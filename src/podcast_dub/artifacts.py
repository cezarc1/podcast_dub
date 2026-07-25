from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from podcast_dub.config import JobConfig
from podcast_dub.types import DevicePlan, ModelIdentity, StageName

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ArtifactError(RuntimeError):
    """A current-schema artifact is corrupt or violates its declared contract."""


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputDigest(_ArtifactModel):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)


class ArtifactProvenance(_ArtifactModel):
    input_digests: tuple[InputDigest, ...] = ()
    config_digest: str = Field(pattern=SHA256_PATTERN)
    parameters_digest: str = Field(pattern=SHA256_PATTERN)
    model: ModelIdentity | None = None
    backend_lock_digest: str | None = Field(default=None, pattern=SHA256_PATTERN)
    execution_plan: DevicePlan | None = None


class ArtifactEnvelope[T](_ArtifactModel):
    schema_version: int = Field(gt=0)
    stage: StageName
    provenance: ArtifactProvenance
    payload: T


@dataclass(frozen=True)
class ArtifactSpec[T]:
    stage: StageName
    payload_type: Any
    schema_version: int = 1

    def adapter(self) -> TypeAdapter[ArtifactEnvelope[T]]:
        return TypeAdapter(ArtifactEnvelope[self.payload_type])


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return value


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        default=_jsonable,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: str | os.PathLike[str]) -> str:
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Write text via a sibling .tmp file and atomic rename (no torn files on kill)."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def build_provenance(
    cfg: JobConfig,
    *,
    input_files: Mapping[str, str | os.PathLike[str]],
    parameters: Any,
    model: ModelIdentity | None = None,
    execution_plan: DevicePlan | None = None,
) -> ArtifactProvenance:
    inputs = tuple(InputDigest(name=name, sha256=file_digest(path)) for name, path in sorted(input_files.items()))
    return ArtifactProvenance(
        input_digests=inputs,
        config_digest=stable_digest(cfg.provenance_config()),
        parameters_digest=stable_digest(parameters),
        model=model,
        execution_plan=execution_plan,
    )


def load_cached_artifact[T](
    path: str | os.PathLike[str],
    spec: ArtifactSpec[T],
    expected_provenance: ArtifactProvenance,
) -> T | None:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON artifact {artifact_path}: {exc}") from exc
    if not isinstance(raw, dict) or "schema_version" not in raw:
        return None
    if raw.get("schema_version") != spec.schema_version or raw.get("stage") != spec.stage:
        return None
    try:
        envelope = spec.adapter().validate_python(raw)
    except ValidationError as exc:
        raise ArtifactError(f"invalid current-schema artifact {artifact_path}: {exc}") from exc
    if envelope.provenance != expected_provenance:
        return None
    return envelope.payload


def read_artifact[T](path: str | os.PathLike[str], spec: ArtifactSpec[T]) -> ArtifactEnvelope[T]:
    """Read a required current-schema artifact without applying cache expectations."""
    artifact_path = Path(path)
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"required artifact is missing: {artifact_path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid JSON artifact {artifact_path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != spec.schema_version or raw.get("stage") != spec.stage:
        raise ArtifactError(
            f"legacy or incompatible artifact {artifact_path}; regenerate the producing stage before resuming"
        )
    try:
        return spec.adapter().validate_python(raw)
    except ValidationError as exc:
        raise ArtifactError(f"invalid current-schema artifact {artifact_path}: {exc}") from exc


def write_artifact_atomic[T](
    path: str | os.PathLike[str],
    spec: ArtifactSpec[T],
    provenance: ArtifactProvenance,
    payload: T,
) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        envelope = spec.adapter().validate_python(
            {
                "schema_version": spec.schema_version,
                "stage": spec.stage,
                "provenance": provenance,
                "payload": payload,
            }
        )
    except ValidationError as exc:
        raise ArtifactError(f"refusing to write invalid artifact {artifact_path}: {exc}") from exc
    serialized = json.dumps(
        envelope.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(artifact_path, serialized + "\n")
