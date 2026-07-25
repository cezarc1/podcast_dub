"""Regression pins for artifact digest helpers and read_artifact error paths."""

import hashlib
import json

import pytest

from podcast_dub.artifacts import ArtifactError, ArtifactSpec, file_digest, read_artifact, stable_digest
from podcast_dub.types import StageName

NUMBERS: ArtifactSpec[tuple[int, ...]] = ArtifactSpec(stage=StageName.ASR, payload_type=tuple[int, ...])


def test_stable_digest_ignores_key_insertion_order() -> None:
    first = {"b": 1, "a": {"z": [1, 2], "y": "x"}}
    second = {"a": {"y": "x", "z": [1, 2]}, "b": 1}

    assert list(first) != list(second)
    assert stable_digest(first) == stable_digest(second)


def test_stable_digest_distinguishes_values() -> None:
    assert stable_digest({"a": 1}) != stable_digest({"a": 2})
    assert stable_digest({"a": [1, 2]}) != stable_digest({"a": [2, 1]})


@pytest.mark.parametrize("size", [0, 1, 1024 * 1024 + 7])
def test_file_digest_matches_hashlib_sha256(tmp_path, size) -> None:
    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)

    assert file_digest(path) == hashlib.sha256(payload).hexdigest()


def test_file_digest_on_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        file_digest(tmp_path / "absent.bin")


def test_read_artifact_missing_file_is_an_artifact_error(tmp_path) -> None:
    path = tmp_path / "absent.json"

    with pytest.raises(ArtifactError, match="required artifact is missing"):
        read_artifact(path, NUMBERS)


def test_read_artifact_corrupt_json_is_an_artifact_error(tmp_path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ArtifactError, match="invalid JSON artifact"):
        read_artifact(path, NUMBERS)


def test_read_artifact_wrong_schema_version_is_an_artifact_error(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"schema_version": 99, "stage": "asr", "payload": []}), encoding="utf-8")

    with pytest.raises(ArtifactError, match="legacy or incompatible artifact"):
        read_artifact(path, NUMBERS)
