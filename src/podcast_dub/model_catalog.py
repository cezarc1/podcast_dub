"""Pinned model identities used by local pipeline stages."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ASR_ID = "Qwen/Qwen3-ASR-1.7B-hf"
ASR_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"

ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGNER_REVISION = "c07281df297b9905d24a508279258cccf987a064"

TTS_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TTS_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"

SORTFORMER_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
SORTFORMER_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
SORTFORMER_FILE = "diar_streaming_sortformer_4spk-v2.1.nemo"

_UNSAFE_CONFIG_FIELDS = frozenset(
    {
        "_attn_implementation_internal",
        "_experts_implementation_internal",
        "auto_map",
        "trust_remote_code",
    }
)


def _unsafe_field(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in _UNSAFE_CONFIG_FIELDS:
                return str(key)
            found = _unsafe_field(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _unsafe_field(nested)
            if found is not None:
                return found
    return None


def validate_model_snapshot(snapshot_path: str | Path) -> None:
    """Reject model configuration hooks that can execute downloaded code."""
    root = Path(snapshot_path)
    config_paths = tuple(root.rglob("config.json"))
    if not config_paths:
        raise RuntimeError(f"model snapshot has no config.json: {root}")
    for config_path in config_paths:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid model config {config_path}: {exc}") from exc
        field = _unsafe_field(config)
        if field is not None:
            raise RuntimeError(f"unsafe model config field {field!r} in {config_path}")
