"""Atomically persisted, typed translation and rewrite event manifest."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from podcast_dub.artifacts import atomic_write_text
from podcast_dub.types import ManifestEvent

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "translations.jsonl")
_EVENT_ADAPTER = TypeAdapter(ManifestEvent)
_WRITE_LOCK = threading.Lock()


class ManifestError(RuntimeError):
    """The manifest cannot be validated or written safely."""


def configure(log_dir: str) -> None:
    """Point the manifest at a job-specific directory."""
    global LOG_DIR, LOG_FILE
    LOG_DIR = log_dir
    LOG_FILE = os.path.join(LOG_DIR, "translations.jsonl")


def log_event(event: ManifestEvent) -> None:
    """Validate and atomically append one typed manifest event."""
    validated = _EVENT_ADAPTER.validate_python(event)
    encoded = json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    path = Path(LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        try:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            atomic_write_text(path, existing + encoded + "\n")
        except OSError as exc:
            raise ManifestError(f"failed to write manifest {path}: {exc}") from exc


def read_events() -> Iterator[ManifestEvent]:
    path = Path(LOG_FILE)
    if not path.exists():
        raise FileNotFoundError(f"manifest file not found: {path}")
    with path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            try:
                yield _EVENT_ADAPTER.validate_json(line)
            except ValidationError as exc:
                raise ManifestError(f"invalid manifest event {path}:{line_number}: {exc}") from exc
