from typing import Any

import pytest

from podcast_dub.config import JobConfig


@pytest.fixture
def make_job_config(tmp_path):
    """JobConfig factory bound to tmp_path with the minimal required fields."""

    def _make(**overrides: Any) -> JobConfig:
        values = {
            "video": str(tmp_path / "input.mp4"),
            "source_lang": "zh",
            "target_lang": "en",
            "workdir": str(tmp_path),
        }
        values.update(overrides)
        return JobConfig(**values)

    return _make
