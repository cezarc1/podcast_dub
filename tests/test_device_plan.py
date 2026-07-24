import os
from pathlib import Path

import pytest

from podcast_dub import device_utils
from podcast_dub.device_utils import resolve_device_plan
from podcast_dub.models import ConcreteDevice, ModelStage


@pytest.mark.parametrize(
    ("stage", "cuda", "mps", "expected"),
    [
        (ModelStage.ASR, True, False, ConcreteDevice.CUDA),
        (ModelStage.TTS, False, True, ConcreteDevice.MPS),
        (ModelStage.DIARIZE, False, True, ConcreteDevice.CPU),
        (ModelStage.ASR, False, False, ConcreteDevice.CPU),
    ],
)
def test_auto_device_plan_is_stage_specific(stage, cuda, mps, expected) -> None:
    plan = resolve_device_plan(stage, "auto", cuda_available=cuda, mps_available=mps)

    assert plan.device == expected
    assert plan.stage == stage


def test_explicit_unavailable_device_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device_plan(ModelStage.TTS, "cuda", cuda_available=False, mps_available=True)


def test_helper_python_defaults_to_current_project(monkeypatch, tmp_path) -> None:
    helper_python = getattr(device_utils, "helper_python", None)
    assert helper_python is not None
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DUB_ASR_PYTHON", raising=False)

    assert helper_python("DUB_ASR_PYTHON", ".venv-asr") == str(tmp_path / ".venv-asr" / "bin" / "python")


def test_helper_environment_exposes_installed_package_and_preserves_pythonpath(monkeypatch) -> None:
    helper_process_env = getattr(device_utils, "helper_process_env", None)
    assert helper_process_env is not None
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

    env = helper_process_env()
    entries = tuple(Path(entry) for entry in env["PYTHONPATH"].split(os.pathsep))

    assert entries[0].joinpath("podcast_dub").is_dir()
    assert entries[1] == Path("/existing/pythonpath")
