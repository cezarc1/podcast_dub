import logging
import os
from pathlib import Path

import pytest

from podcast_dub import device_utils
from podcast_dub.device_utils import resolve_device_plan
from podcast_dub.types import ConcreteDevice, ModelStage


def test_hardware_probe_failure_logs_exception_and_assumes_cpu(monkeypatch, caplog) -> None:
    def fail_probe() -> bool:
        raise RuntimeError("broken torch runtime")

    monkeypatch.setattr(device_utils.torch.cuda, "is_available", fail_probe)

    with caplog.at_level(logging.ERROR, logger=device_utils.__name__):
        plan = resolve_device_plan(ModelStage.TTS, "auto")

    assert plan.device == ConcreteDevice.CPU
    assert "hardware probe failed; assuming CPU-only" in caplog.text
    assert "broken torch runtime" in caplog.text


def test_missing_flash_attention_logs_debug_fallback(monkeypatch, caplog) -> None:
    def missing_flash_attention(name: str) -> None:
        if name == "flash_attn":
            raise ModuleNotFoundError(name)

    monkeypatch.setattr(device_utils.importlib, "import_module", missing_flash_attention)

    with caplog.at_level(logging.DEBUG, logger=device_utils.__name__):
        plan = resolve_device_plan(ModelStage.TTS, "auto", cuda_available=True, mps_available=False)

    assert plan.attention == "sdpa"
    assert "flash-attn unavailable; using sdpa" in caplog.text


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
