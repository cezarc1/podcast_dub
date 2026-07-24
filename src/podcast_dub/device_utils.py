import importlib
import os
from pathlib import Path
from typing import Any, Literal

import torch

from podcast_dub.models import ConcreteDevice, DevicePlan, ModelStage

PACKAGE_PARENT = Path(__file__).resolve().parent.parent


def helper_python(env_var: str, environment_dir: str) -> str:
    """Resolve a helper interpreter from an override or the current project."""
    configured = os.environ.get(env_var)
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return str(Path.cwd() / environment_dir / "bin" / "python")


def helper_process_env() -> dict[str, str]:
    """Make this installed/source package importable in a helper interpreter."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    entries = [str(PACKAGE_PARENT)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _availability() -> tuple[bool, bool]:
    try:
        cuda = bool(torch.cuda.is_available())
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return cuda, mps
    except Exception:
        return False, False


def resolve_device_plan(
    stage: ModelStage,
    requested: str,
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> DevicePlan:
    """Resolve a requested device once; never change it after a backend error."""
    detected_cuda, detected_mps = _availability()
    cuda = detected_cuda if cuda_available is None else cuda_available
    mps = detected_mps if mps_available is None else mps_available
    if requested == "auto":
        if cuda:
            device = ConcreteDevice.CUDA
        elif mps and stage != ModelStage.DIARIZE:
            device = ConcreteDevice.MPS
        else:
            device = ConcreteDevice.CPU
    elif requested == "cuda":
        if not cuda:
            raise RuntimeError(f"{stage}: cuda was requested but is unavailable")
        device = ConcreteDevice.CUDA
    elif requested == "mps":
        if stage == ModelStage.DIARIZE:
            raise RuntimeError("diarize: mps is not supported; choose auto, cuda, or cpu")
        if not mps:
            raise RuntimeError(f"{stage}: mps was requested but is unavailable")
        device = ConcreteDevice.MPS
    elif requested == "cpu":
        device = ConcreteDevice.CPU
    else:
        raise RuntimeError(f"{stage}: unsupported device {requested!r}")

    dtype = "bfloat16" if device in (ConcreteDevice.CUDA, ConcreteDevice.MPS) else "float32"
    if device == ConcreteDevice.CUDA:
        try:
            importlib.import_module("flash_attn")
            attention = "flash_attention_2"
        except Exception:
            attention = "sdpa"
    else:
        attention = "eager"
    return DevicePlan(stage=stage, device=device, dtype=dtype, attention=attention)


def model_kwargs(
    requested: str = "auto", *, stage: Literal[ModelStage.ASR, ModelStage.TTS] = ModelStage.TTS
) -> dict[str, Any]:
    plan = resolve_device_plan(stage, requested)
    dtype = torch.bfloat16 if plan.dtype == "bfloat16" else torch.float32
    return {
        "device_map": plan.device,
        "dtype": dtype,
        "attn_implementation": plan.attention,
    }


if __name__ == "__main__":
    plan = resolve_device_plan(ModelStage.TTS, "auto")
    print(f"device={plan.device} dtype={plan.dtype} attn={plan.attention}")
