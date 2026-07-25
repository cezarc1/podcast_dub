import importlib
import logging
import os
from pathlib import Path
from typing import Any, assert_never

import torch

from podcast_dub.types import ConcreteDevice, DeviceChoice, DevicePlan, ModelStage

logger = logging.getLogger(__name__)

PACKAGE_PARENT = Path(__file__).resolve().parent.parent


def helper_python(env_var: str, environment_dir: str) -> str:
    """Resolve a helper interpreter from an override or the current project."""
    if configured := os.environ.get(env_var):
        return os.path.abspath(os.path.expanduser(configured))
    return str(Path.cwd() / environment_dir / "bin" / "python")


def helper_process_env() -> dict[str, str]:
    """Make this installed/source package importable in a helper interpreter."""
    env = dict(os.environ)
    entries = [str(PACKAGE_PARENT)]
    if existing := env.get("PYTHONPATH"):
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _availability() -> tuple[bool, bool]:
    try:
        cuda = bool(torch.cuda.is_available())
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return cuda, mps
    except Exception:
        logger.exception("device: hardware probe failed; assuming CPU-only")
        return False, False


def resolve_device_plan(
    stage: ModelStage,
    requested: DeviceChoice,
    *,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> DevicePlan:
    """Resolve a requested device once; never change it after a backend error."""
    detected_cuda, detected_mps = _availability()
    cuda = detected_cuda if cuda_available is None else cuda_available
    mps = detected_mps if mps_available is None else mps_available
    match requested:
        case DeviceChoice.AUTO:
            if cuda:
                device = ConcreteDevice.CUDA
            elif mps and stage != ModelStage.DIARIZE:
                device = ConcreteDevice.MPS
            else:
                device = ConcreteDevice.CPU
        case DeviceChoice.CUDA:
            if not cuda:
                raise RuntimeError(f"{stage}: cuda was requested but is unavailable")
            device = ConcreteDevice.CUDA
        case DeviceChoice.MPS:
            if stage == ModelStage.DIARIZE:
                raise RuntimeError("diarize: mps is not supported; choose auto, cuda, or cpu")
            if not mps:
                raise RuntimeError(f"{stage}: mps was requested but is unavailable")
            device = ConcreteDevice.MPS
        case DeviceChoice.CPU:
            device = ConcreteDevice.CPU
        case _:
            assert_never(requested)

    dtype = "bfloat16" if device in (ConcreteDevice.CUDA, ConcreteDevice.MPS) else "float32"
    if device == ConcreteDevice.CUDA:
        try:
            importlib.import_module("flash_attn")
            attention = "flash_attention_2"
        except Exception as exc:
            logger.warning("device: flash-attn unavailable; using sdpa: %s", exc)
            attention = "sdpa"
    else:
        attention = "eager"
    return DevicePlan(stage=stage, device=device, dtype=dtype, attention=attention)


def _serialize_mps_weight_loading(plan: DevicePlan) -> None:
    """Load weights on a single thread when targeting MPS.

    Transformers 5.x materializes checkpoint tensors from a thread pool. Each
    worker calls ``tensor.to(device, dtype)``, and on MPS those casts contend
    inside PyTorch's Metal shader cache: the load livelocks rather than merely
    running slowly, with several threads spinning indefinitely. Loading is not
    I/O-bound at this model size, so serializing costs nothing measurable.
    """
    if plan.device != ConcreteDevice.MPS:
        return
    try:
        from transformers import core_model_loading
    except ImportError:  # transformers < 5 has no parallel loader
        return
    if core_model_loading.GLOBAL_WORKERS != 1:
        core_model_loading.GLOBAL_WORKERS = 1
        logger.info("device: serialized weight loading (MPS + Metal shader cache contention)")


def model_kwargs_for(plan: DevicePlan) -> dict[str, Any]:
    _serialize_mps_weight_loading(plan)
    dtype = torch.bfloat16 if plan.dtype == "bfloat16" else torch.float32
    return {
        "device_map": plan.device,
        "dtype": dtype,
        "attn_implementation": plan.attention,
    }


if __name__ == "__main__":
    from podcast_dub.logging_config import configure_logging

    configure_logging()
    plan = resolve_device_plan(ModelStage.TTS, DeviceChoice.AUTO)
    logger.info("device=%s dtype=%s attn=%s", plan.device, plan.dtype, plan.attention)
