"""Pure-Python checks for ASR chunking/validation helpers and device kwargs.

Deliberately free of torch/transformers/ffmpeg/network usage: the ASR stage
imports torch lazily inside ``_run_asr_inline``, so these helpers stay callable
without a model runtime. ``dtype`` is asserted by name for the same reason.
"""

import pytest

from podcast_dub.device_utils import model_kwargs_for
from podcast_dub.stages.asr import (
    CHUNK_S,
    MAX_COLLAPSED_WORD_RUN,
    _chunk_ranges,
    _validate_alignment_quality,
)
from podcast_dub.types import AlignedWord, ConcreteDevice, DevicePlan, ModelStage


def test_chunk_ranges_tile_an_exact_multiple_of_chunk_s() -> None:
    ranges = _chunk_ranges(CHUNK_S * 3)

    assert ranges == ((0.0, CHUNK_S), (CHUNK_S, CHUNK_S), (CHUNK_S * 2, CHUNK_S))


def test_chunk_ranges_end_with_a_short_remainder_chunk() -> None:
    remainder = CHUNK_S / 4
    ranges = _chunk_ranges(CHUNK_S * 2 + remainder)

    assert ranges == ((0.0, CHUNK_S), (CHUNK_S, CHUNK_S), (CHUNK_S * 2, remainder))


def test_chunk_ranges_returns_a_single_short_range_below_one_chunk() -> None:
    assert _chunk_ranges(CHUNK_S / 2) == ((0.0, CHUNK_S / 2),)


def test_chunk_ranges_is_empty_for_a_zero_limit() -> None:
    assert _chunk_ranges(0.0) == ()


def _collapsed_words(count: int, *, at: float = 4.0) -> tuple[AlignedWord, ...]:
    return tuple(AlignedWord(text="字", start=at, end=at) for _ in range(count))


def test_validate_alignment_quality_allows_max_collapsed_run() -> None:
    _validate_alignment_quality(_collapsed_words(MAX_COLLAPSED_WORD_RUN))


def test_validate_alignment_quality_rejects_one_past_max_collapsed_run() -> None:
    with pytest.raises(RuntimeError, match=f"collapsed {MAX_COLLAPSED_WORD_RUN + 1} consecutive words"):
        _validate_alignment_quality(_collapsed_words(MAX_COLLAPSED_WORD_RUN + 1))


def test_validate_alignment_quality_resets_the_run_on_a_timed_word() -> None:
    words = (
        *_collapsed_words(MAX_COLLAPSED_WORD_RUN),
        AlignedWord(text="好", start=4.0, end=4.5),
        *_collapsed_words(MAX_COLLAPSED_WORD_RUN, at=4.5),
    )

    _validate_alignment_quality(words)


@pytest.mark.parametrize(
    ("device", "dtype", "attention", "expected_dtype"),
    [
        (ConcreteDevice.CPU, "float32", "eager", "torch.float32"),
        (ConcreteDevice.MPS, "bfloat16", "eager", "torch.bfloat16"),
        (ConcreteDevice.CUDA, "bfloat16", "flash_attention_2", "torch.bfloat16"),
    ],
)
def test_model_kwargs_for_uses_the_given_plan_not_probed_hardware(device, dtype, attention, expected_dtype) -> None:
    plan = DevicePlan(stage=ModelStage.TTS, device=device, dtype=dtype, attention=attention)

    kwargs = model_kwargs_for(plan)

    assert kwargs["device_map"] == device
    assert kwargs["attn_implementation"] == attention
    assert str(kwargs["dtype"]) == expected_dtype
