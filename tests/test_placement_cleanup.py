"""Unit tests for placement/timing plumbing that needs no ffmpeg.

Covers the mix-buffer overrun guard, logical-turn identity validation, and the
single-part passthrough in the shared timeline evaluator.
"""

from __future__ import annotations

import numpy as np
import pytest

from podcast_dub.audio_utils import SR
from podcast_dub.stages import place
from podcast_dub.timing import INTER_CHUNK_GAP_S, MIN_SILENCE_GAP_S, evaluate_timeline, group_by_logical_turns
from podcast_dub.types import TurnChunk


def _chunk(*, turn_id: int, part_index: int = 0, start: float = 0.0, end: float = 1.0) -> TurnChunk:
    return TurnChunk(
        start=start,
        end=end,
        speaker="host",
        text="spoken text",
        turn_id=turn_id,
        part_index=part_index,
        audio_file=f"/tmp/turn-{turn_id}-{part_index}.wav",
        audio_duration_s=end - start,
        audio_sha256="a" * 64,
    )


def test_mix_add_accumulates_without_trimming_when_the_turn_fits() -> None:
    mix = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    place._mix_add(mix, 2, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert np.array_equal(mix, np.array([0.5, 0.5, 1.5, 2.5, 3.5, 0.5], dtype=np.float32))


def test_mix_add_trims_a_turn_that_overruns_the_buffer() -> None:
    mix = np.zeros(4, dtype=np.float32)

    place._mix_add(mix, 2, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    assert np.array_equal(mix, np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32))


def test_mix_add_drops_a_turn_that_starts_at_the_buffer_end() -> None:
    mix = np.zeros(3, dtype=np.float32)

    place._mix_add(mix, 3, np.array([1.0, 2.0], dtype=np.float32))

    assert np.array_equal(mix, np.zeros(3, dtype=np.float32))


def test_group_logical_turns_rejects_a_non_contiguous_repeated_turn_id() -> None:
    chunks = (_chunk(turn_id=0), _chunk(turn_id=1), _chunk(turn_id=0, part_index=1))

    with pytest.raises(RuntimeError, match=r"place: turn 0 reappears"):
        group_by_logical_turns(chunks, stage="place")


def test_group_logical_turns_groups_contiguous_repeats() -> None:
    chunks = (_chunk(turn_id=0), _chunk(turn_id=0, part_index=1), _chunk(turn_id=1))

    grouped = group_by_logical_turns(chunks, stage="place")

    assert len(grouped) == 2
    assert [turn.turn_id for turn in grouped] == [0, 1]
    assert len(grouped[0].chunks) == 2
    assert len(grouped[1].chunks) == 1


def test_evaluate_timeline_passes_a_single_part_turn_through_untouched() -> None:
    window_s = 2.0
    total_s = window_s + MIN_SILENCE_GAP_S
    source = np.ones(round(window_s * SR), dtype=np.float32)

    fitted = evaluate_timeline(
        group_by_logical_turns((_chunk(turn_id=0, start=0.0, end=window_s),), stage="place"),
        total_s,
        lambda _path: source,
        stage="place",
    )

    assert fitted[0].audio is source
    assert fitted[0].assessment.fit_notes == ()


def test_evaluate_timeline_joins_multi_part_turns_with_one_gap_each() -> None:
    part_s = 1.0
    gap = np.zeros(round(INTER_CHUNK_GAP_S * SR), dtype=np.float32)
    window_s = 2 * part_s + gap.size / SR
    chunks = (
        _chunk(turn_id=0, part_index=0, start=0.0, end=window_s),
        _chunk(turn_id=0, part_index=1, start=0.0, end=window_s),
    )

    fitted = evaluate_timeline(
        group_by_logical_turns(chunks, stage="place"),
        window_s + MIN_SILENCE_GAP_S,
        lambda _path: np.ones(round(part_s * SR), dtype=np.float32),
        stage="place",
    )

    part = np.ones(round(part_s * SR), dtype=np.float32)
    assert np.array_equal(fitted[0].audio, np.concatenate((part, gap, part)))
    assert fitted[0].assessment.fit_notes == ()
