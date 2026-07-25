"""Contract tests for the shared TTS/placement timing evaluator."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from podcast_dub.audio_utils import SR
from podcast_dub.types import TurnChunk


def _timing_module():
    spec = importlib.util.find_spec("podcast_dub.timing")
    assert spec is not None, "TTS and placement need one shared production timing evaluator"
    return importlib.import_module("podcast_dub.timing")


def _chunk(
    *,
    turn_id: int,
    part_index: int,
    start: float,
    end: float,
    duration_s: float,
) -> TurnChunk:
    return TurnChunk(
        start=start,
        end=end,
        speaker=f"speaker-{turn_id}",
        text="spoken text",
        turn_id=turn_id,
        part_index=part_index,
        audio_file=f"/tmp/turn-{turn_id}-{part_index}.wav",
        audio_duration_s=duration_s,
        audio_sha256="a" * 64,
    )


def test_evaluator_rewrites_remote_turn_that_old_ratio_gate_accepted() -> None:
    timing = _timing_module()
    decoded_duration_s = 4.3165
    chunk = _chunk(turn_id=1, part_index=0, start=9.9, end=12.94, duration_s=4.368)

    fitted = timing.evaluate_timeline(
        timing.group_by_logical_turns((chunk,), stage="tts"),
        total_s=13.76,
        load_audio=lambda _path: np.ones(round(decoded_duration_s * SR), dtype=np.float32),
        stage="tts",
    )

    assessment = fitted[0].assessment
    assert assessment.input_duration_s / assessment.window_s < 1.22
    assert assessment.window_s == pytest.approx(3.61)
    assert assessment.fitted_duration_s > assessment.window_s * timing.LONG_OK
    assert assessment.rewrite_direction == "tighter"


def test_evaluator_includes_interchunk_gap_and_sequential_lag() -> None:
    timing = _timing_module()
    durations = {
        Path("/tmp/turn-0-0.wav"): 0.8,
        Path("/tmp/turn-0-1.wav"): 0.8,
        Path("/tmp/turn-1-0.wav"): 0.5,
    }
    chunks = (
        _chunk(turn_id=0, part_index=0, start=0.0, end=1.0, duration_s=0.8),
        _chunk(turn_id=0, part_index=1, start=0.0, end=1.0, duration_s=0.8),
        _chunk(turn_id=1, part_index=0, start=1.9, end=2.5, duration_s=0.5),
    )

    fitted = timing.evaluate_timeline(
        timing.group_by_logical_turns(chunks, stage="place"),
        total_s=3.0,
        load_audio=lambda path: np.ones(round(durations[path] * SR), dtype=np.float32),
        stage="place",
    )

    first = fitted[0].assessment
    second = fitted[1].assessment
    assert first.input_duration_s == pytest.approx(1.6 + timing.INTER_CHUNK_GAP_S)
    assert first.window_s == pytest.approx(1.9 - timing.MIN_SILENCE_GAP_S - first.start_s)
    assert second.start_s >= first.end_s + timing.MIN_SILENCE_GAP_S


def test_evaluator_rewrites_long_turn_with_large_uncovered_tail() -> None:
    timing = _timing_module()
    chunk = _chunk(
        turn_id=14,
        part_index=0,
        start=145.52,
        end=274.64,
        duration_s=119.26,
    )

    fitted = timing.evaluate_timeline(
        timing.group_by_logical_turns((chunk,), stage="tts"),
        total_s=300.0,
        load_audio=lambda _path: np.ones(round(119.26 * SR), dtype=np.float32),
        stage="tts",
    )

    assessment = fitted[0].assessment
    assert assessment.window_s - assessment.fitted_duration_s > 30.0
    assert assessment.fitted_duration_s / assessment.window_s > 0.72
    assert assessment.rewrite_direction == "fuller"


def test_evaluator_rewrites_any_tail_beyond_verifier_tolerance() -> None:
    timing = _timing_module()
    chunk = _chunk(
        turn_id=13,
        part_index=0,
        start=145.52,
        end=300.0,
        duration_s=132.1,
    )

    fitted = timing.evaluate_timeline(
        timing.group_by_logical_turns((chunk,), stage="tts"),
        total_s=300.0,
        load_audio=lambda _path: np.ones(round(132.1 * SR), dtype=np.float32),
        stage="tts",
    )

    assessment = fitted[0].assessment
    assert assessment.fitted_duration_s / assessment.window_s > 0.85
    assert assessment.window_s - assessment.fitted_duration_s > timing.FULLER_GAP_MIN_S
    assert assessment.rewrite_direction == "fuller"


def test_evaluator_accepts_audio_within_the_fitters_long_tolerance() -> None:
    timing = _timing_module()
    chunk = _chunk(
        turn_id=13,
        part_index=0,
        start=145.511,
        end=300.0,
        duration_s=154.3525,
    )

    fitted = timing.evaluate_timeline(
        timing.group_by_logical_turns((chunk,), stage="tts"),
        total_s=300.0,
        load_audio=lambda _path: np.ones(round(154.3525 * SR), dtype=np.float32),
        stage="tts",
    )

    assessment = fitted[0].assessment
    assert assessment.fitted_duration_s - assessment.window_s == pytest.approx(0.1135, abs=0.001)
    assert assessment.fitted_duration_s / assessment.window_s < 1.005
    assert assessment.rewrite_direction is None
