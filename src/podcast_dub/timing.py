from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from podcast_dub.audio_utils import SR
from podcast_dub.fit import LONG_OK, fit_audio
from podcast_dub.types import LogicalTurn, TurnChunk, TurnFitAssessment

MIN_SILENCE_GAP_S = 0.25
INTER_CHUNK_GAP_S = 0.15
EARLY_START_LIMIT_S = 1.5
LATE_START_LIMIT_S = 1.0
FULLER_GAP_MIN_S = 1.5
FULLER_COVERAGE_MAX = 0.85
FULLER_WINDOW_MIN_S = 2.5  # windows shorter than this are too small to be worth refilling
TIMING_POLICY_VERSION = 4

TimingStage = Literal["tts", "place"]


@dataclass(frozen=True, slots=True)
class FittedTurn:
    logical_turn: LogicalTurn
    audio: np.ndarray
    assessment: TurnFitAssessment


def rewrite_can_help(item: FittedTurn, *, word_ceiling: int) -> bool:
    """Returns True if a turn rewrite could still change this turn's duration based on turns current word count and word ceiling."""
    if item.assessment.rewrite_direction != "fuller":
        return True
    return any(len(chunk.text.split()) < word_ceiling for chunk in item.logical_turn.chunks)


def group_by_logical_turns(
    chunks: Sequence[TurnChunk],
    *,
    stage: TimingStage,
) -> tuple[LogicalTurn, ...]:
    """Group consecutive chunks while validating logical-turn identity."""
    grouped = []
    current = []
    closed_turn_ids = set()
    for chunk in chunks:
        if current and chunk.turn_id != current[0].turn_id:
            closed_turn_ids.add(current[0].turn_id)
            grouped.append(_logical_turn(current))
            current = []
        if chunk.turn_id in closed_turn_ids:
            raise RuntimeError(f"{stage}: turn {chunk.turn_id} reappears after a different turn")
        if current and chunk.speaker != current[0].speaker:
            raise RuntimeError(f"{stage}: turn {chunk.turn_id} contains multiple speakers")
        current.append(chunk)
    if current:
        grouped.append(_logical_turn(current))
    return tuple(grouped)


def _logical_turn(chunks: Sequence[TurnChunk]) -> LogicalTurn:
    first = chunks[0]
    return LogicalTurn(
        start=first.start,
        end=max(chunk.end for chunk in chunks),
        speaker=first.speaker,
        turn_id=first.turn_id,
        chunks=tuple(chunks),
    )


def evaluate_timeline(
    turns: Sequence[LogicalTurn],
    total_s: float,
    load_audio: Callable[[Path], np.ndarray],
    *,
    stage: TimingStage,
) -> tuple[FittedTurn, ...]:
    """Decode, assemble, fit, and assess turns using placement's exact policy."""
    fitted = []
    previous_end_s = None
    gap = np.zeros(round(INTER_CHUNK_GAP_S * SR), dtype=np.float32)

    for index, turn in enumerate(turns):
        parts = [load_audio(Path(chunk.audio_file)) for chunk in turn.chunks]
        if any(part.size == 0 for part in parts):
            raise RuntimeError(f"{stage}: turn {turn.turn_id} has empty audio")
        if len(parts) == 1:
            audio = parts[0]  # single-part turn: hand the decoded array straight through, no copy
        else:
            pieces = [parts[0]]
            for part in parts[1:]:
                pieces.extend((gap, part))
            audio = np.concatenate(pieces)

        cue_start_s = turn.start
        if previous_end_s is None:
            start_s = cue_start_s
        else:
            earliest_start_s = max(
                previous_end_s + MIN_SILENCE_GAP_S,
                cue_start_s - EARLY_START_LIMIT_S,
            )
            latest_start_s = cue_start_s + LATE_START_LIMIT_S
            if earliest_start_s > latest_start_s:
                raise RuntimeError(
                    f"{stage}: turn {turn.turn_id} earliest start {earliest_start_s:.3f}s "
                    f"exceeds drift limit {latest_start_s:.3f}s"
                )
            start_s = earliest_start_s

        next_start_s = turns[index + 1].start if index + 1 < len(turns) else total_s
        window_s = next_start_s - MIN_SILENCE_GAP_S - start_s
        if window_s <= 0:
            raise RuntimeError(f"{stage}: turn {turn.turn_id} has nonpositive window {window_s:.3f}s")

        input_duration_s = audio.size / SR
        fit_result = fit_audio(audio, window_s)
        fitted_audio = fit_result.audio
        fitted_duration_s = fitted_audio.size / SR
        direction = None
        if fitted_duration_s > window_s * LONG_OK:
            direction = "tighter"
        elif (
            window_s >= FULLER_WINDOW_MIN_S
            and window_s - fitted_duration_s > FULLER_GAP_MIN_S
            and (index == len(turns) - 1 or fitted_duration_s / window_s < FULLER_COVERAGE_MAX)
        ):
            direction = "fuller"

        end_s = start_s + fitted_duration_s
        assessment = TurnFitAssessment(
            turn_id=turn.turn_id,
            start_s=start_s,
            end_s=end_s,
            cue_start_s=cue_start_s,
            window_s=window_s,
            input_duration_s=input_duration_s,
            fitted_duration_s=fitted_duration_s,
            lag_s=start_s - cue_start_s,
            fit_notes=fit_result.notes,
            rewrite_direction=direction,
        )
        fitted.append(FittedTurn(logical_turn=turn, audio=fitted_audio, assessment=assessment))

        # Flagged audio is never published. Clamp only for this diagnostic pass so
        # later turns can still be assessed and all required rewrites reported.
        previous_end_s = min(end_s, start_s + window_s) if direction else end_s

    return tuple(fitted)
