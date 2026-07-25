#!/usr/bin/env python3
"""Typed turn-building and placement rehearsal without model inference."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from podcast_dub.artifacts import read_artifact
from podcast_dub.pipeline_artifacts import TRANSLATION_UNITS
from podcast_dub.stages.tts import SENT_END, build_turns
from podcast_dub.timing import EARLY_START_LIMIT_S, LATE_START_LIMIT_S, MIN_SILENCE_GAP_S
from podcast_dub.types import SimulationTurn, SubtitleCue, SubtitleUnit, TurnChunkDraft

logger = logging.getLogger(__name__)


def group_units(cues: Sequence[SubtitleCue]) -> tuple[SubtitleUnit, ...]:
    """Group subtitle cues without mutating their source values."""
    units = []
    for cue in cues:
        frames = tuple(range(int(cue.start * 2), int(cue.end * 2)))
        if (
            units
            and units[-1].speaker == cue.speaker
            and not SENT_END.search(units[-1].subtitle_text)
            and cue.start - units[-1].end <= 1.2
            and cue.end - units[-1].start <= 15.0
        ):
            previous = units[-1]
            units[-1] = previous.validated_copy(
                end=max(previous.end, cue.end),
                subtitle_text=f"{previous.subtitle_text} {cue.text}",
                frames=previous.frames + frames,
            )
        else:
            units.append(
                SubtitleUnit(
                    speaker=cue.speaker,
                    start=cue.start,
                    end=cue.end,
                    subtitle_text=cue.text,
                    frames=frames,
                )
            )
    return tuple(units)


def simulate(
    turns: Sequence[SimulationTurn | TurnChunkDraft],
    total: float,
) -> tuple[tuple[float, float], ...]:
    """Predict anchored placement windows for typed turn records."""
    placements = []
    previous_end = None
    for index, turn in enumerate(turns):
        next_start = turns[index + 1].start if index + 1 < len(turns) else total
        if previous_end is None:
            start = turn.start
        else:
            start = min(
                max(previous_end + MIN_SILENCE_GAP_S, turn.start - EARLY_START_LIMIT_S),
                turn.start + LATE_START_LIMIT_S,
            )
        window = max(next_start - MIN_SILENCE_GAP_S - start, 0.05)
        placements.append((start, window))
        duration = turn.audio_duration_s if isinstance(turn, SimulationTurn) else max(len(turn.text.split()) / 3.6, 0.5)
        previous_end = start + min(duration, window)
    return tuple(placements)


def main() -> None:
    from podcast_dub.logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", help="pipeline workdir containing units.json")
    parser.add_argument("--max-t", type=float, default=None)
    parser.add_argument("--words-per-second", type=float, default=3.6)
    args = parser.parse_args()

    units = read_artifact(Path(args.workdir) / "units.json", TRANSLATION_UNITS).payload
    if args.max_t is not None:
        units = tuple(unit for unit in units if unit.start < args.max_t)
    chunks = build_turns(units)
    turns = tuple(
        SimulationTurn(
            **chunk.model_dump(),
            audio_duration_s=max(len(chunk.text.split()) / args.words_per_second, 0.1),
        )
        for chunk in chunks
    )
    total = args.max_t if args.max_t is not None else (turns[-1].end + 0.5)
    for turn, (start, window) in zip(turns, simulate(turns, total), strict=True):
        logger.info(
            "t%dp%d %s: cue=%.2fs placed=%.2fs window=%.2fs",
            turn.turn_id,
            turn.part_index,
            turn.speaker,
            turn.start,
            start,
            window,
        )


if __name__ == "__main__":
    main()
