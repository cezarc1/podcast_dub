#!/usr/bin/env python3
"""Generate an HTML and SRT review report from typed translation artifacts."""

import argparse
import html
import logging
from pathlib import Path

from podcast_dub.artifacts import read_artifact
from podcast_dub.audio_utils import srt_ts
from podcast_dub.pipeline_artifacts import TRANSLATION_UNITS
from podcast_dub.stages.tts import build_turns
from podcast_dub.tools.turn_tts_sample import simulate

# Speaker colors are assigned from this palette in order of first appearance.
PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

logger = logging.getLogger(__name__)


def main() -> None:
    from podcast_dub.logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--max-t", type=float, default=300.0)
    args = parser.parse_args()
    workdir = Path(args.workdir)

    units = read_artifact(workdir / "units.json", TRANSLATION_UNITS).payload
    units = tuple(unit for unit in units if unit.start < args.max_t)
    chunks = build_turns(units)
    placements = simulate(chunks, args.max_t + 0.5)
    placement_by_turn = {}
    for chunk, placement in zip(chunks, placements, strict=True):
        placement_by_turn.setdefault(chunk.turn_id, placement)

    logical = []
    part_counts = {}
    for chunk in chunks:
        part_counts[chunk.turn_id] = part_counts.get(chunk.turn_id, 0) + 1
        if logical and logical[-1].turn_id == chunk.turn_id:
            previous = logical[-1]
            logical[-1] = previous.validated_copy(end=chunk.end, text=f"{previous.text} {chunk.text}")
        else:
            logical.append(chunk)

    with (workdir / "turns_debug.srt").open("w", encoding="utf-8") as subtitles:
        for index, turn in enumerate(logical, start=1):
            start, window = placement_by_turn[turn.turn_id]
            subtitles.write(f"{index}\n{srt_ts(start)} --> {srt_ts(start + window)}\n[{turn.speaker}] {turn.text}\n\n")

    rows = []
    speaker_colors = {}
    for index, turn in enumerate(logical):
        start, window = placement_by_turn[turn.turn_id]
        if turn.speaker not in speaker_colors:
            speaker_colors[turn.speaker] = PALETTE[len(speaker_colors) % len(PALETTE)]
        color = speaker_colors[turn.speaker]
        rows.append(
            f"<tr><td>{index}</td><td style='color:{color}'><b>{html.escape(turn.speaker)}</b></td>"
            f"<td>{part_counts[turn.turn_id]}</td><td>{turn.start:.1f}</td><td>{turn.end:.1f}</td>"
            f"<td>{start:.1f}</td><td>{window:.1f}</td><td>{start - turn.start:+.1f}</td>"
            f"<td>{html.escape(turn.text)}</td></tr>"
        )
    report = f"""<!doctype html><meta charset="utf-8"><title>turn report</title>
<style>body{{font:13px system-ui}}td,th{{padding:4px 8px;border-bottom:1px solid #ddd;vertical-align:top}}table{{border-collapse:collapse}}</style>
<h2>Turn report — {len(logical)} logical turns ({len(chunks)} chunks)</h2>
<table><tr><th>#</th><th>speaker</th><th>parts</th><th>cue start</th><th>cue end</th><th>placed start</th><th>window</th><th>lag</th><th>text</th></tr>
{"".join(rows)}</table>"""
    (workdir / "turns_report.html").write_text(report, encoding="utf-8")
    logger.info("wrote %s and %s", workdir / "turns_debug.srt", workdir / "turns_report.html")


if __name__ == "__main__":
    main()
