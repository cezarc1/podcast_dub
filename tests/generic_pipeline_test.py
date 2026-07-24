#!/usr/bin/env python3
"""Standalone integration audit for a completed typed pipeline workdir.

Usage:
    uv run python tests/generic_pipeline_test.py <workdir>
"""

from __future__ import annotations

import argparse
import os

from podcast_dub.artifacts import file_digest, read_artifact
from podcast_dub.audio_utils import dur_of
from podcast_dub.pipeline_artifacts import (
    DIARIZATION_SEGMENTS,
    PHRASES,
    PLACEMENT_RESULT,
    SPEAKER_PHRASES,
    SPEAKER_REFERENCES,
    TRANSLATION_UNITS,
    TURN_CHUNKS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    args = parser.parse_args()
    workdir = args.workdir

    phrases = read_artifact(os.path.join(workdir, "phrases.json"), PHRASES).payload
    segments = read_artifact(os.path.join(workdir, "diar_segments.json"), DIARIZATION_SEGMENTS).payload
    speaker_phrases = read_artifact(os.path.join(workdir, "phrases_spk.json"), SPEAKER_PHRASES).payload
    references = read_artifact(
        os.path.join(workdir, "refs", "references.json"),
        SPEAKER_REFERENCES,
    ).payload
    units = read_artifact(os.path.join(workdir, "units.json"), TRANSLATION_UNITS).payload
    turns = read_artifact(os.path.join(workdir, "turns.json"), TURN_CHUNKS).payload
    placement = read_artifact(os.path.join(workdir, "placement.json"), PLACEMENT_RESULT).payload

    if len(speaker_phrases) != len(units):
        raise RuntimeError(
            f"translation cardinality mismatch: {len(speaker_phrases)} speaker phrases, {len(units)} units"
        )
    expected_speakers = {phrase.speaker for phrase in speaker_phrases}
    reference_speakers = {reference.speaker for reference in references}
    if expected_speakers != reference_speakers:
        raise RuntimeError(
            f"reference speaker mismatch: expected {sorted(expected_speakers)}, got {sorted(reference_speakers)}"
        )
    for reference in references:
        if file_digest(reference.audio_file) != reference.audio_sha256:
            raise RuntimeError(f"reference digest mismatch: {reference.audio_file}")
    for turn in turns:
        if file_digest(turn.audio_file) != turn.audio_sha256:
            raise RuntimeError(f"turn digest mismatch: {turn.audio_file}")
        measured = dur_of(turn.audio_file)
        if abs(measured - turn.audio_duration_s) > 0.05:
            raise RuntimeError(
                f"turn duration mismatch: {turn.audio_file}: recorded={turn.audio_duration_s:.3f}, measured={measured:.3f}"
            )
    if not placement.verification.passed:
        raise RuntimeError("placement artifact records failed verification")
    if not os.path.exists(placement.output_file):
        raise RuntimeError(f"final output is missing: {placement.output_file}")

    print(
        "PASS "
        f"phrases={len(phrases)} segments={len(segments)} speakers={len(expected_speakers)} "
        f"units={len(units)} turns={len(turns)} coverage={placement.verification.coverage:.3f}"
    )


if __name__ == "__main__":
    main()
