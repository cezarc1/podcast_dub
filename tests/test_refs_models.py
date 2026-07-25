import logging

from podcast_dub import types
from podcast_dub.stages import refs
from podcast_dub.stages.refs import _select_reference_segments
from podcast_dub.types import DiarizationSegment


def test_empty_reference_transcript_logs_warning(caplog) -> None:
    build_transcript = getattr(refs, "_build_reference_transcript", None)
    assert build_transcript is not None
    picks = (types.DiarizationSegment(start=0.0, end=5.0, speaker="speaker_0"),)

    with caplog.at_level(logging.WARNING, logger=refs.__name__):
        transcript = build_transcript((), picks, "host")

    assert transcript == ""
    assert "refs: no transcript text found for host reference audio" in caplog.text


def test_reference_selection_returns_longest_clean_segments_as_models() -> None:
    segments = (
        DiarizationSegment(start=0.0, end=5.0, speaker="host"),
        DiarizationSegment(start=6.0, end=14.0, speaker="host"),
        DiarizationSegment(start=15.0, end=20.0, speaker="guest"),
    )

    selected, duration = _select_reference_segments(segments, "host", target_s=6.0)

    assert selected == (segments[1],)
    assert duration == 8.0
