from podcast_dub.models import DiarizationSegment
from podcast_dub.stages.refs import _select_reference_segments


def test_reference_selection_returns_longest_clean_segments_as_models() -> None:
    segments = (
        DiarizationSegment(start=0.0, end=5.0, speaker="host"),
        DiarizationSegment(start=6.0, end=14.0, speaker="host"),
        DiarizationSegment(start=15.0, end=20.0, speaker="guest"),
    )

    selected, duration = _select_reference_segments(segments, "host", target_s=6.0)

    assert selected == (segments[1],)
    assert duration == 8.0
