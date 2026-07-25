"""Regression tests for the diarize/refs cleanup pass.

Covers two behavior fixes and the helpers they touch:
- merge_orphans forward merge must widen the span end, not only the start
  (an orphan reaching past the next phrase used to fail phrase validation)
- speaker_mapping must break equal-duration ties on the raw speaker label so
  the display-name mapping is identical across processes
- _join_with_silence keeps the single-part passthrough and the int16 dtype
- _select_reference_segments rejects candidates that fail the clearance or the
  duration window
"""

import numpy as np
import pytest

from podcast_dub.stages.diarize import merge_orphans, speaker_mapping
from podcast_dub.stages.refs import CLEAR_S, MAX_D, MIN_D, _join_with_silence, _select_reference_segments
from podcast_dub.types import DiarizationSegment, PhraseWord, SpeakerPhrase


def _word(text: str, start: float, end: float) -> PhraseWord:
    return PhraseWord(text=text, start=start, end=end)


def test_forward_merge_widens_span_end() -> None:
    # the orphan reaches past the end of the phrase it merges into; keeping the
    # next phrase's end would leave its word outside the merged span and the
    # pydantic model would reject the copy
    orphan = SpeakerPhrase(start=1.0, end=2.0, text="嗯", speaker="a", words=(_word("嗯", 1.0, 2.0),))
    following = SpeakerPhrase(start=1.5, end=1.9, text="对对对", speaker="a", words=(_word("对", 1.5, 1.9),))

    merged = merge_orphans([orphan, following])

    assert len(merged) == 1
    assert merged[0].text == "嗯对对对"
    assert merged[0].start == 1.0
    assert merged[0].end == 2.0
    assert len(merged[0].words) == 2


def test_forward_merge_keeps_wider_next_end() -> None:
    orphan = SpeakerPhrase(start=1.0, end=1.2, text="嗯", speaker="a", words=(_word("嗯", 1.0, 1.2),))
    following = SpeakerPhrase(start=1.5, end=3.0, text="对对对", speaker="a", words=(_word("对", 1.5, 1.9),))

    merged = merge_orphans([orphan, following])

    assert len(merged) == 1
    assert merged[0].start == 1.0
    assert merged[0].end == 3.0


def test_equal_duration_speakers_map_deterministically() -> None:
    # both speakers hold exactly 10s; the tie must resolve on the raw label
    segments = [
        DiarizationSegment(start=0.0, end=10.0, speaker="speaker_1"),
        DiarizationSegment(start=10.5, end=20.5, speaker="speaker_0"),
    ]

    _, mapping, totals = speaker_mapping(segments, ["host", "guest"])

    assert totals["speaker_0"] == totals["speaker_1"]
    assert mapping == {"speaker_0": "host", "speaker_1": "guest"}
    # input order must not move the tie either
    _, reversed_mapping, _ = speaker_mapping(list(reversed(segments)), ["host", "guest"])
    assert reversed_mapping == mapping


def test_speaker_mapping_totals_do_not_autovivify_absent_speakers() -> None:
    # totals is accumulated in a defaultdict(float) but must not be returned as one:
    # a caller probing an unknown speaker should get KeyError, not a silent 0.0.
    segments = [DiarizationSegment(start=0.0, end=4.0, speaker="speaker_0")]

    _, _, totals = speaker_mapping(segments, ["host"])

    assert totals == {"speaker_0": 4.0}
    with pytest.raises(KeyError):
        totals["speaker_absent"]
    assert totals == {"speaker_0": 4.0}


def test_join_with_silence_single_part_passthrough() -> None:
    part = np.array([1, 2, 3], dtype=np.int16)
    silence = np.zeros(2, dtype=np.int16)

    joined = _join_with_silence([part], silence)

    assert joined is part
    assert joined.dtype == np.int16
    np.testing.assert_array_equal(joined, np.array([1, 2, 3], dtype=np.int16))


def test_join_with_silence_three_parts() -> None:
    parts = [
        np.array([1, 2], dtype=np.int16),
        np.array([3], dtype=np.int16),
        np.array([4, 5], dtype=np.int16),
    ]
    silence = np.zeros(2, dtype=np.int16)

    joined = _join_with_silence(parts, silence)

    assert joined.dtype == np.int16
    np.testing.assert_array_equal(joined, np.array([1, 2, 0, 0, 3, 0, 0, 4, 5], dtype=np.int16))


def test_selection_rejects_candidate_without_clearance() -> None:
    crowded = DiarizationSegment(start=0.0, end=6.0, speaker="host")
    clean = DiarizationSegment(start=20.0, end=26.0, speaker="host")
    segments = (
        crowded,
        # starts inside the clearance margin of `crowded`, so it is disqualified
        DiarizationSegment(start=crowded.end + CLEAR_S / 2, end=8.0, speaker="guest"),
        clean,
    )

    selected, total = _select_reference_segments(segments, "host", target_s=5.0)

    assert selected == (clean,)
    assert total == 6.0


def test_selection_rejects_durations_outside_window() -> None:
    too_short = DiarizationSegment(start=0.0, end=MIN_D - 0.5, speaker="host")
    too_long = DiarizationSegment(start=30.0, end=30.0 + MAX_D + 1.0, speaker="host")
    in_window = DiarizationSegment(start=60.0, end=65.0, speaker="host")

    selected, total = _select_reference_segments((too_short, too_long, in_window), "host", target_s=100.0)

    assert selected == (in_window,)
    assert total == 5.0
