"""Tests for placement invariants: minimum silence gap between turns and
window sanity, via podcast_dub.tools.turn_tts_sample.simulate().

Policy under test:
- consecutive turns are separated by >= MIN_SILENCE_GAP_S whenever the
  drift clamp [cue-1.5, cue+1.0] doesn't force a smaller gap
- that clamp is the ONLY exception (start = cue+1.0 when prev overruns badly)
- windows never go negative
"""

import numpy as np
import pytest

from podcast_dub.audio_utils import SR
from podcast_dub.models import SimulationTurn, TurnChunk
from podcast_dub.stages import place
from podcast_dub.timing import MIN_SILENCE_GAP_S
from podcast_dub.tools.turn_tts_sample import simulate


def turn(start, dur, text=None):
    words = " ".join(["word"] * max(1, int(dur * 3.6))) if text is None else text
    return SimulationTurn(
        speaker="yang",
        start=start,
        end=start + dur,
        text=words,
        turn_id=int(start),
        part_index=0,
        audio_duration_s=dur,
    )


def placements(turns, total=200.0):
    return simulate(turns, total)


class TestMinGap:
    def test_first_turn_anchors_on_cue(self):
        turns = [turn(0, 5), turn(10, 5)]
        assert abs(placements(turns)[0][0] - 0.0) < 1e-9

    def test_basic_gap_respected(self):
        turns = [turn(0, 5), turn(10, 5), turn(20, 5)]
        pl = placements(turns)
        ends = [pl[i][0] + min(turns[i].audio_duration_s, pl[i][1]) for i in range(len(turns))]
        for i in range(1, len(turns)):
            assert pl[i][0] >= ends[i - 1] + MIN_SILENCE_GAP_S - 1e-9

    def test_gap_respected_after_trimmed_overrun(self):
        # prev audio (8s) overruns but is trimmed at next_cue - gap;
        # next turn starts exactly on its cue, gap preserved
        turns = [turn(0, 8), turn(5, 3)]
        pl = placements(turns)
        assert abs(pl[1][0] - 5.0) < 1e-9
        prev_end = pl[0][0] + min(8, pl[0][1])
        assert pl[1][0] >= prev_end + MIN_SILENCE_GAP_S - 1e-9

    def test_overruns_never_propagate(self):
        # heavy overruns: every turn still starts on its cue (drift cap)
        # and windows end at next cue - gap
        turns = [turn(0, 10), turn(2, 3), turn(4, 3), turn(6, 3)]
        pl = placements(turns)
        ends = [pl[i][0] + min(turns[i].audio_duration_s, pl[i][1]) for i in range(len(turns))]
        for i in range(1, len(turns)):
            assert pl[i][0] >= ends[i - 1] + MIN_SILENCE_GAP_S - 1e-9
            assert pl[i][0] <= turns[i].start + 1.0 + 1e-9

    def test_backward_drift_bounded(self):
        # prev ends early: next may start early, but no earlier than cue-1.5
        turns = [turn(0, 1), turn(10, 3)]
        pl = placements(turns)
        assert pl[1][0] >= 10 - 1.5 - 1e-9


class TestWindows:
    def test_no_negative_windows(self):
        turns = [turn(0, 50), turn(10, 3), turn(12, 3)]
        for _start, window in placements(turns):
            assert window > 0

    def test_last_turn_window_to_total(self):
        turns = [turn(0, 5), turn(10, 5)]
        pl = placements(turns, total=60.0)
        assert pl[-1][1] > 5  # last turn gets room to the end


def test_place_uses_the_shared_post_fit_assessment(monkeypatch) -> None:
    chunk = TurnChunk(
        start=9.9,
        end=12.94,
        speaker="host",
        text="Over the past year, TikTok has swung between peaks and valleys.",
        turn_id=1,
        part_index=0,
        audio_file="/tmp/turn-1.mp3",
        audio_duration_s=4.368,
        audio_sha256="a" * 64,
    )
    monkeypatch.setattr(
        place,
        "decode_f32",
        lambda _path, _tempo=1.0: np.ones(round(4.3165 * SR), dtype=np.float32),
    )

    fitted = place._evaluate_chunks((chunk,), total_s=13.76)

    assert fitted[0].assessment.rewrite_direction == "tighter"
    assert fitted[0].audio.size / SR == fitted[0].assessment.fitted_duration_s
    with pytest.raises(RuntimeError, match=r"place: TTS timing contract.*turn 1"):
        place._require_placeable(fitted)


def test_place_rejects_a_residual_underfilled_turn(monkeypatch) -> None:
    chunk = TurnChunk(
        start=0.0,
        end=2.0,
        speaker="host",
        text="A brief answer.",
        turn_id=12,
        part_index=0,
        audio_file="/tmp/turn-12.mp3",
        audio_duration_s=2.0,
        audio_sha256="b" * 64,
    )
    monkeypatch.setattr(
        place,
        "decode_f32",
        lambda _path, _tempo=1.0: np.ones(round(2.0 * SR), dtype=np.float32),
    )

    fitted = place._evaluate_chunks((chunk,), total_s=10.25)

    assert fitted[0].assessment.rewrite_direction == "fuller"
    with pytest.raises(RuntimeError, match=r"place: TTS timing contract.*turn 12"):
        place._require_placeable(fitted)
