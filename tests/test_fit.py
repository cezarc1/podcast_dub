"""Unit tests for the symmetric fit engine (podcast_dub.fit)."""

import numpy as np

from podcast_dub.fit import SR, atempo, fit_audio, needs_rewrite, pause_compress, pause_stretch, silence_intervals


def beep(dur_s, freq=440.0):
    t = np.arange(int(dur_s * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speech_like(n_phrases=4, phrase_s=1.0, gap_s=0.5):
    parts = []
    for i in range(n_phrases):
        parts.append(beep(phrase_s, 300 + 50 * i))
        if i < n_phrases - 1:
            parts.append(np.zeros(int(gap_s * SR), dtype=np.float32))
    return np.concatenate(parts)


def dur(a):
    return len(a) / SR


class TestSilenceDetect:
    def test_finds_gaps(self):
        a = speech_like(3, 1.0, 0.5)
        iv = silence_intervals(a)
        assert len(iv) == 2
        for x, y in iv:
            assert 0.4 < (y - x) / SR < 0.6

    def test_no_gaps_in_continuous(self):
        assert silence_intervals(beep(2.0)) == []


class TestPauseOps:
    def test_compress_shorter(self):
        a = speech_like(3, 1.0, 0.8)
        b, saved = pause_compress(a)
        assert dur(b) < dur(a)
        assert saved > 0.5

    def test_stretch_longer(self):
        a = speech_like(3, 1.0, 0.4)
        want = dur(a) + 1.0
        b, added = pause_stretch(a, want)
        assert dur(b) > dur(a)
        assert added > 0.5

    def test_speech_content_preserved(self):
        a = speech_like(4, 1.0, 0.5)
        b, _ = pause_compress(a)
        # speech still present: energy roughly conserved (only silences removed)
        assert np.abs(b).mean() > np.abs(a).mean() * 0.8


class TestFitPolicy:
    def test_long_within_reach_fits(self):
        a = speech_like(6, 1.5, 0.5)  # ~11.5s into a 10s window (1.15x)
        b, rep = fit_audio(a, 10.0)
        assert dur(b) < dur(a)
        assert dur(b) <= 10.0 * 1.005 + 0.05
        assert not needs_rewrite(b, 10.0)
        assert rep

    def test_long_beyond_cap_is_accepted_or_flagged(self):
        a = speech_like(6, 2.0, 0.8)  # ~16s into a 10s window (1.6x)
        b, rep = fit_audio(a, 10.0)
        assert dur(b) < dur(a)
        # policy: full cap applied; leftover misfit either <=20% or flagged
        from podcast_dub.fit import misfit

        assert misfit(b, 10.0) <= 1.2 or needs_rewrite(b, 10.0)
        assert rep

    def test_short_within_stretch_cap_fits(self):
        a = speech_like(5, 1.5, 0.5)  # ~9.5s... in a 10.5s window -> +1s stretch
        b, rep = fit_audio(a, 10.5)
        assert dur(b) > dur(a)
        assert dur(b) >= 10.5 * 0.98 - 0.05
        assert not needs_rewrite(b, 10.5)
        assert rep

    def test_short_beyond_stretch_cap_goes_to_rewrite(self):
        a = speech_like(3, 1.5, 0.5)  # ~5.5s into a 10s window: needs +4.5s
        b, rep = fit_audio(a, 10.0)
        assert dur(b) > dur(a)  # stretched the allowed +2s
        assert needs_rewrite(b, 10.0)  # remainder (>10% short) -> DSPy.Refine fuller text

    def test_short_beyond_reach_is_flagged(self):
        a = speech_like(3, 0.8, 0.3)  # ~3.2s into a 10s window (0.32x)
        b, rep = fit_audio(a, 10.0)
        assert dur(b) > dur(a)
        assert needs_rewrite(b, 10.0)  # can't stretch 3x: rewrite with fuller text

    def test_in_window_untouched(self):
        a = speech_like(4, 1.0, 0.3)  # ~4.9s; window 5.0s
        b, rep = fit_audio(a, 5.0)
        assert np.allclose(dur(b), dur(a), atol=0.05)
        assert rep == []

    def test_rewrite_flag(self):
        assert needs_rewrite(beep(20.0), 10.0)
        assert needs_rewrite(beep(3.0), 10.0)
        assert not needs_rewrite(beep(9.5), 10.0)


class TestAtempo:
    def test_duration_scales(self):
        a = beep(4.0)
        assert abs(dur(atempo(a, 2.0)) - 2.0) < 0.1
        assert abs(dur(atempo(a, 0.5)) - 8.0) < 0.1
