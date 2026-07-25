"""Unit tests for podcast_dub.audio_utils (pure helpers only: no ffmpeg, no network)."""

import math
import wave
from pathlib import Path

import numpy as np

from podcast_dub.audio_utils import SR, atempo_filters, srt_ts, write_wav_pcm16


def _tempo_product(filters: list[str]) -> float:
    product = 1.0
    for f in filters:
        assert f.startswith("atempo=")
        product *= float(f.split("=", 1)[1])
    return product


class TestAtempoFilters:
    def test_single_factor_under_cap(self) -> None:
        filters = atempo_filters(1.5)
        assert filters == ["atempo=1.50000"]
        assert math.isclose(_tempo_product(filters), 1.5, rel_tol=1e-6)

    def test_chains_above_cap(self) -> None:
        filters = atempo_filters(5.0)
        assert filters == ["atempo=2.0", "atempo=2.0", "atempo=1.25000"]
        assert all(float(f.split("=", 1)[1]) <= 2.0 for f in filters)
        assert math.isclose(_tempo_product(filters), 5.0, rel_tol=1e-6)


class TestSrtTs:
    def test_zero(self) -> None:
        assert srt_ts(0.0) == "00:00:00,000"

    def test_hours_minutes_seconds_millis(self) -> None:
        assert srt_ts(3661.507) == "01:01:01,507"

    def test_rolls_over_instead_of_emitting_second_60(self) -> None:
        assert srt_ts(59.9996) == "00:01:00,000"


class TestWriteWavPcm16:
    def test_round_trip(self, tmp_path: Path) -> None:
        pcm = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        out = tmp_path / "tone.wav"
        write_wav_pcm16(out, pcm, SR)

        with wave.open(str(out), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == SR
            assert wav_file.getnframes() == len(pcm)
            frames = wav_file.readframes(wav_file.getnframes())

        assert np.array_equal(np.frombuffer(frames, dtype="<i2"), pcm)
