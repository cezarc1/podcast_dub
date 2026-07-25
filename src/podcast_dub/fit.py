#!/usr/bin/env python3
"""Symmetric audio-to-window fitting engine for the dub pipeline.

Policy:
  long  (D/W > LONG_OK):  pause-compress -> atempo <= TEMPO_UP  -> evaluator requests rewrite if still long
  short (D/W < SHORT_OK): pause-stretch  -> leave gap (speech is never slowed)

Speech rate is only changed within perceptual caps; silences absorb the rest.
"""

import subprocess
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from podcast_dub.audio_utils import SR, atempo_filters

LONG_OK = 1.005  # fit to essentially exact window (no tolerance overrun -> no tail trims)
SHORT_OK = 0.98
TEMPO_UP = 1.12
SIL_FLOOR = 0.12  # pause-compress: shorten silences to this
SIL_CAP = 1.5  # pause-stretch: max seconds per pause
STRETCH_MAX = 2.0  # pause-stretch: max scale factor
STRETCH_ADD_MAX = 2.0  # pause-stretch: max total seconds added per turn


class SampleInterval(NamedTuple):
    start_sample: int
    end_sample: int

    @property
    def length(self) -> int:
        return self.end_sample - self.start_sample


class AudioFitResult(NamedTuple):
    audio: np.ndarray
    notes: tuple[str, ...]


def silence_intervals(
    audio: np.ndarray, thr: float = 0.008, min_frames: int = 5, frame_ms: int = 25
) -> list[SampleInterval]:
    """Return list of [start_sample, end_sample) silence intervals."""
    frame = int(frame_ms / 1000 * SR)
    n = len(audio) // frame
    if n < 4:
        return []
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    sil = rms < thr
    out = []
    i = 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if j - i >= min_frames:
                out.append(SampleInterval(start_sample=i * frame, end_sample=j * frame))
            i = j
        else:
            i += 1
    return out


def _rebuild(
    audio: np.ndarray, intervals: Sequence[SampleInterval], new_lens: Sequence[float], fade: float = 0.008
) -> np.ndarray:
    parts, pos = [], 0
    f = np.linspace(0, 1, int(fade * SR), dtype=np.float32)
    for interval, new_len in zip(intervals, new_lens, strict=True):
        if interval.start_sample > pos:
            chunk = audio[pos : interval.start_sample].copy()
            m = min(len(f), len(chunk) // 2)
            if m > 0:
                chunk[:m] *= f[:m]
                chunk[-m:] *= f[:m][::-1]
            parts.append(chunk)
        parts.append(np.zeros(int(new_len), dtype=np.float32))
        pos = interval.end_sample
    tail = audio[pos:].copy()
    m = min(len(f), len(tail) // 2)
    if m > 0:
        tail[:m] *= f[:m]
    parts.append(tail)
    return np.concatenate(parts)


def pause_compress(audio: np.ndarray, floor: float = SIL_FLOOR) -> tuple[np.ndarray, float]:
    iv = silence_intervals(audio)
    if not iv:
        return audio, 0.0
    new_lens = [min(interval.length / SR, floor) * SR for interval in iv]
    saved = sum(interval.length - new_len for interval, new_len in zip(iv, new_lens, strict=True)) / SR
    return _rebuild(audio, iv, new_lens), saved


def pause_stretch(
    audio: np.ndarray, want_s: float, cap: float = SIL_CAP, max_scale: float = STRETCH_MAX
) -> tuple[np.ndarray, float]:
    iv = silence_intervals(audio)
    if not iv:
        return audio, 0.0
    total_sil = sum(interval.length for interval in iv) / SR
    need = max(want_s - len(audio) / SR, 0.0)
    if need <= 0 or total_sil < 0.1:
        return audio, 0.0
    scale = min(1.0 + need / total_sil, max_scale)
    new_lens = [min(interval.length * scale, cap * SR) for interval in iv]
    added = sum(new_len - interval.length for interval, new_len in zip(iv, new_lens, strict=True)) / SR
    return _rebuild(audio, iv, new_lens), added


def atempo(audio: np.ndarray, r: float) -> np.ndarray:
    if abs(r - 1.0) < 0.001:
        return audio
    p = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-i",
            "pipe:0",
            "-af",
            ",".join(atempo_filters(r)),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "pipe:1",
        ],
        input=audio.tobytes(),
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"atempo failed: {p.stderr.decode()[:300]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def fit_audio(audio: np.ndarray, window_s: float, tempo_up: float = TEMPO_UP) -> AudioFitResult:
    """Fit generated audio to window_s seconds and return the applied transformations."""
    report = []
    d = len(audio) / SR
    if d > window_s * LONG_OK:
        audio, saved = pause_compress(audio)
        if saved > 0.05:
            report.append(f"pause-compress -{saved:.1f}s")
        d = len(audio) / SR
        if d > window_s * LONG_OK:
            r = min(d / window_s, tempo_up)
            audio = atempo(audio, r)
            report.append(f"fast {r:.2f}")
    elif d < window_s * SHORT_OK:
        audio, added = pause_stretch(audio, min(window_s, d + STRETCH_ADD_MAX))
        if added > 0.05:
            report.append(f"pause-stretch +{added:.1f}s")
        # No atempo here: slowing speech sounds draggy, so it is forbidden outright;
        # so any remaining shortfall stays a natural gap and the evaluator asks for
        # fuller text instead.
    return AudioFitResult(audio=audio, notes=tuple(report))


def misfit(audio: np.ndarray, window_s: float) -> float:
    """How far the fitted audio still misses the window, as a ratio (>1 long, <1 short)."""
    return (len(audio) / SR) / window_s


def needs_rewrite(audio: np.ndarray, window_s: float, tol_long: float = 0.20, tol_short: float = 0.10) -> bool:
    """Misfit beyond repair: >tol_long over window or >tol_short under it.
    (A 2-10% shortfall is a natural conversational gap, not a defect.)"""
    m = misfit(audio, window_s)
    return m > 1 + tol_long or m < 1 - tol_short
