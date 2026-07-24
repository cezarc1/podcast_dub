#!/usr/bin/env python3
"""Symmetric audio-to-window fitting engine for the dub pipeline.

Policy:
  long  (D/W > LONG_OK):  pause-compress -> atempo <= TEMPO_UP  -> evaluator requests rewrite if still long
  short (D/W < SHORT_OK): pause-stretch  -> atempo >= TEMPO_DN -> leave gap

Speech rate is only changed within perceptual caps; silences absorb the rest.
"""

import subprocess

import numpy as np

from podcast_dub.audio_utils import atempo_filters

SR = 44100
LONG_OK = 1.005  # fit to essentially exact window (no tolerance overrun -> no tail trims)
SHORT_OK = 0.98
TEMPO_UP = 1.12
TEMPO_DN = 1.0  # NEVER slow speech: no atempo < 1.0 (rewrite fuller text instead)
SIL_FLOOR = 0.12  # pause-compress: shorten silences to this
SIL_CAP = 1.5  # pause-stretch: max seconds per pause
STRETCH_MAX = 2.0  # pause-stretch: max scale factor
STRETCH_ADD_MAX = 2.0  # pause-stretch: max total seconds added per turn


def silence_intervals(audio, thr=0.008, min_frames=5, frame_ms=25):
    """Return list of [start_sample, end_sample) silence intervals."""
    frame = int(frame_ms / 1000 * SR)
    n = len(audio) // frame
    if n < 4:
        return []
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    sil = rms < thr
    out, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if j - i >= min_frames:
                out.append([i * frame, j * frame])
            i = j
        else:
            i += 1
    return out


def _rebuild(
    audio: np.ndarray, intervals: list[tuple[int, int]], new_lens: list[float], fade: float = 0.008
) -> np.ndarray:
    parts, pos = [], 0
    f = np.linspace(0, 1, int(fade * SR), dtype=np.float32)
    for (a, b), nl in zip(intervals, new_lens, strict=True):
        if a > pos:
            chunk = audio[pos:a].copy()
            m = min(len(f), len(chunk) // 2)
            if m > 0:
                chunk[:m] *= f[:m]
                chunk[-m:] *= f[:m][::-1]
            parts.append(chunk)
        parts.append(np.zeros(int(nl), dtype=np.float32))
        pos = b
    tail = audio[pos:].copy()
    m = min(len(f), len(tail) // 2)
    if m > 0:
        tail[:m] *= f[:m]
    parts.append(tail)
    return np.concatenate(parts) if parts else audio


def pause_compress(audio: np.ndarray, floor: float = SIL_FLOOR) -> tuple[np.ndarray, float]:
    iv = silence_intervals(audio)
    if not iv:
        return audio, 0.0
    new_lens = [min((b - a) / SR, floor) * SR for a, b in iv]
    saved = sum((b - a) - nl for (a, b), nl in zip(iv, new_lens, strict=True)) / SR
    return _rebuild(audio, iv, new_lens), saved


def pause_stretch(
    audio: np.ndarray, want_s: float, cap: float = SIL_CAP, max_scale: float = STRETCH_MAX
) -> tuple[np.ndarray, float]:
    iv = silence_intervals(audio)
    if not iv:
        return audio, 0.0
    total_sil = sum(b - a for a, b in iv) / SR
    need = max(want_s - len(audio) / SR, 0.0)
    if need <= 0 or total_sil < 0.1:
        return audio, 0.0
    scale = min(1.0 + need / total_sil, max_scale)
    new_lens = [min((b - a) * scale, cap * SR) for a, b in iv]
    added = sum(nl - (b - a) for (a, b), nl in zip(iv, new_lens, strict=True)) / SR
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


def fit_audio(
    audio: np.ndarray, window_s: float, tempo_up: float = TEMPO_UP, tempo_dn: float = TEMPO_DN
) -> tuple[np.ndarray, list[str]]:
    """Fit generated audio to window_s seconds. Returns (audio, report list)."""
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
        d = len(audio) / SR
        if d < window_s * SHORT_OK:
            r = max(d / window_s, tempo_dn)
            if abs(r - 1.0) >= 0.001:
                audio = atempo(audio, r)
                report.append(f"slow {r:.2f}")
    return audio, report


def misfit(audio: np.ndarray, window_s: float) -> float:
    """How far the fitted audio still misses the window, as a ratio (>1 long, <1 short)."""
    return (len(audio) / SR) / window_s


def needs_rewrite(audio: np.ndarray, window_s: float, tol_long: float = 0.20, tol_short: float = 0.10) -> bool:
    """Misfit beyond repair: >tol_long over window or >tol_short under it.
    (A 2-10% shortfall is a natural conversational gap, not a defect.)"""
    m = misfit(audio, window_s)
    return m > 1 + tol_long or m < 1 - tol_short
