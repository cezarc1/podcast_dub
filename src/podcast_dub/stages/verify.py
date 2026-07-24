#!/usr/bin/env python3
"""Verify a dub build: speech coverage + dead-air scan (the v5 gate, generic).

Compares <workdir>/dub_voice.wav (placed dub speech track) against the job's
original audio: wherever the ORIGINAL has speech, the dub should have speech
too, within a placement-tolerance window (the dub may legitimately lag).
Reports coverage and contiguous dead-air runs longer than DEAD_S.

Usage:
    python -m podcast_dub.stages.verify <workdir> [--dead-s 5] [--tol-s 1.5] [--min-coverage 0.99]

Exit code 1 when coverage < --min-coverage or any dead-air run > --dead-s
exists (usable as a build gate).
"""

import argparse
import os
import sys

import numpy as np

from podcast_dub.audio_utils import decode_f32
from podcast_dub.models import VerificationResult

SR = 16000
FRAME = int(0.025 * SR)


def load_mono(path: str, sr: int = SR) -> np.ndarray:
    return decode_f32(path, tempo=None, sr=sr)


def active_mask(audio: np.ndarray, floor: float, rel: float = 0.15) -> np.ndarray:
    n = len(audio) // FRAME
    rms = np.sqrt((audio[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
    thr = max(floor, rel * float(np.median(rms)))
    return rms > thr


def dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    """True wherever mask is True within +-frames (max filter, placement lag)."""
    if frames <= 0:
        return mask
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask)
    for i in idx:  # sparse: active frames are the minority in dub tracks
        out[max(0, i - frames) : i + frames + 1] = True
    return out


def runs(dead: np.ndarray, fps: float) -> list[tuple[float, float]]:
    out, i = [], 0
    while i < len(dead):
        if dead[i]:
            j = i
            while j < len(dead) and dead[j]:
                j += 1
            out.append((i / fps, j / fps))
            i = j
        else:
            i += 1
    return out


def evaluate_masks(
    original: np.ndarray,
    dubbed: np.ndarray,
    *,
    tolerance_frames: int,
    frames_per_second: float,
    max_dead_s: float,
    min_coverage: float = 0.99,
) -> VerificationResult:
    n = min(len(dubbed), len(original))
    dubbed, original = dubbed[:n], original[:n]
    dubbed_near = dilate(dubbed, tolerance_frames)
    coverage = float((original & dubbed_near).sum()) / max(int(original.sum()), 1)
    dead_runs = runs(original & ~dubbed_near, frames_per_second)
    longest_dead_air_s = max((end - start for start, end in dead_runs), default=0.0)
    return VerificationResult(
        coverage=coverage,
        longest_dead_air_s=longest_dead_air_s,
        passed=coverage >= min_coverage and longest_dead_air_s <= max_dead_s,
    )


def verify_media(
    original_path: str,
    dub_path: str,
    *,
    dead_s: float = 5.0,
    tolerance_s: float = 1.5,
    min_coverage: float = 0.99,
) -> VerificationResult:
    dubbed = active_mask(load_mono(dub_path), floor=0.005)
    original = active_mask(load_mono(original_path), floor=0.006)
    fps = SR / FRAME
    return evaluate_masks(
        original,
        dubbed,
        tolerance_frames=int(tolerance_s * fps),
        frames_per_second=fps,
        max_dead_s=dead_s,
        min_coverage=min_coverage,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--dead-s", type=float, default=5.0)
    ap.add_argument("--tol-s", type=float, default=1.5)
    ap.add_argument("--min-coverage", type=float, default=0.99)
    ap.add_argument("--original", default=None, help="original audio path (default: <workdir>/<stem>.wav from probe)")
    a = ap.parse_args()

    dub_path = os.path.join(a.workdir, "dub_voice.wav")
    orig_path = a.original
    if orig_path is None:
        wavs = [
            f
            for f in os.listdir(a.workdir)
            if f.endswith(".wav") and f not in ("dub_voice.wav", "dub_mix.wav", "bed.wav")
        ]
        if len(wavs) != 1:
            sys.exit(f"cannot infer original audio in {a.workdir}: {wavs}")
        orig_path = os.path.join(a.workdir, wavs[0])

    result = verify_media(
        orig_path,
        dub_path,
        dead_s=a.dead_s,
        tolerance_s=a.tol_s,
        min_coverage=a.min_coverage,
    )
    print(f"coverage (dub active within ±{a.tol_s}s): {result.coverage * 100:.1f}%")
    print(f"longest dead-air run: {result.longest_dead_air_s:.1f}s")
    print("PASS" if result.passed else "FAIL")
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
