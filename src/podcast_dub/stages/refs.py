#!/usr/bin/env python3
"""Refs stage: auto-mine ~60s of clean solo reference audio per speaker.

Mines the full-timeline diarization (diar_segments.json), NOT the (possibly
windowed) ASR phrases — a speaker who barely appears inside a short job window
still gets full-length clone refs. Candidates are merged same-speaker segments
4-12s long with >=0.2s clearance from other speakers. Segments >12s are
skipped because arbitrary mid-word cuts make bad clone refs. Diarization
segments end at natural pauses, so the pieces have clean boundaries.

Input:  <workdir>/diar_segments.json + job audio
Output: <workdir>/refs/ref_<speaker>.wav (+ .txt best-effort transcript)
"""

import logging
import os
import subprocess
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from podcast_dub.artifacts import (
    build_provenance,
    file_digest,
    load_cached_artifact,
    read_artifact,
    write_artifact_atomic,
)
from podcast_dub.audio_utils import write_wav_pcm16
from podcast_dub.config import JobConfig
from podcast_dub.model_catalog import TTS_ID, TTS_REVISION
from podcast_dub.pipeline_artifacts import DIARIZATION_SEGMENTS, SPEAKER_PHRASES, SPEAKER_REFERENCES
from podcast_dub.stages.diarize import speaker_mapping
from podcast_dub.types import DiarizationSegment, ModelIdentity, SpeakerPhrase, SpeakerReference

SR = 24000
TARGET_S = 60.0
MIN_D, MAX_D = 4.0, 12.0
CLEAR_S = 0.2
JOIN_SIL_S = 0.35
MIN_TOTAL_S = 30.0  # hard floor: below this the run fails, clone quality suffers

logger = logging.getLogger(__name__)


def _select_reference_segments(
    segments: tuple[DiarizationSegment, ...],
    speaker: str,
    *,
    target_s: float = TARGET_S,
) -> tuple[tuple[DiarizationSegment, ...], float]:
    others = tuple(segment for segment in segments if segment.speaker != speaker)
    candidates = []
    for segment in segments:
        duration = segment.end - segment.start
        if segment.speaker != speaker or not (MIN_D <= duration <= MAX_D):
            continue
        if any(other.start < segment.end + CLEAR_S and other.end > segment.start - CLEAR_S for other in others):
            continue
        candidates.append((duration, segment))
    candidates.sort(key=lambda item: -item[0])
    selected: list[DiarizationSegment] = []
    total = 0.0
    for duration, segment in candidates:
        selected.append(segment)
        total += duration
        if total >= target_s:
            break
    return tuple(selected), total


def _join_with_silence(parts: Sequence[np.ndarray], silence: np.ndarray) -> np.ndarray:
    if len(parts) == 1:
        return parts[0]
    pieces: list[np.ndarray] = [parts[0]]
    for part in parts[1:]:
        pieces.append(silence)
        pieces.append(part)
    return np.concatenate(pieces)


def _build_reference_transcript(
    phrases: Sequence[SpeakerPhrase],
    segments: Sequence[DiarizationSegment],
    speaker: str,
) -> str:
    texts = [
        "".join(
            phrase.text
            for phrase in phrases
            if phrase.speaker == speaker and max(phrase.start, segment.start) < min(phrase.end, segment.end)
        ).strip()
        for segment in segments
    ]
    transcript = "。".join(text for text in texts if text)
    if not transcript:
        logger.warning("refs: no transcript text found for %s reference audio", speaker)
    return transcript


def run_refs(cfg: JobConfig) -> str:
    workdir = cfg.resolved_workdir()
    refs_dir = os.path.join(workdir, "refs")
    metadata_path = os.path.join(refs_dir, "references.json")
    os.makedirs(refs_dir, exist_ok=True)
    segments_path = os.path.join(workdir, "diar_segments.json")
    phrases_path = os.path.join(workdir, "phrases_spk.json")
    provenance_inputs = {"segments": segments_path, "audio": cfg.resolved_audio()}
    if os.path.exists(phrases_path):
        provenance_inputs["speaker_phrases"] = phrases_path
    provenance = build_provenance(
        cfg,
        input_files=provenance_inputs,
        parameters={
            "target_s": TARGET_S,
            "minimum_s": MIN_TOTAL_S,
            "duration_range": (MIN_D, MAX_D),
            "clearance_s": CLEAR_S,
            "join_silence_s": JOIN_SIL_S,
        },
        model=ModelIdentity(identifier=TTS_ID, revision=TTS_REVISION),
    )
    cached = load_cached_artifact(metadata_path, SPEAKER_REFERENCES, provenance)
    if cached is not None and all(
        os.path.exists(reference.audio_file) and file_digest(reference.audio_file) == reference.audio_sha256
        for reference in cached
    ):
        logger.info("refs: cached %s", metadata_path)
        return refs_dir

    raw = read_artifact(segments_path, DIARIZATION_SEGMENTS).payload
    merged, mapping, _ = speaker_mapping(raw, cfg.speaker_names)
    audio_path = cfg.resolved_audio()
    phrases = read_artifact(phrases_path, SPEAKER_PHRASES).payload if os.path.exists(phrases_path) else ()
    references: list[SpeakerReference] = []
    progress = tqdm(mapping.items(), desc="refs", unit="speaker")
    for raw_spk, disp in progress:
        wav_out = os.path.join(refs_dir, f"ref_{disp}.wav")
        txt_out = os.path.join(refs_dir, f"ref_{disp}.txt")
        picks, total = _select_reference_segments(merged, raw_spk)
        if total < MIN_TOTAL_S:
            raise RuntimeError(
                f"refs: only {total:.0f}s clean audio for {disp}; at least {MIN_TOTAL_S:.0f}s is required"
            )

        ordered_picks = sorted(picks, key=lambda item: item.start)
        parts = []
        for segment in ordered_picks:
            raw_pcm = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    str(segment.start),
                    "-t",
                    str(segment.end - segment.start),
                    "-i",
                    audio_path,
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-ar",
                    str(SR),
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            ).stdout
            parts.append(np.frombuffer(raw_pcm, dtype=np.int16))
        silence = np.zeros(int(JOIN_SIL_S * SR), dtype=np.int16)
        joined = _join_with_silence(parts, silence)
        write_wav_pcm16(wav_out, joined, SR)
        # best-effort transcript (unused by x_vector_only cloning; for records)
        with open(txt_out, "w", encoding="utf-8") as transcript_file:
            transcript_file.write(_build_reference_transcript(phrases, ordered_picks, disp))
        references.append(
            SpeakerReference(
                speaker=disp,
                audio_file=wav_out,
                transcript_file=txt_out,
                duration_s=len(joined) / SR,
                audio_sha256=file_digest(wav_out),
            )
        )
        progress.set_postfix_str(f"{disp} {total:.0f}s / {len(picks)} segs")
    write_artifact_atomic(metadata_path, SPEAKER_REFERENCES, provenance, tuple(references))
    return refs_dir
