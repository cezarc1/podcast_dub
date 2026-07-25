#!/usr/bin/env python3
"""Place validated TTS chunks and publish only a verified final dub."""

from __future__ import annotations

import logging
import os
import subprocess

import numpy as np
from tqdm import tqdm

from podcast_dub.artifacts import build_provenance, load_cached_artifact, read_artifact, write_artifact_atomic
from podcast_dub.audio_utils import SR, decode_f32, dur_of, srt_ts, write_wav_pcm16
from podcast_dub.config import JobConfig
from podcast_dub.pipeline_artifacts import PLACEMENT_RESULT, TURN_CHUNKS
from podcast_dub.stages.tts import MAX_REWRITE_WORDS
from podcast_dub.stages.verify import verify_media
from podcast_dub.timing import (
    TIMING_POLICY_VERSION,
    FittedTurn,
    evaluate_timeline,
    group_by_logical_turns,
    rewrite_can_help,
)
from podcast_dub.types import PlacementResult, TurnChunk

BED_FILTER = (
    "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.01:ratio=20:"
    "attack=15:release=650:makeup=1[duck];[duck]volume=0.35[bed];"
    "[bed][mix]amix=inputs=2:normalize=0,alimiter=limit=0.89[out]"
)

logger = logging.getLogger(__name__)


def _mix_add(mix: np.ndarray, offset: int, audio: np.ndarray) -> int:
    room = max(len(mix) - offset, 0)
    if trimmed := max(len(audio) - room, 0):
        audio = audio[:room]
    mix[offset : offset + len(audio)] += audio
    return trimmed


def _bed_is_stale(bed_path: str, video_path: str) -> bool:
    if not os.path.exists(bed_path):
        return True
    return os.path.getmtime(bed_path) < os.path.getmtime(video_path)


def _evaluate_chunks(chunks: tuple[TurnChunk, ...], *, total_s: float) -> tuple[FittedTurn, ...]:
    return evaluate_timeline(
        group_by_logical_turns(chunks, stage="place"),
        total_s,
        lambda path: decode_f32(str(path), 1.0),
        stage="place",
    )


def _require_placeable(fitted_turns: tuple[FittedTurn, ...]) -> None:
    failures = [
        item.assessment
        for item in fitted_turns
        if item.assessment.rewrite_direction is not None and rewrite_can_help(item, word_ceiling=MAX_REWRITE_WORDS)
    ]
    if not failures:
        return
    details = ", ".join(
        f"turn {item.turn_id}: {item.fitted_duration_s:.2f}s/{item.window_s:.2f}s ({item.rewrite_direction})"
        for item in failures
    )
    raise RuntimeError(f"place: TTS timing contract is not satisfied: {details}")


def run_place(cfg: JobConfig) -> str:
    workdir = cfg.resolved_workdir()
    turns_path = os.path.join(workdir, "turns.json")
    result_path = os.path.join(workdir, "placement.json")
    provenance = build_provenance(
        cfg,
        input_files={"turns": turns_path, "video": cfg.video},
        parameters={"timing_policy_version": TIMING_POLICY_VERSION, "bed_filter": BED_FILTER},
    )
    if (cached := load_cached_artifact(result_path, PLACEMENT_RESULT, provenance)) is not None:
        required = (cached.output_file, cached.voice_file, cached.mix_file, cached.subtitles_file)
        if all(os.path.exists(path) for path in required):
            verification = verify_media(cfg.resolved_audio(), cached.voice_file)
            if verification.passed:
                logger.info("place: cached %s", cached.output_file)
                return cached.output_file

    chunks = read_artifact(turns_path, TURN_CHUNKS).payload
    if not chunks:
        raise RuntimeError("place: no TTS chunks to place")
    total = cfg.window_s or max(chunk.end for chunk in chunks) + 0.5
    fitted_turns = _evaluate_chunks(chunks, total_s=total)
    _require_placeable(fitted_turns)
    logical = tuple(item.logical_turn for item in fitted_turns)

    mix = np.zeros(int(total * SR), dtype=np.float32)
    progress = tqdm(fitted_turns, desc="place", unit="turn")
    for fitted_turn in progress:
        logical_turn = fitted_turn.logical_turn
        assessment = fitted_turn.assessment
        audio = fitted_turn.audio
        offset = int(assessment.start_s * SR)
        if trimmed := _mix_add(mix, offset, audio):
            progress.write(
                f"place: t{logical_turn.turn_id} tail trimmed {trimmed / SR:.2f}s past the end of the mix buffer"
            )
        if assessment.fit_notes:
            progress.write(
                f"place: t{logical_turn.turn_id} ({logical_turn.speaker}, {len(logical_turn.chunks)} chunk(s)): "
                f"dur {assessment.fitted_duration_s:.1f}s {' '.join(assessment.fit_notes)} "
                f"lag {assessment.lag_s:.1f} [win {assessment.window_s:.2f}]"
            )
        progress.set_postfix_str(f"t{logical_turn.turn_id} {logical_turn.speaker} lag{assessment.lag_s:+.1f}")

    if (peak := float(np.max(np.abs(mix)))) > 0.89:
        mix *= 0.89 / peak
    voice = os.path.join(workdir, "dub_voice.wav")
    write_wav_pcm16(voice, (np.clip(mix, -1, 1) * 32767).astype(np.int16), SR)

    subtitles = os.path.join(workdir, f"dub_{cfg.target_lang}.srt")
    ordered_chunks = [chunk for logical_turn in logical for chunk in logical_turn.chunks]
    with open(subtitles, "w", encoding="utf-8") as subtitle_file:
        for sequence, chunk in enumerate(ordered_chunks, start=1):
            subtitle_file.write(f"{sequence}\n{srt_ts(chunk.start)} --> {srt_ts(chunk.end)}\n{chunk.text}\n\n")

    bed_source = os.path.join(workdir, "bed.wav")
    if _bed_is_stale(bed_source, cfg.video):
        temporary_bed = bed_source + ".tmp.wav"
        subprocess.run(
            # the bed must share the voice track's rate for BED_FILTER's amix
            ["ffmpeg", "-v", "error", "-y", "-i", cfg.video, "-vn", "-ac", "1", "-ar", str(SR), temporary_bed],
            check=True,
        )
        os.replace(temporary_bed, bed_source)
    mixed = os.path.join(workdir, "dub_mix.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            bed_source,
            "-i",
            voice,
            "-filter_complex",
            BED_FILTER,
            "-map",
            "[out]",
            mixed,
        ],
        check=True,
    )

    verification = verify_media(cfg.resolved_audio(), voice)
    if not verification.passed:
        raise RuntimeError(
            f"place: verification failed: coverage={verification.coverage:.3f}, "
            f"longest_dead_air_s={verification.longest_dead_air_s:.2f}"
        )

    output = cfg.resolved_output()
    temporary_output = output + ".tmp.mp4"
    mux = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        cfg.video,
        "-i",
        mixed,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
    ]
    source_duration = dur_of(cfg.video)
    if total < source_duration - 1:
        mux += ["-t", str(total + 2)]
    mux += [temporary_output]
    subprocess.run(mux, check=True)
    os.replace(temporary_output, output)

    result = PlacementResult(
        output_file=output,
        voice_file=voice,
        mix_file=mixed,
        subtitles_file=subtitles,
        verification=verification,
    )
    write_artifact_atomic(result_path, PLACEMENT_RESULT, provenance, result)
    logger.info("place: wrote %s", output)
    return output
