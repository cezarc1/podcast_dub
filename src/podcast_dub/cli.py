#!/usr/bin/env python3
"""podcast_dub CLI — generic entry point for the pipeline.

Usage:
  podcast_dub <video> --from zh --to en [--config dub.toml] \
      [--names host,guest] [--output out.mp4] [--workdir DIR]

Stages (each resumable; artifacts cached in the job workdir):
  probe     -> extract audio + duration from the input video
  asr       -> Qwen3-ASR + ForcedAligner -> phrase units with word timestamps
  diarize   -> NVIDIA Sortformer diarization -> speaker labels per phrase
  refs      -> auto-mine ~60s clean solo reference audio per speaker
  translate -> DSPy spoken-ASR translation with preceding conversation turns
  tts       -> Qwen3-TTS clones + measurement-verified rewrite loop
  place     -> sentence-flow placement + ducked bed -> final mp4

This file is the orchestrator; stages live in their own modules.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Final

from pydantic import ValidationError

from podcast_dub.audio_utils import dur_of
from podcast_dub.config import JobConfig, JobConfigInput, load_toml, merge_cli
from podcast_dub.device_utils import resolve_device_plan
from podcast_dub.models import ModelStage
from podcast_dub.stages.asr import run_asr
from podcast_dub.stages.diarize import run_diarize
from podcast_dub.stages.place import run_place
from podcast_dub.stages.refs import run_refs
from podcast_dub.stages.translate import run_translate
from podcast_dub.stages.tts import run_tts

logger = logging.getLogger(__name__)


def fetch_if_url(cfg: JobConfig) -> JobConfig:
    """Download the input video into the workdir when cfg.video is a URL."""
    if not cfg.video.startswith(("http://", "https://")):
        return cfg
    workdir = cfg.resolved_workdir()
    os.makedirs(workdir, exist_ok=True)
    existing = [f for f in os.listdir(workdir) if f.startswith("input.")]
    if existing:
        fetched = cfg.validated_copy(video=os.path.join(workdir, existing[0]))
        print(f"fetch: cached {fetched.video}")
        return fetched
    tmpl = os.path.join(workdir, "input.%(ext)s")
    subprocess.run(["yt-dlp", "-f", "bv*+ba/b", "-o", tmpl, cfg.video], check=True)
    got = [f for f in os.listdir(workdir) if f.startswith("input.")]
    if len(got) != 1:
        sys.exit(f"fetch: expected one input.* in {workdir}, got {got}")
    fetched = cfg.validated_copy(video=os.path.join(workdir, got[0]))
    print(f"fetch: downloaded -> {fetched.video}")
    return fetched


def probe(cfg: JobConfig) -> float:
    """Extract audio track to the workdir; return duration in seconds."""
    workdir = cfg.resolved_workdir()
    os.makedirs(workdir, exist_ok=True)
    audio = cfg.resolved_audio()
    if not os.path.exists(audio):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", cfg.video, "-vn", "-ac", "1", "-ar", "16000", audio], check=True
        )
    duration = dur_of(cfg.video)
    print(f"probe: {cfg.video} -> {audio} ({duration:.1f}s)")
    return duration


def main():
    ap = argparse.ArgumentParser(
        prog="podcast_dub", description="Dub a podcast video into another language, preserving voices."
    )
    ap.add_argument("video", nargs="?", help="input video file path")
    ap.add_argument("--from", dest="source_lang", help="source language code (e.g. zh)")
    ap.add_argument("--to", dest="target_lang", help="target language code (e.g. en)")
    ap.add_argument("--config", dest="config", help="dub.toml job file path")
    ap.add_argument("--names", dest="names", help="comma-separated speaker display names")
    ap.add_argument("--output", dest="output", help="final mp4 file path")
    ap.add_argument("--workdir", dest="workdir", help="artifacts directory path")
    ap.add_argument(
        "--stages",
        dest="stages",
        default=",".join(["probe", *STAGES]),
        help="comma-separated stages to run (default: all)",
    )
    args = ap.parse_args()

    cfg_input = load_toml(args.config) if args.config else JobConfigInput()
    try:
        cfg = merge_cli(cfg_input, args)
    except ValidationError as exc:
        print(f"error: invalid job configuration\n{exc}", file=sys.stderr)
        print("\nusage: podcast_dub <video> --from <lang> --to <lang> [--config dub.toml]", file=sys.stderr)
        sys.exit(2)
    cfg = fetch_if_url(cfg)
    problems = cfg.validation_problems()
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        print("\nusage: podcast_dub <video> --from <lang> --to <lang> [--config dub.toml]", file=sys.stderr)
        sys.exit(2)

    print(f"job: {os.path.basename(cfg.video)}  {cfg.source_lang} -> {cfg.target_lang}")
    print(f"workdir: {cfg.resolved_workdir()}")
    print(f"output: {cfg.resolved_output()}")
    stages = [s.strip() for s in args.stages.split(",")]
    print(f"stages: {', '.join(stages)}")
    plans = (
        resolve_device_plan(ModelStage.ASR, cfg.asr_device),
        resolve_device_plan(ModelStage.DIARIZE, cfg.diarize_device),
        resolve_device_plan(ModelStage.TTS, cfg.tts_device),
    )
    print(
        "devices: "
        + ", ".join(
            f"{plan.stage}={plan.device}/{plan.dtype}/{plan.attention}" for plan in plans if plan.stage in stages
        )
    )

    duration = probe(cfg) if "probe" in stages else cfg.window_s
    if not duration:
        duration = dur_of(cfg.video)
    if not cfg.window_s:
        cfg = cfg.validated_copy(window_s=duration + 0.5)

    # stage dispatch (each implemented in its own module; later stages need earlier artifacts)
    for stage in stages:
        if stage == "probe":
            continue
        fn = STAGES.get(stage)
        if fn is None:
            sys.exit(f"unknown stage: {stage}")
        logger.info("Running %s stage...", stage)
        started = time.monotonic()
        fn(cfg)
        logger.info("%s stage completed in %.2f seconds", stage, time.monotonic() - started)


STAGES: Final[dict[str, Callable[[JobConfig], str]]] = {
    "asr": run_asr,
    "diarize": run_diarize,
    "refs": run_refs,
    "translate": run_translate,
    "tts": run_tts,
    "place": run_place,
}
