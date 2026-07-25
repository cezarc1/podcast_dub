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
import tomllib
from collections.abc import Callable
from typing import Final

from pydantic import ValidationError

from podcast_dub.audio_utils import dur_of
from podcast_dub.branding import show_banner
from podcast_dub.config import JobConfig, JobConfigInput, load_toml, merge_cli
from podcast_dub.device_utils import resolve_device_plan
from podcast_dub.logging_config import configure_logging
from podcast_dub.stages.asr import run_asr
from podcast_dub.stages.diarize import run_diarize
from podcast_dub.stages.place import run_place
from podcast_dub.stages.refs import run_refs
from podcast_dub.stages.translate import run_translate
from podcast_dub.stages.tts import run_tts
from podcast_dub.types import ModelStage

logger = logging.getLogger(__name__)

# yt-dlp leaves scratch files beside the real download (input.mp4.part,
# input.f137.mp4.ytdl, input.mp4.part-Frag3); adopting one as the job input
# would feed the pipeline a truncated file.
_PARTIAL_SUFFIXES: Final[tuple[str, ...]] = (".part", ".ytdl", ".temp", ".tmp")


def _is_partial(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(_PARTIAL_SUFFIXES) or ".part-frag" in lowered


def _complete_inputs(workdir: str) -> list[str]:
    """Return the finished ``input.*`` downloads in workdir, skipping partials."""
    return sorted(name for name in os.listdir(workdir) if name.startswith("input.") and not _is_partial(name))


def fetch_if_url(cfg: JobConfig) -> JobConfig:
    """Download the input video into the workdir when cfg.video is a URL."""
    if not cfg.video.startswith(("http://", "https://")):
        return cfg
    workdir = cfg.resolved_workdir()
    os.makedirs(workdir, exist_ok=True)
    inputs = _complete_inputs(workdir)
    action = "cached"
    if not inputs:
        tmpl = os.path.join(workdir, "input.%(ext)s")
        subprocess.run(["yt-dlp", "-f", "bv*+ba/b", "-o", tmpl, cfg.video], check=True)
        inputs = _complete_inputs(workdir)
        action = "downloaded ->"
    # An interrupted merge can leave yt-dlp's per-format streams (input.f137.mp4,
    # input.f140.m4a) behind; adopting one silently would dub a video-only input.
    if len(inputs) != 1:
        sys.exit(f"fetch: expected one input.* in {workdir}, got {inputs}")
    fetched = cfg.validated_copy(video=os.path.join(workdir, inputs[0]))
    logger.info(f"fetch: {action} {fetched.video}")
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
    logger.info(f"probe: {cfg.video} -> {audio} ({duration:.1f}s)")
    return duration


def main() -> None:
    configure_logging()
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

    try:
        cfg_input = load_toml(args.config) if args.config else JobConfigInput()
        cfg = merge_cli(cfg_input, args)
    except (ValidationError, tomllib.TOMLDecodeError, OSError) as exc:
        logger.error(f"error: invalid job configuration\n{exc}")
        logger.error("\nusage: podcast_dub <video> --from <lang> --to <lang> [--config dub.toml]")
        sys.exit(2)
    cfg = fetch_if_url(cfg)
    if problems := cfg.validation_problems():
        for p in problems:
            logger.error(f"error: {p}")
        logger.error("\nusage: podcast_dub <video> --from <lang> --to <lang> [--config dub.toml]")
        sys.exit(2)

    stages = [s.strip() for s in args.stages.split(",")]
    # Validate the whole selection up front: a typo in a later stage must not cost a
    # full run of the earlier ones before it is reported.
    if unknown := [stage for stage in stages if stage not in _KNOWN_STAGES]:
        sys.exit(f"unknown stage: {', '.join(unknown)}")

    show_banner()
    logger.info(f"job: {os.path.basename(cfg.video)}  {cfg.source_lang} -> {cfg.target_lang}")
    logger.info(f"workdir: {cfg.resolved_workdir()}")
    logger.info(f"output: {cfg.resolved_output()}")
    logger.info(f"stages: {', '.join(stages)}")
    plans = (
        resolve_device_plan(ModelStage.ASR, cfg.asr_device),
        resolve_device_plan(ModelStage.DIARIZE, cfg.diarize_device),
        resolve_device_plan(ModelStage.TTS, cfg.tts_device),
    )
    logger.info(
        "devices: "
        + ", ".join(
            f"{plan.stage}={plan.device}/{plan.dtype}/{plan.attention}" for plan in plans if plan.stage in stages
        )
    )

    if not (duration := probe(cfg) if "probe" in stages else cfg.window_s):
        duration = dur_of(cfg.video)
    if not cfg.window_s:
        cfg = cfg.validated_copy(window_s=duration + 0.5)

    # stage dispatch (each implemented in its own module; later stages need earlier artifacts)
    for stage in stages:
        if stage == "probe":
            continue
        fn = STAGES[stage]
        logger.info("Running %s stage...", stage)
        started = time.monotonic()
        try:
            fn(cfg)
        except RuntimeError as exc:
            logger.error(f"error: {exc}")
            sys.exit(1)
        logger.info("%s stage completed in %.2f seconds", stage, time.monotonic() - started)


STAGES: Final[dict[str, Callable[[JobConfig], str]]] = {
    "asr": run_asr,
    "diarize": run_diarize,
    "refs": run_refs,
    "translate": run_translate,
    "tts": run_tts,
    "place": run_place,
}

_KNOWN_STAGES: Final[frozenset[str]] = frozenset({"probe", *STAGES})
