#!/usr/bin/env python3
"""Build typed turns, synthesize validated audio, and persist TTS artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from itertools import batched, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

import podcast_dub.manifest as manifest
from podcast_dub.artifacts import (
    ArtifactError,
    atomic_write_text,
    build_provenance,
    file_digest,
    load_cached_artifact,
    read_artifact,
    stable_digest,
    write_artifact_atomic,
)
from podcast_dub.audio_utils import decode_f32, dur_of
from podcast_dub.config import JobConfig, lang_name, resolve_translation_api
from podcast_dub.device_utils import model_kwargs_for, resolve_device_plan
from podcast_dub.model_catalog import TTS_ID, TTS_REVISION, validate_model_snapshot
from podcast_dub.pipeline_artifacts import SPEAKER_REFERENCES, TRANSLATION_UNITS, TURN_CHUNKS
from podcast_dub.timing import (
    TIMING_POLICY_VERSION,
    FittedTurn,
    evaluate_timeline,
    group_by_logical_turns,
    rewrite_can_help,
)
from podcast_dub.translate import Translator
from podcast_dub.translate import make_translator as _make
from podcast_dub.types import (
    ModelIdentity,
    ModelStage,
    RewriteCache,
    RewriteCacheEntry,
    RewriteEvent,
    TranslationUnit,
    TurnChunk,
    TurnChunkDraft,
)

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel

SENT_END = re.compile(r'[.!?…]["\')\]]?\s*$')
TURN_GAP = 2.5
MIN_UNIT_SPAN_S = 0.3
MAX_CHUNK_S = 110.0
# Assumed delivered speaking rate; converts between word counts and seconds.
WORDS_PER_SECOND = 2.5
MAX_REWRITE_WORDS = round(MAX_CHUNK_S * WORDS_PER_SECOND)
REWRITE_CACHE_VERSION = 6
MAX_REWRITE_ATTEMPTS = 3
VOCAL_ONSET_POLICY_VERSION = 2
MIN_CACHED_AUDIO_BYTES = 1000

logger = logging.getLogger(__name__)


def _split_target_text(text: str) -> tuple[str, ...]:
    sentences = tuple(part for part in re.split(r"(?<=[.!?…])\s+", text) if part)
    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > MAX_REWRITE_WORDS:
            if current:
                parts.append(" ".join(current))
                current = []
                current_words = 0
            parts.extend(" ".join(batch) for batch in batched(words, MAX_REWRITE_WORDS))
        elif current and current_words + len(words) > MAX_REWRITE_WORDS:
            parts.append(" ".join(current))
            current = [sentence]
            current_words = len(words)
        else:
            current.append(sentence)
            current_words += len(words)
    if current:
        parts.append(" ".join(current))
    return tuple(parts)


def _partition_source_text(source_text: str, weights: tuple[int, ...]) -> tuple[str, ...]:
    tokens = source_text.split()
    separator = " "
    if len(tokens) < len(weights):
        tokens = list(source_text)
        separator = ""
    if len(tokens) < len(weights):
        raise RuntimeError("tts: cannot preserve source mapping while splitting an oversized translation unit")

    boundaries = [0]
    cumulative_weight = 0
    total_weight = sum(weights)
    for index, weight in enumerate(weights[:-1]):
        cumulative_weight += weight
        boundary = round(len(tokens) * cumulative_weight / total_weight)
        boundary = max(boundaries[-1] + 1, boundary)
        boundary = min(len(tokens) - (len(weights) - index - 1), boundary)
        boundaries.append(boundary)
    boundaries.append(len(tokens))
    return tuple(separator.join(tokens[start:end]).strip() for start, end in pairwise(boundaries))


def _split_oversized_unit(unit: TranslationUnit) -> tuple[TranslationUnit, ...]:
    if len(unit.target_text.split()) / WORDS_PER_SECOND <= MAX_CHUNK_S:
        return (unit,)

    target_parts = _split_target_text(unit.target_text)
    weights = tuple(len(part.split()) for part in target_parts)
    source_parts = _partition_source_text(unit.source_text, weights)
    start = unit.start
    end = unit.end if unit.end > unit.start else unit.start + MIN_UNIT_SPAN_S
    duration = end - start
    total_weight = sum(weights)
    offset = start
    fragments: list[TranslationUnit] = []
    cumulative_weight = 0
    for index, (source, target, weight) in enumerate(zip(source_parts, target_parts, weights, strict=True)):
        cumulative_weight += weight
        fragment_end = end if index == len(target_parts) - 1 else start + duration * cumulative_weight / total_weight
        fragments.append(
            TranslationUnit(
                start=offset,
                end=fragment_end,
                speaker=unit.speaker,
                source_text=source,
                target_text=target,
            )
        )
        offset = fragment_end
    return tuple(fragments)


def _balance_turn_parts(turn: list[TranslationUnit], part_count: int) -> list[list[TranslationUnit]]:
    """Partition ordered units near equal word capacity without crossing the per-chunk ceiling."""
    if part_count <= 1:
        return [turn]

    remaining_words = sum(len(unit.target_text.split()) for unit in turn)
    start = 0
    parts: list[list[TranslationUnit]] = []
    for part_index in range(part_count):
        parts_left = part_count - part_index
        if parts_left == 1:
            parts.append(turn[start:])
            break

        target_words = remaining_words / parts_left
        minimum_words = max(0, remaining_words - MAX_REWRITE_WORDS * (parts_left - 1))
        latest_end = len(turn) - (parts_left - 1)
        end = start
        part_words = 0
        while end < latest_end:
            next_words = len(turn[end].target_text.split())
            if end > start and part_words + next_words > MAX_REWRITE_WORDS:
                break
            if (
                end > start
                and part_words >= minimum_words
                and abs(part_words - target_words) <= abs(part_words + next_words - target_words)
            ):
                break
            part_words += next_words
            end += 1

        if end == start:
            end += 1
            part_words = len(turn[start].target_text.split())
        parts.append(turn[start:end])
        remaining_words -= part_words
        start = end
    return parts


def _greedy_part_count(turn: Sequence[TranslationUnit]) -> int:
    count = 0
    part_started = False
    current_words = 0
    for unit in turn:
        unit_words = len(unit.target_text.split())
        if part_started and (current_words + unit_words) / WORDS_PER_SECOND > MAX_CHUNK_S:
            count += 1
            part_started = False
            current_words = 0
        part_started = True
        current_words += unit_words
    if part_started:
        count += 1
    return count


def build_turns(units: Sequence[TranslationUnit]) -> tuple[TurnChunkDraft, ...]:
    """Coalesce adjacent units, then split oversized logical turns."""
    turns: list[list[TranslationUnit]] = []
    for unit in units:
        for fragment in _split_oversized_unit(unit):
            previous = turns[-1][-1] if turns else None
            if (
                previous is not None
                and previous.speaker == fragment.speaker
                and fragment.start - previous.end <= TURN_GAP
            ):
                turns[-1].append(fragment)
            else:
                turns.append([fragment])

    chunks: list[TurnChunkDraft] = []
    for turn_id, turn in enumerate(turns):
        turn_start = turn[0].start
        turn_end = max(unit.end if unit.end > unit.start else unit.start + MIN_UNIT_SPAN_S for unit in turn)
        required_parts = math.ceil((turn_end - turn_start) / MAX_CHUNK_S)
        part_count = min(len(turn), max(_greedy_part_count(turn), required_parts))
        parts = _balance_turn_parts(turn, part_count)

        total_words = sum(len(unit.target_text.split()) for unit in turn)
        cumulative_words = 0
        offset = turn_start
        for part_index, part in enumerate(parts):
            cumulative_words += sum(len(unit.target_text.split()) for unit in part)
            end = (
                turn_end
                if part_index == len(parts) - 1
                else turn_start + (turn_end - turn_start) * cumulative_words / total_words
            )
            chunks.append(
                TurnChunkDraft(
                    speaker=part[0].speaker,
                    start=offset,
                    end=end,
                    text=" ".join(unit.target_text for unit in part),
                    source_text=" ".join(unit.source_text for unit in part).strip(),
                    turn_id=turn_id,
                    part_index=part_index,
                )
            )
            offset = end
    return tuple(chunks)


def snap_turn_starts(
    chunks: tuple[TurnChunkDraft, ...],
    audio: np.ndarray,
    sr: int = 16000,
    search_back: float = 1.5,
    search_fwd: float = 0.3,
    pad: float = 0.05,
) -> tuple[TurnChunkDraft, ...]:
    """Return new chunks with each logical turn's first cue snapped to vocal onset."""
    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio), model, sampling_rate=sr, min_speech_duration_ms=80, min_silence_duration_ms=150
    )
    speech_segments = [(timestamp["start"] / sr, timestamp["end"] / sr) for timestamp in timestamps]
    onsets = [start for start, _end in speech_segments]
    if not onsets:
        raise RuntimeError("tts: no speech detected in source audio")
    adjusted = list(chunks)
    seen: set[int] = set()
    nudged = 0
    dropped: set[int] = set()
    for index, chunk in enumerate(chunks):
        if chunk.turn_id in seen or chunk.part_index != 0:
            continue
        seen.add(chunk.turn_id)
        candidates = [onset for onset in onsets if chunk.start - search_back <= onset <= chunk.start + search_fwd]
        if not candidates:
            # Continuous speech may cross the cue without producing a fresh VAD
            # onset. Keep those turns; only drop cues with no acoustic backing.
            if any(start <= chunk.start <= end for start, end in speech_segments):
                continue
            dropped.add(chunk.turn_id)
            continue
        onset = max(candidates)
        nudge = (onset - pad) - chunk.start
        if abs(nudge) > 0.05:
            adjusted[index] = chunk.validated_copy(
                start=round(onset - pad, 2),
                end=round(chunk.end + nudge, 2),
            )
            nudged += 1
    if dropped:
        adjusted = [chunk for chunk in adjusted if chunk.turn_id not in dropped]
        logger.warning(
            "tts: dropped %d turn(s) with no nearby speech (likely ASR hallucination): %s",
            len(dropped),
            sorted(dropped),
        )
    logger.info("tts: vocal-onset snapping adjusted %d/%d turn starts", nudged, len(seen))
    return tuple(adjusted)


def _gen_file(
    model: Qwen3TTSModel,
    prompts: Mapping[str, Any],
    chunk: TurnChunkDraft,
    dst: str,
    target_language: str,
) -> None:
    wavs, sample_rate = model.generate_voice_clone(
        text=chunk.text,
        language=target_language,
        voice_clone_prompt=prompts[chunk.speaker],
    )
    raw = dst.replace(".mp3", ".wav")
    sf.write(raw, wavs[0], sample_rate)
    temporary = dst + ".tmp.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", raw, "-codec:a", "libmp3lame", "-b:a", "128k", temporary],
        check=True,
    )
    os.replace(temporary, dst)
    os.remove(raw)


def _generate_unless_cached(
    model: Qwen3TTSModel | None,
    prompts: Mapping[str, Any],
    chunk: TurnChunkDraft,
    dst: str,
    target_language: str,
) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > MIN_CACHED_AUDIO_BYTES:
        return
    if model is None:
        raise ValueError("model is required when cached audio is unavailable")
    _gen_file(model, prompts, chunk, dst, target_language)


class _CachedTranslator:
    """Idempotence wrapper: rewrite results persist to a JSON cache (sha1 of
    namespace+budget+text), so a crashed/resumed TTS stage never re-fires an LLM
    rewrite (or regenerates its audio) for work already done."""

    def __init__(self, translator: Translator, cache_path: str, namespace: str = ""):
        self._translator = translator
        self._namespace = namespace
        self._path = Path(cache_path)
        self._cache: dict[str, str] = {}
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "entries" in raw:
                    self._cache = RewriteCache.model_validate(raw).as_dict()
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"tts: invalid current rewrite cache {self._path}: {exc}") from exc

    def _key(self, text: str, budget: int) -> str:
        payload = f"{self._namespace}\0{budget}\0{text}".encode()
        return hashlib.sha1(payload, usedforsecurity=False).hexdigest()

    def _store(self, key: str, value: str) -> None:
        value = value.strip()
        if not value:
            raise RuntimeError("tts: rewrite returned empty text")
        self._cache[key] = value
        model = RewriteCache(
            entries=tuple(
                RewriteCacheEntry(key=entry_key, value=entry_value)
                for entry_key, entry_value in sorted(self._cache.items())
            )
        )
        atomic_write_text(self._path, model.model_dump_json(indent=2))

    def __call__(self, *, previous_context: str, text: str, target_word_count: int) -> str:
        key = self._key(text, target_word_count)
        if key not in self._cache:
            self._store(
                key,
                self._translator(
                    previous_context=previous_context,
                    text_to_translate=text,
                    target_word_count=target_word_count,
                ),
            )
        return self._cache[key]


def make_cached_translator(
    translator: Translator,
    cache_path: str,
    namespace: str = "",
) -> _CachedTranslator:
    return _CachedTranslator(translator, cache_path, namespace)


def make_translator(
    *,
    source_language: str,
    target_language: str,
    base_url: str,
    model: str,
    key: str,
    job_context: str = "",
    proper_nouns: tuple[str, ...] = (),
    glossary: Mapping[str, str] | None = None,
) -> Translator:
    return _make(
        api_key=key,
        model_name=model,
        source_language=source_language,
        target_language=target_language,
        base_url=base_url,
        job_context=job_context,
        proper_nouns=proper_nouns,
        glossary=glossary,
    )


class _RewriteJob(NamedTuple):
    """One pending rewrite in an attempt round, computed from round-start state."""

    chunk_index: int
    chunk: TurnChunk
    budget: int
    event_kind: Literal["rewrite_tighter", "rewrite_fuller"]
    window: float
    duration: float


class _BudgetSample(NamedTuple):
    word_counts: tuple[int, ...]
    duration_s: float


def _allocate_rewrite_budgets(
    word_counts: tuple[int, ...],
    *,
    scale: float,
    floor: int,
    ceiling: int,
) -> tuple[int, ...]:
    """Preserve a logical turn's word target without oversizing any TTS chunk."""
    if not word_counts or any(count <= 0 for count in word_counts):
        raise ValueError("rewrite word counts must be positive")
    if scale <= 0 or floor <= 0 or ceiling < floor:
        raise ValueError("rewrite budget bounds must be positive and ordered")

    target = min(ceiling * len(word_counts), max(floor * len(word_counts), round(sum(word_counts) * scale)))
    budgets = [floor] * len(word_counts)
    while sum(budgets) < target:
        candidates = [index for index, budget in enumerate(budgets) if budget < ceiling]
        if not candidates:
            break
        index = min(candidates, key=lambda candidate: (budgets[candidate] / word_counts[candidate], candidate))
        budgets[index] += 1
    return tuple(budgets)


def _interpolate_rewrite_budgets(
    *,
    lower_word_counts: tuple[int, ...],
    lower_duration_s: float,
    upper_word_counts: tuple[int, ...],
    upper_duration_s: float,
    target_duration_s: float,
    floor: int,
    ceiling: int,
) -> tuple[int, ...]:
    """Interpolate word budgets between measured short and long candidates."""
    if not lower_word_counts or len(lower_word_counts) != len(upper_word_counts):
        raise ValueError("rewrite budget bounds must have matching nonempty word counts")
    if not lower_duration_s < target_duration_s < upper_duration_s:
        raise ValueError("rewrite duration target must be strictly bracketed")
    if floor <= 0 or ceiling < floor:
        raise ValueError("rewrite budget bounds must be positive and ordered")
    if any(lower >= upper for lower, upper in zip(lower_word_counts, upper_word_counts, strict=True)):
        raise ValueError("rewrite word-count bounds must be strictly ordered")

    fraction = (target_duration_s - lower_duration_s) / (upper_duration_s - lower_duration_s)
    budgets: list[int] = []
    for lower, upper in zip(lower_word_counts, upper_word_counts, strict=True):
        budget = round(lower + fraction * (upper - lower))
        if upper - lower > 1:
            budget = min(upper - 1, max(lower + 1, budget))
        budgets.append(min(ceiling, max(floor, budget)))
    return tuple(budgets)


def _evaluate_chunks(chunks: Sequence[TurnChunk], *, total_s: float) -> tuple[FittedTurn, ...]:
    return evaluate_timeline(
        group_by_logical_turns(chunks, stage="tts"),
        total_s,
        lambda path: decode_f32(str(path), 1.0),
        stage="tts",
    )


def _select_rewrite_work(
    fitted: tuple[FittedTurn, ...],
) -> list[FittedTurn]:
    return [
        item
        for item in fitted
        if item.assessment.rewrite_direction is not None and rewrite_can_help(item, word_ceiling=MAX_REWRITE_WORDS)
    ]


def _require_final_fit(fitted: tuple[FittedTurn, ...]) -> None:
    failures = [
        item.assessment
        for item in fitted
        if item.assessment.rewrite_direction is not None and rewrite_can_help(item, word_ceiling=MAX_REWRITE_WORDS)
    ]
    for item in fitted:
        if item.assessment.rewrite_direction is not None and not rewrite_can_help(item, word_ceiling=MAX_REWRITE_WORDS):
            logger.warning(
                "tts: turn %s left %.1fs short of its window; every chunk is already at the %d-word ceiling",
                item.assessment.turn_id,
                item.assessment.window_s - item.assessment.fitted_duration_s,
                MAX_REWRITE_WORDS,
            )
    if not failures:
        return
    details = ", ".join(f"turn {item.turn_id}: {item.fitted_duration_s:.2f}s/{item.window_s:.2f}s" for item in failures)
    raise RuntimeError(f"tts: fit exhausted after {MAX_REWRITE_ATTEMPTS} rewrites: {details}")


def run_tts(cfg: JobConfig) -> str:
    workdir = cfg.resolved_workdir()
    out_json = os.path.join(workdir, "turns.json")
    units_path = os.path.join(workdir, "units.json")
    references_path = os.path.join(workdir, "refs", "references.json")
    translation_api = resolve_translation_api(cfg)
    base = translation_api.base_url
    llm_model = translation_api.model_name
    target_language = lang_name(cfg.target_lang)
    plan = resolve_device_plan(ModelStage.TTS, cfg.tts_device)
    provenance = build_provenance(
        cfg,
        input_files={
            "translation_units": units_path,
            "speaker_references": references_path,
            "source_audio": cfg.resolved_audio(),
        },
        parameters={
            "tts_model": TTS_ID,
            "tts_revision": TTS_REVISION,
            "turn_gap_s": TURN_GAP,
            "max_chunk_s": MAX_CHUNK_S,
            "max_rewrite_words": MAX_REWRITE_WORDS,
            "rewrite_cache_version": REWRITE_CACHE_VERSION,
            "timing_policy_version": TIMING_POLICY_VERSION,
            "vocal_onset_policy_version": VOCAL_ONSET_POLICY_VERSION,
            "rewrite_model": llm_model,
            "rewrite_base": base,
            "target_language": target_language,
        },
        model=ModelIdentity(identifier=TTS_ID, revision=TTS_REVISION),
        execution_plan=plan,
    )
    cached = load_cached_artifact(out_json, TURN_CHUNKS, provenance)
    if cached is not None:
        for chunk in cached:
            if not os.path.exists(chunk.audio_file) or file_digest(chunk.audio_file) != chunk.audio_sha256:
                raise ArtifactError(f"tts: cached audio does not match artifact metadata: {chunk.audio_file}")
        logger.info("tts: cached %s", out_json)
        return out_json

    manifest.configure(workdir)
    units = read_artifact(units_path, TRANSLATION_UNITS).payload
    if cfg.window_s:
        units = tuple(unit for unit in units if unit.start < cfg.window_s)
    if not units:
        raise RuntimeError("tts: no translation units to synthesize")
    total = cfg.window_s or units[-1].end + 0.5
    drafts = build_turns(units)
    logger.info("tts: %d units -> %d turn-chunks", len(units), len(drafts))

    raw_audio = decode_f32(cfg.resolved_audio(), tempo=None, sr=16000, duration_s=total + 2)
    drafts = snap_turn_starts(drafts, raw_audio)

    out_dir = os.path.join(workdir, "turn_tts")
    os.makedirs(out_dir, exist_ok=True)

    def cache_path(index: int, chunk: TurnChunkDraft) -> str:
        digest = stable_digest(
            {
                "model": TTS_ID,
                "revision": TTS_REVISION,
                "language": target_language,
                "speaker": chunk.speaker,
                "text": chunk.text,
            }
        )[:12]
        return os.path.join(out_dir, f"turn_{index:03d}_{digest}.mp3")

    model_path = snapshot_download(repo_id=TTS_ID, revision=TTS_REVISION)
    validate_model_snapshot(model_path)
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(model_path, **model_kwargs_for(plan))
    references = read_artifact(references_path, SPEAKER_REFERENCES).payload
    reference_by_speaker = {reference.speaker: reference for reference in references}
    prompts: dict[str, Any] = {}
    for speaker in sorted({draft.speaker for draft in drafts}):
        reference = reference_by_speaker.get(speaker)
        if reference is None:
            raise RuntimeError(f"tts: missing clone reference metadata for {speaker}")
        if file_digest(reference.audio_file) != reference.audio_sha256:
            raise RuntimeError(f"tts: clone reference digest mismatch for {speaker}: {reference.audio_file}")
        reference_text = Path(reference.transcript_file).read_text(encoding="utf-8")
        prompts[speaker] = model.create_voice_clone_prompt(
            ref_audio=reference.audio_file,
            ref_text=reference_text,
            x_vector_only_mode=True,
        )
        logger.info("tts: clone prompt ready: %s", speaker)

    chunks: list[TurnChunk] = []
    progress = tqdm(enumerate(drafts), total=len(drafts), desc="tts", unit="chunk")
    for index, draft in progress:
        destination = cache_path(index, draft)
        started = time.monotonic()
        _generate_unless_cached(model, prompts, draft, destination, target_language)
        duration = dur_of(destination)
        chunks.append(
            TurnChunk.from_draft(
                draft,
                audio_file=destination,
                audio_duration_s=duration,
                audio_sha256=file_digest(destination),
            )
        )
        progress.set_postfix_str(f"{time.monotonic() - started:.0f}s/chunk")

    key = translation_api.api_key
    cached_translator: _CachedTranslator | None = None
    if key:
        translator = make_translator(
            source_language=lang_name(cfg.source_lang),
            target_language=target_language,
            base_url=base,
            model=llm_model,
            key=key,
            job_context=cfg.context,
            proper_nouns=cfg.proper_nouns,
            glossary=cfg.glossary_map,
        )
        cached_translator = make_cached_translator(
            translator,
            os.path.join(workdir, "rewrite_cache.json"),
            namespace=stable_digest(
                {
                    "version": REWRITE_CACHE_VERSION,
                    "target_language": target_language,
                    "base_url": base,
                    "model": llm_model,
                    "job_context": cfg.context,
                    "proper_nouns": cfg.proper_nouns,
                    "glossary": cfg.glossary_map,
                }
            ),
        )

    lower_samples: dict[int, _BudgetSample] = {}
    upper_samples: dict[int, _BudgetSample] = {}
    for attempt in range(MAX_REWRITE_ATTEMPTS):
        work = _select_rewrite_work(_evaluate_chunks(chunks, total_s=total))
        if not work:
            break
        if cached_translator is None:
            raise RuntimeError("tts: measured audio requires rewriting but no translation API key is configured")

        # Compute every rewrite job from round-start state before regenerating any audio.
        chunk_index_by_part = {(chunk.turn_id, chunk.part_index): index for index, chunk in enumerate(chunks)}
        jobs: list[_RewriteJob] = []
        for fitted_turn in work:
            logical_turn = fitted_turn.logical_turn
            assessment = fitted_turn.assessment
            window = assessment.window_s
            duration = assessment.input_duration_s
            fuller = assessment.rewrite_direction == "fuller"
            current_word_counts = tuple(len(chunk.text.split()) for chunk in logical_turn.chunks)
            sample = _BudgetSample(word_counts=current_word_counts, duration_s=duration)
            samples = lower_samples if fuller else upper_samples
            samples[logical_turn.turn_id] = sample
            ratio = window / duration
            floor = 1 if window < 2.5 else 4
            scale = ratio * (1.07 if fuller else 0.95)
            lower = lower_samples.get(logical_turn.turn_id)
            upper = upper_samples.get(logical_turn.turn_id)
            if (
                lower is not None
                and upper is not None
                and lower.duration_s < window < upper.duration_s
                and all(
                    lower_count < upper_count
                    for lower_count, upper_count in zip(lower.word_counts, upper.word_counts, strict=True)
                )
            ):
                budgets = _interpolate_rewrite_budgets(
                    lower_word_counts=lower.word_counts,
                    lower_duration_s=lower.duration_s,
                    upper_word_counts=upper.word_counts,
                    upper_duration_s=upper.duration_s,
                    target_duration_s=window,
                    floor=floor,
                    ceiling=MAX_REWRITE_WORDS,
                )
            elif fuller:
                budgets = _allocate_rewrite_budgets(
                    current_word_counts,
                    scale=scale,
                    floor=floor,
                    ceiling=MAX_REWRITE_WORDS,
                )
            else:
                budgets = tuple(max(floor, round(word_count * scale)) for word_count in current_word_counts)
            for old_chunk, budget in zip(logical_turn.chunks, budgets, strict=True):
                try:
                    chunk_index = chunk_index_by_part[old_chunk.turn_id, old_chunk.part_index]
                except KeyError:
                    raise RuntimeError(
                        f"tts: no chunk for turn {old_chunk.turn_id} part {old_chunk.part_index}"
                    ) from None
                jobs.append(
                    _RewriteJob(
                        chunk_index=chunk_index,
                        chunk=old_chunk,
                        budget=budget,
                        event_kind="rewrite_fuller" if fuller else "rewrite_tighter",
                        window=window,
                        duration=duration,
                    )
                )
        # Re-translate each job from its SOURCE text to a tighter/fuller word
        # budget, then regenerate its audio (one GPU, serial).
        for job in jobs:
            source = job.chunk.source_text
            if not source:
                raise RuntimeError(f"tts: turn {job.chunk.turn_id} has no source text to re-translate")
            rewritten_text = cached_translator(previous_context="", text=source, target_word_count=job.budget)
            draft = TurnChunkDraft(
                **job.chunk.model_dump(exclude={"audio_file", "audio_duration_s", "audio_sha256", "text"}),
                text=rewritten_text,
            )
            destination = cache_path(job.chunk_index, draft)
            _generate_unless_cached(model, prompts, draft, destination, target_language)
            replacement = TurnChunk.from_draft(
                draft,
                audio_file=destination,
                audio_duration_s=dur_of(destination),
                audio_sha256=file_digest(destination),
            )
            chunks[job.chunk_index] = replacement
            manifest.log_event(
                RewriteEvent(
                    kind=job.event_kind,
                    turn=f"t{job.chunk.turn_id}p{job.chunk.part_index}",
                    speaker=job.chunk.speaker,
                    before=job.chunk.text,
                    after=replacement.text,
                    budget_words=job.budget,
                    words_before=len(job.chunk.text.split()),
                    words_after=len(replacement.text.split()),
                    duration_before_s=job.chunk.audio_duration_s,
                    duration_after_s=replacement.audio_duration_s,
                    window_s=job.window,
                    ratio=job.duration / job.window,
                    attempt=attempt,
                    model=llm_model,
                )
            )

    _require_final_fit(_evaluate_chunks(chunks, total_s=total))
    write_artifact_atomic(out_json, TURN_CHUNKS, provenance, tuple(chunks))
    logger.info("tts: wrote %s", out_json)
    return out_json
