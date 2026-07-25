#!/usr/bin/env python3
"""Translate validated speaker phrases into validated target-language units."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path

from tqdm import tqdm

import podcast_dub.manifest as manifest
from podcast_dub.artifacts import (
    atomic_write_text,
    build_provenance,
    load_cached_artifact,
    read_artifact,
    stable_digest,
    write_artifact_atomic,
)
from podcast_dub.config import JobConfig, lang_name, resolve_translation_api
from podcast_dub.pipeline_artifacts import SPEAKER_PHRASES, TRANSLATION_UNITS
from podcast_dub.translate import make_translator
from podcast_dub.types import (
    ModelIdentity,
    TranslateEvent,
    TranslationBatch,
    TranslationManifestLine,
    TranslationUnit,
)

BATCH = 20
WORKERS = 3
PREVIOUS_TURNS = 3
WORDS_PER_SECOND = 2.5  # target-language speaking rate for the initial word budget
TRANSLATION_CACHE_VERSION = 4

logger = logging.getLogger(__name__)


def _read_batch(path: Path) -> TranslationBatch:
    try:
        return TranslationBatch.model_validate_json(path.read_bytes())
    except Exception as exc:
        raise RuntimeError(f"translate: invalid batch cache {path}: {exc}") from exc


def _write_batch(path: Path, batch: TranslationBatch) -> None:
    atomic_write_text(path, batch.model_dump_json(indent=2))


def run_translate(cfg: JobConfig) -> str:
    workdir = cfg.resolved_workdir()
    out = os.path.join(workdir, "units.json")
    phrases_path = os.path.join(workdir, "phrases_spk.json")
    translation_api = resolve_translation_api(cfg)
    provenance = build_provenance(
        cfg,
        input_files={"speaker_phrases": phrases_path},
        parameters={
            "cache_version": TRANSLATION_CACHE_VERSION,
            "source_language": lang_name(cfg.source_lang),
            "target_language": lang_name(cfg.target_lang),
            "base": translation_api.base_url,
            "window_s": cfg.window_s,
            "batch_size": BATCH,
            "previous_turns": PREVIOUS_TURNS,
            "words_per_second": WORDS_PER_SECOND,
        },
        model=ModelIdentity(identifier=translation_api.model_name),
    )
    if load_cached_artifact(out, TRANSLATION_UNITS, provenance) is not None:
        logger.info("translate: cached %s", out)
        return out

    manifest.configure(workdir)
    if not translation_api.api_key:
        raise RuntimeError("translate: no API key (set DUB_TRANSLATE_API_KEY)")
    translator = make_translator(
        translation_api.api_key,
        source_language=lang_name(cfg.source_lang),
        target_language=lang_name(cfg.target_lang),
        base_url=translation_api.base_url,
        model_name=translation_api.model_name,
        job_context=cfg.context,
        proper_nouns=cfg.proper_nouns,
        glossary=cfg.glossary_map,
    )
    phrases = read_artifact(phrases_path, SPEAKER_PHRASES).payload
    if cfg.window_s:
        phrases = tuple(phrase for phrase in phrases if phrase.start < cfg.window_s)

    cache_key = stable_digest(provenance)[:16]
    tr_dir = Path(workdir, "translated", cache_key)
    tr_dir.mkdir(parents=True, exist_ok=True)
    batches = list(batched(range(len(phrases)), BATCH))
    cached_batches = {path.name: _read_batch(path) for path in tr_dir.glob("batch_*.json")}
    todo = [
        (batch_index, indexes, tr_dir / f"batch_{batch_index:03d}.json")
        for batch_index, indexes in enumerate(batches)
        if f"batch_{batch_index:03d}.json" not in cached_batches
    ]
    logger.info(
        "translate: %d phrases; %d batches to do (%d cached)",
        len(phrases),
        len(todo),
        len(batches) - len(todo),
    )

    def _budget(index: int) -> int:
        return max(1, round((phrases[index].end - phrases[index].start) * WORDS_PER_SECOND))

    def _previous_context(index: int) -> str:
        return " ".join(phrases[j].text for j in range(max(0, index - PREVIOUS_TURNS), index))

    def work(task: tuple[int, tuple[int, ...], Path]) -> int:
        batch_index, indexes, batch_path = task
        translations = {}
        for i in indexes:
            translated = translator(
                previous_context=_previous_context(i),
                text_to_translate=phrases[i].text,
                target_word_count=_budget(i),
            ).strip()
            if not translated:
                raise RuntimeError(f"translate: phrase {i} returned empty translation")
            translations[i] = translated
        batch = TranslationBatch(batch_index=batch_index, translations=translations)
        _write_batch(batch_path, batch)
        manifest.log_event(
            TranslateEvent(
                batch_index=batch_index,
                ids=indexes,
                model=translation_api.model_name,
                lines=tuple(
                    TranslationManifestLine(
                        id=i,
                        source_text=phrases[i].text,
                        target_text=translations[i],
                    )
                    for i in indexes
                ),
            )
        )
        return batch_index

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for batch_index in tqdm(executor.map(work, todo), total=len(todo), desc="translate", unit="batch"):
            logger.info("translate: batch %d completed", batch_index)

    merged = {}
    for path in sorted(tr_dir.glob("batch_*.json")):
        batch = _read_batch(path)
        if overlap := set(merged).intersection(batch.translations):
            raise RuntimeError(f"translate: duplicate cached translation ids {sorted(overlap)} in {path}")
        merged.update(batch.translations)
    if missing := [i for i in range(len(phrases)) if i not in merged]:
        raise RuntimeError(f"translate: missing translations: {missing[:10]}")

    units = tuple(
        TranslationUnit.from_phrase(phrase, target_text=merged[index]) for index, phrase in enumerate(phrases)
    )
    write_artifact_atomic(out, TRANSLATION_UNITS, provenance, units)
    logger.info("translate: wrote %s (%d units)", out, len(units))
    return out
