#!/usr/bin/env python3
"""Diarize stage: who speaks when -> speaker label per ASR phrase.

Primary: NVIDIA Sortformer streaming diarization
(nvidia/diar_streaming_sortformer_4spk-v2.1, ungated) via NeMo. NeMo runs in
an isolated helper environment for reproducibility and dependency stability,
so when it is not importable here we re-execute this file with the NeMo venv
interpreter (DUB_NEMO_PYTHON env var, default
<project>/.venv-nemo/bin/python) — the same pattern as the ASR stage.

Input:  <workdir>/phrases.json + job audio (16kHz mono wav from probe)
Output: <workdir>/diar_segments.json  raw diarization segments
        <workdir>/phrases_spk.json    phrases with "speaker" (cfg name or spk_N)
"""

import logging
import os
import subprocess

from podcast_dub.artifacts import build_provenance, load_cached_artifact, read_artifact, write_artifact_atomic
from podcast_dub.config import JobConfig
from podcast_dub.device_utils import helper_process_env, helper_python, resolve_device_plan
from podcast_dub.model_catalog import SORTFORMER_FILE, SORTFORMER_ID, SORTFORMER_REVISION
from podcast_dub.models import (
    DevicePlan,
    DiarizationBackendRequest,
    DiarizationBackendResult,
    DiarizationSegment,
    ModelIdentity,
    ModelStage,
    Phrase,
    SpeakerPhrase,
)
from podcast_dub.pipeline_artifacts import DIARIZATION_SEGMENTS, PHRASES, SPEAKER_PHRASES

VDIR = os.path.dirname(os.path.abspath(__file__))

MERGE_GAP_S = 0.3  # merge adjacent same-speaker segments across shorter gaps
MIN_SPEECH_FRAC = 0.01  # drop speakers below this share of total speech (noise)
MIN_SPLIT_S = 0.4  # split a phrase at a speaker change only if BOTH sides
# hold >= this long inside the phrase (else blip)


def run_diarize(cfg: JobConfig) -> str:
    workdir = cfg.resolved_workdir()
    out = os.path.join(workdir, "phrases_spk.json")
    phrases_path = os.path.join(workdir, "phrases.json")
    segs_path = os.path.join(workdir, "diar_segments.json")
    plan = resolve_device_plan(ModelStage.DIARIZE, cfg.diarize_device)

    def _phrase_provenance(input_files: dict[str, str]):
        return build_provenance(
            cfg,
            input_files=input_files,
            parameters={
                "merge_gap_s": MERGE_GAP_S,
                "min_speech_frac": MIN_SPEECH_FRAC,
                "min_split_s": MIN_SPLIT_S,
                "speaker_names": cfg.speaker_names,
            },
            model=ModelIdentity(identifier=SORTFORMER_ID, revision=SORTFORMER_REVISION),
            execution_plan=plan,
        )

    phrase_provenance = _phrase_provenance(
        {"phrases": phrases_path, "segments": segs_path} if os.path.exists(segs_path) else {"phrases": phrases_path}
    )
    if load_cached_artifact(out, SPEAKER_PHRASES, phrase_provenance) is not None:
        print(f"diarize: cached {out}")
        return out
    phrases = read_artifact(phrases_path, PHRASES).payload
    segment_provenance = build_provenance(
        cfg,
        input_files={"audio": cfg.resolved_audio()},
        parameters={"model": SORTFORMER_ID, "revision": SORTFORMER_REVISION, "merge_gap_s": MERGE_GAP_S},
        model=ModelIdentity(identifier=SORTFORMER_ID, revision=SORTFORMER_REVISION),
        execution_plan=plan,
    )
    raw = load_cached_artifact(segs_path, DIARIZATION_SEGMENTS, segment_provenance)
    if raw is None:
        raw = _produce_segments(DiarizationBackendRequest(audio_file=cfg.resolved_audio(), plan=plan), workdir)
        write_artifact_atomic(segs_path, DIARIZATION_SEGMENTS, segment_provenance, raw)
    phrase_provenance = _phrase_provenance({"phrases": phrases_path, "segments": segs_path})
    phrases, mapping, totals = label_phrases(phrases, raw, cfg.speaker_names)
    pretty = {mapping[s]: round(t) for s, t in totals.items() if s in mapping}
    print(f"diarize: speakers {pretty}s -> {mapping}")
    write_artifact_atomic(out, SPEAKER_PHRASES, phrase_provenance, phrases)
    print(f"diarize: wrote {out}")
    return out


def _have_nemo() -> bool:
    try:
        import nemo.collections.asr.models  # noqa: F401  # ty: ignore[unresolved-import]

        return True
    except Exception:
        logging.warning("diarize: WARNING no NeMo (set DUB_NEMO_PYTHON or create .venv-nemo)")
        return False


def _produce_segments(request: DiarizationBackendRequest, workdir: str) -> tuple[DiarizationSegment, ...]:
    if _have_nemo():
        return _run_sortformer(request)
    nemo_py = helper_python("DUB_NEMO_PYTHON", ".venv-nemo")
    if os.path.exists(nemo_py):
        result_path = os.path.join(workdir, "_diarize_backend_result.json")
        # the helper venv has no podcast_dub installed; make the package importable
        env = helper_process_env()
        subprocess.run(
            [
                nemo_py,
                os.path.abspath(__file__),
                "--audio",
                request.audio_file,
                "--out",
                result_path,
                "--device",
                request.plan.device,
                "--dtype",
                request.plan.dtype,
                "--attention",
                request.plan.attention,
            ],
            check=True,
            env=env,
        )
        result = DiarizationBackendResult.model_validate_json(open(result_path, "rb").read())
        os.remove(result_path)
        return result.segments
    raise RuntimeError(f"diarize: required NeMo backend is unavailable; set DUB_NEMO_PYTHON or create {nemo_py}")


def _run_sortformer(request: DiarizationBackendRequest) -> tuple[DiarizationSegment, ...]:
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel  # ty: ignore[unresolved-import]

    checkpoint = hf_hub_download(
        repo_id=SORTFORMER_ID,
        filename=SORTFORMER_FILE,
        revision=SORTFORMER_REVISION,
    )
    m = SortformerEncLabelModel.restore_from(
        restore_path=checkpoint,
        map_location=torch.device(request.plan.device),
        strict=False,
    )
    m.eval()
    # Longer streaming chunks reduce segmentation churn on long-form audio.
    m.sortformer_modules.chunk_len = 340
    m = m.to(request.plan.device)
    segs = m.diarize(audio=[request.audio_file], batch_size=1)
    out: list[DiarizationSegment] = []
    for s in segs[0]:  # lines of "start end speaker_N"
        a, b, spk = s.split()
        out.append(DiarizationSegment(start=float(a), end=float(b), speaker=spk))
    print(f"diarize: sortformer -> {len(out)} raw segments", flush=True)
    return tuple(out)


def merge_segments(
    raw_segs: tuple[DiarizationSegment, ...] | list[DiarizationSegment],
) -> tuple[DiarizationSegment, ...]:
    segs = sorted(raw_segs, key=lambda segment: (segment.start, segment.end))
    merged: list[DiarizationSegment] = []
    for s in segs:
        if merged and merged[-1].speaker == s.speaker and s.start - merged[-1].end < MERGE_GAP_S:
            merged[-1] = merged[-1].validated_copy(end=max(merged[-1].end, s.end))
        else:
            merged.append(s)
    return tuple(merged)


def speaker_mapping(
    raw_segs: tuple[DiarizationSegment, ...] | list[DiarizationSegment],
    names: tuple[str, ...] | list[str],
) -> tuple[tuple[DiarizationSegment, ...], dict[str, str], dict[str, float]]:
    """Merge segments, drop noise speakers, order survivors by speaking time.

    Returns (merged, mapping, totals): merged segment list, mapping raw
    speaker -> display name (only kept speakers), totals raw speaker -> kept
    speech seconds.
    """
    merged = merge_segments(raw_segs)
    totals: dict[str, float] = {}
    for s in merged:
        totals[s.speaker] = totals.get(s.speaker, 0.0) + s.end - s.start
    total_all = sum(totals.values()) or 1.0
    kept = {spk for spk, t in totals.items() if t / total_all >= MIN_SPEECH_FRAC}
    order = sorted(kept, key=lambda spk: -totals[spk])
    mapping = {spk: (names[r] if r < len(names) else f"spk_{r}") for r, spk in enumerate(order)}
    return merged, mapping, totals


def label_phrases(
    phrases: tuple[Phrase, ...] | list[Phrase],
    raw_segs: tuple[DiarizationSegment, ...] | list[DiarizationSegment],
    names: tuple[str, ...] | list[str],
) -> tuple[tuple[SpeakerPhrase, ...], dict[str, str], dict[str, float]]:
    """Split phrases at real speaker handoffs, label by max overlap.

    Cross-speaker phrases make TTS use the wrong voice, so phrases with word
    timings are SPLIT at each sustained speaker change. Changes where either
    side holds < MIN_SPLIT_S inside the phrase are treated as diarization blips
    and ignored.

    Returns (phrases, mapping, totals): mapping raw speaker -> display name,
    totals raw speaker -> kept speech seconds.
    """
    merged, mapping, totals = speaker_mapping(raw_segs, names)
    labeled = tuple((s.start, s.end, mapping[s.speaker]) for s in merged if s.speaker in mapping)
    dominant = next(iter(mapping.values()), "spk_0")
    out: list[SpeakerPhrase] = []
    for p in phrases:
        out.extend(_split_and_label(p, labeled, dominant))
    return merge_orphans(out), mapping, totals


def merge_orphans(out: tuple[SpeakerPhrase, ...] | list[SpeakerPhrase]) -> tuple[SpeakerPhrase, ...]:
    """Merge sub-2-char orphan phrases into the adjacent same-speaker phrase.

    Aligner artifacts sometimes split a sentence-final particle (的, 嗯) into a
    zero-duration orphan; it has no standalone meaning and an LLM asked to
    translate it alone returns empty. Prefer merging into the previous
    same-speaker phrase (usually its true parent sentence), else the next;
    isolated fragments between speakers are real backchannels and stay.
    """
    merged: list[SpeakerPhrase] = []
    i = 0
    while i < len(out):
        p = out[i]
        if len(p.text.strip()) < 2:
            if merged and merged[-1].speaker == p.speaker:
                prev = merged[-1]
                merged[-1] = prev.validated_copy(
                    end=max(prev.end, p.end),
                    text=prev.text + p.text,
                    words=prev.words + p.words,
                )
                i += 1
                continue
            if i + 1 < len(out) and out[i + 1].speaker == p.speaker:
                nxt = out[i + 1]
                out = list(out)
                out[i + 1] = nxt.validated_copy(
                    start=min(nxt.start, p.start),
                    text=p.text + nxt.text,
                    words=p.words + nxt.words,
                )
                i += 1
                continue
        merged.append(p)
        i += 1
    return tuple(merged)


def _overlap_label(start: float, end: float, labeled: tuple[tuple[float, float, str], ...], dominant: str) -> str:
    best, best_ov = None, 0.0
    for a, b, spk in labeled:
        ov = max(0.0, min(b, end) - max(a, start))
        if ov > best_ov:
            best, best_ov = spk, ov
    return best or dominant


def _split_and_label(
    p: Phrase,
    labeled: tuple[tuple[float, float, str], ...],
    dominant: str,
) -> tuple[SpeakerPhrase, ...]:
    words = p.words
    cuts: list[float] = []
    for (a0, b0, spk0), (a1, b1, spk1) in zip(labeled, labeled[1:], strict=False):
        if spk0 == spk1:
            continue
        cut = (b0 + a1) / 2
        if not (p.start + 0.05 < cut < p.end - 0.05):
            continue
        # real handoff = BOTH sides hold >= MIN_SPLIT_S inside the phrase;
        # a brief flip (backchannel/noise) fails the test on one side
        ov_out = min(b0, p.end) - max(a0, p.start)
        ov_in = min(b1, p.end) - max(a1, p.start)
        if ov_out >= MIN_SPLIT_S and ov_in >= MIN_SPLIT_S:
            cuts.append(cut)
    if not cuts or not words:
        return (
            SpeakerPhrase(
                **p.model_dump(),
                speaker=_overlap_label(p.start, p.end, labeled, dominant),
            ),
        )

    cuts.sort()
    # bucket index = number of cuts below the word's midpoint
    buckets: dict[int, list] = {}
    for w in words:
        mid = (w.start + w.end) / 2
        k = sum(1 for c in cuts if mid > c)
        buckets.setdefault(k, []).append(w)
    pieces: list[SpeakerPhrase] = []
    for k in sorted(buckets):
        ws = buckets[k]
        start = round(ws[0].start, 3)
        end = round(ws[-1].end, 3)
        piece = SpeakerPhrase(
            start=start,
            end=end,
            text="".join(w.text for w in ws),
            words=tuple(ws),
            speaker=_overlap_label(start, end, labeled, dominant),
        )
        pieces.append(piece)
    return tuple(pieces)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--attention", required=True)
    a = ap.parse_args()
    request = DiarizationBackendRequest(
        audio_file=a.audio,
        plan=DevicePlan(stage=ModelStage.DIARIZE, device=a.device, dtype=a.dtype, attention=a.attention),
    )
    result = DiarizationBackendResult(
        segments=_run_sortformer(request),
        model=ModelIdentity(identifier=SORTFORMER_ID, revision=SORTFORMER_REVISION),
    )
    with open(a.out, "w", encoding="utf-8") as result_file:
        result_file.write(result.model_dump_json(indent=2))
