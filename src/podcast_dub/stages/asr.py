#!/usr/bin/env python3
"""ASR stage: caption-free phrase timing via Qwen3-ASR-1.7B + ForcedAligner.

Input: job audio (16kHz wav from probe). Chunks it, transcribes with
Qwen3-ASR, force-aligns words to the audio, groups words into speech-natural
phrases. Output: <workdir>/phrases.json [{start, end, text}].

Environment note: Qwen3-ASR needs transformers>=5.x while Qwen3-TTS needs 4.x.
If this module runs under transformers<5, it re-executes itself with the ASR
venv interpreter (DUB_ASR_PYTHON env var, default <project>/.venv-asr/bin/python).
"""

import logging
import os
import re
import subprocess
from collections.abc import Sequence

from tqdm import tqdm

from podcast_dub.artifacts import build_provenance, load_cached_artifact, write_artifact_atomic
from podcast_dub.audio_utils import dur_of
from podcast_dub.config import JobConfig
from podcast_dub.device_utils import helper_process_env, helper_python, resolve_device_plan
from podcast_dub.model_catalog import ALIGNER_ID, ALIGNER_REVISION, ASR_ID, ASR_REVISION
from podcast_dub.models import (
    AlignedWord,
    AsrBackendRequest,
    AsrBackendResult,
    DevicePlan,
    ModelIdentity,
    ModelStage,
    Phrase,
    PhraseWord,
)
from podcast_dub.pipeline_artifacts import PHRASES

logger = logging.getLogger(__name__)

CHUNK_S = 100.0
MAX_COLLAPSED_WORD_RUN = 8
SENT_END_ZH = re.compile(r"[，。！？；、…,.!?;]$")
VDIR = os.path.dirname(os.path.abspath(__file__))


def _tf_major():
    try:
        import transformers

        return int(transformers.__version__.split(".")[0])
    except Exception:
        logger.exception("asr: transformers major version detection failed")
        return 0


def run_asr(cfg: JobConfig) -> str:
    """Library entry: run the stage (re-exec under transformers>=5 if needed)."""
    out = os.path.join(cfg.resolved_workdir(), "phrases.json")
    plan = resolve_device_plan(ModelStage.ASR, cfg.asr_device)
    request = AsrBackendRequest(
        audio_file=cfg.resolved_audio(),
        source_language=cfg.source_lang,
        window_s=cfg.window_s or None,
        plan=plan,
    )
    provenance = build_provenance(
        cfg,
        input_files={"audio": cfg.resolved_audio()},
        parameters={
            "chunk_s": CHUNK_S,
            "max_collapsed_word_run": MAX_COLLAPSED_WORD_RUN,
            "source_language": cfg.source_lang,
            "window_s": cfg.window_s,
            "aligner": {"identifier": ALIGNER_ID, "revision": ALIGNER_REVISION},
        },
        model=ModelIdentity(identifier=ASR_ID, revision=ASR_REVISION),
        execution_plan=plan,
    )
    if load_cached_artifact(out, PHRASES, provenance) is not None:
        print(f"asr: cached {out}")
        return out
    if _tf_major() >= 5:
        result = _run_asr_inline(request)
    else:
        asr_py = helper_python("DUB_ASR_PYTHON", ".venv-asr")
        if not os.path.exists(asr_py):
            raise RuntimeError(
                f"transformers {_tf_major()} < 5 and no ASR venv at {asr_py}. "
                "Create it: uv venv .venv-asr && uv pip install --python .venv-asr/bin/python "
                "'transformers==5.14.1' torch torchaudio accelerate librosa soundfile pydantic"
            )
        result_path = os.path.join(cfg.resolved_workdir(), "_asr_backend_result.json")
        cmd = [
            asr_py,
            os.path.abspath(__file__),
            "--audio",
            request.audio_file,
            "--out",
            result_path,
            "--source-language",
            request.source_language,
            "--device",
            plan.device,
            "--dtype",
            plan.dtype,
            "--attention",
            plan.attention,
        ]
        if request.window_s:
            cmd += ["--window", str(request.window_s)]
        env = helper_process_env()
        subprocess.run(cmd, check=True, env=env)
        result = AsrBackendResult.model_validate_json(open(result_path, "rb").read())
        os.remove(result_path)
    write_artifact_atomic(out, PHRASES, provenance, result.phrases)
    return out


def _group_phrases(timestamps: tuple[AlignedWord, ...] | list[AlignedWord], offset: float) -> tuple[Phrase, ...]:
    words = tuple(PhraseWord(text=word.text, start=word.start + offset, end=word.end + offset) for word in timestamps)
    if not words:
        raise RuntimeError("asr: forced aligner returned no words")
    phrases: list[list[PhraseWord]] = []
    cur: list[PhraseWord] = [words[0]]
    for a, b in zip(words, words[1:], strict=False):
        if b.start - a.end > 0.3 or SENT_END_ZH.search(a.text) or b.start - cur[0].start > 15.0:
            phrases.append(cur)
            cur = [b]
        else:
            cur.append(b)
    phrases.append(cur)
    return tuple(
        Phrase(
            start=round(phrase[0].start, 3),
            end=round(phrase[-1].end, 3),
            text="".join(word.text for word in phrase),
            words=tuple(word.validated_copy(start=round(word.start, 3), end=round(word.end, 3)) for word in phrase),
        )
        for phrase in phrases
    )


def _chunk_ranges(limit_s: float) -> tuple[tuple[float, float], ...]:
    """Return bounded ``(offset, duration)`` ranges that never cross the requested limit."""
    ranges: list[tuple[float, float]] = []
    offset = 0.0
    while offset < limit_s:
        duration = min(CHUNK_S, limit_s - offset)
        ranges.append((offset, duration))
        offset += duration
    return tuple(ranges)


def _validate_alignment_quality(words: Sequence[AlignedWord]) -> None:
    """Reject timestamp saturation while allowing small quantization ties."""
    collapsed_at: float | None = None
    collapsed_run = 0
    for word in words:
        if word.end == word.start:
            if collapsed_at == word.start:
                collapsed_run += 1
            else:
                collapsed_at = word.start
                collapsed_run = 1
        else:
            collapsed_at = None
            collapsed_run = 0
        if collapsed_run > MAX_COLLAPSED_WORD_RUN:
            raise RuntimeError(f"asr: forced aligner collapsed {collapsed_run} consecutive words at {word.start:.3f}s")


def _run_asr_inline(request: AsrBackendRequest) -> AsrBackendResult:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForMultimodalLM,  # ty: ignore[unresolved-import]  # transformers>=5 only (ASR venv)
        AutoModelForTokenClassification,
        AutoProcessor,
    )

    dur = dur_of(request.audio_file)
    limit = min(dur, request.window_s) if request.window_s else dur
    chunk_ranges = _chunk_ranges(limit)

    dtype = torch.bfloat16 if request.plan.dtype == "bfloat16" else torch.float32
    print("asr: loading models...", flush=True)
    from podcast_dub.model_catalog import validate_model_snapshot

    asr_path = snapshot_download(repo_id=ASR_ID, revision=ASR_REVISION)
    aligner_path = snapshot_download(repo_id=ALIGNER_ID, revision=ALIGNER_REVISION)
    validate_model_snapshot(asr_path)
    validate_model_snapshot(aligner_path)
    asr_p = AutoProcessor.from_pretrained(asr_path)
    asr_m = AutoModelForMultimodalLM.from_pretrained(
        asr_path,
        device_map=request.plan.device,
        dtype=dtype,
        attn_implementation=request.plan.attention,
    )
    al_p = AutoProcessor.from_pretrained(aligner_path)
    al_m = AutoModelForTokenClassification.from_pretrained(
        aligner_path,
        dtype=dtype,
        device_map=request.plan.device,
    )

    all_words: list[AlignedWord] = []
    progress = tqdm(enumerate(chunk_ranges), total=len(chunk_ranges), desc="asr", unit="chunk")
    for i, (off, chunk_duration_s) in progress:
        seg = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(off),
                "-t",
                str(chunk_duration_s),
                "-i",
                request.audio_file,
                "-f",
                "wav",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        ).stdout
        tmp = os.path.join(os.path.dirname(request.audio_file), f"_asr_chunk_{i}.wav")
        open(tmp, "wb").write(seg)
        inputs = asr_p.apply_transcription_request(audio=tmp)
        inputs = inputs.to(asr_m.device, asr_m.dtype)
        out_ids = asr_m.generate(**inputs, max_new_tokens=4096)
        gen = out_ids[:, inputs["input_ids"].shape[1] :]
        parsed = asr_p.decode(gen, return_format="parsed")[0]
        transcript, language = parsed["transcription"], parsed["language"] or "auto"
        al_inputs, word_lists = al_p.prepare_forced_aligner_inputs(audio=tmp, transcript=transcript, language=language)
        al_inputs = al_inputs.to(al_m.device, al_m.dtype)
        with torch.inference_mode():
            outputs = al_m(**al_inputs)
        ts = al_p.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=al_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=al_m.config.timestamp_token_id,
        )[0]
        chunk_words = tuple(
            AlignedWord(text=w["text"], start=w["start_time"] + off, end=w["end_time"] + off) for w in ts
        )
        _validate_alignment_quality(chunk_words)
        all_words.extend(chunk_words)
        os.remove(tmp)
        progress.set_postfix(words=len(all_words))

    phrases = _group_phrases(all_words, 0)
    print(f"asr: produced {len(phrases)} phrases")
    return AsrBackendResult(
        words=tuple(all_words),
        phrases=phrases,
        model=ModelIdentity(identifier=ASR_ID, revision=ASR_REVISION),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--source-language", required=True)
    ap.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--attention", required=True)
    a = ap.parse_args()
    result = _run_asr_inline(
        AsrBackendRequest(
            audio_file=a.audio,
            source_language=a.source_language,
            window_s=a.window,
            plan=DevicePlan(stage=ModelStage.ASR, device=a.device, dtype=a.dtype, attention=a.attention),
        )
    )
    with open(a.out, "w", encoding="utf-8") as result_file:
        result_file.write(result.model_dump_json(indent=2))
