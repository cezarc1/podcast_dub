from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import dspy
from dspy import Prediction

DEFAULT_BASE = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k3"


class TranslateSpokenASR(dspy.Signature):
    """
    Translate machine-transcribed spoken conversation into natural spoken
    target-language text suitable for TTS.

    Silently repair obvious ASR recognition noise, duplicated fragments, and
    broken punctuation without inventing meaning. Use previous_turns only to
    understand the conversation and translate. Preserve meaning,
    intent, tone, and conversational register and technical jargon. Return complete, naturally
    pronounceable sentences.
    """

    source_language: str = dspy.InputField(desc="language spoken in the ASR transcript")
    target_language: str = dspy.InputField(desc="language the translated text will be spoken in")
    job_context: str = dspy.InputField(desc="job-level subject context; use only to disambiguate the current text")
    proper_nouns: str = dspy.InputField(desc="comma-separated spellings to preserve when relevant")
    glossary: str = dspy.InputField(desc="source-to-target terminology preferences, one mapping per line")
    previous_context: str = dspy.InputField(
        desc="preceding source-language conversation turns, oldest to newest; never translate these"
    )
    text_to_translate: str = dspy.InputField(desc="current ASR utterances to translate, in conversation order")
    target_word_count: int = dspy.InputField(desc="exact ideal number of words for the translated text")

    target_translated_text: str = dspy.OutputField(
        desc="natural spoken target-language translations with exactly the target word count"
    )


def _make_lm(api_key: str, base: str, model: str, *, num_retries: int = 4):
    return dspy.LM(
        f"openai/{model}",
        api_key=api_key,
        api_base=base,
        allowed_openai_params=["reasoning_effort"],
        reasoning_effort="low",
        num_retries=num_retries,
    )


def _word_fit_reward(args: dict, pred: Prediction) -> float:
    """Reward exact word count, then decay with normalized squared error."""
    target_word_count = int(args["target_word_count"])
    actual_word_count = len(pred.target_translated_text.split())
    if not actual_word_count:
        return 0.0
    error = abs(actual_word_count - target_word_count)
    tolerance = max(1.0, 0.05 * target_word_count)
    return 1.0 / (1.0 + (error / tolerance) ** 2)


class Translator(Protocol):
    def __call__(self, *, previous_context: str, text_to_translate: str, target_word_count: int) -> str: ...


def make_translator(
    api_key: str,
    source_language: str,
    target_language: str,
    base_url: str,
    model_name: str,
    *,
    n_attempts: int = 3,
    job_context: str = "",
    proper_nouns: tuple[str, ...] = (),
    glossary: Mapping[str, str] | None = None,
) -> Translator:
    lm = _make_lm(api_key, base_url, model_name, num_retries=n_attempts)
    refiner = dspy.Refine(
        module=dspy.Predict(TranslateSpokenASR),
        N=n_attempts,
        reward_fn=_word_fit_reward,
        threshold=1.0,
    )
    proper_nouns_text = ", ".join(proper_nouns)
    glossary_text = "\n".join(f"{source} -> {target}" for source, target in (glossary or {}).items())

    def _translate(previous_context: str, text_to_translate: str, target_word_count: int) -> str:
        with dspy.context(lm=lm):
            pred = refiner(
                source_language=source_language,
                target_language=target_language,
                job_context=job_context,
                proper_nouns=proper_nouns_text,
                glossary=glossary_text,
                previous_context=previous_context,
                text_to_translate=text_to_translate,
                target_word_count=target_word_count,
            )
        return pred.target_translated_text

    return _translate
