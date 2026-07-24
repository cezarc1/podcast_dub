import importlib
import importlib.util

import dspy
import pytest
from dspy import Prediction

from podcast_dub.translate import (
    TranslateSpokenASR,
    _word_fit_reward,
    make_translator,
)


def test_dspy_translation_module_has_domain_name():
    assert importlib.util.find_spec("podcast_dub.translate") is not None


def test_translation_module_exposes_unified_translator_api():
    translation = importlib.import_module("podcast_dub.translate")

    assert hasattr(translation, "TranslateSpokenASR")
    assert hasattr(translation, "make_translator")


def test_translation_signature_fits_a_single_text_to_a_word_count():
    assert set(TranslateSpokenASR.input_fields) == {
        "source_language",
        "target_language",
        "job_context",
        "proper_nouns",
        "glossary",
        "previous_context",
        "text_to_translate",
        "target_word_count",
    }
    assert set(TranslateSpokenASR.output_fields) == {"target_translated_text"}


def reward_for(target: int, actual: int) -> float:
    text = " ".join(f"w{i}" for i in range(actual))
    return _word_fit_reward({"target_word_count": target}, Prediction(target_translated_text=text))


def test_exact_target_gets_full_reward():
    assert reward_for(3, 3) == 1.0


def test_reward_decreases_as_normalized_error_grows():
    assert reward_for(40, 40) > reward_for(40, 39) > reward_for(40, 38) > reward_for(40, 36)


@pytest.mark.parametrize(("target", "actual"), [(20, 19), (40, 38)])
def test_tolerance_boundary_scores_one_half(target, actual):
    assert reward_for(target, actual) == pytest.approx(0.5)


def test_empty_translation_gets_no_reward():
    assert reward_for(1, 0) == 0.0


def test_translator_binds_config_and_refines_toward_the_word_budget(monkeypatch):
    calls = {}
    fake_lm = object()

    class FakeRefine:
        def __init__(self, **kwargs):
            calls["refine"] = kwargs

        def __call__(self, **kwargs):
            calls["inputs"] = kwargs
            calls["active_lm"] = dspy.settings.lm
            return Prediction(target_translated_text="Natural spoken English.")

    monkeypatch.setattr(dspy, "LM", lambda *args, **kwargs: fake_lm)
    monkeypatch.setattr(dspy, "Refine", FakeRefine)
    previous_lm = dspy.settings.lm

    translate = make_translator(
        "secret",
        source_language="Chinese",
        target_language="English",
        base_url="https://example.test/v1",
        model_name="test-model",
        job_context="A database performance interview.",
        proper_nouns=("AcmeDB", "Nova"),
        glossary={"查询": "query"},
    )
    result = translate(
        previous_context="我们刚才谈到了这个模型。",
        text_to_translate="嗯这个这个很好",
        target_word_count=4,
    )

    assert result == "Natural spoken English."
    assert calls["active_lm"] is fake_lm
    assert dspy.settings.lm is previous_lm
    assert calls["inputs"] == {
        "source_language": "Chinese",
        "target_language": "English",
        "job_context": "A database performance interview.",
        "proper_nouns": "AcmeDB, Nova",
        "glossary": "查询 -> query",
        "previous_context": "我们刚才谈到了这个模型。",
        "text_to_translate": "嗯这个这个很好",
        "target_word_count": 4,
    }
    assert calls["refine"]["threshold"] == 1.0
    assert calls["refine"]["reward_fn"] is _word_fit_reward
