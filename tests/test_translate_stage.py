from podcast_dub.artifacts import ArtifactProvenance, stable_digest, write_artifact_atomic
from podcast_dub.models import SpeakerPhrase
from podcast_dub.pipeline_artifacts import SPEAKER_PHRASES, TRANSLATION_UNITS
from podcast_dub.stages import translate as stage


def test_translate_stage_translates_each_phrase_with_source_only_context(monkeypatch, tmp_path, make_job_config):
    monkeypatch.delenv("DUB_TRANSLATE_BASE_URL", raising=False)
    monkeypatch.delenv("DUB_TRANSLATE_MODEL", raising=False)
    monkeypatch.delenv("DUB_TRANSLATE_API_KEY", raising=False)
    phrases = (
        SpeakerPhrase(start=0.0, end=1.0, speaker="S0", text="first"),
        SpeakerPhrase(start=1.0, end=2.0, speaker="S0", text="second"),
        SpeakerPhrase(start=2.0, end=3.0, speaker="S1", text="third"),
        SpeakerPhrase(start=3.0, end=4.0, speaker="S1", text="fourth"),
    )
    source_provenance = ArtifactProvenance(
        config_digest=stable_digest({"source": "test"}),
        parameters_digest=stable_digest({}),
    )
    write_artifact_atomic(tmp_path / "phrases_spk.json", SPEAKER_PHRASES, source_provenance, phrases)
    translate_calls: list[dict[str, object]] = []
    calls: dict[str, object] = {"translations": translate_calls}

    def fake_factory(
        api_key,
        source_language,
        target_language,
        base_url,
        model_name,
        *,
        job_context,
        proper_nouns,
        glossary,
    ):
        calls["factory"] = {
            "api_key": api_key,
            "source_language": source_language,
            "target_language": target_language,
            "base_url": base_url,
            "model_name": model_name,
            "job_context": job_context,
            "proper_nouns": proper_nouns,
            "glossary": glossary,
        }

        def translate(*, previous_context, text_to_translate, target_word_count):
            translate_calls.append(
                {
                    "previous_context": previous_context,
                    "text_to_translate": text_to_translate,
                    "target_word_count": target_word_count,
                }
            )
            return f"translated {text_to_translate}"

        return translate

    monkeypatch.setattr(stage, "BATCH", 2)
    monkeypatch.setattr(stage, "WORKERS", 1)
    monkeypatch.setattr(stage, "PREVIOUS_TURNS", 3)
    monkeypatch.setattr(stage, "make_translator", fake_factory, raising=False)

    assert not hasattr(stage, "llm_batch")
    cfg = make_job_config(
        context="A database performance interview.",
        proper_nouns=("AcmeDB", "Nova"),
        glossary={"查询": "query"},
        llm_base="https://example.test/v1",
        llm_model="test-model",
        llm_key="secret",
    )
    out = stage.run_translate(cfg)

    units = TRANSLATION_UNITS.adapter().validate_json((tmp_path / "units.json").read_bytes()).payload
    assert [unit.target_text for unit in units] == [
        "translated first",
        "translated second",
        "translated third",
        "translated fourth",
    ]
    assert out == str(tmp_path / "units.json")
    assert calls["factory"] == {
        "api_key": "secret",
        "source_language": "Chinese",
        "target_language": "English",
        "base_url": "https://example.test/v1",
        "model_name": "test-model",
        "job_context": "A database performance interview.",
        "proper_nouns": ("AcmeDB", "Nova"),
        "glossary": {"查询": "query"},
    }
    # Each phrase is translated on its own, with source-only rolling history and a
    # duration-based word budget (1.0s * 2.5 words/s -> 2).
    assert translate_calls == [
        {"previous_context": "", "text_to_translate": "first", "target_word_count": 2},
        {"previous_context": "first", "text_to_translate": "second", "target_word_count": 2},
        {"previous_context": "first second", "text_to_translate": "third", "target_word_count": 2},
        {"previous_context": "first second third", "text_to_translate": "fourth", "target_word_count": 2},
    ]

    # A different model changes provenance -> cache miss -> everything re-translates.
    calls.pop("factory")
    translate_calls.clear()

    stage.run_translate(cfg.validated_copy(llm_model="second-model"))

    assert calls["factory"] == {
        "api_key": "secret",
        "source_language": "Chinese",
        "target_language": "English",
        "base_url": "https://example.test/v1",
        "model_name": "second-model",
        "job_context": "A database performance interview.",
        "proper_nouns": ("AcmeDB", "Nova"),
        "glossary": {"查询": "query"},
    }
    assert len(translate_calls) == 4
