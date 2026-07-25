"""Unit tests for the rewrite idempotence cache (podcast_dub.stages.tts.make_cached_translator).

The TTS rewrite loop must be resumable: a crashed stage that rebuilds turns
from scratch re-derives the same (text, budget) pairs, and must NOT re-fire
LLM calls or regenerate audio for work already done. The cache is keyed by
(namespace, target_word_count, text) — previous_context does not affect it.
"""

import os

import podcast_dub.stages.tts as tts
from podcast_dub.stages.tts import MIN_CACHED_AUDIO_BYTES, make_cached_translator
from podcast_dub.types import RewriteCache, TurnChunkDraft


class TestCachedTranslator:
    def test_second_call_same_args_is_cached(self, tmp_path):
        calls = []

        def fake(*, previous_context, text_to_translate, target_word_count):
            calls.append((text_to_translate, target_word_count))
            return f"rw:{text_to_translate}"

        rw = make_cached_translator(fake, str(tmp_path / "rc.json"))
        assert rw(previous_context="", text="hello world", target_word_count=5) == "rw:hello world"
        assert rw(previous_context="", text="hello world", target_word_count=5) == "rw:hello world"
        assert calls == [("hello world", 5)]

    def test_different_budget_is_a_different_entry(self, tmp_path):
        calls = []

        def fake(*, previous_context, text_to_translate, target_word_count):
            calls.append((text_to_translate, target_word_count))
            return f"rw{target_word_count}:{text_to_translate}"

        rw = make_cached_translator(fake, str(tmp_path / "rc.json"))
        assert rw(previous_context="", text="hello", target_word_count=5) == "rw5:hello"
        assert rw(previous_context="", text="hello", target_word_count=9) == "rw9:hello"
        assert len(calls) == 2

    def test_different_namespace_is_a_different_entry(self, tmp_path):
        rc = str(tmp_path / "rc.json")
        calls = []

        def fake(*, previous_context, text_to_translate, target_word_count):
            calls.append((text_to_translate, target_word_count))
            return f"result-{len(calls)}"

        first = make_cached_translator(fake, rc, namespace="model-a")
        second = make_cached_translator(fake, rc, namespace="model-b")

        assert first(previous_context="", text="hello", target_word_count=5) == "result-1"
        assert second(previous_context="", text="hello", target_word_count=5) == "result-2"
        assert calls == [("hello", 5), ("hello", 5)]

    def test_cache_persists_across_instances(self, tmp_path):
        rc = str(tmp_path / "rc.json")
        calls = []

        def fake(*, previous_context, text_to_translate, target_word_count):
            calls.append(text_to_translate)
            return "out"

        make_cached_translator(fake, rc)(previous_context="", text="some text", target_word_count=4)
        # new instance over the same file (the resume case): no LLM call
        rw2 = make_cached_translator(fake, rc)
        assert rw2(previous_context="", text="some text", target_word_count=4) == "out"
        assert calls == ["some text"]
        on_disk = RewriteCache.model_validate_json(open(rc, "rb").read())
        assert len(on_disk.entries) == 1 and "out" in on_disk.as_dict().values()

    def test_cache_file_is_valid_json_after_each_write(self, tmp_path):
        rc = str(tmp_path / "rc.json")
        rw = make_cached_translator(
            lambda *, previous_context, text_to_translate, target_word_count: text_to_translate.upper(), rc
        )
        rw(previous_context="", text="a", target_word_count=1)
        rw(previous_context="", text="b", target_word_count=2)
        assert RewriteCache.model_validate_json(open(rc, "rb").read()).entries
        assert not os.path.exists(rc + ".tmp")


def test_tts_translator_binds_job_configuration(monkeypatch):
    calls = {}

    def fake_make(api_key, **kwargs):
        calls["factory"] = {"api_key": api_key, **kwargs}

        def translate(*, previous_context, text_to_translate, target_word_count):
            calls["translate"] = (previous_context, text_to_translate, target_word_count)
            return "short result"

        return translate

    monkeypatch.setattr(tts, "_make", fake_make)
    translate = tts.make_translator(
        source_language="Chinese",
        target_language="English",
        base_url="https://example.test/v1",
        model="test-model",
        key="secret",
        job_context="A database performance interview.",
        proper_nouns=("AcmeDB", "Nova"),
        glossary={"查询": "query"},
    )

    assert translate(previous_context="", text_to_translate="源文本", target_word_count=2) == "short result"
    assert calls["translate"] == ("", "源文本", 2)
    assert calls["factory"] == {
        "api_key": "secret",
        "model_name": "test-model",
        "source_language": "Chinese",
        "target_language": "English",
        "base_url": "https://example.test/v1",
        "job_context": "A database performance interview.",
        "proper_nouns": ("AcmeDB", "Nova"),
        "glossary": {"查询": "query"},
    }


def test_valid_cached_audio_skips_generation(monkeypatch, tmp_path):
    dst = tmp_path / "rewritten.mp3"
    dst.write_bytes(b"x" * (MIN_CACHED_AUDIO_BYTES + 1))
    generated = []
    monkeypatch.setattr(tts, "_gen_file", lambda *args: generated.append(args))

    draft = TurnChunkDraft(start=0.0, end=1.0, speaker="host", text="cached rewrite", turn_id=0, part_index=0)
    tts._generate_unless_cached(None, {}, draft, str(dst), "English")

    assert generated == []
