"""Regression cover for the TTS stage helpers touched by the cleanup pass.

Pure unit tests: no model loading, no network, no ffmpeg. Importing
podcast_dub.stages.tts pulls torch/soundfile/silero_vad/huggingface_hub at
module scope, but nothing here invokes them.
"""

from __future__ import annotations

import pytest

from podcast_dub.stages.tts import (
    MAX_REWRITE_WORDS,
    _greedy_part_count,
    _partition_source_text,
    _split_target_text,
    make_cached_translator,
)
from podcast_dub.types import TranslationUnit


def _unavailable_translator(*, previous_context: str, text_to_translate: str, target_word_count: int) -> str:
    raise AssertionError("the cache key must be computed without calling the translator")


# ---------------- _CachedTranslator._key ----------------


@pytest.mark.parametrize(
    ("namespace", "budget", "text", "expected"),
    [
        ("ns", 12, "hello world", "d7674e9604dc813feb76ca0cc5c95610a21d9160"),
        ("", 4, "你好", "f661c6c4eaebc5762d37613fd37306191ce8a64d"),
        ("digest-abc", 275, "The quick brown fox.", "4908095c60c8701d3f78030fc695e83d0c620662"),
    ],
)
def test_rewrite_cache_key_hex_is_pinned(tmp_path, namespace: str, budget: int, text: str, expected: str) -> None:
    """The rewrite cache key is on-disk state; its sha1 hex must never drift."""
    cached = make_cached_translator(_unavailable_translator, str(tmp_path / "rewrite_cache.json"), namespace)
    assert cached._key(text, budget) == expected


def test_rewrite_cache_key_separates_its_fields(tmp_path) -> None:
    cached = make_cached_translator(_unavailable_translator, str(tmp_path / "rewrite_cache.json"), "ns")
    # A NUL separator keeps neighbouring fields from bleeding into one another.
    assert cached._key("hello world", 12) != cached._key("hello world", 120)
    other = make_cached_translator(_unavailable_translator, str(tmp_path / "other.json"), "ns\0")
    assert other._key("hello world", 12) != cached._key("hello world", 12)


# ---------------- _split_target_text ----------------


def test_split_target_text_conserves_words_in_order() -> None:
    max_words = MAX_REWRITE_WORDS
    text = " ".join(f"w{index}" for index in range(max_words * 2 + 7)) + "."
    parts = _split_target_text(text)
    assert len(parts) > 1
    assert all(len(part.split()) <= max_words for part in parts)
    assert [word for part in parts for word in part.split()] == text.split()


def test_split_target_text_fans_out_a_single_oversized_sentence() -> None:
    """The batched fan-out must keep the short trailing batch."""
    max_words = MAX_REWRITE_WORDS
    words = [f"w{index}" for index in range(max_words * 2 + 3)]
    parts = _split_target_text(" ".join(words))
    assert [len(part.split()) for part in parts] == [max_words, max_words, 3]
    assert [word for part in parts for word in part.split()] == words


def test_split_target_text_keeps_short_text_whole() -> None:
    assert _split_target_text("One. Two. Three.") == ("One. Two. Three.",)
    assert _split_target_text("") == ()


# ---------------- _partition_source_text ----------------


def test_partition_source_text_splits_on_whitespace_tokens() -> None:
    parts = _partition_source_text("a b c d e f", (1, 1))
    assert parts == ("a b c", "d e f")
    assert " ".join(parts).split() == "a b c d e f".split()


def test_partition_source_text_falls_back_to_characters_for_cjk() -> None:
    """CJK source text has no spaces, so a per-character split is the only mapping."""
    source = "这是一个很长的中文句子"
    parts = _partition_source_text(source, (1, 1, 1))
    assert len(parts) == 3
    assert all(" " not in part for part in parts)
    assert "".join(parts) == source


def test_partition_source_text_rejects_unsplittable_source() -> None:
    with pytest.raises(RuntimeError, match="cannot preserve source mapping"):
        _partition_source_text("ab", (1, 1, 1))


# ---------------- _greedy_part_count ----------------


def _turn(word_counts: list[int]) -> list[TranslationUnit]:
    return [
        TranslationUnit(
            start=float(index),
            end=float(index) + 1.0,
            speaker="A",
            source_text="src",
            target_text=" ".join(["w"] * count) or "x",
        )
        for index, count in enumerate(word_counts)
    ]


@pytest.mark.parametrize(
    ("word_counts", "expected"),
    [
        ([], 0),
        ([1], 1),
        # a single unit is never split, however oversized
        ([600], 1),
        # MAX_REWRITE_WORDS (275) is the per-part ceiling: 274+1 still fits
        ([274, 1], 1),
        ([274, 2], 2),
        ([275, 275], 2),
        ([1, 1, 1], 1),
        ([137, 137, 1, 1], 2),
        ([300, 300, 300], 3),
    ],
)
def test_greedy_part_count(word_counts: list[int], expected: int) -> None:
    assert _greedy_part_count(_turn(word_counts)) == expected


def test_greedy_part_count_never_exceeds_the_unit_count() -> None:
    for word_counts in ([], [500], [500, 500], [1, 2, 3]):
        turn = _turn(word_counts)
        assert _greedy_part_count(turn) <= len(turn)
