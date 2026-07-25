import numpy as np
import pytest

from podcast_dub.audio_utils import SR
from podcast_dub.stages import tts
from podcast_dub.stages.tts import build_turns
from podcast_dub.types import TranslationUnit, TurnChunk, TurnChunkDraft


def test_build_turns_returns_typed_immutable_drafts() -> None:
    units = (
        TranslationUnit(
            start=0.0,
            end=1.0,
            speaker="host",
            source_text="你好",
            target_text="Hello.",
        ),
        TranslationUnit(
            start=1.2,
            end=2.0,
            speaker="host",
            source_text="再见",
            target_text="Goodbye.",
        ),
    )

    chunks = build_turns(units)

    assert chunks == (
        TurnChunkDraft(
            start=0.0,
            end=2.0,
            speaker="host",
            text="Hello. Goodbye.",
            source_text="你好 再见",
            turn_id=0,
            part_index=0,
        ),
    )


def test_build_turns_keeps_matching_source_text_when_splitting(monkeypatch) -> None:
    units = (
        TranslationUnit(
            start=0.0,
            end=1.0,
            speaker="host",
            source_text="源一",
            target_text="one two three.",
        ),
        TranslationUnit(
            start=1.1,
            end=2.0,
            speaker="host",
            source_text="源二",
            target_text="four five six.",
        ),
    )
    monkeypatch.setattr(tts, "MAX_CHUNK_S", 2.0)

    chunks = build_turns(units)

    assert [chunk.text for chunk in chunks] == ["one two three.", "four five six."]
    assert [chunk.source_text for chunk in chunks] == ["源一", "源二"]


def test_build_turns_adds_rewrite_capacity_for_a_long_sparse_turn(monkeypatch) -> None:
    units = (
        TranslationUnit(
            start=0.0,
            end=6.0,
            speaker="host",
            source_text="第一部分",
            target_text="First part.",
        ),
        TranslationUnit(
            start=6.1,
            end=12.0,
            speaker="host",
            source_text="第二部分",
            target_text="Second part.",
        ),
    )
    monkeypatch.setattr(tts, "MAX_CHUNK_S", 10.0)

    chunks = build_turns(units)

    assert len(chunks) == 2
    assert [chunk.source_text for chunk in chunks] == ["第一部分", "第二部分"]
    assert chunks[0].end == chunks[1].start
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 12.0


def test_build_turns_balances_rewrite_capacity_across_chunks(monkeypatch) -> None:
    units = tuple(
        TranslationUnit(
            start=float(index * 2),
            end=float((index + 1) * 2),
            speaker="host",
            source_text=f"源{index}",
            target_text=f"part {index} done.",
        )
        for index in range(4)
    )
    monkeypatch.setattr(tts, "MAX_CHUNK_S", 4.0)

    chunks = build_turns(units)

    assert [len(chunk.text.split()) for chunk in chunks] == [6, 6]
    assert [chunk.source_text for chunk in chunks] == ["源0 源1", "源2 源3"]


def test_tts_preflight_rewrites_audio_that_capped_fit_cannot_place(monkeypatch) -> None:
    decoded_duration_s = 4.3165
    chunk = TurnChunk(
        start=9.9,
        end=12.94,
        speaker="host",
        text="Over the past year, TikTok has swung between peaks and valleys.",
        turn_id=1,
        part_index=0,
        audio_file="/tmp/turn-1.mp3",
        audio_duration_s=4.368,
        audio_sha256="a" * 64,
    )
    monkeypatch.setattr(
        tts,
        "decode_f32",
        lambda _path, _tempo=1.0: np.ones(round(decoded_duration_s * SR), dtype=np.float32),
    )

    fitted = tts._evaluate_chunks((chunk,), total_s=13.76)

    assert chunk.audio_duration_s / fitted[0].assessment.window_s < 1.22
    assert fitted[0].assessment.rewrite_direction == "tighter"
    with pytest.raises(RuntimeError, match=r"tts: fit exhausted.*turn 1"):
        tts._require_final_fit(fitted)


def test_tts_final_fit_rejects_an_unresolved_fuller_turn(monkeypatch) -> None:
    chunk = TurnChunk(
        start=0.0,
        end=2.0,
        speaker="host",
        text="A brief answer.",
        turn_id=12,
        part_index=0,
        audio_file="/tmp/turn-12.mp3",
        audio_duration_s=2.0,
        audio_sha256="b" * 64,
    )
    monkeypatch.setattr(
        tts,
        "decode_f32",
        lambda _path, _tempo=1.0: np.ones(round(2.0 * SR), dtype=np.float32),
    )
    fitted = tts._evaluate_chunks((chunk,), total_s=10.25)

    assert fitted[0].assessment.rewrite_direction == "fuller"
    assert tts._select_rewrite_work(fitted) == list(fitted)
    with pytest.raises(RuntimeError, match=r"tts: fit exhausted.*turn 12"):
        tts._require_final_fit(fitted)


def test_fuller_turn_at_the_word_ceiling_is_not_rewritten_or_failed(monkeypatch) -> None:
    # Same shape as the test above, but the chunk already holds MAX_REWRITE_WORDS.
    # The budget allocator caps every chunk at that ceiling, so asking for "fuller"
    # text cannot change the duration — the loop must not spend three rounds of LLM
    # calls on it, and must not fail the run over a gap nothing could close.
    chunk = TurnChunk(
        start=0.0,
        end=2.0,
        speaker="host",
        text=" ".join(["word"] * tts.MAX_REWRITE_WORDS),
        turn_id=12,
        part_index=0,
        audio_file="/tmp/turn-12.mp3",
        audio_duration_s=2.0,
        audio_sha256="b" * 64,
    )
    monkeypatch.setattr(
        tts,
        "decode_f32",
        lambda _path, _tempo=1.0: np.ones(round(2.0 * SR), dtype=np.float32),
    )
    fitted = tts._evaluate_chunks((chunk,), total_s=10.25)

    assert fitted[0].assessment.rewrite_direction == "fuller"
    assert tts._select_rewrite_work(fitted) == []
    tts._require_final_fit(fitted)


def test_fuller_word_budgets_redistribute_oversized_chunk_budget() -> None:
    budgets = tts._allocate_rewrite_budgets(
        (260, 161),
        scale=1.39,
        floor=4,
        ceiling=275,
    )

    assert budgets == (275, 275)
    assert sum(budgets) == 550
    assert max(budgets) <= 275


def test_rewrite_budgets_interpolate_between_measured_short_and_long_audio() -> None:
    budgets = tts._interpolate_rewrite_budgets(
        lower_word_counts=(239, 239),
        lower_duration_s=130.278,
        upper_word_counts=(275, 275),
        upper_duration_s=168.99,
        target_duration_s=154.239,
        floor=4,
        ceiling=275,
    )

    assert budgets == (261, 261)


def test_snap_turn_starts_keeps_turn_inside_continuous_speech(monkeypatch) -> None:
    sample_rate = 16_000
    draft = TurnChunkDraft(
        start=2.0,
        end=3.0,
        speaker="host",
        text="Continuous speech.",
        turn_id=4,
        part_index=0,
    )
    monkeypatch.setattr(tts, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        tts,
        "get_speech_timestamps",
        lambda *_args, **_kwargs: [{"start": 0, "end": 5 * sample_rate}],
    )

    snapped = tts.snap_turn_starts((draft,), np.ones(5 * sample_rate, dtype=np.float32), sr=sample_rate)

    assert snapped == (draft,)


def test_snap_turn_starts_drops_turn_without_onset_or_speech_overlap(monkeypatch) -> None:
    sample_rate = 16_000
    draft = TurnChunkDraft(
        start=2.0,
        end=3.0,
        speaker="host",
        text="Hallucinated speech.",
        turn_id=4,
        part_index=0,
    )
    monkeypatch.setattr(tts, "load_silero_vad", lambda: object())
    monkeypatch.setattr(
        tts,
        "get_speech_timestamps",
        lambda *_args, **_kwargs: [{"start": 0, "end": sample_rate}],
    )

    snapped = tts.snap_turn_starts((draft,), np.ones(5 * sample_rate, dtype=np.float32), sr=sample_rate)

    assert snapped == ()
