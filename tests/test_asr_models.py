import pytest

from podcast_dub.models import AlignedWord, Phrase
from podcast_dub.stages import asr
from podcast_dub.stages.asr import _group_phrases


def test_group_phrases_returns_immutable_phrase_models() -> None:
    words = (
        AlignedWord(text="你", start=0.0, end=0.2),
        AlignedWord(text="好。", start=0.2, end=0.5),
        AlignedWord(text="再见", start=1.0, end=1.4),
    )

    phrases = _group_phrases(words, offset=2.0)

    assert all(isinstance(phrase, Phrase) for phrase in phrases)
    assert [phrase.text for phrase in phrases] == ["你好。", "再见"]
    assert phrases[0].words[0].start == 2.0


def test_asr_chunk_ranges_stop_at_the_requested_window(monkeypatch) -> None:
    monkeypatch.setattr(asr, "CHUNK_S", 100.0)

    assert asr._chunk_ranges(250.0) == ((0.0, 100.0), (100.0, 100.0), (200.0, 50.0))


def test_alignment_quality_rejects_a_collapsed_timestamp_run() -> None:
    words = tuple(AlignedWord(text=str(index), start=99.0, end=99.0) for index in range(9))

    with pytest.raises(RuntimeError, match=r"forced aligner collapsed 9 consecutive words at 99\.000s"):
        asr._validate_alignment_quality(words)


def test_alignment_quality_allows_small_quantized_timestamp_runs() -> None:
    words = (
        AlignedWord(text="你", start=1.0, end=1.0),
        AlignedWord(text="好", start=1.0, end=1.0),
        AlignedWord(text="啊", start=1.0, end=1.0),
        AlignedWord(text="。", start=1.0, end=1.08),
    )

    asr._validate_alignment_quality(words)
