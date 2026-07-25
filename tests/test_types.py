import pytest
from pydantic import ValidationError

from podcast_dub.types import (
    AlignedWord,
    DevicePlan,
    DiarizationSegment,
    Phrase,
    PhraseWord,
    SpeakerPhrase,
    TranslationUnit,
    TurnChunk,
)


def test_timed_models_reject_non_positive_spans() -> None:
    with pytest.raises(ValidationError, match="end must be greater than start"):
        DiarizationSegment(start=1.0, end=1.0, speaker="speaker_0")


def test_phrase_rejects_out_of_order_words() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        Phrase(
            start=0.0,
            end=2.0,
            text="ab",
            words=(
                PhraseWord(text="b", start=1.0, end=1.2),
                PhraseWord(text="a", start=0.2, end=0.4),
            ),
        )


def test_models_are_deeply_immutable_value_objects() -> None:
    phrase = SpeakerPhrase(
        start=0.0,
        end=1.0,
        text="hello",
        words=(PhraseWord(text="hello", start=0.0, end=1.0),),
        speaker="host",
    )

    translated = TranslationUnit.from_phrase(phrase, target_text="bonjour")

    assert translated.source_text == "hello"
    assert translated.target_text == "bonjour"
    assert isinstance(translated.words, tuple)
    with pytest.raises(ValidationError, match="frozen"):
        translated.target_text = "salut"  # ty: ignore[invalid-assignment]

    with pytest.raises(ValidationError, match="greater than or equal"):
        phrase.validated_copy(end=-1.0)


def test_turn_chunk_requires_complete_audio_metadata() -> None:
    with pytest.raises(ValidationError):
        TurnChunk(
            start=0.0,
            end=1.0,
            speaker="host",
            text="hello",
            turn_id=0,
            part_index=0,
            audio_file="",
            audio_duration_s=0.0,
            audio_sha256="",
        )


def test_backend_and_device_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AlignedWord.model_validate({"text": "hello", "start": 0.0, "end": 1.0, "confidence": 0.9})

    with pytest.raises(ValidationError):
        DevicePlan(stage="diarize", device="mps", dtype="float32", attention="default")
