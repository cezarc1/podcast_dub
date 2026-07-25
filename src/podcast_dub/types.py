from __future__ import annotations

import math
import time
from enum import StrEnum, auto
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, model_validator


class StageName(StrEnum):
    ASR = auto()
    DIARIZE = auto()
    REFS = auto()
    TRANSLATE = auto()
    TTS = auto()
    PLACE = auto()


class ConcreteDevice(StrEnum):
    CUDA = auto()
    MPS = auto()
    CPU = auto()


class ModelStage(StrEnum):
    """The pipeline stages that load an ML model onto a device — a subset of StageName."""

    ASR = auto()
    DIARIZE = auto()
    TTS = auto()


class StrictModel(BaseModel):
    """Default contract for pipeline records: closed schema and immutable values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def validated_copy(self, **changes: Any) -> Self:
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(changes)
        return type(self).model_validate(values)


class DevicePlan(StrictModel):
    stage: ModelStage
    device: ConcreteDevice
    dtype: str = Field(min_length=1)
    attention: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stage_device(self) -> Self:
        if self.stage == ModelStage.DIARIZE and self.device == ConcreteDevice.MPS:
            raise ValueError("diarize does not support the mps device")
        return self


class ModelIdentity(StrictModel):
    identifier: str = Field(min_length=1)
    revision: str | None = None


class _PositiveTimeSpan(StrictModel):
    start: NonNegativeFloat
    end: NonNegativeFloat

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("timestamps must be finite")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class _NonNegativeTimeSpan(StrictModel):
    start: NonNegativeFloat
    end: NonNegativeFloat

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("timestamps must be finite")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class _WordTimeSpan(StrictModel):
    text: str = Field(min_length=1)
    start: NonNegativeFloat
    end: NonNegativeFloat

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("timestamps must be finite")
        if self.end < self.start:
            raise ValueError("word end must be greater than or equal to start")
        return self


class AlignedWord(_WordTimeSpan):
    """Word timing returned by the forced aligner."""


class PhraseWord(_WordTimeSpan):
    """Normalized word timing persisted with an ASR phrase."""


class Phrase(_NonNegativeTimeSpan):
    text: str = Field(min_length=1)
    words: tuple[PhraseWord, ...] = ()

    @model_validator(mode="after")
    def validate_words(self) -> Self:
        previous_start = -1.0
        for word in self.words:
            if word.start < previous_start:
                raise ValueError("phrase words must be ordered by start time")
            if word.start < self.start - 0.001 or word.end > self.end + 0.001:
                raise ValueError("phrase word timing must be inside the phrase span")
            previous_start = word.start
        return self


class DiarizationSegment(_PositiveTimeSpan):
    speaker: str = Field(min_length=1)


class SpeakerPhrase(Phrase):
    speaker: str = Field(min_length=1)


class TranslationUnit(_NonNegativeTimeSpan):
    speaker: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    target_text: str = Field(min_length=1)
    words: tuple[PhraseWord, ...] = ()

    @classmethod
    def from_phrase(cls, phrase: SpeakerPhrase, *, target_text: str) -> TranslationUnit:
        return cls(
            start=phrase.start,
            end=phrase.end,
            speaker=phrase.speaker,
            source_text=phrase.text,
            target_text=target_text,
            words=phrase.words,
        )


class SubtitleCue(_PositiveTimeSpan):
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SubtitleUnit(_PositiveTimeSpan):
    speaker: str = Field(min_length=1)
    subtitle_text: str = Field(min_length=1)
    frames: tuple[int, ...]


class TurnChunkDraft(_PositiveTimeSpan):
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    # Source-language text this turn was translated from. Carried so the TTS fit
    # loop can re-translate (not paraphrase) a turn to a tighter word budget.
    # Optional: built turns always set it; ad-hoc drafts (tests/tools) may omit it.
    source_text: str = ""
    turn_id: int = Field(ge=0)
    part_index: int = Field(ge=0)


class TurnChunk(TurnChunkDraft):
    audio_file: str = Field(min_length=1)
    audio_duration_s: PositiveFloat
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_draft(
        cls,
        draft: TurnChunkDraft,
        *,
        audio_file: str,
        audio_duration_s: float,
        audio_sha256: str,
    ) -> TurnChunk:
        return cls(
            **draft.model_dump(),
            audio_file=audio_file,
            audio_duration_s=audio_duration_s,
            audio_sha256=audio_sha256,
        )


class SimulationTurn(TurnChunkDraft):
    audio_duration_s: PositiveFloat


class LogicalTurn(_PositiveTimeSpan):
    speaker: str = Field(min_length=1)
    turn_id: int = Field(ge=0)
    chunks: tuple[TurnChunk, ...]

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        if not self.chunks:
            raise ValueError("logical turn requires at least one chunk")
        if any(chunk.turn_id != self.turn_id or chunk.speaker != self.speaker for chunk in self.chunks):
            raise ValueError("logical turn chunks must share turn id and speaker")
        return self


class TurnFitAssessment(StrictModel):
    turn_id: int = Field(ge=0)
    start_s: NonNegativeFloat
    end_s: NonNegativeFloat
    cue_start_s: NonNegativeFloat
    window_s: PositiveFloat
    input_duration_s: PositiveFloat
    fitted_duration_s: PositiveFloat
    lag_s: float
    fit_notes: tuple[str, ...] = ()
    rewrite_direction: Literal["tighter", "fuller"] | None = None


class SpeakerReference(StrictModel):
    speaker: str = Field(min_length=1)
    audio_file: str = Field(min_length=1)
    transcript_file: str
    duration_s: PositiveFloat
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VerificationResult(StrictModel):
    coverage: float = Field(ge=0.0, le=1.0)
    longest_dead_air_s: NonNegativeFloat
    passed: bool


class PlacementResult(StrictModel):
    output_file: str = Field(min_length=1)
    voice_file: str = Field(min_length=1)
    mix_file: str = Field(min_length=1)
    subtitles_file: str = Field(min_length=1)
    verification: VerificationResult


class TranslationBatch(StrictModel):
    batch_index: int = Field(ge=0)
    translations: dict[int, str]

    @model_validator(mode="after")
    def validate_translations(self) -> Self:
        if any(not text.strip() for text in self.translations.values()):
            raise ValueError("translation batch contains empty target text")
        return self


class RewriteCacheEntry(StrictModel):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)


class RewriteCache(StrictModel):
    entries: tuple[RewriteCacheEntry, ...] = ()

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        keys = [entry.key for entry in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("rewrite cache keys must be unique")
        return self

    def as_dict(self) -> dict[str, str]:
        return {entry.key: entry.value for entry in self.entries}


class TranslationManifestLine(StrictModel):
    id: int = Field(ge=0)
    source_text: str = Field(min_length=1)
    target_text: str = Field(min_length=1)


class _ManifestEventBase(StrictModel):
    timestamp_s: float = Field(default_factory=lambda: round(time.time(), 3), gt=0)


class TranslateEvent(_ManifestEventBase):
    kind: Literal["translate"] = "translate"
    batch_index: int = Field(ge=0)
    ids: tuple[int, ...]
    model: str = Field(min_length=1)
    lines: tuple[TranslationManifestLine, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("manifest translation ids must be unique")
        if tuple(line.id for line in self.lines) != self.ids:
            raise ValueError("manifest lines must match ids in order")
        return self


class RewriteEvent(_ManifestEventBase):
    kind: Literal["rewrite_tighter", "rewrite_fuller"]
    turn: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)
    budget_words: int = Field(gt=0)
    words_before: int = Field(gt=0)
    words_after: int = Field(gt=0)
    duration_before_s: PositiveFloat
    duration_after_s: PositiveFloat
    window_s: PositiveFloat
    ratio: PositiveFloat
    attempt: int = Field(ge=0)
    model: str = Field(min_length=1)


ManifestEvent = Annotated[TranslateEvent | RewriteEvent, Field(discriminator="kind")]


class AsrBackendRequest(StrictModel):
    audio_file: str = Field(min_length=1)
    source_language: str = Field(min_length=1)
    window_s: PositiveFloat | None = None
    plan: DevicePlan

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.plan.stage != ModelStage.ASR:
            raise ValueError("ASR request requires an ASR device plan")
        return self


class AsrBackendResult(StrictModel):
    words: tuple[AlignedWord, ...]
    phrases: tuple[Phrase, ...]
    model: ModelIdentity


class DiarizationBackendRequest(StrictModel):
    audio_file: str = Field(min_length=1)
    plan: DevicePlan

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.plan.stage != ModelStage.DIARIZE:
            raise ValueError("diarization request requires a diarization device plan")
        return self


class DiarizationBackendResult(StrictModel):
    segments: tuple[DiarizationSegment, ...]
    model: ModelIdentity
