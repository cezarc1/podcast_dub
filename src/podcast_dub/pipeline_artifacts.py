"""Artifact specifications for each persisted pipeline record family."""

from podcast_dub.artifacts import ArtifactSpec
from podcast_dub.models import (
    DiarizationSegment,
    Phrase,
    PlacementResult,
    SpeakerPhrase,
    SpeakerReference,
    StageName,
    TranslationUnit,
    TurnChunk,
)

PHRASES: ArtifactSpec[tuple[Phrase, ...]] = ArtifactSpec(stage=StageName.ASR, payload_type=tuple[Phrase, ...])
DIARIZATION_SEGMENTS: ArtifactSpec[tuple[DiarizationSegment, ...]] = ArtifactSpec(
    stage=StageName.DIARIZE, payload_type=tuple[DiarizationSegment, ...]
)
SPEAKER_PHRASES: ArtifactSpec[tuple[SpeakerPhrase, ...]] = ArtifactSpec(
    stage=StageName.DIARIZE, payload_type=tuple[SpeakerPhrase, ...]
)
SPEAKER_REFERENCES: ArtifactSpec[tuple[SpeakerReference, ...]] = ArtifactSpec(
    stage=StageName.REFS, payload_type=tuple[SpeakerReference, ...]
)
TRANSLATION_UNITS: ArtifactSpec[tuple[TranslationUnit, ...]] = ArtifactSpec(
    stage=StageName.TRANSLATE, payload_type=tuple[TranslationUnit, ...]
)
TURN_CHUNKS: ArtifactSpec[tuple[TurnChunk, ...]] = ArtifactSpec(stage=StageName.TTS, payload_type=tuple[TurnChunk, ...])
PLACEMENT_RESULT: ArtifactSpec[PlacementResult] = ArtifactSpec(stage=StageName.PLACE, payload_type=PlacementResult)
