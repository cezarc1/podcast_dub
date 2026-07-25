import json

import pytest

from podcast_dub.artifacts import (
    ArtifactError,
    ArtifactProvenance,
    ArtifactSpec,
    InputDigest,
    load_cached_artifact,
    stable_digest,
    write_artifact_atomic,
)
from podcast_dub.types import Phrase, PhraseWord, StageName

PHRASES: ArtifactSpec[tuple[Phrase, ...]] = ArtifactSpec(stage=StageName.ASR, payload_type=tuple[Phrase, ...])


def provenance(config: str = "a") -> ArtifactProvenance:
    return ArtifactProvenance(
        input_digests=(InputDigest(name="audio", sha256="a" * 64),),
        config_digest=stable_digest({"config": config}),
        parameters_digest=stable_digest({"chunk_s": 300}),
    )


def test_artifact_round_trip_and_stable_serialization(tmp_path) -> None:
    path = tmp_path / "phrases.json"
    payload = (
        Phrase(
            start=0.0,
            end=1.0,
            text="hello",
            words=(PhraseWord(text="hello", start=0.0, end=1.0),),
        ),
    )

    write_artifact_atomic(path, PHRASES, provenance(), payload)
    first = path.read_bytes()
    loaded = load_cached_artifact(path, PHRASES, provenance())
    write_artifact_atomic(path, PHRASES, provenance(), payload)

    assert loaded == payload
    assert path.read_bytes() == first
    assert not (tmp_path / "phrases.json.tmp").exists()


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        json.dumps({"schema_version": 99, "stage": "asr", "provenance": {}, "payload": []}),
    ],
)
def test_legacy_or_wrong_version_is_a_cache_miss(tmp_path, contents) -> None:
    path = tmp_path / "phrases.json"
    path.write_text(contents)

    assert load_cached_artifact(path, PHRASES, provenance()) is None


def test_provenance_mismatch_is_a_cache_miss(tmp_path) -> None:
    path = tmp_path / "phrases.json"
    write_artifact_atomic(path, PHRASES, provenance("first"), ())

    assert load_cached_artifact(path, PHRASES, provenance("second")) is None


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        json.dumps(
            {
                "schema_version": 1,
                "stage": "asr",
                "provenance": {
                    "input_digests": [{"name": "audio", "sha256": "a" * 64}],
                    "config_digest": stable_digest({"config": "a"}),
                    "parameters_digest": stable_digest({"chunk_s": 300}),
                },
                "payload": [{"start": 2, "end": 1, "text": "broken", "words": []}],
            }
        ),
    ],
)
def test_current_schema_corruption_is_a_hard_error(tmp_path, contents) -> None:
    path = tmp_path / "phrases.json"
    path.write_text(contents)

    with pytest.raises(ArtifactError, match=str(path)):
        load_cached_artifact(path, PHRASES, provenance())
