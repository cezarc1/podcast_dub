from podcast_dub import manifest
from podcast_dub.types import RewriteEvent, TranslateEvent, TranslationManifestLine


def test_manifest_round_trips_translation_event(tmp_path):
    manifest.configure(str(tmp_path))
    event = TranslateEvent(
        batch_index=2,
        ids=(7,),
        model="test-model",
        lines=(TranslationManifestLine(id=7, source_text="你好", target_text="Hello"),),
    )

    manifest.log_event(event)

    assert list(manifest.read_events()) == [event]


def test_manifest_uses_discriminated_rewrite_events(tmp_path):
    manifest.configure(str(tmp_path))
    event = RewriteEvent(
        kind="rewrite_tighter",
        turn="t1p0",
        speaker="host",
        before="a much longer sentence",
        after="short sentence",
        budget_words=2,
        words_before=4,
        words_after=2,
        duration_before_s=2.0,
        duration_after_s=1.0,
        window_s=1.2,
        ratio=1.67,
        attempt=0,
        model="test-model",
    )

    manifest.log_event(event)

    assert list(manifest.read_events()) == [event]
