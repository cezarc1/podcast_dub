import argparse
import logging

import pytest
from pydantic import ValidationError

from podcast_dub.config import lang_name, load_toml, merge_cli, resolve_translation_api


def test_unknown_language_name_warns_before_using_code_verbatim(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="podcast_dub.config"):
        display = lang_name("klingon")

    assert display == "klingon"
    assert "unrecognized language code 'klingon'; using it verbatim" in caplog.text


def test_load_toml_reads_llm_configuration(tmp_path):
    config = tmp_path / "dub.toml"
    config.write_text(
        """
video = "episode.mp4"
source_lang = "zh"
target_lang = "en"
llm_model = "test-model"
llm_base = "https://example.test/v1"
llm_key = "test-key"
""".strip()
    )

    cfg = load_toml(str(config))

    assert cfg.llm_model == "test-model"
    assert cfg.llm_base == "https://example.test/v1"
    assert cfg.llm_key == "test-key"


def test_merge_preserves_secret_without_serializing_it(tmp_path):
    config = tmp_path / "dub.toml"
    config.write_text('video = "episode.mp4"\nsource_lang = "zh"\ntarget_lang = "en"\nllm_key = "test-key"')

    cfg = merge_cli(load_toml(config), argparse.Namespace())

    assert cfg.llm_key == "test-key"
    assert "test-key" not in repr(cfg)
    assert "llm_key" not in cfg.model_dump()
    assert "llm_key" not in cfg.provenance_config()


def test_config_restores_context_fields_and_legacy_language_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / "dub.toml"
    config.write_text(
        """
video = "~/episode.mp4"
from = "zh"
to = "en"
context = "An interview"
proper_nouns = ["Kimi"]
glossary = { "月之暗面" = "Moonshot AI" }
""".strip()
    )

    cfg = load_toml(config)

    assert cfg.video == str(tmp_path / "episode.mp4")
    assert cfg.source_lang == "zh"
    assert cfg.target_lang == "en"
    assert cfg.context == "An interview"
    assert cfg.proper_nouns == ("Kimi",)
    assert cfg.glossary_map == {"月之暗面": "Moonshot AI"}


def test_config_rejects_unknown_keys(tmp_path):
    config = tmp_path / "dub.toml"
    config.write_text('video = "x.mp4"\nsource_lang = "zh"\ntarget_lang = "en"\ntargt_lang = "fr"')

    with pytest.raises(ValidationError, match="targt_lang"):
        load_toml(config)


def test_cli_values_override_file_and_create_final_config(tmp_path):
    config = tmp_path / "dub.toml"
    config.write_text('video = "file.mp4"\nsource_lang = "zh"\ntarget_lang = "en"')
    args = argparse.Namespace(
        video="cli.mp4",
        source_lang=None,
        target_lang="fr",
        llm_model=None,
        llm_base=None,
        llm_key=None,
        output=None,
        names="host, guest",
        workdir=None,
    )

    cfg = merge_cli(load_toml(config), args)

    assert cfg.video == "cli.mp4"
    assert cfg.target_lang == "fr"
    assert cfg.speaker_names == ("host", "guest")


def test_final_config_rejects_same_language(tmp_path):
    config = tmp_path / "dub.toml"
    config.write_text('video = "file.mp4"\nsource_lang = "en"\ntarget_lang = "en"')

    with pytest.raises(ValidationError, match="different"):
        merge_cli(load_toml(config), argparse.Namespace())


def test_translation_api_uses_moonshot_defaults(monkeypatch, make_job_config):
    monkeypatch.delenv("DUB_TRANSLATE_BASE_URL", raising=False)
    monkeypatch.delenv("DUB_TRANSLATE_MODEL", raising=False)
    monkeypatch.delenv("DUB_TRANSLATE_API_KEY", raising=False)

    settings = resolve_translation_api(make_job_config())

    assert settings.base_url == "https://api.moonshot.ai/v1"
    assert settings.model_name == "kimi-k3"
    assert settings.api_key == ""


def test_translation_environment_overrides_configured_base_and_model(monkeypatch, make_job_config):
    monkeypatch.setenv("DUB_TRANSLATE_BASE_URL", "https://environment.test/v1")
    monkeypatch.setenv("DUB_TRANSLATE_MODEL", "environment-model")

    settings = resolve_translation_api(make_job_config(llm_base="https://config.test/v1", llm_model="config-model"))

    assert settings.base_url == "https://environment.test/v1"
    assert settings.model_name == "environment-model"


def test_configured_translation_key_takes_precedence(monkeypatch, make_job_config):
    monkeypatch.setenv("DUB_TRANSLATE_API_KEY", "environment-key")

    settings = resolve_translation_api(make_job_config(llm_key="config-key"))

    assert settings.api_key == "config-key"


def test_openai_api_key_is_ignored_for_translation(monkeypatch, make_job_config):
    monkeypatch.delenv("DUB_TRANSLATE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leave-openai")

    settings = resolve_translation_api(make_job_config())

    assert settings.api_key == ""


def test_translation_settings_repr_redacts_api_key(monkeypatch, make_job_config):
    monkeypatch.setenv("DUB_TRANSLATE_API_KEY", "translation-secret")

    settings = resolve_translation_api(make_job_config())

    # the key must actually be resolved, not merely absent from the repr —
    # an empty-string fallback would satisfy the redaction check on its own
    assert settings.api_key == "translation-secret"
    assert "translation-secret" not in repr(settings)


@pytest.mark.parametrize(
    "config_text",
    [
        'video = "file.mp4"\nsource_lang = "xx"\ntarget_lang = "en"',
        'video = "file.mp4"\nsource_lang = "en"\ntarget_lang = "ar"',
    ],
)
def test_config_rejects_unsupported_pipeline_languages(tmp_path, config_text):
    config = tmp_path / "dub.toml"
    config.write_text(config_text)

    with pytest.raises(ValidationError):
        load_toml(config)
