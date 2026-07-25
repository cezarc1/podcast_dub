"""Strict job configuration loaded from TOML and CLI overrides."""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from podcast_dub.types import DeviceChoice, StrictModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse

DEFAULT_TRANSLATE_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_TRANSLATE_MODEL = "kimi-k3"


class Language(StrEnum):
    """A supported language: its ISO code is the string value, with a display name attached.

    Membership is by lowercase code (``Language("zh")``), and because this is a
    ``StrEnum`` each member compares equal to its code (``Language.ZH == "zh"``).
    """

    ZH = "zh", "Chinese"
    EN = "en", "English"
    JA = "ja", "Japanese"
    KO = "ko", "Korean"
    ES = "es", "Spanish"
    FR = "fr", "French"
    DE = "de", "German"
    IT = "it", "Italian"
    PT = "pt", "Portuguese"
    RU = "ru", "Russian"
    AR = "ar", "Arabic"
    HI = "hi", "Hindi"

    display: str

    def __new__(cls, code: str, display: str) -> Language:
        member = str.__new__(cls, code)
        member._value_ = code
        member.display = display
        return member


class TtsLanguage(StrEnum):
    """A target language supported by the pinned Qwen3-TTS model."""

    ZH = "zh"
    EN = "en"
    JA = "ja"
    KO = "ko"
    DE = "de"
    FR = "fr"
    RU = "ru"
    PT = "pt"
    ES = "es"
    IT = "it"


def lang_name(code: str) -> str:
    try:
        return Language(code.lower()).display
    except ValueError:
        logger.warning("config: unrecognized language code %r; using it verbatim", code)
        return code


@dataclass(frozen=True, slots=True)
class TranslationAPISettings:
    """Resolved settings for an OpenAI-compatible translation endpoint."""

    base_url: str
    model_name: str
    api_key: str = field(repr=False)


class _ConfigModel(StrictModel):
    """StrictModel (closed schema, immutable, validated_copy) plus TOML alias support."""

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True)


class GlossaryEntry(_ConfigModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


def _glossary_entries(value: Any) -> Any:
    if value is None or isinstance(value, tuple):
        return value
    if isinstance(value, Mapping):
        return tuple({"source": str(source), "target": str(target)} for source, target in value.items())
    return value


class _JobConfigBase(_ConfigModel):
    """Path expansion + glossary normalization shared by the partial and final job models."""

    glossary: tuple[GlossaryEntry, ...] | None = None

    @field_validator("video", "output", "workdir", mode="before", check_fields=False)
    @classmethod
    def expand_paths(cls, value: Any) -> Any:
        return os.path.expanduser(value) if isinstance(value, str) else value

    @field_validator("glossary", mode="before")
    @classmethod
    def normalize_glossary(cls, value: Any) -> Any:
        return _glossary_entries(value)

    @property
    def glossary_map(self) -> dict[str, str]:
        return {entry.source: entry.target for entry in self.glossary or ()}


class JobConfigInput(_JobConfigBase):
    video: str | None = None
    source_lang: Language | None = Field(default=None, validation_alias=AliasChoices("source_lang", "from"))
    target_lang: TtsLanguage | None = Field(default=None, validation_alias=AliasChoices("target_lang", "to"))
    output: str | None = None
    context: str | None = None
    proper_nouns: tuple[str, ...] | None = None
    speaker_names: tuple[str, ...] | None = None
    workdir: str | None = None
    window_s: float | None = Field(default=None, ge=0)
    llm_model: str | None = None
    llm_base: str | None = None
    llm_key: str | None = Field(default=None, repr=False, exclude=True)
    asr_device: DeviceChoice | None = None
    diarize_device: DeviceChoice | None = None
    tts_device: DeviceChoice | None = None


class JobConfig(_JobConfigBase):
    video: str = Field(min_length=1)
    source_lang: Language
    target_lang: TtsLanguage
    output: str = ""
    context: str = ""
    proper_nouns: tuple[str, ...] = ()
    glossary: tuple[GlossaryEntry, ...] = ()
    speaker_names: tuple[str, ...] = ()
    workdir: str = ""
    window_s: float = Field(default=0.0, ge=0)
    llm_model: str = Field(default=DEFAULT_TRANSLATE_MODEL, min_length=1)
    llm_base: str = Field(default=DEFAULT_TRANSLATE_BASE_URL, min_length=1)
    llm_key: str = Field(default="", repr=False, exclude=True)
    asr_device: DeviceChoice = DeviceChoice.AUTO
    diarize_device: DeviceChoice = DeviceChoice.AUTO
    tts_device: DeviceChoice = DeviceChoice.AUTO

    @model_validator(mode="after")
    def validate_languages(self) -> JobConfig:
        if self.source_lang == self.target_lang:
            raise ValueError("source and target languages must be different")
        return self

    def resolved_output(self) -> str:
        if self.output:
            return self.output
        stem, _ = os.path.splitext(self.video)
        return f"{stem}_{self.target_lang}.mp4"

    def resolved_workdir(self) -> str:
        if self.workdir:
            return self.workdir
        stem, _ = os.path.splitext(os.path.basename(self.video))
        return os.path.join(os.path.dirname(os.path.abspath(self.video)), f"{stem}_dubwork")

    def resolved_audio(self) -> str:
        stem, _ = os.path.splitext(os.path.basename(self.video))
        return os.path.join(self.resolved_workdir(), f"{stem}.wav")

    def validation_problems(self) -> list[str]:
        if self.video.startswith(("http://", "https://")) or os.path.exists(self.video):
            return []
        return [f"input video not found: {self.video!r}"]

    def provenance_config(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"llm_key"})


def resolve_translation_api(cfg: JobConfig) -> TranslationAPISettings:
    """Resolve per-run translation settings without consulting unrelated provider variables."""
    return TranslationAPISettings(
        base_url=os.environ.get("DUB_TRANSLATE_BASE_URL", cfg.llm_base),
        model_name=os.environ.get("DUB_TRANSLATE_MODEL", cfg.llm_model),
        api_key=cfg.llm_key or os.environ.get("DUB_TRANSLATE_API_KEY", ""),
    )


def load_toml(path: str | os.PathLike[str]) -> JobConfigInput:
    with open(path, "rb") as config_file:
        return JobConfigInput.model_validate(tomllib.load(config_file))


def merge_cli(cfg: JobConfigInput, args: argparse.Namespace) -> JobConfig:
    """Apply CLI precedence, then construct the fully validated job."""
    values = cfg.model_dump(exclude_none=True)
    if cfg.llm_key is not None:
        # Field(exclude=True) keeps the secret out of repr/dumps/provenance, so
        # carry it explicitly across the partial -> final config boundary.
        values["llm_key"] = cfg.llm_key
    scalar_overrides = {
        "video": getattr(args, "video", None),
        "source_lang": getattr(args, "source_lang", None),
        "target_lang": getattr(args, "target_lang", None),
        "output": getattr(args, "output", None),
        "workdir": getattr(args, "workdir", None),
    }
    values.update({key: value for key, value in scalar_overrides.items() if value is not None})
    if names := getattr(args, "names", None):
        values["speaker_names"] = tuple(name.strip() for name in names.split(",") if name.strip())
    return JobConfig.model_validate(values)
