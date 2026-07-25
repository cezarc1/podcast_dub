"""CLI robustness: partial-download filtering, up-front stage validation, config errors."""

import sys

import pytest

from podcast_dub import cli
from podcast_dub.artifacts import ArtifactProvenance, stable_digest, write_artifact_atomic
from podcast_dub.config import JobConfig
from podcast_dub.pipeline_artifacts import SPEAKER_PHRASES
from podcast_dub.stages import translate as translate_stage
from podcast_dub.types import SpeakerPhrase


def test_complete_inputs_skips_partial_downloads_and_sorts(tmp_path) -> None:
    """yt-dlp scratch files must never be adopted as the job input."""
    for name in (
        "input.webm",
        "input.mp4",
        "input.mp4.part",
        "input.f137.mp4.ytdl",
        "input.mp4.part-Frag3",
        "input.mkv.temp",
        "unrelated.mp4",
    ):
        (tmp_path / name).write_bytes(b"x")

    assert cli._complete_inputs(str(tmp_path)) == ["input.mp4", "input.webm"]


def test_complete_inputs_returns_empty_when_only_partials_exist(tmp_path) -> None:
    (tmp_path / "input.mp4.part").write_bytes(b"x")
    (tmp_path / "input.f137.mp4.ytdl").write_bytes(b"x")

    assert cli._complete_inputs(str(tmp_path)) == []


def test_unknown_stage_runs_no_stage_function(tmp_path, monkeypatch) -> None:
    """`--stages asr,bogus` must fail before ASR burns a full run."""
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    called: list[str] = []

    def _recorder(name: str):
        def _run(cfg: JobConfig) -> str:
            called.append(name)
            return name

        return _run

    monkeypatch.setattr(cli, "STAGES", {name: _recorder(name) for name in cli.STAGES})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "podcast_dub",
            str(video),
            "--from",
            "zh",
            "--to",
            "en",
            "--workdir",
            str(tmp_path),
            "--stages",
            "asr,bogus",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == "unknown stage: bogus"
    assert called == []


def test_valid_stages_still_dispatch(tmp_path, monkeypatch) -> None:
    """The up-front check must not reject legitimate selections."""
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    called: list[str] = []

    def _recorder(name: str):
        def _run(cfg: JobConfig) -> str:
            called.append(name)
            return name

        return _run

    monkeypatch.setattr(cli, "STAGES", {name: _recorder(name) for name in cli.STAGES})
    monkeypatch.setattr(cli, "probe", lambda cfg: 12.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "podcast_dub",
            str(video),
            "--from",
            "zh",
            "--to",
            "en",
            "--workdir",
            str(tmp_path),
            "--stages",
            "probe,asr,translate",
        ],
    )

    cli.main()

    assert called == ["asr", "translate"]


def test_malformed_toml_exits_two(tmp_path, monkeypatch) -> None:
    """A broken job file reports a configuration error, not a TOMLDecodeError traceback."""
    config = tmp_path / "dub.toml"
    config.write_text('from = "zh"\nthis is not valid toml\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["podcast_dub", "--config", str(config)])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2


def test_missing_config_file_exits_two(tmp_path, monkeypatch) -> None:
    """A typo'd --config path is the likeliest config error; it must not traceback."""
    monkeypatch.setattr(sys, "argv", ["podcast_dub", "--config", str(tmp_path / "absent.toml")])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2


def test_translate_stage_raises_runtime_error_without_api_key(tmp_path, monkeypatch, make_job_config) -> None:
    """A missing key is a library-level error, not a process exit."""
    monkeypatch.delenv("DUB_TRANSLATE_API_KEY", raising=False)
    source_provenance = ArtifactProvenance(
        config_digest=stable_digest({"source": "test"}),
        parameters_digest=stable_digest({}),
    )
    write_artifact_atomic(
        tmp_path / "phrases_spk.json",
        SPEAKER_PHRASES,
        source_provenance,
        (SpeakerPhrase(start=0.0, end=1.0, speaker="S0", text="first"),),
    )
    cfg = make_job_config(llm_key="")

    with pytest.raises(RuntimeError, match=r"translate: no API key \(set DUB_TRANSLATE_API_KEY\)"):
        translate_stage.run_translate(cfg)
