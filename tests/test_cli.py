import subprocess
import sys


def test_cli_imports_in_main_environment() -> None:
    """The CLI must load before helper-only ASR dependencies are available."""
    from podcast_dub import cli

    assert callable(cli.main)


def test_cli_help_does_not_import_tts_runtime() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "podcast_dub", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "SoX could not be found" not in result.stdout + result.stderr
    assert "flash-attn is not installed" not in result.stdout + result.stderr


def test_fetch_if_url_returns_config_with_cached_local_video(tmp_path, make_job_config) -> None:
    from podcast_dub.cli import fetch_if_url

    (tmp_path / "input.mp4").write_bytes(b"cached")
    cfg = make_job_config(video="https://example.test/video")

    fetched = fetch_if_url(cfg)

    assert fetched.video == str(tmp_path / "input.mp4")
    assert cfg.video == "https://example.test/video"
