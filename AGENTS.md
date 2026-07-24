# Repository Guidelines

## Project Structure & Module Organization

The installable Python package lives in `src/podcast_dub/`. `cli.py` orchestrates the pipeline; individual stages are
under `src/podcast_dub/stages/` (`asr`, `diarize`, `refs`, `translate`, `tts`, `place`, and `verify`). Shared timing,
audio, configuration, and device helpers remain at the package root. Developer-only simulation and review utilities
belong in `src/podcast_dub/tools/`.

Tests live in `tests/`; keep fixtures and unit tests near the behavior they exercise. Example job configuration belongs
in `jobs/`, while `dub.toml.example` documents user-facing settings. Do not commit generated media or
`<video_stem>_dubwork/` artifacts.

## Build, Test, and Development Commands

- `uv sync` creates the Python 3.12 environment and installs runtime and development dependencies.
- `uv sync --extra cuda` also installs Linux/NVIDIA-only extras (currently `flash-attn` for TTS).
- `uv run pytest` runs the configured pytest suite, including Hypothesis property tests.
- `uv run ruff check .` checks imports, undefined names, and selected Python errors.
- `uv run ty check .` type-checks the maintained package and test surface.
- `uv build` builds source and wheel distributions through Hatchling.
- `uv run podcast_dub --config jobs/my_podcast.toml` runs a configured dubbing job.

ASR and diarization use the separate helper environments described in `README.md`. ASR has a Transformers conflict
with the main TTS environment; NeMo is isolated for reproducibility and dependency stability.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.12 syntax, type hints on public interfaces, and a 120-character maximum line length.
Follow Ruff’s configured `E4`, `E7`, `E9`, and `F` rules. Name modules, functions, and variables `snake_case`, classes
`PascalCase`, and constants `UPPER_SNAKE_CASE`. Keep stage orchestration thin and put stage-specific work in `stages/`.

## Testing Guidelines

Use pytest names such as `tests/test_fit.py` and `test_duration_scales`. Add Hypothesis tests for timing invariants and
deterministic unit tests for regressions; avoid model downloads or live API calls in the normal suite. The standalone
integration harness runs with `uv run python tests/generic_pipeline_test.py`. For completed media, run
`uv run python -m podcast_dub.stages.verify <workdir>` to enforce coverage and dead-air gates.

## Commit & Pull Request Guidelines

Use concise, imperative subjects, optionally scoped with a component prefix: `tts: tighten short turns` or
`README: correct helper-venv commands`. Keep each commit focused and explain measured behavior in the body when relevant.
Pull requests should summarize the pipeline impact, list validation commands, identify configuration or environment
changes, and link related issues. For audio or placement changes, include sample artifacts or before/after metrics and
listening notes; never include API keys, downloaded model weights, or private source media.
