# podcast_dub

Dubs a video podcast into another language using all open-weight models:

* caption-free speech timing
* automatic speaker separation (diarization)
* fluent LLM translation
* voice-cloned TTS per speaker (up to 4 speakers)
* the original audio kept underneath, deep-ducked

## Demo

| Original | English dub |
|:---:|:---:|
| [![Original Mandarin interview](https://img.youtube.com/vi/91fmhAnECVc/maxresdefault.jpg)](https://www.youtube.com/watch?v=91fmhAnECVc) | [![English dub of the same interview](https://img.youtube.com/vi/92BQg2oozBg/maxresdefault.jpg)](https://www.youtube.com/watch?v=92BQg2oozBg) |
| [Watch original](https://www.youtube.com/watch?v=91fmhAnECVc) | [Watch English dub](https://www.youtube.com/watch?v=92BQg2oozBg) |

## Dub a local video
Note: You need [Git](https://git-scm.com/install/), [uv](https://docs.astral.sh/uv/getting-started/installation/),
and [FFmpeg](https://ffmpeg.org/download.html) pre-installed.

```bash
git clone https://github.com/cezarc1/podcast_dub.git
cd podcast_dub
uv sync --locked --python 3.12
```

Then, from the repository root:

```bash
uv run podcast_dub "./interview.mp4" --from zh --to en
```

Translation defaults to Kimi K3 through Moonshot's OpenAI-compatible API.
`DUB_TRANSLATE_API_KEY` must already be set to the corresponding API key.

Run the command from the repository root: the ASR and diarization stages look
for their helper environments at `./.venv-asr` and `./.venv-nemo`, relative to
the current directory. Set `DUB_ASR_PYTHON` and `DUB_NEMO_PYTHON` to run from
elsewhere.

The complete pipeline runs and writes:

| Path | Contents |
|---|---|
| `./interview_en.mp4` | final dubbed video |
| `./interview_dubwork/dub_mix.wav` | complete soundtrack with the original audio ducked underneath |
| `./interview_dubwork/dub_voice.wav` | synthesized voices only |
| `./interview_dubwork/` | resumable stage artifacts, logs, and subtitles |

The first run downloads the model weights and can take a while. Local inputs
must be media files that FFmpeg can decode. HTTP(S) video URLs are also
accepted and downloaded into the workdir with `yt-dlp`.

## Use a job file

For a repeatable job, create `dub.toml`:

```toml
video = "./interview.mp4"
source_lang = "zh"
target_lang = "en"
output = "./interview_en.mp4"
context = "A technical interview about database performance."
proper_nouns = ["AcmeDB", "Nova"]
speaker_names = ["host", "guest"]
# Names are assigned from most to least detected speaking time.
```

Then run:

```bash
uv run podcast_dub --config dub.toml
```

See [dub.toml.example](dub.toml.example) for the configuration template. The
demo above has ready-to-run configs for a
[five-minute development run](jobs/ref_kimi_5min.toml) and the
[full showcase](jobs/ref_kimi_full.toml).

Rerunning the same job reuses matching stage artifacts. If the source file's
contents change without its filename changing, choose a fresh `--workdir` so
the extracted source audio cannot be reused. Version 0.1 does not yet include
the ASR/NeMo helper dependency locks in artifact provenance, so also use a
fresh workdir after changing either helper environment.

## Configure translation

Moonshot and Kimi K3 are the defaults. To use another OpenAI-compatible
service, such as OpenAI, OpenRouter, or Ollama, set the endpoint, model, and
key together:

```bash
export DUB_TRANSLATE_BASE_URL="https://provider.example/v1"
export DUB_TRANSLATE_MODEL="provider-model-id"
export DUB_TRANSLATE_API_KEY="<translation-api-key>"
```

`DUB_TRANSLATE_BASE_URL` is the API base URL; do not include
`/chat/completions`. The pipeline appends the operation path. A non-empty key
is still required when a local endpoint ignores authentication, so use the
placeholder value documented by that endpoint.

For a repeatable job, `llm_base` and `llm_model` can instead be set in
`dub.toml`; the environment variables override those values for one-off runs.
Keep the key in `DUB_TRANSLATE_API_KEY` rather than storing it in the job file.

The configured endpoint must speak the OpenAI-compatible protocol. A direct
Anthropic Messages API URL is not supported; use an OpenAI-compatible gateway
such as OpenRouter to run Claude models.

## Install

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg
(`ffmpeg` and `ffprobe`).

### macOS / Apple Silicon

```bash
uv sync --locked --python 3.12
```

The automatic stage plan is ASR on MPS, Sortformer diarization on CPU, and TTS
on MPS. FlashAttention is not used on macOS.

### Linux / NVIDIA CUDA

Confirm the NVIDIA driver works before installing:

```bash
nvidia-smi
uv sync --locked --python 3.12 --extra cuda
```

Linux currently resolves the project's pinned CUDA 13.0 PyTorch wheels.
`--extra cuda` additionally installs FlashAttention for the main TTS
environment; it is not GPU auto-detection.

### ASR helper

Diarization runs in the main environment: NeMo asks for `transformers~=4.57.0`,
which already admits the `4.57.3` that Qwen3-TTS pins, so `uv sync` installs
both together.

ASR is the one stage that cannot share it — Qwen3-ASR needs Transformers 5.x
while Qwen3-TTS pins 4.x. Create that single helper environment once:

```bash
uv venv --python 3.12 .venv-asr
uv pip install --python .venv-asr/bin/python --torch-backend=auto \
    'transformers==5.14.1' torch torchaudio accelerate librosa soundfile 'pydantic>=2.10,<3'

# Linux/NVIDIA only: the CUDA plan also uses FlashAttention in the ASR helper
uv pip install --python .venv-asr/bin/python --no-build-isolation 'flash-attn>=2.7.4'

export DUB_TRANSLATE_API_KEY="<translation-api-key>"
```

The default helper path is `.venv-asr/bin/python`; set `DUB_ASR_PYTHON` only
when using a different location. If no compatible FlashAttention wheel exists
for the CUDA ASR environment, building it requires a matching CUDA toolkit.

`DUB_NEMO_PYTHON` still works if you prefer to keep diarization in its own
environment: point it at that interpreter and the stage runs there instead.

## Pipeline

`video → probe (16 kHz audio) → ASR timing → diarization → clone refs →
translation → TTS → placement + mix → mp4`

| Stage | What runs | Notes |
|---|---|---|
| probe | ffmpeg | extracts mono 16 kHz audio |
| asr | Qwen3-ASR-1.7B + Qwen3-ForcedAligner | caption-free phrase + word timings (runs in `.venv-asr`, transformers 5.x) |
| diarize | NVIDIA Sortformer (NeMo, in `.venv-nemo`) | up to four speakers; splits phrases at sustained speaker handoffs to reduce cross-speaker TTS |
| refs | auto-mined from diarization | ~60 s clean solo audio per speaker, full timeline |
| translate | DSPy + kimi-k3 (Moonshot default) | repairs spoken-ASR noise, uses preceding source conversation turns, emits TTS-ready speech, and logs every batch to `<workdir>/translations.jsonl` |
| tts | Qwen3-TTS-12Hz-1.7B (local) | stage-specific CUDA/MPS/CPU selection, x_vector-only voice cloning, measurement-verified DSPy.Refine rewrite loop |
| place | ffmpeg + `fit.py` + verification | anchored chaining, hard anti-drift windows, capped speedups only, sidechain-ducked original bed; publishes the mp4 only after coverage/dead-air verification passes |

The CLI resolves and prints a stage-specific plan before doing model work:

| Available accelerator | ASR `auto` | Diarization `auto` | TTS `auto` |
|---|---|---|---|
| NVIDIA CUDA | CUDA | CUDA | CUDA |
| Apple MPS | MPS | CPU | MPS |
| none | CPU | CPU | CPU |

MPS and CPU use eager attention. CUDA ASR and TTS select FlashAttention 2 when
it is installed and importable, otherwise SDPA; Sortformer does not consume
that attention setting. Set `asr_device`, `diarize_device`, and `tts_device`
explicitly in TOML when a particular accelerator is required.

These are fail-fast routing rules, not a promise that every driver and package
combination works. Confirm `nvidia-smi` before a CUDA job and run the final
media verifier on every completed dub.

The pinned TTS model supports `zh`, `en`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`,
`es`, and `it` as target languages. Reference mining requires at least 30
seconds of clean solo speech per detected speaker, so very short or heavily
overlapping clips fail fast instead of producing a weak voice clone.

Placement runs the coverage and dead-air gates automatically. They can also be
run directly when inspecting an existing workdir:

```bash
uv run python -m podcast_dub.stages.verify <workdir>
```

## Repo layout

* `src/podcast_dub/` — the installable package: `cli.py` (console entry point
  `podcast_dub`), `types.py` (Pydantic contracts), `artifacts.py`
  (versioned/provenance-aware I/O), `config.py`, `translate.py` (DSPy
  programs), and the pipeline stages in `stages/`
  (`asr`, `diarize`, `refs`, `translate`, `tts`, `place`, `verify`)
* `src/podcast_dub/fit.py`, `device_utils.py`, `audio_utils.py` — timing-fit engine, stage-specific device planning, audio helpers
* `src/podcast_dub/tools/` — typed placement simulation and HTML turn-review
  tools that consume a pipeline workdir
* `tests/` — unit, regression, integration-audit, and Hypothesis property tests

## Develop

```bash
uv run pytest                  # tests (testpaths configured in pyproject.toml)
uv run ruff format --check .   # formatting
uv run ruff check .            # lint (configured ruleset)
uv run ty check .              # types
uv build                       # source + wheel distributions

# audit the typed artifacts and media in a completed workdir
uv run python tests/generic_pipeline_test.py <workdir>
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). Model weights are governed by their own licenses
(Qwen3-ASR/TTS: Apache-2.0; NVIDIA Sortformer: NVIDIA Open Model License).
