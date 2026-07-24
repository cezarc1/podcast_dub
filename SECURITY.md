# Security policy

`podcast_dub` 0.1 is an alpha, local command-line pipeline. It is not designed
as a multi-user service or for processing untrusted model repositories.

## Trust boundaries

- Keep the virtual environments, Hugging Face cache, and pipeline workdirs
  writable only by the user running the job.
- Use only the model identifiers and immutable revisions shipped with this
  release. Do not change the code to load an untrusted model, checkpoint, or
  remote kernel.
- Put translation API keys in `DUB_TRANSLATE_API_KEY`. Do not commit keys in a
  TOML file, shell script, log, issue, or workdir.
- Treat input media as untrusted data and keep FFmpeg and `yt-dlp` updated.

## Known dependency constraints

The current `qwen-tts` release requires `transformers==4.57.3`. That
Transformers version has published advisories involving malicious model
configuration and checkpoint inputs. This project does not expose a
user-selectable model ID, pins the model revisions it loads, supplies the
attention implementation explicitly, and does not install the optional
`kernels` package. These controls reduce the reachable surface but do not
replace an upstream fix. A future release will move to a fixed Transformers
version when Qwen TTS supports it.

DSPy currently brings in `diskcache==5.6.3`, whose cache format can execute
pickled data if an attacker can write to the cache directory. Do not share a
writable cache or workdir with untrusted users.

See the upstream advisories:

- [Transformers GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72)
- [DiskCache GHSA-w8v5-vhqr-4h9v](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v)

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do
not open a public issue for an undisclosed vulnerability.
