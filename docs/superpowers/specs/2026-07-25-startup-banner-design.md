# Startup Banner Design

## Goal

Show a distinctive `podcast-dub` wordmark when a valid dubbing job begins, without adding runtime dependencies or
changing help and error output.

## Design

Add `src/podcast_dub/branding.py` with:

- a module-level constant containing the selected slanted ASCII wordmark;
- a `show_banner()` function that emits the complete wordmark as one `INFO` logging record.

Call `show_banner()` from `podcast_dub.cli.main()` after argument parsing, configuration loading, and job validation
succeed. Call it immediately before the existing job, workdir, output, and stage summary. This placement means
`--help`, argument-parser exits, invalid configuration, and invalid job settings do not display the banner.

The artwork will contain no ANSI color codes. Static terminal-safe ASCII keeps rendering deterministic in interactive
terminals, captured test output, and redirected logs. The implementation will use logging rather than `print()`, in
keeping with the package-wide print-free output contract.

## Testing

Add focused tests that verify:

- `show_banner()` emits the complete multiline wordmark at `INFO`;
- a valid job reaches the banner call;
- invalid job configuration exits before the banner call;
- the existing package-wide no-`print()` regression remains satisfied.

Run the targeted banner and CLI tests, Ruff, and the maintained type checker.

## Scope

This change adds only static startup branding and its tests. It does not add dynamic font generation, ANSI styling,
configuration flags, terminal-width adaptation, or banner output for help and error paths.
