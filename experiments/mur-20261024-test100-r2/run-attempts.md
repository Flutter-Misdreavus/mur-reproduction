# Run attempts — mur-20261024-test100-r2

## Attempt run-01 — FAILED before any test generation

- Run commit: `2ed1d21ef298dee73ec066a662054762a176f574` (clean tree).
- Engineering identity check: **PASSED** on dev indices 0 and 3; 4 trajectories,
  1306 generated tokens, 77.04 GPU allocation seconds, no genuine MUR trigger,
  first proposals and all common pre-trigger steps matched. Prefix caching off.
- Chunk 0 generation: exited 1 after 1.23 s with **zero trajectories**.
  `mur.run.load_questions` passed `runtime/data` (`data_root()`) to
  `testdata.load_test_generation_file`, which appends `/data` again, so the loader
  looked for `runtime/data/data/test100.jsonl`. It failed before loading a policy
  or generating any test token.
- The driver conservatively charged the full 300000-token chunk grant because
  `run-status.json` was absent; actual test-chunk generation was 0 tokens.
  Total attempt charge: 78.27 GPU allocation seconds, 1306 reported tokens plus
  the conservative 300000 reservation.
- No test question was generated, graded, replaced or selected on outcome.
- Raw failure evidence (stage-driver, resource usage, GPU probe, engineering
  traces and logs) is preserved in the ignored run directory under
  `attempts/run-01-chunk0-loader-failure/`.

## Repair

Minimal implementation fix: `load_questions(split="test")` now resolves the runtime
root once and passes it to both `load_source_rows` and
`load_test_generation_file`. A regression test builds a synthetic frozen split in a
temporary runtime root and asserts that `load_questions` returns the same 100 IDs.
No manifest, question, split, seed, prompt, sampling parameter, token limit,
verifier, metric or budget was changed. The rerun uses a new commit, not the failed
run commit.
