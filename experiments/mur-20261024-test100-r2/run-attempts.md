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

## Attempt run-02 — PARTIAL (fixed100 not completed)

- Run commit: `5c91a8a178e2c39ba6aa50c3d116524eae8ecf6e` (clean tree; loader fix).
- Engineering identity check: **PASSED** again (77.19 s, 1306 tokens).
- Chunk 0: generation 2157.9 s, grading 133.8 s, 75/75 trajectories, 145790
  generation tokens and 9162 verifier tokens; complete.
- Chunk 1: generation 2472.4 s, grading 85.7 s, 75/75 trajectories, 170703
  generation tokens and 6608 verifier tokens; complete.
- Chunk 2: generation killed by the frozen combined-phase cap at 2941.2 s after
  46/75 trajectories (15 complete questions plus one CoT-only question); no
  grading, no restart. Its 179940 generated tokens are preserved in raw traces.
- Chunk 3: never started.
- Driver allocation: 7868.17 s (2.1856 GPU-hours); wall 7868.93 s.
- Conservative driver ledger: 633569 tokens because the killed chunk reserved its
  full 300000-token grant; actual trace-derived total is 513509 tokens including
  verifier.
- No OOM, NaN, leak, context limit, budget stop or unowned process was observed.
- Result: 50 complete paired questions only, exploratory. Point target not met
  (MUR accuracy 0.74 versus per-step 0.80); token savings 0.8320. This is not a
  fixed100 or MATH-500 result.
