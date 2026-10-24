# Report summary — mur-20261024-test100-r2

Status: **FAILED / PARTIAL** (engineering gate passed; fixed100 test incomplete).

## What actually ran

- Actual model-run commit: `5c91a8a178e2c39ba6aa50c3d116524eae8ecf6e`, clean tree,
  pinned physical GPU index 2, prefix caching disabled.
- Engineering identity check passed on dev indices 0 and 3 (4 trajectories,
  1,306 tokens, 77.19 s, no genuine MUR trigger).
- Chunks 0 and 1 completed generation and grading: 150 rows, 50 questions x3
  methods, no errors/truncations/context limits/budget stops/fallbacks.
- Chunk 2 was killed by the frozen 2,940 s phase cap after 46/75 trajectories
  (15 complete questions plus one CoT-only question). It was not graded and not
  restarted. Chunk 3 never started.
- No test question was replaced or selected using outcomes; no parameter tuning.

## Partial 50-question result (exploratory only)

| Method | Correct /50 | Accuracy | Mean policy+critic tokens |
|---|---:|---:|---:|
| CoT | 37 | 0.74 | 563.60 |
| Per-step Guided Search | 40 | 0.80 | 4936.96 |
| MUR | 37 | 0.74 | 829.30 |

- Token savings vs per-step: **0.8320** (paired 95% CI 0.7626 to 0.8844).
- Accuracy delta vs per-step: **−0.0600** (paired 95% CI −0.1800 to +0.0600).
- Point target **not met** because accuracy loss exceeds 2 pp; non-inferiority
  not supported. Bootstrap: 10,000 resamples, seed 20261024.
- Trigger rates: per-step 140/140 eligible steps; MUR 7/95 eligible non-first
  steps (7/145 all-step).
- Nine method-discordant questions among the 50 have a seeded, method-blinded
  audit; labels remain `pending_supervisor`. No verifier label was changed.

## Resources

- Driver GPU allocation: 7,868.17 s = 2.1856 GPU-hours; wall 7,868.93 s.
- Actual generated tokens: 513,509 (497,739 policy/critic, 15,770 verifier).
- Conservative driver ledger: 633,569 because the killed chunk reserved its full
  300,000-token grant.
- Peak new GPU memory 12,109 MiB; minimum free 8,770 MiB; no OOM/NaN.
- Disk remained above the 25 GiB floor.

## Failures and deviations

- First model attempt (`2ed1d21`) stopped in 1.23 s before any test trajectory on
  a test-split loader path bug; evidence preserved under `attempts/`.
- Authorization gate accepted the frozen r2 task only after a minimal wording
  fix; the gate was not weakened.
- Post-run CPU aggregation referenced nonexistent verifier summary keys; fixed
  without changing observed metrics.
- Partial result, not a fixed100 or MATH-500 claim.

## Recommendation to supervisor

Review the 50/100 partial result and the chunk-2 cap failure. Decide whether a
future task should pre-register a larger per-chunk allocation (or smaller chunk
size) for the harder remaining questions. Do not resume/restart this failed
matrix automatically, do not treat the 50-question subset as the fixed100 result,
and do not launch a full500 run from this evidence.
