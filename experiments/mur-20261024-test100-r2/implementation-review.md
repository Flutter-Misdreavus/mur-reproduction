# Implementation review — mur-20261024-test100-r2

Supervisor note: see [review.md](review.md) for completed answer review, both-attempt costs, and reporting corrections. Original automatic scores are unchanged.

Gate status: **FAILED / PARTIAL**. Engineering identity passed; two of four frozen
test chunks completed. No complete fixed100 result exists and no full MATH-500
claim is made.

## Frozen implementation

The task reused the committed fixed100 driver (`scripts/test_stage_driver.py`) and
the runner/evaluator/aggregator from the accepted r6 implementation. Scientific
settings were unchanged: Qwen3-1.7B BF16 policy, GenPRM-1.5B critic,
TIGER-Lab/general-verifier, seed 42, alpha=0.9, gamma=0.9, four candidates,
temperature 0.6, top_p 0.8, top_k 20, presence penalty 1.5, step/critic token
limits and the frozen first100 manifest. Prefix caching was disabled uniformly.
GPU index 2 was pinned; no other GPU was selected.

The frozen split was verified from all 500 source rows and copied byte-identically
to the run root. Gold answers were stripped before generation and joined only by
the separate grader after each chunk's policy/critic processes exited.

## Engineering identity check

PASSED on development indices 0 and 3 with one policy instance and identical
seeds: 4 trajectories, 1,306 generated tokens, 77.19 GPU allocation seconds,
no genuine MUR trigger. First proposals matched in text and token IDs and all
common pre-trigger accepted steps matched. Raw trace SHA256:
`72ae9a1657f6faa4cb2ebc2d6cde470b253cd52a9354bcaf574a329b5f7cbfd3`.

## Code changes made during this task (no scientific changes)

1. **Authorization wording (pre-run, `723b477`).** The gate required the literal
   token `fixed100`; the frozen r2 controls use “identical committed100-question
   manifest” plus “exactly100 questions x3 methods in four25-question chunks”.
   The gate now accepts either exact wording and still requires a ready task, the
   `ready task` control, a private repository and an accepted parent review.
   A regression test covers acceptance and rejection.
2. **Test-split loader path (before the second attempt, `5c91a8a`).** The first
   attempt exited in 1.23 s with zero test trajectories because
   `load_questions` passed `runtime/data` to a loader that appends `/data`.
   It now resolves the runtime root once. A regression test builds a synthetic
   frozen split and asserts the loader returns the same 100 IDs.
3. **Post-run verifier accounting (results commit).** The unexercised
   four-chunk aggregator referenced nonexistent `verifier_calls` and
   `verifier_tokens_total` keys. The verifier summary now separates actual
   cache-deduplicated verifier tokens/calls from per-answer attributed tokens,
   with a regression test. This did not change any observed metric.

All 60 remote CPU tests pass after the fixes. The public audit reports zero
findings over tracked history.

## Model-run provenance and outcome

Actual model-run commit: `5c91a8a178e2c39ba6aa50c3d116524eae8ecf6e` (clean tree).
Chunks 0 and 1 completed generation and grading: 150 graded rows, no errors,
no context limits, no budget stops, no truncations, no fallbacks; all three
methods have exactly 50 questions.

Chunk 2 generation was stopped by the frozen combined phase cap at 2,941.2 s
after 46/75 trajectories (15 complete questions plus one CoT-only question). It
was not graded and was not restarted. Chunk 3 never started. The 46 partial
trajectories are preserved raw and are not included in the 50-question metrics.

| Method | Correct /50 | Accuracy | Mean policy+critic tokens |
|---|---:|---:|---:|
| CoT | 37 | 0.74 | 563.60 |
| Per-step Guided Search | 40 | 0.80 | 4936.96 |
| MUR | 37 | 0.74 | 829.30 |

Partial 50-question paired result: token savings 0.8320 (95% CI 0.7626–0.8844),
accuracy delta −0.0600 (95% CI −0.1800 to +0.0600). The point target is not met
because the accuracy guardrail fails; non-inferiority is not supported. Bootstrap
used 10,000 resamples, seed 20261024. CoT remains reported and is cheaper than
MUR but equally accurate here.

Auxiliary trigger rates: per-step 140/140 eligible steps; MUR 7/95 eligible
non-first steps (7/145 over all steps). Nine method-discordant questions among the
50 complete questions have a seeded, method-blinded audit file; labels remain
`pending_supervisor` and no verifier label was changed.

## Resources and artifacts

Driver allocation: 7,868.17 s (2.1856 GPU-hours), wall 7,868.93 s. Known saved token lower bound for this driver attempt: 513,509 (497,739 policy/critic + 15,770 verifier).
The driver's conservative ledger is 633,569 because the ungraded killed chunk
reserved its full 300,000-token grant. Peak new GPU memory 12,109 MiB; minimum
free 8,770 MiB; no OOM or NaN. Disk stayed above the 25 GiB floor.

Raw traces, per-chunk evidence, failed-attempt records and the blind audit stay in
the ignored run directory. The tracked `experiments/` copy contains only the
sanitized environment, source manifest, engineering summary, numeric metrics,
partial resource summary, blind-audit summary and this review.

## Limitations

- 50/100 complete questions; not a fixed100 result and not MATH-500 reproduction.
- The 50-question subset is the first two frozen chunks, not outcome-selected.
- Chunk 2 was cut by the per-chunk allocation, so its harder questions are absent.
- API-returned tokens are a compute proxy, not exact GPU FLOPs.
- One seed; shared-GPU latency is descriptive.
- The blind audit is pending supervisor review; no audited labels were invented.
