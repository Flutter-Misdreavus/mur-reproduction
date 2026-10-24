# mur-20261024-prep-r5 — summary

Status: **BLOCKED** at the direct same-forward probability-contract gate.
The 20x3 development smoke and the general-verifier evaluation were **not
launched**. No MATH-500 test item was generated or tuned on. No scientific
accuracy or token-savings claim is made.

## What ran
- CPU contract suite: **44/44 tests passed**, including direct-observer
  arithmetic/negative-control contracts, strict gate-summary checks, and
  prior-stage conservative budget charges.
- `mur.prepare` verified the pinned 500-row MATH-500 file, the seven Hendrycks
  MATH training shards, and the policy/critic/verifier weight, config and
  tokenizer SHA256 values. It froze the 20 deterministic development question
  IDs after normalized-question deduplication against all 500 test questions.
- One bounded foreground GPU attempt on physical GPU 2, run commit
  `e931a6d7e2cc235b4a6e26f60f1ebf5faa8c20c3`.
- The driver ran `mur.validate_forward` only. It measured 32.06 GPU seconds
  including cleanup, 172 generated tokens and peak new GPU usage 6,851 MiB
  (minimum free memory 14,027 MiB). It did not launch smoke because the gate
  did not complete.

## Direct-gate result
`logprob-validation.json`: **passed=false**, `probe_count=5`.
Five of the eight frozen probes completed with **exactly zero** observed
arithmetic error against the unmodified logits and the returned client
`Generation` logprobs:

| probe | tokens | max_abs (limit 5e-5) | negative control max | result |
|---|---:|---:|---:|---|
| repeat | 15 | 0.0 | 0.26924 | pass |
| count | 29 | 0.0 | 0.13531 | pass |
| ones | 63 | 0.0 | 0.14057 | pass |
| sum | 15 | 0.0 | 0.04535 | pass |
| double | 21 | 0.0 | 0.14027 | pass |

Probe six (`fraction`) generated 29 tokens and then raised
`ValueError: sampler and client token IDs/counts differ` inside
`compare_returned`. The gate stopped at that point; probes seven
(`product`) and eight (`repeat_sequence`) were not run, so the required
eight complete probes, exact alignment on every probe, and the full
same-forward gate were not established.

The saved evidence is consistent with a client-side stop/abort alignment
issue, not a probability arithmetic mismatch: validation probes were generated
with the scientific `stop=["\nStep"]`, and the client output may omit a
trailing token that the sampler still observed when a stop string triggers an
abort. This was not re-run or proven in this task. The first five probes
demonstrate that the observer/API arithmetic contract itself is exact when the
token sequences align.

## Failures retained
1. Direct gate did not complete: five probes passed arithmetic, probe six
   aborted on exact token ID/count alignment, probes seven and eight were not
   run.
2. Because the gate did not pass, the 20x3 development smoke and verifier were
   not launched. There are no accuracy, token-savings or 1500-trajectory
   timing metrics; missing values are unknown, not zero.
3. The r3 HF cross-engine parity failure and the r4 independent-prefill
   failure are retained separately and are **not** relabelled.
4. `raw-smoke-traces.jsonl` is present at the run root but empty because no
   smoke was executed.

## Resource evidence
- R5 measured GPU allocation: 32.06 s (0.0089 GPU-hours), including cleanup;
  172 generated tokens; driver wall 32.26 s.
- Peak new GPU usage: 6,851 MiB (< 14,336 MiB cap); minimum free memory:
  14,027 MiB (> 4,096 MiB floor).
- Conservative prep-ceiling charges, distinct from measurements: R1 3,600 s,
  R2 0 s, R3 280 s, R4 62.77 s, R5 at most 3,000 s. The conservative sum is
  6,942.77 s (1.9285 GPU-hours), below the 2 GPU-hour combined preparation
  ceiling; the 12 GPU-hour project ceiling is unchanged.
- No smoke or evaluation tokens/GPU seconds were spent.

## Minimal decision for the supervisor
The failure is an alignment/compatibility error, not a numerical probability
mismatch. Do not launch the development smoke or MATH-500 test on this gate.
A minimal next run could remove the step stop string from the **synthetic
validation probes only** (or compare the observer sequence against the
client's stop-trimmed output explicitly), keep the eight frozen prompts,
seed, sampling parameters and 5e-5 bound unchanged, and rerun the gate once
under a new commit/run_id. No tolerance, reference, model, backend or
scientific protocol change is proposed. This task did not perform that rerun
because it had one bounded direct-gate attempt.

## Version control
Actual run commit: `e931a6d7e2cc235b4a6e26f60f1ebf5faa8c20c3`.
Reporting commit: the commit containing this sanitized summary (see
`report.json` for the final HEAD and push state). Repository is private;
branch `exp/mur-20261024`. Push mode is relay and no remote is configured on
the worker, so the reporting commit is **pending** relay by the local
supervisor.
