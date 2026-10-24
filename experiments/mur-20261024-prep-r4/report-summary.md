# mur-20261024-prep-r4 — summary

Status: **BLOCKED** at the amended same-engine raw-signal gate. The 20x3
development smoke and the general-verifier evaluation were **not** launched.
No MATH-500 test item was generated or tuned on. No scientific accuracy or
token-savings claim is made.

## What ran
- CPU contract suite: 38/38 tests passed, including new tests for the eight-probe
  signal comparison, sampler-source ordering evidence, stale-pass rejection,
  reused-output rejection, and actual-versus-reserved token accounting.
- `mur.prepare` verified the pinned 500-row MATH-500 file, the seven Hendrycks MATH
  training shards, and the policy/critic/verifier weight, config and tokenizer
  SHA256 values. It froze 20 deterministic development question IDs after
  normalized-question deduplication against all 500 test questions.
- One foreground GPU attempt on physical GPU 2, run commit
  `d2e0408165055f975504b1c5b509c379dad3b7c7`.
- The driver ran `mur.validate_signal` only. It recorded 62.77 s of GPU allocation
  including cleanup and 243 generated tokens. It did not launch smoke because the
  signal gate failed.

## Signal gate result
`logprob-validation.json`: **passed=false**. All eight frozen probes were
nonempty, repeated-token coverage was present, and the exact saved prompt/output
token IDs were re-scored in a fresh pinned BF16 vLLM V1 process with
`prompt_logprobs=0`, identity sampling and prefix caching disabled. The generation
and reference engines both used the default Flash Attention backend. The unchanged
r3 tolerances were applied to every probe; four probes passed and four failed.

| probe | tokens | max_abs (limit 0.15) | mean_abs (limit 0.02) | abs_mean (limit 0.005) | result |
|---|---:|---:|---:|---:|---|
| repeat | 15 | 0.42634 | 0.03007 | 0.02678 | FAIL |
| count | 29 | 0.08731 | 0.00301 | 0.00301 | pass |
| ones | 63 | 0.06981 | 0.00187 | 0.00121 | pass |
| sum | 15 | 0.01580 | 0.00105 | 0.00105 | pass |
| double | 21 | 0.07361 | 0.00443 | 0.00264 | pass |
| fraction | 29 | 0.26273 | 0.01465 | 0.01224 | FAIL |
| product | 50 | 0.16668 | 0.00654 | 0.00582 | FAIL |
| repeat_sequence | 13 | 0.05440 | 0.01388 | 0.01047 | FAIL |

Most token positions matched exactly; the failures are concentrated at a few
low-margin positions, including the first output token of `fraction`, an interior
position of `product`, the final EOS position of `repeat`, and several positions of
`repeat_sequence`. The generation values reproduce the r3 vLLM values (for example,
`repeat` mean NLL 0.0383395 in both runs), while the fresh-prefill reference gives
0.0115611 for that probe. This is evidence that a fresh prefill and the generating
decode path are not numerically interchangeable at every position under the default
backend, even with the same pinned weights and exact token IDs. It is not proof of a
sampling-transform mismatch. The original HF cross-engine failure remains a
separate FAILED diagnostic, and `cross_engine_hf_parity=false`.

Installed sampler source evidence: `runtime/venv/lib/python3.10/site-packages/vllm/v1/sample/sampler.py`,
SHA256 `9d1c04c6f44adbaabb96388ca0766791351a6c84fa95672be70a1116498972b1`.
The inspected V1 source computes `raw_logprobs` before penalties and before
`sample` applies temperature (lines 36, 47, 49 and 108). The reference probe tokens
(8) were counted, so the total is 235 generation tokens + 8 probe tokens = 243.

No tolerance was changed, no sampler/attention variant was tried, and no further
GPU diagnosis or repetition was performed after the failed probe, as required.

## Resource evidence
- R4 GPU allocation: 62.77 s (0.0174 GPU-hours), including cleanup; 243 generated
  tokens (235 generation + 8 reference probe tokens).
- Measured peak new GPU usage: 6,849 MiB (< 14,336 MiB cap); minimum free memory
  observed: 14,029 MiB (> 4,096 MiB floor).
- R4 wall time at driver completion: 63.21 s.
- R3 prior allocation reported separately: 280 conservative GPU seconds and 300
  conservative diagnostic tokens. Combined conservative preparation total:
  342.77 GPU seconds (0.0952 GPU-hours) and 543 tokens, below the 2 GPU-hour
  combined-preparation ceiling and the 12 GPU-hour project ceiling.
- No smoke/evaluation tokens or GPU seconds were spent.

## Failures retained
1. Same-engine signal gate failed on four of eight probes, as tabulated above.
2. Because the signal gate failed, the 20x3 development smoke and verifier were
   not launched. There are no accuracy, token-savings, or 1500-trajectory timing
   metrics; missing values are unknown, not zero.
3. The r3 HF cross-engine parity failure is retained and not overridden.

## Minimal decision for the supervisor
Do not launch the development smoke or MATH-500 test. The supervisor must decide
whether the gate's reference should match the generation path more exactly (for
example, a same-path prefix-cache/decode reference) or whether a documented
kernel/backend reference change is acceptable. This task did not make that change,
did not relax the tolerances and did not run additional GPU diagnostics. Any
approved change needs a new run_id and a fresh commit before the next GPU attempt.

## Version control
Actual run commit: `d2e0408165055f975504b1c5b509c379dad3b7c7`.
Reporting commit: the commit containing this sanitized summary (see `report.json`
for the final HEAD and push state). Repository is private; branch
`exp/mur-20261024`. Push mode is relay and no remote is configured on the worker,
so the reporting commit is **pending** relay by the local supervisor.
