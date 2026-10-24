# Implementation review — mur-20261024-prep-r5

Gate status: **BLOCKED**.

## What was implemented
- Independent three-method runner (`mur/run.py`) sharing prompt, stop, sampling, critic and ledger code.
- vLLM 0.8.5 V1 backend that reads raw sampled-token logprobs from the generating forward pass
  (inspected `vllm/v1/sample/sampler.py`: `raw_logprobs` precedes penalties/temperature), so no extra
  prompt-rescoring call is used for `m_t`.
- `mur.validate_forward` installs a read-only observer in the exact generating sampler, clones
  logits before transforms, computes float32 log-softmax independently, calls the original sampler
  once, and compares the sampled column with the client `Generation` logprob on exact token IDs.
  A temperature-only negative control must differ by more than 0.001 somewhere. Prior
  independent-prefill (r4) and cross-engine (r3) comparisons remain FAILED diagnostics;
  this observer establishes neither kind of numerical parity.
- Driver bookkeeping includes cleanup time in GPU allocation, fails closed without NVML/compute
  telemetry, refuses reused output directories, and marks unknown token reservations separately.
- Momentum detector with `U_t = alpha*U_(t-1)+(1-alpha)*m_accepted`, bias correction, and trigger
  `m_proposal > U_(t-1)/(1-alpha**(t-1)) - log(gamma)`; first MUR step is never scaled.
- GenPRM two-call analyze/judgment critic with an anchored initial `\boxed{Yes|No}` verdict token,
  deterministic first-candidate fallback and malformed logging.
- General-verifier evaluation after solutions are frozen, strict `Final Decision:` parsing and a
  versioned hash cache including model revision, prompt and sampling settings.
- Model manifests verify pinned SHA256 for weight, config and tokenizer files as well as the MATH-500
  and development data.
- One foreground driver (`scripts/stage_driver.py`) runs validation, smoke and verifier in fresh
  child processes, samples `nvidia-smi`, and enforces cumulative GPU/token/wall budgets.
- vLLM V1 memory fractions are 0.42 (policy), 0.62 (critic) and 0.40 (verifier).  vLLM's
  `determine_available_memory` subtracts non-torch allocations that include co-tenant GPU use, so
  the supervisor's 0.26 left no KV-cache blocks on the shared GPU; measured new usage stays below
  14 GiB and free memory stays above 4 GiB.

## Verified
- Source/test/dev manifest: True
- Same-forward probability-contract gate, all eight probes: False
  (validation_reference=same_forward_unmodified_logits,
  cross_engine_hf_parity=False,
  independent_prefill_parity=False,
  temperature_only_negative_control=True)
- Complete 20x3 smoke and verifier: False
- No budget/context-truncated trajectory: True
- Token-ledger closure: False
- Cumulative GPU/token/wall budget evidence: True
  (GPU 32.06s including cleanup, reported tokens 172, unknown reservations 0, wall 32.26s)
- At least one MUR triggered branch inspected or a marked engineering forced-trigger case: False

## Validation details
- Installed sampler source: `runtime/venv/lib/python3.10/site-packages/vllm/v1/sample/sampler.py`
  SHA256 `9d1c04c6f44adbaabb96388ca0766791351a6c84fa95672be70a1116498972b1`;
  raw logprobs computed before penalties/temperature:
  `True`.
- Direct-gate arithmetic limit 5e-05 on identical logits; observed max
  0.0, temperature-only max
  0.26924046874046326 (negative-control threshold
  0.001).
- r3 cross-engine HF parity and r4 independent-prefill parity both remain **FAILED** and are not passing results.
- Conservative preparation charges: R1 3600 s, R2 0 s, R3 280 s, R4 62.77 s, R5 <=3000 s;
  combined 6942.77 s
  (1.9285 GPU-hours) below the 2 GPU-hour ceiling.
  These are budget charges, distinct from the measured R5 allocation.

## Blocking condition
same-forward probability-contract gate is missing or failed

## Deviations / limitations
- Development questions only; no MATH-500 test item was generated or tuned on.
- The smoke is not a scientific success claim.
- Latency on a shared GPU is descriptive, not a controlled speedup.
- GPU memory/headroom is reported only when an `nvidia-smi` observation is actually available.

## Failures
[
  "same-forward probability-contract gate is missing or failed"
]

## Direct-gate attempt evidence (r5)
- Actual run commit: `e931a6d7e2cc235b4a6e26f60f1ebf5faa8c20c3` (pre-GPU
  implementation commit). One bounded GPU attempt, no retry.
- The eight frozen synthetic probes, seed, sampling parameters and 5e-5
  arithmetic bound were unchanged.
- Five probes completed and passed: `repeat` (15 tokens), `count` (29),
  `ones` (63), `sum` (15), `double` (21). Every returned token comparison had
  max absolute error `0.0` for raw-reference, sampler column and client
  `Generation` logprob; the temperature-only negative control reached
  `0.26924`, well above 0.001.
- Probe six (`fraction`) generated 29 tokens, then
  `mur.validate_forward.compare_returned` raised
  `ValueError: sampler and client token IDs/counts differ`. The gate stopped
  there; probes seven and eight were not run and no smoke or verifier was
  launched.
- Measured R5 allocation was 32.06 GPU seconds including cleanup, with 172
  generated tokens, peak new GPU usage 6,851 MiB and minimum free memory
  14,027 MiB.
- The saved evidence is consistent with a client-side stop/abort alignment
  issue rather than a probability arithmetic mismatch: validation probes were
  generated with `stop=["\nStep"]`, and the client output can omit a trailing
  sampler token when a stop string is detected. The exact cause was not
  re-run or proven in this task.
- No tolerance, reference, model, backend, sampling parameter, seed, prompt or
  split was changed after the failure. No additional GPU diagnosis was
  performed.
