# Implementation review — mur-20261024-prep-r6

Gate status: **PASS**.

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
- Same-forward probability-contract gate, all eight probes: True
  (validation_reference=same_forward_unmodified_logits,
  cross_engine_hf_parity=False,
  independent_prefill_parity=False,
  temperature_only_negative_control=True)
- Complete 20x3 smoke and verifier: True
- No budget/context-truncated trajectory: True
- Token-ledger closure: True
- Cumulative GPU/token/wall budget evidence: True
  (GPU 2115.96s including cleanup, reported tokens 140863, unknown reservations 0, wall 2116.64s)
- At least one MUR triggered branch inspected or a marked engineering forced-trigger case: True

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
- Conservative preparation charges: R1 3600 s, R2 0 s, R3 280 s, R4 62.77 s,
  R5 32.06 s measured, R6 <=3000 s limit;
  combined 6974.83 s
  (1.9375 GPU-hours) below the 2 GPU-hour ceiling.
  These are budget charges, distinct from the measured R6 allocation.

## Blocking condition
None.

## Deviations / limitations
- Development questions only; no MATH-500 test item was generated or tuned on.
- The smoke is not a scientific success claim.
- Latency on a shared GPU is descriptive, not a controlled speedup.
- GPU memory/headroom is reported only when an `nvidia-smi` observation is actually available.

## Failures
[]
