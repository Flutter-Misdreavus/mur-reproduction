# Implementation review — mur-20261024-prep-r4

Gate status: **BLOCKED**.

## What was implemented
- Independent three-method runner (`mur/run.py`) sharing prompt, stop, sampling, critic and ledger code.
- vLLM 0.8.5 V1 backend that reads raw sampled-token logprobs from the generating forward pass
  (inspected `vllm/v1/sample/sampler.py`: `raw_logprobs` precedes penalties/temperature), so no extra
  prompt-rescoring call is used for `m_t`.
- `mur.validate_signal` re-scores the exact saved prompt/output ids with an identity-sampling
  teacher-forced prefill in a fresh pinned BF16 vLLM V1 process. The r3 HF cross-engine comparison
  remains a separate FAILED diagnostic; `cross_engine_hf_parity=false`.
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
- Same-engine raw signal gate, all eight probes: False
  (validation_reference=same_engine_identity_sampling_teacher_forced_prefill,
  cross_engine_hf_parity=False)
- Complete 20x3 smoke and verifier: False
- No budget/context-truncated trajectory: True
- Token-ledger closure: False
- Cumulative GPU/token/wall budget evidence: True
  (GPU 62.77s including cleanup, reported tokens 243, unknown reservations 0, wall 63.21s)
- At least one MUR triggered branch inspected or a marked engineering forced-trigger case: False

## Validation details
- Installed sampler source: `runtime/venv/lib/python3.10/site-packages/vllm/v1/sample/sampler.py`
  SHA256 `9d1c04c6f44adbaabb96388ca0766791351a6c84fa95672be70a1116498972b1`;
  raw logprobs computed before penalties/temperature:
  `True`.
- r3 cross-engine HF parity remains **FAILED** and is not a passing result.
- r3 prior allocation reported separately: 280 conservative GPU seconds and 300 conservative
  diagnostic tokens (excluded from the r4 budget totals).

## Blocking condition
same-engine raw generation-logprob signal gate is missing or failed

## Deviations / limitations
- Development questions only; no MATH-500 test item was generated or tuned on.
- The smoke is not a scientific success claim.
- Latency on a shared GPU is descriptive, not a controlled speedup.
- GPU memory/headroom is reported only when an `nvidia-smi` observation is actually available.

## Failures
[
  "same-engine raw generation-logprob signal gate is missing or failed"
]
