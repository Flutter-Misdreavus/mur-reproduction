# Implementation review — mur-20261024-prep-r3

Gate status: **BLOCKED**.

## What was implemented
- Independent three-method runner (`mur/run.py`) sharing prompt, stop, sampling, critic and ledger code.
- vLLM 0.8.5 V1 backend that reads raw sampled-token logprobs from the generating forward pass
  (inspected `vllm/v1/sample/sampler.py`: `raw_logprobs` precedes penalties/temperature), so no extra
  prompt-rescoring call is used for `m_t`.
- `mur.validate_logprobs` compares those raw generation logprobs against an HF BF16 teacher-forced
  forward pass on the same token ids.
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
- Raw generation-logprob validation: False
- Complete 20x3 smoke and verifier: False
- No budget/context-truncated trajectory: False
- Token-ledger closure: False
- Cumulative GPU/token/wall budget evidence: True
  (GPU 255s, tokens 245, wall 1080s)
- At least one MUR triggered branch inspected or a marked engineering forced-trigger case: False

## Blocking condition
raw generation-logprob validation is missing or failed

## Deviations / limitations
- Development questions only; no MATH-500 test item was generated or tuned on.
- The smoke is not a scientific success claim.
- Latency on a shared GPU is descriptive, not a controlled speedup.
- GPU memory/headroom is reported only when an `nvidia-smi` observation is actually available.

## Failures
[
  "raw generation-logprob validation is missing or failed"
]

## Logprob gate diagnosis (observed)

The gate stopped at `mur.validate_logprobs`; no smoke or verifier was launched.

- Attempt 01 failed before model initialization: Triton imported `setuptools`, which was absent
  from the project venv. `setuptools 82.0.1` was installed offline into that project-local venv only.
- Attempt 02 failed in V1 engine initialization: on the shared GPU, vLLM counted co-tenant
  allocations as non-torch use, so `gpu_memory_utilization=0.26` left zero KV-cache blocks.
  A diagnostic measured both engines at `0.42/0.62`: used 14894 MiB, free 9238 MiB, new usage
  11637 MiB, both models generated, and shutdown returned to the 3257 MiB baseline.
- Attempt 03 ran five short probes with the frozen sampling settings, repeated tokens and
  `presence_penalty=1.5`, producing 112 tokens. Three probes passed; `repeat` and `ones` exceeded
  the frozen per-token/mean tolerance. No tolerance was relaxed.
- A separate vLLM `prompt_logprobs` call on the exact saved ids reproduced the generation logprobs,
  including the outlier positions (e.g. `double` token 11 = 0.2520 in both paths). The returned
  generation logprobs are therefore raw model values, not a penalty/temperature transform.
- HF eager/SDPA and BF16/float32 references were compared as bounded diagnostics. Triton attention
  reduced but did not eliminate the outliers; no tested configuration made every frozen probe pass.

Minimal decision needed from the supervisor: accept a documented validation-reference change
(attention backend and/or teacher-forced precision), revise the frozen tolerance with a stated
rationale, or keep the gate blocked. No benchmark test item was generated.
