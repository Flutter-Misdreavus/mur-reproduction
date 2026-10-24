# Implementation review — mur-20261024-prep-r2

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

## Verified
- Source/test/dev manifest: True
- Raw generation-logprob validation: False
- Complete 20x3 smoke and verifier: False
- Token-ledger closure: True
- At least one MUR triggered branch inspected or a marked engineering forced-trigger case: False

## Blocking condition
GPU/NVML is unavailable to the managed foreground execution sandbox

## Deviations / limitations
- Development questions only; no MATH-500 test item was generated or tuned on.
- The smoke is not a scientific success claim.
- Latency on a shared GPU is descriptive, not a controlled speedup.
- GPU memory/headroom is reported only when an `nvidia-smi` observation is actually available.

## Failures
[
  "GPU/NVML is unavailable to the managed foreground execution sandbox: Failed to initialize NVML: Unknown Error"
]
