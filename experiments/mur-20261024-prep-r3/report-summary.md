# mur-20261024-prep-r3 — summary

Status: **BLOCKED** at the raw generation-logprob gate. The 20x3 development smoke and
the general-verifier evaluation were **not** launched. No MATH-500 test item was generated.

## What ran
- CPU contract suite: 24/24 tests passed.
- `mur.prepare` verified the pinned MATH-500 file (500 rows), seven Hendrycks MATH training
  shards, and the policy/critic/verifier weight, config and tokenizer SHA256 values. The 20
  deterministic development IDs exclude normalized test questions.
- GPU access was available in the managed foreground cgroup on physical GPU 2.
- The foreground driver ran validation in a fresh child process, monitored `nvidia-smi`, and
  enforced cumulative GPU/token/wall limits. It did not launch smoke because validation failed.

## Gate result
`logprob-validation.json`: **passed=false**. Five short probes (112 generated tokens) were
generated with the frozen sampling settings, including repeated tokens under
`presence_penalty=1.5`. Three probes passed; `repeat` and `ones` exceeded the frozen
per-token/mean tolerances. The tolerances were not relaxed.

A separate vLLM `prompt_logprobs` call on the exact saved token ids reproduced the generation
logprobs, including the outlier positions. The returned generation logprobs are raw model
values; the mismatch is a vLLM-vs-HF forward/kernel precision difference on a few low-margin
tokens, not a penalty/temperature transform. HF eager/SDPA and BF16/float32 references and a
Triton attention backend were tested as bounded diagnostics; none made every frozen probe pass.

## Resource evidence
- Cumulative GPU allocation: 255 s (0.071 GPU-hours), including three failed validation
  attempts and bounded diagnostics; conservative estimate 280 s.
- Generated tokens: 245 observed (300 conservative), including validation and diagnostics.
- Session wall: 1080 s at evidence write time. All limits (3600 s GPU, 1,000,000 tokens,
  7200 s wall) were respected.
- Peak new GPU usage during the two-engine diagnostic: 11,637 MiB (< 14,336 MiB cap);
  free memory 9,238 MiB (> 4,096 MiB floor).

## Failures retained
1. Attempt 01: Triton required `setuptools`, absent from the project venv. Installed
   `setuptools 82.0.1` offline into the project-local venv only.
2. Attempt 02: vLLM V1 `gpu_memory_utilization=0.26` left no KV-cache blocks on the shared
   GPU because co-tenant allocations are counted as non-torch use. Measured 0.42/0.62 works
   within the 14 GiB new-usage cap.
3. Attempt 03: raw-logprob validation tolerance failure, as above.

## Minimal decision for the supervisor
Accept a documented validation-reference configuration (attention backend and/or
teacher-forced precision) and rerun this gate with a new commit; or revise the frozen
numerical tolerance with a stated rationale; or keep the gate blocked. Do not treat this
run as a scientific result and do not launch the full 500.

## Version control
Implementation commit before model execution: `bc68244`.
Environment observability fix: `282b8da`.
Shared-GPU memory-fraction fix: `0a69204`.
This summary/evidence commit is the head recorded in the final report.
Push mode is relay; no remote is configured on the worker.
