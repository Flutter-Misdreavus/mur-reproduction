# Report summary — mur-20261024-prep-r2

Status: **BLOCKED** (no GPU execution; preparation evidence retained).

## Observed
- CPU contract/regression tests: 20 tests pass in the pinned runtime.
- `mur.prepare` verified the pinned MATH-500 file (500 rows, SHA256
  `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`) and produced the same
  20 deterministic development IDs (seed 42, normalized-test dedup) as the earlier attempt.
- Model manifests verified pinned SHA256 for all weight files and for config/tokenizer files of
  Qwen3-1.7B, GenPRM-1.5B and general-verifier.
- Runtime: Python 3.10.12, vLLM 0.8.5, transformers 4.51.3, torch 2.6.0+cu124,
  tokenizers 0.21.4, pyarrow 25.0.1.
- GPU is not usable inside the managed foreground sandbox. `nvidia-smi` fails with
  `Failed to initialize NVML: Unknown Error`; torch reports `cuda_available=False`, device count 0,
  CUDA error 304; `/dev/nvidia*` opens are denied. The wider tool sandbox required approval and no
  approval channel was available.
- The task explicitly forbids detached `systemd-run`/`nohup`/`setsid` and requires the child cgroup
  to belong to the current wrapper service. No GPU job was started through an escape hatch.

## Not run
Raw generation-logprob validation, the 20x3 smoke, critic verification and general-verifier
evaluation were not run because no GPU process could be started in the required cgroup. Metrics are
empty, not zero. `logprob-validation.json` records `NOT_RUN`.

## Gate
`source_manifest`, `dev_rows_20`, `test_rows_500` and `complete_token_ledger` are true; all GPU-
dependent checks are false. Gate status is `BLOCKED`.

## Budget
No GPU allocation; generated model tokens = 0. Wall time is reported from the worker interval.

## Commits
- `72f2bd8` — implementation/config fixes: config/tokenizer hash verification, observed-environment
  reporting, 0.26 policy memory fraction, regression test.
- This commit — sanitized prep evidence and report summary.

## Next recommendation
Provide GPU access inside the managed DSH foreground cgroup (or explicitly authorize a tracked
scheduler job with automatic cleanup), then rerun logprob validation and the full 20x3 smoke from
the committed code. Do not treat this blocked run as a scientific result or reuse the earlier
14 partial trajectories.
