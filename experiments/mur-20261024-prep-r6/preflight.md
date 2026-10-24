# Preflight — mur-20261024-prep-r6

Status: CPU checks passed; one bounded GPU attempt not yet started.

## Code and installed contract
- Branch `exp/mur-20261024`, base commit `bcc60c7` (r6 text-stop tail retention commit).
- Reviewed installed vLLM 0.8.5 V1 `output_processor.py`: a client-side text-stop
  match requests engine abort (lines 343-373), so the engine can sample a trailing
  token after the client has already formed its stop-trimmed output.
- Reviewed installed `detokenizer.py`: `stop_reason` is the exact configured stop
  string; client `token_ids` retain the sampled tokens while output text is truncated.
  This supports retaining a sampler tail only when `finish_reason=stop` and
  `stop_reason="\nStep"`.
- Installed sampler source hash and raw-logprob ordering evidence are produced by
  `mur.validate_forward` during the run; no vLLM installation file was modified.

## CPU checks
- `runtime/venv/bin/python -m unittest discover -s tests -v`: 45/45 passed.
- `python3 scripts/audit_public.py --history`: 147 file versions audited, 0 findings.
- `runtime/venv/bin/python -m mur.prepare ...`: test rows 500, dev rows 20,
  pinned file and model hashes verified, no `prepare-problems.json`.
- Versions: Python 3.10.12, torch 2.6.0+cu124, vLLM 0.8.5, transformers 4.51.3.

## Resource preflight
- Physical GPU 2 visible through `CUDA_VISIBLE_DEVICES=2`; observed 20878 MiB free
  before start, above the 4096 MiB floor.
- Data volume had about 34 GB free before start, above the 25 GiB stop floor.
- No downloads, global environment changes or shared-package edits are planned.
- Budget: one foreground driver, <=3000 s GPU allocation including cleanup,
  <=1,000,000 generated tokens, <=7000 s driver wall, 14 GiB new-memory and
  4 GiB free-memory guards.

## Gate ordering
The driver runs the eight-probe same-forward direct gate first. Only a clean pass
with exact returned-token alignment, retained/validated text-stop tails, repeated
tokens and the temperature-only negative control can launch the 20x3 development
smoke and verifier. No MATH-500 test generation is permitted in this task.
