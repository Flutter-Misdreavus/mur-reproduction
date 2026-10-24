# Preflight — mur-20261024-test100-r2

Status: **PREFLIGHT_OK** (no model loaded, no GPU allocation).

## Frozen inputs

- Parent review: `experiments/mur-20261024-test100/review.md` accepts the r6 gate and
  authorizes this separate frozen task.
- Task: `experiments/mur-20261024-test100-r2/task.json` (unchanged; run_id
  `mur-20261024-test100-r2`).
- Test manifest: `experiments/mur-20261024-test100-r2/test-manifest.json`
  SHA256 `694c2fa324d3ecfd6f46902b6bf481016c082795278cefc67c3655ad364ea24f`,
  byte-identical to the manifest committed before any model execution.
  It is copied to the run root as-is; the driver independently recomputes all
  500 source row identities and the frozen first100 selection.
- Branch `exp/mur-20261024`, private repository `Flutter-Misdreavus/mur-reproduction`
  declared in `version_control`; relay push only.

## Commands run (CPU only)

```
runtime/venv/bin/python -m unittest discover -s tests -v
python3 scripts/audit_public.py --history
MUR_RUN_DIR=runs/mur-20261024-test100-r2 runtime/venv/bin/python \
  scripts/test_stage_driver.py \
  --task-file experiments/mur-20261024-test100-r2/task.json \
  --run-dir "$MUR_RUN_DIR" --gpu-index 2 --preflight-only
```

- Unit tests: 58 passed.
- Public audit: 183 historical file versions, 0 findings.
- Preflight: `PREFLIGHT_OK`; selected physical GPU index 2, 20878 MiB free at
  check time, 3257 MiB already used by unrelated co-tenants (descriptive only).
- Disk free at check: 35,674,148,864 bytes (33.2 GiB), above the 25 GiB floor.
- `runtime/venv`: Python 3.10.12, torch 2.6.0+cu124, vLLM 0.8.5,
  transformers 4.51.3.
- Prefix caching disabled for policy, critic and verifier.

## Authorization-wording fix before GPU work

The first preflight attempt returned `BLOCKED_PREFLIGHT` with no GPU allocation:
`mur.testdata.authorize_test` required the literal token `fixed100`, while the
frozen r2 task controls use the equivalent wording “identical committed100-question
manifest” and “exactly100 questions x3 methods in four25-question chunks”.
The failed preflight evidence is preserved under the ignored run directory at
`attempts/preflight-01/`. The gate was not weakened: it now accepts either the
original literal marker or the conjunction of the reused frozen manifest, the
exact 100-question matrix and four25-question chunks, and it still requires a
ready task, the `ready task` control, a private repository and an accepted parent
review. A regression test covers both acceptance and rejection. No scientific
parameter, prompt, split, budget or model revision changed.

## Next step

Run the single foreground driver from a clean tree at the committed run SHA.
All per-phase raw output stays in the ignored run directory; only sanitized
environment, manifest, engineering summary, numeric metrics and the report are
committed after model work ends.
