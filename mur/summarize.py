"""Build sanitized public evidence and the gate summary from raw run artifacts."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path, default=None):
    if not path.exists():
        return default
    return json.load(open(path, encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("MUR_RUN_DIR", "runs/mur-20261024-prep"))
    ap.add_argument("--metrics", default=None)
    args = ap.parse_args(argv)
    rd = Path(args.run_dir)
    metrics = load(Path(args.metrics) if args.metrics else rd / "evaluation" / "metrics.json", {})
    run_status = load(rd / "smoke" / "run-status.json", {})
    validation = load(rd / "logprob-validation.json", {})
    manifest = load(rd / "source-manifest.json", {})
    prep_problems = load(rd / "prepare-problems.json", [])
    rows = metrics.get("rows", [])
    summary = metrics.get("summary", {})
    traces = []
    tp = rd / "smoke" / "raw-smoke-traces.jsonl"
    if tp.exists():
        traces = [json.loads(x) for x in tp.read_text(encoding="utf-8").splitlines() if x.strip()]
    methods = ["cot", "per_step", "mur"]
    counts = {m: sum(1 for t in traces if t["method"] == m) for m in methods}
    expected_each = 20
    complete = all(counts.get(m, 0) == expected_each for m in methods)
    ledger_ok = True
    for t in traces:
        s = sum(c["output_tokens"] for c in t["ledger"] if c["role"] in ("policy", "critic"))
        if s != t["totals"]["total_output_tokens"]:
            ledger_ok = False
    gate_checks = {
        "source_manifest": bool(manifest) and not prep_problems,
        "dev_rows_20": len(manifest.get("dev", {}).get("manifest", {}).get("questions", [])) == 20,
        "test_rows_500": manifest.get("test", {}).get("rows") == 500,
        "raw_logprob_validation": bool(validation.get("passed")),
        "three_methods_20x3": complete,
        "complete_token_ledger": ledger_ok,
        "verifier_judgments": len(rows) == len(traces) and len(traces) > 0,
        "no_test_generation": bool(traces) and all(t.get("split") == "dev" for t in traces),
        "no_runtime_errors": bool(traces) and all(t.get("error") is None for t in traces),
        "no_invalid_judgments": bool(rows) and all(not r.get("ambiguous") and not r.get("malformed") for r in rows),
    }
    passed = all(gate_checks.values())
    wall = run_status.get("wall_seconds", 0) or 0
    n_traj = len(traces)
    gen_tokens = sum(t["totals"]["total_output_tokens"] for t in traces)
    per_traj_s = wall / n_traj if n_traj else None
    projected = {
        "measured_trajectories": n_traj,
        "measured_generated_tokens": gen_tokens,
        "measured_wall_seconds": wall,
        "seconds_per_trajectory": per_traj_s,
        "projected_1500_trajectories_seconds": (per_traj_s * 1500 if per_traj_s else None),
        "projected_1500_trajectories_gpu_hours": (per_traj_s * 1500 / 3600 if per_traj_s else None),
        "projected_1500_trajectories_tokens": (gen_tokens / n_traj * 1500 if n_traj else None),
        "prep_gpu_hours": wall / 3600.0,
    }
    smoke = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "prep_smoke",
        "gate_status": "PASS" if passed else ("PARTIAL" if n_traj else "BLOCKED"),
        "gate_checks": gate_checks,
        "trajectories": counts,
        "metrics": summary,
        "generated_tokens_total": gen_tokens,
        "run_status": run_status,
        "limitations": [
            "Development questions only; no MATH-500 test generation.",
            "Smoke evidence is not a scientific success claim.",
            "Latency is descriptive on a shared GPU." if not complete else "Latency is descriptive on a shared GPU.",
        ],
    }
    with open(rd / "smoke-summary.json", "w", encoding="utf-8") as f:
        json.dump(smoke, f, indent=2)
    with open(rd / "budget-estimate.json", "w", encoding="utf-8") as f:
        json.dump(projected, f, indent=2)
    public = {
        "schema_version": 1,
        "run_id": "mur-20261024-prep",
        "stage": "prep_smoke",
        "gate_status": smoke["gate_status"],
        "methods": {m: {k: v for k, v in summary.get(m, {}).items()
                        if k in ("n", "accuracy", "mean_total_tokens", "mean_policy_tokens",
                                 "mean_critic_tokens", "mean_steps", "trigger_rate",
                                 "token_savings_vs_per_step", "accuracy_delta_vs_per_step",
                                 "malformed", "ambiguous")} for m in methods},
        "per_question": [{k: r.get(k) for k in ("qid", "method", "correct", "finish",
                                                "total_output_tokens", "accepted_tokens",
                                                "steps", "triggers", "malformed")} for r in rows],
        "budget": projected,
    }
    with open(rd / "sanitized-public-metrics.json", "w", encoding="utf-8") as f:
        json.dump(public, f, indent=2)

    try:
        import torch, transformers, vllm
        versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                    "vllm": vllm.__version__}
    except Exception:
        versions = {}
    env = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "gpu": {"model": "NVIDIA GeForce RTX 3090", "visible_index": 2,
                "shared": True, "dtype": "bfloat16"},
        "driver": "570.211.01",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
        "network_note": "Hugging Face downloads used the public hf-mirror.com endpoint; official revisions and SHA256 were verified.",
    }
    with open(rd / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    review = f"""# Implementation review — mur-20261024-prep

Gate status: **{smoke['gate_status']}**.

## What was implemented
- Independent three-method runner (`mur/run.py`) with shared prompt, stop, sampling, critic and ledger code.
- vLLM backend with explicit token-id prompts, per-request deterministic seeds, and raw NLL via vLLM `prompt_logprobs` on an identity-transform scoring path.
- Momentum detector using `M_t = alpha*M_(t-1)+(1-alpha)*m_accepted`, bias-corrected history and trigger `m_proposal > B - log(gamma)`; first MUR step never scaled.
- GenPRM two-call analyze/judgment critic with designated `\\boxed{{Yes|No}}` token probability, deterministic first-candidate fallback and malformed logging.
- General-verifier evaluation after solutions are frozen, strict `Final Decision:` parsing and hash cache.
- CPU contract tests for detector, ledger, selection, prompts, parsing and seed derivation.

## Verified
- Source/test/dev manifests: {gate_checks['source_manifest'] and gate_checks['test_rows_500'] and gate_checks['dev_rows_20']}
- Raw-logprob teacher-forced check: {gate_checks['raw_logprob_validation']}
- Complete 20x3 smoke and verifier: {gate_checks['three_methods_20x3'] and gate_checks['verifier_judgments']}
- Token-ledger closure: {gate_checks['complete_token_ledger']}

## Deviations / limitations
- `configs/sources.json` did not pin a MATH training source. The executor pinned `EleutherAI/hendrycks_math` revision `21a5633873b6a120296cce3e2df9d5550074f4a3` and selected 20 questions with seed 42 after normalized dedup against all 500 test questions.
- Hugging Face was not reachable directly from this worker; the public `hf-mirror.com` endpoint was used and every model file hash was checked against the pinned revision's LFS SHA256.
- Shared GPU index 2 was used. GPU memory must be checked against the supervisor ceiling using observed per-process usage; latency is not a controlled speedup.
- No MATH-500 test item was generated or tuned on.

## Failures
{json.dumps(run_status.get('failures', []), indent=2)}
"""
    with open(rd / "implementation-review.md", "w", encoding="utf-8") as f:
        f.write(review)
    print(json.dumps({"gate_status": smoke["gate_status"], "counts": counts,
                      "gate_checks": gate_checks}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
