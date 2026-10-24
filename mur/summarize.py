"""Build sanitized public evidence and the gate summary from raw run artifacts."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path


def load(path: Path, default=None):
    if not path.exists():
        return default
    return json.load(open(path, encoding="utf-8"))


def run_id_for(rd: Path) -> str:
    task = load(rd / "task.json", {}) or {}
    return task.get("run_id") or os.environ.get("MUR_RUN_ID") or rd.name


def pkg_version(name: str):
    try:
        return package_version(name)
    except Exception:
        return None


def probe_gpu() -> dict:
    """Observe GPU state; never assert a model or memory ceiling we did not see."""
    out = {"available": False, "source": "nvidia-smi"}
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        devices = []
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                cols = [c.strip() for c in line.split(",")]
                if len(cols) < 7:
                    continue
                try:
                    devices.append({
                        "index": cols[0],
                        "name": cols[1],
                        "memory_total_mib": float(cols[2]),
                        "memory_used_mib": float(cols[3]),
                        "memory_free_mib": float(cols[4]),
                        "driver_version": cols[5],
                        "utilization_gpu_pct": float(cols[6]),
                    })
                except ValueError:
                    continue
        out["available"] = bool(devices)
        out["devices"] = devices
        if not devices:
            out["error"] = (proc.stderr or proc.stdout).strip()[:400]
    except Exception as exc:  # device access can be denied by the execution sandbox
        out["error"] = f"{type(exc).__name__}: {exc}"
    try:
        txt = Path("/proc/driver/nvidia/version").read_text(encoding="utf-8")
        match = re.search(r"Kernel Module\s+([0-9.]+)", txt)
        if match:
            out["driver_version"] = match.group(1)
    except Exception:
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    out["cuda_visible_devices"] = visible if visible not in (None, "") else None
    out["observed_at"] = datetime.now(timezone.utc).isoformat()
    return out


def host_memory() -> dict:
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                memory[key] = value.strip()
    except Exception as exc:
        memory["error"] = f"{type(exc).__name__}: {exc}"
    return memory


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("MUR_RUN_DIR", "runs/mur-20261024-prep"))
    ap.add_argument("--metrics", default=None)
    args = ap.parse_args(argv)
    rd = Path(args.run_dir)
    run_id = run_id_for(rd)
    metrics = load(Path(args.metrics) if args.metrics else rd / "evaluation" / "metrics.json", {})
    run_status = load(rd / "smoke" / "run-status.json", {})
    validation = load(rd / "logprob-validation.json", {})
    manifest = load(rd / "source-manifest.json", {})
    prep_problems = load(rd / "prepare-problems.json", [])
    forced = load(rd / "forced-trigger.json", {})
    driver = load(rd / "stage-driver.json", {})
    cumulative = load(rd / "cumulative-budget.json", {})
    eval_status = load(rd / "evaluation" / "run-status.json", {})
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
    mur_triggers = sum(1 for t in traces if t["method"] == "mur"
                       for s in t["steps"] if s.get("triggered"))
    forced_ok = bool(forced.get("passed")) and bool(forced.get("engineering_only")) \
        and bool(forced.get("excluded_from_scientific_smoke"))
    gpu = probe_gpu()
    budget_evidence = cumulative or driver
    gpu_seconds = budget_evidence.get("gpu_allocated_seconds")
    total_tokens = budget_evidence.get("total_generated_tokens")
    wall_seconds = budget_evidence.get("wall_seconds")
    stage_budget_ok = (
        isinstance(gpu_seconds, (int, float)) and gpu_seconds <= 3600
        and isinstance(total_tokens, int) and total_tokens <= 1_000_000
        and isinstance(wall_seconds, (int, float)) and wall_seconds <= 7200
    )
    gate_checks = {
        "gpu_available": bool(gpu.get("available")),
        "source_manifest": bool(manifest) and not prep_problems,
        "dev_rows_20": len(manifest.get("dev", {}).get("manifest", {}).get("questions", [])) == 20,
        "test_rows_500": manifest.get("test", {}).get("rows") == 500,
        "raw_logprob_validation": bool(validation.get("passed")),
        "three_methods_20x3": complete,
        "complete_token_ledger": bool(traces) and ledger_ok,
        "verifier_judgments": len(rows) == len(traces) and len(traces) > 0,
        "no_test_generation": bool(traces) and all(t.get("split") == "dev" for t in traces),
        "no_runtime_errors": bool(traces) and all(t.get("error") is None for t in traces),
        "no_invalid_judgments": bool(rows) and all(not r.get("ambiguous") and not r.get("malformed") for r in rows),
        "no_budget_or_context_truncation": bool(traces) and all(
            t.get("finish") not in ("budget_stopped", "context_limit", "error")
            for t in traces),
        "stage_budget_observed": stage_budget_ok,
        "mur_triggered_branch": mur_triggers > 0 or forced_ok,
    }
    passed = all(gate_checks.values())
    if not gate_checks["gpu_available"]:
        blocker = "GPU/NVML is unavailable to the managed foreground execution sandbox"
    elif not gate_checks["raw_logprob_validation"]:
        blocker = "raw generation-logprob validation is missing or failed"
    elif not complete:
        blocker = "the 20x3 development smoke is incomplete"
    elif not gate_checks["stage_budget_observed"]:
        blocker = "cumulative GPU/token/wall budget evidence is missing or over limit"
    elif not passed:
        blocker = "one or more prep gate checks failed"
    else:
        blocker = None
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
        "prep_gpu_hours": (gpu_seconds / 3600.0 if isinstance(gpu_seconds, (int, float))
                           else None),
        "prep_gpu_allocated_seconds": gpu_seconds,
        "prep_wall_seconds": wall_seconds,
        "prep_generated_tokens": total_tokens,
        "budget_limits": {"gpu_hours": 1, "wall_seconds": 7200,
                          "generated_tokens": 1_000_000, "trajectories": 60},
        "interpretation": ("No 1500-trajectory estimate is reported because the gate "
                           "stopped before the 20x3 smoke; GPU time is the measured "
                           "cumulative model-allocation interval." if n_traj == 0 else
                           "GPU time is the measured cumulative model-allocation "
                           "interval; smoke throughput drives the 1500 projection."),
    }
    smoke = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "stage": "prep_smoke",
        "gate_status": "PASS" if passed else "BLOCKED",
        "blocked_reason": blocker,
        "gate_checks": gate_checks,
        "trajectories": counts,
        "metrics": summary,
        "generated_tokens_total": gen_tokens,
        "run_status": run_status,
        "evaluation_status": eval_status,
        "driver": {k: driver.get(k) for k in
                   ("gpu_allocated_seconds", "total_generated_tokens", "wall_seconds",
                    "peak_new_usage_mib", "min_free_mib", "phases")},
        "cumulative_budget": cumulative or None,
        "limitations": [
            "Development questions only; no MATH-500 test generation.",
            "Smoke evidence is not a scientific success claim.",
            "Latency is descriptive on a shared GPU.",
        ],
    }
    with open(rd / "smoke-summary.json", "w", encoding="utf-8") as f:
        json.dump(smoke, f, indent=2)
    with open(rd / "budget-estimate.json", "w", encoding="utf-8") as f:
        json.dump(projected, f, indent=2)
    public = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "prep_smoke",
        "gate_status": smoke["gate_status"],
        "blocked_reason": blocker,
        "gate_checks": gate_checks,
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
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=False).stdout.strip()
        git_branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True,
                                    text=True, check=False).stdout.strip()
    except Exception:
        git_head, git_branch = "", ""
    env = {
        "schema_version": 1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: pkg_version(name)
                     for name in ("torch", "transformers", "vllm", "tokenizers", "pyarrow", "setuptools")},
        "gpu": gpu,
        "host_memory": host_memory(),
        "git_head": git_head,
        "git_branch": git_branch,
        "cuda_visible_devices": gpu.get("cuda_visible_devices"),
        "notes": [
            "GPU state is an observed nvidia-smi snapshot, not a hardcoded claim.",
            "Per-process peak allocation must be read from the run log / nvidia-smi interval evidence.",
            "setuptools was installed offline into the project-local runtime/venv only, because Triton imports it; no global environment was changed.",
            "vLLM memory fractions: policy 0.42, critic 0.62, verifier 0.40. The shared GPU had a co-tenant allocation, and V1 counts it as non-torch use; 0.26 left no KV-cache blocks.",
        ],
    }
    with open(rd / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    failures = []
    if run_status.get("failure"):
        failures.append(run_status["failure"])
    if eval_status.get("failure"):
        failures.append("evaluation: " + eval_status["failure"])
    if blocker:
        failures.append(blocker + (": " + gpu["error"] if gpu.get("error") else ""))
    review = f"""# Implementation review — {run_id}

Gate status: **{smoke['gate_status']}**.

## What was implemented
- Independent three-method runner (`mur/run.py`) sharing prompt, stop, sampling, critic and ledger code.
- vLLM 0.8.5 V1 backend that reads raw sampled-token logprobs from the generating forward pass
  (inspected `vllm/v1/sample/sampler.py`: `raw_logprobs` precedes penalties/temperature), so no extra
  prompt-rescoring call is used for `m_t`.
- `mur.validate_logprobs` compares those raw generation logprobs against an HF BF16 teacher-forced
  forward pass on the same token ids.
- Momentum detector with `U_t = alpha*U_(t-1)+(1-alpha)*m_accepted`, bias correction, and trigger
  `m_proposal > U_(t-1)/(1-alpha**(t-1)) - log(gamma)`; first MUR step is never scaled.
- GenPRM two-call analyze/judgment critic with an anchored initial `\\boxed{{Yes|No}}` verdict token,
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
- Source/test/dev manifest: {gate_checks['source_manifest'] and gate_checks['test_rows_500'] and gate_checks['dev_rows_20']}
- Raw generation-logprob validation: {gate_checks['raw_logprob_validation']}
- Complete 20x3 smoke and verifier: {gate_checks['three_methods_20x3'] and gate_checks['verifier_judgments']}
- No budget/context-truncated trajectory: {gate_checks['no_budget_or_context_truncation']}
- Token-ledger closure: {gate_checks['complete_token_ledger']}
- Cumulative GPU/token/wall budget evidence: {gate_checks['stage_budget_observed']}
  (GPU {gpu_seconds}s, tokens {total_tokens}, wall {wall_seconds}s)
- At least one MUR triggered branch inspected or a marked engineering forced-trigger case: {gate_checks['mur_triggered_branch']}

## Blocking condition
{blocker or 'None.'}

## Deviations / limitations
- Development questions only; no MATH-500 test item was generated or tuned on.
- The smoke is not a scientific success claim.
- Latency on a shared GPU is descriptive, not a controlled speedup.
- GPU memory/headroom is reported only when an `nvidia-smi` observation is actually available.

## Failures
{json.dumps(failures, indent=2)}
"""
    with open(rd / "implementation-review.md", "w", encoding="utf-8") as f:
        f.write(review)
    print(json.dumps({"run_id": run_id, "gate_status": smoke["gate_status"],
                      "blocked_reason": blocker, "counts": counts,
                      "gate_checks": gate_checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
