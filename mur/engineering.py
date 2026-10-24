"""Bounded no-trigger identity check on development indices 0 and 3.

One policy instance, identical seeds, CoT and MUR in sequence.  The check records
the first proposals and every common pre-trigger accepted step, including exact
token IDs.  A genuine MUR trigger is never suppressed; only the shared portion
before it is compared.  Failure blocks the test before any test generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .backend import VLLMBackend
from .limits import StageBudget, StageBudgetExceeded
from .run import load_json, model_path, run_one, write_json_atomic


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_pair(cot: dict, mur: dict) -> dict:
    problems = []
    cot_steps = cot.get("steps") or []
    mur_steps = mur.get("steps") or []
    if not cot_steps or not mur_steps:
        return {"qid": cot.get("qid"), "passed": False,
                "problems": ["empty trajectory"], "first_trigger_step": None}
    c0, m0 = cot_steps[0], mur_steps[0]
    if c0.get("proposal") != m0.get("proposal"):
        problems.append("first proposal text differs")
    if c0.get("proposal_token_ids") != m0.get("proposal_token_ids"):
        problems.append("first proposal token IDs differ")
    triggers = [i for i, s in enumerate(mur_steps) if s.get("triggered")]
    first_trigger = min(triggers) if triggers else None
    if first_trigger is None:
        prefix_len = min(len(cot_steps), len(mur_steps))
        compared_steps = prefix_len
        if len(cot_steps) != len(mur_steps):
            problems.append(
                f"no trigger but trajectory lengths differ "
                f"({len(cot_steps)} != {len(mur_steps)})")
    else:
        prefix_len = first_trigger
        compared_steps = prefix_len
    for i in range(prefix_len):
        cs, ms = cot_steps[i], mur_steps[i]
        if bool(cs.get("final_fallback")) != bool(ms.get("final_fallback")):
            problems.append(f"step {i}: final-fallback flag differs")
        if cs.get("accepted_text") != ms.get("accepted_text"):
            problems.append(f"step {i}: accepted text differs")
        if cs.get("accepted_token_ids") != ms.get("accepted_token_ids"):
            problems.append(f"step {i}: accepted token IDs differ")
    return {
        "qid": cot.get("qid"),
        "passed": not problems,
        "problems": problems,
        "first_trigger_step": first_trigger,
        "triggered_steps": triggers,
        "compared_steps_before_trigger": compared_steps,
        "cot_steps": len(cot_steps),
        "mur_steps": len(mur_steps),
        "first_proposal_sha256": hashlib.sha256(
            (c0.get("proposal") or "").encode("utf-8")).hexdigest(),
        "first_proposal_tokens_sha256": hashlib.sha256(
            json.dumps(c0.get("proposal_token_ids") or []).encode()).hexdigest(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default="configs/core.json")
    ap.add_argument("--run-id", default="mur-20261024-test100")
    ap.add_argument("--dev-indices", default="0,3")
    ap.add_argument("--max-generated-tokens", type=int, default=30000)
    ap.add_argument("--max-wall-seconds", type=float, default=280)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "raw-engineering-traces.jsonl"
    if trace_path.exists():
        raise SystemExit(f"refusing to overwrite existing engineering traces: {trace_path}")
    root = Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime"))
    dev = [json.loads(x) for x in (root / "data" / "dev.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    indices = [int(x) for x in args.dev_indices.split(",") if x.strip()]
    questions = []
    for i in indices:
        if i < 0 or i >= len(dev):
            raise SystemExit(f"dev index out of range: {i}")
        questions.append(dev[i])
    cfg = load_json(args.config)
    cfg["run_id"] = args.run_id
    cfg["_split"] = "dev"
    budget = StageBudget(args.max_generated_tokens, args.max_wall_seconds,
                         args.max_model_len)
    traces = []
    failure = None
    started = time.time()
    policy = critic = None
    exit_code = 0
    try:
        sources = load_json("configs/sources.json")
        policy = VLLMBackend(str(model_path(sources["policy"]["id"])),
                             gpu_memory_utilization=0.42,
                             max_model_len=args.max_model_len,
                             seed=int(cfg["seed"]), enable_prefix_caching=False)
        critic = VLLMBackend(str(model_path(sources["critic"]["id"])),
                             gpu_memory_utilization=0.62,
                             max_model_len=args.max_model_len,
                             seed=int(cfg["seed"]), enable_prefix_caching=False)
        for q in questions:
            for method in ("cot", "mur"):
                rec = run_one(q, method, cfg, policy, critic, trace_path, budget)
                traces.append(rec)
                if rec.get("error") is not None:
                    raise RuntimeError("engineering trajectory failed: " + rec["error"])
                print(f"engineering {q['id']} {method} finish={rec['finish']} "
                      f"tokens={rec['totals']['total_output_tokens']}", flush=True)
    except StageBudgetExceeded as exc:
        failure = f"StageBudgetExceeded: {exc}"
        exit_code = 3
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        exit_code = 2
        traceback.print_exc()
    finally:
        for backend in (policy, critic):
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass

    pairs = []
    for q in questions:
        cot = next((t for t in traces if t["qid"] == q["id"] and t["method"] == "cot"), None)
        mur = next((t for t in traces if t["qid"] == q["id"] and t["method"] == "mur"), None)
        if cot is None or mur is None:
            pairs.append({"qid": q["id"], "passed": False,
                          "problems": ["missing CoT or MUR trajectory"]})
        else:
            pairs.append(compare_pair(cot, mur))
    passed = (failure is None and len(traces) == len(questions) * 2
              and all(p.get("passed") for p in pairs))
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "stage": "engineering_identity_check",
        "passed": passed,
        "failure": failure,
        "dev_indices": indices,
        "dev_ids": [q["id"] for q in questions],
        "pairs": pairs,
        "trajectories": len(traces),
        "generated_tokens": budget.generated,
        "wall_seconds": time.time() - started,
        "budget": {"max_generated_tokens": args.max_generated_tokens,
                   "max_wall_seconds": args.max_wall_seconds,
                   "max_model_len": args.max_model_len},
        "prefix_caching": False,
        "criteria": (
            "First proposals must match in text and token IDs; every accepted step "
            "before the first genuine MUR trigger must match in text and token IDs. "
            "No trigger was suppressed and no numerical variant was retried."),
        "raw_trace": "raw-engineering-traces.jsonl",
    }
    if trace_path.exists():
        payload["raw_trace_sha256"] = sha256_file(trace_path)
    write_json_atomic(out / "engineering-check.json", payload)
    print(json.dumps({k: payload[k] for k in
                      ("passed", "failure", "dev_ids", "pairs", "trajectories",
                       "generated_tokens", "wall_seconds")}, indent=2))
    return exit_code if not passed else 0


if __name__ == "__main__":
    sys.exit(main())
