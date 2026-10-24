"""Frozen-solution evaluation with TIGER-Lab/general-verifier.

Grading is a separate process from generation.  For the fixed100 test split the
gold answer is joined back only here, after every generation was frozen, and the
join is strict: unknown, duplicate or missing IDs fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

from .backend import VLLMBackend
from .limits import ContextLimit, StageBudget, StageBudgetExceeded
from .prompts import parse_decision, request_seed, verifier_prompt
from . import testdata


def model_path(repo: str) -> Path:
    root = Path(os.environ.get("MUR_MODEL_ROOT") or
                (Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime")) / "models"))
    return root / repo.split("/")[-1]


def write_json_atomic(path: Path, payload):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def trigger_stats(method: str, rows: list[dict]) -> dict:
    """Correct auxiliary trigger-rate denominators.

    Per-step Guided Search scales every step: eligible = all steps.
    MUR never scales the first step: eligible = sum(max(steps-1, 0)).
    Also retain the all-step rate so the two definitions cannot be conflated.
    """
    numerator = sum(int(r.get("triggers", 0)) for r in rows)
    all_steps = sum(int(r.get("steps", 0)) for r in rows)
    if method == "mur":
        eligible = sum(max(0, int(r.get("steps", 0)) - 1) for r in rows)
        definition = "sum(max(steps-1,0)); MUR never scales the first step"
    elif method == "per_step":
        eligible = all_steps
        definition = "all steps; per-step Guided Search scales every step"
    else:
        eligible = 0
        definition = "not applicable"
    return {
        "numerator": numerator,
        "denominator": eligible,
        "rate": (numerator / eligible) if eligible else None,
        "all_step_numerator": numerator,
        "all_step_denominator": all_steps,
        "all_step_rate": (numerator / all_steps) if all_steps else None,
        "denominator_definition": definition,
    }


def load_reference(split: str, sources: dict, test_manifest: str | None,
                   task_file: str | None, acceptance_review: str | None,
                   run_id: str) -> tuple[dict, dict | None, dict | None]:
    """Return qid -> (problem, gold answer), test manifest and authorization."""
    root = Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime"))
    if split == "dev":
        reference = {}
        for line in (root / "data" / "dev.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                reference[r["id"]] = {"problem": r["problem"], "answer": r["answer"],
                                      "index": r.get("row_index")}
        return reference, None, None
    if split != "test":
        raise ValueError(f"unsupported split: {split}")
    if not (test_manifest and task_file and acceptance_review):
        raise PermissionError("test grading requires manifest, task and accepted review")
    authorization = testdata.authorize_test(Path(task_file), Path(acceptance_review), run_id)
    manifest = json.loads(Path(test_manifest).read_text(encoding="utf-8"))
    rows, _ = testdata.load_source_rows(root, sources)
    problems = testdata.verify_test_manifest(manifest, rows, sources)
    if problems:
        raise ValueError("test manifest verification failed: " + "; ".join(problems))
    by_index = {q["index"]: q for q in manifest["questions"]}
    reference = {}
    for index, q in by_index.items():
        source = rows[int(index)]
        if testdata.sha256_text(source["problem"]) != q["question_sha256"]:
            raise ValueError(f"source problem hash mismatch at row {index}")
        reference[q["id"]] = {"problem": source["problem"],
                              "answer": source["answer"],
                              "index": int(index)}
    return reference, manifest, authorization


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--test-manifest", default=None)
    ap.add_argument("--task-file", default=None)
    ap.add_argument("--acceptance-review", default=None)
    ap.add_argument("--expected-ids", default=None)
    ap.add_argument("--run-id", default="mur-20261024-prep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-generated-tokens", type=int, default=250_000)
    ap.add_argument("--max-wall-seconds", type=float, default=900)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = json.load(open(args.sources, encoding="utf-8"))
    seed = args.seed
    reference, manifest, authorization = load_reference(
        args.split, sources, args.test_manifest, args.task_file,
        args.acceptance_review, args.run_id)
    traces_path = inp / "raw-smoke-traces.jsonl"
    if not traces_path.exists():
        traces_path = inp / "raw-traces.jsonl"
    traces = []
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))

    # Strict identity join.  A chunk may cover a subset of the manifest but must
    # name exactly the expected IDs when the driver supplies them.
    seen = set()
    expected_ids = None
    if args.expected_ids:
        expected_ids = [x.strip() for x in Path(args.expected_ids).read_text(
            encoding="utf-8").splitlines() if x.strip()]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("duplicate IDs in expected-ids file")
    for t in traces:
        qid = t.get("qid")
        if qid not in reference:
            raise ValueError(f"unknown test/dev ID in trace: {qid!r}")
        pair = (qid, t.get("method"))
        if pair in seen:
            raise ValueError(f"duplicate trace row: {pair}")
        seen.add(pair)
        if t.get("split") != args.split:
            raise ValueError(f"split mismatch for {qid}: {t.get('split')} != {args.split}")
    if expected_ids is not None:
        expected_pairs = {(qid, method) for qid in expected_ids
                          for method in ("cot", "per_step", "mur")}
        if seen != expected_pairs:
            raise ValueError("missing or unknown method/question pairs before grading")
        got_ids = {t["qid"] for t in traces}
        if got_ids != set(expected_ids):
            missing = sorted(set(expected_ids) - got_ids)
            extra = sorted(got_ids - set(expected_ids))
            raise ValueError(f"ID join mismatch missing={missing} extra={extra}")

    cache_path = Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime")) / "cache" / "verifier.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r

    # Cache key includes model revision, exact prompt, sampling, seed, context
    # length and effective output cap.  Tokenization is CPU-only.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path(sources["verifier"]["id"])), trust_remote_code=True)
    todo = {}
    for t in traces:
        q = reference[t["qid"]]
        prompt = verifier_prompt(q["problem"], q["answer"], t.get("final_answer", ""))
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        effective_cap = min(4096, args.max_model_len - prompt_len)
        key_payload = {
            "model": sources["verifier"],
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens_requested": 4096,
            "max_tokens_effective": effective_cap,
            "max_model_len": args.max_model_len,
            "seed": request_seed(seed, t["qid"], 0, "verifier", 0),
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        t["_prompt"] = prompt
        t["_key"] = key
        t["_prompt_tokens"] = prompt_len
        t["_effective_cap"] = effective_cap
        t["_was_cached"] = key in cache
        if key not in cache and key not in todo:
            todo[key] = t

    budget = StageBudget(max_generated=args.max_generated_tokens,
                         max_wall_seconds=args.max_wall_seconds,
                         max_model_len=args.max_model_len)
    failure = None
    exit_code = 0
    verified_now = 0
    backend = None
    try:
        if todo:
            backend = VLLMBackend(str(model_path(sources["verifier"]["id"])),
                                  gpu_memory_utilization=0.40,
                                  max_model_len=args.max_model_len, seed=seed,
                                  enable_prefix_caching=False)
            pending = list(todo.values())
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start:start + args.batch_size]
                caps = []
                for t in chunk:
                    if t["_effective_cap"] <= 0:
                        raise ContextLimit("verifier prompt leaves no output context")
                    caps.append(budget.fit(t["_prompt_tokens"], 4096, "verifier"))
                    if caps[-1] != t["_effective_cap"]:
                        raise ValueError("cache effective-cap mismatch")
                budget.reserve(caps, "verifier")
                from vllm import SamplingParams
                params = [SamplingParams(max_tokens=cap, temperature=0.7, n=1,
                                         seed=request_seed(seed, t["qid"], 0, "verifier", 0))
                          for t, cap in zip(chunk, caps)]
                gens = backend.generate_many([t["_prompt"] for t in chunk], params)
                with open(cache_path, "a", encoding="utf-8") as cf:
                    for t, g, cap in zip(chunk, gens, caps):
                        budget.account("verifier", len(g.token_ids))
                        rec = {
                            "key": t["_key"],
                            "model": sources["verifier"],
                            "revision": sources["verifier"]["revision"],
                            "text": g.text,
                            "finish": g.finish_reason,
                            "tokens": len(g.token_ids),
                            "prompt_tokens": g.prompt_tokens,
                            "max_tokens_effective": cap,
                            "max_model_len": args.max_model_len,
                            "temperature": 0.7,
                            "seed": request_seed(seed, t["qid"], 0, "verifier", 0),
                        }
                        cache[t["_key"]] = rec
                        cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                verified_now += len(chunk)
                print(f"verified {verified_now}/{len(pending)}", flush=True)
    except (StageBudgetExceeded, ContextLimit) as e:
        failure = f"{type(e).__name__}: {e}"
        exit_code = 3
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
        exit_code = 2
        traceback.print_exc()
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        tokenizer = None

    rows = []
    missing = []
    for t in traces:
        c = cache.get(t["_key"])
        if c is None:
            missing.append(t["qid"])
            continue
        label, ambiguous = parse_decision(c["text"])
        rows.append({
            "qid": t["qid"], "method": t["method"], "split": args.split,
            "source_index": t.get("source_index"),
            "finish": t["finish"],
            "verifier_label": label,
            "ambiguous": ambiguous,
            "correct": int(label == "Yes" and not ambiguous
                           and t["finish"] not in ("error", "budget_stopped")),
            "correct_automated": int(label == "Yes" and not ambiguous
                                     and t["finish"] not in ("error", "budget_stopped")),
            "audited_label": None,
            "audit_status": "pending",
            "accepted_tokens": t["accepted_tokens"],
            "policy_output_tokens": t["totals"]["policy_output_tokens"],
            "critic_output_tokens": t["totals"]["critic_output_tokens"],
            "total_output_tokens": t["totals"]["total_output_tokens"],
            "policy_input_tokens": t["totals"].get("policy_input_tokens", 0),
            "critic_input_tokens": t["totals"].get("critic_input_tokens", 0),
            "steps": len(t["steps"]),
            "triggers": sum(1 for s in t["steps"] if s.get("triggered")),
            "fallbacks": sum(1 for s in t["steps"] if s.get("fallback")),
            "malformed": sum(1 for s in t["steps"] for c2 in s.get("candidates", [])
                             if c2["verify"]["judgment"]["label"] is None),
            "wall_seconds": t["wall_seconds"],
            "verifier_tokens": c["tokens"],
            "verifier_cached": bool(t.get("_was_cached")),
        })
    summary = {}
    for m in sorted({r["method"] for r in rows}):
        mr = [r for r in rows if r["method"] == m]
        ts = trigger_stats(m, mr)
        summary[m] = {
            "n": len(mr),
            "accuracy": sum(r["correct"] for r in mr) / len(mr),
            "mean_total_tokens": statistics.mean(r["total_output_tokens"] for r in mr),
            "mean_policy_tokens": statistics.mean(r["policy_output_tokens"] for r in mr),
            "mean_critic_tokens": statistics.mean(r["critic_output_tokens"] for r in mr),
            "mean_accepted_tokens": statistics.mean(r["accepted_tokens"] for r in mr),
            "mean_steps": statistics.mean(r["steps"] for r in mr),
            "trigger_numerator": ts["numerator"],
            "trigger_denominator": ts["denominator"],
            "trigger_rate": ts["rate"],
            "trigger_rate_all_steps": ts["all_step_rate"],
            "trigger_all_step_numerator": ts["all_step_numerator"],
            "trigger_all_step_denominator": ts["all_step_denominator"],
            "trigger_denominator_definition": ts["denominator_definition"],
            "malformed": sum(r["malformed"] for r in mr),
            "fallbacks": sum(r["fallbacks"] for r in mr),
            "finish_counts": {k: sum(1 for r in mr if r["finish"] == k)
                              for k in sorted({r["finish"] for r in mr})},
            "ambiguous": sum(1 for r in mr if r["ambiguous"]),
            "verifier_tokens_total": sum(r["verifier_tokens"] for r in mr),
            "verifier_tokens_mean": statistics.mean(r["verifier_tokens"] for r in mr),
        }
    if "per_step" in summary and "mur" in summary and summary["per_step"]["mean_total_tokens"]:
        base = summary["per_step"]["mean_total_tokens"]
        summary["mur"]["token_savings_vs_per_step"] = (
            1 - summary["mur"]["mean_total_tokens"] / base)
        summary["mur"]["accuracy_delta_vs_per_step"] = (
            summary["mur"]["accuracy"] - summary["per_step"]["accuracy"])
    metrics = {
        "schema_version": 1,
        "run_id": args.run_id,
        "split": args.split,
        "rows": rows,
        "summary": summary,
        "verifier": sources["verifier"],
        "verifier_cache_deduplicated_pending": len(todo),
        "missing_verifications": missing,
        "test_manifest_sha256": (testdata.sha256_file(Path(args.test_manifest))
                                 if args.test_manifest else None),
        "test_authorization": authorization,
    }
    write_json_atomic(out / "metrics.json", metrics)
    write_json_atomic(out / "verifier-summary.json", summary)
    status = {
        "schema_version": 1,
        "generated_tokens": budget.generated,
        "wall_seconds": budget.elapsed,
        "max_generated_tokens": budget.max_generated,
        "max_wall_seconds": budget.max_wall_seconds,
        "max_model_len": budget.max_model_len,
        "split": args.split,
        "prefix_caching": False,
        "trajectories": len(traces),
        "verified_rows": len(rows),
        "unique_pending_prompts": len(todo),
        "missing_verifications": missing,
        "verified_now": verified_now,
        "failure": failure,
        "exit_code": exit_code,
        "partial": len(rows) != len(traces),
    }
    write_json_atomic(out / "run-status.json", status)
    print(json.dumps(summary, indent=2))
    print(json.dumps(status), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
