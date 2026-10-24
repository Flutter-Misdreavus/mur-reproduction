"""Aggregate the four frozen test chunks, paired bootstrap and blind audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

from . import testdata
from .evaluate import trigger_stats


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile; q in [0,1]."""
    if not values:
        raise ValueError("empty percentile sample")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def paired_bootstrap(per_q: dict[str, dict[str, dict]], seed: int = 20261024,
                     replicates: int = 10000) -> dict:
    qids = sorted(per_q)
    n = len(qids)
    if n == 0:
        raise ValueError("no paired questions")
    for qid in qids:
        for method in ("cot", "per_step", "mur"):
            if method not in per_q[qid]:
                raise ValueError(f"missing {method} for {qid}")
    rng = random.Random(int(seed))
    savings, acc_delta, mur_acc, per_acc, cot_acc = [], [], [], [], []
    for _ in range(int(replicates)):
        sample = [qids[rng.randrange(n)] for _ in range(n)]
        per_tokens = sum(per_q[q]["per_step"]["tokens"] for q in sample) / n
        mur_tokens = sum(per_q[q]["mur"]["tokens"] for q in sample) / n
        cot_tokens = sum(per_q[q]["cot"]["tokens"] for q in sample) / n
        savings.append(1 - mur_tokens / per_tokens)
        pd = sum(per_q[q]["per_step"]["correct"] for q in sample) / n
        md = sum(per_q[q]["mur"]["correct"] for q in sample) / n
        cd = sum(per_q[q]["cot"]["correct"] for q in sample) / n
        acc_delta.append(sum(per_q[q]["mur"]["correct"] -
                             per_q[q]["per_step"]["correct"] for q in sample) / n)
        mur_acc.append(md)
        per_acc.append(pd)
        cot_acc.append(cd)
    point_per = statistics.mean(per_q[q]["per_step"]["tokens"] for q in qids)
    point_mur = statistics.mean(per_q[q]["mur"]["tokens"] for q in qids)
    point_cot = statistics.mean(per_q[q]["cot"]["tokens"] for q in qids)
    point_savings = 1 - point_mur / point_per
    point_delta = sum(per_q[q]["mur"]["correct"] -
                      per_q[q]["per_step"]["correct"] for q in qids) / n
    return {
        "seed": int(seed),
        "replicates": int(replicates),
        "n_paired": n,
        "point": {
            "mean_cot_tokens": point_cot,
            "mean_per_step_tokens": point_per,
            "mean_mur_tokens": point_mur,
            "token_savings_vs_per_step": point_savings,
            "accuracy_cot": statistics.mean(per_q[q]["cot"]["correct"] for q in qids),
            "accuracy_per_step": statistics.mean(
                per_q[q]["per_step"]["correct"] for q in qids),
            "accuracy_mur": statistics.mean(per_q[q]["mur"]["correct"] for q in qids),
            "accuracy_delta_vs_per_step": point_delta,
        },
        "token_savings_ci95": [percentile(savings, 0.025), percentile(savings, 0.975)],
        "accuracy_delta_ci95": [percentile(acc_delta, 0.025),
                                percentile(acc_delta, 0.975)],
        "accuracy_mur_ci95": [percentile(mur_acc, 0.025), percentile(mur_acc, 0.975)],
        "accuracy_per_step_ci95": [percentile(per_acc, 0.025),
                                   percentile(per_acc, 0.975)],
        "accuracy_cot_ci95": [percentile(cot_acc, 0.025), percentile(cot_acc, 0.975)],
    }


def decision_from(paired: dict) -> dict:
    point = paired["point"]
    savings = point["token_savings_vs_per_step"]
    delta = point["accuracy_delta_vs_per_step"]
    lower = paired["accuracy_delta_ci95"][0]
    point_met = bool(savings >= 0.40 and delta >= -0.02)
    non_inferior = bool(lower > -0.02)
    return {
        "target_token_savings_ge_0_40": bool(savings >= 0.40),
        "target_accuracy_delta_ge_minus_0_02": bool(delta >= -0.02),
        "point_target_met": point_met,
        "non_inferiority_lower_bound_gt_minus_0_02": non_inferior,
        "evidence_supported_non_inferiority": bool(point_met and non_inferior),
        "status": ("point_target_and_non_inferiority_met" if point_met and non_inferior
                   else "inconclusive_non_inferiority" if point_met
                   else "point_target_not_met"),
        "note": ("A fixed 100-question exploratory result does not establish a complete "
                 "MATH-500 reproduction."),
    }


def aggregate_rows(rows: list[dict], manifest: dict) -> tuple[dict, dict[str, dict]]:
    expected_ids = [q["id"] for q in manifest["questions"]]
    by_method: dict[str, dict[str, dict]] = {m: {} for m in ("cot", "per_step", "mur")}
    for row in rows:
        method = row.get("method")
        qid = row.get("qid")
        if method not in by_method:
            raise ValueError(f"unknown method {method!r}")
        if qid not in set(expected_ids):
            raise ValueError(f"unknown test ID {qid!r}")
        if qid in by_method[method]:
            raise ValueError(f"duplicate row {method}/{qid}")
        by_method[method][qid] = row
    for method in by_method:
        missing = set(expected_ids) - set(by_method[method])
        if missing:
            raise ValueError(f"{method} missing {len(missing)} questions")
    per_q = {}
    for qid in expected_ids:
        per_q[qid] = {
            "cot": {"tokens": by_method["cot"][qid]["total_output_tokens"],
                    "correct": int(by_method["cot"][qid]["correct"])},
            "per_step": {"tokens": by_method["per_step"][qid]["total_output_tokens"],
                         "correct": int(by_method["per_step"][qid]["correct"])},
            "mur": {"tokens": by_method["mur"][qid]["total_output_tokens"],
                    "correct": int(by_method["mur"][qid]["correct"])},
        }
    summary = {}
    for method in ("cot", "per_step", "mur"):
        mr = [by_method[method][q] for q in expected_ids]
        ts = trigger_stats(method, mr)
        summary[method] = {
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
            "fallbacks": sum(r.get("fallbacks", 0) for r in mr),
            "malformed_critic_judgments": sum(r.get("malformed", 0) for r in mr),
            "ambiguous_verifier_labels": sum(1 for r in mr if r.get("ambiguous")),
            "finish_counts": {k: sum(1 for r in mr if r["finish"] == k)
                              for k in sorted({r["finish"] for r in mr})},
            "context_limit_count": sum(1 for r in mr if r["finish"] == "context_limit"),
            "budget_stop_count": sum(1 for r in mr if r["finish"] == "budget_stopped"),
            "error_count": sum(1 for r in mr if r["finish"] == "error"),
            "verifier_tokens_attributed_total": sum(r["verifier_tokens"] for r in mr),
            "verifier_tokens_mean": statistics.mean(r["verifier_tokens"] for r in mr),
            "graded_rows": len(mr),
        }
    cot = summary["cot"]
    per = summary["per_step"]
    mur = summary["mur"]
    summary["mur"]["token_savings_vs_per_step"] = (
        1 - mur["mean_total_tokens"] / per["mean_total_tokens"])
    summary["mur"]["accuracy_delta_vs_per_step"] = mur["accuracy"] - per["accuracy"]
    summary["cot"]["token_savings_vs_per_step"] = (
        1 - cot["mean_total_tokens"] / per["mean_total_tokens"])
    summary["cot"]["accuracy_delta_vs_per_step"] = cot["accuracy"] - per["accuracy"]
    return summary, per_q


def build_blind_audit(rows: list[dict], source_rows: list[dict], manifest: dict,
                      final_answers: dict, seed: int = 20261024) -> tuple[list[dict], dict, dict]:
    by_q: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_q.setdefault(row["qid"], {})[row["method"]] = row
    discordant = []
    for q in manifest["questions"]:
        qid = q["id"]
        methods = by_q.get(qid, {})
        if len({int(methods[m]["correct"]) for m in methods}) > 1:
            discordant.append(qid)
    rng = random.Random(int(seed))
    audit_rows = []
    mapping = {}
    for qid in discordant:
        methods = by_q[qid]
        labels = list(methods)
        rng.shuffle(labels)
        audit_id = "audit-" + hashlib.sha256(
            (qid + "|" + str(seed)).encode("utf-8")).hexdigest()[:12]
        index = int(next(q["index"] for q in manifest["questions"] if q["id"] == qid))
        source = source_rows[index]
        candidates = {}
        method_map = {}
        for label, method in zip("ABC", labels):
            candidates[label] = {
                "final_answer": final_answers.get((qid, method), ""),
                "finish": methods[method].get("finish"),
            }
            method_map[label] = method
        audit_rows.append({
            "schema_version": 1,
            "audit_id": audit_id,
            "qid": qid,
            "source_index": index,
            "problem": source["problem"],
            "gold_answer": source["answer"],
            "candidates": candidates,
            "instructions": ("Judge answer equivalence only; do not infer method from "
                             "ordering. Record audited_label per candidate before "
                             "revealing the mapping."),
            "audited_labels": {label: None for label in candidates},
            "audit_status": "pending_supervisor",
        })
        mapping[audit_id] = {
            "qid": qid,
            "source_index": index,
            "labels": method_map,
            "verifier_labels": {method: methods[method].get("verifier_label")
                                for method in methods},
            "automated_correct": {method: int(methods[method]["correct"])
                                  for method in methods},
        }
    summary = {
        "schema_version": 1,
        "seed": int(seed),
        "n_method_discordant_questions": len(discordant),
        "audit_ids": [r["audit_id"] for r in audit_rows],
        "status": "pending_supervisor",
        "note": ("Labels were assigned by a seeded random permutation per question. "
                 "The mapping file is separate; no verifier label was changed."),
        "method_discordant_qids_sha256": hashlib.sha256(
            "\n".join(discordant).encode()).hexdigest(),
    }
    return audit_rows, mapping, summary


def verifier_cost(chunk_status: list[dict], summary: dict) -> dict:
    """Separate actual cache-deduplicated verifier calls from attributed tokens.

    ``summary[m]["verifier_tokens_total"]`` attributes a shared generation to every
    row that reuses its cache entry, so it is reported separately from the tokens
    actually generated by the verifier process.
    """
    evaluations = [c.get("evaluation") or {} for c in chunk_status]
    return {
        "input_tokens": None,
        "output_tokens_total": sum(int(e.get("generated_tokens", 0))
                                   for e in evaluations),
        "attributed_output_tokens_total": sum(
            s["verifier_tokens_attributed_total"] for s in summary.values()),
        "calls": sum(int(e.get("verified_now", 0)) for e in evaluations),
        "unique_pending_prompts": sum(
            int(e.get("unique_pending_prompts", 0)) for e in evaluations),
        "per_method_output_tokens": {m: summary[m]["verifier_tokens_attributed_total"]
                                     for m in ("cot", "per_step", "mur")},
        "wall_seconds": sum(e.get("wall_seconds", 0) for e in evaluations),
        "cost_role": "separate from policy+critic generation token metric",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--replicates", type=int, default=10000)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    task = load_json(Path(args.task_file), {})
    run_id = task["run_id"]
    sources = load_json(Path(args.sources))
    manifest = load_json(Path(args.test_manifest))
    root = Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime"))
    source_rows, _ = testdata.load_source_rows(root, sources)
    problems = testdata.verify_test_manifest(manifest, source_rows, sources)
    if problems:
        raise ValueError("test manifest verification failed: " + "; ".join(problems))

    final_answers = {}
    raw_trace_path = run_dir / "raw-traces.jsonl"
    raw_lines = []
    if raw_trace_path.exists():
        raw_lines = [json.loads(x) for x in raw_trace_path.read_text(
            encoding="utf-8").splitlines() if x.strip()]
    rows = []
    chunk_status = []
    for i in range(4):
        chunk = run_dir / "chunks" / f"chunk-{i:02d}"
        metrics = load_json(chunk / "evaluation" / "metrics.json")
        if not metrics:
            raise ValueError(f"missing evaluation metrics for chunk {i}")
        rows.extend(metrics.get("rows") or [])
        for trace in (chunk / "generation" / "raw-smoke-traces.jsonl").read_text(
                encoding="utf-8").splitlines():
            if trace.strip():
                rec = json.loads(trace)
                final_answers[(rec["qid"], rec["method"])] = rec.get("final_answer", "")
        gen_phase = load_json(
            chunk / "generation" / f"phase-chunk-{i:02d}-generate.json", {}) or {}
        eval_phase = load_json(
            chunk / "evaluation" / f"phase-chunk-{i:02d}-evaluate.json", {}) or {}
        chunk_status.append({
            "chunk": i,
            "generation": load_json(chunk / "generation" / "run-status.json"),
            "evaluation": load_json(chunk / "evaluation" / "run-status.json"),
            "generation_commit": gen_phase.get("run_commit"),
            "evaluation_commit": eval_phase.get("run_commit"),
            "rows": len(metrics.get("rows") or []),
        })
    summary, per_q = aggregate_rows(rows, manifest)
    paired = paired_bootstrap(per_q, args.seed, args.replicates)
    decision = decision_from(paired)
    audit_rows, mapping, audit_summary = build_blind_audit(
        rows, source_rows, manifest, final_answers, args.seed)

    verifier = verifier_cost(chunk_status, summary)
    generation_tokens = sum((c["generation"] or {}).get("generated_tokens", 0)
                            for c in chunk_status)
    generation_wall = sum((c["generation"] or {}).get("wall_seconds", 0)
                          for c in chunk_status)
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "fixed100_exploratory_test",
        "split": "test",
        "n_questions": len(manifest["questions"]),
        "methods": ["cot", "per_step", "mur"],
        "primary_metric": "mean policy+critic API-returned generated tokens",
        "summary": summary,
        "paired_bootstrap": paired,
        "decision": decision,
        "verifier": verifier,
        "counts": {
            "rows": len(rows),
            "expected_rows": 300,
            "per_method_rows": {m: summary[m]["n"] for m in summary},
            "truncation": {m: summary[m]["finish_counts"].get("accepted_token_cap", 0)
                           for m in summary},
            "step_cap": {m: summary[m]["finish_counts"].get("step_cap", 0)
                         for m in summary},
            "context_limit": {m: summary[m]["context_limit_count"] for m in summary},
            "budget_stops": {m: summary[m]["budget_stop_count"] for m in summary},
            "errors": {m: summary[m]["error_count"] for m in summary},
            "ambiguous_verifier": {m: summary[m]["ambiguous_verifier_labels"]
                                   for m in summary},
            "final_fallback": {m: summary[m]["fallbacks"] for m in summary},
        },
        "test_manifest_sha256": testdata.sha256_file(Path(args.test_manifest)),
        "source_sha256": sources["test"]["sha256"],
        "rows": [{k: r.get(k) for k in (
            "qid", "method", "split", "source_index", "finish", "verifier_label",
            "ambiguous", "correct", "correct_automated", "audited_label",
            "audit_status", "accepted_tokens", "policy_output_tokens",
            "critic_output_tokens", "total_output_tokens", "policy_input_tokens",
            "critic_input_tokens", "steps", "triggers", "fallbacks", "malformed",
            "verifier_tokens", "wall_seconds")} for r in rows],
        "chunks": chunk_status,
        "resource": {
            "generation_generated_tokens": generation_tokens,
            "generation_wall_seconds": generation_wall,
            "verifier_output_tokens": verifier["output_tokens_total"],
            "verifier_wall_seconds": verifier["wall_seconds"],
        },
        "audit": audit_summary,
        "limitations": [
            "Fixed 100-question exploratory subset; not a full MATH-500 result.",
            "API-returned tokens are a compute proxy, not exact GPU FLOPs.",
            "Latency on a shared GPU is descriptive, not a controlled speedup.",
            "One seed only; no tuning or outcome-based stopping.",
        ],
    }
    testdata.write_json_atomic(run_dir / "metrics.json", metrics)
    with open(run_dir / "blind-audit.jsonl", "w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    testdata.write_json_atomic(run_dir / "blind-audit-mapping.json",
                               {"schema_version": 1, "run_id": run_id,
                                "created_after_labels_fixed": True,
                                "mapping": mapping})
    testdata.write_json_atomic(run_dir / "blind-audit-summary.json", audit_summary)
    testdata.write_json_atomic(run_dir / "aggregate-status.json", {
        "schema_version": 1, "run_id": run_id, "rows": len(rows),
        "chunks": chunk_status, "verifier": verifier,
        "generation_generated_tokens": generation_tokens,
        "generation_wall_seconds": generation_wall,
    })

    # Sanitized tracked copy: numeric rows only, no question or answer text.
    public = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "fixed100_exploratory_test",
        "split": "test",
        "n_questions": len(manifest["questions"]),
        "summary": summary,
        "paired_bootstrap": paired,
        "decision": decision,
        "verifier": verifier,
        "counts": metrics["counts"],
        "resource": metrics["resource"],
        "audit": audit_summary,
        "test_manifest_sha256": metrics["test_manifest_sha256"],
        "source_sha256": metrics["source_sha256"],
        "rows": metrics["rows"],
        "limitations": metrics["limitations"],
    }
    exp = Path("experiments") / run_id
    testdata.write_json_atomic(exp / "metrics.json", public)
    testdata.write_json_atomic(exp / "blind-audit-summary.json", audit_summary)
    print(json.dumps({"run_id": run_id, "decision": decision,
                      "summary": summary, "paired_bootstrap": paired}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
