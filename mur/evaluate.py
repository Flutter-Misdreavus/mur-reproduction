"""Frozen-solution evaluation with TIGER-Lab/general-verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from .backend import VLLMBackend
from .prompts import parse_decision, request_seed, verifier_prompt


def model_path(repo: str) -> Path:
    root = Path(os.environ.get("MUR_MODEL_ROOT") or
                (Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime")) / "models"))
    return root / repo.split("/")[-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = json.load(open(args.sources, encoding="utf-8"))
    seed = args.seed
    dev = {}
    root = Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime"))
    for line in (root / "data" / "dev.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); dev[r["id"]] = r
    traces = []
    for line in (inp / "raw-smoke-traces.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))

    cache_path = root / "cache" / "verifier.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); cache[r["key"]] = r
    todo = []
    for t in traces:
        q = dev[t["qid"]]
        prompt = verifier_prompt(q["problem"], q["answer"], t.get("final_answer", ""))
        key_payload = {"model": sources["verifier"], "prompt": prompt, "temperature": 0.7,
                       "max_tokens": 4096, "seed": request_seed(seed, t["qid"], 0, "verifier", 0)}
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        t["_prompt"], t["_key"] = prompt, key
        if key not in cache:
            todo.append(t)

    if todo:
        backend = VLLMBackend(str(model_path(sources["verifier"]["id"])),
                              gpu_memory_utilization=0.45, max_model_len=12288, seed=seed)
        from vllm import SamplingParams
        params = [SamplingParams(max_tokens=4096, temperature=0.7, n=1,
                                 seed=request_seed(seed, t["qid"], 0, "verifier", 0))
                  for t in todo]
        gens = backend.generate_many([t["_prompt"] for t in todo], params)
        with open(cache_path, "a", encoding="utf-8") as cf:
            for t, g in zip(todo, gens):
                rec = {"key": t["_key"], "text": g.text, "finish": g.finish_reason,
                       "tokens": len(g.token_ids), "prompt_tokens": g.prompt_tokens}
                cache[t["_key"]] = rec
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        backend.close()

    rows = []
    for t in traces:
        c = cache[t["_key"]]
        label, ambiguous = parse_decision(c["text"])
        rows.append({
            "qid": t["qid"], "method": t["method"], "finish": t["finish"],
            "verifier_label": label, "ambiguous": ambiguous,
            "correct": int(label == "Yes" and not ambiguous and t["finish"] not in ("error", "budget_stopped")),
            "accepted_tokens": t["accepted_tokens"],
            "policy_output_tokens": t["totals"]["policy_output_tokens"],
            "critic_output_tokens": t["totals"]["critic_output_tokens"],
            "total_output_tokens": t["totals"]["total_output_tokens"],
            "steps": len(t["steps"]),
            "triggers": sum(1 for s in t["steps"] if s.get("triggered")),
            "malformed": sum(1 for s in t["steps"] for c2 in s.get("candidates", [])
                             if c2["verify"]["judgment"]["label"] is None),
            "wall_seconds": t["wall_seconds"],
            "verifier_tokens": c["tokens"],
        })
    summary = {}
    for m in sorted({r["method"] for r in rows}):
        mr = [r for r in rows if r["method"] == m]
        summary[m] = {
            "n": len(mr),
            "accuracy": sum(r["correct"] for r in mr) / len(mr),
            "mean_total_tokens": statistics.mean(r["total_output_tokens"] for r in mr),
            "mean_policy_tokens": statistics.mean(r["policy_output_tokens"] for r in mr),
            "mean_critic_tokens": statistics.mean(r["critic_output_tokens"] for r in mr),
            "mean_steps": statistics.mean(r["steps"] for r in mr),
            "trigger_rate": (sum(r["triggers"] for r in mr) /
                             max(1, sum(max(1, r["steps"] - 1) for r in mr))),
            "malformed": sum(r["malformed"] for r in mr),
            "finish_counts": {k: sum(1 for r in mr if r["finish"] == k)
                              for k in sorted({r["finish"] for r in mr})},
            "ambiguous": sum(1 for r in mr if r["ambiguous"]),
        }
    if "per_step" in summary and "mur" in summary and summary["per_step"]["mean_total_tokens"]:
        base = summary["per_step"]["mean_total_tokens"]
        summary["mur"]["token_savings_vs_per_step"] = 1 - summary["mur"]["mean_total_tokens"] / base
        summary["mur"]["accuracy_delta_vs_per_step"] = (summary["mur"]["accuracy"] -
                                                        summary["per_step"]["accuracy"])
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary, "verifier": sources["verifier"]}, f, indent=2)
    with open(out / "verifier-summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
