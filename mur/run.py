"""Three-method stepwise runner. Real execution is remote-only and budget bounded."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .control import Ledger, Momentum, select_judgment
from .prompts import (analyze_prompt, critic_prompt, extract_answer, finalize_prompt,
                      judgment_prompt, policy_prompt, request_seed, strip_step_prefix,
                      to_step_string)
from .backend import VLLMBackend, judgment_from_generation


class BudgetExceeded(RuntimeError):
    pass


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_root() -> Path:
    return Path(os.environ.get("MUR_MODEL_ROOT") or
                (Path(os.environ.get("MUR_RUNTIME_ROOT", Path.cwd() / "runtime")) / "models"))


def data_root() -> Path:
    return Path(os.environ.get("MUR_RUNTIME_ROOT", Path.cwd() / "runtime")) / "data"


def model_path(repo: str) -> Path:
    return model_root() / repo.split("/")[-1]


def load_questions(split: str, limit: int | None) -> list[dict]:
    if split != "dev":
        raise SystemExit("this task only permits --split dev")
    path = data_root() / "dev.jsonl"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def run_one(question: dict, method: str, cfg: dict, policy: VLLMBackend,
            critic: VLLMBackend | None, out_path: Path, budget: dict) -> dict:
    qid = question["id"]
    seed = int(cfg["seed"])
    max_steps = int(cfg["max_steps"])
    max_step_tokens = int(cfg["max_step_tokens"])
    max_accepted = int(cfg["max_accepted_tokens"])
    temperature = float(cfg["temperature"])
    top_p = float(cfg["top_p"])
    top_k = int(cfg["top_k"])
    presence = float(cfg["presence_penalty"])
    momentum = Momentum(alpha=float(cfg["alpha"]), gamma=float(cfg["gamma"]),
                        bias_correction=bool(cfg["bias_correction"]))
    ledger = Ledger()
    trajectory: list[str] = []
    step_records: list[dict] = []
    accepted_tokens = 0
    finish = "step_cap"
    error = None
    started = time.time()

    def spend(role, phase, inp, out):
        ledger.record(role, phase, inp, out)
        budget["generated"] += out if role in ("policy", "critic") else 0
        if budget["generated"] > budget["max_generated"]:
            raise BudgetExceeded(f"generated token budget exceeded: {budget['generated']}")
        if time.time() - budget["wall_start"] > budget["max_wall"]:
            raise BudgetExceeded("wall budget exceeded")

    def policy_call(prompt, max_tokens, s, stop):
        gen = policy.generate(prompt, max_tokens=max_tokens, temperature=temperature,
                              top_p=top_p, top_k=top_k, presence_penalty=presence,
                              seed=s, stop=stop)
        return gen

    def critic_verify_batch(candidate_texts, step_idx):
        assert critic is not None
        aps = [analyze_prompt(critic.tokenizer,
                              critic_prompt(question["problem"], trajectory, c), step_idx)
               for c in candidate_texts]
        akw = [dict(max_tokens=int(cfg["critic_analysis_tokens"]),
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    presence_penalty=presence,
                    seed=request_seed(seed, qid, step_idx, "critic_analyze", ci),
                    stop=["</analyze>\n", "```python"],
                    include_stop_str_in_output=True) for ci in range(len(aps))]
        analyses = critic.generate_batch(aps, akw)
        for a in analyses:
            spend("critic", "analysis", a.prompt_tokens, len(a.token_ids))
        jps = [judgment_prompt(ap, a.text) for ap, a in zip(aps, analyses)]
        jkw = [dict(max_tokens=int(cfg["critic_output_tokens"]),
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    presence_penalty=presence,
                    seed=request_seed(seed, qid, step_idx, "critic_judgment", ci),
                    stop=["</output>\n", "</think>\n", "```python"],
                    include_stop_str_in_output=True, logprobs=1)
               for ci in range(len(jps))]
        judgments = critic.generate_batch(jps, jkw)
        for j in judgments:
            spend("critic", "judgment", j.prompt_tokens, len(j.token_ids))
        return [{
            "analysis": a.text,
            "judgment_text": j.text,
            "judgment": judgment_from_generation(j),
            "analyze_tokens": len(a.token_ids),
            "judgment_tokens": len(j.token_ids),
        } for a, j in zip(analyses, judgments)]

    try:
        for step_idx in range(max_steps):
            remaining = max_accepted - accepted_tokens
            if remaining <= 0:
                finish = "accepted_token_cap"
                break
            prompt = policy_prompt(policy.tokenizer, question["problem"], trajectory, step_idx)
            prompt_ids = policy.encode(prompt)
            proposal = policy_call(prompt, min(max_step_tokens, remaining),
                                   request_seed(seed, qid, step_idx, "proposal", 0),
                                   ["\nStep"])
            spend("policy", "proposal", proposal.prompt_tokens, len(proposal.token_ids))
            proposal_text = strip_step_prefix(proposal.text)
            rec = {"step": step_idx, "proposal": proposal_text,
                   "proposal_finish": proposal.finish_reason,
                   "triggered": False, "candidates": [], "selected": 0,
                   "fallback": False, "accepted_nll": None,
                   "score": None}
            accepted = proposal
            accepted_text = proposal_text
            accepted_ids = list(proposal.token_ids)
            proposal_nll = None
            triggered = False
            if method == "mur":
                sc = policy.score_nll(prompt_ids, proposal.token_ids)
                spend("policy", "proposal_score", len(prompt_ids), sc["probe_tokens"])
                proposal_nll = sc["mean_nll"]
                rec["score"] = sc
                triggered = momentum.trigger(proposal_nll)
                rec["triggered"] = triggered
            if method == "per_step" or (method == "mur" and triggered):
                cand_prompt = prompt
                ckw = [dict(max_tokens=min(max_step_tokens, remaining),
                            temperature=temperature, top_p=top_p, top_k=top_k,
                            presence_penalty=presence,
                            seed=request_seed(seed, qid, step_idx, "candidate", ci),
                            stop=["\nStep"]) for ci in range(int(cfg["candidates"]))]
                cgen = policy.generate_batch([cand_prompt] * len(ckw), ckw)
                for g in cgen:
                    spend("policy", "candidate", g.prompt_tokens, len(g.token_ids))
                judgments = critic_verify_batch(
                    [strip_step_prefix(g.text) for g in cgen], step_idx)
                sel, fallback = select_judgment(
                    [{"label": v["judgment"]["label"],
                      "probability": v["judgment"]["probability"]} for v in judgments])
                rec["candidates"] = [{"text": strip_step_prefix(g.text),
                                      "finish": g.finish_reason,
                                      "tokens": len(g.token_ids),
                                      "verify": v} for g, v in zip(cgen, judgments)]
                rec["selected"] = sel
                rec["fallback"] = fallback
                accepted = cgen[sel]
                accepted_text = strip_step_prefix(accepted.text)
                accepted_ids = list(accepted.token_ids)
            if method == "mur":
                if triggered:
                    sc2 = policy.score_nll(prompt_ids, accepted_ids)
                    spend("policy", "accepted_score", len(prompt_ids), sc2["probe_tokens"])
                    accepted_nll = sc2["mean_nll"]
                else:
                    accepted_nll = proposal_nll
                rec["accepted_nll"] = accepted_nll
                momentum.accept(accepted_nll)
            trajectory.append(to_step_string(step_idx, accepted_text))
            accepted_tokens += len(accepted_ids)
            rec["accepted_text"] = accepted_text
            rec["accepted_tokens"] = len(accepted_ids)
            step_records.append(rec)
            if "the answer is" in accepted_text.lower():
                finish = "answer"
                break
            if accepted_tokens >= max_accepted:
                finish = "accepted_token_cap"
                break
        if finish == "step_cap" and accepted_tokens < max_accepted:
            remaining = max_accepted - accepted_tokens
            fp = finalize_prompt(policy.tokenizer, question["problem"], trajectory)
            g = policy_call(fp, min(max_step_tokens, remaining),
                            request_seed(seed, qid, 999, "final_fallback", 0), None)
            spend("policy", "final_fallback", g.prompt_tokens, len(g.token_ids))
            trajectory.append(to_step_string(len(step_records), strip_step_prefix(g.text)))
            accepted_tokens += len(g.token_ids)
            step_records.append({"step": len(step_records), "final_fallback": True,
                                 "accepted_text": strip_step_prefix(g.text),
                                 "accepted_tokens": len(g.token_ids)})
            finish = "final_fallback"
    except BudgetExceeded as e:
        finish = "budget_stopped"
        error = str(e)
    except Exception as e:
        finish = "error"
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    totals = {
        "policy_output_tokens": ledger.total("policy"),
        "critic_output_tokens": ledger.total("critic"),
        "total_output_tokens": ledger.total(),
        "policy_input_tokens": ledger.total("policy", "input_tokens"),
        "critic_input_tokens": ledger.total("critic", "input_tokens"),
        "calls": len(ledger.calls),
    }
    record = {
        "run_id": cfg.get("run_id", "mur-20261024-prep"),
        "method": method,
        "qid": qid,
        "subject": question.get("subject"),
        "seed": seed,
        "split": "dev",
        "steps": step_records,
        "trajectory": "\n".join(trajectory),
        "final_answer": extract_answer("\n".join(trajectory)),
        "accepted_tokens": accepted_tokens,
        "finish": finish,
        "error": error,
        "ledger": ledger.calls,
        "totals": totals,
        "wall_seconds": time.time() - started,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default="mur-20261024-prep")
    args = ap.parse_args(argv)
    cfg = load_json(args.config)
    cfg["run_id"] = args.run_id
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    traces = outdir / "raw-smoke-traces.jsonl"
    if traces.exists():
        raise SystemExit(f"refusing to overwrite existing traces: {traces}")
    questions = load_questions(args.split, args.limit)
    with open(outdir / "run-config.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "methods": args.methods, "questions": len(questions),
                   "seed": cfg["seed"]}, f, indent=2)
    budget = {"generated": 0, "max_generated": 1_000_000,
              "wall_start": time.time(), "max_wall": 3300}
    policy = VLLMBackend(str(model_path(load_json("configs/sources.json")["policy"]["id"])),
                         gpu_memory_utilization=0.29, max_model_len=16384,
                         seed=int(cfg["seed"]))
    critic = None
    if any(m in ("per_step", "mur") for m in args.methods):
        critic = VLLMBackend(
            str(model_path(load_json("configs/sources.json")["critic"]["id"])),
            gpu_memory_utilization=0.38, max_model_len=16384, seed=int(cfg["seed"]))
    records = []
    try:
        for q in questions:
            for method in args.methods:
                if time.time() - budget["wall_start"] > budget["max_wall"]:
                    break
                rec = run_one(q, method, cfg, policy, critic, traces, budget)
                records.append(rec)
                print(f"{method} {q['id']} finish={rec['finish']} "
                      f"tokens={rec['totals']['total_output_tokens']} "
                      f"accepted={rec['accepted_tokens']}", flush=True)
    finally:
        summary = {
            "trajectories": len(records),
            "generated_tokens": budget["generated"],
            "wall_seconds": time.time() - budget["wall_start"],
            "budget_max_generated": budget["max_generated"],
            "budget_max_wall": budget["max_wall"],
            "partial": len(records) < len(questions) * len(args.methods),
            "methods": args.methods,
            "questions": len(questions),
        }
        with open(outdir / "run-status.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
