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
from .limits import ContextLimit, StageBudget, StageBudgetExceeded
from .prompts import (analyze_prompt, critic_prompt, extract_answer, finalize_prompt, has_final_answer,
                      judgment_prompt, policy_prompt, request_seed, strip_step_prefix,
                      to_step_string)
from .backend import VLLMBackend, judgment_from_generation, generation_nll
from . import testdata


# Backwards-compatible name used by older callers/tests.
BudgetExceeded = StageBudgetExceeded


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def model_root() -> Path:
    return Path(os.environ.get("MUR_MODEL_ROOT") or
                (Path(os.environ.get("MUR_RUNTIME_ROOT", Path.cwd() / "runtime")) / "models"))


def data_root() -> Path:
    return Path(os.environ.get("MUR_RUNTIME_ROOT", Path.cwd() / "runtime")) / "data"


def model_path(repo: str) -> Path:
    return model_root() / repo.split("/")[-1]


def load_questions(split: str, limit: int | None,
                   test_manifest: str | None = None) -> list[dict]:
    if split == "dev":
        path = data_root() / "dev.jsonl"
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif split == "test":
        if not test_manifest:
            raise SystemExit("--split test requires --test-manifest")
        manifest = load_json(test_manifest)
        runtime_root = Path(os.environ.get("MUR_RUNTIME_ROOT", Path.cwd() / "runtime"))
        sources = load_json("configs/sources.json")
        expected = testdata.load_source_rows(runtime_root, sources)[0]
        problems = testdata.verify_test_manifest(manifest, expected, sources)
        if problems:
            raise SystemExit("test manifest verification failed: " + "; ".join(problems))
        rows = testdata.load_test_generation_file(runtime_root, manifest)
    else:
        raise SystemExit(f"unsupported split: {split}")
    if limit:
        rows = rows[:limit]
    return rows


def run_one(question: dict, method: str, cfg: dict, policy: VLLMBackend,
            critic: VLLMBackend | None, out_path: Path, budget: StageBudget) -> dict:
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
    context_error = None
    abort_budget = None
    started = time.time()
    force_trigger_step = cfg.get("_force_trigger_step")
    engineering_forced = force_trigger_step is not None

    def spend(role, phase, inp, out):
        ledger.record(role, phase, inp, out)
        budget.account(role, out)

    def policy_call(prompt, prompt_len, desired_tokens, s, stop, label):
        effective = budget.fit(prompt_len, desired_tokens, label)
        budget.reserve(effective, label)
        gen = policy.generate(prompt, max_tokens=effective, temperature=temperature,
                              top_p=top_p, top_k=top_k, presence_penalty=presence,
                              seed=s, stop=stop, logprobs=0)
        spend("policy", label, gen.prompt_tokens, len(gen.token_ids))
        return gen, effective

    def policy_batch(prompts, desired_tokens, seeds, label):
        prompt_lens = [len(policy.encode(p)) for p in prompts]
        effective = [budget.fit(n, desired_tokens, label) for n in prompt_lens]
        budget.reserve(effective, label)
        ckw = [dict(max_tokens=e, temperature=temperature, top_p=top_p,
                    top_k=top_k, presence_penalty=presence, seed=s,
                    stop=["\nStep"], logprobs=0)
               for e, s in zip(effective, seeds)]
        gens = policy.generate_batch(prompts, ckw)
        for g in gens:
            spend("policy", label, g.prompt_tokens, len(g.token_ids))
        return gens, effective

    def critic_verify_batch(candidate_texts, step_idx):
        assert critic is not None
        aps = [analyze_prompt(critic.tokenizer,
                              critic_prompt(question["problem"], trajectory, c), step_idx)
               for c in candidate_texts]
        a_lens = [len(critic.encode(p)) for p in aps]
        a_eff = [budget.fit(n, int(cfg["critic_analysis_tokens"]), "critic_analysis")
                 for n in a_lens]
        budget.reserve(a_eff, "critic_analysis")
        akw = [dict(max_tokens=e,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    presence_penalty=presence,
                    seed=request_seed(seed, qid, step_idx, "critic_analyze", ci),
                    stop=["</analyze>\n", "```python"],
                    include_stop_str_in_output=True) for ci, e in enumerate(a_eff)]
        analyses = critic.generate_batch(aps, akw)
        for a in analyses:
            spend("critic", "analysis", a.prompt_tokens, len(a.token_ids))
        jps = [judgment_prompt(ap, a.text) for ap, a in zip(aps, analyses)]
        j_lens = [len(critic.encode(p)) for p in jps]
        j_eff = [budget.fit(n, int(cfg["critic_output_tokens"]), "critic_judgment")
                 for n in j_lens]
        budget.reserve(j_eff, "critic_judgment")
        jkw = [dict(max_tokens=e,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    presence_penalty=presence,
                    seed=request_seed(seed, qid, step_idx, "critic_judgment", ci),
                    stop=["</output>\n", "</think>\n", "```python"],
                    include_stop_str_in_output=True, logprobs=1)
               for ci, e in enumerate(j_eff)]
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
            desired = min(max_step_tokens, remaining)
            proposal, proposal_cap = policy_call(
                prompt, len(prompt_ids), desired,
                request_seed(seed, qid, step_idx, "proposal", 0),
                ["\nStep"], "proposal")
            proposal_text = strip_step_prefix(proposal.text)
            rec = {"step": step_idx, "proposal": proposal_text,
                   "proposal_token_ids": list(proposal.token_ids),
                   "proposal_finish": proposal.finish_reason,
                   "proposal_max_tokens": proposal_cap,
                   "triggered": False, "candidates": [], "selected": 0,
                   "fallback": False, "accepted_nll": None,
                   "score": None}
            accepted = proposal
            accepted_text = proposal_text
            accepted_ids = list(proposal.token_ids)
            proposal_nll = None
            triggered = False
            if method == "mur":
                sc = generation_nll(proposal)
                proposal_nll = sc["mean_nll"]
                rec["score"] = sc
                triggered = momentum.trigger(proposal_nll)
                rec["triggered"] = triggered
                rec["history_before"] = momentum.history
                rec["proposal_nll"] = proposal_nll
            if method == "per_step" or (method == "mur" and triggered):
                cand_prompt = prompt
                cand_seeds = [request_seed(seed, qid, step_idx, "candidate", ci)
                              for ci in range(int(cfg["candidates"]))]
                cgen, cand_caps = policy_batch(
                    [cand_prompt] * len(cand_seeds),
                    min(max_step_tokens, remaining), cand_seeds, "candidate")
                judgments = critic_verify_batch(
                    [strip_step_prefix(g.text) for g in cgen], step_idx)
                sel, fallback = select_judgment(
                    [{"label": v["judgment"]["label"],
                      "probability": v["judgment"]["probability"]} for v in judgments])
                rec["triggered"] = True
                rec["candidate_max_tokens"] = cand_caps
                rec["candidates"] = [{"text": strip_step_prefix(g.text),
                                      "finish": g.finish_reason,
                                      "tokens": len(g.token_ids),
                                      "max_tokens": cap,
                                      "verify": v} for g, v, cap in
                                     zip(cgen, judgments, cand_caps)]
                rec["selected"] = sel
                rec["fallback"] = fallback
                accepted = cgen[sel]
                accepted_text = strip_step_prefix(accepted.text)
                accepted_ids = list(accepted.token_ids)
            if method == "mur":
                if triggered:
                    sc2 = generation_nll(accepted)
                    accepted_nll = sc2["mean_nll"]
                else:
                    accepted_nll = proposal_nll
                rec["accepted_nll"] = accepted_nll
                momentum.accept(accepted_nll)
            trajectory.append(to_step_string(step_idx, accepted_text))
            accepted_tokens += len(accepted_ids)
            rec["accepted_text"] = accepted_text
            rec["accepted_token_ids"] = list(accepted_ids)
            rec["accepted_tokens"] = len(accepted_ids)
            step_records.append(rec)
            if has_final_answer(accepted_text):
                finish = "answer"
                break
            if accepted_ids[-1:] == [policy.tokenizer.eos_token_id]:
                finish = "eos"
                break
            if accepted_tokens >= max_accepted:
                finish = "accepted_token_cap"
                break
            if engineering_forced and step_idx == int(force_trigger_step) - 1:
                # Engineering-only branch exercise: deliberately make the
                # momentum estimate low so the next step is scaled.  Never
                # used in the scientific smoke.
                momentum.raw = -1.0
        if finish == "step_cap" and accepted_tokens < max_accepted:
            remaining = max_accepted - accepted_tokens
            fp = finalize_prompt(policy.tokenizer, question["problem"], trajectory)
            fp_ids = policy.encode(fp)
            g, cap = policy_call(fp, len(fp_ids), min(max_step_tokens, remaining),
                                 request_seed(seed, qid, 999, "final_fallback", 0),
                                 None, "final_fallback")
            trajectory.append(to_step_string(len(step_records), strip_step_prefix(g.text)))
            accepted_tokens += len(g.token_ids)
            step_records.append({"step": len(step_records), "final_fallback": True,
                                 "accepted_text": strip_step_prefix(g.text),
                                 "accepted_token_ids": list(g.token_ids),
                                 "accepted_tokens": len(g.token_ids),
                                 "max_tokens": cap})
            finish = "final_fallback"
    except ContextLimit as e:
        finish = "context_limit"
        context_error = str(e)
    except StageBudgetExceeded as e:
        finish = "budget_stopped"
        error = str(e)
        abort_budget = str(e)
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
        "split": cfg.get("_split", "dev"),
        "source_index": question.get("index"),
        "question_sha256": question.get("question_sha256"),
        "engineering_forced_trigger": engineering_forced,
        "steps": step_records,
        "trajectory": "\n".join(trajectory),
        "final_answer": extract_answer("\n".join(trajectory)),
        "accepted_tokens": accepted_tokens,
        "finish": finish,
        "error": error,
        "context_error": context_error,
        "ledger": ledger.calls,
        "totals": totals,
        "wall_seconds": time.time() - started,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if abort_budget is not None:
        raise StageBudgetExceeded(abort_budget)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--run-id", default="mur-20261024-prep")
    ap.add_argument("--max-generated-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-wall-seconds", type=float, default=3300)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--force-trigger-step", type=int, default=None)
    ap.add_argument("--test-manifest", default=None)
    ap.add_argument("--task-file", default=None)
    ap.add_argument("--acceptance-review", default=None)
    args = ap.parse_args(argv)
    cfg = load_json(args.config)
    cfg["run_id"] = args.run_id
    cfg["_split"] = args.split
    test_authorization = None
    if args.split == "test":
        if not (args.test_manifest and args.task_file and args.acceptance_review):
            raise SystemExit("--split test requires --test-manifest, --task-file and "
                             "--acceptance-review")
        if args.force_trigger_step is not None:
            raise SystemExit("forced engineering triggers are forbidden on test")
        test_authorization = testdata.authorize_test(
            Path(args.task_file), Path(args.acceptance_review), args.run_id)
    if args.force_trigger_step is not None:
        cfg["_force_trigger_step"] = args.force_trigger_step
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    traces = outdir / "raw-smoke-traces.jsonl"
    if traces.exists():
        raise SystemExit(f"refusing to overwrite existing traces: {traces}")
    questions = load_questions(args.split, None, args.test_manifest)
    questions = questions[args.offset:(args.offset + args.limit if args.limit else None)]
    if not questions:
        raise SystemExit("empty question slice")
    test_manifest_obj = load_json(args.test_manifest) if args.test_manifest else None
    write_json_atomic(outdir / "run-config.json",
                      {"config": cfg, "methods": args.methods, "questions": len(questions),
                       "question_ids": [q["id"] for q in questions],
                       "split": args.split, "seed": cfg["seed"],
                       "prefix_caching": False,
                       "test_manifest_sha256": (
                           testdata.sha256_file(Path(args.test_manifest))
                           if args.test_manifest else None),
                       "test_manifest_rows": (
                           test_manifest_obj.get("rows") if test_manifest_obj else None),
                       "test_authorization": test_authorization,
                       "limits": {
                           "max_generated_tokens": args.max_generated_tokens,
                           "max_wall_seconds": args.max_wall_seconds,
                           "max_model_len": args.max_model_len}})
    budget = StageBudget(max_generated=args.max_generated_tokens,
                         max_wall_seconds=args.max_wall_seconds,
                         max_model_len=args.max_model_len)
    setup_started = time.time()
    policy = None
    critic = None
    records = []
    failure = None
    exit_code = 0
    try:
        policy = VLLMBackend(
            str(model_path(load_json("configs/sources.json")["policy"]["id"])),
            gpu_memory_utilization=0.42, max_model_len=args.max_model_len,
            seed=int(cfg["seed"]), enable_prefix_caching=False)
        if any(m in ("per_step", "mur") for m in args.methods):
            critic = VLLMBackend(
                str(model_path(load_json("configs/sources.json")["critic"]["id"])),
                gpu_memory_utilization=0.62, max_model_len=args.max_model_len,
                seed=int(cfg["seed"]), enable_prefix_caching=False)
        for q in questions:
            for method in args.methods:
                if budget.elapsed >= budget.max_wall_seconds:
                    raise StageBudgetExceeded(
                        f"wall budget exhausted before next trajectory: "
                        f"{budget.elapsed:.1f}s")
                rec = run_one(q, method, cfg, policy, critic, traces, budget)
                records.append(rec)
                if rec["error"] is not None:
                    raise RuntimeError("Trajectory failed; stopped stage: " + rec["error"])
                print(f"{method} {q['id']} finish={rec['finish']} "
                      f"tokens={rec['totals']['total_output_tokens']} "
                      f"accepted={rec['accepted_tokens']}", flush=True)
    except StageBudgetExceeded as e:
        failure = str(e)
        exit_code = 3
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
        exit_code = 2
        traceback.print_exc()
    finally:
        for backend in (policy, critic):
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
        rows = []
        if traces.exists():
            for line in traces.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        counts = {m: sum(1 for r in rows if r["method"] == m) for m in args.methods}
        context_limits = sum(1 for r in rows if r["finish"] == "context_limit")
        budget_stops = sum(1 for r in rows if r["finish"] == "budget_stopped")
        errors = sum(1 for r in rows if r["error"] is not None)
        summary = {
            "trajectories": len(rows),
            "method_counts": counts,
            "generated_tokens": budget.generated,
            "wall_seconds": budget.elapsed,
            "setup_seconds": time.time() - setup_started,
            "budget_max_generated": budget.max_generated,
            "budget_max_wall": budget.max_wall_seconds,
            "budget_max_model_len": budget.max_model_len,
            "context_capped_calls": budget.context_capped_calls,
            "context_limits": context_limits,
            "budget_stops": budget_stops,
            "errors": errors,
            "partial": len(rows) < len(questions) * len(args.methods),
            "complete": len(rows) == len(questions) * len(args.methods) and not failure,
            "methods": args.methods,
            "questions": len(questions),
            "failure": failure,
            "exit_code": exit_code,
        }
        write_json_atomic(outdir / "run-status.json", summary)
        print(json.dumps(summary), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
