"""Validate raw V1 generation logprobs against an HF teacher-forced reference.

The vLLM generation and the HF reference run in *separate* subprocesses, so the
vLLM EngineCore process is fully released before the reference model allocates.
At least three short probes are generated with the frozen sampling settings
(temperature 0.6, top_p 0.8, top_k 20, presence_penalty 1.5) and the set must
contain a probe with repeated token ids.  Tolerances are frozen before values
are seen.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from .backend import VLLMBackend, generation_nll
from .limits import ContextLimit, StageBudget, StageBudgetExceeded
from .prompts import policy_prompt, request_seed

PROBES = [
    ("repeat", "Repeat the word echo exactly four times separated by spaces. "
               "Output only the repeated words."),
    ("count", "Write the numbers 1 1 2 2 3 3 separated by spaces. "
              "Output only the numbers."),
    ("ones", "Write ten copies of the digit 1 separated by spaces. "
             "Output only the digits."),
    ("sum", "What is 2+2? Give a very short answer."),
    ("double", "Write the sequence a a b b c c. Output only the letters."),
]

# Frozen before observing any comparison values.
TOLERANCE = {"max_abs_token_nll": 0.15, "mean_abs_token_nll": 0.02,
             "mean_nll": 0.005}


def write_json_atomic(path: Path, payload):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def model_root() -> Path:
    return Path(os.environ.get("MUR_MODEL_ROOT") or
                (Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime")) / "models"))


def failure(path: str, message: str, **extra):
    payload = {"schema_version": 1, "passed": False, "error": message,
               "scoring_path": "vLLM 0.8.5 V1 raw generation logprobs",
               "tolerance": TOLERANCE, **extra}
    write_json_atomic(Path(path), payload)
    print(json.dumps(payload, indent=2), file=sys.stderr)
    return 1


def generate_phase(args) -> int:
    sources = json.load(open(args.sources, encoding="utf-8"))
    path = str(model_root() / sources["policy"]["id"].split("/")[-1])
    budget = StageBudget(max_generated=args.max_generated_tokens,
                         max_wall_seconds=args.max_wall_seconds,
                         max_model_len=args.max_model_len)
    backend = None
    try:
        backend = VLLMBackend(path, gpu_memory_utilization=0.26,
                              max_model_len=args.max_model_len, seed=args.seed)
        probes = []
        for index, (name, question) in enumerate(PROBES):
            prompt = policy_prompt(backend.tokenizer, question, [], 0)
            prompt_ids = backend.encode(prompt)
            effective = budget.fit(len(prompt_ids), args.max_tokens, name)
            budget.reserve(effective, name)
            gen = backend.generate(
                prompt, max_tokens=effective, temperature=0.6, top_p=0.8,
                top_k=20, presence_penalty=1.5,
                seed=request_seed(args.seed, "logprob-check", index, "proposal", 0),
                stop=["\nStep"], logprobs=0)
            budget.account("policy", len(gen.token_ids))
            sc = generation_nll(gen)
            ids = list(gen.token_ids)
            probes.append({
                "name": name,
                "question": question,
                "prompt_ids": prompt_ids,
                "output_ids": ids,
                "output_text": gen.text,
                "finish_reason": gen.finish_reason,
                "generation_mean_nll": sc["mean_nll"],
                "generation_token_nll": sc["token_nll"],
                "repeated_token_ids": len(ids) != len(set(ids)),
                "generated_tokens": len(ids),
            })
        result = {
            "schema_version": 1,
            "phase": "generation",
            "model": sources["policy"]["id"],
            "revision": sources["policy"]["revision"],
            "sampling": {"temperature": 0.6, "top_p": 0.8, "top_k": 20,
                         "presence_penalty": 1.5, "logprobs": 0},
            "probes": probes,
            "probe_count": len(probes),
            "repeated_token_probe": any(p["repeated_token_ids"] for p in probes),
            "generated_tokens": budget.generated,
            "wall_seconds": budget.elapsed,
            "engine": "vllm-0.8.5-v1",
        }
        write_json_atomic(Path(args.output), result)
        print(json.dumps({"phase": "generation", "probes": len(probes),
                          "generated_tokens": budget.generated,
                          "repeated_token_probe": result["repeated_token_probe"]},
                         indent=2))
        return 0
    except (StageBudgetExceeded, ContextLimit) as e:
        return failure(args.output, f"{type(e).__name__}: {e}")
    except Exception as e:
        return failure(args.output, f"{type(e).__name__}: {e}")
    finally:
        if backend is not None:
            backend.close()


def reference_phase(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    generated = json.load(open(args.probes, encoding="utf-8"))
    sources = json.load(open(args.sources, encoding="utf-8"))
    path = str(model_root() / sources["policy"]["id"].split("/")[-1])
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to("cuda")
    model.eval()
    results = []
    try:
        for p in generated["probes"]:
            full = list(p["prompt_ids"]) + list(p["output_ids"])
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([full], device=model.device)).logits[0].float()
                logp = torch.log_softmax(logits, dim=-1)
            hf_nll = []
            for i in range(len(p["prompt_ids"]), len(full)):
                hf_nll.append(-float(logp[i - 1, full[i]]))
            vllm_nll = [float(x) for x in p["generation_token_nll"]]
            diffs = [a - b for a, b in zip(vllm_nll, hf_nll)]
            max_abs = max(abs(x) for x in diffs)
            mean_abs = sum(abs(x) for x in diffs) / len(diffs)
            vllm_mean = sum(vllm_nll) / len(vllm_nll)
            hf_mean = sum(hf_nll) / len(hf_nll)
            diff = abs(vllm_mean - hf_mean)
            passed = (max_abs < TOLERANCE["max_abs_token_nll"]
                      and mean_abs < TOLERANCE["mean_abs_token_nll"]
                      and diff < TOLERANCE["mean_nll"])
            results.append({
                "name": p["name"],
                "passed": passed,
                "output_tokens": len(p["output_ids"]),
                "repeated_token_ids": bool(p.get("repeated_token_ids")),
                "vllm_mean_nll": vllm_mean,
                "hf_teacher_forced_mean_nll": hf_mean,
                "abs_mean_diff": diff,
                "max_abs_token_logprob_diff": max_abs,
                "mean_abs_token_logprob_diff": mean_abs,
                "token_diffs": diffs,
            })
    finally:
        model = None
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    all_pass = (len(results) >= 3 and all(r["passed"] for r in results)
                and any(r["repeated_token_ids"] for r in results))
    payload = {
        "schema_version": 1,
        "passed": all_pass,
        "probe_count": len(results),
        "repeated_token_probe": any(r["repeated_token_ids"] for r in results),
        "probes": results,
        "max_abs_token_logprob_diff": max(r["max_abs_token_logprob_diff"] for r in results) if results else None,
        "mean_abs_token_logprob_diff": (sum(r["mean_abs_token_logprob_diff"] for r in results) / len(results)) if results else None,
        "tolerance": TOLERANCE,
        "tolerance_reason": ("BF16 vLLM/HF kernel differences; an unaccounted "
                             "sampling transform (presence_penalty=1.5) would be "
                             "far larger."),
        "model": sources["policy"]["id"],
        "revision": sources["policy"]["revision"],
        "generated_tokens": int(generated.get("generated_tokens", 0)),
        "vllm_generation_wall_seconds": generated.get("wall_seconds"),
        "scoring_path": ("vLLM 0.8.5 V1 raw sampled-token logprobs before "
                         "penalties/temperature, no extra scoring call"),
        "generation_logprobs_used_for_nll": True,
        "separate_processes": True,
        "note": "The check verifies the exact scoring path used by mur.run for m_t.",
    }
    write_json_atomic(Path(args.output), payload)
    print(json.dumps(payload, indent=2))
    return 0 if all_pass else 1


def orchestrator(args) -> int:
    out = Path(args.output)
    gen_path = out.with_name(out.stem + "-generation.json")
    base = [sys.executable, "-m", "mur.validate_logprobs",
            "--sources", args.sources, "--seed", str(args.seed),
            "--max-tokens", str(args.max_tokens),
            "--max-generated-tokens", str(args.max_generated_tokens),
            "--max-wall-seconds", str(args.max_wall_seconds),
            "--max-model-len", str(args.max_model_len)]
    started = time.time()
    rc = subprocess.call(base + ["--phase", "generate", "--output", str(gen_path)],
                         env=os.environ.copy())
    if rc != 0:
        return failure(args.output, f"generation phase exited {rc}",
                       separate_processes=True, generated_tokens=None)
    # Give the engine child a moment to disappear before the reference loads.
    time.sleep(1.0)
    rc = subprocess.call(base + ["--phase", "reference", "--probes", str(gen_path),
                                 "--output", str(out)], env=os.environ.copy())
    if rc != 0 and not out.exists():
        return failure(args.output, f"reference phase exited {rc}",
                       separate_processes=True)
    try:
        payload = json.load(open(out, encoding="utf-8"))
    except Exception as e:
        return failure(args.output, f"unreadable reference output: {e}")
    payload["orchestrator_wall_seconds"] = time.time() - started
    payload["separate_processes"] = True
    write_json_atomic(out, payload)
    return 0 if payload.get("passed") else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--phase", choices=("generate", "reference"), default=None)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--max-generated-tokens", type=int, default=10000)
    ap.add_argument("--max-wall-seconds", type=float, default=600)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args(argv)
    if args.phase == "generate":
        return generate_phase(args)
    if args.phase == "reference":
        if not args.probes:
            raise SystemExit("--probes is required for the reference phase")
        return reference_phase(args)
    return orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
