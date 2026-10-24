"""Teacher-forced check that vLLM prompt_logprobs are raw model log-probabilities."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from .backend import VLLMBackend
from .prompts import policy_prompt, request_seed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args(argv)
    sources = json.load(open(args.sources, encoding="utf-8"))
    root = Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime")) / "models"
    path = str(root / sources["policy"]["id"].split("/")[-1])
    backend = VLLMBackend(path, gpu_memory_utilization=0.29, max_model_len=4096, seed=args.seed)
    question = "What is 2+2? Give a short answer."
    prompt = policy_prompt(backend.tokenizer, question, [], 0)
    gen = backend.generate(prompt, max_tokens=args.max_tokens, temperature=0.6,
                           top_p=0.8, top_k=20, presence_penalty=1.5,
                           seed=request_seed(args.seed, "logprob-check", 0, "proposal", 0),
                           stop=["\nStep"])
    prompt_ids = backend.encode(prompt)
    output_ids = list(gen.token_ids)
    if not output_ids:
        raise SystemExit("validation generation produced no tokens")
    sc = backend.score_nll(prompt_ids, output_ids)
    backend.close()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    model.eval()
    full = prompt_ids + output_ids
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([full], device=model.device)).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
    hf_total = 0.0
    for i in range(len(prompt_ids), len(full)):
        hf_total -= float(logp[i - 1, full[i]])
    hf_mean = hf_total / len(output_ids)
    diff = abs(sc["mean_nll"] - hf_mean)
    result = {
        "schema_version": 1,
        "passed": diff < 1e-4,
        "model": sources["policy"]["id"],
        "revision": sources["policy"]["revision"],
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(output_ids),
        "vllm_mean_nll": sc["mean_nll"],
        "hf_teacher_forced_mean_nll": hf_mean,
        "abs_diff": diff,
        "tolerance": 1e-4,
        "scoring_path": "vllm prompt_logprobs with identity transforms (temperature=0, top_p=1, top_k=-1, presence_penalty=0)",
        "generation_logprobs_used_for_nll": False,
        "note": "The check verifies the exact scoring path used by mur.run for m_t.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
