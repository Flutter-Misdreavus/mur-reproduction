"""Thin vLLM backend with explicit token-level and raw prompt-logprob access."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class Generation:
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    logprobs: list = field(default_factory=list)
    prompt_tokens: int = 0


class VLLMBackend:
    def __init__(self, model_path: str, gpu_memory_utilization: float,
                 max_model_len: int = 16384, seed: int = 42):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            disable_log_stats=True,
            seed=seed,
            trust_remote_code=True,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @staticmethod
    def _sampling_params(**kw):
        from vllm import SamplingParams
        return SamplingParams(**kw)

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 top_p: float, top_k: int, presence_penalty: float,
                 seed: int, stop: list[str] | None = None,
                 logprobs: int | None = None,
                 include_stop_str_in_output: bool = False) -> Generation:
        params = self._sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            logprobs=logprobs,
            include_stop_str_in_output=include_stop_str_in_output,
        )
        ids = self.encode(prompt)
        req = {"prompt_token_ids": ids}
        out = self.llm.generate([req], params, use_tqdm=False)[0]
        return self._generation(out, len(ids))

    def generate_many(self, prompts: list[str], params: list) -> list[Generation]:
        reqs = [{"prompt_token_ids": self.encode(p)} for p in prompts]
        outs = self.llm.generate(reqs, params, use_tqdm=False)
        return [self._generation(o, len(r["prompt_token_ids"])) for o, r in zip(outs, reqs)]

    def generate_batch(self, prompts: list[str], kwargs_list: list[dict]) -> list[Generation]:
        params = [self._sampling_params(**kw) for kw in kwargs_list]
        return self.generate_many(prompts, params)

    def _generation(self, out, prompt_tokens: int) -> Generation:
        if not out.outputs:
            return Generation(prompt_tokens=prompt_tokens)
        o = out.outputs[0]
        return Generation(
            text=o.text,
            token_ids=list(o.token_ids),
            finish_reason=getattr(o, "finish_reason", None),
            logprobs=list(o.logprobs or []),
            prompt_tokens=prompt_tokens,
        )

    def score_nll(self, prompt_ids: list[int], output_ids: list[int]) -> dict:
        """Raw teacher-forced NLL of output_ids conditioned on prompt_ids.

        Uses vLLM prompt_logprobs, which are computed before sampling transforms.
        Returns mean/sum NLL, scored token count and the one probe token generated
        by the scoring call (counted in the compute ledger, not in the step).
        """
        if not output_ids:
            raise ValueError("cannot score an empty generation")
        full = list(prompt_ids) + list(output_ids)
        params = self._sampling_params(
            max_tokens=1, temperature=0.0, top_p=1.0, top_k=-1,
            presence_penalty=0.0, seed=0, prompt_logprobs=0,
        )
        out = self.llm.generate([{"prompt_token_ids": full}], params, use_tqdm=False)[0]
        plp = getattr(out, "prompt_logprobs", None)
        if plp is None or len(plp) < len(full):
            raise RuntimeError("vLLM did not return complete prompt_logprobs")
        total = 0.0
        for i in range(len(prompt_ids), len(full)):
            tid = full[i]
            entry = plp[i]
            if not entry or tid not in entry:
                raise RuntimeError(f"missing raw logprob at position {i}")
            lp = float(entry[tid].logprob)
            if not math.isfinite(lp):
                raise RuntimeError("non-finite raw logprob")
            total -= lp
        probe = out.outputs[0].token_ids if out.outputs else []
        return {
            "sum_nll": total,
            "mean_nll": total / len(output_ids),
            "tokens": len(output_ids),
            "probe_tokens": len(probe),
        }

    def close(self):
        self.llm = None
        self.tokenizer = None


def judgment_from_generation(gen: Generation) -> dict:
    """Locate the designated Yes/No token after the final \\boxed marker."""
    prefix = ""
    tokens = []
    for tid, info in enumerate(gen.logprobs or []):
        # vLLM SampleLogprobs is keyed by token id; the sampled token is present.
        if not info:
            continue
        sampled = None
        target = gen.token_ids[tid] if tid < len(gen.token_ids) else None
        for tok_id, lp in info.items():
            if tok_id == target:
                sampled = lp
                break
        if sampled is None:
            sampled = next(iter(info.values()))
        prefix += sampled.decoded_token or ""
        tokens.append((sampled.decoded_token or "", float(sampled.logprob)))
        stripped = (sampled.decoded_token or "").strip()
        if re.search(r"\\boxed\s*\{?\s*$", prefix) and stripped in ("Yes", "No"):
            return {"label": stripped, "probability": math.exp(float(sampled.logprob)),
                    "position": tid, "method": "boxed"}
    # Fallback: a standalone Yes/No token after a boxed marker, or the last one.
    for i, (tok, lp) in enumerate(tokens):
        if tok.strip() in ("Yes", "No") and i and "boxed" in "".join(t for t, _ in tokens[:i]):
            return {"label": tok.strip(), "probability": math.exp(lp),
                    "position": i, "method": "boxed_fallback"}
    for i in range(len(tokens) - 1, -1, -1):
        tok, lp = tokens[i]
        if tok.strip() in ("Yes", "No"):
            return {"label": tok.strip(), "probability": math.exp(lp),
                    "position": i, "method": "last_yes_no"}
    return {"label": None, "probability": None, "position": None, "method": "missing"}
