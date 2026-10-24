"""Thin vLLM backend with explicit token-level and raw prompt-logprob access."""
from __future__ import annotations

import gc
import math
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_USE_V1"] = "1"


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
        import vllm
        from transformers import AutoTokenizer

        if vllm.__version__ != "0.8.5":
            raise RuntimeError("Revalidate raw-generation logprob semantics before changing vLLM")

        self.model_path = model_path
        self.max_model_len = int(max_model_len)
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
            max_num_seqs=8,
            max_num_batched_tokens=2048,
        )
        from vllm.v1.engine.llm_engine import LLMEngine as V1Engine
        if not isinstance(self.llm.llm_engine, V1Engine):
            raise RuntimeError("V1 engine required: V0 reports transformed generation logprobs")

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
        token_nll = []
        for i in range(len(prompt_ids), len(full)):
            tid = full[i]
            entry = plp[i]
            if not entry or tid not in entry:
                raise RuntimeError(f"missing raw logprob at position {i}")
            lp = float(entry[tid].logprob)
            if not math.isfinite(lp):
                raise RuntimeError("non-finite raw logprob")
            total -= lp
            token_nll.append(-lp)
        probe = out.outputs[0].token_ids if out.outputs else []
        return {
            "sum_nll": total,
            "mean_nll": total / len(output_ids),
            "tokens": len(output_ids),
            "probe_tokens": len(probe),
            "token_nll": token_nll,
        }

    def close(self):
        """Shut down the V1 engine-core child, then drop model references.

        vLLM 0.8.5 V1 uses a separate EngineCore process by default.  Setting
        ``self.llm = None`` alone leaves that process alive, so explicitly
        request client shutdown and wait for the child to disappear.
        """
        llm = getattr(self, "llm", None)
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        if core is not None:
            try:
                core.shutdown()
            except Exception:
                pass
        self.llm = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def generation_nll(gen: Generation) -> dict:
    """Read V1 raw logprobs from the generating forward pass; no rescoring call."""
    if not gen.token_ids or len(gen.logprobs) != len(gen.token_ids):
        raise ValueError("missing sampled-token logprobs")
    values = []
    for tid, entry in zip(gen.token_ids, gen.logprobs):
        if tid not in entry:
            raise ValueError("sampled token is absent from logprob response")
        lp = float(entry[tid].logprob)
        if not math.isfinite(lp) or lp > 1e-6:
            raise ValueError("invalid raw sampled-token logprob")
        values.append(-lp)
    return {"mean_nll": sum(values)/len(values), "token_nll": values,
            "tokens": len(values), "sum_nll": sum(values), "probe_tokens": 0}


def judgment_from_generation(gen: Generation) -> dict:
    """The prompt ends in \"\\boxed\"; only its first completed brace is the verdict.

    The actual standalone Yes/No token must overlap the matched label in the output.
    Never select an unrelated later Yes/No token from explanation text.
    """
    missing = {"label": None, "probability": None, "position": None, "method": "missing"}
    match = re.match(r"^\s*\{\s*(Yes|No)\s*\}", gen.text)
    if not match:
        return missing
    offset = 0
    for i, (tid, entry) in enumerate(zip(gen.token_ids, gen.logprobs)):
        if tid not in entry:
            return missing
        info = entry[tid]
        token = info.decoded_token or ""
        end = offset + len(token)
        if offset < match.end(1) and end > match.start(1):
            if token.strip() != match.group(1):
                return missing
            lp = float(info.logprob)
            if not math.isfinite(lp) or lp > 1e-6:
                return missing
            return {"label": match.group(1), "probability": math.exp(lp),
                    "position": i, "method": "initial_boxed_verdict"}
        offset = end
    return missing
