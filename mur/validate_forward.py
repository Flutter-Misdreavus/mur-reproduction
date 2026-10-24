"""Inspect raw probabilities in the SAME forward pass, without changing sampling.

The older independent BF16 forward comparisons remain failed diagnostics. This
gate isolates the sampler/API contract from differences between forward kernels.
The observer is installed only in this short-lived validation engine via vLLM's
public collective_rpc API. Main inference never installs it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .backend import VLLMBackend, generation_nll
from .limits import StageBudget
from .prompts import policy_prompt, request_seed
from .validate_signal import sampler_source_evidence

REFERENCE = "same_forward_unmodified_logits"
ATOL = 5e-5  # Float32 log-softmax arithmetic, not a BF16 forward tolerance.
NEGATIVE_CONTROL_MIN_DIFF = 1e-3  # Temperature-only transform must disagree somewhere.


def install_observer(worker):
    """Runs inside the existing model worker after model initialization."""
    import torch
    sampler = worker.model_runner.sampler
    if hasattr(worker, "_mur_capture"):
        raise RuntimeError("observer already installed")
    worker._mur_capture = []
    original = sampler.forward

    def observed(logits, sampling_metadata):
        # Clone BEFORE the sampler can apply any in-place transforms. This
        # consumes no RNG and does not modify logits or sampling metadata.
        raw = logits.detach().float().clone()
        raw_logp = torch.log_softmax(raw, dim=-1)
        temp = sampling_metadata.temperature
        if temp is None:
            raise RuntimeError("observer requires one positive temperature per row")
        temp = temp.reshape(-1)
        if temp.numel() != raw.shape[0]:
            raise RuntimeError("observer requires one positive temperature per row")
        if not bool(torch.all(torch.isfinite(temp) & (temp > 0))):
            raise RuntimeError("observer requires finite positive sampling temperature")
        temperature_logp = torch.log_softmax(raw / temp[:, None].to(raw.device), dim=-1)
        result = original(logits, sampling_metadata)
        if result.logprobs_tensors is None:
            raise RuntimeError("observer requires sampled-token logprobs")
        ids = result.sampled_token_ids.reshape(-1).long()
        if ids.numel() != raw.shape[0]:
            raise RuntimeError("unexpected speculative-decoding sampler output shape")
        rows = torch.arange(ids.numel(), device=ids.device)
        tensor_ids, reported, _ = result.logprobs_tensors
        if not torch.equal(tensor_ids[:, 0].long(), ids):
            raise RuntimeError("sampled token is not in the expected logprob column")
        worker._mur_capture.extend([
            {"token_id": int(tid), "raw_logprob": float(raw_lp),
             "sampler_logprob": float(lp), "temperature_only_logprob": float(tlp)}
            for tid, raw_lp, lp, tlp in zip(
                ids.cpu().tolist(), raw_logp[rows, ids].cpu().tolist(),
                reported[:, 0].cpu().tolist(), temperature_logp[rows, ids].cpu().tolist())
        ])
        return result

    sampler.forward = observed
    return {"installed": True, "observer_changes_sampling": False}


def drain_observer(worker):
    rows = worker._mur_capture
    worker._mur_capture = []
    return rows


def compare_returned(generation, captured):
    if not captured:
        raise ValueError("empty observer evidence")
    if generation.token_ids != [r["token_id"] for r in captured]:
        raise ValueError("sampler and client token IDs/counts differ")
    nll = generation_nll(generation)["token_nll"]
    if not nll:
        raise ValueError("empty generation")
    if len(nll) != len(captured):
        raise ValueError("client and observer token counts differ")
    errors = []
    negative_control = []
    for value, row in zip(nll, captured):
        if not all(math.isfinite(row[k]) for k in
                   ("raw_logprob", "sampler_logprob", "temperature_only_logprob")):
            raise ValueError("nonfinite observer evidence")
        errors.extend((abs(value + row["raw_logprob"]),
                       abs(row["sampler_logprob"] - row["raw_logprob"]),
                       abs(value + row["sampler_logprob"])))
        negative_control.append(abs(value + row["temperature_only_logprob"]))
    max_negative = max(negative_control)
    return {"passed": max(errors) <= ATOL, "max_abs_error": max(errors),
            "mean_abs_error": sum(errors) / len(errors),
            "temperature_only_max_error": max_negative,
            "negative_control_passed": max_negative > NEGATIVE_CONTROL_MIN_DIFF,
            "output_tokens": len(nll), "client_token_nll": nll,
            "captured": captured}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--config", default="configs/validation.json")
    ap.add_argument("--max-generated-tokens", type=int, default=10000)
    ap.add_argument("--max-wall-seconds", type=float, default=600)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    sources = json.loads(Path(args.sources).read_text())
    model = Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime")) / "models" / sources["policy"]["id"].split("/")[-1]
    probes = cfg["probes"]
    if len(probes) != 8 or len(probes) * cfg["max_tokens"] > args.max_generated_tokens:
        raise ValueError("frozen probe count or validation token budget violated")
    results, backend, error = [], None, None
    budget = StageBudget(args.max_generated_tokens, args.max_wall_seconds, args.max_model_len)
    evidence = sampler_source_evidence()
    try:
        if not evidence.get("raw_before_penalties_and_temperature"):
            raise ValueError("installed sampler source contract failed")
        backend = VLLMBackend(str(model), gpu_memory_utilization=0.42,
                              max_model_len=args.max_model_len, seed=cfg["seed"])
        installed = backend.llm.collective_rpc(install_observer, timeout=30)
        if len(installed) != 1 or not installed[0]["installed"]:
            raise ValueError("single observed worker required")
        for index, (name, question) in enumerate(probes):
            budget.reserve(cfg["max_tokens"], name)
            prompt = policy_prompt(backend.tokenizer, question, [], 0)
            if len(backend.encode(prompt)) + cfg["max_tokens"] > args.max_model_len:
                raise ValueError("probe exceeds context")
            gen = backend.generate(prompt, max_tokens=cfg["max_tokens"], temperature=0.6,
                                   top_p=0.8, top_k=20, presence_penalty=1.5,
                                   seed=request_seed(cfg["seed"], "signal-validation", index, "proposal", 0),
                                   stop=["\nStep"], logprobs=0)
            budget.account("policy", len(gen.token_ids))
            captured = backend.llm.collective_rpc(drain_observer, timeout=30)
            if len(captured) != 1:
                raise ValueError("single observed worker required")
            result = compare_returned(gen, captured[0])
            result.update(name=name, repeated_token_ids=len(gen.token_ids) != len(set(gen.token_ids)))
            results.append(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.close()
    passed = (error is None and len(results) == 8 and all(r["passed"] for r in results)
              and all(r["output_tokens"] > 0 for r in results)
              and any(r["repeated_token_ids"] for r in results)
              and any(r["negative_control_passed"] for r in results))
    # Only a portable relative source path belongs in public summaries.
    evidence["path"] = "vllm/v1/sample/sampler.py"
    negative_control_ok = any(r["negative_control_passed"] for r in results)
    repeated_ok = any(r["repeated_token_ids"] for r in results)
    payload = {"schema_version": 1, "passed": passed, "error": error,
               "validation_reference": REFERENCE, "atol": ATOL,
               "negative_control_min_diff": NEGATIVE_CONTROL_MIN_DIFF,
               "probe_count": len(results),
               "complete_nonempty_probes": len(results) == 8 and all(r["output_tokens"] > 0 for r in results),
               "repeated_token_probe": repeated_ok,
               "temperature_only_negative_control_passed": negative_control_ok,
               "max_temperature_only_diff": max(
                   (r["temperature_only_max_error"] for r in results), default=None),
               "max_abs_token_logprob_diff": max((r["max_abs_error"] for r in results), default=None),
               "mean_abs_token_logprob_diff": (
                   sum(r["mean_abs_error"] for r in results) / len(results) if results else None),
               "reference_run": "mur-20261024-prep-r5",
               "cross_engine_hf_parity": False, "independent_prefill_parity": False,
               "prior_failed_runs": ["mur-20261024-prep-r3", "mur-20261024-prep-r4"],
               "generation_logprobs_used_for_nll": True,
               "generated_tokens": budget.generated,
               "sampler_source": evidence, "probes": results,
               "no_extra_model_forward": True, "observer_installed_only_for_validation": True}

    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k:v for k,v in payload.items() if k != "probes"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
