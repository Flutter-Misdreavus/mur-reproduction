"""Validate the raw *generation* logprob signal used for ``m_t``.

The frozen r3 cross-engine comparison (vLLM generation vs a Hugging Face
forward pass) failed and remains a separate diagnostic.  This module implements
the explicitly amended, narrower gate: re-score the *exact saved token ids* with
an identity-sampling teacher-forced prefill in a fresh instance of the same
pinned vLLM V1 engine.

Generation and reference run in two short-lived child processes.  The
generation child writes prompt/output ids and the raw sampled-token NLL vector.
The reference child starts after that child (and its EngineCore process) has
exited, opens the same pinned BF16 weights with the default attention backend,
and asks vLLM for ``prompt_logprobs`` on those exact ids.  No decoded text is
ever re-tokenized for scoring.

Every probe must satisfy all three unchanged tolerances.  A single failed or
empty probe blocks the gate; there is no diagnostic follow-up in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

DEFAULT_CONFIG = "configs/validation.json"
DEFAULT_SOURCES = "configs/sources.json"
VALIDATION_REFERENCE = "same_engine_identity_sampling_teacher_forced_prefill"
REFERENCE_RUN = "mur-20261024-prep-r3"
CROSS_ENGINE_HF_PARITY = False


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_root() -> Path:
    return Path(os.environ.get("MUR_MODEL_ROOT") or
                (Path(os.environ.get("MUR_RUNTIME_ROOT", "runtime")) / "models"))


def run_id() -> str | None:
    return os.environ.get("MUR_RUN_ID") or None


def _vllm_sampler_path() -> Path | None:
    """Locate the installed sampler without importing the CUDA stack."""
    try:
        spec = importlib.util.find_spec("vllm")
        if spec and spec.submodule_search_locations:
            candidate = Path(list(spec.submodule_search_locations)[0]) / "v1" / "sample" / "sampler.py"
            if candidate.exists():
                return candidate
    except Exception:
        pass
    for base in (Path(sys.prefix) / "lib", Path("/usr/local/lib")):
        if not base.exists():
            continue
        for candidate in base.glob("python*/site-packages/vllm/v1/sample/sampler.py"):
            return candidate
    return None


def sampler_source_evidence(path: str | None = None) -> dict:
    """Record the installed sampler hash and the raw-logprob ordering proof.

    In vLLM 0.8.5 V1, ``raw_logprobs`` are computed from the untouched logits
    before allowed-token/bad-word/bias/penalty transforms and before ``sample``
    applies temperature.  The gate relies on that ordering.
    """
    sampler = Path(path) if path else _vllm_sampler_path()
    if sampler is None or not sampler.exists():
        return {"available": False, "error": "installed vllm sampler source not found",
                "raw_before_penalties_and_temperature": False}
    text = sampler.read_text(encoding="utf-8")
    lines = text.splitlines()

    def line_of(needle: str) -> int | None:
        for i, line in enumerate(lines, 1):
            if needle in line:
                return i
        return None

    raw_line = line_of("raw_logprobs = self.compute_logprobs(logits)")
    penalties_line = line_of("logits = self.apply_penalties(logits, sampling_metadata)")
    temperature_line = line_of("self.apply_temperature(logits, sampling_metadata")
    sample_line = line_of("sampled = self.sample(logits, sampling_metadata)")
    ordering = (raw_line is not None and penalties_line is not None
                and temperature_line is not None and sample_line is not None
                and raw_line < penalties_line and raw_line < temperature_line
                and raw_line < sample_line)
    return {
        "available": True,
        "path": str(sampler),
        "sha256": sha256_file(sampler),
        "raw_logprobs_line": raw_line,
        "apply_penalties_line": penalties_line,
        "apply_temperature_line": temperature_line,
        "sample_line": sample_line,
        "raw_before_penalties_and_temperature": ordering,
        "comment": ("raw logits are used for top-k/sampled-token logprobs; "
                    "sampling penalties and temperature are applied afterwards"),
    }


def validate_config(cfg: dict) -> None:
    probes = cfg.get("probes")
    if not isinstance(probes, list) or len(probes) != 8:
        raise ValueError("config must freeze exactly eight probes")
    names = set()
    for item in probes:
        if (not isinstance(item, list) or len(item) != 2
                or not all(isinstance(x, str) and x.strip() for x in item)):
            raise ValueError("each probe must be [name, question]")
        if item[0] in names:
            raise ValueError(f"duplicate probe name {item[0]}")
        names.add(item[0])
    tol = cfg.get("tolerance") or {}
    for key in ("max_abs_token_nll", "mean_abs_token_nll", "mean_nll"):
        if not isinstance(tol.get(key), (int, float)) or tol[key] <= 0:
            raise ValueError(f"missing/invalid tolerance {key}")
    if int(cfg.get("max_tokens", 0)) <= 0:
        raise ValueError("max_tokens must be positive")


def failure(path: str, message: str, **extra):
    payload = {
        "schema_version": 1,
        "run_id": extra.pop("run_id", run_id()),
        "passed": False,
        "error": message,
        "validation_reference": VALIDATION_REFERENCE,
        "reference_run": REFERENCE_RUN,
        "cross_engine_hf_parity": CROSS_ENGINE_HF_PARITY,
        "scoring_path": "same-engine vLLM 0.8.5 V1 prompt_logprobs on exact saved ids",
        "separate_processes": True,
        **extra,
    }
    write_json_atomic(Path(path), payload)
    print(json.dumps(payload, indent=2), file=sys.stderr)
    return 1


def compare_probe(gen_token_nll: list, ref_token_nll: list,
                  tolerance: dict, *, output_tokens: int | None = None) -> dict:
    """Compute unchanged per-token/mean tolerances for one probe."""
    gen = [float(x) for x in gen_token_nll]
    ref = [float(x) for x in ref_token_nll]
    if not gen or not ref:
        raise ValueError("empty token NLL vector")
    if len(gen) != len(ref):
        raise ValueError(f"token count mismatch: generation {len(gen)} reference {len(ref)}")
    if output_tokens is not None and int(output_tokens) != len(gen):
        raise ValueError(f"output token count {output_tokens} != scored {len(gen)}")
    if any(not math.isfinite(x) for x in gen + ref):
        raise ValueError("non-finite token NLL")
    diffs = [g - r for g, r in zip(gen, ref)]
    max_abs = max(abs(x) for x in diffs)
    mean_abs = sum(abs(x) for x in diffs) / len(diffs)
    gen_mean = sum(gen) / len(gen)
    ref_mean = sum(ref) / len(ref)
    abs_mean_diff = abs(gen_mean - ref_mean)
    passed = (max_abs < float(tolerance["max_abs_token_nll"])
              and mean_abs < float(tolerance["mean_abs_token_nll"])
              and abs_mean_diff < float(tolerance["mean_nll"]))
    return {
        "passed": bool(passed),
        "generation_mean_nll": gen_mean,
        "reference_mean_nll": ref_mean,
        "abs_mean_diff": abs_mean_diff,
        "max_abs_token_logprob_diff": max_abs,
        "mean_abs_token_logprob_diff": mean_abs,
        "token_diffs": diffs,
        "generation_token_nll": gen,
        "reference_token_nll": ref,
        "scored_tokens": len(gen),
    }


def generate_phase(args) -> int:
    cfg = load_json(args.config)
    validate_config(cfg)
    sources = load_json(args.sources)
    path = str(model_root() / sources["policy"]["id"].split("/")[-1])
    seed = int(cfg["seed"])
    max_tokens = int(cfg["max_tokens"])
    budget = StageBudget(max_generated=args.max_generated_tokens,
                         max_wall_seconds=args.max_wall_seconds,
                         max_model_len=args.max_model_len)
    backend = None
    probes = []
    try:
        backend = VLLMBackend(path, gpu_memory_utilization=0.42,
                              max_model_len=args.max_model_len, seed=seed)
        for index, (name, question) in enumerate(cfg["probes"]):
            prompt = policy_prompt(backend.tokenizer, question, [], 0)
            prompt_ids = backend.encode(prompt)
            effective = budget.fit(len(prompt_ids), max_tokens, f"signal-{name}")
            budget.reserve(effective, f"signal-{name}")
            gen = backend.generate(
                prompt, max_tokens=effective, temperature=0.6, top_p=0.8,
                top_k=20, presence_penalty=1.5,
                seed=request_seed(seed, "signal-validation", index, "proposal", 0),
                stop=["\nStep"], logprobs=0)
            budget.account("policy", len(gen.token_ids))
            record = {
                "name": name,
                "question": question,
                "prompt_ids": list(prompt_ids),
                "output_ids": list(gen.token_ids),
                "output_text": gen.text,
                "finish_reason": gen.finish_reason,
                "max_tokens_effective": effective,
                "generated_tokens_this_probe": len(gen.token_ids),
                "repeated_token_ids": len(gen.token_ids) != len(set(gen.token_ids)),
            }
            if gen.token_ids:
                scored = generation_nll(gen)
                record.update({
                    "generation_mean_nll": scored["mean_nll"],
                    "generation_token_nll": scored["token_nll"],
                    "scored_tokens": scored["tokens"],
                })
            probes.append(record)
        empty = [p["name"] for p in probes
                 if not p.get("output_ids") or not p.get("generation_token_nll")]
        result = {
            "schema_version": 1,
            "run_id": run_id(),
            "phase": "generation",
            "validation_reference": VALIDATION_REFERENCE,
            "reference_run": REFERENCE_RUN,
            "cross_engine_hf_parity": CROSS_ENGINE_HF_PARITY,
            "model": sources["policy"]["id"],
            "revision": sources["policy"]["revision"],
            "dtype": "bfloat16",
            "attention_backend": "default",
            "engine": "vllm-0.8.5-v1",
            "sampling": {"temperature": 0.6, "top_p": 0.8, "top_k": 20,
                         "presence_penalty": 1.5, "logprobs": 0},
            "prompt_template": "mur.prompts.policy_prompt(step 0, empty trajectory)",
            "probes": probes,
            "probe_count": len(probes),
            "empty_probes": empty,
            "repeated_token_probe": any(p["repeated_token_ids"] for p in probes),
            "generated_tokens": budget.generated,
            "wall_seconds": budget.elapsed,
            "sampler_source": sampler_source_evidence(),
            "exact_saved_ids": True,
        }
        write_json_atomic(Path(args.output), result)
        print(json.dumps({"phase": "generation", "probes": len(probes),
                          "generated_tokens": budget.generated,
                          "empty_probes": empty,
                          "repeated_token_probe": result["repeated_token_probe"]},
                         indent=2))
        return 1 if empty else 0
    except (StageBudgetExceeded, ContextLimit) as e:
        return failure(args.output, f"{type(e).__name__}: {e}",
                       generated_tokens=budget.generated)
    except Exception as e:
        return failure(args.output, f"{type(e).__name__}: {e}",
                       generated_tokens=budget.generated)
    finally:
        if backend is not None:
            backend.close()


def reference_phase(args) -> int:
    cfg = load_json(args.config)
    validate_config(cfg)
    sources = load_json(args.sources)
    generated = load_json(args.probes)
    if generated.get("error"):
        return failure(args.output, "generation phase recorded an error",
                       generated_tokens=generated.get("generated_tokens"))
    probes = generated.get("probes") or []
    if len(probes) != 8:
        return failure(args.output, f"expected eight generated probes, got {len(probes)}",
                       generated_tokens=generated.get("generated_tokens"))
    gen_tokens = int(generated.get("generated_tokens") or 0)
    remaining = int(args.max_generated_tokens) - gen_tokens
    if remaining < len(probes):
        return failure(args.output,
                       f"insufficient generated-token budget for reference probes: "
                       f"remaining={remaining} probes={len(probes)}",
                       generated_tokens=gen_tokens)
    seed = int(cfg["seed"])
    tolerance = cfg["tolerance"]
    path = str(model_root() / sources["policy"]["id"].split("/")[-1])
    budget = StageBudget(max_generated=remaining,
                         max_wall_seconds=args.max_wall_seconds,
                         max_model_len=args.max_model_len)
    backend = None
    results = []
    try:
        backend = VLLMBackend(path, gpu_memory_utilization=0.42,
                              max_model_len=args.max_model_len, seed=seed,
                              enable_prefix_caching=False)
        eos_id = backend.tokenizer.eos_token_id
        for p in probes:
            prompt_ids = [int(x) for x in p["prompt_ids"]]
            output_ids = [int(x) for x in p["output_ids"]]
            if not prompt_ids or not output_ids:
                raise ValueError(f"probe {p['name']} has empty prompt/output ids")
            budget.reserve(1, f"signal-reference-{p['name']}")
            scored = backend.score_nll(prompt_ids, output_ids)
            budget.account("policy", scored["probe_tokens"])
            token_count_ok = (scored["tokens"] == len(output_ids)
                              and len(scored["token_nll"]) == len(output_ids))
            boundary_ok = token_count_ok and scored["tokens"] > 0
            eos_scored = bool(output_ids and output_ids[-1] == eos_id)
            record = {
                "name": p["name"],
                "output_tokens": len(output_ids),
                "prompt_tokens": len(prompt_ids),
                "repeated_token_ids": bool(p.get("repeated_token_ids")),
                "finish_reason": p.get("finish_reason"),
                "first_output_token_id": output_ids[0],
                "last_output_token_id": output_ids[-1],
                "eos_token_id": eos_id,
                "eos_scored": eos_scored,
                "first_scored_position": len(prompt_ids),
                "last_scored_position": len(prompt_ids) + len(output_ids) - 1,
                "scored_token_count_matches": token_count_ok,
                "boundary_scored": boundary_ok,
                "exact_saved_ids_rescored": True,
                "reference_probe_tokens": scored["probe_tokens"],
            }
            try:
                cmp = compare_probe(p.get("generation_token_nll") or [],
                                    scored["token_nll"], tolerance,
                                    output_tokens=len(output_ids))
                record.update(cmp)
                record["passed"] = bool(cmp["passed"] and boundary_ok)
                if not boundary_ok:
                    record["boundary_error"] = "not every saved output position was scored"
            except Exception as e:
                record.update({
                    "passed": False,
                    "error": f"{type(e).__name__}: {e}",
                    "max_abs_token_logprob_diff": None,
                    "mean_abs_token_logprob_diff": None,
                    "abs_mean_diff": None,
                })
            results.append(record)
        reference_tokens = budget.generated
        total_tokens = gen_tokens + reference_tokens
        complete = (len(results) == 8 and all(r["output_tokens"] > 0 for r in results)
                    and all(r.get("boundary_scored") for r in results))
        repeated = any(r["repeated_token_ids"] for r in results)
        all_pass = bool(complete and repeated and all(r.get("passed") for r in results))
        payload = {
            "schema_version": 1,
            "run_id": run_id(),
            "passed": all_pass,
            "validation_reference": VALIDATION_REFERENCE,
            "reference_run": REFERENCE_RUN,
            "cross_engine_hf_parity": CROSS_ENGINE_HF_PARITY,
            "hf_parity_status": "FAILED in r3; retained as a separate diagnostic",
            "probe_count": len(results),
            "required_probe_count": 8,
            "complete_nonempty_probes": complete,
            "repeated_token_probe": repeated,
            "probes": results,
            "tolerance": tolerance,
            "tolerance_reason": ("Unchanged r3 absolute tolerances; the reference "
                                 "engine is the same pinned BF16 vLLM V1 instance, "
                                 "so no cross-engine kernel allowance is being added."),
            "model": sources["policy"]["id"],
            "revision": sources["policy"]["revision"],
            "dtype": "bfloat16",
            "attention_backend": "default",
            "engine": "vllm-0.8.5-v1",
            "reference_sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                                   "presence_penalty": 0.0, "max_tokens": 1,
                                   "prompt_logprobs": 0},
            "reference_prefix_caching": False,
            "generation_sampling": generated.get("sampling"),
            "prompt_template": generated.get("prompt_template"),
            "scoring_path": ("vLLM 0.8.5 V1 prompt_logprobs on exact saved prompt+output "
                             "ids; no decoded-text re-tokenization"),
            "generation_generated_tokens": gen_tokens,
            "reference_probe_tokens": reference_tokens,
            "generated_tokens": total_tokens,
            "extra_probe_tokens_counted": True,
            "generation_wall_seconds": generated.get("wall_seconds"),
            "reference_wall_seconds": budget.elapsed,
            "separate_processes": True,
            "sampler_source": generated.get("sampler_source") or sampler_source_evidence(),
            "max_abs_token_logprob_diff": (max(r["max_abs_token_logprob_diff"] for r in results
                                               if r.get("max_abs_token_logprob_diff") is not None)
                                           if any(r.get("max_abs_token_logprob_diff") is not None
                                                  for r in results) else None),
            "mean_abs_token_logprob_diff": (sum(r["mean_abs_token_logprob_diff"] for r in results
                                                if r.get("mean_abs_token_logprob_diff") is not None)
                                            / max(1, sum(1 for r in results
                                                         if r.get("mean_abs_token_logprob_diff") is not None))),
            "empty_probes": [r["name"] for r in results if r["output_tokens"] == 0],
            "note": ("This gate validates raw generation logprobs against an identity-sampling "
                     "teacher-forced prefill in the same engine. It is not HF numerical parity."),
        }
        write_json_atomic(Path(args.output), payload)
        print(json.dumps(payload, indent=2))
        return 0 if all_pass else 1
    except (StageBudgetExceeded, ContextLimit) as e:
        return failure(args.output, f"{type(e).__name__}: {e}",
                       generated_tokens=gen_tokens + budget.generated)
    except Exception as e:
        return failure(args.output, f"{type(e).__name__}: {e}",
                       generated_tokens=gen_tokens + budget.generated)
    finally:
        if backend is not None:
            backend.close()


def orchestrator(args) -> int:
    out = Path(args.output)
    gen_path = out.with_name(out.stem + "-generation.json")
    if gen_path.exists() or out.exists():
        return failure(args.output, "refusing to reuse an existing validation output",
                       generation_output=str(gen_path))
    base = [sys.executable, "-m", "mur.validate_signal",
            "--config", args.config, "--sources", args.sources,
            "--max-generated-tokens", str(args.max_generated_tokens),
            "--max-wall-seconds", str(args.max_wall_seconds),
            "--max-model-len", str(args.max_model_len)]
    started = time.time()
    rc = subprocess.call(base + ["--phase", "generate", "--output", str(gen_path)],
                         env=os.environ.copy())
    if rc != 0:
        gen_tokens = None
        try:
            gen_tokens = load_json(gen_path).get("generated_tokens")
        except Exception:
            pass
        return failure(args.output, f"generation phase exited {rc}",
                       generated_tokens=gen_tokens,
                       generation_output=str(gen_path))
    # subprocess.call already waited for the short-lived generation process;
    # its own close() shut down the V1 EngineCore child.  A short grace period
    # makes the fresh reference process unambiguous.
    time.sleep(1.0)
    rc = subprocess.call(base + ["--phase", "reference", "--probes", str(gen_path),
                                 "--output", str(out)], env=os.environ.copy())
    if rc != 0 and not out.exists():
        return failure(args.output, f"reference phase exited {rc}",
                       generated_tokens=None)
    try:
        payload = load_json(out)
    except Exception as e:
        return failure(args.output, f"unreadable reference output: {e}")
    payload["orchestrator_wall_seconds"] = time.time() - started
    payload["separate_processes"] = True
    write_json_atomic(out, payload)
    return 0 if payload.get("passed") else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--phase", choices=("generate", "reference"), default=None)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--max-generated-tokens", type=int, default=10_000)
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
