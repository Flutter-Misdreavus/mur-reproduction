#!/usr/bin/env python3
"""One foreground driver for the GPU prep stage.

Runs validation, smoke and verifier as sequential fresh child processes in
their own process groups, samples GPU memory, and enforces cumulative GPU,
token and wall budgets.  It never kills anything outside its own child groups.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def run_text(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def gpu_snapshot(index: int) -> dict:
    rc, out, err = run_text([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    snap = {"available": False, "error": None, "devices": []}
    if rc != 0:
        snap["error"] = (err or out).strip()[:400]
        return snap
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        try:
            dev = {"index": int(cols[0]), "name": cols[1],
                   "total_mib": float(cols[2]), "used_mib": float(cols[3]),
                   "free_mib": float(cols[4]), "util_pct": float(cols[5])}
        except ValueError:
            continue
        snap["devices"].append(dev)
    snap["available"] = any(d["index"] == index for d in snap["devices"])
    for d in snap["devices"]:
        if d["index"] == index:
            snap.update(d)
    return snap


def compute_apps() -> dict:
    rc, out, _ = run_text([
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    apps = {}
    if rc != 0:
        return apps
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 2:
            continue
        try:
            apps[int(cols[0])] = float(cols[1])
        except ValueError:
            continue
    return apps


def pids_in_group(pgid: int) -> set[int]:
    pids = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text(encoding="utf-8")
            rest = stat.rsplit(")", 1)[1].split()
            # state, ppid, pgrp
            if int(rest[2]) == pgid:
                pids.add(int(entry))
        except Exception:
            continue
    return pids


def kill_group(pgid: int, sig, label: str, events: list):
    try:
        os.killpg(pgid, sig)
        events.append({"at": now_iso(), "action": "killpg", "signal": int(sig),
                       "pgid": pgid, "label": label})
        return True
    except ProcessLookupError:
        return False
    except Exception as e:
        events.append({"at": now_iso(), "action": "killpg_error", "error": str(e),
                       "pgid": pgid, "label": label})
        return False


def read_json(path: Path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def tokens_from_status(name: str, run_dir: Path, grant: int):
    if name == "validate":
        payload = read_json(run_dir / "logprob-validation.json", {}) or {}
        value = payload.get("generated_tokens")
    elif name == "smoke":
        payload = read_json(run_dir / "smoke" / "run-status.json", {}) or {}
        value = payload.get("generated_tokens")
    elif name == "forced":
        payload = read_json(run_dir / "forced" / "run-status.json", {}) or {}
        value = payload.get("generated_tokens")
    elif name == "evaluate":
        payload = read_json(run_dir / "evaluation" / "run-status.json", {}) or {}
        value = payload.get("generated_tokens")
    else:
        value = None
    if isinstance(value, int) and value >= 0:
        return value, "reported"
    return int(grant), "grant_assumed"


class Driver:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.phases = []
        self.failures = []
        self.gpu_allocated = 0.0
        self.spent_tokens = 0
        self.wall_start = time.monotonic()
        self.baseline = gpu_snapshot(args.gpu_index)
        self.peak_new = 0.0
        self.min_free = None
        self.max_our_mib = 0.0

    @property
    def wall_elapsed(self):
        return time.monotonic() - self.wall_start

    @property
    def gpu_remaining(self):
        return max(0.0, self.args.gpu_budget_seconds - self.gpu_allocated)

    @property
    def wall_remaining(self):
        return max(0.0, self.args.wall_budget_seconds - self.wall_elapsed)

    @property
    def token_remaining(self):
        return max(0, self.args.token_budget - self.spent_tokens)

    def base_cmd(self):
        return [sys.executable, "-m"]

    def phase(self, name: str, module_cmd: list[str], token_grant: int,
              reserve_seconds: float = 0.0):
        if self.gpu_remaining <= 30:
            self.failures.append(f"{name}: skipped, no GPU budget remaining")
            return {"name": name, "status": "skipped_no_gpu_budget", "exit_code": None,
                    "tokens": 0}
        if self.token_remaining <= 0:
            self.failures.append(f"{name}: skipped, no token budget remaining")
            return {"name": name, "status": "skipped_no_token_budget", "exit_code": None,
                    "tokens": 0}
        wall_cap = min(self.wall_remaining, self.gpu_remaining - reserve_seconds)
        if wall_cap <= 30:
            self.failures.append(f"{name}: skipped, phase cap {wall_cap:.0f}s too small")
            return {"name": name, "status": "skipped_no_phase_budget", "exit_code": None,
                    "tokens": 0}
        log_path = self.log_dir / f"{name}.log"
        env = os.environ.copy()
        env["MUR_RUN_DIR"] = str(self.run_dir)
        env.setdefault("MUR_RUNTIME_ROOT", str(Path.cwd() / "runtime"))
        proc = subprocess.Popen(module_cmd, cwd=str(Path.cwd()), env=env,
                                stdout=open(log_path, "w", encoding="utf-8"),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        pgid = proc.pid
        start = time.monotonic()
        start_iso = now_iso()
        status = "ok"
        samples = []
        last_sample = 0.0
        data = compute_apps()
        while proc.poll() is None:
            now = time.monotonic()
            if self.gpu_allocated + (now - start) > self.args.gpu_budget_seconds:
                status = "killed_gpu_budget"
                kill_group(pgid, signal.SIGTERM, "gpu_budget", self.events)
                break
            if self.wall_elapsed > self.args.wall_budget_seconds:
                status = "killed_wall_budget"
                kill_group(pgid, signal.SIGTERM, "wall_budget", self.events)
                break
            if now - start > wall_cap:
                status = "killed_phase_cap"
                kill_group(pgid, signal.SIGTERM, "phase_cap", self.events)
                break
            if now - last_sample >= 2.0:
                last_sample = now
                snap = gpu_snapshot(self.args.gpu_index)
                pids = pids_in_group(pgid)
                apps = compute_apps()
                our_mib = sum(v for p, v in apps.items() if p in pids)
                usage = snap.get("used_mib")
                new_usage = (max(0.0, usage - self.baseline.get("used_mib", 0.0))
                             if isinstance(usage, (int, float)) else 0.0)
                observed_new = max(new_usage, our_mib)
                free = snap.get("free_mib")
                self.peak_new = max(self.peak_new, observed_new)
                self.max_our_mib = max(self.max_our_mib, our_mib)
                if isinstance(free, (int, float)):
                    self.min_free = free if self.min_free is None else min(self.min_free, free)
                samples.append({"t": round(now - start, 1), "used_mib": usage,
                                "free_mib": free, "our_apps_mib": our_mib,
                                "new_usage_mib": round(observed_new, 1),
                                "pids": sorted(pids)})
                if observed_new > self.args.max_new_mib:
                    status = "killed_memory_ceiling"
                    kill_group(pgid, signal.SIGTERM, "memory_ceiling", self.events)
                    break
                if isinstance(free, (int, float)) and free < self.args.min_free_mib:
                    status = "killed_free_memory_floor"
                    kill_group(pgid, signal.SIGTERM, "free_memory_floor", self.events)
                    break
            time.sleep(1.0)
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            kill_group(pgid, signal.SIGKILL, "wait_timeout", self.events)
            rc = proc.wait()
            status = "killed_sigkill"
        duration = time.monotonic() - start
        # Wait for the engine child to fully disappear before the next model loads.
        leftover = pids_in_group(pgid)
        cleanup = "clean"
        deadline = time.monotonic() + 15
        while leftover and time.monotonic() < deadline:
            time.sleep(0.5)
            leftover = pids_in_group(pgid)
        if leftover:
            kill_group(pgid, signal.SIGTERM, "leftover", self.events)
            time.sleep(2)
            leftover = pids_in_group(pgid)
            if leftover:
                kill_group(pgid, signal.SIGKILL, "leftover_sigkill", self.events)
                time.sleep(1)
                leftover = pids_in_group(pgid)
            cleanup = "killed_leftover" if leftover else "terminated_leftover"
            self.events.append({"at": now_iso(), "action": "leftover_pids",
                                "pids": sorted(leftover)})
        self.gpu_allocated += duration
        tokens, token_source = tokens_from_status(name, self.run_dir, token_grant)
        self.spent_tokens += tokens
        result = {
            "name": name,
            "status": status,
            "exit_code": rc,
            "started_at": start_iso,
            "ended_at": now_iso(),
            "duration_seconds": round(duration, 2),
            "tokens": tokens,
            "token_source": token_source,
            "token_grant": token_grant,
            "gpu_allocated_after_seconds": round(self.gpu_allocated, 2),
            "max_new_usage_mib": round(max((s["new_usage_mib"] for s in samples), default=0.0), 1),
            "min_free_mib": min((s["free_mib"] for s in samples if isinstance(s["free_mib"], (int, float))), default=None),
            "samples": samples[-120:],
            "cleanup": cleanup,
            "log": f"logs/{name}.log",
        }
        if status != "ok" or rc != 0:
            self.failures.append(f"{name}: status={status} exit={rc} cleanup={cleanup}")
        self.phases.append(result)
        self.write_evidence()
        return result

    def write_evidence(self):
        driver = {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "started_at": now_iso(),
            "wall_seconds": round(self.wall_elapsed, 2),
            "gpu_allocated_seconds": round(self.gpu_allocated, 2),
            "total_generated_tokens": self.spent_tokens,
            "phases": self.phases,
            "failures": self.failures,
            "limits": {
                "gpu_seconds": self.args.gpu_budget_seconds,
                "token_budget": self.args.token_budget,
                "wall_seconds": self.args.wall_budget_seconds,
                "max_new_mib": self.args.max_new_mib,
                "min_free_mib": self.args.min_free_mib,
                "gpu_index": self.args.gpu_index,
            },
            "events": self.events,
        }
        write_json_atomic(self.run_dir / "stage-driver.json", driver)
        usage = {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "wall_seconds": round(self.wall_elapsed, 2),
            "gpu_allocated_seconds": round(self.gpu_allocated, 2),
            "generated_tokens_total": self.spent_tokens,
            "gpu_budget_seconds": self.args.gpu_budget_seconds,
            "token_budget": self.args.token_budget,
            "wall_budget_seconds": self.args.wall_budget_seconds,
            "peak_new_usage_mib": round(self.peak_new, 1),
            "peak_our_processes_mib": round(self.max_our_mib, 1),
            "min_free_mib": self.min_free,
            "per_phase": {p["name"]: {"seconds": p["duration_seconds"],
                                     "tokens": p["tokens"],
                                     "status": p["status"],
                                     "max_new_usage_mib": p["max_new_usage_mib"],
                                     "min_free_mib": p["min_free_mib"]}
                          for p in self.phases},
            "cleanup": {p["name"]: p["cleanup"] for p in self.phases},
        }
        write_json_atomic(self.run_dir / "resource-usage.json", usage)
        probe = {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "gpu_index": self.args.gpu_index,
            "observed_at": now_iso(),
            "baseline": self.baseline,
            "final": gpu_snapshot(self.args.gpu_index),
            "peak_new_usage_mib": round(self.peak_new, 1),
            "min_free_mib": self.min_free,
            "max_new_usage_limit_mib": self.args.max_new_mib,
            "min_free_limit_mib": self.args.min_free_mib,
            "phases": [{"name": p["name"], "max_new_usage_mib": p["max_new_usage_mib"],
                        "min_free_mib": p["min_free_mib"], "status": p["status"]}
                       for p in self.phases],
        }
        write_json_atomic(self.run_dir / "gpu-probe.json", probe)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", default="mur-20261024-prep")
    ap.add_argument("--config", default="configs/core.json")
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--methods", nargs="+", default=["cot", "per_step", "mur"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--gpu-index", type=int, default=2)
    ap.add_argument("--gpu-budget-seconds", type=float, default=3600)
    ap.add_argument("--wall-budget-seconds", type=float, default=7200)
    ap.add_argument("--token-budget", type=int, default=1_000_000)
    ap.add_argument("--max-new-mib", type=float, default=14336)
    ap.add_argument("--min-free-mib", type=float, default=4096)
    args = ap.parse_args(argv)
    d = Driver(args)
    py = sys.executable
    out = d.run_dir
    eval_reserve_tokens = 60_000
    forced_reserve_tokens = 10_000

    validate_grant = min(10_000, d.token_remaining)
    d.phase("validate", [
        py, "-m", "mur.validate_logprobs",
        "--output", str(out / "logprob-validation.json"),
        "--sources", args.sources,
        "--max-generated-tokens", str(validate_grant),
        "--max-wall-seconds", str(min(600, max(60, d.gpu_remaining - 300))),
        "--max-model-len", "4096",
    ], validate_grant, reserve_seconds=300)
    validation_ok = read_json(out / "logprob-validation.json", {}).get("passed") is True

    if validation_ok:
        smoke_grant = max(0, d.token_remaining - eval_reserve_tokens - forced_reserve_tokens)
        if smoke_grant < 20_000:
            d.failures.append("smoke: insufficient remaining token budget")
        else:
            reserve = 400 if d.gpu_remaining > 700 else 100
            d.phase("smoke", [
                py, "-m", "mur.run",
                "--config", args.config, "--split", "dev",
                "--methods", *args.methods,
                "--run-id", args.run_id,
                "--output-dir", str(out / "smoke"),
                "--limit", str(args.limit),
                "--max-generated-tokens", str(smoke_grant),
                "--max-wall-seconds", str(max(60, d.gpu_remaining - reserve)),
                "--max-model-len", "16384",
            ], smoke_grant, reserve_seconds=reserve)
    else:
        d.failures.append("validation did not pass; smoke not launched")

    smoke_status = read_json(out / "smoke" / "run-status.json", {}) or {}
    smoke_ok = smoke_status.get("complete") is True
    forced_ok = False
    if validation_ok and smoke_ok:
        traces_path = out / "smoke" / "raw-smoke-traces.jsonl"
        triggers = 0
        if traces_path.exists():
            for line in traces_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("method") == "mur":
                    triggers += sum(1 for s in rec.get("steps", []) if s.get("triggered"))
        if triggers == 0:
            grant = min(forced_reserve_tokens, d.token_remaining - eval_reserve_tokens)
            grant = max(2000, grant)
            res = d.phase("forced", [
                py, "-m", "mur.run",
                "--config", args.config, "--split", "dev",
                "--methods", "mur", "--run-id", args.run_id + "-forced",
                "--output-dir", str(out / "forced"), "--limit", "1",
                "--max-generated-tokens", str(grant),
                "--max-wall-seconds", str(max(30, d.gpu_remaining - 120)),
                "--max-model-len", "16384", "--force-trigger-step", "1",
            ], grant, reserve_seconds=120)
            forced_triggers = 0
            if (out / "forced" / "raw-smoke-traces.jsonl").exists():
                for line in (out / "forced" / "raw-smoke-traces.jsonl").read_text(
                        encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        forced_triggers += sum(1 for s in rec.get("steps", [])
                                               if s.get("triggered"))
            forced_ok = forced_triggers > 0 and res.get("status") == "ok"
            write_json_atomic(out / "forced-trigger.json", {
                "schema_version": 1,
                "run_id": args.run_id,
                "engineering_only": True,
                "excluded_from_scientific_smoke": True,
                "passed": forced_ok,
                "triggered_steps": forced_triggers,
                "note": ("Engineering-only forced trigger at step 1 after zero "
                         "genuine MUR triggers; not part of the 20x3 metrics."),
                "phase": res,
            })
        else:
            write_json_atomic(out / "forced-trigger.json", {
                "schema_version": 1, "run_id": args.run_id,
                "engineering_only": True, "excluded_from_scientific_smoke": True,
                "passed": True, "genuine_triggers": triggers,
                "note": "Genuine MUR triggers observed; no forced case needed."})

    traces_exist = (out / "smoke" / "raw-smoke-traces.jsonl").exists()
    smoke_rc = smoke_status.get("exit_code", None)
    should_evaluate = traces_exist and (smoke_ok or smoke_rc == 3)
    if should_evaluate and d.gpu_remaining > 30:
        eval_grant = d.token_remaining
        d.phase("evaluate", [
            py, "-m", "mur.evaluate",
            "--input-dir", str(out / "smoke"),
            "--output-dir", str(out / "evaluation"),
            "--sources", args.sources,
            "--max-generated-tokens", str(eval_grant),
            "--max-wall-seconds", str(max(30, d.gpu_remaining)),
            "--max-model-len", "4096",
        ], eval_grant, reserve_seconds=0)
    elif traces_exist:
        d.failures.append("evaluate: not launched (smoke incomplete/failed or no GPU budget)")
    else:
        d.failures.append("evaluate: no smoke traces")

    d.write_evidence()
    # CPU-only summary after all model processes are gone.
    subprocess.call([py, "-m", "mur.summarize", "--run-dir", str(out)],
                    cwd=str(Path.cwd()), env=os.environ.copy())
    d.write_evidence()
    print(json.dumps({"run_id": args.run_id,
                      "gpu_allocated_seconds": round(d.gpu_allocated, 2),
                      "tokens": d.spent_tokens,
                      "wall_seconds": round(d.wall_elapsed, 2),
                      "phases": [(p["name"], p["status"], p["exit_code"])
                                 for p in d.phases],
                      "failures": d.failures}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
