#!/usr/bin/env python3
"""One foreground driver for the GPU prep stage.

Runs signal validation, smoke and verifier as sequential fresh child processes
in their own process groups, samples GPU memory, and enforces cumulative GPU,
token and wall budgets.  It never kills anything outside its own child groups.

Fail-closed rules:
* refuse a run directory that already contains GPU output,
* refuse to start without usable NVML/compute-process telemetry,
* refuse to start from a dirty Git worktree and record the run commit,
* include process cleanup time in the GPU allocation,
* require a phase's exit code, status and cleanup to succeed before advancing,
* count reported tokens exactly and count unknown reservations separately.
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

GPU_OUTPUT_PATHS = [
    "logprob-validation.json",
    "logprob-validation-generation.json",
    "smoke/raw-smoke-traces.jsonl",
    "smoke/run-status.json",
    "smoke/run-config.json",
    "forced/raw-smoke-traces.jsonl",
    "forced/run-status.json",
    "forced/run-config.json",
    "evaluation/metrics.json",
    "evaluation/run-status.json",
    "stage-driver.json",
    "resource-usage.json",
    "gpu-probe.json",
    "smoke-summary.json",
    "forced-trigger.json",
]


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


def find_existing_outputs(run_dir: Path) -> list[str]:
    return [rel for rel in GPU_OUTPUT_PATHS if (run_dir / rel).exists()]


def gpu_snapshot(index: int) -> dict:
    rc, out, err = run_text([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    snap = {"available": False, "error": None, "devices": [], "query_ok": rc == 0}
    if rc != 0:
        snap["error"] = (err or out).strip()[:400] or f"nvidia-smi exit {rc}"
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
    selected = next((d for d in snap["devices"] if d["index"] == index), None)
    if selected is None:
        snap["error"] = f"GPU index {index} not present in nvidia-smi output"
        return snap
    snap.update(selected)
    snap["available"] = True
    return snap


def compute_apps() -> tuple[dict, bool, str | None]:
    """Return (pid->MiB, telemetry_ok, error)."""
    rc, out, err = run_text([
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return {}, False, (err or out).strip()[:400] or f"nvidia-smi exit {rc}"
    apps = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 2:
            return {}, False, f"unparseable compute-app row: {line!r}"
        try:
            apps[int(cols[0])] = float(cols[1])
        except ValueError:
            return {}, False, f"unparseable compute-app row: {line!r}"
    return apps, True, None


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
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def tokens_from_status(name: str, run_dir: Path, grant: int):
    """Return (tokens, source, unknown_reservation)."""
    mapping = {
        "validate": ("logprob-validation.json", "generated_tokens"),
        "smoke": ("smoke/run-status.json", "generated_tokens"),
        "forced": ("forced/run-status.json", "generated_tokens"),
        "evaluate": ("evaluation/run-status.json", "generated_tokens"),
    }
    rel, key = mapping.get(name, (None, None))
    if rel is not None:
        payload = read_json(run_dir / rel, {}) or {}
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value, "reported", False
    return int(grant), "unknown_conservative_reservation", True


def fresh_file(path: Path, since_epoch: float) -> bool:
    try:
        return path.exists() and path.stat().st_mtime >= since_epoch - 0.5
    except OSError:
        return False


def git_state() -> dict:
    rc, out, err = run_text(["git", "rev-parse", "HEAD"])
    head = out.strip() if rc == 0 else None
    rc2, out2, err2 = run_text(["git", "status", "--porcelain"])
    dirty = bool(out2.strip()) if rc2 == 0 else None
    return {"head": head, "dirty": dirty,
            "error": None if (head is not None and dirty is not None)
            else (err or err2 or "git unavailable").strip()[:400]}


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
        self.gpu_work_only = 0.0
        self.spent_tokens = 0
        self.reported_tokens = 0
        self.unknown_reserved_tokens = 0
        self.wall_start = time.monotonic()
        self.started_at = now_iso()
        self.baseline = gpu_snapshot(args.gpu_index)
        self.git = git_state()
        self.existing_outputs = find_existing_outputs(self.run_dir)
        self.peak_new = 0.0
        self.min_free = None
        self.max_our_mib = 0.0
        self.preflight_errors = []
        self.check_preflight()

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

    def check_preflight(self):
        if self.existing_outputs:
            self.preflight_errors.append(
                "refusing reused GPU-output directory; found " +
                ", ".join(self.existing_outputs))
        if not self.baseline.get("available"):
            self.preflight_errors.append(
                "NVML GPU telemetry unavailable: " + str(self.baseline.get("error")))
        else:
            free = self.baseline.get("free_mib")
            if not isinstance(free, (int, float)):
                self.preflight_errors.append("NVML free-memory telemetry unavailable")
            elif free < self.args.min_free_mib:
                self.preflight_errors.append(
                    f"free GPU memory {free:.0f} MiB below floor "
                    f"{self.args.min_free_mib:.0f} MiB")
            used = self.baseline.get("used_mib")
            if not isinstance(used, (int, float)):
                self.preflight_errors.append("NVML used-memory telemetry unavailable")
        apps, ok, err = compute_apps()
        if not ok:
            self.preflight_errors.append(
                "compute-process telemetry unavailable: " + str(err))
        if self.git.get("dirty") is not False:
            self.preflight_errors.append(
                "Git worktree is dirty or unreadable; commit before GPU work: " +
                str(self.git.get("error") or "dirty"))
        if not self.git.get("head"):
            self.preflight_errors.append("cannot record actual run commit (git HEAD missing)")
        task_path = self.run_dir / "task.json"
        if task_path.exists():
            task = read_json(task_path, {}) or {}
            if task.get("run_id") and task.get("run_id") != self.args.run_id:
                self.preflight_errors.append(
                    f"run-id mismatch: task {task.get('run_id')} != driver {self.args.run_id}")

    def phase(self, name: str, module_cmd: list[str], token_grant: int,
              reserve_seconds: float = 0.0):
        if self.preflight_errors:
            return {"name": name, "status": "skipped_preflight", "exit_code": None,
                    "tokens": 0, "phase_ok": False}
        if self.gpu_remaining <= 30:
            self.failures.append(f"{name}: skipped, no GPU budget remaining")
            return {"name": name, "status": "skipped_no_gpu_budget", "exit_code": None,
                    "tokens": 0, "phase_ok": False}
        if self.token_remaining <= 0:
            self.failures.append(f"{name}: skipped, no token budget remaining")
            return {"name": name, "status": "skipped_no_token_budget", "exit_code": None,
                    "tokens": 0, "phase_ok": False}
        wall_cap = min(self.wall_remaining, self.gpu_remaining - reserve_seconds)
        if wall_cap <= 30:
            self.failures.append(f"{name}: skipped, phase cap {wall_cap:.0f}s too small")
            return {"name": name, "status": "skipped_no_phase_budget", "exit_code": None,
                    "tokens": 0, "phase_ok": False}
        log_path = self.log_dir / f"{name}.log"
        env = os.environ.copy()
        env["MUR_RUN_DIR"] = str(self.run_dir)
        env.setdefault("MUR_RUNTIME_ROOT", str(Path.cwd() / "runtime"))
        start = time.monotonic()
        start_epoch = time.time()
        started_at = now_iso()
        proc = subprocess.Popen(module_cmd, cwd=str(Path.cwd()), env=env,
                                stdout=open(log_path, "w", encoding="utf-8"),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        pgid = proc.pid
        status = "ok"
        samples = []
        last_sample = 0.0
        telemetry_failed = False
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
                apps, apps_ok, apps_err = compute_apps()
                if (not snap.get("available")
                        or not isinstance(snap.get("used_mib"), (int, float))
                        or not isinstance(snap.get("free_mib"), (int, float))
                        or not apps_ok):
                    status = "killed_telemetry_unavailable"
                    telemetry_failed = True
                    self.events.append({"at": now_iso(),
                                        "action": "telemetry_unavailable",
                                        "gpu_error": snap.get("error"),
                                        "apps_error": apps_err})
                    kill_group(pgid, signal.SIGTERM, "telemetry_unavailable", self.events)
                    break
                pids = pids_in_group(pgid)
                our_mib = sum(v for p, v in apps.items() if p in pids)
                usage = snap["used_mib"]
                free = snap["free_mib"]
                new_usage = max(0.0, usage - self.baseline.get("used_mib", 0.0))
                observed_new = max(new_usage, our_mib)
                self.peak_new = max(self.peak_new, observed_new)
                self.max_our_mib = max(self.max_our_mib, our_mib)
                self.min_free = free if self.min_free is None else min(self.min_free, free)
                samples.append({"t": round(now - start, 1), "used_mib": usage,
                                "free_mib": free, "our_apps_mib": our_mib,
                                "new_usage_mib": round(observed_new, 1),
                                "pids": sorted(pids)})
                if observed_new > self.args.max_new_mib:
                    status = "killed_memory_ceiling"
                    kill_group(pgid, signal.SIGTERM, "memory_ceiling", self.events)
                    break
                if free < self.args.min_free_mib:
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
        work_seconds = time.monotonic() - start
        # Wait for the engine child to fully disappear before the next model
        # loads.  Cleanup time is part of this phase's GPU allocation.
        leftover = pids_in_group(pgid)
        cleanup_start = time.monotonic()
        cleanup = "clean"
        if leftover:
            cleanup = "waited_clean"
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
                    cleanup = "leftover_survived" if leftover else "killed_leftover"
                else:
                    cleanup = "terminated_leftover"
            self.events.append({"at": now_iso(), "action": "leftover_pids",
                                "pids": sorted(leftover)})
        cleanup_seconds = time.monotonic() - cleanup_start
        duration = time.monotonic() - start  # includes cleanup
        self.gpu_allocated += duration
        self.gpu_work_only += work_seconds
        tokens, token_source, unknown = tokens_from_status(name, self.run_dir, token_grant)
        self.spent_tokens += tokens
        if unknown:
            self.unknown_reserved_tokens += tokens
        else:
            self.reported_tokens += tokens
        phase_ok = (status == "ok" and rc == 0
                    and cleanup in ("clean", "waited_clean")
                    and not telemetry_failed)
        result = {
            "name": name,
            "status": status,
            "exit_code": rc,
            "phase_ok": phase_ok,
            "started_at": started_at,
            "started_epoch": start_epoch,
            "ended_at": now_iso(),
            "work_seconds": round(work_seconds, 2),
            "cleanup_seconds": round(cleanup_seconds, 2),
            "duration_seconds": round(duration, 2),
            "tokens": tokens,
            "token_source": token_source,
            "unknown_reservation": unknown,
            "token_grant": token_grant,
            "gpu_allocated_after_seconds": round(self.gpu_allocated, 2),
            "max_new_usage_mib": round(max((s["new_usage_mib"] for s in samples),
                                           default=0.0), 1),
            "min_free_mib": min((s["free_mib"] for s in samples), default=None),
            "samples": samples[-120:],
            "cleanup": cleanup,
            "telemetry_failed": telemetry_failed,
            "log": f"logs/{name}.log",
        }
        if not phase_ok:
            self.failures.append(
                f"{name}: status={status} exit={rc} cleanup={cleanup} "
                f"telemetry_failed={telemetry_failed}")
        self.phases.append(result)
        self.write_evidence()
        return result

    def write_evidence(self):
        phase_summary = {p["name"]: {"seconds": p["duration_seconds"],
                                     "work_seconds": p["work_seconds"],
                                     "cleanup_seconds": p["cleanup_seconds"],
                                     "tokens": p["tokens"],
                                     "token_source": p["token_source"],
                                     "status": p["status"],
                                     "phase_ok": p["phase_ok"],
                                     "cleanup": p["cleanup"],
                                     "max_new_usage_mib": p["max_new_usage_mib"],
                                     "min_free_mib": p["min_free_mib"]}
                          for p in self.phases}
        driver = {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "run_commit": self.git.get("head"),
            "git_dirty_at_start": self.git.get("dirty"),
            "git_error": self.git.get("error"),
            "started_at": self.started_at,
            "updated_at": now_iso(),
            "wall_seconds": round(self.wall_elapsed, 2),
            "gpu_allocated_seconds": round(self.gpu_allocated, 2),
            "gpu_work_seconds": round(self.gpu_work_only, 2),
            "total_generated_tokens": self.spent_tokens,
            "reported_generated_tokens": self.reported_tokens,
            "unknown_reserved_tokens": self.unknown_reserved_tokens,
            "token_accounting": ("total = reported actual tokens + conservative "
                                 "reservations for phases with no reported count"),
            "phases": self.phases,
            "phase_summary": phase_summary,
            "failures": self.failures,
            "preflight_errors": self.preflight_errors,
            "existing_gpu_outputs": self.existing_outputs,
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
            "run_commit": self.git.get("head"),
            "wall_seconds": round(self.wall_elapsed, 2),
            "gpu_allocated_seconds": round(self.gpu_allocated, 2),
            "gpu_work_seconds": round(self.gpu_work_only, 2),
            "generated_tokens_total": self.spent_tokens,
            "reported_generated_tokens": self.reported_tokens,
            "unknown_reserved_tokens": self.unknown_reserved_tokens,
            "gpu_budget_seconds": self.args.gpu_budget_seconds,
            "token_budget": self.args.token_budget,
            "wall_budget_seconds": self.args.wall_budget_seconds,
            "peak_new_usage_mib": round(self.peak_new, 1),
            "peak_our_processes_mib": round(self.max_our_mib, 1),
            "min_free_mib": self.min_free,
            "per_phase": phase_summary,
            "cleanup": {p["name"]: p["cleanup"] for p in self.phases},
            "preflight_errors": self.preflight_errors,
        }
        write_json_atomic(self.run_dir / "resource-usage.json", usage)
        final = gpu_snapshot(self.args.gpu_index)
        probe = {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "run_commit": self.git.get("head"),
            "gpu_index": self.args.gpu_index,
            "observed_at": now_iso(),
            "baseline": self.baseline,
            "final": final,
            "peak_new_usage_mib": round(self.peak_new, 1),
            "min_free_mib": self.min_free,
            "max_new_usage_limit_mib": self.args.max_new_mib,
            "min_free_limit_mib": self.args.min_free_mib,
            "telemetry_ok": bool(self.baseline.get("available") and final.get("available")),
            "phases": [{"name": p["name"], "max_new_usage_mib": p["max_new_usage_mib"],
                        "min_free_mib": p["min_free_mib"], "status": p["status"],
                        "phase_ok": p["phase_ok"], "cleanup": p["cleanup"]}
                       for p in self.phases],
            "preflight_errors": self.preflight_errors,
        }
        write_json_atomic(self.run_dir / "gpu-probe.json", probe)


def validation_passed(run_dir: Path, result: dict) -> bool:
    if not result.get("phase_ok"):
        return False
    path = run_dir / "logprob-validation.json"
    if not fresh_file(path, result.get("started_epoch", 0)):
        return False
    payload = read_json(path, {}) or {}
    return (payload.get("passed") is True
            and payload.get("validation_reference")
            == "same_engine_identity_sampling_teacher_forced_prefill"
            and payload.get("cross_engine_hf_parity") is False
            and payload.get("probe_count") == 8
            and payload.get("complete_nonempty_probes") is True)


def smoke_complete(run_dir: Path, result: dict) -> bool:
    if not result.get("phase_ok"):
        return False
    status = read_json(run_dir / "smoke" / "run-status.json", {}) or {}
    traces = run_dir / "smoke" / "raw-smoke-traces.jsonl"
    return (fresh_file(run_dir / "smoke" / "run-status.json",
                       result.get("started_epoch", 0))
            and status.get("complete") is True and status.get("exit_code") == 0
            and status.get("errors") == 0 and traces.exists())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", default="mur-20261024-prep")
    ap.add_argument("--config", default="configs/core.json")
    ap.add_argument("--validation-config", default="configs/validation.json")
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--methods", nargs="+", default=["cot", "per_step", "mur"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--gpu-index", type=int, default=2)
    ap.add_argument("--gpu-budget-seconds", type=float, default=3000)
    ap.add_argument("--wall-budget-seconds", type=float, default=7000)
    ap.add_argument("--token-budget", type=int, default=1_000_000)
    ap.add_argument("--max-new-mib", type=float, default=14336)
    ap.add_argument("--min-free-mib", type=float, default=4096)
    args = ap.parse_args(argv)
    d = Driver(args)
    py = sys.executable
    out = d.run_dir
    eval_reserve_tokens = 60_000
    forced_reserve_tokens = 10_000

    if d.preflight_errors:
        d.write_evidence()
        print(json.dumps({"run_id": args.run_id, "status": "BLOCKED_PREFLIGHT",
                          "preflight_errors": d.preflight_errors}, indent=2))
        return 2

    sres = None
    eres = None
    validate_grant = min(10_000, d.token_remaining)
    vres = d.phase("validate", [
        py, "-m", "mur.validate_signal",
        "--output", str(out / "logprob-validation.json"),
        "--config", args.validation_config,
        "--sources", args.sources,
        "--max-generated-tokens", str(validate_grant),
        "--max-wall-seconds", str(min(600, max(60, d.gpu_remaining - 300))),
        "--max-model-len", "4096",
    ], validate_grant, reserve_seconds=300)
    validation_ok = validation_passed(out, vres)
    if not validation_ok:
        d.failures.append("validation did not pass cleanly; smoke not launched")

    if validation_ok:
        smoke_grant = max(0, d.token_remaining - eval_reserve_tokens - forced_reserve_tokens)
        if smoke_grant < 20_000:
            d.failures.append("smoke: insufficient remaining token budget")
        else:
            reserve = 400 if d.gpu_remaining > 700 else 100
            sres = d.phase("smoke", [
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
            if not smoke_complete(out, sres):
                d.failures.append(
                    "smoke did not complete cleanly; forced/evaluate not launched")

    smoke_ok = bool(validation_ok and sres and smoke_complete(out, sres))
    if smoke_ok:
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
            forced_path = out / "forced" / "raw-smoke-traces.jsonl"
            if forced_path.exists():
                for line in forced_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        forced_triggers += sum(1 for s in rec.get("steps", [])
                                               if s.get("triggered"))
            forced_ok = res.get("phase_ok") is True and forced_triggers > 0
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
            if not forced_ok:
                d.failures.append("forced trigger exercise did not pass cleanly")
        else:
            write_json_atomic(out / "forced-trigger.json", {
                "schema_version": 1, "run_id": args.run_id,
                "engineering_only": True, "excluded_from_scientific_smoke": True,
                "passed": True, "genuine_triggers": triggers,
                "note": "Genuine MUR triggers observed; no forced case needed."})

    traces_path = out / "smoke" / "raw-smoke-traces.jsonl"
    traces_exist = traces_path.exists()
    if smoke_ok and traces_exist and d.gpu_remaining > 30:
        eval_grant = d.token_remaining
        eres = d.phase("evaluate", [
            py, "-m", "mur.evaluate",
            "--input-dir", str(out / "smoke"),
            "--output-dir", str(out / "evaluation"),
            "--sources", args.sources,
            "--max-generated-tokens", str(eval_grant),
            "--max-wall-seconds", str(max(30, d.gpu_remaining)),
            "--max-model-len", "4096",
        ], eval_grant, reserve_seconds=0)
        estatus = read_json(out / "evaluation" / "run-status.json", {}) or {}
        metrics = read_json(out / "evaluation" / "metrics.json", {}) or {}
        traces = []
        if traces_path.exists():
            traces = [json.loads(x) for x in traces_path.read_text(encoding="utf-8").splitlines()
                      if x.strip()]
        eval_ok = (eres.get("phase_ok") is True
                   and estatus.get("exit_code") == 0
                   and estatus.get("partial") is False
                   and len(metrics.get("rows", [])) == len(traces)
                   and len(traces) > 0)
        if not eval_ok:
            d.failures.append("evaluate did not complete cleanly")
    elif smoke_ok:
        d.failures.append("evaluate: not launched (no traces or no GPU budget)")
    else:
        d.failures.append("evaluate: not launched (validation/smoke gate failed)")

    d.write_evidence()
    # CPU-only summary after all model processes are gone.
    summary_rc = subprocess.call([py, "-m", "mur.summarize", "--run-dir", str(out)],
                                 cwd=str(Path.cwd()), env=os.environ.copy())
    if summary_rc != 0:
        d.failures.append(f"summarize exited {summary_rc}")
    d.write_evidence()
    print(json.dumps({"run_id": args.run_id,
                      "run_commit": d.git.get("head"),
                      "gpu_allocated_seconds": round(d.gpu_allocated, 2),
                      "reported_tokens": d.reported_tokens,
                      "unknown_reserved_tokens": d.unknown_reserved_tokens,
                      "tokens": d.spent_tokens,
                      "wall_seconds": round(d.wall_elapsed, 2),
                      "phases": [(p["name"], p["status"], p["exit_code"], p["cleanup"])
                                 for p in d.phases],
                      "failures": d.failures}, indent=2))
    return 0 if not d.failures else 1


if __name__ == "__main__":
    sys.exit(main())
