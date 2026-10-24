#!/usr/bin/env python3
"""Bounded foreground driver for the frozen fixed100 exploratory test.

Phases:
1. engineering identity check on dev indices 0 and 3 (<=300 GPU seconds)
2. four independent 25-question chunks, each generation process followed by a
   separate verifier-grading process after policy/critic have fully exited
3. CPU-only aggregation, paired bootstrap and blind-audit construction

It never restarts a chunk, never stops on accuracy, and kills only its own child
process groups.  GPU allocation includes process cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mur import testdata  # noqa: E402

GPU_MAX_NEW_MIB = 14336
GPU_MIN_FREE_MIB = 4096
GPU_PREFLIGHT_MIN_FREE_MIB = 8192
DISK_MIN_FREE_BYTES = 25 * (1 << 30)
NEW_STORAGE_LIMIT_BYTES = 20 * (1 << 30)
CHUNK_GENERATION_SECONDS = 3300
GRADING_RESERVE_SECONDS = 600
GRADING_RESERVE_TOKENS = 30000
CHUNK_TOKEN_GRANT = 300000
EXPECTED_ROWS = 300


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
    except Exception as exc:
        return 1, "", f"{type(exc).__name__}: {exc}"


def all_gpu_snapshot() -> dict:
    snap = {"available": False, "devices": [], "error": None}
    rc, out, err = run_text([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        snap["error"] = (err or out).strip()[:400]
        return snap
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        try:
            snap["devices"].append({
                "index": int(cols[0]), "name": cols[1],
                "total_mib": float(cols[2]), "used_mib": float(cols[3]),
                "free_mib": float(cols[4]), "util_pct": float(cols[5]),
            })
        except ValueError:
            continue
    snap["available"] = bool(snap["devices"])
    return snap


def select_gpu(snap: dict) -> dict | None:
    candidates = [d for d in snap.get("devices", [])
                  if d.get("free_mib", 0) >= GPU_PREFLIGHT_MIN_FREE_MIB]
    if not candidates:
        return None
    # Prefer the most free memory, then the least busy device.
    return sorted(candidates, key=lambda d: (-d["free_mib"], d["util_pct"], d["index"]))[0]


def compute_apps() -> tuple[dict, bool, str | None]:
    rc, out, err = run_text([
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return {}, False, (err or out).strip()[:400]
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
    except ProcessLookupError:
        pass
    except Exception as exc:
        events.append({"at": now_iso(), "action": "killpg_error", "error": str(exc),
                       "pgid": pgid, "label": label})


def git_state() -> dict:
    rc, out, err = run_text(["git", "rev-parse", "HEAD"])
    head = out.strip() if rc == 0 else None
    rc2, out2, err2 = run_text(["git", "status", "--porcelain"])
    dirty = bool(out2.strip()) if rc2 == 0 else None
    rc3, out3, _ = run_text(["git", "branch", "--show-current"])
    branch = out3.strip() if rc3 == 0 else None
    return {"head": head, "dirty": dirty, "branch": branch,
            "error": None if head and dirty is not None else (err or err2 or "git failed")}


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class TestDriver:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.phases = []
        self.failures = []
        self.preflight_errors = []
        self.started_at = now_iso()
        self.wall_start = time.monotonic()
        self.gpu_allocated = 0.0
        self.gpu_work = 0.0
        self.spent_tokens = 0
        self.reported_tokens = 0
        self.unknown_reserved_tokens = 0
        self.peak_new = 0.0
        self.max_our_mib = 0.0
        self.min_free = None
        self.task = load_json(Path(args.task_file), {}) or {}
        self.run_id = self.task.get("run_id") or args.run_id
        self.git = git_state()
        self.initial_commit = self.git.get("head")
        self.gpu_snap = all_gpu_snapshot()
        if args.gpu_index is not None:
            self.gpu = next((d for d in self.gpu_snap.get("devices", [])
                             if d["index"] == args.gpu_index), None)
        else:
            self.gpu = select_gpu(self.gpu_snap) if self.gpu_snap.get("available") else None
        self.gpu_index = int(self.gpu["index"]) if self.gpu else args.gpu_index
        self.baseline = self.gpu
        self.check_preflight()

    @property
    def wall_elapsed(self):
        return time.monotonic() - self.wall_start

    @property
    def gpu_remaining(self):
        return max(0.0, self.args.gpu_budget_seconds - self.gpu_allocated)

    @property
    def token_remaining(self):
        return max(0, self.args.token_budget - self.spent_tokens)

    def check_preflight(self):
        task_file = Path(self.args.task_file)
        if not task_file.exists():
            self.preflight_errors.append(f"missing task file: {task_file}")
            return
        if self.task.get("run_id") != self.run_id:
            self.preflight_errors.append("task run_id mismatch")
        if self.task.get("ready") is not True:
            self.preflight_errors.append("task is not ready")
        vc = self.task.get("version_control") or {}
        if vc.get("visibility") != "private" or not vc.get("repository"):
            self.preflight_errors.append("missing private version_control")
        if vc.get("branch") and self.git.get("branch") != vc.get("branch"):
            self.preflight_errors.append(
                f"branch {self.git.get('branch')} != {vc.get('branch')}")
        if self.git.get("dirty") is not False:
            self.preflight_errors.append("Git worktree is dirty; commit before GPU work")
        if not self.git.get("head"):
            self.preflight_errors.append("cannot record actual run commit")
        # Test access authorization and manifest.
        try:
            parent = self.task.get("parent_run_id")
            review = Path("experiments") / str(parent) / "review.md"
            self.authorization = testdata.authorize_test(
                task_file, review, self.run_id)
        except Exception as exc:
            self.authorization = None
            self.preflight_errors.append(f"test authorization failed: {exc}")
        manifest_path = self.run_dir / "test-manifest.json"
        manifest = load_json(manifest_path)
        self.manifest = manifest
        if not manifest:
            self.preflight_errors.append(f"missing test manifest: {manifest_path}")
        else:
            try:
                sources = load_json(Path("configs/sources.json"), {})
                rows, _ = testdata.load_source_rows(Path("runtime"), sources)
                problems = testdata.verify_test_manifest(manifest, rows, sources)
                if problems:
                    self.preflight_errors.append(
                        "test manifest failed verification: " + "; ".join(problems))
                split_path = Path("runtime") / "data" / "test100.jsonl"
                if not split_path.exists():
                    info = testdata.build_test_generation_file(rows, manifest, Path("runtime"))
                    self.events.append({"at": now_iso(), "action": "rebuilt_test_split",
                                        "sha256": info["sha256"]})
                got = testdata.sha256_file(split_path)
                want = (manifest.get("generation_split") or {}).get("sha256")
                if got != want:
                    self.preflight_errors.append(
                        f"test100 file sha256 {got} != manifest {want}")
            except Exception as exc:
                self.preflight_errors.append(f"test split check failed: {exc}")
        # GPU and storage headroom.
        if not self.gpu_snap.get("available"):
            self.preflight_errors.append(
                "NVML GPU telemetry unavailable: " + str(self.gpu_snap.get("error")))
        elif self.gpu is None:
            self.preflight_errors.append(
                f"no GPU with at least {GPU_PREFLIGHT_MIN_FREE_MIB} MiB free")
        else:
            apps, ok, err = compute_apps()
            if not ok:
                self.preflight_errors.append(f"compute-process telemetry failed: {err}")
        disk = shutil.disk_usage(Path("runtime"))
        self.disk_preflight = {"free_bytes": disk.free, "total_bytes": disk.total}
        if disk.free < DISK_MIN_FREE_BYTES:
            self.preflight_errors.append(
                f"disk free {disk.free / (1 << 30):.1f} GiB below 25 GiB")
        # Refuse reused GPU outputs.
        existing = []
        for rel in ("engineering/engineering-check.json", "raw-traces.jsonl",
                    "metrics.json", "report.json"):
            if (self.run_dir / rel).exists():
                existing.append(rel)
        for i in range(4):
            if (self.run_dir / "chunks" / f"chunk-{i:02d}").exists():
                existing.append(f"chunks/chunk-{i:02d}")
        self.existing_outputs = existing
        if existing:
            self.preflight_errors.append(
                "refusing reused GPU-output run directory: " + ", ".join(existing))
        # Keep the declared scientific/resource limits fixed.
        if self.args.gpu_budget_seconds > 14400 or self.args.token_budget > 2500000:
            self.preflight_errors.append("driver budget exceeds the frozen task limit")

    def write_evidence(self):
        phase_summary = {p["name"]: {k: p[k] for k in (
            "status", "exit_code", "phase_ok", "duration_seconds", "work_seconds",
            "cleanup_seconds", "tokens", "token_source", "cleanup",
            "max_new_usage_mib", "min_free_mib")} for p in self.phases}
        driver = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_commit": self.initial_commit,
            "git_dirty_at_start": self.git.get("dirty"),
            "started_at": self.started_at,
            "updated_at": now_iso(),
            "wall_seconds": round(self.wall_elapsed, 2),
            "gpu_allocated_seconds": round(self.gpu_allocated, 2),
            "gpu_work_seconds": round(self.gpu_work, 2),
            "total_generated_tokens": self.spent_tokens,
            "reported_generated_tokens": self.reported_tokens,
            "unknown_reserved_tokens": self.unknown_reserved_tokens,
            "phases": self.phases,
            "phase_summary": phase_summary,
            "failures": self.failures,
            "preflight_errors": self.preflight_errors,
            "existing_gpu_outputs": self.existing_outputs,
            "gpu_selection": {"selected_index": self.gpu_index,
                              "preflight_device": self.gpu,
                              "all_devices": self.gpu_snap.get("devices"),
                              "cuda_visible_devices": str(self.gpu_index),
                              "one_gpu": True},
            "limits": {
                "gpu_seconds": self.args.gpu_budget_seconds,
                "token_budget": self.args.token_budget,
                "wall_seconds": self.args.wall_budget_seconds,
                "max_new_mib": GPU_MAX_NEW_MIB,
                "min_free_mib": GPU_MIN_FREE_MIB,
                "gpu_index": self.gpu_index,
                "chunk_generation_seconds": CHUNK_GENERATION_SECONDS,
                "grading_reserve_seconds": GRADING_RESERVE_SECONDS,
                "grading_reserve_tokens": GRADING_RESERVE_TOKENS,
            },
            "events": self.events,
        }
        write_json_atomic(self.run_dir / "stage-driver.json", driver)
        usage = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_commit": self.initial_commit,
            "wall_seconds": round(self.wall_elapsed, 2),
            "gpu_allocated_seconds": round(self.gpu_allocated, 2),
            "gpu_work_seconds": round(self.gpu_work, 2),
            "gpu_hours": round(self.gpu_allocated / 3600.0, 6),
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
            "gpu_index": self.gpu_index,
        }
        write_json_atomic(self.run_dir / "resource-usage.json", usage)
        final = all_gpu_snapshot()
        write_json_atomic(self.run_dir / "gpu-probe.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_commit": self.initial_commit,
            "observed_at": now_iso(),
            "gpu_index": self.gpu_index,
            "baseline": self.baseline,
            "final_device": next((d for d in final.get("devices", [])
                                  if d["index"] == self.gpu_index), None),
            "peak_new_usage_mib": round(self.peak_new, 1),
            "min_free_mib": self.min_free,
            "limits": {"max_new_mib": GPU_MAX_NEW_MIB,
                       "min_free_mib": GPU_MIN_FREE_MIB},
            "preflight_errors": self.preflight_errors,
            "phases": [{k: p[k] for k in ("name", "status", "phase_ok", "cleanup",
                                          "max_new_usage_mib", "min_free_mib")}
                       for p in self.phases],
        })

    def tokens_from(self, name: str) -> tuple[int, str, bool]:
        mapping = {
            "engineering": (self.run_dir / "engineering" / "engineering-check.json",
                            "generated_tokens"),
        }
        for i in range(4):
            mapping[f"chunk-{i:02d}-generate"] = (
                self.run_dir / "chunks" / f"chunk-{i:02d}" / "generation" / "run-status.json",
                "generated_tokens")
            mapping[f"chunk-{i:02d}-evaluate"] = (
                self.run_dir / "chunks" / f"chunk-{i:02d}" / "evaluation" / "run-status.json",
                "generated_tokens")
        path, key = mapping.get(name, (None, None))
        if path is not None:
            payload = load_json(path, {}) or {}
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value, "reported", False
        return 0, "missing_reported_zero", True

    def phase(self, name: str, cmd: list[str], wall_cap: float, result_dir: Path | None = None):
        if self.preflight_errors:
            return {"name": name, "status": "skipped_preflight", "phase_ok": False,
                    "exit_code": None, "tokens": 0, "cleanup": "not_started"}
        head_now = git_state()
        if head_now.get("head") != self.initial_commit:
            self.failures.append(f"{name}: code HEAD changed during run")
            return {"name": name, "status": "aborted_head_changed", "phase_ok": False,
                    "exit_code": None, "tokens": 0, "cleanup": "not_started"}
        wall_cap = min(wall_cap, self.args.wall_budget_seconds - self.wall_elapsed,
                       self.gpu_remaining)
        if wall_cap <= 30:
            self.failures.append(f"{name}: no bounded wall budget remains")
            return {"name": name, "status": "skipped_no_budget", "phase_ok": False,
                    "exit_code": None, "tokens": 0, "cleanup": "not_started"}
        log_path = self.log_dir / f"{name}.log"
        env = os.environ.copy()
        env["MUR_RUN_DIR"] = str(self.run_dir)
        env["MUR_RUNTIME_ROOT"] = str(Path.cwd() / "runtime")
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        env["TOKENIZERS_PARALLELISM"] = "false"
        started_epoch = time.time()
        started = time.monotonic()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        pgid = proc.pid
        samples = []
        last = 0.0
        status = "ok"
        telemetry_failed = False
        while proc.poll() is None:
            now = time.monotonic()
            if now - started > wall_cap:
                status = "killed_phase_cap"
                kill_group(pgid, signal.SIGTERM, "phase_cap", self.events)
                break
            if self.wall_elapsed > self.args.wall_budget_seconds:
                status = "killed_wall_budget"
                kill_group(pgid, signal.SIGTERM, "wall_budget", self.events)
                break
            if self.gpu_allocated + (now - started) > self.args.gpu_budget_seconds:
                status = "killed_gpu_budget"
                kill_group(pgid, signal.SIGTERM, "gpu_budget", self.events)
                break
            if now - last >= 2.0:
                last = now
                snap = all_gpu_snapshot()
                dev = next((d for d in snap.get("devices", [])
                            if d["index"] == self.gpu_index), None)
                apps, apps_ok, apps_err = compute_apps()
                if not snap.get("available") or dev is None or not apps_ok:
                    status = "killed_telemetry_unavailable"
                    telemetry_failed = True
                    self.events.append({"at": now_iso(), "action": "telemetry_unavailable",
                                        "gpu_error": snap.get("error"),
                                        "apps_error": apps_err})
                    kill_group(pgid, signal.SIGTERM, "telemetry_unavailable", self.events)
                    break
                pids = pids_in_group(pgid)
                our_mib = sum(v for p, v in apps.items() if p in pids)
                free = dev["free_mib"]
                new_usage = max(0.0, dev["used_mib"] - self.baseline["used_mib"])
                observed_new = max(new_usage, our_mib)
                self.peak_new = max(self.peak_new, observed_new)
                self.max_our_mib = max(self.max_our_mib, our_mib)
                self.min_free = free if self.min_free is None else min(self.min_free, free)
                samples.append({"t": round(now - started, 1), "used_mib": dev["used_mib"],
                                "free_mib": free, "our_apps_mib": our_mib,
                                "new_usage_mib": round(observed_new, 1),
                                "pids": sorted(pids)})
                if observed_new > GPU_MAX_NEW_MIB:
                    status = "killed_memory_ceiling"
                    kill_group(pgid, signal.SIGTERM, "memory_ceiling", self.events)
                    break
                if free < GPU_MIN_FREE_MIB:
                    status = "killed_free_memory_floor"
                    kill_group(pgid, signal.SIGTERM, "free_memory_floor", self.events)
                    break
            time.sleep(1.0)
        try:
            rc = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            kill_group(pgid, signal.SIGKILL, "wait_timeout", self.events)
            rc = proc.wait()
            status = "killed_sigkill"
        work_seconds = time.monotonic() - started
        leftover = pids_in_group(pgid)
        cleanup_start = time.monotonic()
        cleanup = "clean"
        if leftover:
            cleanup = "waited_clean"
            deadline = time.monotonic() + 20
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
        cleanup_seconds = time.monotonic() - cleanup_start
        duration = time.monotonic() - started
        self.gpu_allocated += duration
        self.gpu_work += work_seconds
        tokens, token_source, unknown = self.tokens_from(name)
        self.spent_tokens += tokens
        if unknown:
            self.unknown_reserved_tokens += tokens
        else:
            self.reported_tokens += tokens
        phase_ok = (status == "ok" and rc == 0 and not telemetry_failed
                    and cleanup in ("clean", "waited_clean"))
        result = {
            "name": name, "status": status, "exit_code": rc, "phase_ok": phase_ok,
            "command": cmd, "started_at": datetime.fromtimestamp(
                started_epoch, timezone.utc).isoformat(),
            "ended_at": now_iso(), "work_seconds": round(work_seconds, 2),
            "cleanup_seconds": round(cleanup_seconds, 2),
            "duration_seconds": round(duration, 2), "tokens": tokens,
            "token_source": token_source, "unknown_reservation": unknown,
            "gpu_allocated_after_seconds": round(self.gpu_allocated, 2),
            "max_new_usage_mib": round(max((s["new_usage_mib"] for s in samples),
                                           default=0.0), 1),
            "min_free_mib": min((s["free_mib"] for s in samples), default=None),
            "samples": samples[-120:], "cleanup": cleanup,
            "telemetry_failed": telemetry_failed, "log": f"logs/{name}.log",
        }
        if not phase_ok:
            self.failures.append(
                f"{name}: status={status} exit={rc} cleanup={cleanup} "
                f"telemetry_failed={telemetry_failed}")
        self.phases.append(result)
        if result_dir is not None:
            result_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(result_dir / f"phase-{name}.json", {
                "phase": name, "run_commit": self.initial_commit,
                "cmd": cmd, "status": status, "exit_code": rc,
                "phase_ok": phase_ok, "tokens": tokens,
                "duration_seconds": result["duration_seconds"],
                "cleanup": cleanup,
            })
        self.write_evidence()
        return result

    def run(self) -> int:
        self.write_evidence()
        if self.preflight_errors:
            return 2
        sources = load_json(Path("configs/sources.json"), {})
        manifest = self.manifest
        run_id = self.run_id
        task_file = str(Path(self.args.task_file))
        review = str(Path("experiments") / str(self.task["parent_run_id"]) / "review.md")
        py = sys.executable

        # Engineering identity check (bounded, no test generation).
        eng_dir = self.run_dir / "engineering"
        eng = self.phase("engineering", [
            py, "-m", "mur.engineering", "--output-dir", str(eng_dir),
            "--config", "configs/core.json", "--run-id", run_id,
            "--dev-indices", "0,3",
            "--max-generated-tokens", str(min(30000, self.token_remaining)),
            "--max-wall-seconds", str(max(60, min(280, self.gpu_remaining - 20))),
            "--max-model-len", "16384",
        ], min(300, self.gpu_remaining), eng_dir)
        eng_payload = load_json(eng_dir / "engineering-check.json", {}) or {}
        if not (eng.get("phase_ok") and eng_payload.get("passed") is True):
            self.failures.append("engineering identity check did not pass; test not launched")
            self.write_evidence()
            return 3
        if eng.get("duration_seconds", 0) > 300:
            self.failures.append("engineering check exceeded 300 GPU seconds")
            self.write_evidence()
            return 3
        write_json_atomic(self.run_dir / "engineering-check.json", eng_payload)
        write_json_atomic(Path("experiments") / run_id / "engineering-check.json",
                          eng_payload)

        chunks_ok = True
        for i in range(4):
            chunk_dir = self.run_dir / "chunks" / f"chunk-{i:02d}"
            gen_dir = chunk_dir / "generation"
            eval_dir = chunk_dir / "evaluation"
            qids = [q["id"] for q in manifest["questions"][25 * i:25 * i + 25]]
            (chunk_dir).mkdir(parents=True, exist_ok=True)
            (chunk_dir / "expected-ids.txt").write_text("\n".join(qids) + "\n",
                                                        encoding="utf-8")
            gen_cap = min(CHUNK_GENERATION_SECONDS,
                          self.gpu_remaining - GRADING_RESERVE_SECONDS,
                          self.args.wall_budget_seconds - self.wall_elapsed)
            gen_tokens = min(CHUNK_TOKEN_GRANT,
                             self.token_remaining - GRADING_RESERVE_TOKENS)
            if gen_cap <= 60 or gen_tokens < 10000:
                self.failures.append(f"chunk {i}: insufficient reserved generation budget")
                chunks_ok = False
                break
            gres = self.phase(f"chunk-{i:02d}-generate", [
                py, "-m", "mur.run", "--config", "configs/core.json",
                "--split", "test", "--test-manifest", str(self.run_dir / "test-manifest.json"),
                "--task-file", task_file, "--acceptance-review", review,
                "--methods", "cot", "per_step", "mur", "--run-id", run_id,
                "--output-dir", str(gen_dir), "--offset", str(25 * i),
                "--limit", "25", "--max-generated-tokens", str(gen_tokens),
                "--max-wall-seconds", str(max(60, gen_cap - 30)),
                "--max-model-len", "16384",
            ], gen_cap, chunk_dir)
            status = load_json(gen_dir / "run-status.json", {}) or {}
            config = load_json(gen_dir / "run-config.json", {}) or {}
            expected_qids = set(qids)
            config_ok = (set(config.get("question_ids") or []) == expected_qids
                         and config.get("split") == "test"
                         and config.get("prefix_caching") is False)
            status_ok = (status.get("complete") is True and status.get("exit_code") == 0
                         and status.get("errors") == 0
                         and status.get("trajectories") == 75
                         and status.get("method_counts") == {"cot": 25, "per_step": 25,
                                                            "mur": 25})
            gen_ok = bool(gres.get("phase_ok") and status_ok and config_ok)
            if not gen_ok:
                self.failures.append(f"chunk {i}: generation incomplete or invalid; "
                                     "no restart and no grading")
                chunks_ok = False
                break
            # The generation process and its engine children have exited before
            # the grader starts.  Full stop is asserted by the driver cleanup.
            eval_cap = min(600, self.gpu_remaining,
                           self.args.wall_budget_seconds - self.wall_elapsed)
            eval_tokens = self.token_remaining
            eres = self.phase(f"chunk-{i:02d}-evaluate", [
                py, "-m", "mur.evaluate", "--input-dir", str(gen_dir),
                "--output-dir", str(eval_dir), "--sources", "configs/sources.json",
                "--split", "test", "--test-manifest",
                str(self.run_dir / "test-manifest.json"), "--task-file", task_file,
                "--acceptance-review", review, "--run-id", run_id,
                "--expected-ids", str(chunk_dir / "expected-ids.txt"),
                "--max-generated-tokens", str(eval_tokens),
                "--max-wall-seconds", str(max(30, eval_cap - 20)),
                "--max-model-len", "4096",
            ], eval_cap, chunk_dir)
            estatus = load_json(eval_dir / "run-status.json", {}) or {}
            emetrics = load_json(eval_dir / "metrics.json", {}) or {}
            eval_ok = (eres.get("phase_ok") and estatus.get("exit_code") == 0
                       and estatus.get("partial") is False
                       and len(emetrics.get("rows") or []) == 75)
            if not eval_ok:
                self.failures.append(f"chunk {i}: grading incomplete; no restart")
                chunks_ok = False
                break
            write_json_atomic(chunk_dir / "chunk-complete.json", {
                "chunk": i, "questions": len(qids), "rows": len(emetrics["rows"]),
                "complete": True, "generation_tokens": status.get("generated_tokens"),
                "evaluation_tokens": estatus.get("generated_tokens"),
            })

        if not chunks_ok:
            self.write_evidence()
            return 4

        # Concatenate raw traces after all model processes are gone.
        raw_lines = []
        for i in range(4):
            p = self.run_dir / "chunks" / f"chunk-{i:02d}" / "generation" / "raw-smoke-traces.jsonl"
            raw_lines.extend([x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()])
        if len(raw_lines) != EXPECTED_ROWS:
            self.failures.append(f"raw trace rows {len(raw_lines)} != {EXPECTED_ROWS}")
            self.write_evidence()
            return 5
        tmp = self.run_dir / "raw-traces.jsonl.tmp"
        tmp.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        os.replace(tmp, self.run_dir / "raw-traces.jsonl")

        agg = subprocess.run([
            py, "-m", "mur.aggregate", "--run-dir", str(self.run_dir),
            "--task-file", task_file,
            "--test-manifest", str(self.run_dir / "test-manifest.json"),
            "--sources", "configs/sources.json",
        ], cwd=str(Path.cwd()), capture_output=True, text=True)
        (self.run_dir / "logs" / "aggregate.log").write_text(
            agg.stdout + "\n" + agg.stderr, encoding="utf-8")
        if agg.returncode != 0:
            self.failures.append(f"aggregate exited {agg.returncode}: {agg.stderr[-1000:]}")
            self.write_evidence()
            return 6

        # Copy the accepted source manifest and write environment evidence.
        src_manifest = Path("experiments/mur-20261024-prep-r6/source-manifest.json")
        if not src_manifest.exists():
            self.failures.append("missing accepted source manifest for the run")
            self.write_evidence()
            return 7
        shutil.copyfile(src_manifest, self.run_dir / "source-manifest.json")
        shutil.copyfile(src_manifest,
                        Path("experiments") / run_id / "source-manifest.json")
        self.write_environment(sources)
        self.write_evidence()
        shutil.copyfile(self.run_dir / "resource-usage.json",
                        Path("experiments") / run_id / "resource-usage.json")
        return 0

    def write_environment(self, sources):
        import platform
        from importlib.metadata import version as package_version

        def pkg(name):
            try:
                return package_version(name)
            except Exception:
                return None
        snap = all_gpu_snapshot()
        dev = next((d for d in snap.get("devices", [])
                    if d["index"] == self.gpu_index), None)
        metrics = load_json(self.run_dir / "metrics.json", {}) or {}
        env = {
            "schema_version": 1,
            "observed_at": now_iso(),
            "run_id": self.run_id,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {n: pkg(n) for n in
                         ("torch", "transformers", "vllm", "tokenizers", "pyarrow",
                          "setuptools")},
            "gpu": {"selected_index": self.gpu_index,
                    "cuda_visible_devices": str(self.gpu_index),
                    "preflight_device": self.gpu,
                    "final_device": dev,
                    "driver": "nvidia-smi"},
            "host_memory": self.host_memory(),
            "git_head": git_state().get("head"),
            "git_branch": git_state().get("branch"),
            "actual_run_commit": self.initial_commit,
            "prefix_caching": False,
            "test_manifest_sha256": testdata.sha256_file(
                self.run_dir / "test-manifest.json"),
            "source_sha256": sources.get("test", {}).get("sha256"),
            "models": {
                "policy": {"id": sources["policy"]["id"],
                           "revision": sources["policy"]["revision"]},
                "critic": {"id": sources["critic"]["id"],
                           "revision": sources["critic"]["revision"]},
                "verifier": {"id": sources["verifier"]["id"],
                             "revision": sources["verifier"]["revision"]},
            },
            "limits": {
                "gpu_seconds": self.args.gpu_budget_seconds,
                "generated_tokens": self.args.token_budget,
                "wall_seconds": self.args.wall_budget_seconds,
                "one_gpu": True,
                "max_new_mib": GPU_MAX_NEW_MIB,
                "min_free_mib": GPU_MIN_FREE_MIB,
            },
            "notes": [
                "No downloads and no shared-environment mutation.",
                "GPU state is an observed nvidia-smi snapshot.",
                "Latency on a shared GPU is descriptive.",
                "Prefix caching was disabled for policy, critic and verifier.",
            ],
        }
        write_json_atomic(self.run_dir / "environment.json", env)
        tracked = dict(env)
        tracked["notes"] = list(env["notes"])
        write_json_atomic(Path("experiments") / self.run_id / "environment.json", tracked)

    @staticmethod
    def host_memory():
        out = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    out[key] = value.strip()
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", default="mur-20261024-test100")
    ap.add_argument("--gpu-index", type=int, default=None)
    ap.add_argument("--gpu-budget-seconds", type=float, default=14400)
    ap.add_argument("--token-budget", type=int, default=2500000)
    ap.add_argument("--wall-budget-seconds", type=float, default=21600)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    d = TestDriver(args)
    if args.preflight_only:
        d.write_evidence()
        print(json.dumps({"run_id": d.run_id, "status": (
            "PREFLIGHT_OK" if not d.preflight_errors else "BLOCKED_PREFLIGHT"),
            "gpu_index": d.gpu_index, "gpu": d.gpu,
            "preflight_errors": d.preflight_errors,
            "disk_preflight": getattr(d, "disk_preflight", None)}, indent=2))
        return 0 if not d.preflight_errors else 2
    try:
        rc = d.run()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        d.failures.append(f"driver exception: {type(exc).__name__}: {exc}")
        d.write_evidence()
        rc = 7
    print(json.dumps({
        "run_id": d.run_id,
        "run_commit": d.initial_commit,
        "status": ("SUCCEEDED" if rc == 0 else
                   "BLOCKED_PREFLIGHT" if rc == 2 else "FAILED"),
        "gpu_index": d.gpu_index,
        "gpu_allocated_seconds": round(d.gpu_allocated, 2),
        "tokens": d.spent_tokens,
        "reported_tokens": d.reported_tokens,
        "wall_seconds": round(d.wall_elapsed, 2),
        "preflight_errors": d.preflight_errors,
        "failures": d.failures,
        "phases": [(p["name"], p["status"], p.get("exit_code"), p.get("cleanup"))
                   for p in d.phases],
    }, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
