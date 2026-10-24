"""Frozen 100-question MATH-500 test split and test-access gate.

The split is a deterministic shuffle of the 500 canonical source rows:
``random.Random(20261024).shuffle(range(500))`` and the first 100 indices in
shuffled order.  Row order of the source file is never changed.  The canonical
ID is ``test-<three-digit row index>-<first 12 hex of sha256(problem UTF-8)>``.

Only IDs, indices, source hashes and the stripped generation file live in the
runtime workspace.  Gold answers are joined back only during grading, after all
generations are frozen, and are never written into the generation split.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

SPLIT_SEED = 20261024
SPLIT_SIZE = 100
FORBIDDEN_GENERATION_FIELDS = {"solution", "answer", "gold", "gold_answer"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def canonical_test_id(index: int, problem: str) -> str:
    if index < 0:
        raise ValueError("negative test row index")
    if not isinstance(problem, str) or not problem:
        raise ValueError("empty test problem")
    return f"test-{index:03d}-{sha256_text(problem)[:12]}"


def load_source_rows(runtime_root: Path, sources: dict) -> tuple[list[dict], str]:
    spec = sources["test"]
    path = Path(runtime_root) / "data" / "MATH-500_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing pinned test file: {path}")
    got = sha256_file(path)
    if got != spec["sha256"]:
        raise ValueError(f"test sha256 mismatch: {got} != {spec['sha256']}")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    if len(rows) != int(spec["rows"]):
        raise ValueError(f"test rows {len(rows)} != {spec['rows']}")
    return rows, got


def row_hashes(rows: list[dict]) -> list[str]:
    return [sha256_json(r) for r in rows]


def frozen_index_order(total: int = 500, seed: int = SPLIT_SEED,
                       size: int = SPLIT_SIZE) -> list[int]:
    """Frozen selection: shuffle *all* indices, then take the first size."""
    indices = list(range(total))
    random.Random(int(seed)).shuffle(indices)
    if len(indices) < size:
        raise ValueError("source has fewer rows than the requested split size")
    return indices[:size]


def freeze_test_manifest(rows: list[dict], sources: dict,
                         total: int = 500, seed: int = SPLIT_SEED,
                         size: int = SPLIT_SIZE) -> dict:
    if len(rows) != total:
        raise ValueError(f"expected {total} source rows, got {len(rows)}")
    hashes = row_hashes(rows)
    order = frozen_index_order(total, seed, size)
    problems = []
    questions = []
    for position, index in enumerate(order):
        problem = rows[index].get("problem")
        if not isinstance(problem, str) or not problem:
            problems.append(f"row {index} has no problem text")
            continue
        questions.append({
            "order": position,
            "id": canonical_test_id(index, problem),
            "index": index,
            "question_sha256": sha256_text(problem),
            "row_sha256": hashes[index],
        })
    if len({q["id"] for q in questions}) != len(questions):
        problems.append("duplicate canonical test ID")
    if len({q["index"] for q in questions}) != len(questions):
        problems.append("duplicate source row index")
    if problems:
        raise ValueError("; ".join(problems))
    manifest = {
        "schema_version": 1,
        "stage": "first_independent_test100",
        "frozen_before_inference": True,
        "source": {
            "id": sources["test"]["id"],
            "revision": sources["test"]["revision"],
            "split": sources["test"]["split"],
            "rows": int(sources["test"]["rows"]),
            "file_sha256": sources["test"]["sha256"],
            "row_hashes_sha256": sha256_text("\n".join(hashes)),
        },
        "selection": {
            "seed": int(seed),
            "method": ("list(range(500)); random.Random(seed).shuffle(indices); "
                       "take first 100 in shuffled order"),
            "size": size,
            "outcome_based": False,
            "gold_used": False,
        },
        "rows": len(questions),
        "questions": questions,
    }
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                      for r in generation_rows(rows, manifest))
    manifest["generation_split"] = {
        "format": "jsonl; only id,source,revision,split,index,problem,question_sha256",
        "rows": len(questions),
        "sha256": sha256_text(payload),
    }
    return manifest


def verify_test_manifest(manifest: dict, rows: list[dict], sources: dict,
                         total: int = 500, seed: int = SPLIT_SEED,
                         size: int = SPLIT_SIZE) -> list[str]:
    """Independently recompute the frozen split and compare every identity."""
    problems = []
    expected = freeze_test_manifest(rows, sources, total, seed, size)
    for key in ("schema_version", "stage", "frozen_before_inference"):
        if manifest.get(key) != expected.get(key):
            problems.append(f"manifest field {key} mismatch")
    for key in ("source", "selection", "rows", "generation_split"):
        if manifest.get(key) != expected.get(key):
            problems.append(f"manifest field {key} mismatch")
    got_q = manifest.get("questions")
    exp_q = expected["questions"]
    if not isinstance(got_q, list) or len(got_q) != size:
        problems.append(f"manifest questions length != {size}")
    elif got_q != exp_q:
        for i, (a, b) in enumerate(zip(got_q, exp_q)):
            if a != b:
                problems.append(f"manifest question {i} differs from frozen selection")
                break
    return problems


def generation_rows(rows: list[dict], manifest: dict) -> list[dict]:
    """Strip gold/solution fields before any test generation."""
    out = []
    for q in manifest["questions"]:
        index = int(q["index"])
        problem = rows[index].get("problem")
        if sha256_text(problem) != q["question_sha256"]:
            raise ValueError(f"question hash mismatch at row {index}")
        if canonical_test_id(index, problem) != q["id"]:
            raise ValueError(f"canonical ID mismatch at row {index}")
        out.append({
            "id": q["id"],
            "source": manifest["source"]["id"],
            "revision": manifest["source"]["revision"],
            "split": "test",
            "index": index,
            "problem": problem,
            "question_sha256": q["question_sha256"],
        })
    for row in out:
        forbidden = FORBIDDEN_GENERATION_FIELDS.intersection(row)
        if forbidden:
            raise ValueError(f"generation row contains gold fields: {sorted(forbidden)}")
    return out


def build_test_generation_file(rows: list[dict], manifest: dict,
                               runtime_root: Path) -> dict:
    path = Path(runtime_root) / "data" / "test100.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    qrows = generation_rows(rows, manifest)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in qrows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return {
        "path": path,
        "rows": len(qrows),
        "sha256": sha256_file(path),
        "ids_sha256": sha256_text("\n".join(r["id"] for r in qrows)),
    }


def load_test_generation_file(runtime_root: Path, manifest: dict) -> list[dict]:
    path = Path(runtime_root) / "data" / "test100.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen test split: {path}")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    if len(rows) != int(manifest["rows"]):
        raise ValueError(f"test split rows {len(rows)} != manifest {manifest['rows']}")
    for position, (row, want) in enumerate(zip(rows, manifest["questions"])):
        if row.get("id") != want["id"]:
            raise ValueError(f"test split ID mismatch at position {position}")
        if int(row.get("index", -1)) != int(want["index"]):
            raise ValueError(f"test split index mismatch at position {position}")
        if row.get("split") != "test":
            raise ValueError(f"test split marker missing at position {position}")
        if row.get("question_sha256") != want["question_sha256"]:
            raise ValueError(f"question hash mismatch at position {position}")
        if sha256_text(row.get("problem", "")) != want["question_sha256"]:
            raise ValueError(f"problem text mismatch at position {position}")
        forbidden = FORBIDDEN_GENERATION_FIELDS.intersection(row)
        if forbidden:
            raise ValueError(f"generation split contains gold fields: {sorted(forbidden)}")
    return rows


def authorize_test(task_file: Path, review_file: Path, run_id: str) -> dict:
    """Fail closed unless a ready task and an accepted parent review authorize test."""
    task = json.loads(Path(task_file).read_text(encoding="utf-8"))
    if task.get("run_id") != run_id:
        raise PermissionError(f"task run_id {task.get('run_id')} != {run_id}")
    if task.get("ready") is not True:
        raise PermissionError("task is not ready")
    controls = " ".join(task.get("controls") or [])
    if "fixed100" not in controls or "ready task" not in controls:
        raise PermissionError("task does not contain the frozen fixed100 test controls")
    vc = task.get("version_control") or {}
    if not vc.get("repository") or vc.get("visibility") != "private":
        raise PermissionError("task lacks a private version_control repository")
    review_path = Path(review_file)
    if not review_path.exists():
        raise PermissionError(f"missing accepted parent review: {review_path}")
    text = review_path.read_text(encoding="utf-8")
    head = text[:4000]
    if not re.search(r"Decision:\s*\*\*accept", head, flags=re.I):
        raise PermissionError("parent review does not accept the preparation gate")
    if re.search(r"Decision:\s*\*\*reject", head, flags=re.I):
        raise PermissionError("parent review rejects the preparation gate")
    if "Authorize the separate frozen" not in head:
        raise PermissionError("parent review does not authorize the frozen test task")
    return {
        "task_file": str(task_file),
        "task_run_id": task["run_id"],
        "ready": True,
        "review_file": str(review_path),
        "review_accepted": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
