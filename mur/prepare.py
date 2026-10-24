"""Freeze public source manifests and select the 20-question development split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from .prompts import boxed_values, question_key


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_dir(root: Path, repo: str) -> Path:
    return root / "models" / repo.split("/")[-1]


def verify_model(root: Path, spec: dict, problems: list[str]) -> dict:
    d = model_dir(root, spec["id"])
    files = {}
    for name, expected in spec.get("weights_sha256", {}).items():
        p = d / name
        if not p.exists():
            problems.append(f"missing model file {spec['id']}/{name}")
            continue
        got = sha256_file(p)
        files[name] = {"bytes": p.stat().st_size, "sha256": got}
        if got != expected:
            problems.append(f"sha256 mismatch {spec['id']}/{name}: {got} != {expected}")
    return {"id": spec["id"], "revision": spec["revision"], "files": files}


def extract_gold(solution: str) -> str:
    vals = boxed_values(solution)
    return vals[-1] if vals else ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="configs/sources.json")
    ap.add_argument("--runtime-root", default=os.environ.get("MUR_RUNTIME_ROOT", "runtime"))
    ap.add_argument("--run-dir", default=os.environ.get("MUR_RUN_DIR", "runs/mur-20261024-prep"))
    ap.add_argument("--num-dev", type=int, default=20)
    args = ap.parse_args(argv)
    root = Path(args.runtime_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    sources = json.load(open(args.sources, encoding="utf-8"))
    problems: list[str] = []

    test_path = data_dir / "MATH-500_test.jsonl"
    if not test_path.exists():
        raise SystemExit(f"missing test file: {test_path}")
    test_sha = sha256_file(test_path)
    if test_sha != sources["test"]["sha256"]:
        problems.append(f"test sha256 mismatch: {test_sha}")
    test_rows = [json.loads(x) for x in test_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(test_rows) != sources["test"]["rows"]:
        problems.append(f"test rows {len(test_rows)} != {sources['test']['rows']}")
    test_row_hashes = [sha256_json(r) for r in test_rows]
    test_keys = {question_key(r.get("problem", "")) for r in test_rows}

    candidates = []
    dev_files = {}
    for subject in sources["dev"]["subjects"]:
        p = data_dir / "hendrycks_math" / f"{subject}-train.parquet"
        if not p.exists():
            problems.append(f"missing dev shard {subject}")
            continue
        got = sha256_file(p)
        dev_files[p.name] = {"bytes": p.stat().st_size, "sha256": got}
        if got != sources["dev"]["files_sha256"][p.name]:
            problems.append(f"dev shard sha256 mismatch {p.name}")
        import pyarrow.parquet as pq
        rows = pq.read_table(p).to_pylist()
        for i, r in enumerate(rows):
            prob = r.get("problem", "")
            if question_key(prob) in test_keys:
                continue
            candidates.append({
                "subject": subject, "row_index": i, "problem": prob,
                "solution": r.get("solution", ""), "answer": extract_gold(r.get("solution", "")),
                "type": r.get("type"), "level": r.get("level"),
            })
    candidates.sort(key=lambda r: (r["subject"], r["row_index"]))
    rng = random.Random(int(sources["dev"]["selection_seed"]))
    chosen = rng.sample(candidates, min(args.num_dev, len(candidates)))
    if len(chosen) != args.num_dev:
        problems.append(f"only {len(chosen)} dev candidates available")
    dev_rows = []
    for r in chosen:
        qsha = hashlib.sha256(r["problem"].encode("utf-8")).hexdigest()
        dev_rows.append({
            "id": f"dev-{qsha[:12]}", "source": sources["dev"]["id"],
            "revision": sources["dev"]["revision"], "split": "train",
            "subject": r["subject"], "row_index": r["row_index"],
            "problem": r["problem"], "solution": r["solution"],
            "answer": r["answer"], "question_sha256": qsha,
        })
    with open(data_dir / "dev.jsonl", "w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dev_manifest = {
        "source": sources["dev"]["id"], "revision": sources["dev"]["revision"],
        "split": "train", "selection_seed": sources["dev"]["selection_seed"],
        "selection_method": "stable sort (subject,row_index), random.Random(seed).sample",
        "normalized_test_dedup": True, "rows": len(dev_rows),
        "questions": [{"id": r["id"], "subject": r["subject"], "row_index": r["row_index"],
                       "question_sha256": r["question_sha256"]} for r in dev_rows],
    }

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper": sources["paper"],
        "upstream": sources["upstream"],
        "policy": verify_model(root, sources["policy"], problems),
        "critic": verify_model(root, sources["critic"], problems),
        "verifier": verify_model(root, sources["verifier"], problems),
        "test": {"id": sources["test"]["id"], "revision": sources["test"]["revision"],
                 "split": "test", "rows": len(test_rows), "file_sha256": test_sha,
                 "row_hashes_sha256": hashlib.sha256("\n".join(test_row_hashes).encode()).hexdigest()},
        "dev": {"source": sources["dev"]["id"], "revision": sources["dev"]["revision"],
                "split": "train", "rows": len(dev_rows), "selection_seed": sources["dev"]["selection_seed"],
                "files": dev_files, "manifest": dev_manifest},
    }
    with open(run_dir / "source-manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(run_dir / "dev-manifest.json", "w", encoding="utf-8") as f:
        json.dump(dev_manifest, f, indent=2)
    if problems:
        with open(run_dir / "prepare-problems.json", "w", encoding="utf-8") as f:
            json.dump(problems, f, indent=2)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(json.dumps({"test_rows": len(test_rows), "dev_rows": len(dev_rows),
                      "manifest": str(run_dir / "source-manifest.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
