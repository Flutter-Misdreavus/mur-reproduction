"""Recompute paired summaries from sanitized supervisor rows (Python stdlib only)."""
import argparse
import json
import math
import random
from pathlib import Path

METHODS = ("cot", "per_step", "mur")


def interval(values):
    values = sorted(values)
    def percentile(p):
        pos = p * (len(values) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return [percentile(.025), percentile(.975)]


def summarize(rows, label, *, repeats=10000, seed=20261024):
    groups = {}
    for row in rows:
        assert row[label] in (0, 1)
        assert row["method"] not in groups.setdefault(row["qid"], {})
        groups[row["qid"]][row["method"]] = row
    assert groups and all(set(g) == set(METHODS) for g in groups.values())
    qids = sorted(groups)
    n = len(qids)
    summary = {}
    for method in METHODS:
        selected = [g[method] for g in groups.values()]
        tokens = sum(r["total_output_tokens"] for r in selected)
        correct = sum(r[label] for r in selected)
        summary[method] = {
            "n": n, "correct": correct, "accuracy": correct / n,
            "total_output_tokens": tokens, "mean_output_tokens": tokens / n,
            "wall_seconds": sum(r["wall_seconds"] for r in selected),
        }
    rng = random.Random(seed)
    savings, differences = [], []
    for _ in range(repeats):
        chosen = [qids[rng.randrange(n)] for _ in range(n)]
        savings.append(1 - sum(groups[q]["mur"]["total_output_tokens"] for q in chosen) /
                       sum(groups[q]["per_step"]["total_output_tokens"] for q in chosen))
        differences.append(sum(groups[q]["mur"][label] - groups[q]["per_step"][label]
                               for q in chosen) / n)
    saving = 1 - summary["mur"]["total_output_tokens"] / summary["per_step"]["total_output_tokens"]
    delta = (summary["mur"]["correct"] - summary["per_step"]["correct"]) / n
    delta_ci = interval(differences)
    return {
        "questions": n, "methods": summary,
        "token_saving_vs_per_step": saving, "token_saving_ci95": interval(savings),
        "accuracy_delta_vs_per_step": delta, "accuracy_delta_ci95": delta_ci,
        "mur_only_correct": sum(g["mur"][label] > g["per_step"][label] for g in groups.values()),
        "per_step_only_correct": sum(g["mur"][label] < g["per_step"][label] for g in groups.values()),
        "point_target_met": saving >= .4 and delta >= -.02 - 1e-12,
        "non_inferiority_supported": saving >= .4 and delta_ci[0] > -.02 + 1e-12,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.rows.read_text())
    rows = source["paired_rows"]
    primary = [r for r in rows if r["automated_correct"] is not None]
    result = {
        "run_id": source["run_id"], "task_complete": False,
        "primary_automated_scored_subset": summarize(primary, "automated_correct"),
        "secondary_reviewed_same_scored_subset": summarize(primary, "audited_correct"),
        "supplementary_reviewed_all_complete_pairs": summarize(rows, "audited_correct"),
        "bootstrap": {"replicates": 10000, "seed": 20261024, "unit": "question",
                      "sampling": "random.randrange on sorted IDs, with replacement",
                      "interval": "two-sided 95% percentile, linear interpolation"},
        "scope": "Resource-truncated exploratory subsets; neither estimates completion of the fixed100 task.",
        "reviewer": "GPT supervisor mathematical-equivalence review, not human annotation",
        "changed_automated_labels": [r for r in primary if r["automated_correct"] != r["audited_correct"]],
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
