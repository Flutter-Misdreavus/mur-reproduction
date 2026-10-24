import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mur import run as runner
from mur import testdata
from mur.aggregate import decision_from, paired_bootstrap, verifier_cost
from mur.engineering import compare_pair
from mur.evaluate import trigger_stats


def synthetic_rows(n=500):
    return [{"problem": f"question {i}", "solution": f"gold {i}",
             "answer": f"answer {i}"} for i in range(n)]


def step(i, text, triggered=False):
    return {"step": i, "proposal": text, "proposal_token_ids": [i, i + 1],
            "accepted_text": text, "accepted_token_ids": [i, i + 1],
            "accepted_tokens": 2, "triggered": triggered}


class FrozenSplitContracts(unittest.TestCase):
    def test_selection_is_deterministic_and_shuffled(self):
        rows = synthetic_rows()
        a = testdata.freeze_test_manifest(rows, {"test": {
            "id": "HuggingFaceH4/MATH-500", "revision": "r", "split": "test",
            "rows": 500, "sha256": "0" * 64}})
        b = testdata.freeze_test_manifest(rows, {"test": {
            "id": "HuggingFaceH4/MATH-500", "revision": "r", "split": "test",
            "rows": 500, "sha256": "0" * 64}})
        self.assertEqual(a, b)
        self.assertEqual(len(a["questions"]), 100)
        self.assertEqual(len({q["index"] for q in a["questions"]}), 100)
        self.assertNotEqual([q["index"] for q in a["questions"]], list(range(100)))
        for q in a["questions"][:5]:
            self.assertEqual(q["id"], testdata.canonical_test_id(
                q["index"], rows[q["index"]]["problem"]))

    def test_manifest_tamper_and_gold_strip(self):
        rows = synthetic_rows()
        sources = {"test": {"id": "HuggingFaceH4/MATH-500", "revision": "r",
                            "split": "test", "rows": 500, "sha256": "0" * 64}}
        man = testdata.freeze_test_manifest(rows, sources)
        self.assertEqual(testdata.verify_test_manifest(man, rows, sources), [])
        bad = json.loads(json.dumps(man))
        bad["questions"][3]["id"] = "test-000-deadbeef0000"
        self.assertTrue(testdata.verify_test_manifest(bad, rows, sources))
        gen = testdata.generation_rows(rows, man)
        for row in gen:
            self.assertEqual(set(row), {"id", "source", "revision", "split",
                                        "index", "problem", "question_sha256"})

    def test_build_and_load_stripped_split(self):
        rows = synthetic_rows()
        sources = {"test": {"id": "HuggingFaceH4/MATH-500", "revision": "r",
                            "split": "test", "rows": 500, "sha256": "0" * 64}}
        man = testdata.freeze_test_manifest(rows, sources)
        with tempfile.TemporaryDirectory() as td:
            info = testdata.build_test_generation_file(rows, man, Path(td))
            loaded = testdata.load_test_generation_file(Path(td), man)
            self.assertEqual(len(loaded), 100)
            self.assertEqual(info["sha256"], man["generation_split"]["sha256"])
            self.assertFalse(any("solution" in r or "answer" in r for r in loaded))

    def test_load_questions_reads_runtime_root_once(self):
        rows = synthetic_rows()
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                          for r in rows)
        sources = {"test": {"id": "HuggingFaceH4/MATH-500", "revision": "r",
                            "split": "test", "rows": 500,
                            "sha256": hashlib.sha256(payload.encode()).hexdigest()}}
        man = testdata.freeze_test_manifest(rows, sources)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "data" / "MATH-500_test.jsonl").write_text(payload, encoding="utf-8")
            testdata.build_test_generation_file(rows, man, root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(man), encoding="utf-8")

            def fake_load(path):
                if str(path) == "configs/sources.json":
                    return sources
                with open(path, encoding="utf-8") as f:
                    return json.load(f)

            with mock.patch.dict(os.environ, {"MUR_RUNTIME_ROOT": str(root)}):
                with mock.patch("mur.run.load_json", side_effect=fake_load):
                    loaded = runner.load_questions("test", None, str(manifest_path))
            self.assertEqual(len(loaded), 100)
            self.assertEqual([r["id"] for r in loaded],
                             [q["id"] for q in man["questions"]])


class TestAuthorizationContracts(unittest.TestCase):
    def test_accept_and_reject(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            task = {"run_id": "run", "ready": True,
                    "controls": ["fixed100 test", "ready task gate"],
                    "version_control": {"repository": "Owner/Repo",
                                        "visibility": "private"}}
            (td / "task.json").write_text(json.dumps(task), encoding="utf-8")
            review = ("# Review\n\nDecision: **accept the engineering preparation gate**\n"
                      "Authorize the separate frozen `run` task.\n")
            (td / "review.md").write_text(review, encoding="utf-8")
            out = testdata.authorize_test(td / "task.json", td / "review.md", "run")
            self.assertTrue(out["review_accepted"])
            (td / "review.md").write_text("Decision: **reject**\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                testdata.authorize_test(td / "task.json", td / "review.md", "run")
            task["ready"] = False
            (td / "task.json").write_text(json.dumps(task), encoding="utf-8")
            with self.assertRaises(PermissionError):
                testdata.authorize_test(td / "task.json", td / "review.md", "run")

    def test_r2_frozen_wording_requires_reused_manifest_and_exact_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            task = {"run_id": "run", "ready": True,
                    "controls": ["Reuse the identical committed100-question manifest, "
                                 "no reselection.",
                                 "Run exactly100 questions x3 methods in four25-question "
                                 "chunks with policy/critic fully stopped.",
                                 "Test access gated by this ready task and accepted review."],
                    "version_control": {"repository": "Owner/Repo",
                                        "visibility": "private"}}
            (td / "task.json").write_text(json.dumps(task), encoding="utf-8")
            (td / "review.md").write_text(
                "# Review\n\nDecision: **accept the preparation gate**\n"
                "Authorize the separate frozen `run` task.\n", encoding="utf-8")
            self.assertTrue(testdata.authorize_test(
                td / "task.json", td / "review.md", "run")["review_accepted"])
            task["controls"] = task["controls"][1:]
            (td / "task.json").write_text(json.dumps(task), encoding="utf-8")
            with self.assertRaises(PermissionError):
                testdata.authorize_test(td / "task.json", td / "review.md", "run")


class TriggerDenominatorRegression(unittest.TestCase):
    def test_single_step_mur_has_zero_eligible_but_all_step_rate_reported(self):
        rows = [{"steps": 1, "triggers": 0}]
        mur = trigger_stats("mur", rows)
        self.assertEqual(mur["denominator"], 0)
        self.assertIsNone(mur["rate"])
        self.assertEqual(mur["all_step_denominator"], 1)
        self.assertEqual(mur["all_step_rate"], 0.0)
        per = trigger_stats("per_step", rows)
        self.assertEqual(per["denominator"], 1)
        self.assertEqual(per["rate"], 0.0)

    def test_mur_denominator_excludes_first_step_only(self):
        rows = [{"steps": 3, "triggers": 2}, {"steps": 1, "triggers": 0}]
        mur = trigger_stats("mur", rows)
        self.assertEqual((mur["numerator"], mur["denominator"]), (2, 2))
        self.assertAlmostEqual(mur["rate"], 1.0)
        self.assertAlmostEqual(mur["all_step_rate"], 0.5)


class EngineeringIdentityContracts(unittest.TestCase):
    def pair(self, mur_steps):
        cot = {"qid": "q", "steps": [step(0, "a"), step(1, "b")]}
        mur = {"qid": "q", "steps": mur_steps}
        return cot, mur

    def test_identical_prefix_passes_and_trigger_cuts_comparison(self):
        cot, mur = self.pair([step(0, "a"), step(1, "c", True)])
        out = compare_pair(cot, mur)
        self.assertTrue(out["passed"])
        self.assertEqual(out["first_trigger_step"], 1)
        cot, mur = self.pair([step(0, "a"), step(1, "b")])
        self.assertTrue(compare_pair(cot, mur)["passed"])
        cot, mur = self.pair([step(0, "a"), step(1, "DIFFERENT")])
        self.assertFalse(compare_pair(cot, mur)["passed"])

    def test_first_proposal_or_token_mismatch_fails(self):
        cot, mur = self.pair([step(0, "a"), step(1, "b")])
        mur["steps"][0]["proposal_token_ids"] = [9]
        out = compare_pair(cot, mur)
        self.assertFalse(out["passed"])
        self.assertIn("first proposal token IDs differ", out["problems"])


class BootstrapContracts(unittest.TestCase):
    def test_paired_bootstrap_and_decision(self):
        per_q = {}
        for i in range(20):
            per_q[f"q{i}"] = {
                "cot": {"tokens": 10, "correct": 1},
                "per_step": {"tokens": 100, "correct": 1},
                "mur": {"tokens": 50, "correct": 1},
            }
        out = paired_bootstrap(per_q, seed=20261024, replicates=200)
        self.assertAlmostEqual(out["point"]["token_savings_vs_per_step"], 0.5)
        self.assertAlmostEqual(out["point"]["accuracy_delta_vs_per_step"], 0.0)
        self.assertEqual(out["token_savings_ci95"], [0.5, 0.5])
        self.assertEqual(out["accuracy_delta_ci95"], [0.0, 0.0])
        dec = decision_from(out)
        self.assertTrue(dec["point_target_met"])
        self.assertTrue(dec["evidence_supported_non_inferiority"])


class VerifierCostContracts(unittest.TestCase):
    def test_actual_and_attributed_verifier_tokens_are_separate(self):
        chunk_status = [
            {"evaluation": {"generated_tokens": 10, "verified_now": 3,
                            "unique_pending_prompts": 4, "wall_seconds": 1.0}},
            {"evaluation": {"generated_tokens": 5, "verified_now": 2,
                            "unique_pending_prompts": 3, "wall_seconds": 1.5}},
        ]
        summary = {m: {"verifier_tokens_attributed_total": t} for m, t in
                   (("cot", 2), ("per_step", 4), ("mur", 6))}
        out = verifier_cost(chunk_status, summary)
        self.assertEqual(out["output_tokens_total"], 15)
        self.assertEqual(out["attributed_output_tokens_total"], 12)
        self.assertEqual(out["calls"], 5)
        self.assertEqual(out["unique_pending_prompts"], 7)
        self.assertEqual(out["wall_seconds"], 2.5)
        self.assertEqual(out["per_method_output_tokens"],
                         {"cot": 2, "per_step": 4, "mur": 6})


if __name__ == "__main__":
    unittest.main()
