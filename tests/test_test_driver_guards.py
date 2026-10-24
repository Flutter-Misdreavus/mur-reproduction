"""Regression cases found during supervisor review; no GPU work."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "test_driver_guards", Path(__file__).resolve().parents[1] / "scripts/test_stage_driver.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class DriverBudgetRegression(unittest.TestCase):
    def test_grading_and_generation_share_one_chunk_budget(self):
        generation, grading = driver.chunk_caps(14400, 21600)
        self.assertEqual(generation + grading, 3300)
        _, remaining_grading = driver.chunk_caps(12000, 19000, spent=3200)
        self.assertEqual(remaining_grading, 100)
        self.assertEqual(driver.chunk_caps(20, 19000), (0, 20))

    def test_missing_or_invalid_counter_reserves_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            d = object.__new__(driver.TestDriver)
            d.run_dir = Path(directory)
            self.assertEqual(d.tokens_from("engineering", 30000),
                             (30000, "unknown_conservative_reservation", True))
            p = d.run_dir / "engineering/engineering-check.json"
            p.parent.mkdir()
            for value in [None, -1, True, "12"]:
                p.write_text(json.dumps({"generated_tokens": value}))
                self.assertTrue(d.tokens_from("engineering", 30000)[2])
            p.write_text(json.dumps({"generated_tokens": 712}))
            self.assertEqual(d.tokens_from("engineering", 30000), (712, "reported", False))

    def test_exact_two_point_loss_is_on_point_boundary(self):
        from mur.aggregate import paired_bootstrap, decision_from
        q = {str(i): {"cot": {"tokens": 1, "correct": int(i < 81)},
                      "per_step": {"tokens": 10, "correct": int(i < 81)},
                      "mur": {"tokens": 2, "correct": int(i < 79)}} for i in range(100)}
        stats = paired_bootstrap(q, replicates=10)
        self.assertEqual(stats["point"]["accuracy_delta_vs_per_step"], -0.02)
        self.assertTrue(decision_from(stats)["point_target_met"])
        stats["accuracy_delta_ci95"][0] = -0.02
        self.assertFalse(decision_from(stats)["evidence_supported_non_inferiority"])


if __name__ == "__main__":
    unittest.main()
