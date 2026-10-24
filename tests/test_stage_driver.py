import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "stage_driver", Path(__file__).resolve().parents[1] / "scripts" / "stage_driver.py")
stage_driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_driver)


class TokenAccountingContracts(unittest.TestCase):
    def test_reported_tokens_are_used_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            (rd / "smoke").mkdir()
            (rd / "smoke" / "run-status.json").write_text(
                json.dumps({"generated_tokens": 123}), encoding="utf-8")
            value, source, unknown = stage_driver.tokens_from_status("smoke", rd, 999)
            self.assertEqual(value, 123)
            self.assertEqual(source, "reported")
            self.assertFalse(unknown)

    def test_missing_count_is_a_separate_conservative_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            value, source, unknown = stage_driver.tokens_from_status("evaluate", Path(td), 777)
            self.assertEqual(value, 777)
            self.assertEqual(source, "unknown_conservative_reservation")
            self.assertTrue(unknown)

    def test_invalid_reported_count_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            (rd / "evaluation").mkdir()
            (rd / "evaluation" / "run-status.json").write_text(
                json.dumps({"generated_tokens": -1}), encoding="utf-8")
            value, source, unknown = stage_driver.tokens_from_status("evaluate", rd, 42)
            self.assertEqual((value, source, unknown),
                             (42, "unknown_conservative_reservation", True))


class FreshnessAndOutputContracts(unittest.TestCase):
    def test_fresh_file_rejects_stale_pass(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "logprob-validation.json"
            p.write_text("{}", encoding="utf-8")
            self.assertTrue(stage_driver.fresh_file(p, 0))
            self.assertFalse(stage_driver.fresh_file(p, 10**12))

    def test_reused_gpu_output_detected(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            self.assertEqual(stage_driver.find_existing_outputs(rd), [])
            (rd / "smoke").mkdir()
            (rd / "smoke" / "run-status.json").write_text("{}", encoding="utf-8")
            self.assertIn("smoke/run-status.json",
                          stage_driver.find_existing_outputs(rd))

    def test_validation_requires_new_clean_pass(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            payload = {
                "passed": True,
                "validation_reference": "same_forward_unmodified_logits",
                "cross_engine_hf_parity": False,
                "probe_count": 8,
                "complete_nonempty_probes": True,
            }
            (rd / "logprob-validation.json").write_text(json.dumps(payload), encoding="utf-8")
            good = {"phase_ok": True, "started_epoch": 0}
            bad = {"phase_ok": False, "started_epoch": 0}
            self.assertTrue(stage_driver.validation_passed(rd, good))
            self.assertFalse(stage_driver.validation_passed(rd, bad))
            payload["cross_engine_hf_parity"] = None
            (rd / "logprob-validation.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(stage_driver.validation_passed(rd, good))

    def test_smoke_requires_complete_clean_status_and_traces(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            (rd / "smoke").mkdir()
            (rd / "smoke" / "run-status.json").write_text(json.dumps(
                {"complete": True, "exit_code": 0, "errors": 0}), encoding="utf-8")
            (rd / "smoke" / "raw-smoke-traces.jsonl").write_text("{}\n", encoding="utf-8")
            result = {"phase_ok": True, "started_epoch": 0}
            self.assertTrue(stage_driver.smoke_complete(rd, result))
            self.assertFalse(stage_driver.smoke_complete(rd, {"phase_ok": False,
                                                              "started_epoch": 0}))
            (rd / "smoke" / "run-status.json").write_text(json.dumps(
                {"complete": True, "exit_code": 0, "errors": 1}), encoding="utf-8")
            self.assertFalse(stage_driver.smoke_complete(rd, result))


if __name__ == "__main__":
    unittest.main()
