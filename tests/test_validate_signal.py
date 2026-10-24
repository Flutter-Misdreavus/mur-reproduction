import json
import math
import tempfile
import unittest
from pathlib import Path

from mur.validate_signal import (compare_probe, sampler_source_evidence,
                                 validate_config)


TOL = {"max_abs_token_nll": 0.15, "mean_abs_token_nll": 0.02, "mean_nll": 0.005}


class SignalComparisonContracts(unittest.TestCase):
    def test_all_three_tolerances_pass(self):
        out = compare_probe([0.10, 0.20], [0.11, 0.19], TOL, output_tokens=2)
        self.assertTrue(out["passed"])
        self.assertAlmostEqual(out["max_abs_token_logprob_diff"], 0.01)
        self.assertAlmostEqual(out["mean_abs_token_logprob_diff"], 0.01)
        self.assertAlmostEqual(out["abs_mean_diff"], 0.0)
        self.assertEqual(out["scored_tokens"], 2)

    def test_max_abs_fails(self):
        out = compare_probe([0.0], [0.151], TOL, output_tokens=1)
        self.assertFalse(out["passed"])
        self.assertGreaterEqual(out["max_abs_token_logprob_diff"], 0.15)

    def test_mean_abs_fails(self):
        out = compare_probe([0.0, 0.0], [0.021, -0.021], TOL, output_tokens=2)
        self.assertFalse(out["passed"])
        self.assertGreaterEqual(out["mean_abs_token_logprob_diff"], 0.02)

    def test_abs_mean_is_strict(self):
        out = compare_probe([0.0, 0.0], [0.004, 0.006], TOL, output_tokens=2)
        self.assertFalse(out["passed"])
        self.assertAlmostEqual(out["abs_mean_diff"], 0.005)

    def test_empty_mismatch_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            compare_probe([], [], TOL)
        with self.assertRaises(ValueError):
            compare_probe([0.1], [0.1, 0.2], TOL)
        with self.assertRaises(ValueError):
            compare_probe([0.1], [0.1], TOL, output_tokens=2)
        with self.assertRaises(ValueError):
            compare_probe([float("nan")], [0.1], TOL)

    def test_config_requires_eight_unique_probes(self):
        with open("configs/validation.json", encoding="utf-8") as f:
            cfg = json.load(f)
        validate_config(cfg)
        bad = dict(cfg)
        bad["probes"] = cfg["probes"][:7]
        with self.assertRaises(ValueError):
            validate_config(bad)
        dup = dict(cfg)
        dup["probes"] = cfg["probes"][:7] + [cfg["probes"][0]]
        with self.assertRaises(ValueError):
            validate_config(dup)


class SamplerEvidenceContracts(unittest.TestCase):
    def test_installed_sampler_source_proves_ordering(self):
        evidence = sampler_source_evidence()
        self.assertTrue(evidence["available"])
        self.assertEqual(len(evidence["sha256"]), 64)
        self.assertTrue(evidence["raw_before_penalties_and_temperature"])
        self.assertLess(evidence["raw_logprobs_line"], evidence["apply_penalties_line"])
        self.assertLess(evidence["raw_logprobs_line"], evidence["apply_temperature_line"])


if __name__ == "__main__":
    unittest.main()
