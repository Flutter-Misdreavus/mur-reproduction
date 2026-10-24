import unittest
from types import SimpleNamespace

from mur.backend import Generation
from mur.validate_forward import compare_returned


class ForwardObserverContracts(unittest.TestCase):
    def fixture(self, client_lp=-0.3):
        gen = Generation(token_ids=[9], logprobs=[{9: SimpleNamespace(logprob=client_lp)}])
        observed = [{"token_id": 9, "raw_logprob": -0.3, "sampler_logprob": -0.3,
                     "temperature_only_logprob": -0.2}]
        return gen, observed

    def test_raw_values_pass_and_temperature_only_values_fail(self):
        gen, observed = self.fixture()
        self.assertTrue(compare_returned(gen, observed)["passed"])
        gen, observed = self.fixture(-0.2)
        self.assertFalse(compare_returned(gen, observed)["passed"])

    def test_token_alignment_is_required(self):
        gen, observed = self.fixture()
        observed[0]["token_id"] = 10
        with self.assertRaises(ValueError):
            compare_returned(gen, observed)

    def test_nonfinite_observer_fails(self):
        gen, observed = self.fixture()
        observed[0]["raw_logprob"] = float("nan")
        with self.assertRaises(ValueError):
            compare_returned(gen, observed)
