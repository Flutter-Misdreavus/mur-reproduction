import math
import unittest
from mur.control import Momentum, Ledger, select_judgment


class Contracts(unittest.TestCase):
    def test_constant_sequence_has_no_cold_start_triggers(self):
        m = Momentum()
        for _ in range(20):
            self.assertFalse(m.trigger(0.3))
            m.accept(0.3)
            self.assertAlmostEqual(m.history, 0.3)

    def test_detect_before_accept_and_update_selected_step(self):
        m = Momentum()
        self.assertFalse(m.trigger(2.0))
        m.accept(0.2)
        boundary = 0.2 - math.log(0.9)
        self.assertFalse(m.trigger(boundary - 1e-9))
        self.assertTrue(m.trigger(boundary + 1e-9))
        m.accept(0.1)  # selected replacement, not discarded uncertain proposal
        self.assertAlmostEqual(m.history, (0.9*0.02+0.01)/0.19)
        self.assertEqual(m.count, 2)

    def test_total_includes_discarded_and_both_critic_phases(self):
        l = Ledger()
        l.record("policy", "proposal_discarded", 10, 5)
        for _ in range(4):
            l.record("policy", "candidate", 10, 7)
            l.record("critic", "analysis", 20, 3)
            l.record("critic", "judgment", 23, 2)
        l.record("verifier", "evaluation", 20, 100)
        self.assertEqual(l.total(), 53)
        self.assertEqual(l.total("policy"), 33)
        self.assertEqual(l.total("critic"), 20)
        self.assertEqual(l.total("verifier"), 100)

    def test_yes_no_and_invalid_selection(self):
        self.assertEqual(select_judgment([
            dict(label="No", probability=0.9), dict(label="No", probability=0.2)]), (1, False))
        self.assertEqual(select_judgment([
            dict(label="Yes", probability=0.4), dict(label="Yes", probability=0.8)]), (1, False))
        self.assertEqual(select_judgment([
            dict(label=None, probability=None), dict(label="No", probability=0.8)]), (0, True))

    def test_invalid_signals_fail(self):
        for x in (float("nan"), float("inf"), -0.1):
            with self.assertRaises(ValueError):
                Momentum().trigger(x)


if __name__ == "__main__":
    unittest.main()
