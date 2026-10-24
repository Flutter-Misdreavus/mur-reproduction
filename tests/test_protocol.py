import math
import unittest

from mur.control import Momentum, Ledger, select_judgment
from mur.prompts import (boxed_values, extract_answer, parse_decision, policy_prompt,
                         request_seed, strip_step_prefix, to_step_string, question_key)


class FakeTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, chat, tokenize=False, enable_thinking=False,
                            add_generation_prompt=True):
        assert tokenize is False
        assert enable_thinking is False
        assert add_generation_prompt is True
        return "<|im_start|>user\n" + chat[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"


class PromptContracts(unittest.TestCase):
    def test_step_prompt_shape(self):
        t = FakeTokenizer()
        p0 = policy_prompt(t, "Q text", [], 0)
        self.assertIn("Start your solution with 'Step0:'", p0)
        self.assertTrue(p0.endswith("\nStep0:"))
        p1 = policy_prompt(t, "Q text", ["Step0: a"], 1)
        self.assertIn("Step0: a", p1)
        self.assertTrue(p1.endswith("\nStep1:"))
        self.assertIn("<|im_end|>\n<|im_start|>assistant", p0)

    def test_answer_and_boxed_parsing(self):
        self.assertEqual(extract_answer("work\nthe answer is $\\frac{1}{2}$"), "\\frac{1}{2}")
        self.assertEqual(extract_answer("so \\boxed{7}"), "7")
        self.assertEqual(boxed_values("a \\boxed{\\frac{1}{2}} b \\boxed{3}"),
                         ["\\frac{1}{2}", "3"])
        self.assertEqual(strip_step_prefix("Step3: hello"), "hello")
        self.assertEqual(to_step_string(3, "hello"), "Step3: hello")

    def test_decision_strict(self):
        self.assertEqual(parse_decision("Final Decision: Yes"), ("Yes", False))
        self.assertEqual(parse_decision("final decision: no"), ("No", False))
        self.assertEqual(parse_decision("Final Decision: Yes\nFinal Decision: No"), (None, True))
        self.assertEqual(parse_decision("no decision"), (None, True))

    def test_seed_deterministic_and_role_separated(self):
        a = request_seed(42, "q1", 2, "candidate", 3)
        self.assertEqual(a, request_seed(42, "q1", 2, "candidate", 3))
        self.assertNotEqual(a, request_seed(42, "q1", 2, "candidate", 4))
        self.assertNotEqual(a, request_seed(42, "q1", 3, "candidate", 3))
        self.assertLess(a, 2**31)

    def test_question_dedup_normalization(self):
        self.assertEqual(question_key("  What   is $x$? "), question_key("what is $x$?"))
        self.assertNotEqual(question_key("what is x"), question_key("what is y"))


class DetectorContracts(unittest.TestCase):
    def test_bias_corrected_history(self):
        m = Momentum(alpha=0.9, gamma=0.9)
        m.accept(0.4)
        self.assertAlmostEqual(m.history, 0.4)
        m.accept(0.2)
        self.assertAlmostEqual(m.history, (0.9 * 0.04 + 0.02) / 0.19)
        self.assertTrue(math.isfinite(m.history))

    def test_trigger_uses_corrected_boundary(self):
        m = Momentum(alpha=0.9, gamma=0.9)
        m.accept(0.2)
        self.assertFalse(m.trigger(m.history - math.log(0.9) - 1e-9))
        self.assertTrue(m.trigger(m.history - math.log(0.9) + 1e-9))

    def test_first_step_never_triggers(self):
        self.assertFalse(Momentum().trigger(99.0))


class LedgerContracts(unittest.TestCase):
    def test_closure_and_roles(self):
        l = Ledger()
        l.record("policy", "proposal", 5, 3)
        l.record("policy", "candidate", 5, 4)
        l.record("critic", "analysis", 8, 2)
        l.record("critic", "judgment", 10, 1)
        self.assertEqual(l.total(), 10)
        self.assertEqual(l.total("policy"), 7)
        self.assertEqual(l.total("critic"), 3)

    def test_judgment_rules(self):
        self.assertEqual(select_judgment([
            {"label": "No", "probability": 0.8},
            {"label": "No", "probability": 0.2}]), (1, False))
        self.assertEqual(select_judgment([
            {"label": "Yes", "probability": 0.1},
            {"label": "Yes", "probability": 0.9}]), (1, False))
        self.assertEqual(select_judgment([
            {"label": None, "probability": None},
            {"label": "No", "probability": 0.2}]), (0, True))


if __name__ == "__main__":
    unittest.main()
