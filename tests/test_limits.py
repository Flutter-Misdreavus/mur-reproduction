import unittest

from mur.limits import ContextLimit, StageBudget, StageBudgetExceeded


class LimitsContracts(unittest.TestCase):
    def test_reserve_blocks_before_call_and_accounts_actual(self):
        b = StageBudget(max_generated=100, max_wall_seconds=60, max_model_len=64)
        b.reserve([20, 20], "batch")
        self.assertEqual(b.generated, 0)
        b.account("policy", 5)
        b.account("critic", 7)
        self.assertEqual(b.generated, 12)
        with self.assertRaises(StageBudgetExceeded):
            b.reserve(99, "next")
        self.assertEqual(b.generated, 12)

    def test_context_fit_never_truncates_prompt(self):
        b = StageBudget(max_generated=100, max_wall_seconds=60, max_model_len=64)
        self.assertEqual(b.fit(10, 100, "call"), 54)
        self.assertEqual(b.context_capped_calls, 1)
        with self.assertRaises(ContextLimit):
            b.fit(64, 10, "full")
        with self.assertRaises(ContextLimit):
            b.fit(65, 10, "over")

    def test_wall_is_global(self):
        clock = [0.0]
        b = StageBudget(max_generated=100, max_wall_seconds=10,
                        max_model_len=64, clock=lambda: clock[0])
        clock[0] = 9.9
        b.check_wall("ok")
        clock[0] = 10.0
        with self.assertRaises(StageBudgetExceeded):
            b.check_wall("late")

    def test_unknown_role_rejected(self):
        b = StageBudget(max_generated=10, max_wall_seconds=10, max_model_len=10)
        with self.assertRaises(ValueError):
            b.account("judge", 1)


if __name__ == "__main__":
    unittest.main()
