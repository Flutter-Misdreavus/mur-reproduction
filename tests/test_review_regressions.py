import math
import unittest
from types import SimpleNamespace
from mur.backend import Generation, generation_nll, judgment_from_generation
from mur.prompts import has_final_answer


def gen(tokens, logps):
    return Generation(text="".join(tokens), token_ids=list(range(len(tokens))),
                      logprobs=[{i: SimpleNamespace(decoded_token=t, logprob=lp)}
                                for i, (t, lp) in enumerate(zip(tokens, logps))])


class ReviewRegressions(unittest.TestCase):
    def test_only_initial_verdict_is_selected(self):
        g = gen(["{", "Yes", "}", " Explanation ", "No"], [-0.1, -0.2, -0.1, -1.0, -0.9])
        j = judgment_from_generation(g)
        self.assertEqual(j["label"], "Yes")
        self.assertEqual(j["position"], 1)
        self.assertAlmostEqual(j["probability"], math.exp(-0.2))

    def test_unrelated_yes_is_malformed(self):
        g = gen(["No judgment. ", "Yes"], [-0.1, -0.2])
        self.assertIsNone(judgment_from_generation(g)["label"])

    def test_raw_nll_uses_sampled_token_not_top_candidate(self):
        g = gen(["a", "b"], [-0.2, -0.8])
        for entry in g.logprobs:
            entry[999] = SimpleNamespace(decoded_token="other", logprob=-0.01)
        self.assertAlmostEqual(generation_nll(g)["mean_nll"], 0.5)
        self.assertEqual(generation_nll(g)["probe_tokens"], 0)

    def test_quoted_marker_or_intermediate_box_does_not_finish(self):
        self.assertFalse(has_final_answer('Remember to say "the answer is" later.'))
        self.assertFalse(has_final_answer(r"Intermediate result: \\boxed{2}"))
        self.assertFalse(has_final_answer("the answer is"))
        self.assertTrue(has_final_answer("Therefore, the answer is 2."))


if __name__ == "__main__":
    unittest.main()
