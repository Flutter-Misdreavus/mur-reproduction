# Minimal MUR reproduction

This project tests whether momentum-guided step selection reduces inference computation
while approximately preserving accuracy on MATH-500.

Scope: Qwen3-1.7B (non-thinking), GenPRM-1.5B, and three methods:
stepwise CoT, Guided Search at every step, and Guided Search with MUR.
There is no training. See PROTOCOL.md for frozen settings and acceptance rules.
Current status: the fixed 100-question test **stopped at its time limit and is incomplete**.
The first 50 questions were generated and automatically graded for all three methods.
MUR used 83.2% fewer policy-plus-critic output tokens than per-step Guided Search,
but scored 37/50 versus 40/50 automatically, or 36/50 versus 40/50 after GPT supervisor
answer-equivalence review. CoT scored 37/50 automatically (36/50 reviewed) and cost
563.6 tokens/question, versus MUR's 829.3. The accuracy-preservation target was **not met**.
An additional 15 fully generated question pairs were reviewed separately; the resulting
65-question supplementary comparison also misses the target. This is an exploratory,
resource-truncated result, not a completed fixed100 or MATH-500 reproduction.
See the [supervisor review](experiments/mur-20261024-test100-r2/review.md) for original
and reviewed labels, confidence intervals, failure provenance and cumulative costs.
The earlier [development review](experiments/mur-20261024-prep-r6/review.md) remains available.

Recompute the sanitized result without models or benchmark text:

```bash
python3 scripts/recompute_supervisor.py experiments/mur-20261024-test100-r2/supervisor-rows.json
```

Reference: Yan et al., *MUR: Momentum Uncertainty guided Reasoning for Large Language Models*,
[arXiv:2507.14958v5](https://arxiv.org/abs/2507.14958v5), 2026-05-24.
Official code reference: [yayayacc/MUR](https://github.com/yayayacc/MUR),
commit 5cc34676f1a0d0515b7940260cdde98a69f2b671.
The paper PDF hash is dc981152b007458f4d77c5e3e89427be988fc07f03e3c0f827314815b0fadbd3.
The paper and upstream repository are references; this repository does not redistribute
their PDF, datasets, weights or source tree. Implementations added here must preserve
attribution and record deviations. Do not assume the upstream source has a permissive
license merely because the paper mentions MIT.

Run the full contract suite inside the pinned remote environment with
`runtime/venv/bin/python -m unittest discover -s tests -v`; an installed-sampler
environment check requires vLLM. Pure controller checks also run locally with
`python3 -m unittest discover -s tests -p test_control.py -v`.
Model execution is remote only; environment validation and smoke must pass before the
full benchmark is launched. Runtime evidence and infrastructure stay outside Git.
