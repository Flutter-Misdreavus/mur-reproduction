# Minimal MUR reproduction

This project tests whether momentum-guided step selection reduces inference computation
while approximately preserving accuracy on MATH-500.

Scope: Qwen3-1.7B (non-thinking), GenPRM-1.5B, and three methods:
stepwise CoT, Guided Search at every step, and Guided Search with MUR.
There is no training. See PROTOCOL.md for frozen settings and acceptance rules.
Current status: implementation/preflight; no benchmark result is available yet.

Reference: Yan et al., *MUR: Momentum Uncertainty guided Reasoning for Large Language Models*,
[arXiv:2507.14958v5](https://arxiv.org/abs/2507.14958v5), 2026-05-24.
Official code reference: [yayayacc/MUR](https://github.com/yayayacc/MUR),
commit 5cc34676f1a0d0515b7940260cdde98a69f2b671.
The paper PDF hash is dc981152b007458f4d77c5e3e89427be988fc07f03e3c0f827314815b0fadbd3.
The paper and upstream repository are references; this repository does not redistribute
their PDF, datasets, weights or source tree. Implementations added here must preserve
attribution and record deviations. Do not assume the upstream source has a permissive
license merely because the paper mentions MIT.

Run CPU contract checks with `python3 -m unittest discover -s tests -v`.
Model execution is remote only; environment validation and smoke must pass before the
full benchmark is launched. Runtime evidence and infrastructure stay outside Git.
