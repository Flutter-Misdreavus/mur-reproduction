# Supervisor review: mur-20261024-prep

Decision: reject the initial implementation gate; cancel partial development smoke and
repair before any MATH-500 evaluation. No benchmark conclusion is accepted.

Bootstrap: 33cb39f1fac7e47e0ed281209ffe0d61605b3cb1.
Implementation: d8d8074. Raw-logprob validation update and 20-item smoke: c5f0c12.
At cancellation, 14 complete development trajectories were retained (CoT5, per-step5,
MUR4). All are diagnostic only and must not be mixed with the repaired run.
The wrapper recorded CANCELLED; no valid final worker report was available.
The separate smoke unit was verified inactive and GPU allocations released.

Findings:
1. Chat role-boundary EOS tokens were removed despite the frozen protocol. A test
   incorrectly enshrined this bug.
2. Critic parsing searched unrelated later Yes/No tokens. The verdict prefix is already
   in the prompt; the parser must anchor to the first generated boxed value.
3. NLL rescoring used a separate prompt forward; input accounting omitted the scored
   output prefix. This is avoidable on vLLM0.8.5 V1, whose installed sampler explicitly
   returns raw logprobs before penalties/temperature. V0 differs.
4. Combined engine memory fractions 0.29+0.38 exceeded the task's14GiB new-memory ceiling.
   The observed inference process used16972MiB.
5. Verifier cache omitted model revision and sampling seed/settings; ambiguous outputs
   could still receive a positive score.
6. Summary read the wrong run-status location, allowed runtime errors and fabricated
   a static memory-compliance sentence. These are not acceptable evidence.
7. Executor launched a separate systemd unit, so wrapper cancellation alone was not
   sufficient cleanup. Both task-specific units were inspected during cancellation.

Supervisor repairs preserve role markers, select the anchored actual verdict token,
require vLLM0.8.5 V1 generation logprobs (no extra score forwards), lower engine allocation,
stop the stage after a trajectory error, fix cache/summary semantics and add regression
tests.19 CPU contract/regression tests pass. GPU validation is still required.
The next run must revalidate direct raw generation logprobs and all budgets, audit source
files and runtime evidence, and perform the same20 development questions from scratch.
No hyperparameters were selected on the test set.
