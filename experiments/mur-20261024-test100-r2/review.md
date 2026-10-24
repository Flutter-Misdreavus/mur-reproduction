# Supervisor review — mur-20261024-test100-r2

**Decision: accept the partial evidence; reject fixed100 completion and an accuracy-preservation claim.**
The frozen 100-question task is incomplete. Generation stopped at the resource limit,
not because of observed accuracy. No further model run is authorized by this review.
The worker completion event is validated as `FAILED` and matches the frozen local task.

## Results and scope

Two complete chunks produced 150 automatically graded trajectories on the first 50
frozen questions. The third chunk produced 46 saved trajectories before timeout:
15 further complete three-method pairs and one CoT-only answer. The interrupted
trajectory was not checkpointed. The third chunk was not automatically graded and
the fourth chunk never started. There are 65 complete generated pairs, 196 saved
trajectories, and only 50 automatically scored pairs. No missing answer is imputed.

| Method | Automatic correct /50 | Reviewed correct /50 | Mean policy+critic output tokens |
|---|---:|---:|---:|
| Stepwise CoT | 37 | 36 | 563.60 |
| Per-step Guided Search | 40 | 40 | 4936.96 |
| MUR | 37 | 36 | 829.30 |

MUR saves **83.20%** of output tokens versus per-step search (paired 95% CI
76.26%–88.44%). Its automatic accuracy difference is **−6 pp** (95% CI −18 to +6 pp).
After GPT supervisor answer-equivalence review, the difference is **−8 pp**
(95% CI −20 to +2 pp). Both miss the preset point target of no more than 2 pp loss;
neither supports statistical non-inferiority. These intervals describe the completed
subset and do not account for resource-dependent missingness from the fixed100 task.
CoT is cheaper than MUR with the same accuracy on these 50 questions.

For a separately identified supplementary analysis, all 195 final answers in the
65 complete pairs were reviewed with method and automatic labels hidden. The
additional 15 questions score CoT 8/15, per-step 6/15 and MUR 6/15. Combined:

| Method | Reviewed correct /65 | Reviewed accuracy | Mean output tokens |
|---|---:|---:|---:|
| Stepwise CoT | 44 | 67.69% | 675.55 |
| Per-step Guided Search | 46 | 70.77% | 5807.17 |
| MUR | 42 | 64.62% | 1126.05 |

This supplementary comparison saves 80.61% of output tokens (95% CI 74.26%–85.86%)
but loses 6.15 pp accuracy (95% CI −15.38 to +1.54 pp). It also misses the target.
The extra 15 questions have **no automatic labels**; this analysis does not replace
the original 50-question automatic result or manufacture a 100-question result.
The unpaired CoT answer contributes to costs but not paired accuracy comparisons.

## Failure and resource reconciliation

The third generation phase was terminated by its frozen 2940-second execution
limit, then cleaned up in 0.54 seconds (2941.22 seconds total allocation). Its
3300-second combined chunk budget reserved 300 seconds for grading and 60 for
cleanup; unused grading time was not reassigned to generation. No OOM, telemetry
failure or model error was recorded. The GPU returned to its preflight shared-use
baseline after cleanup. The fourth chunk was not launched and the third was not
restarted. The whole task's 4-hour allocation budget was not exhausted: this was
an earlier **per-phase** cap.

The inner runner allowed 2970 seconds, longer than the outer 2940-second execution
cap. Consequently the outer termination preceded a graceful runner budget exit.
The killed phase has no final counter, and calls from its unfinished trajectory
may be absent from saved traces. Preserving per-call counters and making the inner
deadline earlier than the outer cleanup deadline are follow-up engineering needs;
no timeout or scientific parameter was changed during this run.

| Resource | Reconciled value |
|---|---:|
| Current test driver allocation | 7868.17 s |
| Earlier loader-failure attempt, including its engineering check | 78.27 s |
| Both driver attempts | **7946.44 s / 2.2073 allocation hours** |
| Known saved generated tokens, both attempts including grading | **at least 514815** |
| Conservative token charge, both attempts | 934875 |
| Actual deduplicated verifier output tokens / calls | 15770 / 81 |
| Historical preparation plus both attempts | 14037.23 s / 3.8992 charged hours |

The current attempt's saved token lower bound is 513509. The incomplete chunk's
179940 saved tokens are already covered by its 300000-token reservation; they are
not added again to the conservative charge. The earlier premodel failure reserved
another 300000 tokens but generated zero test tokens. The final exact number of
actually generated tokens is unknown, and historical preparation charges are not
all measurements. See [resource reconciliation](supervisor-resource-reconciliation.json).
Peak new GPU memory was 12109 MiB and minimum free memory was 8770 MiB.

## Corrections to worker reporting

The immutable worker event/report and original automatic metrics are retained.
This review supersedes three misleading or incomplete descriptions:

- Worker fields named `actual_total_tokens_including_verifier` and similar are
  **known saved lower bounds**, not exact totals after interruption.
- The worker narrative initially cited the first engineering attempt's raw hash.
  The second attempt's verified raw hash is
  `72ae9a1657f6faa4cb2ebc2d6cde470b253cd52a9354bcaf574a329b5f7cbfd3`.
- The retry reused the same run ID despite `max_attempts=1`. The generic protocol's
  repair allowance does not erase this task-specific workflow deviation. Both
  attempts are preserved and charged; no earlier test outcomes existed to select on.

## Verification and reproducibility

All 196 saved output-token ledgers reconcile. On the 195 paired trajectories,
184 MUR steps, 12 genuine triggers and 776 candidate judgments pass independent
checks; all 65 first CoT/MUR proposal token sequences match. All 195 paired final
answers were reviewed, and two automatic labels changed for one underlying shared
verifier error. Both engineering attempts were checked, including all six shared
pre-trigger steps in the second attempt. The final-answer review is GPT review,
not human annotation.

The worker reports 60 passing remote CPU tests. The supervisor independently reran
the 12 fixed100 regression tests after the post-run accounting fix; all passed.
The standalone statistics script exactly reproduces the separately computed
50-question paired bootstrap. It consumes only sanitized numeric rows and IDs:

```bash
python3 scripts/recompute_supervisor.py experiments/mur-20261024-test100-r2/supervisor-rows.json
```

Artifacts: [statistics](supervisor-statistics.json), [numeric rows and review labels](supervisor-rows.json),
[ledger checks](supervisor-ledger-checks.json), [validated receipt](supervisor-receipt.json).
No benchmark text, model answer, credential, model weight or raw DSH log is added.

Post-run worker evidence and verifier-accounting fix:
`e3e12dd06dc9ca64b29c086c52248a13e1b52eb8`.
The supervisor review commit is the commit introducing this file; resolve it with
`git log -1 --format=%H -- experiments/mur-20261024-test100-r2/review.md`.
The delivered current HEAD and verified private-push status are recorded in the
local ignored delivery receipt and the final handoff, rather than assigning that
post-run HEAD to model execution. The worker had no GitHub remote; the supervisor
relays its commits after privacy review. No history rewrite or force push is used.

Do not resume this failed matrix or tune MUR against these observed test results.
A future study needs a separately frozen plan: first repair checkpoint/accounting
and audit implementation differences on development data, then allocate adequate
time to the already fixed unanswered questions or a separately specified test.
This run provides a negative exploratory result, not evidence to expand to full500.

## Independent review method

The supervisor recomputed all complete scored chunks from raw generation traces and
the original per-row grading metrics. The checks cover the API-returned output-token
ledger (including discarded proposals, all candidates and both critic phases),
accepted-token counts, proposal mean NLL, bias-corrected EMA and trigger decisions,
candidate selection, and equality of first CoT/MUR proposal text and token IDs.
Triggered accepted-candidate NLL is retained only as a mean: the EMA check uses
that recorded mean, rather than independently reconstructing its per-token NLL.
This is an evidence limitation, not an additional successful parity test.

The pinned MATH-500 source file, all 500 source rows, the seeded permutation, all
100 frozen manifest identities and their question/row hashes were independently
verified. The manifest is unchanged across attempts. No problem was replaced or
selected using its outcome. The generation split excludes reference answers and
solutions.

The supervisor reviewed final-answer equivalence with method identities and
automated labels hidden, then joined the fixed judgments back to the original
metrics. This was GPT supervisor review, not human annotation and not an
independent human gold set. All completed scored rows were reviewed after a real
verifier false positive was found. Original automated labels remain the
preregistered primary result; reviewed labels form a separately identified
secondary result. Neither grader prompts nor model settings were changed.

One underlying verifier error affected both CoT and MUR on
`test-007-659b49f54a96`: exact cache-key inspection confirmed that the verifier
ignored an incorrect student answer and returned `Final Decision: Yes` after its
own incorrect derivation. This was not a row-join or parser bug. For
`test-100-5b1bb536f3d2`, an output-length-truncated ambiguous verifier response
was conservatively scored incorrect; independent answer-equivalence review
agreed. Duplicate cached judgments must not be counted as independent verifier
calls or separately charged actual grader work.

## Attempt and version provenance

The original `mur-20261024-test100` attempt was cancelled after CPU preparation,
before any GPU phase or test generation. Implementation `fba4a79` was followed by
supervisor guards at `62dfdcf`: combined per-chunk allocation, cleanup/grading
reserves, conservative charging of unknown token counts, pinned GPU selection,
per-phase clean-tree checks, strict grading joins and exact point-threshold
arithmetic.

The first driver attempt under `mur-20261024-test100-r2` ran its engineering check
at `2ed1d21ef298dee73ec066a662054762a176f574`. It passed (4 development
trajectories, 1306 tokens, 77.04 allocation seconds), then the test loader failed
before model initialization because it constructed a duplicated data directory.
The failure phase used 1.23 seconds and generated no test tokens. Its total was
78.27 allocation seconds and 1306 known generated tokens. An additional 300000
tokens were conservatively reserved because the failed phase lacked a final
counter; that reservation is not 300000 observed generated tokens.

The minimal loader fix and regression test were committed at
`5c91a8a178e2c39ba6aa50c3d116524eae8ecf6e`. The worker preserved the first attempt
under an ignored attempt archive but restarted the driver under the same run ID
and run root. This is a workflow deviation from unique attempt IDs. No test
answer existed before the restart, and no scientific setting or frozen sample
changed. The engineering check was unnecessarily repeated (77.19 seconds,
1306 tokens); both checks and the loader failure count toward total costs.
The actual test-generation commit is `5c91a8a178e2c39ba6aa50c3d116524eae8ecf6e`,
not the later evidence or supervisor-review HEAD.

## Interpretation limits

This is a single-seed exploratory subset of MATH-500, not the full paper
benchmark. The comparison target is at least 40% lower policy-plus-critic
output-token use and no more than 2 percentage points lower accuracy than
per-step Guided Search. A point estimate meeting that target is separate from
statistical non-inferiority, which requires the paired 95% accuracy-difference
interval to lie strictly above -2 percentage points. Paired resampling uses
10000 question-level bootstrap replicates with seed 20261024.

API-returned output tokens are a computation proxy, not measured FLOPs. Input
processing, unreturned sampler work around stop strings, model loading and
shared-GPU interference prevent interpreting token savings as an exact GPU
speedup. Grading is a separate cost. Allocation seconds include loading and
cleanup, and are not measurements of active arithmetic time. All three methods,
including cheaper stepwise CoT, must be reported.

The historical preparation budget charge is 6090.79 seconds and combines
conservative reservations for earlier failures with measured later phases. It
must not be described as entirely measured use. Historical first-attempt token
accounting is incomplete, so an exact project-wide generated-token total is
unavailable. Raw traces, benchmark texts, model weights, environments,
credentials and DSH sessions remain outside Git; only sanitized metrics,
identities, configurations and reviews belong in the private repository.
