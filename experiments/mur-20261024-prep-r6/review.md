# Supervisor review: mur-20261024-prep-r6

Decision: **accept the engineering preparation gate; do not accept the accuracy-preservation claim.**
Authorize the separate frozen `mur-20261024-test100` task, conditional on its cache-control
engineering check. No full500 run is authorized by this review.

The validated completion event is
`9d9bc914f5060b8588f7ce275af9242b98a1f17456e3d830ca2ba82a7102b5b2`.
Report SHA256: `e38fe37d818ecf444557e70c6cb25483819335789b52c9c4d034065eb60c9d4e`.
Raw trace SHA256: `9128416c2fcc13676af2c80ffee15453d8a1e4f304d43ce3fdaa862a957a8c53`.
The remote bridge verified completion, artifact identities and the matching frozen task.

## Independent evidence review

The supervisor recomputed the eight-probe raw/sampler/client probability comparisons:
all passed with maximum error0, including both retained text-stop tails (237 sampler
tokens,235 returned). The negative control differs by0.26924. This proves the direct
signal contract only; the earlier HF and independent-prefill checks remain failed.

All60 traces have complete policy+critic ledgers, valid final answers, no run/context/
budget errors and no forced triggers. Independent recomputation checked all54 MUR
steps, every momentum update and trigger, all5 genuine MUR selections, and236 candidate
judgments across the guided methods. Original numeric evidence is retained. No local
model inference was used for this review.

| Method | Correct /20 | Mean policy+critic tokens |
|---|---:|---:|
| CoT |16|528.75|
| Per-step Guided Search |16|5036.90|
| MUR |15|1092.75|

MUR saves78.305% tokens versus per-step, but accuracy is5 percentage points lower.
Paired question bootstrap (10000 replicates,seed20261024) gives accuracy-delta95%
interval[-15,0] percentage points and token-savings interval[66.624%,88.273%]. These
small development-sample intervals do not establish non-inferiority. CoT is cheaper
than MUR and scored higher here; it must remain in all subsequent reports.

Both questions with any cross-method correctness disagreement were audited with
method labels hidden. Six answer-equivalence judgments agreed with the automated
verifier. Labels were fixed before revealing the mapping; no scores were revised.
The sanitized audit is in `independent-review-metrics.json`; raw question/answer
materials remain outside Git.

## Corrections and limitations

The stored auxiliary `trigger_rate` uses an incorrect denominator. Correct rates are
54/54 for per-step and5/34 eligible non-first steps for MUR (5/54 over all steps).
The original artifacts are preserved; this does not alter accuracy or token totals.
The next task must fix and test this denominator.

Only12/20 first proposals match between seeded CoT and MUR, although their common
first-step algorithm is identical. Some zero-trigger final trajectories also differ.
Prefix caching and BF16 execution shapes are a plausible explanation, not an established
cause. The next task disables prefix caching uniformly and checks common no-trigger
paths on two development questions before touching the test subset. No alpha/gamma,
sampling or scientific prompt tuning is permitted.

API-returned output tokens are a compute proxy, not exact FLOPs. Different model
tokenizers and asynchronous stop cancellation limit that interpretation. Shared GPU
latency is descriptive. This is one seed and20 development questions, not MATH-500
test performance or a complete reproduction.

## Resources and next decision

R6 measured2115.96 GPU allocation seconds including cleanup and140863 generated
tokens (237 diagnostic,133168 scientific,7458 grader). Peak new memory11885MiB;
minimum free8994MiB. All three phase process groups cleaned up successfully.
Prior conservative charges plus R6 measured time total6090.79 seconds. This total is
not an all-measured number: R1 and R3 charges are conservative. Historical token
totals omit unknown R1 work and must not be described as an exact project total.

The measured projection for500x3 generation alone is13.97 GPU-hours, exceeding the
12-hour total project budget even before prior work and grading. Freeze an exploratory
100-question subset before any test generation: seeded permutation of all500 IDs,
first100, four25-question chunks, all three methods, no outcome-based stopping.
Allocate at most4 GPU-hours including the engineering check and grading. This fits
the original project ceiling and gives a first independent test; it does not replace
the full500 target or justify a full-benchmark claim.

## Version provenance

Actual model-run commit: `ae8c5a2be16aa52181171019e02817af3938d3a6`.
Post-run summary fix: `a5bd645810850fb21e202aae4907b523c405bf28`.
Evidence: `3fa2f59bab9bddff8d8b28ab1df6661259c4f284`.
Worker final HEAD: `56eba9eafa680a956e163c35b134df5133a5aa47`.
The supervisor fast-forwarded these commits and audited162 historical file versions,
zero findings. Review/task changes are subsequent commits, not the execution version.
Relay push is pending at review creation and is verified separately before dispatch.
