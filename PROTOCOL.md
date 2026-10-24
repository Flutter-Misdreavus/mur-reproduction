# Frozen first experiment

## Question and scope
Can MUR reduce total generated tokens by at least 40% versus per-step Guided Search,
with accuracy no more than 2 percentage points lower? These are reproduction
acceptance thresholds, not statistical claims made by the paper.
Only Qwen3-1.7B and MATH-500; no other model sizes, ablations, gamma search or training.
Initial sampling seed 42. Three methods on the identical 500 questions.
Use BF16 on a remote CUDA GPU. No local inference and no precision substitution.

## Sources
Use exact revisions in configs/sources.json. Verify downloaded files against repository
metadata where available and record dataset file SHA256, row order, row hashes and IDs.
Formal evaluation contains every MATH-500 row exactly once per method.
For development use 20 MATH training questions, deterministically selected by a fixed
seed, after normalized-question deduplication against all 500 test questions. Freeze the
development source revision and manifest before smoke. Never adjust parameters on test.
Do not ship data, prompts containing benchmark questions, responses or weights in Git.

## Algorithm
Generate step proposals using the paper Appendix B user prompt, an empty system prompt
for MATH, and Qwen enable_thinking=False. Keep the original chat template including
special role boundaries. Stop before the next Step marker; inspect actual token/text
alignment and accidental marker matches during smoke. Use the same segmentation in
all methods. A final-answer marker terminates only the accepted step, after the method's
selection, not a discarded proposal. Avoid matching a quoted marker in ordinary text.

m_t is mean negative log probability of actual generated tokens under the model,
not distributional entropy. Record whether the backend returns raw model logprobs
or transformed sampling logprobs; verify this on a small teacher-forced comparison.
A systematic mismatch is a gate failure, not permission to silently change the signal.

Maintain raw EMA U_t = alpha*U_(t-1) + (1-alpha)*m_accepted, with U_0=0.
For t>1, compare the proposal to corrected historical mean
B_(t-1) = U_(t-1)/(1-alpha**(t-1)); trigger iff m_proposal > B_(t-1)-log(gamma).
First MUR step is never scaled. After selection update EMA once from the accepted step.
alpha=gamma=0.9. Expose bias correction for diagnosis but do not choose it on test.

CoT: one proposal per step and no critic.
Per-step search: one proposal plus four fresh candidates and critic selection at EVERY
step, including first step; this follows the upstream baseline's compute accounting.
MUR: one proposal, then four fresh candidates and selection only on triggered steps.
Do not count the original proposal as one of the four. Count discarded proposals too.
No extra tuning, retries-until-correct, rejection sampling or test labels in generation.

Each candidate receives one GenPRM verification path and a Yes/No judgment. Use Appendix B
prompt and the official GenPRM analyze/output format. Choose largest judgment-token
P(Yes) if any verification's actual judgment is Yes; otherwise smallest P(No).
Use the designated judgment position, not unrelated Yes/No tokens anywhere in the text.
Missing or malformed judgments must be logged; use a declared deterministic first-candidate
fallback and report its rate. No label/verifier access during candidate selection.

## Sampling and limits
Policy and critic generation: temperature 0.6, top_p 0.8, top_k 20, presence_penalty 1.5.
Policy step max 2048. Up to 20 accepted steps. Candidate steps use the same settings.
Critic analysis max 1024 and output max 2048, following the upstream two-call structure.
One verification per candidate. Seed every request deterministically from seed, question
ID, step, role, candidate index, so scheduling does not change the random stream.
Candidate/critic requests may be batched with separate per-request seeds.

The paper's 16384 output length is operationalized as a cap on each accepted reasoning
trajectory, not a cap on all candidate and critic compute summed together (Table 1
contains larger aggregate token counts). Keep the accepted-trajectory cap separate
from the total compute ledger. Final completion on reaching a step cap may use remaining
accepted-trajectory budget only and must be identical across methods, recorded as fallback.
No silent prompt truncation. Track EOS, length/step limit, empty output and context limits.
Budget-exhausted or unparseable solutions remain in the denominator and are marked.
The chosen cap semantics are an explicit reproduction interpretation because the released
scripts do not enforce a single aggregate 16384-token budget.

## Cost and evaluation
Primary cost: all generated policy tokens plus all generated critic tokens, including
discarded candidates, proposal tokens, analyze/output phases and final fallbacks.
Also report policy, critic, input tokens, accepted tokens, calls, step counts, trigger
counts, latency and peak allocated GPU memory separately. Distinguish tokenizer-specific
token counts; raw token sums do not assert identical FLOPs. Report GPU model and sharing.
The final-answer evaluation model's tokens/time are separate from inference costs.

Run TIGER-Lab/general-verifier with the frozen revision and official evaluation prompt
only AFTER solutions are frozen. Match the released evaluator's temperature 0.7,
max_tokens 4096, n=1, with deterministic per-question seed. Strictly parse the final
Final Decision: Yes/No field, flag ambiguity. Cache judgments by output/prompt/model
hash and use deterministic symbolic checks and blind manual checks for method-discordant
answers. Preserve both original verifier and audited labels; never selectively relabel
one method. Do not call an answer-equivalence checker a step PRM.

Compute paired differences over the same 500 question IDs. Report accuracy, total token
savings 1-mean(C_MUR)/mean(C_PerStep), 10000 paired question-bootstrap 95% intervals
with bootstrap seed 20261024, raw per-question numeric metrics and denominators.
Point estimate acceptance: savings >=0.40 and accuracy_delta >=-0.02.
Evidence-supported non-inferiority additionally needs the lower paired 95% bound above
-0.02; otherwise mark inconclusive, not success. No automatic additional seeds in this
first experiment. CoT must be reported even if it is more efficient or more accurate.

## Stage gates and resources
Stage prep: implement, validate environment, source manifests and CPU contract tests,
then 20 development questions for all three methods (60 trajectories), plus verifier.
Return evidence and stop for supervisor review. Do not launch the full 500 in this task.
Stage main: after supervisor review freeze implementation and run 3*500 at seed42.
Total experiment ceiling: one GPU simultaneously, 12 GPU-hours across prep/main/evaluation;
prep allocation 1 GPU-hour, 2 hours wall; remaining budget decided from measured throughput.
Data/model/environment incremental disk ceiling 20 GiB; maintain at least 25 GiB free on
data volume. Reuse own compatible environments/caches read-only; never pip-upgrade shared
environments or install through symlinks into other projects.
If no scheduler exists, use a bounded systemd process group. Do not kill unrelated jobs.
Check free memory before loading, cap our allocations at 14 GiB and leave 4 GiB headroom.
If GPU sharing is needed record it; latency then is descriptive, not a controlled speedup.
Stop on OOM, NaN, data leakage, invalid logprobs, gross segmentation failure, budget
exhaustion, disk limit, or need for a scientific protocol change. One bounded repair
attempt may use a new run_id and new commit, keeping all previous evidence.

## Reproducibility and publication
Commit code/config before every real run. Record actual run SHA, fix SHA, current HEAD
and push status separately. Repository stays private during experiments.
Tracked artifacts use portable paths; secrets, private infrastructure/account details,
raw DSH transcripts, raw question/answer traces, data and model files stay untracked.

## Validation reference amendment (before any complete smoke or test)
The r3 cross-engine BF16/HF numerical test failed and remains reported as failed.
It is not a scientific accuracy result. To validate the actual MUR signal, r4 compares
raw generation logprobs with an identity-sampling teacher-forced prefill in a fresh
instance of the same pinned vLLM V1 engine, on exact token IDs. Eight synthetic probes
are frozen in configs/validation.json. Numerical tolerances are unchanged; no probe
may be dropped. Require repeated-token coverage and finite complete token vectors.
The independent HF comparison remains a diagnostic and is not relabelled as passing.
This separates raw-versus-transformed probability correctness from cross-engine BF16
numerical parity. All three scientific methods still use the identical frozen backend.
Keep diagnostic forward/token costs separate from main inference but within the resource
budget. R4 has one attempt and 3000 GPU seconds; no test generation until supervisor
review of the complete development gate. Combined preparation ceiling is 2 GPU-hours.

## Direct probability-contract gate (r5)
R4 also failed its independent-prefill comparison; that result remains failed.
The required property is that the supplied MUR signal equals raw probability in the
actual generating forward. Validate it directly with a read-only sampler observer,
cloning logits before transforms, computing float32 log_softmax, and comparing both
sampler and returned client probabilities for each exact selected token. Call the
original sampler once; consume no RNG and modify no logits/metadata. Remove the
observer by ending the validation process before scientific generation. Run the same
eight frozen probes. Maximum arithmetic error <=5e-5 on identical logits, complete
exact token alignment, finite outputs, repeated tokens, and a temperature-only negative
control differing by >0.001 somewhere are all required. This does not establish HF
or independent-prefill parity. Do not relax or replace this direct gate after failure.
Conservatively charge R1's entire3600-second GPU allocation, R3 280, R4 62.77, and
R5 <=3000; their sum stays below the combined2GPU-hour prep ceiling. These charges
are distinct from actual measured GPU use. No test-set results informed these changes.

## Text-stop observer alignment correction (r6)
The direct r5 gate verified five probes exactly but aborted on a token-count mismatch.
All client-returned tokens must match the leading sampler token sequence exactly.
Retain and validate any trailing sampler entries, permitting them only for an explicitly
reported configured text-stop cancellation. Never drop a returned token or permit an
unexplained tail. Save failing vectors before raising. Keep the same eight probes,
float32 arithmetic bound and negative control. Validation resource token counts include
these tails. Main output-token metrics follow the paper-style API counts; asynchronous
text-stop cancellation is another reason this proxy does not measure exact GPU FLOPs.

## First independent test amendment (after r6, before test generation)
The reviewed development gate passes engineering checks but does not meet the
accuracy guardrail: MUR15/20 versus per-step16/20, with78.3% output-token savings.
The measured full500x3 generation projection is13.97 GPU-hours, beyond the12-hour
project ceiling. The next stage is an explicitly exploratory fixed100-question test,
not a full500 result. Shuffle all500 original row indices with random.Random(20261024),
take the first100, commit their ordered IDs/hashes before generation, and run all three
methods in four25-question chunks. Stop on resource failure, never on accuracy.
Do not tune algorithm/sampling parameters or replace questions. The full500 target
remains a separate later decision. Point criteria and paired confidence-interval
criteria remain unchanged; report inconclusive non-inferiority when appropriate.

Uniformly disable prefix caching for policy,critic and verifier to remove cross-method
cache-order coupling observed in development. Verify common pre-trigger CoT/MUR paths
on dev indices0 and3 before test; at most300 GPU seconds, no repeated variants after
failure. This is an engineering control, not an accuracy-driven search. Correct the
auxiliary trigger-rate denominator and deduplicate equivalent pending grader requests.
Keep all earlier failed diagnostics and original development artifacts unchanged.
The100-question stage receives at most14400 allocation seconds and2.5M generated
tokens including engineering and grading, within the existing12GPU-hour project cap.
Its wrapper may last6hours for implementation plus execution; each compute chunk is
bounded to3300seconds with grading reserved. No local model execution is permitted.
