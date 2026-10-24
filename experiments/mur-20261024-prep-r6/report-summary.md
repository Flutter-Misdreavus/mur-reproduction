# Run summary — mur-20261024-prep-r6

Status: **SUCCEEDED** for the preparation gate; no full-500 test was run.

## Gate outcome
- Direct same-forward probability contract: **PASS**, all 8 frozen probes.
  - Maximum absolute raw/sampler/client logprob error: `0.0` (limit `5e-5`).
  - Temperature-only negative control: max difference `0.26924046874046326` (> `0.001`).
  - Repeated-token coverage present; all probes nonempty and finite.
  - Two client text-stop cases retained a sampler tail and were validated:
    `fraction` 29 returned tokens / 1 retained sampler token and
    `repeat_sequence` 13 returned tokens / 1 retained sampler token.
    Both had `finish_reason=stop` and `stop_reason` exactly the configured
    `\nStep`; client returned tokens 235, sampler tokens 237 overall.
  - The retained tails are counted in validation work: 237 generated tokens.
- Prior numerical checks remain failed and are not relabelled:
  `cross_engine_hf_parity=false`, `independent_prefill_parity=false`.
  This gate validates the signal in the actual generating forward; it does not
  establish kernel or cross-engine numerical parity.
- Complete 20x3 development smoke and verifier: **PASS**, 60/60 trajectories,
  60/60 verifier rows, 0 missing, 0 ambiguous, 0 malformed critic judgments.
- No test generation: all 60 traces are `split=dev`; no MATH-500 test item was
  generated or used for tuning.

## Development smoke metrics (20 questions, seed 42)
These are smoke estimates, not a scientific success claim.

| method | accuracy | mean total tokens | mean policy | mean critic | mean steps |
|---|---:|---:|---:|---:|---:|
| cot | 0.80 | 528.75 | 528.75 | 0 | 2.70 |
| per_step | 0.80 | 5036.90 | 2604.65 | 2432.25 | 2.70 |
| mur | 0.75 | 1092.75 | 830.00 | 262.75 | 2.70 |

- MUR token savings versus per-step: `0.783051` (78.31%).
- MUR accuracy delta versus per-step: `-0.05` (-5.0 pp) on 20 development questions.
- Raw triggers: CoT 0, per-step 54/54 steps, MUR 5 (all genuine, step 1 or 3;
  no engineering forced case needed; `forced-trigger.json` records 5 genuine triggers).
- Token-ledger closure held for all 60 traces; all finishes were `answer`, with no
  budget stop, context limit, error, empty final answer or malformed judgment.
- Note on the stored `trigger_rate` field: `evaluate.py` divides by `steps-1`,
  which makes the per-step value `1.2558` although every one of 54 steps triggered.
  The MUR stored value is `0.1163` (5/43 under that denominator); over the 34
  eligible non-first steps the raw MUR rate is `0.1471`. This field does not affect
  the token ledger, gate, accuracy or validity of the run.

## Resources and budget
- Driver wall: `2116.65 s`; GPU allocation including cleanup: `2115.96 s`
  (`0.5878 GPU-hours`); generated tokens: `140863`, all reported, 0 unknown.
  - validate: `32.01 s`, `237` tokens.
  - smoke: `2014.57 s`, `133168` tokens.
  - evaluate: `69.38 s`, `7458` tokens.
- Peak new GPU usage: `11885 MiB` (limit 14336 MiB); minimum free: `8994 MiB`
  (floor 4096 MiB). GPU 2 was shared; latency is descriptive, not a controlled speedup.
- All limits respected: <=3000 s GPU, <=1,000,000 tokens, <=7000 s driver wall,
  14 GiB new-memory ceiling and 4 GiB free-memory floor.
- Conservative prep charges: R1 3600 s + R2 0 s + R3 280 s + R4 62.77 s +
  R5 32.06 s + R6 <=3000 s = `6974.83 s` (`1.9375 GPU-hours`) < 7200 s
  combined prep ceiling. Prior-plus-actual measured: `6090.79 s`.
- Throughput projection: `50295.59 s` (`13.971 GPU-hours`) for 1500 trajectories
  from the measured smoke wall, which is **above the 12 GPU-hour total project
  ceiling**. This is a decision item for the supervisor before any full test.
  The projection includes one-time model loading and is an estimate.

## Artifacts and limitations
- Raw traces and logs remain untracked under `runs/mur-20261024-prep-r6/`;
  the run-root `raw-smoke-traces.jsonl` is an exact copy of the smoke trace
  (`sha256 9128416c2fcc13676af2c80ffee15453d8a1e4f304d43ce3fdaa862a957a8c53`).
- Sanitized summaries are in `experiments/mur-20261024-prep-r6/`; no benchmark
  question/answer text, data, weights, credentials or raw DSH logs are tracked.
- API-returned policy+critic tokens are a compute proxy; asynchronous text-stop
  cancellation can leave engine work unrepresented. Validation tails are counted
  separately in validation work.
- The 20-question accuracy difference is not evidence for the full-500 decision;
  no full-500 generation is authorized in this task.

## Version control
- Repository: `Flutter-Misdreavus/mur-reproduction`, private, branch `exp/mur-20261024`.
- Actual GPU run commit: `ae8c5a2be16aa52181171019e02817af3938d3a6`.
- Post-run summary/accounting fix: `a5bd645810850fb21e202aae4907b523c405bf28`.
- Sanitized evidence commit: `3fa2f59bab9bddff8d8b28ab1df6661259c4f284`.
- Push mode: relay; worker has no remote. Push status pending for the local
  supervisor, which must pull these commits and push to the private origin.
