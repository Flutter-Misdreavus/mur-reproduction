# Supervisor review: mur-20261024-test100

Decision: **accept the r6 preparation evidence and corrected implementation for a bounded continuation; no test result exists.**
Authorize the separate frozen `mur-20261024-test100-r2` task, conditional on its
unchanged two-question development identity gate. The fixed100 sample must remain
byte-identical to the manifest committed before any model execution.

This worker implemented the test loader, fixed manifest, grader deduplication,
corrected trigger denominators, identity check and four-chunk driver in
`fba4a79e5ea57c944d5160a96e21fd60f6a2c682`. All54 remote CPU checks passed.
The supervisor stopped the unit after CPU preflight and before any model phase:
the resource record contains zero allocation and no phases; the model log directory
is empty; the unit has no live main process. The wrapper recorded CANCELLED because
there was no worker final report. Keep that event; do not call this attempt SUCCEEDED.

Review found the new driver reserved3300 seconds for generation alone, allowing
grading to exceed the stated combined chunk ceiling. It also recorded missing token
counters as zero, auto-selected an unvalidated GPU, and could dirty tracked output
between compute phases. These issues were fixed before the bounded continuation:

- Combined generation+grading cap3300seconds, grading reserve300seconds, per-phase
  cleanup reserve60seconds, all measured time charged including cleanup.
- Missing/invalid counters consume the full granted reservation and fail the phase;
  an unknown count is never presented as measured zero.
- Pin the previously validated physical GPU2; refuse HEAD or dirty-worktree changes
  before each phase. Delay tracked engineering evidence until all compute ends.
- Strict expected method/question pairs before grading; preserve cache-deduplicated
  grader costs separately from attributed per-answer token counts.
- Compute accuracy deltas from paired integer correctness differences, avoiding a
  floating-point rejection at exactly minus2 percentage points.

Three new CPU regression checks and nine existing test100 checks passed locally.
Full environment checks must run remotely before GPU execution. Original scientific
prompts, alpha/gamma,candidate count,sampling,model/data revisions and the100 selected
questions are unchanged. This stop consumed no GPU allocation; the next attempt
retains the original4GPU-hour ceiling and project12GPU-hour ceiling.

The r6 review and its limitations remain applicable. The100-question experiment is
exploratory; development accuracy preservation was not established. Do not claim
complete MATH-500 reproduction or tune against the selected test questions.

Version status: cancelled worker HEAD is the implementation SHA above; subsequent
supervisor corrections and this review are separate commits. Private relay push of
the correction commit is checked before dispatch. No model run SHA exists for this
cancelled implementation attempt.
