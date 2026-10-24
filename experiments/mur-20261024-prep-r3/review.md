# Supervisor review — mur-20261024-prep-r3

Decision: preserve the failed cross-engine gate; approve a separately frozen signal-semantics gate. No scientific outcome has been established.

GPU access, V1 loading, whole-stage accounting and process cleanup now work. The measured two-engine diagnostic used 11637 MiB of new memory. The five BF16 generation/Hugging Face teacher-forced comparisons failed two cases (maximum token NLL difference 0.4610, maximum mean difference 0.03109). No 20x3 smoke or test generation occurred. Do not describe this as numerical parity.

Worker diagnostics independently recomputed vLLM prompt logprobs on the saved exact token IDs; these reproduce the generation values, including outliers. HF attention/precision variations change the discrepancies. This supports an implementation/kernel difference; it does not prove that arbitrary engines are interchangeable.

The protocol asks whether m_t is raw model likelihood, not whether separate BF16 kernels agree to arbitrary absolute tolerances. The next gate compares generation with an identity-sampling teacher-forced path in a fresh instance of the same pinned engine, using exact token IDs and eight frozen synthetic prompts. The original tolerances stay unchanged; the HF failure remains a separate mandatory diagnostic. This explicitly changes the validation reference, not the algorithm, sampling parameters, metrics, test data or acceptance threshold. Extra diagnostic forwards count toward resource use and never enter main generation.

Reviewed run commits: `bc68244` (first load failed: missing setuptools), `282b8da` (KV allocation failed), `0a69204` (completed numerical comparison failed). Evidence fixes: `a38ca18`, `d574464`; reviewed HEAD `d574464ee27499c03530c6bbd6f3ff380a6f0c66`. Full-history privacy audit passed; commits are relayed to the private origin.

Conservative R3 GPU use: 280 seconds; 300 generated diagnostic tokens. Diagnostic timing is estimated, unlike the driver intervals. The next task is bounded to 3000 GPU seconds, one attempt, with no unsupervised follow-up diagnostics after a failed gate. Preparation remains below the combined 2 GPU-hour ceiling and the project below 12 GPU-hours.
