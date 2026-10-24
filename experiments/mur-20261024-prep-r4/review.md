# Supervisor review — mur-20261024-prep-r4

Decision: retain the failed independent-prefill comparison. It is not evidence of a scientific accuracy result or proof of transformed logprobs.

All eight probes were run once, at commit d2e0408165055f975504b1c5b509c379dad3b7c7. Four comparisons failed; the model generated 243 tokens and allocated 62.77 GPU seconds including cleanup. No smoke or test generation followed. Report commit e3e7959 records this honestly; the privacy audit passed. The worker's added post-run checks do not retroactively alter the run commit.

The supervisor's previous validation design conflated the identity of the probability signal with numerical parity across distinct BF16 model forwards. Generation and fresh prefill use different shapes/cache paths and can differ. Instead of increasing the tolerances or dropping probes, use a read-only observer inside the exact generating sampler: clone untouched logits, independently compute float32 log_softmax, then compare selected-token probabilities with the sampler output and the public Generation response. This directly tests the property MUR needs. Both prior parity failures remain failures.

The observer consumes no random numbers, does not modify logits/metadata, calls the original sampler exactly once, and exists only in the validation process. The eight fixed probes and scientific configuration remain unchanged. New numerical bound 5e-5 applies only to float32 probability arithmetic on identical logits; it is not a relaxed BF16-forward bound. A temperature-only negative control must disagree by >0.001 on at least one token. Exact token alignment and every probe must pass. No further reference changes or numerical variants in the new task.

For conservative budgeting, charge the entire original R1 allocation (3600 GPU seconds), R2 zero, R3 280, R4 62.77 and R5 at most3000: total6942.77 seconds, below the combined7200-second prep ceiling. This is a budget charge, not a claim of measured R1 usage. Actual metrics must distinguish measurements from reservations.
