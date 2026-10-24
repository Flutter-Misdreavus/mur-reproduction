# Supervisor review — mur-20261024-prep-r2

Decision: accept CPU preparation evidence; GPU gate remains BLOCKED.

The worker ran 20 CPU tests and verified pinned weights, tokenizer/config files and the deterministic data manifests. Device access was denied by the DSH Landlock command sandbox (not proven to be a cgroup device rule). Zero model generations and zero GPU allocation are reported for this attempt. Do not interpret empty metrics as accuracy or token savings.

Implementation commit: `72f2bd8089b8531dc06edc547a52e4be1566ca14`. Evidence commit and reviewed HEAD: `e7f9e9a1a64ebcba2006c46259c04e21e21795d2`. These commits passed the full-history privacy audit and were pushed to the verified private origin.

Remaining work: multi-probe raw-logprob validation, whole-stage budget enforcement including the verifier, model subprocess cleanup, actual memory monitoring, complete 20x3 development smoke, and strict verdict review. The old 14 incomplete trajectories remain discarded diagnostics.

Next task resumes the unused repair budget after a task-local official DSH permission configuration. No global profiles are modified. The bounded parent service retains control-group termination. One GPU-hour for this resumption; total preparation <=2 GPU-hours and project <=12 GPU-hours remain unchanged. No full test run is authorized until the gate passes supervisor review.
