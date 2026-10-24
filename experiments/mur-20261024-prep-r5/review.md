# Supervisor review — mur-20261024-prep-r5

Decision: preserve BLOCKED; fix observer alignment without changing the direct numerical gate.

Run commit e931a6d completed five probes with exactly zero raw/sampler/client logprob error; the temperature-only negative control differed by as much as0.26924. Probe6 stopped at a client/sampler token-count mismatch. Eight-probe validation and smoke did not complete. Usage:32.06 GPU seconds,172 client tokens. Evidence commits0b879fa and65bffee retain the failure; final reviewedHEAD65bffee.

Installed vLLM0.8.5 output_processor.py lines343-373 detect a text stop in the client and request engine cancellation; lines327-330 discard outputs for already-aborted requests. The engine can therefore sample a trailing token before cancellation arrives. This is a specific hypothesis for the count mismatch, not yet proven by R5 because the failing token vectors were not saved.

The fix persists complete failure evidence and compares every returned token against the exact leading sampler sequence. Extra sampler entries are permitted ONLY when the API explicitly reports finish_reason=stop and stop_reason=the configured newline-Step marker. Every extra entry is retained and its raw-versus-sampler arithmetic is also validated. Mismatched prefix, short capture or unexplained tail fails. Threshold5e-5, eight probes, sampling and negative control are unchanged. This tests the same direct contract; no numerical reference is replaced.

Validation resource accounting includes both returned and trailing sampled tokens. Scientific output-token metrics remain the paper-style API-returned policy+critic output counts; they are a compute proxy, not an exact FLOP counter. Client cancellation and engine work can differ, so report this limitation and retain measured GPU intervals separately. No scientific method or test item has been selected using these checks.

Conservative prep charge: R13600, R20, R3280, R462.77, R532.06, R6<=3000 GPU seconds =6974.83 seconds, below7200. R1 is a conservative allocation charge, not measured use. The current maximum remains within the original combined preparation ceiling.
