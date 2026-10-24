# MUR reproduction

Follow the supervisor-frozen experiment task and PROTOCOL.md. Execute model work remotely only.
Use the research-orchestrator workflow supplied separately in ignored .orchestrator/.
Do not copy infrastructure skills, private hostnames, IPs, usernames, absolute home/data
paths, passwords, tokens, authentication screenshots, raw sessions, or runtime launchers
into tracked files. Paths in public configurations must be relative or environment-based.
Use environment variables for local model/cache locations. Never print credentials.

Keep this repository private until the user explicitly requests publication.
Before any inference run or rerun, commit implementation and configuration.
Use explicit staging paths, inspect staged diffs, and run scripts/audit_public.py.
Commit meaningful progress at stage completion or each 15 minutes of active editing.
Commit messages include run_id, stage and purpose. Record actual run commit separately
from subsequent fixes and report commits. Push via the supervisor's relay only.
Do not amend/rebase/force-push history. Do not stage raw datasets, weights, traces or logs.
Sanitized environment summaries, manifests, per-question numerical metrics, and aggregate
results belong in experiments/<run_id>/. Never include question/answer text in these summaries.
Record all failures and truncated generations. Do not tune on MATH-500.
