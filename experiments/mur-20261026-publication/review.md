# Publication preparation — mur-20261026-publication

The user explicitly requested public visibility and a redesigned README on
2026-10-26. This is a documentation/publication stage, not a new model experiment.
Historical private-visibility fields in frozen tasks remain records of their
original execution conditions.

## Changes

- Chinese README with a vector banner, navigation, evidence-derived comparison
  figure, method flowchart, equations, reproducible CPU commands and source links.
- Automatic and GPT-reviewed results remain separate; incomplete fixed100 status,
  negative accuracy result, missingness and computation-proxy limitations remain
  prominent. No model outputs or benchmark text were added.
- The comparison figure reads the committed supervisor statistics directly.
  Its generator is retained, and SVG/PNG outputs were visually inspected.
- Corrected double-escaped whitespace/dot expressions in the publication auditor
  that could miss credential assignments and IPv4 addresses. Added quoted JSON-key
  support and constructed synthetic regression cases without literal credentials
  or addresses in fixtures. The scanner prints finding categories, never values.

## Verification

- The corrected scanner checked all 210 pre-publication historical file versions
  with zero findings. The final staged tree and complete history are checked again
  before visibility changes. Pattern scanning is a check, not an exhaustive proof.
- Four publication-auditor regression tests and five pure controller tests pass.
- Standalone statistical recomputation exactly matches the committed supervisor
  statistics; inference code, scientific configuration and numeric evidence are
  unchanged.
- README local link targets, explicit anchors, details blocks, code-fence balance
  and SVG XML were checked. The README was also rendered through GitHub Markdown.
- The remote repository has one branch and no tags, releases, issues, Actions runs
  or Actions artifacts at preparation time. Public-facing Git author attribution
  is retained. No commit history is rewritten.
- Data, weights, environment directories, credentials, raw traces and raw DSH
  sessions remain excluded. Plotting dependencies were installed outside the
  repository; model inference was not run.

## Version and delivery record

- Actual test execution: `5c91a8a178e2c39ba6aa50c3d116524eae8ecf6e`.
- Post-run accounting fix/results: `e3e12dd06dc9ca64b29c086c52248a13e1b52eb8`.
- Independent review and pre-publication HEAD:
  `f1d87df26c1283ae4f33eab05e136aa4d7612f60`.
- Branch: `exp/mur-20261024`; repository: `Flutter-Misdreavus/mur-reproduction`.

This publication commit is pushed while the repository is still verified private;
visibility is changed only after that push and the final audit. The resulting HEAD,
push status and server-confirmed visibility belong in the ignored local delivery
receipt and final handoff. This file records preparation and does not itself assert
that the remote visibility mutation has already succeeded.
