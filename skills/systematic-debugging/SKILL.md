---
name: systematic-debugging
description: Use when encountering any bug, test failure, flaky behavior, or unexpected runtime state. Apply a structured, evidence-first workflow before proposing fixes.
---

# Systematic Debugging

## Purpose
Prevent speculative fixes by forcing reproduction, evidence gathering, and owner-path analysis before editing.

## When to use
- bugs and regressions
- flaky behavior and race conditions
- performance regressions
- contract or typing violations
- failures that appear to move between lanes after local fixes

## Workflow
1. Classify the failure plane first.
   - infra/dependency/config
   - contract drift / harness mismatch
   - application logic
2. Reproduce the problem with the smallest reliable lane.
   - If another agent already claims an RCA or fix, reproduce and verify the
     claim from source/runtime first; do not inherit its conclusion.
3. Gather evidence:
   - logs
   - traces
   - timestamps
   - request/socket ids
   - runtime state snapshots
4. Quantify repeated signatures instead of reading only isolated errors.
   - reconnect loops
   - snapshot churn
   - membership churn
   - retry storms
5. Separate operational basics from feature behavior.
   - cold start / attach
   - discovery / readiness publication
   - steady-state short operation
   - concurrent long-running work
   - single-owner/process count
   - stale/dead owner recovery
6. Control variables and isolate the failure point.
7. Form a hypothesis from evidence.
8. Test the hypothesis before editing.
9. Consider architecture, not just local symptoms.

## Required pre-edit gate
Before editing during a debugging task, record:
- failing contract/invariant
- failure plane classification
- suspected owner funnel
- primary path vs repair path
- affected systems/modules
- smallest reproducer / validation lane
- backup checkpoint path

If evidence changes the owner or affected systems, reopen the gate before continuing.

If the debugging result could change the active milestone or trigger another
agent assignment, also load `execution-lineage-control` and reconcile the
authoritative objective, current checkpoint, prior actual changes, active
workers, and integration gate. A newly discovered symptom is not permission
to pivot; classify it and either prove the smallest dependency that unblocks
the current objective or ledger it for its named revisit lane.

## Done criteria
- root cause identified with evidence
- reproduction steps captured
- architectural fix options identified
- contract/typing impact assessed
- failure plane recorded with proof
- any unverified startup/ownership/liveness invariant called out explicitly
