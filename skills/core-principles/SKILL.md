---
name: core-principles
description: Apply durable, architecture-first engineering guardrails to non-trivial software work. Use for implementation, refactors, routing changes, incidents, integration work, and review when a canonical owner, contract boundary, validation plan, or contribution scope must be proven before editing.
---

# Core Principles

Use this as a personal default. The active repository's `AGENTS.md`, contribution guide, nested instructions, and explicit user direction are authoritative and may add stricter requirements.

## Before changing code

1. Read the repository's root instructions, contribution guide, and applicable nested instructions.
2. State the narrow outcome, canonical owner, affected contract(s), and smallest validation surface.
3. Trace direct callers/consumers before changing a shared owner. Use semantic tools when available and structural search otherwise.
4. For external CLIs, prefer the repository-pinned or `mise`-managed invocation.
5. Preserve unrelated user changes. Do not reset, checkout, delete, or overwrite broad paths to make a change easier.

## Mandatory Execution Lineage Gate

For multi-step or agent-assisted work, this gate is mandatory before editing,
dispatching, resuming, or changing lanes. Reconcile the authoritative
objective and root order, authority revision, current checkpoint and dirty
state, actual predecessor commits/diffs, active workers/worktrees/sessions,
canonical owner, allowed scope, and integration gate. A report, receipt, idle
status, or stale plan cannot select the next lane. Missing or contradictory
lineage means audit-only hold. `execution-lineage-control` contains the
detailed ledger and assignment-token procedure; it supplements this rule and
is not permission to skip it.

## Implementation rules

- Extend the canonical funnel; do not add a parallel routing, retry, mapper, cache, or state path.
- Keep one owner for each state plane. Normalize external data once at the boundary, then pass typed values internally.
- Make the primary path correct before adding healing. A retry, fallback, cache refresh, or watchdog is not evidence of normal-path success.
- Treat a compensating repair as a stop signal: if it restores one invariant while regressing another, freeze implementation and reload/rebuild churn, preserve the evidence, and remap first divergence before another edit.
- Prefer a focused adapter at a real boundary over leaking an external library's types, lifecycle, or transport details through the application.
- Remove displaced helpers, wrappers, and dead transitions in the same cut. Do not leave a new abstraction beside the old owner without an explicit compatibility plan.
- Do not invent test-only facts, debug fields, or status strings. Prove the value belongs to the canonical owner and local type surface.

## Evidence and closure

- Validate the changed behavior, direct dependants, and the relevant boundary—not merely compilation.
- For an incident or regression, identify the failure plane (configuration/dependency/application) before proposing a code fix.
- Qualify analysis tools before using their output to decide a code cut. Bind metrics and baselines to source, build/runtime artifact, configuration, schema/model/index version, corpus, and method.
- Keep narrow direct traces distinct from broad derived analysis. If a graph, index, telemetry, or generated artifact is unhealthy or has lost required lifecycle state, fall back to raw source, Git, logs, and targeted tests; state the reduced confidence.
- Do not call a capability semantic, learned, calibrated, or active in production without proving its actual execution path and relevant artifact/runtime identity.
- When an external operational surface recurs, replace ad-hoc commands with one repository-owned harness or client once the correct path is established.
- Treat another agent's completion report as a claim bundle, not proof. Re-read the exact diff, touched owner files, artifact, focused validations, and process/runtime identity before relying on it.
- Keep product repair, certification/provenance repair, and attached-process
  mismatch investigation as separate milestones. A red external product probe
  does not authorize a CI/harness rewrite unless that rewrite is the recorded
  prerequisite to reproducing the same invariant on the exact candidate; a
  harness green can never close the product red by itself.
- For runtime, daemon, queue, watcher, index, or release work, verify the
  operational basics separately from feature behavior: cold start/attach,
  discovery/readiness publication, single-owner/process topology, warm-path
  response, failure/takeover, and cleanup. A warm query green or feature-local
  green does not close a startup or ownership regression.
- Record any compatibility impact as additive, backward-compatible, or breaking; explain the migration path for breaking changes.
- Before handoff, review the final diff adversarially for duplicate paths,
  silent fallbacks, unowned state, missing cleanup, untested failure modes,
  code smells, standards drift, and security posture regressions.
- In a multi-step task, keep a concise live plan and retire or explicitly carry forward each obligation before switching lanes.

## Agent review and prompt handoff

- When handing work to another agent, pass only verified facts. Do not restate
  earlier summaries as if they were proven.
- Before composing a next prompt, complete the execution-lineage preflight:
  verify the active objective and authority revision, current baseline and
  dirt, actual predecessor changes, active session/worktree roster, and the
  gate that makes the next lane eligible. Do not derive a new assignment from
  the latest report or receipt alone.
- The next goal/autonomous prompt pair must encode the verified first
  divergence or the exact clean boundary, the canonical owner family/files,
  rejected hypotheses, the proof surface, non-goals, stop condition, and the
  concrete deliverable.
- If any basic operational invariant remains unexercised on the real attached
  path, classify the result as partial or blocked rather than promoted.

## Companion skills

Use `execution-lineage-control` for the detailed ledger/token procedure;
`architecture-first-remediation` for repeated repair churn or split ownership,
`boundary-evidence-workflow` for meaningful code cuts, and
`systematic-debugging` for failures or flaky behavior.

## Independent certification gate

For an architecture, persistence, protocol, scheduler, indexing, or agent-tool
change, treat unit tests and locally curated fixtures as owner-local checks,
never product certification. Before a rebuild, reload, release, or claim that
the change works:

1. Bind the candidate binary, executable-adjacent signed artifact,
   configuration, schema/spec/model, and source revision to one disposable
   production-equivalent process. Never substitute an ambient local store.
2. Exercise the externally visible protocol against the original failing
   third-party corpus and an independent pinned real corpus. Compare the
   claimed result with a source/Git/runtime oracle outside the implementation.
3. Include one adversarial negative: prove that a nearby-but-wrong symbol,
   path, sink, state, or concurrency schedule is rejected rather than merely
   proving a happy-path result.
4. For a long-running operation, prove an independent short operation remains
   bounded while it runs, using a warm short-operation envelope rather than a
   broad fixed timeout. A final successful response does not prove liveness.
5. Only after that pre-cert passes, run the same sequence through the attached
   product integration. If the two differ, freeze source changes and compare
   process identity, startup configuration, database/schema state, and corpus
   state before changing logic again.

Do not substitute a bespoke fixture, a cargo test, compilation, an internal
helper call, output count, or a restart for any step above. Carry any missing
real-corpus lane as an explicit release blocker with its owner and next proof.
The reusable production runner is the only protocol owner: its versioned public
operation registry must equal the runtime operation list, record every method
actually exercised, and make unexercised methods explicit release blockers.
Every changed public surface must be checked against an independent oracle
rather than split across overlapping partial test paths.

## Priority and deficiency ledger

The user-stated current objective is the priority source of truth. Every new
deficiency must be recorded, but recording it does not authorize work on it.

Classify each finding before any edit as exactly one of:

- **Current blocker:** proved to prevent progress on the active user-visible
  invariant or its smallest owner-local fix.
- **Required follow-up:** architecturally important and mandatory after the
  current objective, but not a proved dependency of it.
- **Release/completeness gate:** required before a release or broader support
  claim, but not before the active objective unless the user says so.

A follow-up may preempt the active objective only when continuity records the
specific blocked invariant, direct dependency evidence, smallest prerequisite
cut, and the proof that returns work to the original objective. “This would be
better architecture,” “the harness is incomplete,” or a newly observed red is
not enough. Otherwise add it to the durable deficiency ledger with owner,
evidence, revisit trigger, and validation order, then resume the current
milestone immediately. Never turn a backlog item into an open-ended rewrite
while the active objective remains blocked.
