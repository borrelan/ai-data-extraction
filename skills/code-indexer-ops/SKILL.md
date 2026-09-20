---
name: code-indexer-ops
description: Use Code Indexer MCP or CLI evidence safely — staged capability disclosure, the full public MCP surface, graph analysis, attempt-memory repair steering, recovery, diagnostics, and index maintenance. Trigger when the task names Code Indexer or depends on its search, graph, diff, quality, semantic, attempt-history, recovery, or index-status output.
---

# Code Indexer Operations

## Purpose

One neutral, host-agnostic workflow for using Code Indexer as evidence without
over-trusting its output, aligned with the staged-capability contract the
binary actually enforces and the checked-in public tool registry.

## When to use

- The task names Code Indexer directly, or depends on its search/graph/diff/
  quality/semantic/attempt results.
- You are about to repeat an edit that may have been tried before.
- You are diagnosing Code Indexer itself (index health, locks, storage).

## Guardrails

- Code Indexer is an evidence source, not automatic authority. Record the
  root, binary identity, and exact operation before relying on an answer.
- The shipped public tool source of truth is
  `crates/mcp/tests/public_tool_contracts.json`. If one attached client shows a
  smaller namespace than this skill documents, debug client/runtime drift; do
  not delete tools from the shipped skill to match one broken session.
- Read `capability_stage` before choosing tools (see Workflow); never guess
  readiness from non-empty output.
- Tools answer with a typed unavailable envelope when their stage is not
  reached — treat that as honest state, not an error to hammer.
- If locked or degraded, fall back to raw source reads for the immediate
  question; do not re-index or mutate a worktree merely to force an answer.

## Staged capability contract

Every status answer carries `capability_stage`. Tool families unlock by stage:

| stage | unlocked | behavior below stage |
|---|---|---|
| `bootstrapping` | status/index lifecycle only | others: typed unavailable |
| `structural_ready` | search (lexical), search_text, symbol_search, get_symbol | graph fields locked |
| `graph_ready` | impact, trace, flow, find_references, graph_query | typed unavailable envelope |
| `semantic_partial` | vector signals + coverage disclosure | semantic field shows `locked` |
| `semantic_ready` | full semantic + rerank | — |

`search` always shows a `semantic` field: `locked` (no coverage) or real
availability — never an availability claim without embeddings.

## Public MCP surface

The public MCP binary exposes these tool families:

- Lifecycle: `init`, `use`, `index`, `status`, `backfill`
- DEFAULT discovery (daemon-era): `locate_code` (MCP) and `code-indexer
  locate` (CLI) — the same shared funnel: one call, symbol + text hits
  merged, typed envelopes (`unpublished` / `no_match` with indexed counts
  instead of silent empties; internal failures are errors, never masked as
  misses). Limits are clamped to 1..=200 and the effective pair is
  disclosed. Use it FIRST for any "find X".
- Discovery and retrieval: `search`, `lexical_search`, `symbol_search`,
  `search_text`, `get_symbol`, `context`
- Graph and navigation: `graph_query`, `impact`, `find_references`, `trace`,
  `flow`, `list_flows`
- Cross-boundary contract resolution: `resolve_xbound` (turns a
  `declared-xbound:<boundary>|<service>` token from `impact` output into every
  producer across the published view's member worktrees, with file:line
  provenance)
- Host policy: `policy_evaluate`
- Runtime evidence (additive L4): `runtime_evidence` (ingest/query observed
  HTTP/channel activity; binds to declared tokens when they exist; never
  exact static truth) (typed evidence + validator gates for
  code answers / mutations / commits / merges; explainable decisions, never
  runs validators)
- Event/channel resolution: `resolve_channel` (turns a `channel:<name>` token
  into every producer (EMITS) and listener (HANDLES_EVENT) across the
  published view, with file:line provenance; string-literal names only)
- Change and mutation: `detect_changes`, `rename_symbol`, `edit_symbol`
- Quality and security: `quality`, `explain`, `security_taint`
- Recovery: `history_status`, `history_list`, `history_diff`,
  `restore_preview`, `restore_apply`
- Diagnostics: `diagnostics_profile`, `diagnostics_query_plan`,
  `diagnostics_maintenance`
- Attempt memory: `attempt_status`, `attempt_history`, `attempt_explain`,
  `attempt_alternatives`, `attempt_preflight`, `attempt_ancestor`,
  `attempt_family`

The public CLI that ships with the same release also exposes operational
surfaces such as `code-indexer maintenance` and `code-indexer attempt`, but the
tool names above are the MCP contract this skill is keyed to.

## Workflow

1. Pin and stage-check: `use` (or pass root), then `status`. Read
   `capability_stage` and the backfill note; work only with unlocked tools.
2. Discover symbol-first: `symbol_search` / `search_text` at
   structural_ready; confirm key definitions in source before drawing
   conclusions. Use `context` for a bounded window; `search` for ranked
   lexical hits.
3. Graph work at graph_ready: `impact` for blast radius, `trace`/`flow` for
   paths, `find_references` for callers, `graph_query` for survey work,
   `quality` / `explain` / `security_taint` for file security posture. A
   typed-unavailable envelope means wait or fall back — never treat as empty
   truth.
4. Repair loops: consult attempt memory BEFORE repeating an edit —
   `attempt_preflight` (repeat risk) before mutating; `attempt_history` /
   `attempt_family` for what was tried; `attempt_explain` for why a candidate
   looks repeated; `attempt_alternatives` for untried owners;
   `attempt_ancestor` for the strongest prior state. Verdicts come only from
   validator evidence (`attempt validate` CLI / hook-observed test runs) —
   never from model text.
5. Changes: `detect_changes` maps the git diff to symbols; `impact` with
   `attempts=true` adds owner-family steering (prior verdicts, do-not-repeat).
   Use `rename_symbol` / `edit_symbol` instead of text rewrites when the
   mutation is symbol-owned.
6. Recovery: `history_list`/`history_diff` before restoring;
   `restore_preview` then `restore_apply` (hash-confirmed) — never raw git
   checkout for recovery. To undo a bad managed cut (`edit_symbol`,
   `rename_symbol`, `restore_apply`): every managed mutation returns
   `recovery.before_snapshot_id` (and `after_snapshot_id`); restore the
   affected file from the before snapshot
   (`restore_preview(before_snapshot_id, <file>)` → `restore_apply`). To redo,
   run the same restore against `after_snapshot_id`. There is no separate
   undo/redo tool by design — the restore workflow is the single recovery
   funnel.
7. Maintenance: `code-indexer maintenance --path <root> --budget-secs N`
   (bounded abandoned-set reclaim + WAL exit); `backfill` to expedite
   embeddings. Diagnose with `diagnostics_profile` /
   `diagnostics_query_plan`, and use `diagnostics_maintenance` only as an
   explicit confirmation-gated maintenance action.

## Disposable-resource hygiene (disk is a proof surface)

The host root filesystem runs near-full in normal operation (observed
756G/720G, 3.3G free). Disposable proofs (temp repos, spawned daemons,
staged binaries, project DBs created under `~/.config/code-indexer/projects/`)
must be treated like open file handles: bounded at creation, enumerated at
close, and provably gone.

- Before starting a live-proof lane: check `df -h /`. If free space is
  under ~10G, prune stale build artifacts first
  (`find target/debug/deps -maxdepth 1 -type f -mmin +720 -delete`);
  do not start a proof you cannot afford to finish.
- Every disposable proof names its residue up front: temp roots under
  `/tmp/ci-<lane>/`, project dirs under
  `~/.config/code-indexer/projects/<id>/`, spawned daemons by PID.
- Cleanup is part of the certification regime, not a courtesy:
  - kill ONLY daemons you spawned, by explicit PID from the project's
    `daemon.json` (never `pkill`/`killall` — user processes must survive);
  - delete ONLY project dirs whose `watched_roots` map to your temp roots;
  - remove your `/tmp/ci-*` scratch trees.
- Close a lane by proving the cleanup: re-list spawned PIDs (must be gone),
  re-resolve your project dirs (must not exist), and state the `df -h /`
  delta in the certification artifact.
- Known residue to watch for: stub project dirs keep a bare `history.git`
  (+`history.lock`) with no `index.sqlite` — one per indexed root ever,
  ~9.2G across 4,021 stubs at last survey. If your lane creates many
  throwaway roots, either reuse ONE temp root per lane or record the stub
  growth explicitly.
- A warm-success proof is not a lifecycle proof; a green suite is not a
  cleanup proof. Both must be shown separately.

## Definition of done

- The answer states the evidence grade and the `capability_stage` it relied on.
- Graph conclusions are backed by a graph_ready-or-later stage.
- Repeated-edit decisions cite attempt memory, not memory of the conversation.
- The primary requested path is proven with source or tests, not inferred
  from adjacent output.
- For live-proof lanes: spawned daemons are dead (by explicit PID), temp
  roots and test-created project dirs are removed, and the artifact records
  the disk state before/after.
