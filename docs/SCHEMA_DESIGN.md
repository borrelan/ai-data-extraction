# Event, episode, and action-window schema

Status: implemented compatibility schema and migration reference.

This document describes the `ai-data-extraction/v1` event, episode, and
action-window behavior that exists in this repository. It no longer selects
the future canonical owner: hardened AgentIR owns canonical offline records and
loss-aware projections. The builder described below remains frozen compatibility
behavior and a parity source for that migration.

The current `ai-data-extraction/v1` records are a useful compatibility format
for SFT, trajectories, tool traces, preferences, and prompt inventories. They
are not sufficient as the long-session canonical model because they treat a
whole provider record as one training example. The next schema adds lineage
and segmentation without changing or rewriting the historical export.

Builder 1.3.2 also stamps each normalized nested event with an
`ai-data-extraction/event/v1` schema, stable event ID, local ordinal, and parent
hash. Native provider event ordering is still an adapter-specific quality gate;
the compatibility builder reports normalized order rather than claiming a lossless
event lake.

The Codex ingress now reads its native event-per-line source in bounded passes
and emits one or more provider records per source session. Each bounded record
contains `_chunk_parent_record_sha256`, chunk ordinal/count, source event-line
range, and a `source_origin` file hash. The compatibility
`extract_codex_session()` helper still returns a full session for explicitly
bounded callers, but the normal CLI path does not retain or serialize a whole
long session. The ingress boundary is additive and provider-specific; the
compatibility builder remains the only trainer-data mapper in this repository.

The Codex ingress target is intentionally 25% of the final character safety
bound. The compatibility builder materializes both conversational messages and
event projections, so a larger raw segment can pass ingress and fail after
normalization. This conservative target is an envelope, not a token budget;
the final validator remains authoritative. Oversized tool observations are
reduced to head/tail evidence with an explicit marker, original-character
count, and SHA-256 in `observation_truncations`; the immutable source file is
the recovery authority. A native tool call that follows a non-assistant
message is normalized into an assistant action rather than attaching a call to
the user turn. If a call has no source observation, the segment carries
`_open_tool_call_ids` and a non-complete cut reason.

## Invariants

- A provider session is a parent container, never automatically a training
  example.
- Every derived unit points to a source file hash, source line, source class,
  extractor revision, and parent session hash.
- Every episode has an ordinal and a continuation link where a boundary cuts
  through an unfinished task.
- A cut or incomplete trace cannot acquire a success label merely because a
  later segment succeeds.
- Tool calls remain adjacent to their observations when the source provides
  that relationship. Detached events are marked as detached; order is not
  invented.
- A tool call is owned by an assistant action turn; provider event ordering
  must not turn a user message into a tool caller.
- Truncated observations remain usable only as explicitly tagged candidates;
  they are not silently treated as complete verifier evidence.
- Hidden reasoning fields and tagged reasoning blocks never cross the
  trainer-facing boundary. A short observable decision label is allowed.
- Rewards, preferences, terminal states, and verifier results are explicit
  source or harness facts. Missing values remain unknown/unscored.
- Privacy and license status are orthogonal to task quality. A high-quality
  private record is not training-eligible.
- Source-selection lanes are explicit: `primary`, `optional_alt`, or
  `quarantine`. A lane is provenance/mixture policy, not a quality label; the
  builder must report selected and skipped lanes in its manifest.
- Session quality is a separate provider-neutral assessment with one gate
  (`candidate`, `review_required`, `quarantine`, or `unassessed`), structured
  evidence dimensions, and zero implied model ranking. Model mixture policy is
  a separate `model_tier` dimension: `tier1_frontier` is the primary release
  tier, `tier2_open_source` is the secondary open-source tier, and
  `tier3_local` is the local/self-hosted tier. Missing or conflicting model
  identity is `unclassified` and cannot enter a Tier 1 build. A tier is not a
  correctness verdict; every tier still requires the same session quality,
  privacy, license, and tool/outcome gates. Quality overrides are keyed to a
  session ID or source-file hash and never change the source lane. Reviewed
  model-tier overrides use the same immutable identity rule and never invent
  model identity from a provider name alone.
- The compatibility builder preflights all input records by source-session
  identity before builder-side chunking. Every segment of one session inherits
  the same `session_quality_id` and gate; the manifest reports both segment
  counts and session counts so duplicated chunks cannot masquerade as new
  quality evidence.

## Three linked units

### Event (`ai-data-extraction/event/v1`)

An event is the smallest ordered observation from an adapter. It is useful for
forensics and re-segmentation, not necessarily a trainer row.

```json
{
  "schema_version": "ai-data-extraction/event/v1",
  "event_id": "sha256:...",
  "parent_session_sha256": "sha256:...",
  "ordinal": 1842,
  "source": {
    "provider": "codex",
    "source_class": "session_active",
    "source_file_sha256": "sha256:...",
    "source_line": 17,
    "native_event_type": "response_item",
    "extractor_version": "..."
  },
  "timestamp": "...",
  "role": "assistant",
  "kind": "tool_call",
  "payload": {
    "name": "shell",
    "call_id": "call-1",
    "arguments": {"command": "pytest"}
  },
  "privacy": {"status": "review"}
}
```

The event payload is sanitized before any export outside the private evidence
plane. Raw adapter output remains local and is not trainer input.

### Task episode (`ai-data-extraction/episode/v1`)

An episode is one objective plus the bounded observable interaction needed to
reach a terminal state or an explicitly bounded failure. This is the primary
unit for verified SFT and environment-backed RL prompts.

```json
{
  "schema_version": "ai-data-extraction/episode/v1",
  "example_id": "sha256:...",
  "parent_session_sha256": "sha256:...",
  "episode_id": "sha256:...",
  "ordinal": 12,
  "source_event_range": {"start": 1801, "end": 1912},
  "continuation": {
    "previous_episode_id": "sha256:...",
    "next_episode_id": null,
    "status": "complete",
    "cut_reason": null
  },
  "messages": [],
  "events": [],
  "tools": [],
  "terminal": {
    "status": "unknown",
    "evidence": [],
    "source": "unscored"
  },
  "quality": {
    "stage": "candidate",
    "tool_contract": "review",
    "verification": "absent",
    "token_count": 0,
    "cut_from_oversized_parent": false
  },
  "tags": ["task:debugging", "tool:shell"],
  "provenance": {
    "provider": "codex",
    "source_class": "session_active",
    "source_file_sha256": "sha256:...",
    "source_line": 17,
    "extractor_version": "...",
    "builder_version": "..."
  },
  "privacy": {"status": "review", "eligible_for_training": false}
}
```

`terminal.status` is intentionally separate from `quality.stage`. A record can
be a high-quality, privacy-approved failure demonstration without being a
successful task, and an apparently successful transcript can remain an
unverified candidate.

### Action window (`ai-data-extraction/action-window/v2`)

An action window is a compact supervised decision around one tool or harness
transition. It is the preferred unit for tool selection, argument validity,
skill/MCP gating, recovery, and no-tool decisions.

```json
{
  "schema_version": "ai-data-extraction/action-window/v2",
  "window_id": "sha256:...",
  "episode_id": "sha256:...",
  "context": {
    "messages": [],
    "state_hash": "sha256:...",
    "available_tools_revision": "sha256:...",
    "available_skills_revision": "sha256:..."
  },
  "decision": {
    "action": "use",
    "basis": "capability",
    "skill": {
      "name": "contract-enforcement",
      "triggered": true,
      "read": true,
      "decision": "use",
      "skip_reason": null,
      "revision_sha256": "sha256:..."
    }
  },
  "tool_call": {
    "name": "lsp.find_references",
    "call_id": "call-1",
    "arguments": {}
  },
  "observation": {
    "status": "unknown",
    "call_id": "call-1",
    "output": "..."
  },
  "evidence": {
    "schema_version": "ai-data-extraction/action-evidence/v1",
    "positive_target_status": "not_adjudicated",
    "action": {
      "signature": "sha256:...",
      "turn_signature": "sha256:...",
      "turn_ordinal": 3,
      "calls_in_turn": 1,
      "families": ["code-navigation"]
    },
    "observation": {
      "joined": true,
      "match": "call-id",
      "match_strength": "exact",
      "status": "unknown",
      "status_source": "absent",
      "result_code": null,
      "result_code_source": "absent",
      "output_digest": "sha256:...",
      "novel_for_same_action": null
    },
    "sequence": {
      "prior_turn_occurrences": 0,
      "nearest_prior_turn_distance": null,
      "immediate_repeat": false,
      "complete_cycle_periods": []
    },
    "artifacts": {
      "hashes_before_next_action": [],
      "content_included": false
    },
    "episode_outcome": {
      "value": "unknown",
      "source": "unscored",
      "step_credit": "absent"
    }
  },
  "verification": {
    "state_delta_hash": null,
    "artifact_hashes": [],
    "tests": [],
    "diagnostics": [],
    "source": "unscored"
  },
  "quality": {"stage": "candidate"},
  "provenance": {"episode_id": "sha256:..."},
  "privacy": {"status": "review", "eligible_for_training": false}
}
```

For a no-tool decision, `tool_call` is null and the harness records why no
tool was necessary. For an optional skill skip, `skip_reason` must be a
concrete negative-return category such as `unavailable`, `out_of_scope`, or
`measured_regression`; mandatory contract/privacy/safety gates cannot be
skipped by model text.

Canonical extraction emits tool-use candidates only. The `evidence` object is
derived from normalized event structure: it may prove a one-to-one observation
join, structured return code, output change, recurrence/cycle, or artifact
boundary, but it cannot prove that the action advanced the task. It never
parses output prose into a status and never promotes
`positive_target_status=not_adjudicated`. `verification` is reserved for the
execution harness or explicit adjudication layer.

## Segmentation algorithm contract

The implementation should follow these steps in order:

1. Normalize native events and preserve their native order, timestamp, and
   source range. Remove hidden reasoning only at the trainer-facing boundary.
2. Start an episode at an objective-bearing user request. Keep system/tool
   registry context attached by hash or reviewed compact copy.
3. Keep assistant tool calls, tool observations, artifacts, verification, and
   the next assistant/user response in the same episode while a task is open.
4. Close at an explicit terminal state, a clearly new objective, or a bounded
   failure. If the boundary is heuristic, write `status=unknown` and a
   `boundary_reason`; do not upgrade it to success.
5. Apply a tokenizer-aware episode budget chosen by the target trainer (initial
   pilot: 8k--32k tokens), while retaining a much larger source safety bound.
   Provider ingress should use a conservative pre-normalization envelope (the
   current Codex adapter uses 25% of the final character bound), split at the
   nearest safe user/tool boundary, and create continuation links.
6. Derive action windows from tool/skill/verification transitions. Windows may
   overlap context, but each target action has one parent event and one
   episode.
7. Validate lineage, role/tool adjacency, duplicate identities, token bounds,
   privacy state, and terminal-label provenance before writing a trainer file.

The existing 250,000-character limit becomes a final safety check for an
individual episode/window. It must not be evaluated on the unsplit parent
session. A single message or an indivisible assistant-tool exchange that is
itself over the bound is rejected with provenance; bounded action windows may
still be recovered from an oversized episode candidate and neighboring
episodes remain available. A bounded observation containing an omission must
carry its omission policy and original-content hash; it cannot be promoted to
verifier-backed evidence solely because the visible head/tail looks plausible.

## Dataset projections

The canonical event/episode store is the source for projections:

| Projection | Unit | Allowed labels |
| --- | --- | --- |
| SFT | complete episode or action window | visible assistant/tool behavior; no inferred success |
| Tool SFT | action window | typed call, observation, recovery, no-tool decision |
| Preference | paired episode/window | explicit or verifier-backed chosen/rejected only |
| RL prompt | episode start/state | prompt/context only; reward supplied by environment |
| Process supervision | action window | verifier-backed step labels with oracle provenance |
| Memory | reviewed fact/summary | retrieval metadata, never hidden reasoning |

Every projection carries the parent IDs and a manifest linking schema,
extractor, builder, privacy, registry, skill, environment, verifier, and model
revisions. This permits the same long session to support multiple bounded
learning units without pretending those units are independent conversations.

### Provider/advisor separation

Prime Agent and Pi-family adapters may observe harness, advisor, model-routing,
goal, and loop events. Those events are evidence for the harness audit and
quality gates, not assistant training messages. Ordinary sessions whose model
quality or advisor influence is not independently reviewed remain in the
`optional_alt` source lane, while their session quality is assessed separately
from dialogue/tool/outcome evidence. Advisor overlays such as `__advisor.jsonl`
carry an explicit contamination flag and default to `quarantine`; that rule is
about the overlay artifact, not about every Pi-family session. The compatibility
default build selects `primary` and preserves the other lanes for explicit,
separately manifested experiments.

### Session quality (`session-quality/v1`)

Each adapter that can inspect a complete session should emit a structure like:

```json
{
  "assessment_version": "session-quality/v1",
  "gate": "candidate",
  "flags": ["model_provenance_local_or_self_hosted", "outcome_unverified"],
  "model_tier": "tier3_local",
  "model_tier_basis": "adapter_declared",
  "dimensions": {
    "model_provenance": {"status": "local_or_self_hosted"},
    "dialogue": {"user_messages": 4, "assistant_messages": 4},
    "tool_trace_integrity": {
      "calls": 3, "observations": 3, "unmatched_calls": 0
    },
    "outcome_evidence": {"status": "unverified"}
  },
  "method": "deterministic_structural_session_v1",
  "provider_neutral": true
}
```

`candidate` means structurally usable for review, not verified success,
privacy approval, or RL eligibility. SFT, tool-trace, preference, and RL
projections apply their own stricter gates. The Prime/Pi adapter supports an
optional JSON override file keyed by `source_file_sha256` or `session_id`:

```json
{
  "schema_version": "ai-data-extraction/quality-overrides/v1",
  "overrides": [
    {"source_file_sha256": "sha256:...", "quality_gate": "candidate", "reviewer": "...", "reason": "..."}
  ]
}
```
