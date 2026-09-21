# Harness trace contract

Status: implemented contract, 2026-09-17

This recorder is the compatibility owner for the local observable-policy
events described below. It is not the online-RL trace store. Fresh RL execution
uses Agent Lightning v1.0.1 rollouts and attempt-scoped append-only
model-request/reward/custom events. OpenTelemetry separately carries runtime
instrumentation, and OpenObserve receives only an allowlisted,
privacy-filtered operational mirror; neither OpenObserve nor this recorder may
invent rewards, policy tokens, or environment transitions.

The executable single-funnel recorder is [`harness_trace.py`](../harness_trace.py).
It is a provider-neutral library used by a live harness or a replay adapter;
it does not execute tools and it does not accept hidden reasoning as evidence.
The recorder validates event shapes, requires a registry and structured
decision before a tool call, requires permission for side-effecting calls,
bounds observations while retaining their original hash, and records explicit
unmatched/loop-stop states.

`record_tool_registry()` is the runtime registry owner. It accepts either
`schema` or MCP-style `inputSchema`, requires an object schema, rejects
duplicate names, sorts definitions for deterministic identity, and records a
`registry_sha256` over the exact revision and exposed schemas. A provider
transcript that contains only a tool name cannot recreate this event safely.

The model can learn to prefer a skill or tool, but the harness must own the
truth that a skill was read, an MCP server was healthy, a permission was
granted, and a side effect occurred. These events are therefore separate from
provider transcripts and may be joined to an episode by `episode_id`.

Adapters must pass the skills that are relevant to the episode as
`required_skills`. A `use` decision for a tool is rejected until every listed
skill has a successful, in-scope `skill_preflight` event. The default is an
empty tuple for backward-compatible replay callers; production adapters must
declare their required set explicitly.

## Required event types

Each event has `schema_version`, `event_id`, `episode_id`, `ordinal`, exact
registry/skill/environment revisions, a privacy state, and a source. Payloads
must contain observable facts only.

| Event | Required evidence |
| --- | --- |
| `skill_preflight` | skill name/revision, mandatory vs optional, trigger, read result, scope decision |
| `tool_registry` | registry revision and digest, exposed schemas, trust/side-effect class |
| `mcp_health` | server/revision, health result, capabilities, timeout/error |
| `decision` | `use`, `skip`, `defer`, or `ask`; short basis; required-gate status |
| `tool_call` | typed name, call ID, JSON arguments, permission decision |
| `tool_observation` | call ID, status, bounded output, truncation/error state |
| `state_delta` | artifact/state/test/diagnostic hashes before and after execution |
| `verification` | verifier revision, checks, result, durable evidence |
| `terminal` | success, failure, partial, unknown, or bounded-stop with evidence |
| `loop_guard` | repeated signature/no-progress hash, limit, action taken |

## Skill decision rules

1. Mandatory contract, privacy, and safety skills cannot be skipped by model
   text or user insistence.
2. An optional skill may be skipped only with a harness-recorded reason such as
   `unavailable`, `out_of_scope`, or `measured_regression`.
3. A retry after user insistence is a new decision event; it does not erase the
   original skip or its negative return.
4. A model statement that it read a skill is not evidence. The harness records
   the resolved file hash and read operation.

## Training projections

Join harness events to the sanitized provider episode only when the episode,
tool registry, skill revision, environment, and verifier revisions all match.
Then derive:

- SFT examples for observable decisions and verified actions;
- action windows for tool selection, argument construction, recovery, and
  no-tool/ask decisions;
- preferences for independently verified chosen/rejected paths;
- process labels from verifier-backed transitions;
- RL prompts and rewards only from resettable environments.

Missing harness evidence remains `not_observed`. It is never filled from a
provider prompt, a final answer, a tool name, or a judge model's guess.

The recorder's `HarnessTrace.record_skill_preflight` method enforces the skill
rule at capture time, and `record_decision` enforces the declared
`required_skills` gate before a tool-use decision. A mandatory skill that is
not read is recorded as a failed preflight and blocks the call path. Optional
skips require one of the explicit negative-return reasons (`unavailable`,
`out_of_scope`, `measured_regression`, or `not_triggered`). `record_tool_registry`,
`record_decision`, `record_tool_call`, `record_tool_observation`,
`record_state_delta`, `record_verification`, and `record_terminal` are the
corresponding observable event funnels.

Example setup (the revisions and hashes must come from the actual harness
runtime):

```python
from harness_trace import HarnessTrace

trace = HarnessTrace(
    episode_id="episode-123",
    source={"agent": "agent", "provider": "provider"},
    registry_revision="registry-sha",
    environment_revision="environment-sha",
    verifier_revision="verifier-sha",
    privacy_state="heuristic",
    required_skills=("core-principles", "contract-enforcement"),
)
trace.record_skill_preflight(
    skill="core-principles",
    skill_revision="skill-sha",
    mandatory=True,
    trigger="implementation",
    read_result="read",
    scope_decision="in_scope",
    content_sha256="sha256:<64 hex characters>",
)
```

The resulting JSONL is a harness evidence stream, not automatically trainer
input. It can be joined to a sanitized episode only when episode, registry,
skill, environment, and verifier revisions match.

## Closure requirements

The recorder permits a successful terminal event only when every recorded tool
call has exactly one observation and the latest verification event is a
revision-matched pass with non-empty checks and durable evidence. A successful
verification must occur after the last tool call. A registry is
single-assignment for an episode; a second registry event is rejected even if
the revision string is unchanged. Non-terminal traces cannot be written.

`harness_tool_projection.py` is the consumer boundary for a captured tool
episode. It accepts one sanitized canonical `tool_trace` row and one harness
trace, then emits review-only `tool_sft.jsonl` only when event identities,
call/observation IDs and arguments, registry schemas/digest, required skill
reads, verifier closure, and privacy/reasoning firewalls all agree. The
projection keeps registry, skill, environment, verifier, terminal, and source
hashes in a lineage sidecar and never exports rewards. A provider transcript
without the matching harness trace remains quarantine evidence.

## Fixture replay

Use [`harness_fixture.py`](../harness_fixture.py) to exercise the contract
against one canonical row without copying its prompt, response, or reasoning
content:

```bash
python3 harness_fixture.py /path/to/sft.jsonl \
  --output-dir .tmp/harness-fixture --overwrite
```

The fixture emits one successful tool episode and one repeated-signature
bounded-stop episode. It is a contract check, not a training corpus or proof
that a live MCP/indexer runtime is healthy.

## Loop and side-effect controls

The harness refuses or terminates a run when it reaches configured call,
turn, wall-time, output-byte, retry, repeated-signature, or no-progress limits.
Destructive calls require a permission event and a verifier-visible state
transition. A timeout, permission denial, stale index, unavailable MCP server,
malformed schema, or verifier disagreement is a labeled transition—not a
successful demonstration.
