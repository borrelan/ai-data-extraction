# Canonical training-data contract

This repository has two data planes:

```text
local provider stores -> raw JSONL -> optional Privacy Filter -> canonical builder
                                                        -> SFT
                                                        -> explicit preferences
                                                        -> trajectories
                                                        -> prompt-only RL input
                                                        -> provenance-only rejections
```

Raw JSONL is private evidence. It is intentionally allowed to contain provider
fields that are useful for forensic recovery, including reasoning, paths, IDs,
tool payloads, and incomplete turns. It must not be sent to a trainer or a
remote skill-synthesis endpoint.

Provider ingress may emit more than one bounded JSONL record for a single
source session. The record's `source_origin` and private chunk fields preserve
the original file hash, source event-line range, and ordered segment lineage;
they are not independent conversations and must not be upweighted as such.

Each normalized example also carries a provider-neutral `session-quality/v1`
assessment. `candidate` means structurally usable for review; it is not a
claim of verified success, privacy approval, or model superiority. The builder
selects `candidate` sessions by default. `review_required`, `quarantine`, and
`unassessed` sessions remain visible in manifests and audit output but require
an explicit `--quality-gates` selection. This is independent of
`training_lane`. Model origin is represented separately as `model_tier`:
`tier1_frontier` is the primary release tier, `tier2_open_source` is the
secondary open-source tier, and `tier3_local` is the local/self-hosted tier.
Rows without authoritative model/deployment evidence are `unclassified` and
are excluded from the CLI's default Tier 1 build. A tier is a mixture policy,
not a correctness verdict; all tiers still require the same session-quality,
privacy, license, and tool/outcome gates.
The builder performs this assessment in a streaming preflight over every input
file before it chunks oversized records. All segments sharing a source-session
identity inherit one `session_quality_id` and one gate; a weak segment boundary
cannot accidentally turn the rest of a good session into a provider-wide
rejection, and a good-looking segment cannot hide a session-level integrity
failure. The preflight runs across selected and unselected provenance lanes.
The canonical builder also accepts the same provider-neutral override format
with `--quality-overrides`; an override changes only the quality gate and is
recorded with its automatic gate and reviewer metadata. It cannot promote a
quarantined provenance lane into the default lane or bypass privacy approval.

The Codex native ingress uses a 25% pre-normalization envelope of the final
250,000-character safety bound because normalization emits both messages and
event projections. Large tool observations are kept as bounded head/tail
evidence and tagged in `metadata.observation_truncations` with their original
size and hash. These rows may support review or tool-shape learning, but a
verified SFT/RL release must filter or separately grade them; the source file
remains the only authority for the omitted payload.

## Build sequence

The standard-library-only path creates review-gated output:

```bash
python3 build_training_data.py extracted_data --output-dir training_data
```

The CLI defaults to `--model-tiers tier1_frontier` for the primary release.
Use an explicit audit mix when measuring lower-tier coverage:

```bash
python3 build_training_data.py extracted_data \
  --output-dir training_audit \
  --model-tiers tier1_frontier,tier2_open_source,tier3_local,unclassified
```

Legacy rows that lack model metadata must be assigned through a reviewed
`--model-tier-overrides` manifest keyed by source-file hash or session ID; a
provider name alone is not sufficient evidence for Tier 1.

The builder treats a provider session as a parent container. By default,
`--oversize-strategy chunk` creates bounded segments rather than discarding a
long session as one row. Each segment carries parent/segment hashes, a source
message range, chunk ordinal/count, continuation status, and adjacent IDs when
those IDs survive validation. `--oversize-strategy reject` is retained only
for controlled historical comparisons. A single indivisible message can still
be rejected by the final `--max-record-chars` safety bound. An oversized
episode candidate can still contribute explicitly tagged candidate action
windows; those windows are not accepted SFT or verified process supervision.

For a first-pass corpus measurement, use `--max-records-per-file N`; the
manifest records that limit and the run must not be treated as a complete
coverage result. Remove the limit only after the bounded sample's lineage,
rejection reasons, tool-window matching, and privacy findings are reviewed.

For a filtered corpus, the stronger path binds every input hash to the
Privacy Filter manifest:

```bash
python3 filter_privacy.py extracted_data --output-dir filtered_data --device cuda
python3 build_training_data.py filtered_data \
  --output-dir training_data \
  --privacy-mode filtered \
  --privacy-manifest filtered_data/privacy_manifest.json \
  --privacy-approved
```

`--privacy-approved` is a human review assertion, not an automated safety
claim. The builder keeps records in `quality.status=review` and
`privacy.eligible_for_training=false` until that flag is supplied. The
`none` privacy mode can never become eligible, even if both flags are passed.

The optional Privacy Filter is a data-minimization layer, not an anonymization
or compliance guarantee. Review the output and apply organization-specific
secret, license, and privacy controls.

## Output files

Compatibility dataset rows use `schema_version=ai-data-extraction/v1`, a
deterministic `example_id`, a deterministic `split` (`train`, `validation`, or
`test`), provider/task/outcome/model-tier/privacy tags, quality metadata, and
hashed provenance. `action_windows.jsonl` uses the dedicated
`ai-data-extraction/action-window/v2` schema. It carries the same
`session_quality_gate`, `session_quality_id`, `model_tier`, and quality flags
as its parent episode plus an `ai-data-extraction/action-evidence/v1` object.
No raw full session ID or full source path is exported.

Accepted rows report `quality.payload_chars` and `quality.event_count`. Token
counts remain `null`/`not_tokenized` until a pinned target tokenizer is chosen;
character counts must not be substituted for trainer token budgets. Rows with
`quality.observation_payload_status=truncated` are explicitly distinguishable
from complete observations. `lineage.open_tool_call_ids` identifies an action
whose source ended without a matching observation; it is not a success or
failure label.

| File | Intended consumer | Label rule |
| --- | --- | --- |
| `sft.jsonl` | SFT / instruction tuning | Complete user-plus-assistant conversations only |
| `trajectories.jsonl` | Offline audit, analysis, or reward-label preparation | Tool actions, observations, artifacts, context, and explicit terminal state |
| `tool_traces.jsonl` | Tool-call SFT adapters, tool-use analysis, and RL adapters | Sanitized messages plus action/observation/artifact/context events; includes rows with tool or context data |
| `action_windows.jsonl` | Tool-use review, process supervision preparation, and verifier enrichment | One observed tool transition with bounded context and deterministic structural evidence; every source action remains `positive_target_status=not_adjudicated` until a separate label owner supplies valid outcome evidence |
| `preferences.jsonl` | DPO / preference trainers | Only source records with explicit `chosen` and `rejected` values |
| `rl_prompts.jsonl` | GRPO / RLOO / other online RL input | Prompt through the latest user turn; always `reward=null`, `reward_status=unscored` |
| `rejected.jsonl` | Audit and remediation | Reason codes plus hashes and source location; no rejected content |
| `manifest.json` | Provenance and release gate | Input/output hashes, counts, policy, task/source/outcome totals |

The builder removes exact reasoning fields and common tagged blocks such as
`<think>`, `<analysis>`, and fenced reasoning blocks. It also drops unknown
message roles, multimodal parts that are not representable as text, and
incomplete conversations. Counts for those actions remain in `quality`.

The structural scrub catches common secrets, e-mail addresses, and home-path
patterns when privacy is enabled. That is not proof that proprietary code,
identifiers, or secrets are absent.

## Trainer mapping

The SFT, tool-trace, and preference files are conversational JSONL. SFT rows
also retain the sanitized `events` column so a tool-aware adapter does not lose
detached tool output; a plain chat trainer should ignore that extra column.
Load them with a JSON dataset reader and select rows by the embedded `split`
field. Keep metadata columns during audit; remove them only in a final trainer
adapter if the trainer rejects extra columns.

For tool-call SFT, records include normalized `messages` tool calls and retain
an optional `tools` column when the source contained function schemas. If a
source did not export schemas, add a reviewed tool-schema catalog before
training tool use; do not fabricate one from tool names alone.

Parseable tool arguments are stored as JSON objects/arrays, matching current
TRL conversational tool-calling guidance. Unparseable provider argument text is
retained only as a string and counted in `quality.tool_arguments_unparsed`; a
trainer adapter must either reject/review those rows or stringify arguments for
an API that requires the older OpenAI-style representation. See the current
[TRL tool-calling dataset format](https://huggingface.co/docs/trl/dataset_formats#tool-calling).

For tool-use training, do not throw away the event stream. Convert each
`action` into an assistant tool call, each `observation` into a `tool` message
linked by `call_id`, and retain `artifact`/`context` as supervised environment
state or verifier inputs. When source order is detached, use
`message_index`/`ordering_note` and mark the trace as requiring review rather
than inventing an order. If an observation is truncated, preserve its marker
and hash metadata outside the trainer message and do not count it as complete
verifier evidence.

`action_windows.jsonl` is a candidate projection, not verified process reward.
Its `evidence` object records deterministic action/turn hashes, one-to-one
observation-match strength, structured status/result codes, output digests,
same-action output novelty, immediate recurrence, complete period-2/period-3
cycles, and artifact hashes before the next action. Parallel calls from one
assistant message are one sequence turn. Output prose is never parsed into
success, tests, or reward, and evidence contains hashes rather than copied
observation or artifact text. Episode outcome is retained with
`step_credit=absent`; it does not label an individual action.

`verification` remains a separate harness-owned object and stays
`source=unscored` in canonical extraction. The projection intentionally does
not claim that a skill was read, a tool was necessary, or an action was a good
positive target. A trainer adapter must fail closed on
`positive_target_status=not_adjudicated`; executable replay, explicit
adjudication, or another named verifier is required before process supervision,
preference optimization, or RL.

TRL's current dataset guidance maps conversational SFT to `messages`, DPO to
explicit `prompt`/`chosen`/`rejected`, and GRPO/RLOO to prompt-only records:
<https://huggingface.co/docs/trl/dataset_formats>. A chat template may still
need to be applied by the selected trainer. The canonical files deliberately
remain trainer-neutral so the same provenance can feed Unsloth, TRL, or a
different stack.

Example inspection:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="training_data/sft.jsonl", split="train")
dataset = dataset.filter(lambda row: row["privacy"]["eligible_for_training"])
train = dataset.filter(lambda row: row["split"] == "train")
validation = dataset.filter(lambda row: row["split"] == "validation")
test = dataset.filter(lambda row: row["split"] == "test")
```

## Unified trainer pilot

`build_trainer_pilot.py` combines already-verified silver pilots with
authorized gold dialogue/tool releases. It does not read source archives or
promote quarantine rows. The output contract is:

- `train.jsonl`, `validation.jsonl`: dialogue SFT rows;
- `tool_train.jsonl`, `tool_validation.jsonl`: schema-bound tool SFT rows;
- `lineage.jsonl`: parent/source identity, quality tier, authorization source,
  and split;
- `decisions.jsonl`: one explicit decision for every input row;
- `manifest.json`: input/output hashes, counts, split policy, and release
  status.

The unified manifest remains unauthorized when it contains silver rows. Use
`validate_trainer_pilot.py` with the selected environment's `datasets` package
to load every trainer file and bind the loader report SHA into the manifest.
The loader check proves JSONL/schema compatibility; it does not authorize
training or prove task success.

### Archived salvage pilot composition

Compose an existing unified trainer pilot with the event-rich historical tool
pilot without changing either input:

```bash
python3 build_archived_salvage_pilot.py \
  .tmp/unified_trainer_pilot_<revision> \
  .tmp/historical_tool_trajectory_pilot_<revision> \
  .tmp/archived_salvage_pilot_<revision>
```

Validate it with `validate_archived_salvage_pilot.py` in the pinned loader
environment. `sft_*` and `tool_sft_*` are trainer-shaped partitions;
`tool_trajectory_*` retains historical messages/events for review and replay
but is not tool SFT when `tool_contract.schema_status=not_observed`. The
composer applies one global parent split across all partitions, retains input
manifest/file hashes and decisions, and keeps `training_authorized=false`.

### Harness-captured tool episodes

Use `capture_code_indexer_episode.py` only at the real read-only CLI boundary:

```bash
python3 capture_code_indexer_episode.py \
  .tmp/code_indexer_tool_capture_<date> \
  --project-root .
```

The adapter records one exact callable schema, installed-binary SHA-256,
required skill reads, command arguments, bounded tool output, verifier checks,
and terminal status. It writes `episode.jsonl` and `trace.jsonl`, then runs the
single-funnel `harness_tool_projection.py` join under `projection/`. The
projection is `trainer_loadable=true` but `training_authorized=false` until a
separate release decision. It exports no reward.

The adapter sanitizes the absolute checkout root in the tool result and keeps
the raw-result hash for provenance; it does not claim semantic completeness
from a Code Indexer capability label. A CLI failure, missing exact definition,
missing skill read, registry mismatch, unmatched call/observation, failed
verification, or privacy finding must reject the episode instead of producing
a trainer row. Provider session parsers must use this same harness boundary
for tool SFT; never infer a callable schema from a historical tool name.

For a bounded live positive-control set, capture multiple exact definitions and
join them without rebuilding the session archive:

```bash
python3 build_code_indexer_tool_batch.py \
  .tmp/code_indexer_tool_capture_batch_<date>/release \
  .tmp/code_indexer_tool_capture_batch_<date>/episode_001 \
  .tmp/code_indexer_tool_capture_batch_<date>/episode_002 \
  .tmp/code_indexer_tool_capture_batch_<date>/episode_003
```

The batch join refuses mixed project/binary/registry/skill/verifier identity,
requires every capture manifest and projection digest, and assigns a global
parent-disjoint train/validation split. Validate it in the pinned loader
environment:

```bash
.tmp/trainer_loader_venv_<revision>/bin/python \
  validate_code_indexer_tool_batch.py \
  .tmp/code_indexer_tool_capture_batch_<date>/release
```

This release is `trainer_loadable=true` but remains
`training_authorized=false`; it contains no reward. The loader report is
recorded in the batch manifest's validation block.

### Historical tool-trajectory salvage

Use `build_historical_tool_trajectory_pilot.py` for an already-bound release
partition when the source has recoverable calls/observations but no exact
callable registry or verifier join:

```bash
python3 build_historical_tool_trajectory_pilot.py \
  /data-sea/dump/ai_data_release_training_<revision> \
  .tmp/historical_tool_trajectory_pilot_<revision>
```

The exporter streams one bounded partition, verifies its manifest count,
bytes, and SHA-256, preserves parent/session boundaries, messages, and
normalized events, and emits explicit row decisions. It assigns
train/validation by parent hash, not by individual segment. The output schema
is `ai-data-extraction/historical-tool-trajectory-pilot/v1` and is review
data, not tool SFT: `tool_contract.schema_status=not_observed`,
`training_authorized=false`, and `reward_status=not_exported` are mandatory.

Provider payloads may encode arguments, event inputs, or optional lineage
fields as incompatible JSON types. For this review format, each event is
preserved as canonical JSON text and each tool-call argument is represented as
JSON text; the representation is explicit so a later schema-bound adapter can
parse it deliberately. Do not silently treat these rows as callable-schema
examples or verified outcomes. Validate with
`validate_historical_tool_trajectory_pilot.py` in the pinned loader
environment; the validator must prove row counts, parent-disjoint splits,
message/event presence, no reasoning markers, and no exported rewards.

Do not use `cat extracted_data/*.jsonl` as a training merge. Provider files
have incompatible schemas, may include duplicate exports, and may include
reasoning or sensitive fields.

### Tool schema-enrichment queue

For a bounded historical pilot, build a replay/adjudication queue instead of
promoting name-only tool traces:

```bash
python3 build_tool_schema_enrichment_queue.py \
  .tmp/historical_tool_trajectory_pilot_<revision> \
  .tmp/tool_schema_enrichment_queue_<revision> \
  --registry .tmp/code_indexer_registry_<revision>/tool_registry.json \
  --replay-limit 64
```

The queue streams the validated pilot, preserves messages and paired
action/observation payloads as explicit JSON text, and emits one decision for
every input row plus a deterministic parent-diverse replay subset. A registry
name match is only a candidate: schema binding requires exact source call
identity and a temporally bound registry revision. The queue never infers a
schema, verifier, reward, or training authorization. Validate the queue with
`validate_tool_schema_enrichment_queue.py` in the pinned loader environment;
the queue and replay files are loaded separately because they intentionally
have different schemas.

### Replay adjudication packet

Materialize the selected queue tasks against a revision-bound registry:

```bash
python3 build_replay_adjudication.py \
  .tmp/tool_schema_enrichment_queue_<revision> \
  .tmp/code_indexer_registry_<revision>/tool_registry.json \
  .tmp/replay_adjudication_<revision>
```

Validate the packet with the pinned loader environment:

```bash
.tmp/trainer_loader_venv_<revision>/bin/python \
  validate_replay_adjudication.py \
  .tmp/replay_adjudication_<revision>
```

The packet contains `replay_tasks.jsonl` and `decisions.jsonl`. It is
trainer-loadable as a review artifact, but must remain
`training_authorized=false`. Exact action-name intersection is only a gate
diagnostic; it does not bind a callable schema. A task cannot become replay,
tool SFT, or RL data until the packet has an exact revision-bound callable
schema, a bound workspace snapshot, an executable verifier, and a recorded
execution result. The packet never infers rewards from assistant text.

### Multi-provider salvage pilot

Use the bounded input binding to compose existing normalized provider
partitions without rebuilding the archive:

```bash
python3 build_multi_provider_salvage_pilot.py \
  .tmp/multi_provider_salvage_inputs_<revision>.json \
  .tmp/multi_provider_salvage_pilot_<revision>
```

Validate the package with the pinned loader environment:

```bash
.tmp/trainer_loader_venv_<revision>/bin/python \
  validate_multi_provider_salvage_pilot.py \
  .tmp/multi_provider_salvage_pilot_<revision>
```

The package keeps `sft_tier1_*` separate from `sft_optional_*`, and keeps
`tool_review_*` separate from trainer SFT because historical tool schemas and
verifiers are not inferred. `lineage.jsonl` carries provider, agent, model
tier/basis, quality gate/flags, source row/file hashes, parent identity, and
episode chunk links. `decisions.jsonl` covers every source row and records
explicit SFT/tool/RL decisions. The package is review-only and must remain
`training_authorized=false` until a separate release decision.

## RL semantics

Historical assistant responses are not automatically a reward signal. The
builder preserves a numeric source reward only when the source explicitly
provided one. It does not score text heuristically, label the assistant's
preferred answer, or convert a successful-looking message into a reward.

`rl_prompts.jsonl` is therefore a prompt inventory, not completed RL data. A
real online RL lane must supply an environment and a versioned reward
function. For coding tasks, a reviewed reward contract could combine isolated
test results, patch applicability, static checks, task-specific assertions,
and leakage/safety penalties. Every component needs its own held-out tests and
weighting record before it is used for optimization.

Each RL prompt also has a `prompt_group_id` derived from the normalized prompt.
Exact repeats are retained for provenance and marked `rl:duplicate-prompt`, but
the manifest counts them and an RL sampler should cap or sample by that group.
Multiple independent trajectories for one prompt become useful when their
verified outcomes differ; simply repeating one rollout does not add signal.
Visible provider names are retained and tagged as `provider-mention:*` and, for
the prompt slice, `prompt:provider-specific`. Filter those rows from a
provider-neutral base mix only when the experiment calls for it; keep them for
provider-adapter or held-out compatibility evaluation rather than silently
deleting legitimate tasks.

## Skills synthesis

`corpus_to_skills.py` accepts canonical or filtered JSONL. Its renderer forwards
only schema, dataset, source label, tags, messages, events, and quality fields;
it drops raw IDs, paths, arbitrary metadata, and reasoning before sampling.
The model endpoint is local by default. Generated `SKILL.md` files remain
review artifacts until a human checks their scope, claims, and instructions.

The portable user-skill snapshots are under [`../skills`](../skills):
`core-principles`, `contract-enforcement`, `systematic-debugging`, and
`code-indexer-ops`.
