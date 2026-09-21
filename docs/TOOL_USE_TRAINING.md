# Tool-use data and runtime integration

This project should train observable engineering behavior, not hidden chain of
thought. The supervised unit is a sequence of user-visible messages, skill and
tool decisions, structured calls, observations, artifacts, verification results,
and a bounded terminal state. A short decision label such as `capability`,
`risk`, `cost`, `negative-return`, or `user-override` is enough; private
deliberation is not a dataset requirement.

## Training platform boundary

The project is provider/trainer neutral in its canonical data contracts, but a
training or rollout run must pin its tokenizer, chat template, tool parser,
optimizer, environment API, and artifact revisions.

The supported owner split is:

1. `ai-data-extraction` owns private provider ingress and immutable source
   manifests. Its existing builder/exporter is a frozen compatibility path.
2. Hardened AgentIR owns canonical offline records, lineage, privacy, quality,
   lane eligibility, and loss-aware projections. ATIF v1.8 is interchange,
   not the source of truth.
3. Hugging Face Datasets/Transformers plus TRL define the first concrete SFT,
   preference, and prompt-only dataset contracts. PEFT is the portable adapter
   artifact boundary: <https://huggingface.co/docs/trl/dataset_formats>.
4. The first model lane is the exact Qwen3.5-9B checkpoint on the local Intel
   Arc Pro B70. The pinned local runtime may use Unsloth acceleration only
   where its XPU/model path is proven; it does not define the corpus. Cloud is
   not a fallback for this 9B lane. A later 27B cloud experiment is separate.
5. Fresh online RL uses resettable environments and executable verifiers, with
   Agent Lightning owning Rollout -> Attempt -> ordered span records. Historical
   transcripts may seed tasks but do not acquire fabricated rewards, token IDs,
   or policy log-probabilities.
6. OpenTelemetry carries runtime instrumentation. OpenObserve stores a
   privacy-filtered operational copy containing correlation IDs, hashes,
   timings, status, counts, and safe reward components. Raw prompts, model
   content, tool arguments/results, code/log payloads, private paths, secrets,
   and hidden reasoning are excluded by default.

The current local SFT runtime contract is documented in
[`../runtime/sft/README.md`](../runtime/sft/README.md). Runtime compatibility,
training loss, or a loadable adapter does not prove agentic improvement; the
untouched and adapted checkpoints must run the same executable task registry.

## The honest answer on open DevOps data

There is no single universally clean, gold-standard, long-chain DevOps corpus
that should be poured directly into a trainer. The strongest open material is a
combination of executable environments, task specifications, trajectories,
verifiers, and adversarial failure cases:

| Source | Best use | Boundary |
| --- | --- | --- |
| [Terminal-Bench](https://github.com/harbor-framework/terminal-bench) and [paper](https://arxiv.org/abs/2601.11868) | Real terminal tasks, Docker environments, oracle solutions, and tests; includes server/system work | Primarily a task and evaluation suite; generate and retain your own trajectories under its versioned harness |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) and [paper](https://arxiv.org/abs/2412.21139) | Executable repository issues, tests, agent trajectories, and verifier training | Software engineering rather than broad infrastructure; audit task licenses and trajectory provenance |
| [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) | Procedurally generated repository environments with tests and scalable trajectory collection | Synthetic/procedural data still needs held-out repository and contamination checks |
| [infra-bench](https://github.com/kubeply/infra-bench) | Realistic Kubernetes infrastructure tasks | Benchmark quality and license must be checked at the pinned revision before training |
| [Evidra Bench](https://github.com/vitas/evidra-bench) | Kubernetes, Helm, Argo CD, Terraform, and AWS/LocalStack incident environments with state/path verification | Promising infrastructure benchmark; treat repository claims as candidate evidence until locally reproduced |
| [MCP-Atlas](https://arxiv.org/abs/2602.00933) and [MCP-Bench](https://arxiv.org/abs/2508.20453) | Multi-server discovery, parameterization, cross-tool chains, and claim-level scoring | Evaluation environments, not automatically a licensed training corpus |
| [ToolBench/ToolLLM](https://github.com/OpenBMB/ToolBench) and [paper](https://arxiv.org/abs/2307.16789) | Broad API/schema diversity and tool-call formatting | Older, largely model-generated trajectories; use for schema coverage and negatives, not unquestioned gold behavior |
| [Terminal Wrench](https://arxiv.org/abs/2604.17596) | Reward-hacking and verifier-bypass trajectories for adversarial training | Explicitly adversarial; label as negative/attack data and never mix with successful demonstrations |

For this work, the highest-value DevOps lane is to pin a small set of these
environments and generate trajectories with a sandboxed harness. A successful
final state is more trustworthy than a persuasive transcript. Add Kubernetes,
Helm, Terraform, cloud emulators, incident diagnosis, rollback, observability,
and safe-change tasks with resettable state and deterministic checks. The
repository's own Code Indexer and LSP traces should be an internal domain slice,
not a replacement for broad held-out tools.

## What the canonical trace must retain

Each tool-bearing example should have, when available:

```json
{
  "tool_registry_snapshot": {"revision": "...", "tools": []},
  "skill_gate": {
    "skill": "contract-enforcement",
    "version_sha256": "...",
    "triggered": true,
    "read": true,
    "decision": "use",
    "skip_reason": null
  },
  "decision": "use",
  "decision_basis": "capability",
  "action": {
    "name": "lsp.find_references",
    "call_id": "call-1",
    "input": {}
  },
  "observation": {
    "call_id": "call-1",
    "status": "success",
    "output": "..."
  },
  "verification": {
    "tests": [],
    "diagnostics": [],
    "artifact_checks": [],
    "terminal": false
  }
}
```

The current `sft.jsonl`, `trajectories.jsonl`, and `tool_traces.jsonl` preserve
the available messages, schemas, calls, observations, artifacts, context,
call IDs, and quality tags. They do not invent missing skill-read or decision
events. The next harness capture should emit the additional fields above and
record the tool-registry and skill revision hashes. `tool:<family>` tags in the
current builder are conservative routing hints inferred from tool names and
sanitized action payloads; they are not semantic labels.

The provider ingress must also preserve omission semantics. Native Codex calls
are owned by assistant turns even when the source event order places a call
after a user item. Tool observations larger than the ingress envelope may be
head/tail bounded, but the emitted record must carry the original-content hash,
original size, source line, call ID, and `quality:observation-truncated` tag.
An unmatched call is retained as an explicit open action; it is useful for
recovery and negative training, but cannot be promoted to a verified transition.

## Skills and MCP enforcement belongs in the harness

Weights can learn a tendency to use a skill, but they cannot reliably enforce
that a particular file was read, that an MCP server was healthy, or that a
side-effecting call was authorized. Use a preflight/execute/verify funnel:

1. Resolve the relevant skill and MCP/tool registry from a pinned manifest.
2. Read the required skill content and record its hash, scope, and trigger.
3. Expose only the relevant tool schemas, with trust and side-effect metadata.
4. Require a structured `use`, `skip`, `defer`, or `ask` decision before a call.
5. Refuse side-effecting execution when a required contract, privacy, or safety
   gate is unread; the model cannot override these gates in text.
6. Permit `skip` only for an optional skill when the harness records a concrete
   negative return (unavailable dependency, incompatible scope, or measured
   regression). A user can insist on retrying an optional skill; user insistence
   does not erase safety, privacy, or contract enforcement.
7. Execute in a resettable sandbox and record the exact observation, artifact,
   state delta, and verifier result.

This makes “religious” skill/MCP behavior auditable instead of relying on a
prompt or hoping that a fine-tune memorizes a policy. Code Indexer and LSP are
first-class tool families, while lifecycle health and provenance remain
preconditions: an unavailable graph or stale index must produce an explicit
fallback/skip event, not a fabricated semantic answer.

## Training recipe for the 9B student and later scale-up

Use a staged recipe and keep the teacher out of the execution authority path:

1. **Non-reasoning SFT warm start.** Train on real, end-to-end, verified
   trajectories. Keep user-visible final responses, tool calls, tool outputs,
   patches, tests, and failure recovery. Remove hidden reasoning and raw private
   metadata. Include both necessary calls and concise no-tool answers.
2. **Tool correctness preferences.** Build explicit chosen/rejected pairs for
   wrong-tool selection, invalid arguments, gratuitous calls, unsafe actions,
   and successful recovery. Use DPO/IPO only for reviewed labels; do not turn a
   high-quality-looking answer into a preference by heuristic.
3. **Process supervision.** Train or apply a verifier over action validity,
   schema compliance, dependency order, progress, safety, and final state. Use
   deterministic tests, policy checks, and state assertions first; use a judge
   as an auxiliary signal with calibration and disagreement sampling.
4. **Sandboxed online RL.** Start with outcome rewards and short horizons using
   GRPO/RLOO or the selected trainer's equivalent. Add step/branch credit only
   after the environment and terminal reward are stable. Keep a replay mixture
   of verified SFT examples and monitor tool-call structure, entropy, invalid
   calls, repeated signatures, reward variance, and held-out OOD tools.
5. **Adversarial evaluation.** Hold out tool names, skill combinations, repo
   families, incident types, and MCP servers. Test prompt injection in tool
   outputs, stale indexes, unavailable skills, permission failures, malformed
   schemas, rollback, and reward-hacking attempts.

This ordering is supported by recent work, but no paper establishes a universal
best recipe. [ToolComp](https://arxiv.org/abs/2501.01290) motivates process-level
signals for multi-tool behavior; [ToolTrain](https://arxiv.org/abs/2508.03012)
combines rejection-sampled SFT with tool-integrated RL; and [Demystifying RL in
Agentic Reasoning](https://arxiv.org/abs/2510.11701) reports that real
end-to-end tool trajectories can provide a stronger SFT initialization than
stitched synthetic traces in its settings.

For multi-step credit, [PORTool](https://arxiv.org/abs/2510.26020) explores
tree rollouts with step-wise shared-prefix/fork-relative rewards, while
[RLFactory](https://arxiv.org/abs/2509.06980) separates asynchronous tool
calling from training. These are advanced second-stage options, not reasons to
start with a sparse, noisy RL objective.

The latest directly relevant warning found in this audit is [Why Multi-Step
Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix
It](https://arxiv.org/abs/2606.26027) (June 2026). It reports catastrophic
collapse of tool-invocation structure under naive multi-turn RL and studies
interleaved SFT plus supervisory signals. The operational implication here is
to interleave verified SFT, keep horizons bounded, and stop on structural
collapse; do not reward verbosity or raw call count.

Two newer methods sharpen the next-stage design. [TRACE](https://arxiv.org/abs/2607.13988)
assigns turn-level credit at tool-call state transitions using reference-model
log-ratio changes, while [Verifiable Process Rewards](https://arxiv.org/abs/2605.10325)
turns reliable intermediate oracles into dense rewards. Both are conditional
on a trustworthy state/verifier boundary; neither justifies assigning rewards
to unverified historical transcripts.

[MCP-Universe RL](https://arxiv.org/abs/2608.22167) separates isolated MCP
environment provisioning and staged rollout orchestration from the RL backend.
That separation is the right architecture for internal, external, DevOps, and
research tools: the dataset builder records the contract and lineage, while a
versioned harness owns execution, reset, health, and reward.

For teacher transfer, [SOD](https://arxiv.org/abs/2605.07725) reports that
student-side tool-call errors can cascade and make later teacher supervision
unreliable. Distillation must therefore be state-aware and gated by tool
validity, with explicit recovery/failure examples rather than blindly copying
teacher text over a divergent student trajectory.

## Preventing doom loops and over-thinking

The harness should enforce all of the following independently of model weights:

- maximum calls, turns, wall time, output bytes, and retry budget per task;
- repeated `(tool, normalized_args, state_hash)` detection with a hard cap;
- no-progress detection from unchanged state, artifact, test, and observation
  hashes;
- exponential backoff or escalation after the same failure, never blind retry;
- a terminal stop on verified success, permission denial, safety violation, or
  bounded failure;
- a compact final response target, with no reward for narrating hidden thought;
- a freeze-and-reopen-first-divergence rule when two repairs alternate in one
  owner family.

Training examples should include concise successful traces, useful partial
failures, explicit asks for missing authority, and “do not call a tool” cases.
The desired behavior is the minimum sufficient verified action sequence, not
the maximum number of steps.

## Data split and release rules

Maintain separate manifests for:

- verified demonstrations;
- generated teacher demonstrations;
- explicit human/model preferences;
- environment outcomes and verifier components;
- adversarial/reward-hack negatives;
- held-out evaluation tasks.

Never let a task's oracle patch, test fixture, or generated trajectory cross a
source-independent split. Pin environment images, tool schemas, skill hashes,
model checkpoints, prompts, and verifier revisions. Release only filtered,
license-approved artifacts. The repository's historical export remains a
volume/quality baseline and is not silently mixed into a later run.
