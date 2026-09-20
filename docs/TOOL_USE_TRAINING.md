# Tool-use and long-chain training plan

This project should train observable engineering behavior, not hidden chain of
thought. The supervised unit is a sequence of user-visible messages, skill and
tool decisions, structured calls, observations, artifacts, verification results,
and a bounded terminal state. A short decision label such as `capability`,
`risk`, `cost`, `negative-return`, or `user-override` is enough; private
deliberation is not a dataset requirement.

## Training platform decision

The project is tool-agnostic at the data and execution-contract layers, not at
the last trainer call. The canonical artifacts remain versioned JSONL records
with portable conversational messages, tool schemas, lineage, and harness
metadata. A thin adapter may materialize them as Hugging Face `datasets`
Arrow tables or another trainer's format; extraction must not be rewritten for
each trainer.

The first supported training stack is:

1. Hugging Face Transformers + Datasets + TRL define dataset semantics and
   provide SFT, preference, process-supervision, and prompt-only RL
   interfaces. TRL's current format guide distinguishes conversational SFT,
   explicit preference, stepwise supervision, and prompt-only GRPO/RLOO data:
   <https://huggingface.co/docs/trl/dataset_formats>.
2. Unsloth Core is the first execution target for a measured 27--35B QLoRA or
   LoRA pilot on the local 64 GB GPU. It is an optimization/runtime choice,
   not the canonical data model; its current project advertises QLoRA, SFT,
   DPO, GRPO, and Qwen3.8 support: <https://github.com/unslothai/unsloth>.
3. TRL GRPO is the small environment-backed prototype. It must use an explicit
   environment, verifier, and `max_tool_calling_iterations`; the current
   documentation notes that the default iteration limit is unbounded and that
   the chat template must be prefix-preserving when tools are enabled:
   <https://huggingface.co/docs/trl/grpo_trainer>.
4. veRL is the scale-up RL backend when asynchronous, multi-turn tool rollout
   is justified. It supports multi-turn tool calling, verifiable rewards, and
   separate rollout/training backends: <https://github.com/verl-project/verl>.
   MCP-Universe RL is an orchestration reference for isolated MCP environments
   and staged rollouts, not a replacement for this repository's canonical
   contract: <https://arxiv.org/abs/2608.22167>.

This gives us one stable foundation and replaceable execution backends. It
does not make chat templates, tool parsers, optimizer behavior, or environment
APIs interchangeable; those are pinned in each run manifest. The official
Qwen repository currently recommends Unsloth, Swift, and Llama-Factory for
fine-tuning its open checkpoints, and lists Qwen3.8-27B and Qwen3.6-35B-A3B as
local candidates: <https://github.com/QwenLM/Qwen3.8>. The initial checkpoint
is not finally pinned until the real training host passes Phase 0
memory/template checks.

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

## Training recipe for a 30--35B student

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
