---
name: contract-enforcement
description: Enforce contract-first, typed, single-funnel boundaries in TypeScript services and integrations. Use when adding or changing APIs, schemas, provider adapters, routing decisions, request/response transforms, persistence mappings, or cross-module state contracts.
---

# Contract Enforcement

Treat the active repository's schema, validation, API, and contribution conventions as the source of truth. This skill supplies the enforcement method; it does not replace project contracts. Program scheduling, agent assignment, checkpoint lineage, and root-order authority belong to `core-principles`, not this product-contract skill.

## Contract rules

- Define finite domains as a central typed union, enum, or generated schema. Do not use free-form strings, scattered literals, `Record<string, ...>`, or casts as a substitute.
- Validate raw external inputs at ingress and emitted payloads at egress using the repository's canonical validator. Normalize once in a boundary adapter.
- Keep one canonical mapper in each direction. Do not duplicate provider-to-domain or domain-to-wire transforms in handlers, services, and callers.
- Keep external SDK and HTTP shapes inside the adapter. Expose a minimal domain request, decision, and failure contract to the rest of the application.
- Make optionality, defaults, unknown values, and failures explicit. Fail closed for invalid route decisions unless the project specifies a safe fallback.
- Preserve client wire compatibility unless the contribution explicitly changes it. Mark every surface additive, backward-compatible, or breaking.

## Routing and model-provider integrations

- Model a route result as a typed decision with a selected target, reason code, confidence/score semantics, and explicit fallback behavior.
- Separate semantic classification, strength scoring, policy/configuration, and final target execution. Each can inform the next stage, but none should silently own the others.
- Treat tool use, explicit user model selection, modality/capability requirements, context limits, and safety constraints as hard compatibility gates before cost or quality ranking.
- Do not relabel a heuristic as a trained or calibrated classifier. Preserve the external router's stated score meaning and calibration requirements.
- Add contract tests for successful decisions, unavailable classifier/service, invalid response, tool-bearing request bypass, unsupported target, and no-eligible-target behavior.

## Proof before merge

1. Map producers, consumers, direct dependants, and relevant transitive adapters.
2. Verify there is one active funnel per direction and that old mappings are removed or deliberately supported.
3. Run the project-required lint, type, schema, and focused test commands.
4. Review the diff for literal-domain leaks, ad-hoc casts, object-spread boundary mutations, and undocumented compatibility changes.
