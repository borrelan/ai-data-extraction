import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_training_data import (
    ACTION_EVIDENCE_SCHEMA_VERSION,
    ACTION_WINDOW_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    PARSER_REVISION,
    SCHEMA_VERSION,
    action_windows_for,
    build_datasets,
    build_session_quality_index,
    canonical_json,
    canonical_json_size,
    chunk_record_variants,
    load_model_tier_overrides,
    model_tier_for,
    normalize_record,
    privacy_eligibility,
    quality_assessment_for,
    quality_session_key,
    sha256_file,
    stable_id,
    validate_action_window,
    validate_dataset_record,
    verify_privacy_manifest,
)


def source_args(record, *, mode="filtered", approved=True):
    return dict(
        record=record,
        source_file=Path("codex.jsonl"),
        source_file_hash="a" * 64,
        source_line=1,
        privacy_mode=mode,
        privacy_approved=approved,
        max_record_chars=250_000,
    )


def long_session_record():
    messages = [{"role": "system", "content": "Follow the repository contract."}]
    for index in range(6):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Objective {index}: investigate and fix the service failure. "
                    + ("details " * 8),
                },
                {
                    "role": "assistant",
                    "content": f"I completed the observable work for objective {index}. "
                    + ("evidence " * 8),
                },
            ]
        )
    return {
        "source": "codex",
        "session_id": "long-session",
        "messages": messages,
    }


def action_trajectory(events):
    """Build a minimal normalized trajectory for action-window contract tests."""

    normalized_events = []
    for ordinal, source in enumerate(events):
        event = copy.deepcopy(source)
        event.setdefault("schema_version", EVENT_SCHEMA_VERSION)
        event.setdefault("event_id", stable_id({"ordinal": ordinal, "event": event}))
        event.setdefault("ordinal", ordinal)
        event.setdefault("parent_record_sha256", "parent-fixture")
        normalized_events.append(event)
    max_message_index = max(
        (
            event["message_index"]
            for event in normalized_events
            if isinstance(event.get("message_index"), int)
        ),
        default=0,
    )
    messages = [{"role": "user", "content": "Investigate the failure."}]
    messages.extend(
        {"role": "assistant", "content": f"Observable step {index}."}
        for index in range(1, max_message_index + 1)
    )
    return {
        "example_id": stable_id(normalized_events),
        "messages": messages,
        "events": normalized_events,
        "tools": [],
        "trajectory": {
            "outcome": "unknown",
            "outcome_source": "unscored",
        },
        "metadata": {
            "parser_revision": PARSER_REVISION,
            "parent_record_sha256": "parent-fixture",
        },
        "quality": {
            "session_quality_gate": "candidate",
            "session_quality_id": "quality-fixture",
            "session_quality_flags": [],
            "model_tier": "tier1_frontier",
        },
        "lineage": {"parent_record_sha256": "parent-fixture"},
        "tags": ["provider:codex", "tier:tier1-frontier"],
        "privacy": {"eligible_for_training": False},
    }


def action(name, arguments, *, message_index, call_id):
    return {
        "kind": "action",
        "message_index": message_index,
        "call_id": call_id,
        "name": name,
        "status": "unknown",
        "status_source": "absent",
        "input": arguments,
    }


def observation(output, *, message_index, call_id, status="unknown"):
    return {
        "kind": "observation",
        "message_index": message_index,
        "call_id": call_id,
        "name": "shell",
        "status": status,
        "status_source": "absent",
        "result_code": None,
        "result_code_source": "absent",
        "output": output,
    }


class SegmentationTests(unittest.TestCase):
    def test_long_parent_is_split_at_user_boundaries_with_lineage(self):
        record = long_session_record()
        record["tool_results"] = [{"tool": "shell", "output": "one result"}]
        variants = chunk_record_variants(record, 450)

        self.assertGreater(len(variants), 1)
        parent_hashes = {variant["_chunk_parent_record_sha256"] for variant in variants}
        self.assertEqual(len(parent_hashes), 1)
        self.assertEqual(
            [variant["_chunk_index"] for variant in variants],
            list(range(len(variants))),
        )
        self.assertTrue(all(variant["_chunk_count"] == len(variants) for variant in variants))
        self.assertNotIn("tool_results", variants[1])
        self.assertEqual(
            [
                message["content"]
                for variant in variants
                for message in variant["messages"]
                if message.get("role") != "system"
            ],
            [
                message["content"]
                for message in record["messages"]
                if message.get("role") != "system"
            ],
        )
        ranges = [
            (variant["_chunk_message_start"], variant["_chunk_message_end"])
            for variant in variants
        ]
        self.assertEqual(ranges[0][0], 1)
        self.assertTrue(all(left <= right for left, right in ranges))
        self.assertTrue(all(ranges[index][1] < ranges[index + 1][0] for index in range(len(ranges) - 1)))

    def test_indivisible_message_is_left_for_final_rejection(self):
        record = {
            "source": "codex",
            "session_id": "one-large-message",
            "messages": [
                {"role": "user", "content": "help"},
                {"role": "assistant", "content": "x" * 1000},
            ],
        }
        variants = chunk_record_variants(record, 100)
        self.assertEqual(len(variants), 1)
        result = normalize_record(
            variants[0],
            source_file=Path("codex.jsonl"),
            source_file_hash="a" * 64,
            source_line=1,
            privacy_mode="filtered",
            privacy_approved=True,
            max_record_chars=100,
        )
        self.assertEqual(result[3], ["record_exceeds_max_chars"])
        self.assertIsNotNone(result[1])

    def test_pending_tool_calls_block_a_cut_through_interleaved_results(self):
        record = {
            "source": "codex",
            "session_id": "interleaved-tools",
            "messages": [
                {"role": "user", "content": "Investigate the failure."},
                {
                    "role": "assistant",
                    "content": "I am checking several independent signals. " + ("context " * 12),
                    "tool_calls": [
                        {"id": "call-a", "function": {"name": "shell", "arguments": {}}},
                        {"id": "call-b", "function": {"name": "shell", "arguments": {}}},
                    ],
                },
                {"role": "user", "content": "Keep the investigation bounded."},
                {"role": "tool", "tool_call_id": "call-a", "content": "first result"},
                {"role": "tool", "tool_call_id": "call-b", "content": "second result"},
                {"role": "assistant", "content": "The evidence is collected."},
                {"role": "user", "content": "Now summarize the result."},
                {"role": "assistant", "content": "The failure is isolated."},
            ],
        }

        variants = chunk_record_variants(record, 220)
        self.assertGreater(len(variants), 1)
        tool_batch_variants = [
            variant
            for variant in variants
            if any(
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and message.get("tool_calls")
                for message in variant["messages"]
            )
        ]
        self.assertEqual(len(tool_batch_variants), 1)
        batch = tool_batch_variants[0]["messages"]
        self.assertEqual(
            {message.get("tool_call_id") for message in batch if message.get("role") == "tool"},
            {"call-a", "call-b"},
        )


class NormalizationTests(unittest.TestCase):
    def test_session_quality_provenance_is_scrubbed_before_trainer_output(self):
        record = {
            "source": "codex",
            "session_id": "quality-assessment-provenance",
            "messages": [
                {"role": "user", "content": "Inspect the failure."},
                {"role": "assistant", "content": "The failure is isolated."},
            ],
        }
        assessment = {
            "assessment_version": "session-quality/v1",
            "scope": "source_session",
            "session_quality_id": "sha256:" + "a" * 64,
            "gate": "review_required",
            "flags": [],
            "dimensions": {
                "model_provenance": {
                    "models": ["<think>private deliberation</think>"],
                    "providers": ["local"],
                }
            },
            "provider_neutral": True,
        }

        sft, trajectory, _preference, _rejection, tool_trace = normalize_record(
            **source_args(record), session_quality_assessment=assessment
        )

        self.assertIsNotNone(sft)
        self.assertIsNotNone(trajectory)
        self.assertIsNone(tool_trace)
        self.assertEqual(
            sft["quality"]["session_quality_assessment"]["dimensions"][
                "model_provenance"
            ]["models"],
            [""],
        )
        validate_dataset_record(sft, "sft")
        validate_dataset_record(trajectory, "trajectory")

    def test_literal_reasoning_search_expression_survives_privacy_redaction(self):
        record = {
            "source": "codex",
            "session_id": "literal-reasoning-search",
            "messages": [
                {"role": "user", "content": "Find the relevant validator."},
                {
                    "role": "assistant",
                    "content": "I am searching the implementation.",
                    "tool_calls": [
                        {
                            "id": "call-search",
                            "function": {
                                "name": "exec",
                                "arguments": (
                                    "rg -n '\"(analysis|thinking|reasoning)\"|"
                                    "<think|<analysis|```(thinking|reasoning|analysis)' "
                                    "/home/borrelan/Projects/Personal/ai-data-extraction"
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-search",
                    "content": "No hidden reasoning fields found.",
                },
                {"role": "assistant", "content": "The search completed."},
            ],
        }

        sft, _trajectory, _preference, _rejection, _tool_trace = normalize_record(
            **source_args(record)
        )

        self.assertIsNotNone(sft)
        arguments = sft["messages"][1]["tool_calls"][0]["function"]["arguments"]
        self.assertIn("<think|<analysis", arguments)
        self.assertIn("<PRIVATE_PATH>", arguments)
        validate_dataset_record(sft, "sft")

    def test_session_quality_is_grouped_before_builder_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long-session.jsonl"
            record = {
                "source": "oh-my-pi",
                "training_lane": "optional_alt",
                "session_id": "one-session",
                "messages": [
                    {"role": "system", "content": "Use the repository contract."}
                ]
                + [
                    item
                    for index in range(8)
                    for item in (
                        {
                            "role": "user",
                            "content": f"Objective {index}: inspect the bounded service behavior.",
                        },
                        {
                            "role": "assistant",
                            "content": f"Evidence for objective {index} is recorded and reviewed.",
                        },
                    )
                ],
            }
            source.write_text(json.dumps(record) + "\n")
            source_hash = sha256_file(source)
            index, summary = build_session_quality_index([source])
            quality = index[quality_session_key(record, source_hash)]

            self.assertGreater(len(chunk_record_variants(record, 800)), 1)
            self.assertEqual(summary["sessions"], 1)
            self.assertEqual(quality["gate"], "candidate")
            self.assertEqual(quality["scope"], "source_session")

            manifest = build_datasets(
                [source],
                output_dir=root / "training",
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=800,
                overwrite=False,
                training_lanes=("optional_alt",),
            )
            rows = [
                json.loads(line)
                for line in (root / "training" / "sft.jsonl").read_text().splitlines()
                if line
            ]
            self.assertGreater(len(rows), 1)
            self.assertEqual({row["quality"]["session_quality_id"] for row in rows}, {quality["session_quality_id"]})
            self.assertEqual({row["quality"]["session_quality_gate"] for row in rows}, {"candidate"})
            self.assertEqual(manifest["counts"]["quality_sessions"], 1)

    def test_harness_summary_outcome_evidence_survives_quality_index(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "harness-session.jsonl"
            record = {
                "source": "prime-agent",
                "session_id": "harness-session",
                "harness_summary": {
                    "dimensions": {
                        "outcome_evidence": {
                            "status": "observed",
                            "terminal_signals": 3,
                        }
                    }
                },
                "messages": [
                    {"role": "user", "content": "Inspect the bounded change."},
                    {"role": "assistant", "content": "The verified result is recorded."},
                ],
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            index, _summary = build_session_quality_index([source])
            quality = index[quality_session_key(record, sha256_file(source))]

        self.assertEqual(quality["dimensions"]["outcome_evidence"]["status"], "observed")
        self.assertEqual(quality["dimensions"]["outcome_evidence"]["terminal_signals"], 3)
        self.assertNotIn("outcome_unverified", quality["flags"])

    def test_training_lanes_keep_optional_sources_out_of_default_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sessions.jsonl"
            rows = [
                {
                    "source": "prime-agent",
                    "training_lane": "optional_alt",
                    "quality_gate": "harness_governed_review",
                    "session_id": "alt",
                    "messages": [
                        {"role": "user", "content": "Inspect this."},
                        {"role": "assistant", "content": "Done."},
                    ],
                },
                {
                    "source": "oh-my-pi",
                    "training_lane": "quarantine",
                    "quality_gate": "advisor_overlay_contaminated",
                    "session_id": "advisor",
                    "messages": [
                        {"role": "user", "content": "Inspect this."},
                        {"role": "assistant", "content": "Done."},
                    ],
                },
                {
                    "source": "codex",
                    "session_id": "main",
                    "messages": [
                        {"role": "user", "content": "Inspect this."},
                        {"role": "assistant", "content": "Done."},
                    ],
                },
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            output = root / "training"
            manifest = build_datasets(
                [source],
                output_dir=output,
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
            )
            self.assertEqual(manifest["counts"]["input_records"], 3)
            self.assertEqual(manifest["counts"]["sft"], 1)
            self.assertEqual(manifest["counts"]["skipped_training_lane_records"], 2)
            self.assertEqual(
                manifest["training_lanes"],
                {"optional_alt": 1, "primary": 1, "quarantine": 1},
            )

    def test_quality_gates_filter_sessions_without_using_provider_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sessions.jsonl"
            assessment = {
                "assessment_version": "session-quality/v1",
                "gate": "review_required",
                "flags": ["payload_truncated"],
                "provider_neutral": True,
            }
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "source": provider,
                            "session_id": session_id,
                            "quality_assessment": (
                                assessment
                                if session_id == "review"
                                else {
                                    "assessment_version": "session-quality/v1",
                                    "gate": "candidate",
                                    "flags": ["model_provenance_local_or_self_hosted"],
                                    "provider_neutral": True,
                                }
                            ),
                            "messages": [
                                {"role": "user", "content": "Inspect this."},
                                {"role": "assistant", "content": "Done."},
                            ],
                        }
                    )
                    for provider, session_id in (("oh-my-pi", "review"), ("codex", "candidate"))
                )
                + "\n"
            )
            manifest = build_datasets(
                [source],
                output_dir=root / "training",
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
            )

        self.assertEqual(manifest["counts"]["sft"], 1)
        self.assertEqual(manifest["counts"]["skipped_quality_gate_records"], 1)
        self.assertEqual(manifest["quality_gates"], {"candidate": 1, "review_required": 1})

    def test_local_model_identity_in_message_is_a_flag_not_a_provider_verdict(self):
        assessment = quality_assessment_for(
            {
                "source": "pi-agent",
                "messages": [
                    {"role": "user", "content": "Check the test."},
                    {
                        "role": "assistant",
                        "content": "The test passes.",
                        "model": "llama.cpp/qwen-local",
                    },
                ],
            },
            messages=[
                {"role": "user", "content": "Check the test."},
                {
                    "role": "assistant",
                    "content": "The test passes.",
                    "model": "llama.cpp/qwen-local",
                },
            ],
            events=[],
        )
        self.assertEqual(assessment["gate"], "candidate")
        self.assertIn("model_provenance_local_or_self_hosted", assessment["flags"])

    def test_model_tier_is_explicit_and_emitted_as_a_release_tag(self):
        record = {
            "source": "codex",
            "model_tier": "tier1_frontier",
            "session_id": "frontier-session",
            "messages": [
                {"role": "user", "content": "Check the test."},
                {"role": "assistant", "content": "The test passes."},
            ],
        }
        tier = model_tier_for(record)
        self.assertEqual(tier["tier"], "tier1_frontier")
        self.assertEqual(tier["confidence"], "declared")
        sft, _trajectory, _preference, _rejection, _tool_trace = normalize_record(
            **source_args(record)
        )
        self.assertIsNotNone(sft)
        self.assertIn("tier:tier1-frontier", sft["tags"])
        self.assertEqual(sft["metadata"]["model_tier"], "tier1_frontier")
        self.assertEqual(sft["quality"]["model_tier"], "tier1_frontier")

    def test_model_registry_separates_frontier_open_source_and_local_routes(self):
        cases = (
            (
                {"source": "claude-code", "model": "claude-opus-4-5-20251101"},
                "tier1_frontier",
            ),
            (
                {"source": "claude-code", "model": "glm-5"},
                "tier2_open_source",
            ),
            (
                {
                    "source": "opencode",
                    "model": "qwen3.7-max",
                    "providerID": "openarc",
                },
                "tier3_local",
            ),
            (
                {"source": "codex", "model_provider": "openai"},
                "tier1_frontier",
            ),
        )
        for record, expected in cases:
            with self.subTest(record=record):
                self.assertEqual(model_tier_for(record)["tier"], expected)

    def test_local_runtime_conflicts_with_nonlocal_declared_tier(self):
        result = model_tier_for(
            {
                "source": "opencode",
                "model_tier": "tier1_frontier",
                "model": "claude-sonnet-4-5-20250929",
                "providerID": "openarc",
            }
        )
        self.assertEqual(result["tier"], "unclassified")
        self.assertTrue(result["conflict"])

    def test_primary_tier_selection_excludes_explicit_tier_two_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sessions.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "source": "opencode",
                        "model_tier": "tier2_open_source",
                        "session_id": "open-source-session",
                        "messages": [
                            {"role": "user", "content": "Check this."},
                            {"role": "assistant", "content": "Done."},
                        ],
                    }
                )
                + "\n"
            )
            manifest = build_datasets(
                [source],
                output_dir=root / "training",
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
                model_tiers=("tier1_frontier",),
            )
            self.assertEqual(manifest["counts"]["sft"], 0)
            self.assertEqual(manifest["counts"]["skipped_model_tier_records"], 1)
            rejected = [
                json.loads(line)
                for line in (root / "training" / "rejected.jsonl").read_text().splitlines()
                if line
            ]
            self.assertEqual(rejected[0]["status"], "skipped_model_tier")
            self.assertEqual(rejected[0]["model_tier"], "tier2_open_source")

    def test_reviewed_model_tier_override_promotes_legacy_identity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sessions.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "source": "codex",
                        "session_id": "legacy-frontier-session",
                        "messages": [
                            {"role": "user", "content": "Check this."},
                            {"role": "assistant", "content": "Done."},
                        ],
                    }
                )
                + "\n"
            )
            tier_file = root / "tier_overrides.json"
            tier_file.write_text(
                json.dumps(
                    {
                        "overrides": [
                            {
                                "session_id": "legacy-frontier-session",
                                "model_tier": "tier1_frontier",
                                "basis": "reviewed_runtime_registry",
                                "reviewer": "test",
                            }
                        ]
                    }
                )
            )
            overrides = load_model_tier_overrides(tier_file)
            manifest = build_datasets(
                [source],
                output_dir=root / "training",
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
                model_tiers=("tier1_frontier",),
                model_tier_overrides=overrides,
            )
            self.assertEqual(manifest["counts"]["sft"], 1)
            self.assertTrue(manifest["policy"]["model_tier_overrides"]["provided"])
            row = json.loads((root / "training" / "sft.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["metadata"]["model_tier"], "tier1_frontier")

    def test_builder_accepts_provider_neutral_session_quality_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sessions.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "source": "any-provider",
                        "session_id": "review-me",
                        "quality_assessment": {
                            "assessment_version": "session-quality/v1",
                            "gate": "review_required",
                            "flags": ["outcome_unverified"],
                            "provider_neutral": True,
                        },
                        "messages": [
                            {"role": "user", "content": "Check this."},
                            {"role": "assistant", "content": "Done."},
                        ],
                    }
                )
                + "\n",
            )
            manifest = build_datasets(
                [source],
                output_dir=root / "training",
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
                quality_overrides={
                    "review-me": {
                        "quality_gate": "candidate",
                        "reviewer": "human",
                        "reason": "verified outside adapter",
                    }
                },
            )
        self.assertEqual(manifest["counts"]["sft"], 1)
        self.assertEqual(manifest["counts"]["skipped_quality_gate_records"], 0)
        self.assertTrue(manifest["policy"]["quality_overrides"]["provided"])

    def test_streaming_identity_matches_canonical_json_identity(self):
        value = {"z": ["é", {"nested": True}], "a": 7}

        self.assertEqual(canonical_json_size(value), len(canonical_json(value)))
        self.assertEqual(
            stable_id(value),
            "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest(),
        )

    def test_normalizes_provider_tools_and_removes_reasoning(self):
        record = {
            "source": "gemini-cli",
            "session_id": "session-1",
            "source_origin": {
                "source_file_name": "rollout.jsonl",
                "source_file_sha256": "b" * 64,
                "source_event_line_range": {"start": 2, "end": 4},
            },
            "project_path": "/home/alice/private-project",
            "messages": [
                {
                    "role": "user",
                    "content": "Fix alice@example.com in the service. token=super-secret-value",
                    "context_items": [{"file": "/home/alice/private-project/app.py"}],
                },
                {
                    "role": "assistant",
                    "content": "<think>Do not export this internal reasoning.</think>Here is the patch.",
                    "thoughts": "also excluded",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "shell",
                                "arguments": {
                                    "command": "pytest",
                                    "reasoning_effort": "high",
                                },
                            },
                        }
                    ],
                },
            ],
            "tool_results": [
                {"type": "tool_result", "tool": "shell", "output": "2 passed"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "observation_truncations": [
                {
                    "field": "content",
                    "source_line": 4,
                    "call_id": "call-1",
                    "original_chars": 200_000,
                    "original_sha256": "c" * 64,
                    "kept_chars": 48_000,
                    "policy": "head_tail",
                }
            ],
        }

        sft, trajectory, preference, rejection, tool_trace = normalize_record(**source_args(record))

        self.assertIsNone(preference)
        self.assertIsNone(rejection)
        self.assertIsNotNone(tool_trace)
        self.assertIsNotNone(sft)
        self.assertIsNotNone(trajectory)
        self.assertEqual(sft["schema_version"], SCHEMA_VERSION)
        self.assertEqual(sft["metadata"]["parser_revision"], PARSER_REVISION)
        self.assertEqual(sft["metadata"]["provider"], "gemini")
        self.assertEqual(sft["messages"][1]["content"], "Here is the patch.")
        self.assertTrue(any(event["kind"] == "observation" for event in sft["events"]))
        self.assertIn("<PRIVATE_EMAIL>", sft["messages"][0]["content"])
        self.assertEqual(
            sft["metadata"]["source_origin"]["source_file_sha256"], "b" * 64
        )
        self.assertEqual(
            trajectory["metadata"]["source_origin"]["source_event_line_range"],
            {"start": 2, "end": 4},
        )
        self.assertIn("<SECRET>", sft["messages"][0]["content"])
        self.assertEqual(sft["tools"][0]["function"]["name"], "shell")
        self.assertEqual(
            sft["messages"][1]["tool_calls"][0]["function"]["arguments"],
            {"command": "pytest"},
        )
        self.assertNotIn("reasoning_effort", json.dumps(sft))
        self.assertIn("tool:shell", sft["tags"])
        self.assertIn("tool:testing", sft["tags"])
        self.assertIn("quality:observation-truncated", sft["tags"])
        self.assertEqual(sft["quality"]["observation_truncation_count"], 1)
        self.assertEqual(
            sft["metadata"]["observation_truncations"][0]["original_chars"],
            200_000,
        )
        self.assertEqual(sft["quality"]["tool_families"], ["shell", "testing"])
        self.assertEqual(trajectory["trajectory"]["reward"], None)
        self.assertEqual(trajectory["trajectory"]["reward_status"], "unscored")
        self.assertTrue(any(event["kind"] == "action" for event in trajectory["events"]))
        self.assertTrue(any(event["kind"] == "observation" for event in trajectory["events"]))
        self.assertTrue(all(event["schema_version"] == EVENT_SCHEMA_VERSION for event in trajectory["events"]))
        self.assertEqual(
            len({event["event_id"] for event in trajectory["events"]}),
            len(trajectory["events"]),
        )
        windows = action_windows_for(trajectory)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["decision"]["skill"]["status"], "not_observed")
        self.assertTrue(windows[0]["quality"]["observation_present"])
        validate_dataset_record(sft, "sft")
        validate_dataset_record(trajectory, "trajectory")

    def test_normalizes_structured_tool_blocks(self):
        record = {
            "source": "claude-code",
            "session_id": "anthropic-1",
            "messages": [
                {
                    "role": "user",
                    "content": "Inspect the failing test.",
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will inspect it."},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "pytest"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "1 failed",
                        }
                    ],
                },
                {"role": "assistant", "content": "The test fails in the assertion."},
            ],
        }

        sft, trajectory, _, rejection, tool_trace = normalize_record(**source_args(record))

        self.assertIsNone(rejection)
        self.assertIsNotNone(tool_trace)
        self.assertEqual(sft["messages"][1]["tool_calls"][0]["function"]["name"], "Bash")
        self.assertTrue(any(event["kind"] == "action" for event in trajectory["events"]))
        self.assertTrue(any(event["kind"] == "observation" for event in trajectory["events"]))

    def test_nested_reasoning_tool_arguments_are_removed_before_validation(self):
        record = {
            "source": "codex",
            "session_id": "nested-reasoning-tool",
            "messages": [
                {"role": "user", "content": "Inspect the implementation."},
                {
                    "role": "assistant",
                    "content": "I will inspect it.",
                    "tool_calls": [
                        {
                            "id": "thought-1",
                            "function": {
                                "name": "mcp__sequential-thinking__sequentialthinking",
                                "arguments": json.dumps(
                                    {
                                        "thought": "private deliberation",
                                        "thoughtNumber": 1,
                                        "totalThoughts": 1,
                                    }
                                ),
                            },
                        },
                        {
                            "id": "call-1",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps({"command": "pytest"}),
                            },
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "passed"},
                {"role": "assistant", "content": "The test passes."},
            ],
        }
        sft, trajectory, _, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(rejection)
        call_names = [
            call["function"]["name"]
            for message in sft["messages"]
            for call in message.get("tool_calls", [])
        ]
        self.assertEqual(call_names, ["shell"])
        self.assertNotIn("thought", json.dumps(sft))
        self.assertTrue(any(event["kind"] == "action" for event in trajectory["events"]))

    def test_visible_reasoning_markers_require_review(self):
        record = {
            "source": "codex",
            "session_id": "visible-marker-review",
            "messages": [
                {"role": "user", "content": "Do not reveal chain of thought."},
                {"role": "assistant", "content": "I will provide the result."},
            ],
        }

        sft, _, _, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(rejection)
        self.assertEqual(sft["quality"]["session_quality_gate"], "review_required")
        self.assertIn(
            "visible_trainer_marker_review",
            sft["quality"]["session_quality_flags"],
        )

    def test_role_tool_messages_become_call_id_observations(self):
        record = {
            "source": "codex",
            "session_id": "role-tool-1",
            "messages": [
                {"role": "user", "content": "Run the tests."},
                {
                    "role": "assistant",
                    "content": "I will run them.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": "pytest"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "2 passed",
                },
                {"role": "assistant", "content": "The tests pass."},
            ],
        }

        _, trajectory, _, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(rejection)
        self.assertEqual(
            [(event["kind"], event["call_id"]) for event in trajectory["events"]],
            [("action", "call-1"), ("observation", "call-1")],
        )
        windows = action_windows_for(trajectory)
        self.assertEqual(windows[0]["quality"]["observation_match"], "call-id")

    def test_none_privacy_mode_cannot_become_eligible(self):
        self.assertEqual(privacy_eligibility("none", True), (False, "unfiltered"))

    def test_unknown_roles_are_dropped_but_record_is_tagged(self):
        record = {
            "source": "codex",
            "session_id": "session-2",
            "messages": [
                {"role": "user", "content": "Run tests"},
                {"role": "internal_reasoning", "content": "private"},
                {"role": "assistant", "content": "Tests ran."},
            ],
        }

        sft, _, _, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(rejection)
        self.assertEqual(sft["quality"]["dropped_roles"], ["internal_reasoning"])
        self.assertNotIn("private", json.dumps(sft))

    def test_incomplete_records_are_rejected_with_provenance_only(self):
        record = {"source": "claude-code", "session_id": "only-user", "messages": [{"role": "user", "content": "help"}]}

        sft, trajectory, preference, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(sft)
        self.assertIsNone(trajectory)
        self.assertIsNone(preference)
        self.assertEqual(rejection, ["assistant_turn_missing"])

    def test_explicit_preference_is_preserved_without_inventing_one(self):
        record = {
            "source": "cursor",
            "session_id": "preference-1",
            "prompt": "How should this be fixed?",
            "chosen": "Use the canonical owner.",
            "rejected": "Add another fallback.",
        }

        sft, trajectory, preference, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(sft)
        self.assertIsNone(trajectory)
        self.assertIsNotNone(preference)
        self.assertIsNone(rejection)
        self.assertEqual(preference["dataset"], "preference")
        self.assertEqual(preference["quality"]["label_source"], "explicit-preference-field")

    def test_no_preference_is_created_from_a_single_assistant_response(self):
        record = {
            "source": "opencode-cli",
            "session_id": "ordinary-1",
            "messages": [
                {"role": "user", "content": "Explain this"},
                {"role": "assistant", "content": "This is the explanation."},
            ],
        }

        _, _, preference, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(preference)
        self.assertIsNone(rejection)

    def test_source_reward_and_outcome_are_preserved_as_explicit_labels(self):
        record = {
            "source": "codex",
            "session_id": "scored-1",
            "outcome": "success",
            "reward": 0.75,
            "messages": [
                {"role": "user", "content": "Run the test."},
                {"role": "assistant", "content": "The test passes."},
            ],
        }

        _, trajectory, _, rejection, _ = normalize_record(**source_args(record))

        self.assertIsNone(rejection)
        self.assertEqual(trajectory["trajectory"]["reward"], 0.75)
        self.assertEqual(trajectory["trajectory"]["reward_status"], "source")
        self.assertTrue(trajectory["trajectory"]["terminal"])


class ActionWindowEvidenceTests(unittest.TestCase):
    def test_unstructured_output_is_observed_but_not_promoted_to_success(self):
        record = {
            "source": "codex",
            "session_id": "unstructured-result",
            "messages": [
                {"role": "user", "content": "Run the tests."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": "pytest"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "2 passed"},
                {"role": "assistant", "content": "The command completed."},
            ],
        }

        _, trajectory, _, rejection, _ = normalize_record(**source_args(record))
        windows = action_windows_for(trajectory)

        self.assertIsNone(rejection)
        self.assertEqual(windows[0]["schema_version"], ACTION_WINDOW_SCHEMA_VERSION)
        evidence = windows[0]["evidence"]
        self.assertEqual(evidence["schema_version"], ACTION_EVIDENCE_SCHEMA_VERSION)
        self.assertEqual(evidence["positive_target_status"], "not_adjudicated")
        self.assertEqual(evidence["observation"]["match"], "call-id")
        self.assertEqual(evidence["observation"]["match_strength"], "exact")
        self.assertEqual(evidence["observation"]["status"], "unknown")
        self.assertEqual(evidence["observation"]["status_source"], "absent")
        self.assertIsNone(evidence["observation"]["result_code"])
        self.assertNotIn("2 passed", json.dumps(evidence))

    def test_structured_result_codes_preserve_success_and_failure(self):
        for result_code, expected_status in ((0, "success"), (7, "failure")):
            with self.subTest(result_code=result_code):
                record = {
                    "source": "codex",
                    "session_id": f"structured-result-{result_code}",
                    "messages": [
                        {"role": "user", "content": "Run the command."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "shell",
                                        "arguments": {"command": "check"},
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call-1",
                            "content": "opaque output",
                            "exit_code": result_code,
                        },
                        {"role": "assistant", "content": "The command completed."},
                    ],
                }

                _, trajectory, _, rejection, _ = normalize_record(**source_args(record))
                window = action_windows_for(trajectory)[0]

                self.assertIsNone(rejection)
                observation_event = next(
                    event
                    for event in trajectory["events"]
                    if event["kind"] == "observation"
                )
                self.assertEqual(observation_event["status"], expected_status)
                self.assertEqual(
                    observation_event["status_source"], "structured_result_code"
                )
                self.assertEqual(observation_event["result_code"], result_code)
                self.assertEqual(
                    window["evidence"]["observation"]["result_code"], result_code
                )
                self.assertEqual(
                    window["evidence"]["observation"]["status"], expected_status
                )

    def test_conflicting_structured_status_and_result_code_remain_unknown(self):
        record = {
            "source": "codex",
            "session_id": "conflicting-result",
            "messages": [
                {"role": "user", "content": "Run the command."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": "check"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "opaque output",
                    "status": "success",
                    "exit_code": 7,
                },
                {"role": "assistant", "content": "The command completed."},
            ],
        }

        _, trajectory, _, rejection, _ = normalize_record(**source_args(record))
        evidence = action_windows_for(trajectory)[0]["evidence"]["observation"]

        self.assertIsNone(rejection)
        self.assertEqual(evidence["status"], "unknown")
        self.assertEqual(evidence["status_source"], "conflicting_structured_evidence")
        self.assertEqual(evidence["result_code"], 7)

    def test_immediate_recurrence_and_observation_novelty_are_separate_facts(self):
        events = [
            action("shell", {"command": "check"}, message_index=1, call_id="a-1"),
            observation("first", message_index=1, call_id="a-1"),
            action("shell", {"command": "check"}, message_index=2, call_id="a-2"),
            observation("second", message_index=2, call_id="a-2"),
            action("shell", {"command": "check"}, message_index=3, call_id="a-3"),
            observation("second", message_index=3, call_id="a-3"),
        ]

        windows = action_windows_for(action_trajectory(events))

        self.assertIsNone(windows[0]["evidence"]["observation"]["novel_for_same_action"])
        self.assertTrue(windows[1]["evidence"]["observation"]["novel_for_same_action"])
        self.assertFalse(windows[2]["evidence"]["observation"]["novel_for_same_action"])
        self.assertTrue(windows[1]["evidence"]["sequence"]["immediate_repeat"])
        self.assertEqual(
            windows[1]["evidence"]["sequence"]["nearest_prior_turn_distance"], 1
        )

    def test_period_two_and_period_three_cycles_are_explicit(self):
        period_two_events = []
        for index, command in enumerate(("a", "b", "a", "b"), start=1):
            call_id = f"p2-{index}"
            period_two_events.extend(
                [
                    action(
                        "shell",
                        {"command": command},
                        message_index=index,
                        call_id=call_id,
                    ),
                    observation(command, message_index=index, call_id=call_id),
                ]
            )
        period_three_events = []
        for index, command in enumerate(("a", "b", "c", "a", "b", "c"), start=1):
            call_id = f"p3-{index}"
            period_three_events.extend(
                [
                    action(
                        "shell",
                        {"command": command},
                        message_index=index,
                        call_id=call_id,
                    ),
                    observation(command, message_index=index, call_id=call_id),
                ]
            )

        period_two = action_windows_for(action_trajectory(period_two_events))[-1]
        period_three = action_windows_for(action_trajectory(period_three_events))[-1]

        self.assertEqual(period_two["evidence"]["sequence"]["complete_cycle_periods"], [2])
        self.assertEqual(
            period_two["evidence"]["sequence"]["nearest_prior_turn_distance"], 2
        )
        self.assertFalse(period_two["evidence"]["sequence"]["immediate_repeat"])
        self.assertEqual(period_three["evidence"]["sequence"]["complete_cycle_periods"], [3])
        self.assertEqual(
            period_three["evidence"]["sequence"]["nearest_prior_turn_distance"], 3
        )

    def test_parallel_calls_share_one_turn_instead_of_forming_a_cycle(self):
        events = [
            action("read_file", {"path": "a.py"}, message_index=1, call_id="parallel-a"),
            action("read_file", {"path": "b.py"}, message_index=1, call_id="parallel-b"),
            observation("a", message_index=1, call_id="parallel-a"),
            observation("b", message_index=1, call_id="parallel-b"),
        ]

        windows = action_windows_for(action_trajectory(events))
        first = windows[0]["evidence"]
        second = windows[1]["evidence"]

        self.assertEqual(first["action"]["calls_in_turn"], 2)
        self.assertEqual(first["action"]["turn_ordinal"], 0)
        self.assertEqual(first["action"]["turn_signature"], second["action"]["turn_signature"])
        self.assertEqual(first["sequence"]["prior_turn_occurrences"], 0)
        self.assertEqual(second["sequence"]["complete_cycle_periods"], [])

    def test_missing_observation_and_artifact_boundary_remain_structural(self):
        artifact = {"path": "src/main.py", "patch_sha": "abc123"}
        events = [
            action("edit", {"path": "src/main.py"}, message_index=1, call_id="edit-1"),
            observation("written", message_index=1, call_id="edit-1"),
            {"kind": "artifact", "message_index": 1, "artifact": artifact},
            action("shell", {"command": "check"}, message_index=2, call_id="check-1"),
        ]

        windows = action_windows_for(action_trajectory(events))

        self.assertEqual(
            windows[0]["evidence"]["artifacts"]["hashes_before_next_action"],
            [stable_id(artifact)],
        )
        self.assertFalse(windows[0]["evidence"]["artifacts"]["content_included"])
        missing = windows[1]["evidence"]["observation"]
        self.assertFalse(missing["joined"])
        self.assertEqual(missing["match"], "unmatched")
        self.assertEqual(missing["match_strength"], "absent")
        self.assertIsNone(missing["output_digest"])

    def test_singleton_fallback_requires_an_idless_action(self):
        events = [
            observation("detached", message_index=0, call_id=None),
            action("shell", {"command": "check"}, message_index=1, call_id=None),
        ]

        window = action_windows_for(action_trajectory(events))[0]

        self.assertTrue(window["evidence"]["observation"]["joined"])
        self.assertEqual(
            window["evidence"]["observation"]["match"], "singleton-fallback"
        )
        self.assertEqual(
            window["evidence"]["observation"]["match_strength"], "heuristic"
        )

    def test_observation_is_consumed_by_only_one_ordered_action(self):
        events = [
            action("read_file", {"path": "a.py"}, message_index=1, call_id=None),
            action("read_file", {"path": "b.py"}, message_index=2, call_id=None),
            observation("only result", message_index=2, call_id=None),
        ]

        windows = action_windows_for(action_trajectory(events))

        self.assertFalse(windows[0]["evidence"]["observation"]["joined"])
        self.assertTrue(windows[1]["evidence"]["observation"]["joined"])
        self.assertEqual(
            windows[1]["evidence"]["observation"]["match"], "event-order"
        )

    def test_validator_rejects_reward_promotion_and_invalid_cycle_period(self):
        events = [
            action("shell", {"command": "check"}, message_index=1, call_id="call-1"),
            observation("opaque", message_index=1, call_id="call-1"),
        ]
        window = action_windows_for(action_trajectory(events))[0]

        promoted = copy.deepcopy(window)
        promoted["evidence"]["positive_target_status"] = "accepted"
        with self.assertRaisesRegex(ValueError, "evidence contract"):
            validate_action_window(promoted)

        invalid_cycle = copy.deepcopy(window)
        invalid_cycle["evidence"]["sequence"]["complete_cycle_periods"] = [4]
        with self.assertRaisesRegex(ValueError, "sequence evidence"):
            validate_action_window(invalid_cycle)


class BuildDatasetTests(unittest.TestCase):
    def test_build_splits_oversized_parent_and_links_derived_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long.jsonl"
            output = root / "training"
            source.write_text(json.dumps(long_session_record()) + "\n", encoding="utf-8")

            manifest = build_datasets(
                [source],
                output_dir=output,
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=450,
                overwrite=False,
            )

            self.assertEqual(manifest["counts"]["input_records"], 1)
            self.assertGreater(manifest["counts"]["sft"], 1)
            self.assertEqual(manifest["counts"]["chunked_parent_records"], 1)
            self.assertEqual(
                manifest["policy"]["oversize_strategy"],
                "chunk",
            )
            self.assertNotIn(
                "record_exceeds_max_chars",
                manifest["rejection_reasons"],
            )

            rows = [json.loads(line) for line in (output / "sft.jsonl").read_text().splitlines()]
            self.assertTrue(all("trace:chunked" in row["tags"] for row in rows))
            self.assertEqual(
                {row["metadata"]["parent_record_sha256"] for row in rows},
                {rows[0]["metadata"]["parent_record_sha256"]},
            )
            rows.sort(key=lambda row: row["lineage"]["chunk_index"])
            for index, row in enumerate(rows):
                self.assertEqual(row["lineage"]["chunk_index"], index)
                if index:
                    self.assertEqual(
                        row["lineage"]["previous_example_id"],
                        rows[index - 1]["example_id"],
                    )
                if index + 1 < len(rows):
                    self.assertEqual(
                        row["lineage"]["next_example_id"],
                        rows[index + 1]["example_id"],
                    )

    def test_reject_strategy_preserves_legacy_whole_record_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long.jsonl"
            output = root / "training"
            source.write_text(json.dumps(long_session_record()) + "\n", encoding="utf-8")

            manifest = build_datasets(
                [source],
                output_dir=output,
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=450,
                oversize_strategy="reject",
                overwrite=False,
            )

            self.assertEqual(manifest["counts"]["input_records"], 1)
            self.assertEqual(manifest["counts"]["sft"], 0)
            self.assertEqual(manifest["rejection_reasons"], {"record_exceeds_max_chars": 1})

    def test_oversize_episode_can_recover_bounded_action_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tool-heavy.jsonl"
            output = root / "training"
            record = {
                "source": "codex",
                "session_id": "tool-heavy",
                "messages": [
                    {"role": "user", "content": "Inspect the service."},
                    {
                        "role": "assistant",
                        "content": "I will inspect it.",
                        "tool_calls": [
                            {
                                "id": "call-heavy",
                                "function": {"name": "shell", "arguments": {}},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-heavy",
                        "content": "large output " * 200,
                    },
                    {"role": "assistant", "content": "The output is available."},
                ],
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            manifest = build_datasets(
                [source],
                output_dir=output,
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=300,
                overwrite=False,
            )

            self.assertEqual(manifest["counts"]["recovered_action_windows"], 1)
            windows = [
                json.loads(line)
                for line in (output / "action_windows.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(windows), 1)
            self.assertEqual(
                windows[0]["quality"]["source_episode_status"],
                "rejected_oversize",
            )
            self.assertEqual(windows[0]["quality"]["observation_match"], "call-id")
            self.assertIsNone(windows[0]["observation"]["output"])
            self.assertEqual(
                windows[0]["quality"]["observation_output_policy"],
                "omitted_rejected_oversize",
            )

    def test_build_deduplicates_repeated_raw_exports_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            output = root / "training"
            inputs.mkdir()
            record = {
                "source": "codex",
                "session_id": "same-session",
                "messages": [
                    {"role": "user", "content": "Fix the bug"},
                    {"role": "assistant", "content": "Apply the fix."},
                ],
            }
            (inputs / "one.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (inputs / "two.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            manifest = build_datasets(
                [inputs],
                output_dir=output,
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
            )

            self.assertEqual(manifest["counts"]["input_records"], 2)
            self.assertEqual(manifest["counts"]["sft"], 1)
            self.assertEqual(manifest["counts"]["duplicates"], 1)
            self.assertTrue(json.loads((output / "manifest.json").read_text())["outputs"]["sft.jsonl"]["sha256"])
            rejected = [json.loads(line) for line in (output / "rejected.jsonl").read_text().splitlines()]
            self.assertEqual(rejected[0]["reasons"], ["duplicate_example_id"])

    def test_unapproved_privacy_is_not_training_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "training"
            source.write_text(
                json.dumps(
                    {
                        "source": "claude-code",
                        "session_id": "privacy-1",
                        "messages": [
                            {"role": "user", "content": "Fix it"},
                            {"role": "assistant", "content": "Done"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = build_datasets(
                [source],
                output_dir=output,
                privacy_mode="heuristic",
                privacy_approved=False,
                max_record_chars=250_000,
                overwrite=False,
            )
            result = json.loads((output / "sft.jsonl").read_text())

            self.assertFalse(result["privacy"]["eligible_for_training"])
            self.assertEqual(result["quality"]["status"], "review")
            self.assertFalse(manifest["policy"]["privacy_approved"])

    def test_rl_prompt_groups_mark_duplicates_without_dropping_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "training"
            records = [
                {
                    "source": "codex",
                    "session_id": f"duplicate-{index}",
                    "messages": [
                        {"role": "user", "content": "Use OpenAI tool calling only if needed."},
                        {"role": "assistant", "content": "I will use the available capability."},
                    ],
                }
                for index in (1, 2)
            ]
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            manifest = build_datasets(
                [source],
                output_dir=output,
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
            )

            prompts = [
                json.loads(line)
                for line in (output / "rl_prompts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(prompts), 2)
            self.assertEqual(manifest["rl_prompt_duplicates"], {
                "groups": 1,
                "extra_rows": 1,
                "policy": "retain for provenance; cap or sample by prompt_group_id during RL",
            })
            self.assertTrue(all("rl:duplicate-prompt" in row["tags"] for row in prompts))
            self.assertTrue(all("prompt:provider-specific" in row["tags"] for row in prompts))
            self.assertEqual(manifest["provider_mentions"], {"openai": 2})

    def test_output_directory_is_excluded_when_input_is_its_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "training"
            source.write_text(
                json.dumps(
                    {
                        "source": "gemini-cli",
                        "session_id": "parent-1",
                        "messages": [
                            {"role": "user", "content": "Fix it"},
                            {"role": "assistant", "content": "Done"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            first = build_datasets(
                [root],
                output_dir=output,
                privacy_mode="filtered",
                privacy_approved=True,
                max_record_chars=250_000,
                overwrite=False,
            )
            self.assertEqual(first["counts"]["input_records"], 1)

    def test_privacy_manifest_binds_exact_filtered_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filtered = root / "filtered"
            filtered.mkdir()
            source = filtered / "source.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            manifest = filtered / "privacy_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "privacy-filter/v1",
                        "model": "openai/privacy-filter",
                        "outputs": [
                            {"name": "source.jsonl", "sha256": sha256_file(source)}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                verify_privacy_manifest(manifest, [source]),
                {"source.jsonl": sha256_file(source)},
            )
            source.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_privacy_manifest(manifest, [source])


if __name__ == "__main__":
    unittest.main()
