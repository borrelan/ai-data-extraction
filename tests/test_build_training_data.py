import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_training_data import (
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
