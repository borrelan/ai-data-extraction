import unittest

from runtime.sft.dataset import (
    resolve_text_tokenizer,
    tokenize_final_assistant,
    validate_example,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        if add_generation_prompt:
            return [10, 11, 12]
        return [10, 11, 12, 20, 21]


def example():
    return {
        "schema_version": "ai-data-extraction/agent-sft-example/v1",
        "example_id": "example-1",
        "split": "train",
        "lane": "reviewed_final_answer",
        "messages": [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "answer"},
        ],
    }


class SftDatasetTests(unittest.TestCase):
    def test_multimodal_processor_resolves_to_owned_text_tokenizer(self):
        tokenizer = FakeTokenizer()
        processor = type("FakeProcessor", (), {"tokenizer": tokenizer})()
        self.assertIs(resolve_text_tokenizer(processor), tokenizer)
        self.assertIs(resolve_text_tokenizer(tokenizer), tokenizer)

    def test_processing_class_requires_chat_template_owner(self):
        with self.assertRaisesRegex(TypeError, "text chat tokenizer"):
            resolve_text_tokenizer(object())

    def test_final_assistant_mask_uses_exact_prefix(self):
        row = example()
        validate_example(row, expected_split="train")
        tokenized = tokenize_final_assistant(row, FakeTokenizer(), max_length=8)
        self.assertEqual(tokenized["input_ids"], [10, 11, 12, 20, 21])
        self.assertEqual(tokenized["labels"], [-100, -100, -100, 20, 21])
        self.assertEqual(tokenized["target_tokens"], 2)

    def test_sequence_is_never_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds 4 tokens"):
            tokenize_final_assistant(example(), FakeTokenizer(), max_length=4)

    def test_requires_final_assistant_target(self):
        row = example()
        row["messages"][-1] = {"role": "tool", "content": "observation"}
        with self.assertRaisesRegex(ValueError, "not an assistant"):
            validate_example(row, expected_split="train")

    def test_tool_call_requires_one_matching_schema(self):
        row = example()
        row["messages"][-1] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "README.md"}},
                }
            ],
        }
        row["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        validate_example(row, expected_split="train")
        row["tools"][0]["function"]["name"] = "search_code"
        with self.assertRaisesRegex(ValueError, "no matching schema"):
            validate_example(row, expected_split="train")


if __name__ == "__main__":
    unittest.main()
