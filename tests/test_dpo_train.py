import unittest

from runtime.dpo.train import select_training_rows, token_lengths


def row(pair_id, lane, prompt, chosen, rejected):
    return {
        "pair_id": pair_id,
        "lane": lane,
        "prompt_input_ids": list(range(prompt)),
        "chosen_input_ids": list(range(chosen)),
        "rejected_input_ids": list(range(rejected)),
    }


class DPOTrainTests(unittest.TestCase):
    def test_token_lengths_count_both_preference_branches(self):
        self.assertEqual(
            token_lengths(row("pair-a", "state", 7, 3, 5)),
            {
                "prompt": 7,
                "chosen_completion": 3,
                "rejected_completion": 5,
                "chosen_sequence": 10,
                "rejected_sequence": 12,
            },
        )

    def test_canary_selects_longest_pair_and_retains_provenance(self):
        rows = [
            row("short", "skill", 8, 2, 2),
            row("long", "state_transition", 20, 4, 3),
            row("middle", "tool_decision", 12, 4, 4),
        ]
        selected, report = select_training_rows(rows, one_step_canary=True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["prompt_input_ids"], list(range(20)))
        self.assertEqual(
            set(selected[0]),
            {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"},
        )
        self.assertEqual(report["pair_id"], "long")
        self.assertEqual(report["source_index"], 1)
        self.assertEqual(report["tokens"]["chosen_sequence"], 24)

    def test_full_release_keeps_order_and_strips_provenance(self):
        rows = [
            row("pair-a", "skill", 8, 2, 2),
            row("pair-b", "state_transition", 20, 4, 3),
        ]
        selected, report = select_training_rows(rows, one_step_canary=False)
        self.assertIsNone(report)
        self.assertEqual(len(selected), 2)
        self.assertNotIn("pair_id", selected[0])
        self.assertEqual(selected[1]["prompt_input_ids"], list(range(20)))

    def test_empty_training_split_fails(self):
        with self.assertRaisesRegex(ValueError, "training split is empty"):
            select_training_rows([], one_step_canary=True)


if __name__ == "__main__":
    unittest.main()
