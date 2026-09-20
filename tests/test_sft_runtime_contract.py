import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime.sft.smoke import _verify_acquisition, _verify_versions, _write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "runtime" / "sft" / "runtime_contract.json"


class SftRuntimeContractTests(unittest.TestCase):
    def test_contract_pins_local_xpu_qlora_boundary(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["schema"], "ai-data-extraction/sft-runtime-contract/v1")
        primary = contract["container"]["primary_image"]
        self.assertEqual(primary["tag"], "ai-data-extraction/unsloth-xpu:torch2120-768d644")
        self.assertTrue(primary["id"].startswith("sha256:"))
        self.assertFalse(contract["container"]["local_pull_allowed"])
        self.assertIn("@sha256:", contract["container"]["base_image"])
        self.assertEqual(contract["accelerator"]["type"], "xpu")
        self.assertEqual(contract["accelerator"]["device"], "Intel Arc Pro B70")
        self.assertEqual(
            contract["model"]["loaded_class"], "Qwen3_5ForConditionalGeneration"
        )
        self.assertEqual(contract["model"]["modality"], "text-only")
        self.assertIn("in_proj_qkv", contract["peft"]["target_modules"])
        self.assertEqual(contract["training"]["max_sequence_tokens"], 8192)
        self.assertEqual(contract["training"]["loss"], "final-assistant-turn-only")
        self.assertFalse(contract["training"]["packing"])
        self.assertTrue(
            contract["training"]["adapter_host_path"].startswith(
                "/data-120/models/adapters/"
            )
        )
        self.assertTrue(contract["qualification"]["adapter_save_reload"])
        self.assertEqual(contract["qualification"]["longest_sequence_tokens"], 7184)
        self.assertLess(
            contract["qualification"]["longest_peak_reserved_bytes"],
            contract["accelerator"]["total_memory_bytes"],
        )

    def test_acquisition_check_binds_revision_size_and_critical_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            import hashlib

            payload = (model / "config.json").read_bytes()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "repo_id": "example/model",
                                "revision": "revision-1",
                                "files": [
                                    {
                                        "name": "config.json",
                                        "bytes": len(payload),
                                        "sha256": hashlib.sha256(payload).hexdigest(),
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = _verify_acquisition(
                manifest,
                model,
                repo_id="example/model",
                revision="revision-1",
            )
            self.assertEqual(result["file_count"], 1)
            self.assertEqual(result["critical_hashes_verified"], 1)

    def test_atomic_report_writer_replaces_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            _write_json_atomic(output, {"status": "passed"})
            self.assertEqual(json.loads(output.read_text()), {"status": "passed"})

    def test_record_policy_captures_immutable_image_versions_without_repinning(self):
        with patch("runtime.sft.smoke.importlib.metadata.version", return_value="9.9.9"):
            self.assertEqual(
                _verify_versions({"example": "1.0.0"}, policy="record"),
                {"example": "9.9.9"},
            )


if __name__ == "__main__":
    unittest.main()
