import tempfile
import unittest
from pathlib import Path

from mur.prepare import sha256_file, verify_model


class SourceManifestContracts(unittest.TestCase):
    def test_weights_and_config_files_are_both_verified(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "models" / "Repo"
            model.mkdir(parents=True)
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "config.json").write_text("config", encoding="utf-8")
            (model / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
            spec = {
                "id": "Owner/Repo",
                "revision": "abc",
                "weights_sha256": {"model.safetensors": sha256_file(model / "model.safetensors")},
                "config_sha256": {
                    "config.json": sha256_file(model / "config.json"),
                    "tokenizer.json": sha256_file(model / "tokenizer.json"),
                },
            }
            problems = []
            manifest = verify_model(Path(td), spec, problems)
            self.assertEqual(problems, [])
            self.assertEqual(manifest["files"]["model.safetensors"]["kind"], "weights")
            self.assertEqual(manifest["files"]["config.json"]["kind"], "config")
            self.assertEqual(manifest["files"]["tokenizer.json"]["kind"], "config")

            spec["config_sha256"]["tokenizer.json"] = "0" * 64
            problems = []
            verify_model(Path(td), spec, problems)
            self.assertTrue(any("tokenizer.json" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
