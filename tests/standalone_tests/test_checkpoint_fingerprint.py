# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness checks for persistent KV checkpoint identity."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import checkpoint_fingerprint  # noqa: E402


class CheckpointFingerprintTests(unittest.TestCase):
    def test_rechecks_earlier_files_after_hashing_all_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.safetensors"
            first.write_bytes(b"old")
            (root / "b.safetensors").write_bytes(b"second")
            scan = checkpoint_fingerprint._checkpoint_files
            scans = 0

            def replace_after_hashing(path, role):
                nonlocal scans
                scans += 1
                if scans == 2:
                    first.write_bytes(b"updated shard")
                return scan(path, role)

            with patch.object(
                checkpoint_fingerprint,
                "_checkpoint_files",
                side_effect=replace_after_hashing,
            ):
                with self.assertRaisesRegex(ValueError, "changed while hashing"):
                    checkpoint_fingerprint.fingerprint_checkpoints(str(root))

    def test_rechecks_file_set_after_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.safetensors").write_bytes(b"old")
            scan = checkpoint_fingerprint._checkpoint_files
            scans = 0

            def add_after_hashing(path, role):
                nonlocal scans
                scans += 1
                if scans == 2:
                    (root / "b.safetensors").write_bytes(b"new")
                return scan(path, role)

            with patch.object(
                checkpoint_fingerprint,
                "_checkpoint_files",
                side_effect=add_after_hashing,
            ):
                with self.assertRaisesRegex(ValueError, "file set changed"):
                    checkpoint_fingerprint.fingerprint_checkpoints(str(root))


if __name__ == "__main__":
    unittest.main()
