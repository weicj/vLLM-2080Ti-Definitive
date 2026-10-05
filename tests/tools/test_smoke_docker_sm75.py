# SPDX-License-Identifier: Apache-2.0
"""CPU-only smoke-tool contracts; runnable with unittest, without torch/pytest."""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "smoke_docker_sm75", ROOT / "tools/smoke_docker_sm75.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeContractTests(unittest.TestCase):
    def setUp(self):
        self.release, self.version = smoke.release_policy(ROOT)
        self.torch = SimpleNamespace(
            __version__=self.release["VALIDATED_TORCH_VERSION"],
            version=SimpleNamespace(cuda=self.release["VALIDATED_CUDA_VERSION"]),
        )
        self.vllm = SimpleNamespace(__version__=self.version)
        major, minor = map(int, self.release["PRIMARY_PYTHON_VERSION"].split("."))
        self.python = patch.object(
            smoke.sys, "version_info", SimpleNamespace(major=major, minor=minor)
        )
        self.python.start()
        self.addCleanup(self.python.stop)

    def test_current_release_metadata_is_the_version_policy(self):
        smoke.check_versions(self.release, self.version, self.torch, self.vllm)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "PROJECT_RELEASE.env").write_text(
                '# comment\nPROJECT_NAME="Name with spaces"\nFORK_RELEASE="9.9"\n'
            )
            (root / "pyproject.toml").write_text(
                '[tool.setuptools_scm]\nfallback_version = "9.9+2080ti"\n'
            )
            release, version = smoke.release_policy(root)
            self.assertEqual(release["PROJECT_NAME"], "Name with spaces")
            self.assertEqual(version, "9.9+2080ti")

    def test_rejects_cpu_or_drifted_torch_and_wrong_vllm(self):
        for version in (self.release["PRIMARY_TORCH_VERSION"] + "+cpu", "0.0+cu130"):
            with self.subTest(version=version):
                self.torch.__version__ = version
                with self.assertRaisesRegex(RuntimeError, "Torch"):
                    smoke.check_versions(
                        self.release, self.version, self.torch, self.vllm
                    )
        self.torch.__version__ = self.release["VALIDATED_TORCH_VERSION"]
        self.torch.version.cuda = None
        with self.assertRaisesRegex(RuntimeError, "Torch CUDA"):
            smoke.check_versions(self.release, self.version, self.torch, self.vllm)
        self.torch.version.cuda = self.release["VALIDATED_CUDA_VERSION"]
        self.vllm.__version__ = "0.0"
        with self.assertRaisesRegex(RuntimeError, "vLLM"):
            smoke.check_versions(self.release, self.version, self.torch, self.vllm)

    def make_flashqla(self, root):
        # Apply the real patch script to a minimal upstream-shaped tree.
        legacy = root / "flash_qla/ops/gated_delta_rule/legacy"
        (legacy / "csrc").mkdir(parents=True)
        for relative in (
            "__init__.py",
            "ops/__init__.py",
            "ops/gated_delta_rule/__init__.py",
            "ops/gated_delta_rule/legacy/sm_legacy.py",
            "ops/gated_delta_rule/legacy/csrc/gdn_forward.cu",
        ):
            (root / "flash_qla" / relative).touch()
        patch_spec = importlib.util.spec_from_file_location(
            "flashqla_patch", ROOT / "tools/patch_flashqla_sm75_imports.py"
        )
        patcher = importlib.util.module_from_spec(patch_spec)
        patch_spec.loader.exec_module(patcher)
        with patch.object(
            sys,
            "argv",
            ["patcher", str(root), str(ROOT / "tools/flashqla_sm75_patches")],
        ):
            self.assertEqual(patcher.main(), 0)
        return legacy / "sm_legacy.py"

    def test_flashqla_checker_accepts_real_patches_and_rejects_stale_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_flashqla(root)
            self.assertEqual(smoke.check_flashqla(root), source)
            source.write_text("# stale legacy implementation\n")
            with self.assertRaisesRegex(RuntimeError, "Missing FlashQLA patch"):
                smoke.check_flashqla(root)

    def test_cpu_smoke_does_not_import_native_or_query_cuda(self):
        # Fake torch has NO .cuda; import_module rejects every native/GPU module.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_flashqla(root)
            packages = {"torch": self.torch, "vllm": self.vllm}
            versions = {"vllm": self.version, "flashinfer-python": "test"}
            with (
                patch.object(
                    smoke, "release_policy", return_value=(self.release, self.version)
                ),
                patch.object(
                    smoke.importlib, "import_module", side_effect=packages.__getitem__
                ),
                patch.object(
                    smoke.importlib.metadata,
                    "version",
                    side_effect=versions.__getitem__,
                ),
                patch.object(
                    smoke.importlib.metadata,
                    "distribution",
                    return_value=SimpleNamespace(
                        files=[
                            "vllm/_C_stable_libtorch.abi3.so",
                            "vllm/_moe_C_stable_libtorch.abi3.so",
                        ]
                    ),
                ),
                patch.object(smoke.shutil, "which", return_value="/usr/bin/tool"),
                patch.object(smoke.subprocess, "run") as run,
                patch.dict(
                    os.environ,
                    {
                        "FLASHQLA_ROOT": str(root),
                        "FLASHQLA_DIR": str(root),
                        "TORCH_EXTENSIONS_DIR": str(root / "cache"),
                    },
                ),
                patch.object(sys, "stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(smoke.cpu_smoke(), (self.torch, source))
                self.assertTrue(run.call_args.kwargs["check"])
                self.assertTrue(
                    run.call_args.args[0][1].endswith("check_torch_inductor_e8m0.py")
                )
                run.side_effect = subprocess.CalledProcessError(1, "checker")
                with self.assertRaises(subprocess.CalledProcessError):
                    smoke.cpu_smoke()

    def test_cli_help_runs_installed_script_and_restores_metadata(self):
        platform = SimpleNamespace(device_type="", is_unspecified=lambda: True)
        original_argv = sys.argv
        console_script = "/opt/venv/bin/vllm"

        def import_platform(name):
            self.assertEqual(name, "vllm.platforms")
            self.assertEqual(sys.argv, [console_script, "serve", "--help"])
            return SimpleNamespace(current_platform=platform)

        def run_script(path, *, run_name):
            self.assertEqual(path, console_script)
            self.assertEqual(run_name, "__main__")
            self.assertEqual(platform.device_type, "cpu")
            raise SystemExit(exit_code)

        for exit_code in (0, 2):
            with (
                self.subTest(exit_code=exit_code),
                patch.object(smoke.shutil, "which", return_value=console_script),
                patch.object(
                    smoke.importlib, "import_module", side_effect=import_platform
                ),
                patch.object(smoke.runpy, "run_path", side_effect=run_script),
                patch.object(sys, "stdout", new_callable=io.StringIO),
                self.assertRaises(SystemExit) as raised,
            ):
                smoke.cli_help_smoke()
            self.assertEqual(raised.exception.code, exit_code)
            self.assertEqual(platform.device_type, "")
            self.assertIs(sys.argv, original_argv)

    def test_cli_help_rejects_missing_script_or_detected_device(self):
        with (
            patch.object(smoke.shutil, "which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "console script missing"),
        ):
            smoke.cli_help_smoke()
        with (
            patch.object(smoke.shutil, "which", return_value="/opt/venv/bin/vllm"),
            patch.object(
                smoke.importlib,
                "import_module",
                return_value=SimpleNamespace(
                    current_platform=SimpleNamespace(is_unspecified=lambda: False)
                ),
            ),
            patch.object(smoke.runpy, "run_path") as run,
            self.assertRaisesRegex(RuntimeError, "GPU-less"),
        ):
            smoke.cli_help_smoke()
        run.assert_not_called()

    def test_cli_help_mode_is_separate_from_metadata_and_gpu_smoke(self):
        with (
            patch.object(smoke, "cli_help_smoke") as help_smoke,
            patch.object(smoke, "cpu_smoke") as cpu,
            patch.object(smoke, "gpu_smoke") as gpu,
        ):
            self.assertEqual(smoke.main(["--cli-help"]), 0)
            help_smoke.assert_called_once()
            cpu.assert_not_called()
            gpu.assert_not_called()
            with (
                patch.object(sys, "stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit) as raised,
            ):
                smoke.main(["--gpu", "--cli-help"])
            self.assertEqual(raised.exception.code, 2)

    def test_gpu_is_explicit_and_errors_are_not_skipped(self):
        with (
            patch.object(smoke, "cpu_smoke", return_value=(self.torch, Path("legacy"))),
            patch.object(smoke, "gpu_smoke") as gpu,
        ):
            self.assertEqual(smoke.main([]), 0)
            gpu.assert_not_called()
            self.torch.inference_mode = lambda: patch.dict(os.environ, {})
            self.assertEqual(smoke.main(["--gpu"]), 0)
            gpu.assert_called_once()
            gpu.side_effect = RuntimeError("kernel failed")
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                smoke.main(["--gpu"])


if __name__ == "__main__":
    unittest.main()
