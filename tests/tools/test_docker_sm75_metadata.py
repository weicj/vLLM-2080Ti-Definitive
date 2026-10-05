# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[2] / "tools" / "docker_sm75_metadata.py"
SPEC = importlib.util.spec_from_file_location("docker_sm75_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
METADATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METADATA)

SHA = "abc01234" * 5
IMAGE = "ghcr.io/example-fork/vllm-2080ti-definitive"


class TestDockerSm75Metadata(unittest.TestCase):
    def metadata(self, **overrides):
        values = {
            "event_name": "push",
            "event": {"repository": {"default_branch": "main"}},
            "repository": "Example-Fork/vLLM-2080Ti-Definitive",
            "ref": "refs/heads/feature/docker",
            "sha": SHA,
            "run_id": "123456",
            "run_attempt": "1",
        }
        values.update(overrides)
        return METADATA.build_metadata(**values)

    def release(self, tag="v0.2.2-post3", prerelease=False, **overrides):
        event = {
            "action": "published",
            "release": {"tag_name": tag, "prerelease": prerelease, "draft": False},
        }
        return self.metadata(event_name="release", event=event, **overrides)

    def assert_no_publish(self, metadata):
        self.assertEqual(metadata["publish"], "false")
        self.assertEqual(metadata["tags"], "")

    def test_feature_branch_is_build_only(self):
        self.assert_no_publish(self.metadata())

    def test_default_branch_publishes_main_and_full_sha_not_latest(self):
        result = self.metadata(ref="refs/heads/main")
        self.assertEqual(result["publish"], "true")
        self.assertEqual(
            result["tags"].split(), [f"{IMAGE}:main", f"{IMAGE}:sha-{SHA}"]
        )

    def test_default_branch_name_is_not_hardcoded(self):
        result = self.metadata(
            ref="refs/heads/trunk",
            event={"repository": {"default_branch": "trunk"}},
        )
        self.assertEqual(
            result["tags"].split(), [f"{IMAGE}:main", f"{IMAGE}:sha-{SHA}"]
        )

    def test_unknown_default_branch_and_tag_push_do_not_publish(self):
        self.assert_no_publish(self.metadata(ref="refs/heads/main", event={}))
        self.assert_no_publish(self.metadata(ref="refs/tags/main"))

    def test_pr_cannot_publish_even_with_release_or_manual_fields(self):
        for event_name in ("pull_request", "pull_request_target"):
            with self.subTest(event_name=event_name):
                self.assert_no_publish(
                    self.metadata(
                        event_name=event_name,
                        ref="refs/heads/main",
                        event={
                            "repository": {"default_branch": "main"},
                            "inputs": {"publish": True},
                            "action": "published",
                            "release": {"tag_name": "v1.0.0", "prerelease": False},
                        },
                    )
                )

    def test_post_release_keeps_literal_tag_and_updates_latest(self):
        result = self.release()
        self.assertEqual(
            result["tags"].split(), [f"{IMAGE}:v0.2.2-post3", f"{IMAGE}:latest"]
        )
        self.assertEqual(result["version"], "v0.2.2-post3")

    def test_prerelease_flag_alone_controls_latest(self):
        for tag in ("v0.2.2-post3", "v1.0.0-rc1", "v1.0.0"):
            with self.subTest(tag=tag):
                self.assertEqual(self.release(tag, True)["tags"], f"{IMAGE}:{tag}")
                self.assertEqual(
                    self.release(tag, False)["tags"].split(),
                    [f"{IMAGE}:{tag}", f"{IMAGE}:latest"],
                )

    def test_prerelease_cannot_use_latest_as_its_literal_tag(self):
        with self.assertRaisesRegex(ValueError, "latest alias"):
            self.release("latest", True)
        self.assertEqual(self.release("latest", False)["tags"], f"{IMAGE}:latest")

    def test_release_uses_payload_tag_not_ref(self):
        result = self.release("V2.0.0-Post4", ref="refs/heads/main")
        self.assertEqual(result["tags"].split()[0], f"{IMAGE}:V2.0.0-Post4")

    def test_nonpublished_release_does_not_publish(self):
        self.assert_no_publish(
            self.metadata(
                event_name="release",
                event={"action": "created", "release": {"tag_name": "v1.0.0"}},
            )
        )

    def test_missing_or_string_prerelease_flag_is_rejected(self):
        for flag in (None, "false", "true", 0, 1):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "boolean"):
                self.release(prerelease=flag)

    def test_release_tags_are_validated_not_sanitized(self):
        for tag in (
            "v1/rc",
            "v1\nimage=other",
            "$(touch owned)",
            "-bad",
            "x" * 129,
            "",
        ):
            with (
                self.subTest(tag=tag),
                self.assertRaisesRegex(ValueError, "Docker tag"),
            ):
                self.release(tag)

    def test_draft_release_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "draft"):
            self.metadata(
                event_name="release",
                event={
                    "action": "published",
                    "release": {"tag_name": "v1", "prerelease": False, "draft": True},
                },
            )

    def test_manual_default_and_false_inputs_do_not_publish(self):
        for event in (
            {},
            {"inputs": {}},
            {"inputs": {"publish": False}},
            {"inputs": {"publish": "false"}},
        ):
            with self.subTest(event=event):
                self.assert_no_publish(
                    self.metadata(
                        event_name="workflow_dispatch",
                        event=event,
                        ref="refs/heads/main",
                    )
                )

    def test_manual_true_only_creates_unique_test_tag(self):
        for value in (True, "true"):
            with self.subTest(value=value):
                result = self.metadata(
                    event_name="workflow_dispatch", event={"inputs": {"publish": value}}
                )
                self.assertEqual(result["tags"], f"{IMAGE}:test-{SHA[:12]}-123456-1")
                self.assertEqual(result["publish"], "true")

    def test_manual_reruns_and_new_runs_have_distinct_tags(self):
        event = {"inputs": {"publish": True}}
        tags = {
            self.metadata(
                event_name="workflow_dispatch",
                event=event,
                run_id=run_id,
                run_attempt=attempt,
            )["tags"]
            for run_id, attempt in (("123456", "1"), ("123456", "2"), ("123457", "1"))
        }
        self.assertEqual(len(tags), 3)

    def test_fork_repository_is_lowercased_without_owner_allowlist(self):
        result = self.metadata(repository="SomeOtherOwner/My_SM75-Fork")
        self.assertEqual(result["image"], "ghcr.io/someotherowner/my_sm75-fork")

    def test_unknown_event_is_build_only(self):
        self.assert_no_publish(
            self.metadata(event_name="schedule", ref="refs/heads/main")
        )

    def test_untrusted_metadata_cannot_inject_outputs_or_shell(self):
        for field, value in (
            ("repository", "owner/repo\npublish=true"),
            ("sha", "$(touch owned)"),
            ("run_id", "123\ntags=latest"),
            ("run_attempt", "1;pwd"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.metadata(**{field: value})

    def run_main(self, directory, event_name, event, request_publish):
        event_path = directory / "event.json"
        output_path = directory / "output"
        event_path.write_text(json.dumps(event), encoding="utf-8")
        environment = {
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_OUTPUT": str(output_path),
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_REPOSITORY": "Example-Fork/vLLM-2080Ti-Definitive",
            "GITHUB_REF": "refs/heads/feature/docker",
            "GITHUB_SHA": SHA,
            "GITHUB_RUN_ID": "123456",
            "GITHUB_RUN_ATTEMPT": "1",
            "REQUEST_PUBLISH": request_publish,
        }
        with patch.dict(os.environ, environment, clear=True):
            METADATA.main()
        return output_path.read_text("utf-8")

    def test_main_writes_single_line_actions_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_main(
                Path(directory),
                "workflow_dispatch",
                {"inputs": {"publish": "true"}},
                "true",
            )
        values = dict(line.split("=", 1) for line in output.splitlines())
        self.assertEqual(len(values), 5)
        self.assertEqual(values["tags"], f"{IMAGE}:test-{SHA[:12]}-123456-1")

    def test_main_rejects_privileged_request_for_pr_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "does not match event policy"):
                self.run_main(path, "pull_request", {}, "true")
            self.assertFalse((path / "output").exists())

    def test_main_rejects_unrequested_manual_publication(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "does not match event policy"),
        ):
            self.run_main(
                Path(directory),
                "workflow_dispatch",
                {"inputs": {"publish": "true"}},
                "false",
            )


if __name__ == "__main__":
    unittest.main()
