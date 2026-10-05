# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[2] / "tools" / "docker_sm75_latest_guard.py"
SPEC = importlib.util.spec_from_file_location("docker_sm75_latest_guard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def release(tag, *, draft=False, prerelease=False):
    return {"tag_name": tag, "draft": draft, "prerelease": prerelease}


class TestNewestStableTag(unittest.TestCase):
    def test_skips_drafts_and_prereleases(self):
        releases = [
            release("v3", prerelease=True),
            release("v2-rc", draft=True),
            release("v2"),
            release("v1"),
        ]
        self.assertEqual(GUARD.newest_stable_tag(releases), "v2")

    def test_returns_none_when_no_stable_release_exists(self):
        self.assertIsNone(GUARD.newest_stable_tag([release("v1", prerelease=True)]))
        self.assertIsNone(GUARD.newest_stable_tag([]))

    def test_ignores_malformed_entries(self):
        releases = [{"tag_name": None}, {"tag_name": ""}, "not-a-dict", release("v1")]
        self.assertEqual(GUARD.newest_stable_tag(releases), "v1")

    def test_post_release_is_not_treated_as_prerelease(self):
        # GitHub's own flag decides stability, so a "post" tag counts as stable.
        releases = [release("v0.2.2-post3"), release("v0.2.1")]
        self.assertEqual(GUARD.newest_stable_tag(releases), "v0.2.2-post3")


class TestMain(unittest.TestCase):
    def run_main(self, releases, *, tag="v2", token="token"):
        environment = {"GITHUB_TOKEN": token} if token else {}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(GUARD, "fetch_releases", return_value=releases or []),
        ):
            return GUARD.main(["--repository", "o/r", "--tag", tag])

    def test_allows_the_newest_release(self):
        self.assertEqual(self.run_main([release("v2"), release("v1")]), 0)

    def test_refuses_an_older_release(self):
        self.assertEqual(self.run_main([release("v3"), release("v2")], tag="v2"), 1)

    def test_refuses_without_a_token(self):
        self.assertEqual(self.run_main([release("v2")], token=""), 1)

    def test_fails_closed_when_the_api_call_fails(self):
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "token"}, clear=True),
            patch.object(
                GUARD,
                "fetch_releases",
                side_effect=urllib.error.URLError("offline"),
            ),
        ):
            self.assertEqual(GUARD.main(["--repository", "o/r", "--tag", "v2"]), 1)

    def test_fails_closed_on_a_malformed_response(self):
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "token"}, clear=True),
            patch.object(GUARD, "fetch_releases", return_value=[{"unexpected": 1}]),
        ):
            self.assertEqual(GUARD.main(["--repository", "o/r", "--tag", "v2"]), 1)


if __name__ == "__main__":
    unittest.main()
