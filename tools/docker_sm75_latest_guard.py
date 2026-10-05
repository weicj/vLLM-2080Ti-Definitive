# SPDX-License-Identifier: Apache-2.0
"""Decide whether a release run may move the shared ``latest`` tag.

``latest`` is shared by every stable release, so two releases building at the
same time would otherwise promote it in completion order: if the older release
finishes last, ``latest`` ends up pointing at it. Before moving ``latest``,
confirm that this run's release is still the newest published non-prerelease
release. The check fails closed, so an API problem leaves ``latest`` alone
rather than moving it blindly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API_ROOT = "https://api.github.com"


def newest_stable_tag(releases: list[dict]) -> str | None:
    """Return the newest published, non-prerelease tag, or None."""
    for release in releases:
        if not isinstance(release, dict):
            continue
        if release.get("draft") or release.get("prerelease"):
            continue
        tag = release.get("tag_name")
        if isinstance(tag, str) and tag:
            return tag
    return None


def fetch_releases(repository: str, token: str, *, timeout: float = 30) -> list[dict]:
    """List releases, newest first, as the GitHub API returns them."""
    request = urllib.request.Request(
        f"{API_ROOT}/repos/{repository}/releases?per_page=50",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "docker-sm75-latest-guard",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("unexpected releases response")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="owner/name")
    parser.add_argument("--tag", required=True, help="release tag of this run")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        print("no API token available; leaving latest unchanged", file=sys.stderr)
        return 1

    try:
        releases = fetch_releases(args.repository, token)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(
            f"cannot verify the newest release ({exc}); leaving latest unchanged",
            file=sys.stderr,
        )
        return 1

    newest = newest_stable_tag(releases)
    if newest == args.tag:
        print(f"{args.tag} is the newest published release")
        return 0

    print(f"{args.tag} is not the newest published release (newest: {newest})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
