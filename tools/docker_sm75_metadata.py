# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-safe GHCR metadata for the SM75 workflow (standard library only)."""

import json
import os
import re
from pathlib import Path
from typing import Any

DOCKER_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
# Docker repository components, including repeated hyphens and double underscores.
REPOSITORY = re.compile(
    r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*/"
    r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
)


def build_metadata(
    *,
    event_name: str,
    event: dict[str, Any],
    repository: str,
    ref: str,
    sha: str,
    run_id: str,
    run_attempt: str,
) -> dict[str, str]:
    """Return single-line Actions outputs; never normalize a release tag.

    Event data, rather than a caller's requested permission, decides publication.
    In particular, PR payloads and non-default branch pushes cannot publish.
    """
    repository = repository.lower()
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be a valid owner/name image path")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("source SHA must be a full lowercase Git commit SHA")
    if not all(re.fullmatch(r"[1-9][0-9]*", value) for value in (run_id, run_attempt)):
        raise ValueError("run ID and attempt must be positive integers")

    image = f"ghcr.io/{repository}"
    tags: list[str] = []
    version = f"sha-{sha}"
    default_branch = event.get("repository", {}).get("default_branch")
    if (
        event_name == "push"
        and default_branch
        and ref == f"refs/heads/{default_branch}"
    ):
        tags = ["main", version]
    elif event_name == "release" and event.get("action") == "published":
        release = event["release"]
        tag = release["tag_name"]
        if not isinstance(tag, str) or not DOCKER_TAG.fullmatch(tag):
            raise ValueError("release tag is not a literal valid Docker tag")
        if not isinstance(release.get("prerelease"), bool):
            raise ValueError("release prerelease flag must be a boolean")
        if release.get("draft", False):
            raise ValueError("a draft release cannot publish an image")
        if release["prerelease"] and tag == "latest":
            raise ValueError("a prerelease cannot overwrite the latest alias")
        tags = [tag]
        version = tag
        # GitHub's flag is authoritative: v0.2.2-post3 is not a prerelease.
        if not release["prerelease"] and tag != "latest":
            tags.append("latest")
    elif event_name == "workflow_dispatch":
        publish = event.get("inputs", {}).get("publish", False)
        if publish is True or publish == "true":
            tags = [f"test-{sha[:12]}-{run_id}-{run_attempt}"]

    return {
        "image": image,
        "local_image": f"sm75-ci:{run_id}-{run_attempt}",
        "publish": str(bool(tags)).lower(),
        "tags": " ".join(f"{image}:{tag}" for tag in tags),
        "version": version,
    }


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text("utf-8"))
    metadata = build_metadata(
        event_name=os.environ["GITHUB_EVENT_NAME"],
        event=event,
        repository=os.environ["GITHUB_REPOSITORY"],
        ref=os.environ["GITHUB_REF"],
        sha=os.environ["GITHUB_SHA"],
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
    )
    # Fail closed if the privileged/unprivileged caller ever drifts from policy.
    if os.environ["REQUEST_PUBLISH"] != metadata["publish"]:
        raise ValueError("caller publish request does not match event policy")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for key, value in metadata.items():
            output.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
