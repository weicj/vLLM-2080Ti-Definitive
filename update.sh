#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"

REMOTE_REPO=${REMOTE_REPO:-https://github.com/weicj/vLLM-2080Ti-Definitive.git}
REMOTE_RELEASE_API=${REMOTE_RELEASE_API:-https://api.github.com/repos/weicj/vLLM-2080Ti-Definitive/releases?per_page=100}
UPDATE_PROMPT_TIMEOUT_SECONDS=${UPDATE_PROMPT_TIMEOUT_SECONDS:-10}
UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS=${UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS:-10}
UPDATE_DOWNLOAD_TIMEOUT_SECONDS=${UPDATE_DOWNLOAD_TIMEOUT_SECONDS:-60}

# Runtime state and site-specific profiles survive a release refresh.
PRESERVE_PATHS=(
  ".cache"
  ".venv"
  ".deps"
  "build-logs"
  "run-logs"
  "results"
  "torchinductor-cache"
  "triton-cache"
  ".omx"
  "backups"
  "profiles/local"
)

prompt_yes_no_timeout() {
  local prompt=$1 timeout_seconds=${2:-10} default_answer=${3:-y} answer=""
  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" || ! -t 0 ]]; then
    printf '%s\n' "$default_answer"
    return 0
  fi
  printf '%s ' "$prompt" >&2
  if read -r -t "$timeout_seconds" answer; then
    case "$answer" in
      n|N|no|NO) printf 'n\n' ;;
      *) printf '%s\n' "$default_answer" ;;
    esac
  else
    printf '%s\n' "$default_answer"
  fi
}

latest_remote_version() {
  python3 - "$REMOTE_RELEASE_API" <<'PY'
import json
import re
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=10) as response:
        releases = json.load(response)
except (OSError, ValueError) as exc:
    raise SystemExit(f"Cannot check releases: {exc}") from exc

if isinstance(releases, dict):
    releases = [releases]
versions = []
for release in releases:
    tag = release.get("tag_name", "").strip()
    match = re.fullmatch(r"v0\.3\.(\d+)(?:[.-]post(\d+))?", tag)
    if match and not release.get("draft") and not release.get("prerelease"):
        patch, post = match.groups()
        versions.append((int(patch), int(post) if post is not None else -1, tag[1:]))
if versions:
    print(max(versions)[2])
PY
}

read_local_version() {
  tr -d '[:space:]' < "$ROOT/VERSION"
}

is_remote_newer() {
  local local_version=$1 remote_version=$2
  if [[ "${local_version%.dev*}" == "$remote_version" && "$local_version" == *.dev* ]]; then
    return 0
  fi
  local_version=${local_version/-post/.post}
  remote_version=${remote_version/-post/.post}
  [[ -n "$remote_version" && "$remote_version" != "$local_version" ]] || return 1
  [[ "$(printf '%s\n%s\n' "$local_version" "$remote_version" | sort -V | tail -n 1)" == "$remote_version" ]]
}

download_archive() (
  local version=$1 tmpdir archive_path strip_root
  tmpdir=$(mktemp -d)
  trap "rm -rf -- $(printf '%q' "$tmpdir")" EXIT
  archive_path="$tmpdir/src.tar.gz"
  strip_root="$tmpdir/source"
  mkdir -p "$strip_root"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --max-time "$UPDATE_DOWNLOAD_TIMEOUT_SECONDS" \
      -o "$archive_path" "${REMOTE_REPO%.git}/archive/refs/tags/v${version}.tar.gz"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive_path" --timeout="$UPDATE_DOWNLOAD_TIMEOUT_SECONDS" \
      "${REMOTE_REPO%.git}/archive/refs/tags/v${version}.tar.gz"
  else
    echo "Neither curl nor wget was found." >&2
    return 1
  fi
  tar -xzf "$archive_path" -C "$strip_root" --strip-components=1
  if [[ ! -f "$strip_root/VERSION" || "$(tr -d '[:space:]' <"$strip_root/VERSION")" != "$version" || ! -f "$strip_root/build.sh" ]]; then
    echo "Release archive contents do not match v$version." >&2
    return 1
  fi

  local path
  # Overlay release files. Keep local checkpoints and runtime state in place,
  # including large caches and environments.
  for path in "${PRESERVE_PATHS[@]}"; do
    rm -rf -- "$strip_root/$path"
  done
  cp -a "$strip_root"/. "$ROOT"/

  # The image build applies every patch in this directory. Drop patches the
  # new release removed so they cannot be replayed on a newer upstream image.
  for path in "$ROOT"/docker/sm75/patches/*.patch; do
    [[ -f "$path" ]] || continue
    [[ -f "$strip_root/${path#"$ROOT"/}" ]] || rm -- "$path"
  done
)

main() {
  echo "$PROJECT_NAME v$FORK_RELEASE update helper"
  local local_version remote_version before after
  if [[ -e "$ROOT/.git" ]]; then
    if ! git -C "$ROOT" rev-parse --abbrev-ref '@{upstream}' >/dev/null 2>&1; then
      echo "This checkout needs a tracking branch for ./update.sh." >&2
      return 1
    fi
    if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then
      echo "Commit or stash tracked changes before updating this checkout." >&2
      return 1
    fi
    before=$(git -C "$ROOT" rev-parse HEAD)
    [[ "$(prompt_yes_no_timeout "Update from the tracking branch? [Y/n]:" "$UPDATE_PROMPT_TIMEOUT_SECONDS" y)" != n ]] || return 0
    git -C "$ROOT" pull --ff-only
    after=$(git -C "$ROOT" rev-parse HEAD)
    [[ "$before" != "$after" ]] || return 0
    if [[ "$(prompt_yes_no_timeout "Build now? [Y/n]:" "$UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS" y)" != n ]]; then
      exec bash "$ROOT/build.sh"
    fi
    return 0
  fi
  local_version=$(read_local_version)
  remote_version=$(latest_remote_version) || return 1
  echo "Local version:  ${local_version:-unknown}"
  echo "Remote version: ${remote_version:-unavailable}"

  if [[ -z "$remote_version" ]]; then
    echo "No stable 0.3.x release is available."
    return 0
  fi

  if ! is_remote_newer "$local_version" "$remote_version"; then
    echo "Already up to date."
    return 0
  fi

  echo "New release available: v$remote_version"
  [[ "$(prompt_yes_no_timeout "Update now? [Y/n]:" "$UPDATE_PROMPT_TIMEOUT_SECONDS" y)" != n ]] || return 0
  echo "Updating from release archive v$remote_version..."
  download_archive "$remote_version"
  echo "Update complete. Current version: $(read_local_version)"

  if [[ "$(prompt_yes_no_timeout "Build now? [Y/n]:" "$UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS" y)" != n ]]; then
    exec bash "$ROOT/build.sh"
  fi
}

main "$@"
