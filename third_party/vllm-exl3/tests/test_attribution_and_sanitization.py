"""Unit tests asserting full upstream attribution and strict hardware sanitization."""

import os
import re
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_internal_hostnames_or_usernames():
    """Ensure zero occurrences of forbidden hostnames in documentation or source files."""
    # Split literals so this audit does not flag its own source file.
    forbidden = ["cruz" + "-spark", "we" + "sche" + "-spark", "9f" + "73", "mark" + "us"]
    text_extensions = {".md", ".py", ".cu", ".cuh", ".cpp", ".toml", ".cff"}

    violations = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip git and cache directories
        if ".git" in root or "__pycache__" in root or ".pytest_cache" in root or ".agent-sync" in root or "dist" in root or "build" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in text_extensions or file in {"NOTICE", "LICENSE", "CITATION"}:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    for term in forbidden:
                        if term in text:
                            violations.append(f"{term} found in {os.path.relpath(file_path, REPO_ROOT)}")
                except Exception:
                    pass

    assert not violations, f"Forbidden internal terms detected:\n" + "\n".join(violations)


def test_upstream_attribution_present():
    """Verify that Mia's AI Lab and ExLlamaV3 (turboderp) are prominently credited."""
    readme_path = os.path.join(REPO_ROOT, "README.md")
    assert os.path.exists(readme_path), "README.md missing"
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()

    assert "Mia's AI Lab" in readme, "Mia's AI Lab attribution missing in README.md"
    assert "turboderp" in readme or "ExLlamaV3" in readme, "ExLlamaV3 attribution missing in README.md"

    notices_path = os.path.join(REPO_ROOT, "THIRD_PARTY_NOTICES.md")
    assert os.path.exists(notices_path), "THIRD_PARTY_NOTICES.md missing"
    with open(notices_path, "r", encoding="utf-8") as f:
        notices = f.read()

    assert "Mia's AI Lab" in notices, "Mia's AI Lab attribution missing in THIRD_PARTY_NOTICES.md"
    assert "Turboderp" in notices or "turboderp" in notices, "Turboderp attribution missing in THIRD_PARTY_NOTICES.md"


def test_package_version_and_metadata():
    """Verify version 0.4.0 in pyproject.toml and setup.py."""
    pyproject_path = os.path.join(REPO_ROOT, "pyproject.toml")
    with open(pyproject_path, "r", encoding="utf-8") as f:
        pyproject = f.read()

    assert 'version = "0.4.0"' in pyproject, "pyproject.toml version is not 0.4.0"
    assert 'name = "vllm-exl3"' in pyproject, "pyproject.toml name is not vllm-exl3"

    setup_path = os.path.join(REPO_ROOT, "setup.py")
    with open(setup_path, "r", encoding="utf-8") as f:
        setup_content = f.read()
    assert 'name="vllm-exl3"' in setup_content, "setup.py name does not match vllm-exl3"


def test_canonical_turboderp_org_urls():
    """Verify that all ExLlamaV3 repository URLs use the canonical turboderp-org namespace."""
    bad_pattern = re.compile(r"https?://github\.com/turboderp/exllamav3[^\s\)\"\`\>]*")
    violations = []
    text_extensions = {".md", ".py", ".toml", ".cff"}

    for root, dirs, files in os.walk(REPO_ROOT):
        if ".git" in root or "__pycache__" in root or ".pytest_cache" in root or "dist" in root or "build" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in text_extensions or file in {"NOTICE", "LICENSE", "CITATION"}:
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    matches = bad_pattern.findall(text)
                    if matches:
                        violations.append(f"{os.path.relpath(path, REPO_ROOT)}: {matches}")
                except Exception:
                    pass

    assert not violations, f"Legacy turboderp/exllamav3 URLs found (must use turboderp-org):\n" + "\n".join(violations)


def test_local_markdown_links_resolve():
    """Verify all relative markdown links in README.md and documentation point to existing files."""
    for md_file in ["README.md", "THIRD_PARTY_NOTICES.md", "ACKNOWLEDGMENTS.md", "AGENTS.md"]:
        path = os.path.join(REPO_ROOT, md_file)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        content = re.sub(r"`[^`]*`", "", content)
        local_links = re.findall(r"(?<!!)\[[^]]*\]\((?!https?://)([^)]*)\)", content)
        for link in local_links:
            clean = link.split("#")[0].strip()
            if clean:
                target = os.path.join(REPO_ROOT, clean)
                assert os.path.exists(target), f"Broken link in {md_file}: {link} -> {clean}"
