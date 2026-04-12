#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

from repo_ops import CONTENT, TOPICS, TOPICS_INDEX, HOME_INDEX, git_tracked, parse_frontmatter, read_text


def check_public_pages(issues: list[str]) -> None:
    for path in CONTENT.rglob("*.md"):
        if any(part in {"local", "drafts"} for part in path.parts):
            continue
        frontmatter, _ = parse_frontmatter(read_text(path))
        if "title" not in frontmatter:
            issues.append(f"{path.relative_to(CONTENT.parent)} missing frontmatter key: title")
        if path.name not in {"index.md", "overview.md", "landscape.md"} and "status" not in frontmatter:
            issues.append(f"{path.relative_to(CONTENT.parent)} missing frontmatter key: status")
        visibility = frontmatter.get("visibility", "")
        # Flag private visibility in ANY tracked public content path, not just content/topics.
        # content/syntheses/, content/conferences/, and future roots all carry the same risk.
        # (content/local/ and content/drafts/ were already excluded at line 12.)
        if visibility in {"draft", "local"}:
            issues.append(
                f"{path.relative_to(CONTENT.parent)} uses visibility={visibility} in a tracked public path"
            )


def check_topics(issues: list[str]) -> None:
    for topic_dir in sorted(path for path in TOPICS.iterdir() if path.is_dir()):
        for required in ("index.md", "overview.md"):
            if not (topic_dir / required).exists():
                issues.append(f"{topic_dir.relative_to(CONTENT.parent)} missing required file: {required}")


def check_tracking(issues: list[str]) -> None:
    for pathspec in ("content/local", "content/drafts", "inbox", "raw", "ops"):
        tracked = git_tracked(pathspec)
        for entry in tracked:
            issues.append(f"git-tracked private path detected: {entry}")


def check_index_markers(issues: list[str]) -> None:
    for path in (HOME_INDEX, TOPICS_INDEX):
        text = read_text(path)
        if "<!-- AUTO:TOPICS:START -->" not in text or "<!-- AUTO:TOPICS:END -->" not in text:
            issues.append(f"{path.relative_to(CONTENT.parent)} is missing auto topic markers")


def main() -> int:
    issues: list[str] = []
    check_public_pages(issues)
    check_topics(issues)
    check_tracking(issues)
    check_index_markers(issues)

    if not issues:
        print("No obvious repository issues found.")
        return 0

    print("Paper-List Lint")
    print("================")
    for issue in issues:
        print(f"- {issue}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
