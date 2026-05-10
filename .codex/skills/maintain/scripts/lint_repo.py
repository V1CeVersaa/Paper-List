#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

from repo_ops import CONTENT, TOPICS, TOPICS_INDEX, HOME_INDEX, git_tracked, parse_frontmatter, read_text


def parse_inline_tags(raw: str) -> list[str] | None:
    raw = raw.strip()
    if not raw:
        return []
    if not raw.startswith("["):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [str(tag).strip() for tag in parsed if str(tag).strip()]


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
        _structural = {"index.md", "overview.md", "landscape.md"}
        # Require explicit visibility on non-structural pages.
        # quartz/plugins/filters/draft.ts publishes by default when visibility is
        # absent (fail-open), so omitting it silently exposes partially migrated content.
        if path.name not in _structural and not visibility:
            issues.append(
                f"{path.relative_to(CONTENT.parent)} missing frontmatter key: visibility"
            )
        # Flag private visibility in ANY tracked public content path, not just content/topics.
        # content/syntheses/, content/conferences/, and future roots all carry the same risk.
        # (content/local/ and content/drafts/ were already excluded at line 12.)
        if visibility in {"draft", "local"}:
            issues.append(
                f"{path.relative_to(CONTENT.parent)} uses visibility={visibility} in a tracked public path"
            )
        parsed_tags = parse_inline_tags(frontmatter.get("tags", ""))
        if parsed_tags is not None and len(parsed_tags) > 3:
            issues.append(
                f"{path.relative_to(CONTENT.parent)} has too many tags ({len(parsed_tags)} > 3)"
            )
        topic = frontmatter.get("topic", "").strip()
        if topic and parsed_tags and len(parsed_tags) > 1 and parsed_tags[0] == topic:
            issues.append(
                f"{path.relative_to(CONTENT.parent)} uses topic={topic} as the primary tag despite having a more specific tag"
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
            if Path(entry).name == ".gitkeep":
                continue
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
