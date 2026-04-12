#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from repo_ops import AUTO_TOPICS_END, AUTO_TOPICS_SECTION, AUTO_TOPICS_START, HOME_INDEX, TOPICS, TOPICS_INDEX, ensure_marked_section, load_title


def build_home_block() -> str:
    entries: list[str] = []
    for topic_dir in sorted(path for path in TOPICS.iterdir() if path.is_dir()):
        title = load_title(topic_dir / "index.md", topic_dir.name.replace("_", " ").title())
        entries.append(f"- [{title}](./topics/{topic_dir.name}/)")
    return "\n".join(entries)


def build_topics_block() -> str:
    entries: list[str] = []
    for topic_dir in sorted(path for path in TOPICS.iterdir() if path.is_dir()):
        title = load_title(topic_dir / "index.md", topic_dir.name.replace("_", " ").title())
        entries.append(f"- [{title}](./{topic_dir.name}/)")
    return "\n".join(entries)


def main() -> int:
    ensure_marked_section(HOME_INDEX, AUTO_TOPICS_SECTION, AUTO_TOPICS_START, AUTO_TOPICS_END, build_home_block())
    ensure_marked_section(TOPICS_INDEX, AUTO_TOPICS_SECTION, AUTO_TOPICS_START, AUTO_TOPICS_END, build_topics_block())
    print(HOME_INDEX)
    print(TOPICS_INDEX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
