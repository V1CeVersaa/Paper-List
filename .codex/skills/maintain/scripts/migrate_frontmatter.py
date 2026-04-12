#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from repo_ops import parse_frontmatter, quote, read_text, update_frontmatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill frontmatter metadata for existing Paper-List content."
    )
    parser.add_argument(
        "--path",
        default="content/topics",
        help="Directory or markdown file to scan relative to repo root.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the proposed changes. Default is dry-run.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only fill fields that are currently missing.",
    )
    return parser.parse_args()


def iter_markdown_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(target.rglob("*.md"))


def infer_visibility(path: Path) -> str:
    parts = set(path.parts)
    if "drafts" in parts:
        return "draft"
    if "local" in parts:
        return "local"
    return "public"


def infer_status(path: Path, body: str, visibility: str) -> str:
    if visibility == "draft":
        return "drafting"
    if "TODO" in body or "逐段翻译" in body:
        return "drafting"
    if path.name in {"index.md", "overview.md"}:
        return "complete"
    if path.name == "landscape.md":
        return "complete" if "TODO" not in body else "drafting"
    return "complete"


def infer_description(path: Path, frontmatter: dict[str, str]) -> str:
    title = frontmatter.get("title", path.stem.replace("_", " ").replace("-", " ").title())
    headline = frontmatter.get("headline", "")

    if path.name == "overview.md":
        if headline.startswith("Overview of "):
            subject = headline.removeprefix("Overview of ").strip()
            return f"Queue and reading progress for {subject}."
        return f"Queue and reading progress for {title}."

    if path.name == "landscape.md":
        if headline.startswith("Landscape of "):
            subject = headline.removeprefix("Landscape of ").strip()
            return f"Topic map, methodology spectrum, and reading roadmap for {subject}."
        return f"Topic map and reading roadmap for {title}."

    if path.name == "index.md":
        if path.parent.name == "topics":
            return "Topic index for the Paper-List repository."
        return f"Landing page for {title}."

    if headline:
        return f"Paper note on {headline}."
    return f"Paper note on {title}."


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[4]
    target = Path(args.path)
    target = target if target.is_absolute() else root / target
    paths = iter_markdown_paths(target)

    if not paths:
        raise SystemExit(f"no markdown files found under {target}")

    proposed_count = 0
    written_count = 0

    for path in paths:
        text = read_text(path)
        frontmatter, body = parse_frontmatter(text)
        if not frontmatter:
            continue

        updates: dict[str, str] = {}
        visibility = frontmatter.get("visibility") or infer_visibility(path)
        status = frontmatter.get("status") or infer_status(path, body, visibility)
        description = frontmatter.get("description") or infer_description(path, frontmatter)

        if not args.only_missing or "visibility" not in frontmatter:
            updates["visibility"] = visibility
        if not args.only_missing or "status" not in frontmatter:
            updates["status"] = status
        if not args.only_missing or "description" not in frontmatter:
            updates["description"] = description

        if not updates:
            continue

        proposed_count += 1
        rel = path.relative_to(root).as_posix()
        print(rel)
        for key, value in updates.items():
            print(f"  + {key}: {quote(value)}")

        if args.write:
            update_frontmatter(path, updates)
            written_count += 1

    mode = "write" if args.write else "dry-run"
    print(f"\nmode={mode} proposed={proposed_count} written={written_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
