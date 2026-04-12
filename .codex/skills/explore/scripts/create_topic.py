#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SUPPORT = Path(__file__).resolve().parents[2] / "maintain" / "scripts"
sys.path.insert(0, str(SUPPORT))

from repo_ops import TOPICS, append_log, render_template, skill_asset, slugify, today, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new topic scaffold for /explore.")
    parser.add_argument("title", help="Human-readable topic title, e.g. 'World Models'")
    parser.add_argument("--slug", default="", help="Optional explicit topic slug")
    parser.add_argument(
        "--description",
        default="",
        help="Short topic description for the topic landing page",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing scaffold files")
    return parser.parse_args()


def topic_slug(value: str) -> str:
    return slugify(value).replace("-", "_")


def ensure_write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    write_text(path, content)


def main() -> int:
    args = parse_args()
    slug = args.slug or topic_slug(args.title)
    topic_dir = TOPICS / slug
    topic_dir.mkdir(parents=True, exist_ok=True)

    description = args.description or f"Topic landing page for {args.title}."
    replacements = {
        "title": args.title,
        "description": description,
        "topic": slug,
        "created": today(),
    }

    ensure_write(topic_dir / "index.md", render_template(skill_asset("explore", "topic-index.md"), replacements), args.force)
    ensure_write(topic_dir / "overview.md", render_template(skill_asset("explore", "overview.md"), replacements), args.force)
    ensure_write(topic_dir / "landscape.md", render_template(skill_asset("explore", "landscape.md"), replacements), args.force)

    append_log("explore", args.title, f"topic={slug} | scaffolded")
    print(topic_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
