#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

SUPPORT = Path(__file__).resolve().parents[2] / "maintain" / "scripts"
sys.path.insert(0, str(SUPPORT))

from repo_ops import ROOT, append_log, ensure_overview_entry, today, update_frontmatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a note into public content and synchronize its metadata.")
    parser.add_argument("source", help="Path to the source note, relative to repo root or absolute.")
    parser.add_argument("--target", default="", help="Optional target note path relative to repo root.")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--venue", default="")
    parser.add_argument("--paper-url", default="")
    parser.add_argument("--inbox", default="")
    parser.add_argument("--status", default="complete")
    parser.add_argument("--visibility", default="public")
    parser.add_argument("--force", action="store_true", help="Overwrite target if it already exists.")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    source = resolve(args.source)
    if not source.exists():
        raise SystemExit(f"source note does not exist: {source}")

    final_path = source
    if args.target:
        target = resolve(args.target)
        if target.exists() and not args.force:
            raise SystemExit(
                f"target already exists: {target.relative_to(ROOT).as_posix()} — use --force to overwrite"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)
        final_path = target

    update_frontmatter(
        final_path,
        {
            "status": args.status,
            "visibility": args.visibility,
            "draft": "false",
            "updated": today(),
        },
    )
    ensure_overview_entry(
        topic=args.topic,
        title=args.title,
        venue=args.venue,
        paper_url=args.paper_url,
        note_path=final_path,
        checked=True,
    )

    if args.inbox:
        inbox_path = resolve(args.inbox)
        if inbox_path.exists():
            inbox_path.unlink()

    append_log("promote", args.title, f"path={final_path.relative_to(ROOT).as_posix()} | topic={args.topic}")
    print(final_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
