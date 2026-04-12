#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

SUPPORT = Path(__file__).resolve().parents[2] / "maintain" / "scripts"
sys.path.insert(0, str(SUPPORT))

from repo_ops import ROOT, TOPICS, append_log, ensure_overview_entry, today, update_frontmatter


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

    # Half-promote recovery: source is gone but target already exists, meaning a
    # previous run moved the file but failed before finishing metadata sync.
    # Resume from the metadata step instead of failing.
    if not source.exists() and args.target:
        target = resolve(args.target)
        if target.exists():
            print(
                f"[promote] resuming half-promote: source gone, target exists at "
                f"{target.relative_to(ROOT).as_posix()} — syncing metadata only",
                file=sys.stderr,
            )
            final_path = target
            args.target = ""  # skip the move block below
        else:
            raise SystemExit(f"source note does not exist: {source}")
    elif not source.exists():
        raise SystemExit(f"source note does not exist: {source}")
    else:
        final_path = source

    # Preflight: verify overview.md exists before the irreversible move.
    # This catches the most common post-move failure mode early.
    overview_path = TOPICS / args.topic / "overview.md"
    if not overview_path.exists():
        raise SystemExit(
            f"overview.md not found for topic '{args.topic}': {overview_path} — "
            f"create the topic with /explore first"
        )

    if args.target:
        target = resolve(args.target)
        if target.exists() and not args.force:
            raise SystemExit(
                f"target already exists: {target.relative_to(ROOT).as_posix()} — use --force to overwrite"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, target)
        final_path = target

    # Topological guard: if the file was moved or recovered to a target path,
    # verify it lives under content/topics/<topic>/ to prevent cross-topic
    # metadata contamination from misrouted or retried promote calls.
    if final_path != source:
        topic_dir = TOPICS / args.topic
        try:
            final_path.relative_to(topic_dir)
        except ValueError:
            raise SystemExit(
                f"target {final_path.relative_to(ROOT).as_posix()!r} is not under "
                f"content/topics/{args.topic}/ — verify --target and --topic match"
            )

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
