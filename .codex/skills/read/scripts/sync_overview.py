#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SUPPORT = Path(__file__).resolve().parents[2] / "maintain" / "scripts"
sys.path.insert(0, str(SUPPORT))

from repo_ops import ROOT, ensure_overview_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure a paper entry exists in a topic overview.")
    parser.add_argument("title")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--venue", default="")
    parser.add_argument("--paper-url", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--checked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    note_path = (ROOT / args.note).resolve() if args.note else None
    ensure_overview_entry(
        topic=args.topic,
        title=args.title,
        venue=args.venue,
        paper_url=args.paper_url,
        note_path=note_path,
        checked=args.checked,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
