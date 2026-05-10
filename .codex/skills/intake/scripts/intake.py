#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SUPPORT = Path(__file__).resolve().parents[2] / "maintain" / "scripts"
sys.path.insert(0, str(SUPPORT))

from repo_ops import (
    INBOX,
    ROOT,
    TOPICS,
    append_log,
    archive_pdf,
    ensure_overview_entry,
    extract_canonical_key,
    find_by_canonical_key,
    normalize_paper_url,
    overview_has_entry,
    parse_frontmatter,
    read_text,
    render_template,
    skill_asset,
    slugify,
    today,
    update_frontmatter,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a structured inbox item and reserve its overview position.")
    parser.add_argument("title")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--venue", default="")
    parser.add_argument("--paper-url", default="")
    parser.add_argument("--arxiv", default="", help=argparse.SUPPRESS)
    parser.add_argument("--priority", default="normal", choices=["high", "normal", "low"])
    parser.add_argument(
        "--pdf",
        default="",
        help=(
            "Supported PDF source to archive locally: local .pdf path, direct .pdf URL, "
            "arXiv abs/pdf URL, or OpenReview forum/pdf URL."
        ),
    )
    parser.add_argument("--why", default="TODO")
    parser.add_argument("--positioning", default="TODO")
    parser.add_argument("--slug", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topic_dir = TOPICS / args.topic
    if not topic_dir.exists():
        raise SystemExit(f"topic does not exist: {args.topic}. Create it first with /explore.")

    slug = args.slug or slugify(args.title)
    inbox_path = INBOX / f"{slug}.md"
    paper_url = normalize_paper_url(args.paper_url or args.arxiv)

    if not args.force:
        # --- Tier 1: canonical key dedup (paper_url / arXiv ID / OpenReview ID) ---
        canonical_key = extract_canonical_key(paper_url) if paper_url else ""
        existing_inbox, existing_note = (
            find_by_canonical_key(canonical_key, args.topic)
            if canonical_key
            else (None, None)
        )

        # --- Tier 2: slug dedup (fallback when no canonical key or no tier-1 hit) ---
        if existing_inbox is None and inbox_path.exists():
            existing_inbox = inbox_path

        # --- Decide action ---

        # Case A: a finished note already exists → nothing to do
        if existing_note is not None:
            print(
                f"note already exists at {existing_note.relative_to(ROOT).as_posix()} "
                f"— skipping intake. Run /refresh if the note needs updating."
            )
            return 0

        # Case B: inbox item exists → check if overview entry is also present
        if existing_inbox is not None:
            existing_slug = existing_inbox.stem
            existing_frontmatter, _ = parse_frontmatter(read_text(existing_inbox))
            updates: dict[str, str] = {}
            existing_source_pdf = existing_frontmatter.get("source_pdf", "")

            if paper_url and not existing_frontmatter.get("paper_url"):
                updates["paper_url"] = paper_url
            if not existing_source_pdf:
                existing_source_pdf = archive_pdf(args.pdf, existing_slug, paper_url=paper_url, force=args.force)
                if existing_source_pdf:
                    updates["source_pdf"] = existing_source_pdf
            if updates:
                update_frontmatter(existing_inbox, updates)

            if overview_has_entry(topic=args.topic, title=args.title, paper_url=paper_url):
                # Full duplicate: both inbox and overview exist → skip entirely
                print(
                    f"duplicate: inbox={existing_inbox.relative_to(ROOT).as_posix()}, "
                    f"overview entry present — skipping."
                )
                return 0
            # Inbox exists but overview entry is missing → add only the overview entry
            print(
                f"inbox item exists at {existing_inbox.relative_to(ROOT).as_posix()}, "
                f"overview entry missing — adding overview entry only."
            )
            ensure_overview_entry(
                topic=args.topic,
                title=args.title,
                venue=args.venue,
                paper_url=paper_url,
                checked=False,
            )
            append_log(
                "intake",
                args.title,
                f"topic={args.topic} | overview-only (inbox={existing_inbox.relative_to(ROOT).as_posix()})",
            )
            return 0

    # --- No duplicate found (or --force): full intake ---
    source_pdf = archive_pdf(args.pdf, slug, paper_url=paper_url, force=args.force)
    rendered = render_template(
        skill_asset("intake", "inbox-item.md"),
        {
            "title": args.title,
            "venue": args.venue,
            "paper_url": paper_url,
            "topic": args.topic,
            "priority": args.priority,
            "created": today(),
            "source_pdf": source_pdf,
            "why": args.why,
            "positioning": args.positioning,
        },
    )
    write_text(inbox_path, rendered)
    ensure_overview_entry(
        topic=args.topic,
        title=args.title,
        venue=args.venue,
        paper_url=paper_url,
        checked=False,
    )
    append_log("intake", args.title, f"topic={args.topic} | inbox={inbox_path.relative_to(ROOT).as_posix()}")
    print(inbox_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
