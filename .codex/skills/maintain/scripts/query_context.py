#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

from repo_ops import CONTENT, parse_frontmatter, read_text


# Matches ASCII tokens or CJK character sequences (Chinese, Japanese, Korean).
# CJK sequences are kept intact so substring matching in score() works correctly.
WORD_RE = re.compile(r"[a-z0-9_+-]+|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
HEADER_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass
class Page:
    path: Path
    title: str
    description: str
    status: str
    visibility: str
    headings: list[str]
    text: str


def tokenize(text: str) -> list[str]:
    return [token for token in WORD_RE.findall(text.lower()) if token]


def load_pages(include_private: bool = False) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(CONTENT.rglob("*.md")):
        if not include_private and any(part in {"local", "drafts"} for part in path.parts):
            continue
        text = read_text(path)
        frontmatter, body = parse_frontmatter(text)
        visibility = frontmatter.get("visibility", "public")
        if not include_private and visibility in {"local", "draft"}:
            continue
        pages.append(
            Page(
                path=path,
                title=frontmatter.get("title", path.stem),
                description=frontmatter.get("description", ""),
                status=frontmatter.get("status", ""),
                visibility=visibility,
                headings=HEADER_RE.findall(body),
                text=body,
            )
        )
    return pages


def score(page: Page, query_tokens: list[str]) -> int:
    if not query_tokens:
        return 0
    haystack = page.text.lower()
    title = page.title.lower()
    description = page.description.lower()
    headings = "\n".join(page.headings).lower()
    value = 0
    for token in query_tokens:
        if token in title:
            value += 20
        if token in description:
            value += 8
        if token in headings:
            value += 6
        value += haystack.count(token)
    if page.status == "complete":
        value += 2
    return value


def best_snippet(page: Page, query_tokens: list[str]) -> str:
    lines = [
        line.strip()
        for line in page.text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for token in query_tokens:
        for line in lines:
            if token in line.lower():
                return line
    return page.description or (lines[0] if lines else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank relevant Paper-List pages for a query and print a compact context pack."
    )
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--include-private", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_tokens = tokenize(args.query)
    pages = load_pages(include_private=args.include_private)
    ranked = []
    for page in pages:
        value = score(page, query_tokens)
        if value > 0:
            ranked.append((value, page))
    ranked.sort(key=lambda item: (-item[0], item[1].path.as_posix()))

    print("Paper-List Query Pack")
    print("=====================")
    print(f"Query: {args.query}")
    if args.include_private:
        print("Mode: include private pages")
    print("")

    if not ranked:
        print("No matching pages found.")
        return 0

    print("Suggested read order:")
    for index, (value, page) in enumerate(ranked[: args.limit], start=1):
        print(f"{index}. {page.path.relative_to(CONTENT.parent).as_posix()} | score={value}")
        print(f"   title: {page.title}")
        if page.description:
            print(f"   description: {page.description}")
        if page.headings:
            print(f"   headings: {', '.join(page.headings[:6])}")
        snippet = best_snippet(page, query_tokens)
        if snippet:
            print(f"   snippet: {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
