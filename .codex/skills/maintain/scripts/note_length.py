#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|\Z)", re.DOTALL)
FENCED_CODE_RE = re.compile(r"(?ms)^(```|~~~).*?^\1\s*$")
HTML_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")

HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x30000, 0x3134F),
    (0x31350, 0x323AF),
)


@dataclass(frozen=True)
class CountResult:
    path: Path
    hanzi: int
    threshold: int | None

    @property
    def delta(self) -> int | None:
        if self.threshold is None:
            return None
        return self.hanzi - self.threshold

    @property
    def status(self) -> str:
        if self.threshold is None:
            return "counted"
        return "ok" if self.hanzi >= self.threshold else "below-threshold"


def is_han(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in HAN_RANGES)


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


def visible_note_text(text: str, *, include_frontmatter: bool, include_code: bool) -> str:
    if not include_frontmatter:
        text = strip_frontmatter(text)
    if not include_code:
        text = FENCED_CODE_RE.sub("", text)
    return HTML_COMMENT_RE.sub("", text)


def count_hanzi(path: Path, *, include_frontmatter: bool, include_code: bool) -> int:
    text = path.read_text(encoding="utf-8")
    text = visible_note_text(
        text,
        include_frontmatter=include_frontmatter,
        include_code=include_code,
    )
    return sum(1 for char in text if is_han(char))


def expand_inputs(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.md")))
        else:
            expanded.append(path)
    return expanded


def resolve_threshold(
    args: argparse.Namespace,
    *,
    include_frontmatter: bool,
    include_code: bool,
) -> int | None:
    if args.threshold is not None:
        return args.threshold
    if args.baseline is None:
        return None
    baseline_count = count_hanzi(
        Path(args.baseline),
        include_frontmatter=include_frontmatter,
        include_code=include_code,
    )
    return round(baseline_count * args.ratio)


def print_text(results: list[CountResult], *, baseline: str | None, ratio: float) -> None:
    if baseline is not None:
        print(f"Baseline: {baseline} (ratio={ratio:g})")
    path_width = max([len("Path"), *(len(item.path.as_posix()) for item in results)])
    threshold_width = len("Threshold")
    print(
        f"{'Path':<{path_width}}  {'Hanzi':>7}  "
        f"{'Threshold':>{threshold_width}}  {'Delta':>7}  Status"
    )
    print(
        f"{'-' * path_width}  {'-' * 7}  "
        f"{'-' * threshold_width}  {'-' * 7}  {'-' * 16}"
    )
    for item in results:
        threshold = "-" if item.threshold is None else str(item.threshold)
        delta = "-" if item.delta is None else f"{item.delta:+d}"
        print(
            f"{item.path.as_posix():<{path_width}}  {item.hanzi:>7}  "
            f"{threshold:>{threshold_width}}  {delta:>7}  {item.status}"
        )


def print_json(results: list[CountResult], *, baseline: str | None, ratio: float) -> None:
    payload = {
        "baseline": baseline,
        "ratio": ratio,
        "results": [
            {
                "path": item.path.as_posix(),
                "hanzi": item.hanzi,
                "threshold": item.threshold,
                "delta": item.delta,
                "status": item.status,
            }
            for item in results
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count Han ideographs in Markdown note bodies. By default this "
            "excludes YAML frontmatter, fenced code blocks, and HTML comments."
        )
    )
    parser.add_argument("paths", nargs="+", help="Markdown files or directories to count.")
    parser.add_argument("--threshold", type=int, help="Minimum Hanzi count required.")
    parser.add_argument(
        "--baseline",
        help="Reference Markdown file used to derive a threshold when --threshold is absent.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Multiplier applied to --baseline count. Defaults to 1.0.",
    )
    parser.add_argument("--include-frontmatter", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.paths)
    missing = [path for path in paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing path: {path}", file=sys.stderr)
        return 2

    threshold = resolve_threshold(
        args,
        include_frontmatter=args.include_frontmatter,
        include_code=args.include_code,
    )
    results = [
        CountResult(
            path=path,
            hanzi=count_hanzi(
                path,
                include_frontmatter=args.include_frontmatter,
                include_code=args.include_code,
            ),
            threshold=threshold,
        )
        for path in paths
    ]
    results.sort(key=lambda item: item.path.as_posix())

    if args.json:
        print_json(results, baseline=args.baseline, ratio=args.ratio)
    else:
        print_text(results, baseline=args.baseline, ratio=args.ratio)

    if threshold is not None and any(item.hanzi < threshold for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
