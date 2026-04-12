#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlparse, urlunparse
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[4]
CONTENT = ROOT / "content"
TOPICS = CONTENT / "topics"
INBOX = ROOT / "inbox"
RAW_PAPERS = ROOT / "raw" / "papers"
OPS = ROOT / "ops"
LOG = OPS / "log.md"
HOME_INDEX = CONTENT / "index.md"
TOPICS_INDEX = TOPICS / "index.md"
SKILLS = ROOT / ".codex" / "skills"
AUTO_TOPICS_START = "<!-- AUTO:TOPICS:START -->"
AUTO_TOPICS_END = "<!-- AUTO:TOPICS:END -->"
AUTO_TOPICS_SECTION = "## Auto Topic Index"
AUTO_QUEUE_START = "<!-- AUTO:QUEUE:START -->"
AUTO_QUEUE_END = "<!-- AUTO:QUEUE:END -->"
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def today() -> str:
    return date.today().isoformat()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "untitled"


def humanize_slug(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").title()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    block = text[4:end]
    body = text[end + 5 :]
    data: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        data[key.strip()] = value
    return data, body


def serialize_frontmatter(data: dict[str, str]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if value.startswith("[") or value.startswith("{"):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {quote(value)}")
    lines.append("---")
    return "\n".join(lines)


def update_frontmatter(path: Path, updates: dict[str, str]) -> None:
    text = read_text(path)
    frontmatter, body = parse_frontmatter(text)
    frontmatter.update({key: value for key, value in updates.items() if value is not None})
    rendered = serialize_frontmatter(frontmatter)
    write_text(path, rendered + "\n\n" + body.lstrip("\n"))


def skill_asset(skill_name: str, asset_name: str) -> Path:
    return SKILLS / skill_name / "assets" / asset_name


def render_template(template_path: Path, replacements: dict[str, str]) -> str:
    template = read_text(template_path)
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def append_log(event: str, title: str, details: str = "") -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        write_text(LOG, "# Operations Log\n")
    line = f"\n## [{today()}] {event} | {title}\n"
    if details:
        line += f"\n- {details}\n"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)


def topic_overview_path(topic: str) -> Path:
    return TOPICS / topic / "overview.md"


def note_link_from_overview(overview: Path, note_path: Path) -> Path:
    return Path(".") / Path(
        note_path.relative_to(overview.parent).as_posix()
    )


def build_overview_entry(
    *,
    title: str,
    venue: str = "",
    paper_url: str = "",
    note_path: Path | None = None,
    checked: bool = False,
    overview_path: Path,
) -> str:
    parts = []
    status = "x" if checked else " "
    prefix = f"- [{status}] "
    if venue:
        prefix += f"{venue}: "
    prefix += title
    parts.append(prefix)
    if paper_url:
        label = "arXiv" if "arxiv.org" in paper_url else "Paper"
        parts.append(f"[{label}]({paper_url})")
    if note_path is not None:
        rel = note_link_from_overview(overview_path, note_path).as_posix()
        parts.append(f"[Note]({rel})")
    return ", ".join(parts)


def normalize_pdf_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"}:
        return source

    host = parsed.netloc.lower()
    path = parsed.path

    if "arxiv.org" in host:
        if path.startswith("/abs/"):
            paper_id = path.removeprefix("/abs/")
            return urlunparse(parsed._replace(path=f"/pdf/{paper_id}.pdf", query="", fragment=""))
        if path.startswith("/pdf/") and not path.endswith(".pdf"):
            return urlunparse(parsed._replace(path=f"{path}.pdf", fragment=""))

    if "openreview.net" in host:
        paper_id = parse_qs(parsed.query).get("id", [""])[0]
        if paper_id and path != "/pdf":
            return urlunparse(parsed._replace(path="/pdf", query=f"id={paper_id}", fragment=""))

    return source


def extract_canonical_key(url: str) -> str:
    """Return a normalized canonical identifier for deduplication.

    - arXiv URLs  → ``"arxiv:<id>"`` with version suffix stripped (e.g. ``"arxiv:2401.00001"``)
    - OpenReview  → ``"openreview:<id>"``
    - Other URLs  → full normalized URL (via normalize_pdf_source)
    - Empty str   → empty string
    """
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return url

    host = parsed.netloc.lower()
    path = parsed.path

    if "arxiv.org" in host:
        for prefix in ("/abs/", "/pdf/"):
            if path.startswith(prefix):
                arxiv_id = path[len(prefix):]
                if arxiv_id.endswith(".pdf"):
                    arxiv_id = arxiv_id[:-4]
                # Strip version suffix: 2401.00001v2 → 2401.00001
                arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                return f"arxiv:{arxiv_id}"

    if "openreview.net" in host:
        paper_id = parse_qs(parsed.query).get("id", [""])[0]
        if paper_id:
            return f"openreview:{paper_id}"

    return normalize_pdf_source(url)


def find_by_canonical_key(
    canonical_key: str, topic: str
) -> tuple[Path | None, Path | None]:
    """Search inbox and topic notes for a paper matching *canonical_key*.

    Returns ``(inbox_path, note_path)``; either may be ``None``.
    Skips structural files (``index.md``, ``overview.md``, ``landscape.md``).
    Tier-2 slug checks are the caller's responsibility.
    """
    if not canonical_key:
        return None, None

    inbox_match: Path | None = None
    note_match: Path | None = None

    for inbox_file in sorted(INBOX.glob("*.md")):
        fm, _ = parse_frontmatter(read_text(inbox_file))
        existing = fm.get("paper_url", "")
        if existing and extract_canonical_key(existing) == canonical_key:
            inbox_match = inbox_file
            break

    topic_dir = TOPICS / topic
    if topic_dir.exists():
        _skip = {"index.md", "overview.md", "landscape.md"}
        for note_file in sorted(topic_dir.glob("*.md")):
            if note_file.name in _skip:
                continue
            fm, _ = parse_frontmatter(read_text(note_file))
            existing = fm.get("paper_url", "")
            if existing and extract_canonical_key(existing) == canonical_key:
                note_match = note_file
                break

    return inbox_match, note_match


def normalize_title_key(title: str) -> str:
    text = title.replace("**", "").strip().lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"[\s_-]+", " ", text)
    return text.strip()


def extract_overview_title(line: str) -> str:
    match = re.match(r"^\s*-\s*\[[ xX]\]\s*(.+)$", line)
    if not match:
        return ""
    text = match.group(1)
    text = re.split(r",\s*\[[^\]]+\]\(", text, maxsplit=1)[0]
    text = text.replace("**", "")
    text = re.sub(r"^[^:]{0,80}\b(?:19|20)\d{2}(?: [A-Za-z]+)?:\s*", "", text)
    return text.strip()


def overview_has_entry(
    *,
    topic: str,
    title: str,
    paper_url: str = "",
    note_path: Path | None = None,
) -> bool:
    overview = topic_overview_path(topic)
    if not overview.exists():
        return False

    overview_text = read_text(overview)
    canonical_key = extract_canonical_key(paper_url) if paper_url else ""
    note_rel = (
        note_link_from_overview(overview, note_path).as_posix() if note_path is not None else ""
    )
    wanted_title = normalize_title_key(title)

    for line in overview_text.splitlines():
        if not re.match(r"^\s*-\s*\[[ xX]\]", line):
            continue

        if canonical_key:
            for target in MARKDOWN_LINK_RE.findall(line):
                if extract_canonical_key(target) == canonical_key:
                    return True

        if note_rel and f"[Note]({note_rel})" in line:
            return True

        if wanted_title:
            existing_title = normalize_title_key(extract_overview_title(line))
            if existing_title and existing_title == wanted_title:
                return True

    return False


def _upsert_overview_lines(
    *,
    lines: list[str],
    title: str,
    venue: str,
    paper_url: str,
    note_path: Path | None,
    checked: bool,
    overview: Path,
) -> tuple[list[str], bool]:
    """Mutate overview lines using the same three-tier match strategy as overview_has_entry.

    Tier 1: canonical URL/ID match via links embedded in the line.
    Tier 2: exact note-path link match.
    Tier 3: exact normalized title match (not substring — avoids GAIL/InfoGAIL collisions).

    Non-checklist lines are passed through untouched.
    """
    updated_lines: list[str] = []
    found = False
    canonical_key = extract_canonical_key(paper_url) if paper_url else ""
    note_rel = (
        note_link_from_overview(overview, note_path).as_posix() if note_path is not None else ""
    )
    wanted_title = normalize_title_key(title)

    for line in lines:
        if not re.match(r"^\s*-\s*\[[ xX]\]", line):
            updated_lines.append(line)
            continue

        line_matches = False
        if canonical_key:
            for target in MARKDOWN_LINK_RE.findall(line):
                if extract_canonical_key(target) == canonical_key:
                    line_matches = True
                    break
        if not line_matches and note_rel and f"[Note]({note_rel})" in line:
            line_matches = True
        if not line_matches and wanted_title:
            existing_title = normalize_title_key(extract_overview_title(line))
            if existing_title and existing_title == wanted_title:
                line_matches = True

        if line_matches:
            found = True
            line = re.sub(r"^\s*-\s*\[[ xX]\]", f"- [{'x' if checked else ' '}]", line, count=1)
            if note_path is not None and "[Note](" not in line:
                rel = note_link_from_overview(overview, note_path).as_posix()
                line += f", [Note]({rel})"
            if paper_url and "[arXiv](" not in line and "[Paper](" not in line:
                label = "arXiv" if "arxiv.org" in paper_url else "Paper"
                line += f", [{label}]({paper_url})"

        updated_lines.append(line)

    if not found:
        updated_lines.append(
            build_overview_entry(
                title=title,
                venue=venue,
                paper_url=paper_url,
                note_path=note_path,
                checked=checked,
                overview_path=overview,
            )
        )

    return updated_lines, found


def ensure_overview_entry(
    *,
    topic: str,
    title: str,
    venue: str = "",
    paper_url: str = "",
    note_path: Path | None = None,
    checked: bool = False,
) -> Path:
    overview = topic_overview_path(topic)
    if not overview.exists():
        raise FileNotFoundError(f"overview.md not found for topic '{topic}': {overview}")

    text = read_text(overview)
    lines = text.splitlines()

    if AUTO_QUEUE_START in text and AUTO_QUEUE_END in text:
        start = lines.index(AUTO_QUEUE_START)
        end = lines.index(AUTO_QUEUE_END)
        queue_lines = [line for line in lines[start + 1 : end] if line.strip()]
        queue_lines, _ = _upsert_overview_lines(
            lines=queue_lines,
            title=title,
            venue=venue,
            paper_url=paper_url,
            note_path=note_path,
            checked=checked,
            overview=overview,
        )
        updated_lines = lines[: start + 1] + queue_lines + lines[end:]
    else:
        updated_lines, found = _upsert_overview_lines(
            lines=lines,
            title=title,
            venue=venue,
            paper_url=paper_url,
            note_path=note_path,
            checked=checked,
            overview=overview,
        )
        if not found and updated_lines[:-1] and updated_lines[-2].strip():
            updated_lines.insert(len(updated_lines) - 1, "")

    write_text(overview, "\n".join(updated_lines).rstrip() + "\n")
    return overview


def _archive_filename(canonical_key: str, slug: str, suffix: str) -> str:
    """Return the archive filename for a PDF.

    Uses the canonical paper ID when available to prevent slug collisions:
      - arXiv   → ``arxiv-2401.00001.pdf``
      - OpenReview → ``openreview-abc123.pdf``
      - Other   → ``<slug>.pdf``  (title-derived, same as the old behaviour)
    """
    if canonical_key.startswith(("arxiv:", "openreview:")):
        safe = canonical_key.replace(":", "-").replace("/", "-")
        return f"{safe}{suffix}"
    return f"{slug}{suffix}"


def archive_pdf(source: str, slug: str, paper_url: str = "", force: bool = False) -> str:
    """Archive a PDF into ``raw/papers/`` and return its repo-relative path.

    Idempotency: if the target file already exists and *force* is False, the
    existing file is returned immediately without re-downloading or overwriting.
    Pass ``force=True`` (propagated from ``--force`` in intake) to replace it.

    Filename priority: canonical arXiv / OpenReview ID > title slug, so two
    papers that happen to share a slug cannot overwrite each other's archive.
    """
    RAW_PAPERS.mkdir(parents=True, exist_ok=True)
    source = normalize_pdf_source(source)
    canonical_key = extract_canonical_key(paper_url) if paper_url else ""
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        suffix = Path(parsed.path).suffix or ".pdf"
        target = RAW_PAPERS / _archive_filename(canonical_key, slug, suffix)
        if target.exists() and not force:
            return target.relative_to(ROOT).as_posix()
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            urlretrieve(source, tmp)
            tmp.rename(target)  # atomic on POSIX same-filesystem
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return target.relative_to(ROOT).as_posix()

    source_path = Path(source).expanduser().resolve()
    suffix = source_path.suffix or ".pdf"
    target = RAW_PAPERS / _archive_filename(canonical_key, slug, suffix)
    if target.exists() and not force:
        return target.relative_to(ROOT).as_posix()
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        shutil.copy2(source_path, tmp)
        tmp.rename(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target.relative_to(ROOT).as_posix()


def load_title(path: Path, fallback: str) -> str:
    frontmatter, _ = parse_frontmatter(read_text(path))
    return frontmatter.get("title", fallback)


def ensure_marked_section(path: Path, heading: str, start: str, end: str, block: str) -> None:
    text = read_text(path)
    generated = f"{heading}\n\n{start}\n{block}\n{end}"
    if start in text and end in text:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        text = pattern.sub(f"{start}\n{block}\n{end}", text)
        write_text(path, text.rstrip() + "\n")
        return

    if text and not text.endswith("\n"):
        text += "\n"
    if text and not text.endswith("\n\n"):
        text += "\n"
    text += generated + "\n"
    write_text(path, text)


def git_tracked(pathspec: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", pathspec],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]
