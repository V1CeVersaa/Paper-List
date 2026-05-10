# Shared Conventions

This file is the expanded reference for rules that are **double-written** in brief in both `AGENTS.md` and the relevant `SKILL.md`. When a rule here conflicts with a SKILL.md or AGENTS.md statement, the AGENTS.md/SKILL.md version takes precedence (they are the authoritative short form; this file is the detail layer).

---

## Source URL Normalization

When archiving a paper from a URL, normalize to a direct PDF download before calling `archive_pdf()` or storing `source_pdf`.

| Input URL pattern | Normalized form |
|---|---|
| `https://arxiv.org/abs/<id>` | `https://arxiv.org/pdf/<id>.pdf` |
| `https://arxiv.org/pdf/<id>` (no `.pdf`) | `https://arxiv.org/pdf/<id>.pdf` |
| `https://openreview.net/forum?id=<id>` | `https://openreview.net/pdf?id=<id>` |
| Any other URL | Leave as-is; store in `paper_url`, leave `source_pdf` empty |
| Local file path | Copy to `raw/papers/<slug><ext>`, return relative path |

Rules:
- Never guess PDF endpoints for publisher-specific sites (NeurIPS, ICML HTML proceedings, ACL Anthology, etc.). Store `paper_url` and leave `source_pdf` empty.
- Only archive actual PDF sources into `raw/papers/`. Do not pass arXiv `e-print` URLs, TeX archives, HTML pages, or other source bundles to `--pdf`; `archive_pdf()` rejects unsupported non-PDF sources instead of recording them as `source_pdf`.
- `paper_url` always stores the canonical **human-facing** link (the abstract page, not the PDF), regardless of whether the PDF was archived.
- `source_pdf` stores the relative path under `raw/papers/` only when a local copy exists.
- The `repo_ops.normalize_pdf_source()` function handles arXiv and OpenReview automatically; call it before `urlretrieve`.

---

## Slug Generation

Slugs are derived from titles using `repo_ops.slugify()`:

1. Strip leading/trailing whitespace
2. Lowercase
3. Remove characters that are not word chars, spaces, or hyphens
4. Replace spaces and underscores with hyphens
5. Collapse multiple consecutive hyphens
6. Strip leading/trailing hyphens
7. Fallback to `"untitled"` if empty

**Topic directories** use underscores (e.g., `reinforcement_learning`). This is a separate convention from paper slugs which use hyphens. When `create_topic.py` creates a topic, it uses underscores; slugify returns hyphens — so topic slugs are set explicitly by the user or script, not derived from `slugify()`.

---

## Path ↔ Visibility Mapping

The path determines visibility semantics. Frontmatter `visibility` must agree with the path.

| Path prefix | Required visibility | Git tracked? |
|---|---|---|
| `content/topics/` | `public` | Yes |
| `content/syntheses/` | `public` | Yes |
| `content/conferences/` | `public` | Yes |
| `content/local/` | `local` | No (gitignored) |
| `content/drafts/` | `draft` | No (gitignored) |
| `inbox/` | — (no visibility field) | No (gitignored) |
| `raw/papers/` | — (binary assets) | No (gitignored) |
| `ops/` | — (operational logs) | No (gitignored) |

**Critical rule**: Do not rely on frontmatter alone to keep private work off GitHub. Private work must stay in gitignored paths. `visibility: local` in a file under `content/topics/` does **not** prevent it from being committed.

---

## Tag Policy

Frontmatter `tags` are for retrieval and synthesis, not for exhaustively restating every concept in the paper.

Rules:
- Keep tags sparse. Default to `0-3` tags; do not add filler tags just because a field exists.
- Order tags from highest fit and importance to lowest.
- The **first tag is the primary tag** and is the one index-style list pages are allowed to display.
- For notes under `content/topics/<topic>/`, do not use that same topic slug as the first tag when a more specific tag is available. Prefer the more discriminative tag first, and usually drop the redundant topic tag entirely.
- Prefer stable, reusable concepts over paper-specific wording.
- If two candidate tags overlap heavily, keep the more informative one and drop the weaker alias.

Examples:
- Good: `["policy_optimization", "trust_region"]`
- Too broad/noisy: `["reinforcement_learning", "optimization", "policy", "trust_region", "theory"]`

---

## Related-Note Pass for Paper Notes

Every normal `/read` topic note should be placed inside the existing reading graph, not written as an isolated summary. After the first full reading and before drafting the final prose:

1. Inspect the target topic's `index.md` and `overview.md` to understand the local reading path.
2. If nearby completed notes are not obvious, run `maintain/scripts/query_context.py` with the paper title plus 3-6 key concepts.
3. Read one or two genuinely relevant completed notes. Prefer notes that are direct precursors, stronger follow-ups, mechanism papers, mitigation papers, benchmark critiques, or bridge papers.
4. Write the relationship into frontmatter `related`, the third paragraph of the `Contributions` callout, and when useful `Related Work & Future Work`.

The relationship must be concrete. Good wording explains whether the current paper prepares, generalizes, contrasts with, mechanizes, or supplies the empirical baseline for another note. Avoid generic claims such as "related to safety alignment." Existing notes can guide positioning and contrast, but current-paper claims must remain grounded in the current paper itself.

---

## Deduplication Priority (Three-Tier)

When checking whether a paper already exists before `/intake` or `/read`:

**Tier 1 — Exact ID match** (highest confidence):
- Compare `paper_url` against all existing `paper_url` fields in `inbox/*.md` and `content/topics/<topic>/*.md` frontmatter
- arXiv ID extraction: `arxiv.org/abs/<id>` and `arxiv.org/pdf/<id>` share the same `<id>`; treat them as equal
- OpenReview ID extraction: `?id=<id>` parameter

**Tier 2 — Slug match**:
- Generate `slug = slugify(title)`
- Check if `inbox/<slug>.md` exists
- Check if `content/topics/<topic>/<slug>.md` exists

**Tier 3 — Agent judgment**:
- If tiers 1 and 2 don't conclusively resolve the question, the agent should compare normalized titles (lowercase, stripped punctuation) and judge similarity
- Do not use a fixed edit-distance threshold; use judgment based on title length, shared key terms, and shared venue/year if available

**On duplicate detected**:
- If inbox has entry but overview does not → add only the overview entry (call `ensure_overview_entry`)
- If both inbox and overview have entries → report and skip; do not create a third entry
- If neither has the entry → proceed normally

---

## promote.py → sync_overview.py Call Order

After `/read` produces a finished note, always run in this order:

1. **`promote.py`** — moves the file to its final path (if it was in `drafts/`), updates `status`, `visibility`, and `updated` frontmatter, and optionally deletes the inbox item
2. **`sync_overview.py`** — ensures the topic `overview.md` has a `[x]` checked entry pointing to the promoted note

Rationale: `promote.py` establishes the canonical path; `sync_overview.py` needs that path to build the correct relative `[Note](...)` link. Running them in reverse order creates a broken or missing link.

If the note is already in its final path (no move needed), skip `promote.py` and call `sync_overview.py` directly with `--checked`.

---

## Ops Log Format

Each workflow appends one entry to `ops/log.md`. Format:

```
## [YYYY-MM-DD] <event> | <Title>

- <key>=<value> | <key>=<value>
```

Event strings by skill:

| Skill | Event string |
|---|---|
| `/intake` | `intake` |
| `/read` | `read` |
| `/read` (promote step) | `promote` |
| `/explore` | `explore` |
| `/discuss` | `discuss` |
| `/refresh` | `refresh` |
| `/synthesize` | `synthesize` |
| `/maintain` | `maintain` |

Event strings follow the work performed, not only the storage path. A single-paper finished note under `content/conferences/<venue-year>/<paper>.md` is still `/read` and uses `event=read`; `/synthesize` is reserved for cross-paper comparisons, conference-level observations, surveys, and landscape updates.

Common detail keys:
- `topic=<slug>`
- `venue=<venue-year>`
- `path=<relative path from repo root>`
- `inbox=<inbox item path>` (for intake/read)
- `source=<source_pdf path>` (for read/promote)

Minimum entry per skill:
- `/intake`: `topic=`, `inbox=`
- `/read`/`promote`: `topic=`, `path=`, `source=` if PDF archived
- `/explore`: `topic=`
- `/discuss`: `path=` of idea file
- `/refresh`: `path=`
- `/synthesize`: `path=` of output file
