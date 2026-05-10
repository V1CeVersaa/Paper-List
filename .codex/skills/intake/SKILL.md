---
name: intake
description: Use whenever the user wants to add one or more papers to the reading pipeline without writing full notes now — including "把这篇加进去"、"先 queue 几篇"、"这些论文之后再读"、"batch add these"、"save for later". Use even when the user does not say /intake explicitly.
---

# `/intake`

## When To Use

- The user wants to add papers into the pipeline but not fully read them yet.
- The user provides a batch of titles, links, or PDFs.
- A `/discuss` session reveals reading gaps that need to be queued.

## Inputs Expected

- title or paper list
- topic
- optional: venue, paper URL, PDF path or URL, priority, positioning notes

## Preflight Checks

1. **Deduplicate** against existing inbox items and overview entries using the three-tier priority:
   - Tier 1: Exact `paper_url` / arXiv ID / OpenReview ID match in any existing `inbox/*.md` or `content/topics/<topic>/*.md` frontmatter
   - Tier 2: Slug match — check if `inbox/<slug>.md` or `content/topics/<topic>/<slug>.md` exists
   - Tier 3: Agent judgment on normalized title similarity (do not use a fixed edit-distance threshold)
   - On duplicate: if inbox exists but overview does not → add only the overview entry; if both exist → report and skip
2. **Confirm topic exists** (`content/topics/<topic>/` must have `index.md` and `overview.md`).
3. If the topic does not exist or the topic boundary is unclear → stop and hand off to `/explore`.
4. **Normalize source URLs** before archiving (see `_shared/conventions.md#source-url-normalization`):
   - `arxiv.org/abs/<id>` → `arxiv.org/pdf/<id>.pdf`
   - `openreview.net/forum?id=<id>` → `openreview.net/pdf?id=<id>`
   - Do not archive arXiv `e-print` URLs, TeX archives, HTML pages, or other source bundles as `source_pdf`.

## Workflow

1. Create a structured inbox item from `assets/inbox-item.md` using `scripts/intake.py`.
2. Archive the PDF into `raw/papers/<base>.pdf` when a supported PDF source is provided. For arXiv papers, also derive the PDF from the canonical `paper_url` when `--pdf` was omitted.
3. Reserve a `- [ ]` entry in the topic `overview.md`.
4. Rebuild the generated topic index if needed.
5. Append an ops log entry (`event=intake`).

**Batch error semantics**: single-item failures do not block the rest of the batch. If a topic is missing, the entire batch hands off to `/explore` before any items are created.

## Priority Semantics

- `high`: user explicitly flags it, or this paper is a direct dependency of the current `/discuss` session
- `low`: user adds it opportunistically, not in the current research focus
- `normal`: default for everything else

## Files It May Write

- `inbox/<slug>.md`
- `raw/papers/<base>.pdf`
- `content/topics/<topic>/overview.md`
- `content/index.md` and `content/topics/index.md` (via `rebuild_index.py`)
- `ops/log.md`

## Scripts

- `scripts/intake.py` — canonical entry point; see `_shared/script-registry.md` for full example
- `../maintain/scripts/rebuild_index.py`

## Done Criteria

- Each paper has a structured inbox item
- Each paper has an `overview.md` placeholder entry
- Each PDF is archived if a supported local file or URL was provided, or if the canonical arXiv paper URL can be normalized to a direct PDF

## Hand-Offs

- Targeted reading of a queued paper → `/read`
- Topic does not exist or boundary is unclear → `/explore`
- If this intake session surfaced a research idea → `/discuss`
