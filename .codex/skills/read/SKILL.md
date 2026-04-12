---
name: read
description: Use when the user wants a finished note for a single paper — including "帮我读这篇"、"整理这个 PDF"、"write up this paper"、"take notes on arXiv:xxxx"、or when the user provides a PDF/URL and names a target topic. Use even if the user does not say /read.
---

# `/read`

## When To Use

- The user wants a full paper note now.
- The input is a PDF path, paper URL, or an already queued inbox item.
- The expected output is a polished note, not just a queue entry.

## Inputs Expected

- paper title
- PDF path or paper URL
- target topic
- optional: inbox item path

## Preflight Checks

1. **Already-noted check**: query `inbox/<slug>.md` first, then `content/topics/<topic>/<slug>.md`.
   - If a `status: complete` note already exists → hand off to `/refresh` instead of rewriting.
   - If an inbox item exists → read it for existing positioning before starting.
2. **Confirm topic** exists (`content/topics/<topic>/`).
3. If the source PDF is not yet archived, normalize the URL and archive it into `raw/papers/`:
   - `arxiv.org/abs/<id>` → `arxiv.org/pdf/<id>.pdf`
   - `openreview.net/forum?id=<id>` → `openreview.net/pdf?id=<id>`
4. Decide whether the working note starts in `content/drafts/` or goes directly to `content/topics/<topic>/`.

## Workflow

1. Read the paper in full at least once before writing — build the overall argument first.
2. Use `assets/paper.md` as the scaffold if starting from scratch.
3. Follow the writing standard in `AGENTS.md §3` (also mirrored at `references/writing-guide.md`). The five-section structure is the default:
   - `Contributions` callout → `Introduction` → `Problem Setup` → `Algorithm/Methods/Model` → `Experiments` → `Related Work & Future Work`
4. **Adjust depth by paper type** (not separate branches — same structure, different emphasis):
   - **Theory-heavy** (main contribution is theorems/derivations): preserve derivation chains step by step; use `> [!todo]-` collapsible blocks for proofs; Method section is the heaviest
   - **Empirical-heavy** (main contribution is system/experiments): Experiments section emphasizes 2–3 key ablations; Method section treats implementation details lightly
   - **Short / Workshop** (< 8 pages): Contributions block can be one paragraph; Related Work may be omitted; sections may be combined
5. Run `python3 .codex/skills/read/scripts/promote.py ...` when the note is ready (moves file, updates frontmatter).
6. Run `python3 .codex/skills/read/scripts/sync_overview.py ...` after promote to mark the overview entry `[x]`.
   - **Order matters**: promote first (establishes canonical path), then sync_overview (builds the correct relative `[Note](...)` link).
7. Delete inbox item if the read started from inbox (promote.py handles this with `--inbox`).
8. Append ops log entry (`event=read` and `event=promote`).

## Files It May Write

- `raw/papers/`
- `content/drafts/`
- `content/topics/<topic>/`
- `ops/log.md`

## Scripts

- `assets/paper.md` — note scaffold
- `scripts/promote.py` — move + frontmatter update + overview sync
- `scripts/sync_overview.py` — overview entry maintenance
- See `_shared/script-registry.md` for full invocation examples

## Done Criteria

- The note exists in its final location (`content/topics/<topic>/`)
- Frontmatter `status` is `complete`, `visibility` is `public`
- The topic `overview.md` entry is marked `[x]` with a `[Note](...)` link
- The inbox item is removed if the read started from inbox
- Writing quality meets the standard in `AGENTS.md §3` (Contributions block, 5-section structure, term pairing, no translation residue)

## Hand-Offs

- Already has a `status: complete` note → `/refresh`
- After reading, if multiple topics or pages were touched → `/maintain`
