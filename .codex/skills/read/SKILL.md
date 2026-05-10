---
name: read
description: Use when the user wants a finished note for a single paper — including "帮我读这篇"、"整理这个 PDF"、"write up this paper"、"take notes on arXiv:xxxx"、or asks to analyze one conference-list paper into a paper note before final topic归档. Use even if the user does not say /read.
---

# `/read`

## When To Use

- The user wants a full paper note now.
- The input is a PDF path, paper URL, or an already queued inbox item.
- The expected output is a polished note, not just a queue entry.
- The user wants a finished note for one paper from a conference list, even when the note should temporarily live under `content/conferences/<venue-year>/` before later topic归档.

## Inputs Expected

- paper title
- PDF path or paper URL
- target topic, or a conference-local destination plus the likely final topic when known
- optional: inbox item path

## Artifact Split

- `content/topics/<topic>/<slug>.md` is the normal polished topic note. It must satisfy `AGENTS.md §3`, reorganize the paper's main line, and remain explicitly **not** a direct translation.
- `content/conferences/<venue-year>/<slug>.md` is a conference-local polished paper note when the user wants to read the paper now but postpone final topic归档. It is still a `/read` output, not a `/synthesize` output.

## Preflight Checks

1. **Already-noted check**: query `inbox/<slug>.md` first, then the expected note destination (`content/topics/<topic>/<slug>.md` or `content/conferences/<venue-year>/<slug>.md`).
   - If a `status: complete` note already exists → hand off to `/refresh` instead of rewriting.
   - If an inbox item exists → read it for existing positioning before starting.
2. **Confirm destination** exists: `content/topics/<topic>/` for topic notes, or `content/conferences/<venue-year>/` for conference-local paper notes.
3. If the source PDF is not yet archived, normalize the URL and archive it into `raw/papers/`:
   - `arxiv.org/abs/<id>` → `arxiv.org/pdf/<id>.pdf`
   - `openreview.net/forum?id=<id>` → `openreview.net/pdf?id=<id>`
   - Archive only PDFs; do not archive arXiv `e-print` URLs, TeX archives, HTML pages, or other source bundles.
4. Decide the working paths:
   - polished topic note path: either `content/drafts/` first or directly `content/topics/<topic>/`
   - polished conference-local note path: directly `content/conferences/<venue-year>/<slug>.md`

## Workflow

1. Read the paper in full at least once before writing — build the overall argument first.
2. Use `assets/paper.md` as the scaffold for the polished note if starting from scratch.
   - The polished note is a separate artifact. It must explain the paper in your own organized prose and must not collapse into sentence-by-sentence translation.
3. Run a related-note pass before drafting the final note:
   - Inspect the target topic's `index.md` and `overview.md`, plus any inbox positioning notes.
   - Use `python3 .codex/skills/maintain/scripts/query_context.py "<paper title and 3-6 key concepts>" --limit 8` when the nearest prior notes are not obvious.
   - Read one or two genuinely relevant completed notes when they exist. Use them to position the new paper in the repo's reading graph; do not use them to invent claims about the current paper.
   - Carry the result into frontmatter `related`, the Contributions positioning paragraph, and `Related Work & Future Work`.
4. Follow the writing standard in `AGENTS.md §3` (also mirrored at `references/writing-guide.md`). The five-section structure is the default:
   - `Contributions` callout → `Introduction` → `Problem Setup` → `Algorithm/Methods/Model` → `Experiments` → `Related Work & Future Work`
   - The note must explicitly answer the minimum reader questions from the writing guide: focused problem, motivating insight or observation, proposed method, stated claims/论断, supporting results, prior-work dependency and downstream influence, and concrete limitations or weaknesses.
   - For normal topic notes, the Contributions callout should usually have three paragraphs: contribution, boundary/caveat, and repo-positioning / downstream role.
   - The positioning paragraph should name one or two completed notes when relevant and explain the concrete relationship, such as precursor, stronger follow-up, mechanism bridge, mitigation, benchmark critique, or downstream use.
5. Set frontmatter `tags` sparingly:
   - Default to `0-3` tags, not a concept dump
   - Sort them from highest fit / highest importance to lowest
   - Treat the first tag as the primary tag that index pages may surface
   - If the note lives under a topic folder, do not make the current `topic` slug the first tag when a more specific tag exists
6. Carry source metadata forward when known:
   - Set `source_pdf` on the polished note.
7. Run the post-read placement check before finalizing:
   - Compare the paper's primary contribution against the target topic's `index.md` scope and the nearest competing topics.
   - If the current topic is clearly wrong, change the target path before running `promote.py` and `sync_overview.py`.
   - If the fit is genuinely ambiguous, record the competing placements and ask the user instead of silently filing the paper.
   - For conference-local paper notes, also check the conference subcategory in `content/conferences/<venue-year>/index.md`; if it is clearly wrong, move the checklist item before finalizing. Record the likely final topic and any boundary ambiguity in a short `Placement Check` section.
8. **Adjust depth by paper type** (not separate branches — same structure, different emphasis):
   - **Theory-heavy** (main contribution is theorems/derivations): preserve derivation chains step by step; use `> [!todo]-` collapsible blocks for proofs; Method section is the heaviest
   - **Empirical-heavy** (main contribution is system/experiments): Experiments section emphasizes 2–3 key ablations; Method section treats implementation details lightly
   - **Short / Workshop** (< 8 pages): Contributions block can be one paragraph; Related Work may be omitted; sections may be combined
9. Run `python3 .codex/skills/maintain/scripts/note_length.py --threshold 3500 <note-path>` before finalization. If it fails, keep expanding the note unless the paper is genuinely too short/thin and the final response explicitly states that exception.
10. For topic notes, run `python3 .codex/skills/read/scripts/promote.py ...` when the note is ready (moves file, updates frontmatter).
11. For topic notes, run `python3 .codex/skills/read/scripts/sync_overview.py ...` after promote to mark the overview entry `[x]`.
   - **Order matters**: promote first (establishes canonical path), then sync_overview (builds the correct relative `[Note](...)` link).
12. For conference-local paper notes, update `content/conferences/<venue-year>/index.md` by checking the paper item and adding `[Note](./<slug>.md)`. Do not run `promote.py` unless the note is being moved into a final topic.
13. Delete inbox item if the read started from inbox (promote.py handles this with `--inbox`; otherwise remove it only when the conference-local note has fully replaced the queue item).
14. Append ops log entry (`event=read`; include `event=promote` only when a promote step actually ran).

## Files It May Write

- `raw/papers/`
- `content/drafts/`
- `content/topics/<topic>/`
- `content/conferences/<venue-year>/<slug>.md`
- `content/conferences/<venue-year>/index.md`
- `ops/log.md`

## Scripts

- `assets/paper.md` — note scaffold
- `scripts/promote.py` — move + frontmatter update + overview sync
- `scripts/sync_overview.py` — overview entry maintenance
- `../maintain/scripts/query_context.py` — related-note discovery when neighboring notes are not obvious
- See `_shared/script-registry.md` for full invocation examples

## Done Criteria

- The note exists in its intended read destination (`content/topics/<topic>/` for final topic notes, or `content/conferences/<venue-year>/` for conference-local paper notes awaiting later归档)
- Frontmatter `status` is `complete`, `visibility` is `public`
- Frontmatter `source_pdf` is set when known
- Frontmatter `tags`, if present, are sparse and ordered by fit/importance with the primary tag first
- Frontmatter `related` names the completed notes that the related-note pass found, when such notes exist
- The Contributions callout includes a repo-positioning / downstream-role paragraph for normal topic notes, naming one or two old notes when relevant
- The post-read placement check confirms that the final topic matches the paper's primary contribution, or the conference-local note records the likely final topic and any ambiguity
- The relevant checklist is marked `[x]` with a `[Note](...)` link: topic `overview.md` for topic notes, or `content/conferences/<venue-year>/index.md` for conference-local paper notes
- The inbox item is removed if the read started from inbox
- Writing quality meets the standard in `AGENTS.md §3` (minimum reader questions, Contributions block, 5-section structure, term pairing, and no translation residue in the polished note)
- `note_length.py --threshold 3500 <note-path>` passes, unless the final response gives a concrete paper-specific exception

## Hand-Offs

- Already has a `status: complete` note → `/refresh`
- After reading, if multiple topics or pages were touched → `/maintain`
