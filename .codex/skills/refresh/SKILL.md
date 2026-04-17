---
name: refresh
description: Use when the user wants to improve an existing note without treating it as a fresh read — "这篇笔记写得不好帮我改一下"、"老笔记格式不对"、"有翻译腔清理一下"、"bring this note up to current standards"、or when the user opens a stale/incomplete note and asks about it. Use instead of /read when a note already exists.
---

# `/refresh`

## When To Use

- A note already exists and has meaningful content.
- The user wants to upgrade, clean, or complete an existing note — not start from scratch.

## When to Use `/refresh` vs `/read`

- **`/refresh`**: note has real content; goal is "upgrade to current standard"; preserve what's already good
- **`/read`**: note is empty or so incomplete that > 50% would need to be rewritten from the PDF; treat as a fresh read

## Upgrade Signals — Trigger `/refresh` When Any of These Are True

1. Obvious translation-residue paragraphs (sentence-by-sentence mapping, no reorganization)
2. Missing `> [!abstract] Contributions` callout block
3. Outdated or missing frontmatter fields (no `headline`, `tags`, `related`)
4. Bloated or weakly ordered tags (too many tags, or the most representative tag is not first)
5. `TODO` placeholders remaining in the body
6. `status: drafting` but content appears substantially complete
7. Missing `## 5. Related Work & Future Work` section
8. User opens the note and comments on quality or asks about it

## Workflow

1. Read the current note first.
2. Keep existing topic placement unless there is a clear structural error.
3. Apply the same writing quality bar as `AGENTS.md §3` (also in `../read/references/writing-guide.md`).
4. Upgrade in place — reorganize, expand thin sections, remove translation residue, fix formatting.
5. Trim and reorder `tags` when needed:
   - keep them sparse (`0-3` is the default target)
   - put the best-fitting and most important tag first
   - if the note already has a specific topical tag, do not leave the folder `topic` slug in first position
6. If the note becomes complete, update `status: complete` in frontmatter.
7. Synchronize overview state: run `../read/scripts/sync_overview.py ... --checked` when the note is now complete.
8. Append ops log entry (`event=refresh`).

## Files It May Write

- Existing note files under `content/topics/`
- Corresponding `overview.md` (via `sync_overview.py`)
- `ops/log.md`

## Scripts

- `../read/assets/paper.md` — reference template
- `../read/scripts/sync_overview.py` — update overview when note becomes complete
- `../read/scripts/promote.py` — if the note needs to move paths

## Done Criteria

- The note is structurally clean and meets current standards
- Translation residue and missing sections are addressed
- Tags are concise and ordered with the primary tag first
- `status` and `overview` entry reflect the improved note state

## Hand-Offs

- If > 50% of the note needs to be rewritten from the PDF → switch to `/read`
- After refresh, if multiple pages were touched → `/maintain`
