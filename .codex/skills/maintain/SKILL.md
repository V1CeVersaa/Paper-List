---
name: maintain
description: Use when the user asks to tidy, check, or sync the repo — "帮我 lint 一下"、"索引好像不对"、"overview 没有更新"、"rebuild indexes"、"check for inconsistencies"、or after any large batch of reads/intakes/explores. Use proactively after multi-paper sessions even if the user does not ask.
---

# `/maintain`

## When To Use

- After batch edits (multiple `/read`, `/intake`, `/explore`, `/refresh`, or `/synthesize` in sequence).
- Before publishing or committing.
- When the user asks to tidy, sync, lint, or check repository health.

## Workflow

1. Run `python3 .codex/skills/maintain/scripts/lint_repo.py` and review the output using the response guide below.
2. Run `python3 .codex/skills/maintain/scripts/rebuild_index.py` when topic structure changed.
3. Run `python3 .codex/skills/read/scripts/sync_overview.py ...` for any known missing overview entries.
4. Summarize inconsistencies that still require manual judgment and present them to the user.
5. Append ops log entry (`event=maintain`).

## Lint Response Guide

| Error Category | What It Means | Auto-Fix | Needs Judgment |
|---|---|---|---|
| `missing frontmatter key: title` | Page has no `title` field | Infer from filename via `migrate_frontmatter.py --only-missing` | If filename is ambiguous, ask user |
| `missing frontmatter key: status` | Page has no `status` field | Set to `drafting` via `migrate_frontmatter.py --only-missing` | If content looks complete, set `complete` |
| `visibility=draft in tracked public path` | Private visibility in public directory | Change `visibility` to `public`, or move file to `content/drafts/` | Confirm user intent before moving |
| `missing required file: overview.md` | Topic directory has no overview | Generate from `../explore/assets/overview.md` template | Content must be filled by user/agent |
| `missing required file: index.md` | Topic directory has no index | Generate from `../explore/assets/topic-index.md` template | Scope section must be filled |
| `git-tracked private path detected` | Private file in git index | `git rm --cached <file>` | Confirm no sensitive content first |
| `missing auto topic markers` | Index file lacks `<!-- AUTO:TOPICS:START/END -->` | Append markers + regenerate with `rebuild_index.py` | Check insertion point is appropriate |

## Script Trigger Guide

Run the right script for the right situation (full invocation examples in `_shared/script-registry.md`):

| Script | When to run |
|---|---|
| `lint_repo.py` | After any batch workflow; before every publish |
| `rebuild_index.py` | After creating/deleting a topic; after bulk `overview.md` changes |
| `query_context.py` | At the start of `/discuss` or `/synthesize` to pre-load relevant notes |
| `migrate_frontmatter.py` | **Only** after a frontmatter schema change requiring bulk backfill; not for daily maintenance |

## Files It May Write

- `content/index.md`
- `content/topics/index.md`
- topic `overview.md` files
- `ops/log.md`

## Done Criteria

- Generated indexes are up to date
- Obvious structure errors are fixed or reported to the user
- No private paths are accidentally git-tracked
