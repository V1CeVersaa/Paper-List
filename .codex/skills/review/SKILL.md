---
name: review
description: >-
  Use when the user wants a quality check on existing notes without editing them — "这篇笔记好不好"、"帮我看看这组笔记质量怎么样"、"哪些笔记应该补充"、"audit my notes on topic X". This is a read-only skill: it produces a diagnostic report but never edits files. Use before deciding whether to run /refresh.
---

# `/review`

## When To Use

- The user wants to assess note quality before deciding whether to refresh them.
- The user wants a systematic audit of an entire topic's notes.
- The user is curious which notes meet current standards and which don't.

## Hard Boundaries (Read-Only)

- Does **not** edit any markdown files
- Does **not** modify `overview.md` or indexes
- Does **not** append to `ops/log.md`
- Output is a diagnostic report in the conversation only

## Workflow

1. Identify the target scope: a single note, all notes in a topic, or a custom file list.
2. For each note, check against the quality criteria from `AGENTS.md §3 §10 Pre-Completion Checklist`:
   - Has the note gone beyond direct translation?
   - Does it explicitly answer the minimum reader questions: focused problem, motivating insight, proposed method, stated claims, supporting results, prior-work dependency / downstream influence, and concrete problems or limitations?
   - Does `Contributions` block exist and summarize both core contribution and boundary condition?
   - Does `Introduction` cover motivation, gap, insight, and origin?
   - Is `Problem Setup` formal and clear?
   - Does `Algorithm/Methods/Model` present backbone before details?
   - Does `Experiments` include both results summary and specific weaknesses?
   - Is `Related Work & Future Work` present and organized?
   - Is there translation residue (sentence-by-sentence mapping)?
   - Are there `TODO` placeholders?
   - Is the frontmatter complete (`headline`, `tags`, `related`, `status: complete`)?
   - If tags exist, are they sparse and ordered from strongest fit to weakest, with the primary tag first and not merely repeating the note's own topic slug?
3. Categorize findings into three tiers:
   - **Must fix**: structural errors, missing required sections, `TODO` placeholders, visible translation residue
   - **Should fix**: thin sections, missing term pairing format, weak Contributions second paragraph
   - **Optional**: minor style improvements, missing figure placeholders
4. Present the report clearly. If the scope covers multiple notes, produce a summary table first, then per-note details.
5. Ask the user which notes they want to run `/refresh` on.

## Output Format

```
## Review Report: <scope>

### Summary
| Note | Must Fix | Should Fix | Overall |
|------|----------|------------|---------|
| PPO  | 0        | 2          | Good    |
| TRPO | 1        | 3          | Needs work |

### Per-Note Details

#### PPO.md
**Should fix:**
- §5 Experiments: no critical evaluation of baselines
- Contributions §2: boundary condition is vague

#### TRPO.md
**Must fix:**
- Missing `> [!abstract] Contributions` callout block
...
```

## Hand-Offs

- After review, user identifies notes to fix → `/refresh`
- If a note is so incomplete it needs a full rewrite → `/read`
