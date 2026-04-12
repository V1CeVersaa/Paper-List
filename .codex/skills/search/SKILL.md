---
name: search
description: Use when the user wants to find existing content in the repo — "有没有关于 X 的笔记"、"我之前读过哪些 Y"、"把和 Z 相关的内容列出来"、"find notes on"、"search for". This is a read-only skill: it never writes files, never modifies overviews, never updates indexes. Use even when the user does not say /search.
---

# `/search`

## When To Use

- The user wants to retrieve or discover existing notes, topics, syntheses, conferences, or local ideas.
- The user is asking "what do we already have" before deciding what to read or synthesize.

## Hard Boundaries (Read-Only)

- Does **not** write any markdown files
- Does **not** modify `overview.md`
- Does **not** update indexes
- Does **not** append to `ops/log.md`
- Does **not** archive PDFs

Contrast with similar skills:
- `/search`: find and list existing content, explain relevance
- `/synthesize`: compare content and produce a new written output
- `/discuss`: use retrieved content to push a research idea forward

## Workflow

1. Extract key terms from the user's query.
2. Run `query_context.py` with the query:
   ```bash
   python3 .codex/skills/maintain/scripts/query_context.py \
     "<user query terms>" \
     --limit 8
   ```
   Add `--include-private` if the user may be looking for local ideas or drafts.
3. Present results in ranked order. For each result, provide:
   - Path and title
   - 1–2 sentences explaining why it's relevant to the query
   - Current `status` if informative (e.g., `drafting` vs `complete`)
4. Recommend a next step for the most relevant results:
   - Note is complete and the user wants to review → `/refresh`
   - Note is missing and user wants to read the paper → `/read` or `/intake`
   - Multiple relevant notes suggest a comparison → `/synthesize`
   - Results spark a research idea → `/discuss`

## Inputs Expected

- Free-text query (topic, paper title fragment, concept, author name, venue)

## Scripts

- `../maintain/scripts/query_context.py` — full invocation example in `_shared/script-registry.md`

## Done Criteria

- Ranked results are presented with relevance explanations
- A next step is suggested for the most relevant hits
