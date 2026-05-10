---
name: synthesize
description: Use when the user wants output spanning multiple papers or topics — "比较一下这几篇"、"帮我写一个综述"、"这几篇的异同是什么"、"conference 趋势/观察笔记"、"更新这个 topic 的 landscape". Use even when the user does not say /synthesize.
---

# `/synthesize`

## When To Use

- The user asks to compare multiple papers.
- The user wants a topic summary, cross-topic connection, or survey-like page.
- The user wants a conference-level note recording emerging directions, methods, oral-session observations, or cross-paper patterns.
- The user wants to update an existing `landscape.md`.

## Inputs Expected

- target papers, topics, or notes
- comparison angle or question
- destination type (see sub-paths below)

## Workflow

1. Read the existing notes first instead of re-deriving from raw PDFs — synthesis is about connecting already-understood material.
2. **Choose the output sub-path** based on what the user needs:

### Sub-path A — Paper Comparison / Survey → `content/syntheses/<name>.md`

For comparing 2+ papers or producing a reusable cross-paper summary.

```md
## Summary
One-liner: what question does this synthesis answer?

## Comparison Axes
2–5 dimensions along which these works differ (e.g., sample efficiency, assumption strength, implementation complexity)

## Key Differences
For each axis: what each paper does and why it matters

## Open Questions
Questions raised by this comparison that remain unanswered

## Sources
- [[topic/paper-slug]] — one-line role in this synthesis
- [[topic/paper-slug]] — ...
```

### Sub-path B — Conference-Level Note → `content/conferences/<venue-year>.md` or `content/conferences/<venue-year>/index.md`

For recording observations from a conference, workshop, or oral session.

```md
## Overview
Theme of this conference/workshop in 1–2 sentences

## Oral / Spotlight Highlights
For each notable paper: one paragraph (contribution + why it matters)

## Emerging Trends
3–5 patterns visible across multiple papers/sessions

## Reading List
Papers worth following up with — route these through `/intake` afterward
```

### Boundary — Conference-Local Paper Notes Belong To `/read`

If the user asks to read one paper from a conference list and write a finished paper note under `content/conferences/<venue-year>/<paper>.md`, route that work to `/read`, not `/synthesize`. The storage path is conference-local, but the workflow type is still a single-paper read. The resulting ops log entry should therefore use `event=read`.

`/synthesize` may later consume those conference-local paper notes to produce a cross-paper comparison, conference trend summary, oral-session observation, or topic landscape update.

### Sub-path C — Landscape Update → existing `content/topics/<topic>/landscape.md`

For expanding an existing topic's landscape page. Do **not** create a new file.
- Read the existing `landscape.md` first
- Identify which sections need updating (new sub-area, new open questions, updated roadmap)
- Edit in place; follow the five-section structure from `/explore` (`_shared/conventions.md` or `AGENTS.md §0`)

3. Link back to all source notes using Obsidian double-bracket links (`[[topic/slug]]`).
4. Append ops log entry (`event=synthesize`).

## Files It May Write

- `content/syntheses/<name>.md`
- `content/conferences/<venue-year>.md`
- `content/conferences/<venue-year>/index.md` when it is being used as the venue-level observation page
- `content/topics/*/landscape.md`
- `ops/log.md`

## Scripts

- `assets/synthesis.md` — template for sub-path A
- `../maintain/scripts/query_context.py` — find relevant existing notes
- `../maintain/scripts/rebuild_index.py` — if a new synthesis file was added

## Done Criteria

- The result is filed back into the repo (not left in chat)
- Source notes are linked
- The synthesis is reusable outside this chat session
- Conference-level notes distinguish direct conference observations from later synthesis across source notes

## Hand-Offs

- After synthesis, run `/maintain` if multiple pages were touched
