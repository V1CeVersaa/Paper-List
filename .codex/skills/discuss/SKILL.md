---
name: discuss
description: Use when the user wants to think through an idea, hypothesis, or research direction interactively — "我有个想法想聊聊"、"你觉得这个方向怎么样"、"我在想能不能做 X"、or any open-ended intellectual conversation about a research topic without a clear deliverable yet. Use even if no slash command is given.
---

# `/discuss`

## When To Use

- The user has an idea, hypothesis, or research direction they want to reason through.
- The user wants to explore interactively before deciding whether to write anything up.
- No immediate paper-reading deliverable is expected.

## Inputs Expected

- seed idea or problem framing
- optional: related papers or topics

## Preflight Checks

1. Do not force file creation too early — let the conversation develop first.
2. When the discussion seems to be converging, check the landing signals below.
3. Keep idea work local by default; do not create public pages for speculative ideas.

## Workflow

1. Discuss freely. Use `query_context.py` to pull relevant notes into context when the conversation touches existing work (see `_shared/script-registry.md`).
2. Watch for **landing signals** — when at least two of the following are true, the idea is ready to record:
   - A falsifiable hypothesis has emerged ("X will cause Y because Z")
   - The user has returned to the same core question across 3+ exchanges
   - Specific experimental design or paper gaps are being discussed
   - The user says "记下来" / "这个值得深入" or equivalent
   - At least 2 concrete related papers have been identified as reading prerequisites
3. When landing: create or update `content/local/ideas/<name>.md` from `assets/idea.md`, filling in:
   - `Trigger`, `Hypothesis`, `Why It Matters`, `Risks`, `Next Steps`, `Connections`
4. If the discussion reveals reading gaps, route those papers through `/intake`.
5. Append ops log entry (`event=discuss`).

## Connections Format

```md
## Connections

- [[topic/paper-slug]] — 支撑（直接支持该假设）
- [[topic/paper-slug]] — 对比（方法不同，值得对比）
- [[topic/paper-slug]] — 延伸（如果假设成立，这篇是自然的下一步）
- Topics: [[topic-name]], [[topic-name]]
```

Use Obsidian double-bracket links. Label each connection as 支撑 / 对比 / 延伸 so the relationship is explicit.

## Upgrade Signals

Route to another skill when:
- Concrete reading gaps emerge → `/intake` those papers before continuing
- The idea matures to the point of needing cross-paper comparison → `/synthesize`
- The discussion reveals an entire undiscovered research area → `/explore`

## Files It May Write

- `content/local/ideas/<name>.md`
- `ops/log.md`

## Scripts

- `assets/idea.md` — idea page template
- `../maintain/scripts/query_context.py` — pull relevant existing notes into context

## Done Criteria

- The current state of the idea is written down (not trapped in chat only)
- Related topics and papers are linked in the Connections section
- Reading gaps have been routed to `/intake`

## Hand-Offs

- Concrete reading gaps → `/intake`
- Broader direction mapping → `/explore`
- Mature comparison output → `/synthesize`
