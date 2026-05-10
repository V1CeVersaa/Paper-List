---
name: explore
description: Use when the user wants to bootstrap a new topic, expand an existing topic boundary, or build a reading roadmap — "开一个新的 topic"、"我想系统读一下 X 领域"、"梳理这个领域的脉络"、"这是一批种子论文帮我建立 topic"、"expand the landscape for Y". Use even if the user does not say /explore.
---

# `/explore`

## When To Use

- The user wants a new topic or a major topic expansion.
- The input is a deep-research list, a field overview, or a discussion about topic boundaries.

## Inputs Expected

- topic name and slug
- user motivation and boundary hints
- optional: seed paper list

## Preflight Checks

1. **New topic vs. expansion** — decide before doing anything:
   - **New topic**: no `content/topics/<slug>/` directory exists; or the new direction's papers have minimal overlap with all existing topics
   - **Expansion**: the topic directory already exists with `index.md`/`overview.md`; new papers clearly belong within the existing scope — only `landscape.md` and `overview.md` need updating, skip `create_topic.py`
2. Confirm the canonical topic slug (use underscores, e.g., `causal_representation_learning`).
3. Decide whether the user wants topic ordering updated in the Quartz explorer (`quartz.layout.ts`).

## Workflow

### For a new topic

1. Run `scripts/create_topic.py "Human-Readable Title" --slug <slug> --description "..."` to scaffold `index.md`, `overview.md`, and `landscape.md`.
2. Fill in `landscape.md` using the structure below.
3. Fill in the `Scope` section of `topic-index.md` to articulate boundaries with neighboring topics.
4. Batch the supplied seed papers through `/intake`.
5. Rebuild indexes.
6. Append ops log entry (`event=explore`).

### For a topic expansion

1. Read the existing `landscape.md` and `overview.md` first.
2. Update the relevant sections of `landscape.md` (add new sub-areas, update roadmap).
3. Batch new papers through `/intake`.
4. Rebuild indexes if needed.
5. Append ops log entry.

## Landscape.md Structure

`landscape.md` should address all five sections. Stubs are acceptable initially, but each section must have at least one substantive paragraph before the topic is considered bootstrapped:

- **Problem Space**: What fundamental problem(s) this field is trying to solve; why they are hard; what past approaches failed and why
- **Methodology Spectrum**: Current technical approaches from the most conservative to the most aggressive, laid out as a spectrum; each major approach labeled with 1–2 representative papers
- **Evolution**: A rough timeline with 3–5 key turning points; each annotated with the representative paper(s) that caused or marked the shift
- **Key Open Questions**: 3–5 specific, concrete unsolved problems recognized by the community — not vague aspirations like "improve generalization"
- **Reading Roadmap**: Recommended reading order for someone new to this topic, divided into `entry` / `core` / `advanced` tiers

## Neighboring Topic Boundaries

When a new topic overlaps with an existing one:
- Write explicit boundary statements in both topics' `index.md` Scope sections
- Use cross-references (`[[topic-slug/index]]`) to link related topics
- If overlap is very high (most papers belong in both), consider whether a merge or a cross-cutting synthesis page is more appropriate than a new topic

## Files It May Write

- `content/topics/<topic>/index.md`
- `content/topics/<topic>/overview.md`
- `content/topics/<topic>/landscape.md`
- `quartz.layout.ts` (if topic ordering is updated)
- `ops/log.md`

## Scripts

- `assets/topic-index.md`, `assets/overview.md`, `assets/landscape.md` — templates
- `scripts/create_topic.py` — scaffolds all three files
- `../intake/scripts/intake.py` — for seeding the paper list
- `../maintain/scripts/rebuild_index.py`
- See `_shared/script-registry.md` for full invocation examples

## Done Criteria

- The topic folder is browsable with `index.md` and `overview.md`
- `landscape.md` has substantive content in all five sections
- The seed paper list has been queued via `/intake`
- Topic boundary is documented in `index.md` Scope section

## Hand-Offs

- Cross-topic comparison work → `/synthesize`
- Idea formation during exploration → `/discuss`
