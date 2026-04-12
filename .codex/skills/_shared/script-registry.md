# Script Registry

Quick reference for all scripts in `.codex/skills/`. Each entry covers: what the script does, when to run it, and a complete example command. All paths are relative to the repo root.

---

## `intake/scripts/intake.py`

**What it does**: Creates a structured inbox item at `inbox/<slug>.md`, reserves a `- [ ]` entry in the topic `overview.md`, optionally archives the PDF, and appends an ops log entry.

**When to run it**: During `/intake`, once per paper. The script is the canonical way to create inbox items — do not create them manually.

**Example**:
```bash
python3 .codex/skills/intake/scripts/intake.py "Proximal Policy Optimization Algorithms" \
  --topic reinforcement_learning \
  --venue "ICML 2017" \
  --paper-url https://arxiv.org/abs/1707.06347 \
  --priority high \
  --why "Foundational PPO paper; needed for agentic RL landscape" \
  --positioning "Core policy gradient method; pairs with TRPO"
```

Optional flags:
- `--pdf <path-or-url>`: Archive PDF locally (accepts local path, direct PDF URL, arXiv abs URL, or OpenReview forum URL)
- `--slug <custom-slug>`: Override auto-generated slug
- `--force`: Overwrite existing inbox item

---

## `read/scripts/promote.py`

**What it does**: Moves a note from its current location to a target path, updates `status`/`visibility`/`updated` frontmatter, optionally deletes the inbox item, calls `ensure_overview_entry` to sync the topic overview, and appends an ops log entry.

**When to run it**: At the end of `/read`, after the note is finished. Run **before** `sync_overview.py`.

**Example**:
```bash
python3 .codex/skills/read/scripts/promote.py \
  content/drafts/ppo.md \
  --target content/topics/reinforcement_learning/PPO.md \
  --topic reinforcement_learning \
  --title "Proximal Policy Optimization Algorithms" \
  --venue "ICML 2017" \
  --paper-url https://arxiv.org/abs/1707.06347 \
  --inbox inbox/proximal-policy-optimization-algorithms.md
```

If note is already in final location (no move needed), omit the positional arg and `--target`; the script will only update frontmatter and sync overview.

---

## `read/scripts/sync_overview.py`

**What it does**: Ensures a paper entry exists in the topic `overview.md` with the correct `[x]`/`[ ]` checkbox state and `[Note](...)` link.

**When to run it**: After `promote.py`, or directly when a note is already in its final location and just needs its overview entry verified or checked.

**Example**:
```bash
# Mark as complete and add note link
python3 .codex/skills/read/scripts/sync_overview.py \
  "Proximal Policy Optimization Algorithms" \
  --topic reinforcement_learning \
  --venue "ICML 2017" \
  --paper-url https://arxiv.org/abs/1707.06347 \
  --note content/topics/reinforcement_learning/PPO.md \
  --checked

# Reserve a queued entry (unchecked)
python3 .codex/skills/read/scripts/sync_overview.py \
  "Some Paper Title" \
  --topic reinforcement_learning
```

---

## `explore/scripts/create_topic.py`

**What it does**: Creates the topic directory scaffold under `content/topics/<topic>/`, writing `index.md`, `overview.md`, and `landscape.md` from templates, and appends an ops log entry. Does **not** rebuild indexes — call `rebuild_index.py` separately after running this script.

**When to run it**: At the start of `/explore` when creating a **new** topic. For topic expansion (existing topic), skip this script and go directly to updating `landscape.md` and running `/intake` on new papers.

**Example**:
```bash
# title is a positional argument; --slug overrides the auto-generated slug
python3 .codex/skills/explore/scripts/create_topic.py \
  "Active Imitation Learning" \
  --slug active_imitation_learning \
  --description "Methods that combine IL with active expert querying."

# Then rebuild indexes
python3 .codex/skills/maintain/scripts/rebuild_index.py
```

---

## `maintain/scripts/rebuild_index.py`

**What it does**: Regenerates the `## Auto Topic Index` sections in `content/index.md` and `content/topics/index.md` by scanning all topic directories and building sorted link lists.

**When to run it**:
- After creating or deleting a topic directory
- After bulk changes to topic `index.md` titles
- As part of `/maintain`

**Example**:
```bash
python3 .codex/skills/maintain/scripts/rebuild_index.py
```

No arguments needed; scans the repo root automatically.

---

## `maintain/scripts/lint_repo.py`

**What it does**: Runs four checks and reports issues:
1. Public pages missing `title` or `status` frontmatter
2. Topic directories missing `index.md` or `overview.md`
3. Private paths (`content/local`, `content/drafts`, `inbox`, `raw`, `ops`) accidentally git-tracked
4. Home and topics index files missing auto-section markers

**When to run it**:
- After any batch of `/read`, `/intake`, `/explore`, `/refresh`, or `/synthesize`
- Before publishing / committing

**Example**:
```bash
python3 .codex/skills/maintain/scripts/lint_repo.py
```

Exit code 0 = clean. Exit code 1 = issues found (printed to stdout).

---

## `maintain/scripts/query_context.py`

**What it does**: Ranks all public content pages by relevance to a free-text query and prints a compact context pack (title, description, headings, snippet, score). Useful for loading relevant notes into context before `/discuss` or `/synthesize`.

**When to run it**: At the start of `/discuss` or `/synthesize` to find related notes; or in response to `/search` requests.

**Example**:
```bash
# Public content only
python3 .codex/skills/maintain/scripts/query_context.py \
  "reward shaping temporal difference" \
  --limit 8

# Include local/draft pages
python3 .codex/skills/maintain/scripts/query_context.py \
  "alignment faking interpretability" \
  --limit 10 \
  --include-private
```

Note: `query` is a **positional argument**, not `--query`.

---

## `maintain/scripts/migrate_frontmatter.py`

**What it does**: Backfills `visibility`, `status`, and `description` frontmatter fields for existing content, inferring values from file path and content when fields are absent.

**When to run it**: Only after a frontmatter schema change that adds new required fields to existing files. Do **not** run in normal daily maintenance — it is a bulk migration tool.

**Example**:
```bash
# Preview proposed changes (default is dry-run — safe to run)
python3 .codex/skills/maintain/scripts/migrate_frontmatter.py --only-missing

# Preview for a single topic
python3 .codex/skills/maintain/scripts/migrate_frontmatter.py \
  --only-missing \
  --path content/topics/reinforcement_learning

# Apply changes (add --write to commit)
python3 .codex/skills/maintain/scripts/migrate_frontmatter.py \
  --only-missing \
  --write
```

Flags:
- Default (no flags): dry-run over all `content/topics`
- `--only-missing`: Only touch fields that are currently absent (safer; recommended)
- `--write`: Apply proposed changes; without this flag nothing is written
- `--path <dir-or-file>`: Scope to a specific directory or file

---

## `maintain/scripts/repo_ops.py`

**What it does**: Shared library used by all other scripts. Provides: `slugify`, `parse_frontmatter`, `update_frontmatter`, `ensure_overview_entry`, `archive_pdf`, `normalize_pdf_source`, `append_log`, `rebuild_index`, `git_tracked`, and more.

**When to run it**: Never directly — it is an internal import. Reference it when writing or debugging other scripts.
