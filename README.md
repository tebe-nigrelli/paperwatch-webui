# paperwatch

Paperwatch is a local, daily research-paper tracker for a curated set of authors. It discovers papers through OpenAlex, fills missing abstract and open-PDF metadata from arXiv, stores everything in SQLite, and produces strict structured summaries with Ollama, OpenAI, OpenAI Codex, or Claude Code models.

Version 0.8.4 fetches linked blogs concurrently with paper discovery and PDF archival. It also removes the Diagnostics page and transient fetch banners, and keeps collection prompt controls folded beneath More filters by default. The Queries table remains the sole source of truth for fetch-time model execution; with both automatic columns clear, Fetch now downloads and indexes content without contacting a model.

## Main flow

```text
tracked OpenAlex authors
  -> daily recent-work query with rolling overlap
  -> identifier-aware deduplication
  -> conditional arXiv abstract/PDF enrichment
  -> SQLite metadata, ordered authorship, and ranked OpenAlex topics
  -> local archival of every exposed paper PDF
  -> optional saved-query processing selected in the Web UI
  -> recent-paper feed
  -> optional model reruns, collection prompts, tags, and exports

tracked author blog links
  -> fetched in parallel with papers by the same daily run
  -> RSS/Atom auto-discovery and individual post ingestion
  -> normalized HTML snapshot and hash comparison when no feed works
  -> new feed posts and changed snapshots stored in one blog-update feed
```

Failures are isolated per author, paper, PDF, model, or saved query. Cross-process locks prevent the timer, manual fetch, and on-demand analysis jobs from modifying the same database concurrently.

## Requirements

- Python 3.11+
- An OpenAlex API key
- Ollama, an OpenAI API key, or metadata-only mode
- Optional OpenAI Codex API or locally authenticated Claude Code model access

## Install on Arch Linux

```bash
sudo pacman -S python
cd paperwatch
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/paperwatch init config.toml
export OPENALEX_API_KEY='...'
```

For Ollama:

```bash
ollama serve
ollama pull qwen3:8b
ollama list
```

Ollama is the local model runtime; `qwen3:8b`, `llama3.1:8b`, and similar names are models downloaded into it. The model named in `config.toml` must appear in `ollama list`.

For OpenAI or OpenAI Codex API profiles:

```bash
export OPENAI_API_KEY='...'
```

Paperwatch calls Codex models directly through the OpenAI Responses API. It does not
shell out to the Codex coding-agent CLI and does not read Codex CLI ChatGPT-login
credentials. This keeps paper analysis deterministic, schema-constrained, and free of
filesystem or tool access.

## Start the Web UI

```bash
.venv/bin/paperwatch --config config.toml web
```

Open `http://127.0.0.1:8765`.

The server binds to localhost by default and has no login system. Do not expose it directly to the internet.

## UI structure

### Recent papers

The home page is a newest-first table with these columns:

- publication date and journal;
- wrapping paper title with direct source links;
- tracked authors, plus the number of other authors;
- the leading OpenAlex topic;
- local/source PDF state, with missing PDFs highlighted;
- the latest structured summary.

Tracked author names link directly to the same table filtered to that author. Search, exact author, searchable multi-select author tags, topic, year, PDF, and summary filters stay on this page. Existing tags can be filtered in a dropdown, added with Enter, and removed from the active chip list; selecting several tags matches papers associated with any of those tagged tracked authors. Checkboxes select an exact paper subset for a saved prompt; selections persist in the browser across filter/search changes and selected rows remain at the top whenever they are visible. The table supports page sizes up to 1000 records. Local PDF links ask the desktop through `xdg-open`, so on Arch Linux/KDE they use the configured system PDF application.

### Paper detail

A paper page centralizes all paper actions:

- complete structured summaries grouped by model and source scope;
- abstract, publication metadata, downloaded PDF, source PDF, JSON export, arXiv, Scholar, and OpenAlex links;
- an author picker for every listed coauthor;
- local paper search for the selected author;
- OpenAlex, Scholar, and arXiv author links adjacent to the author name;
- add, restore, or re-enable tracked authors;
- useful external searches even when a coauthor lacks an OpenAlex identifier;
- rerun one or several configured models against the abstract or full PDF;
- direct, automatic, or explicitly chunked full-text processing;
- a user-selectable context-window size for local-model jobs;
- saved prompts beside the summary controls;
- live progress and streamed Ollama output;
- a Stop button for active summary or saved-prompt jobs;
- in-place result refresh without page reload or scroll reset.

There is no multi-page author drill-down required to perform these actions.

### Authors and tags

The Authors page shows tracked authors in a searchable table. The table includes each tracked author’s name, institution, local paper count, tags, external links, and compact icon controls for pausing, resuming, or non-destructively removing tracking. OpenAlex author discovery is on the separate **Discover** page.

- Search matches name, institution/employment, ORCID, OpenAlex ID, and tags.
- The small **Add tag** field submits on Enter and clears after the redirect.
- Multiple tags are supported and existing tags are suggested.
- Discover results can be tracked with one action.

Removing an author sets `tracked = 0` and `enabled = 0`. It never deletes the author, papers, tags, analyses, or paper-author relationships.

Authors can also be created manually. Supplying an OpenAlex author ID enables paper fetching; omitting it creates a paused, blog-only author. Each author has a blog-management page for linking personal, lab, or company sources. The Authors page can import the research-source CSV and match links by exact author name, optionally creating missing blog-only authors.

### Blog updates

Blogs are fetched during every normal paper run, so the systemd timer and manual `paperwatch run` command use one shared interval. Paperwatch automatically discovers RSS or Atom feeds advertised by a linked page, stores each later feed item as an individual update, and uses conditional feed requests on subsequent runs. The first feed synchronization is a hidden baseline. If no usable feed exists—or a known feed temporarily fails—the linked HTML page is normalized and compared as a snapshot instead. Snapshot changes retain unified diffs. Shared URLs are fetched once even when linked to several authors.

### Discover and import

The Discover page contains the network/import operations that are separate from the local library. Exactly one **Paper** or **Author** search mode is active at a time, and only that mode's results are shown:

- search papers by title on OpenAlex and review up to 100 matches before choosing one to download;
- search OpenAlex authors before adding one to tracking;
- import a local PDF with title, authors, source link, and optional abstract/date/venue/DOI/arXiv metadata;
- run the one-year PDF backfill.

### Saved collection queries

Saved queries work on both research papers and blog entries. **Summarize** is created automatically, remains first, and cannot be deleted. Query creation has its own page, and each query stores the complete execution policy: one or more configured model profiles, context-window size, paper input (`Auto`, `Abstract`, or `Full paper PDF`), and PDF strategy (`Auto`, `Direct`, or `Chunk + merge`). Blog runs use the stored feed item or normalized snapshot text and retain any version diff as additional context.

The Queries table has independent **New papers** and **New blog entries** columns. Saving those checkboxes selects the prompts that run after every normal timer or manual fetch. Each selected query runs only on records created by that fetch and uses its own stored models and settings. Initial blog baselines are excluded. When all boxes are clear, a fetch performs no model preflight and no model inference.

To run a saved prompt manually:

1. Create or edit the query and choose its models, context, paper source, and PDF strategy.
2. Filter Recent or open Blogs and optionally select specific rows.
3. Choose the saved prompt and run it.
4. Inspect persistent per-model output on Queries or the individual record page.

The Runs page shows active saved prompts with links to explicitly selected papers or blog entries. Every completed query run records its runtime selection, models, context, output, and per-model errors.

## Light and dark themes

The Web UI follows the operating system’s light or dark preference by default. The top-right theme button cycles between the system preference, light mode, and dark mode. Explicit overrides are retained in the browser.

## LLM configuration

The primary `[llm]` profile appears in Web UI model selectors and is used only when a saved query or explicit manual analysis selects it. Fetch-time model execution is controlled by the **New papers** and **New blog entries** columns. The legacy `analyze_abstracts` and `analyze_pdfs` keys remain accepted for compatibility with explicit structured-analysis callers, but normal timer and Fetch now runs do not invoke them. PDF queries default to extracted text with chunk-and-merge; Codex and Claude Code instead read the complete PDF natively. Recent papers without a usable local PDF are retried for `fetch.retry_missing_pdf_days` days (90 by default).

```toml
[llm]
provider = "ollama" # ollama | openai | codex | claude-code | none
model = "qwen3:8b"
base_url = "http://localhost:11434"
timeout_seconds = 180
max_input_chars = 100000
context_window_tokens = 16384
analyze_abstracts = false
analyze_pdfs = false
pdf_analysis_strategy = "chunked" # auto | direct | chunked
prompt_version = "v2"
```

Additional profiles are optional:

```toml
[[llm_profiles]]
name = "large-local"
provider = "ollama"
model = "qwen3:30b"
base_url = "http://localhost:11434"
timeout_seconds = 300
max_input_chars = 100000
context_window_tokens = 32768
analyze_pdfs = true
prompt_version = "v2"

[[llm_profiles]]
name = "openai-review"
provider = "openai"
model = "YOUR_SUPPORTED_MODEL"
base_url = "https://api.openai.com/v1"
api_key = "" # Prefer OPENAI_API_KEY
context_window_tokens = 32768
analyze_pdfs = true
prompt_version = "v2"

[[llm_profiles]]
name = "codex-review"
provider = "codex"
model = "gpt-5.3-codex"
base_url = "https://api.openai.com/v1"
api_key = "" # Prefer OPENAI_API_KEY
reasoning_effort = "high"
timeout_seconds = 600
max_input_chars = 1000000
context_window_tokens = 400000
analyze_pdfs = true
prompt_version = "v2"

[[llm_profiles]]
name = "claude-review"
provider = "claude-code"
model = "sonnet"
command = "claude"
reasoning_effort = "high"
timeout_seconds = 600
max_input_chars = 1000000
context_window_tokens = 200000
analyze_pdfs = true
prompt_version = "v2"

```

Profile names must be unique and contain only letters, numbers, dots, dashes, or underscores. Codex profiles accept `low`, `medium`, `high`, or `xhigh` reasoning effort. Claude Code profiles use the local `claude` executable and existing login, and accept `low`, `medium`, `high`, `xhigh`, or `max`. Selected models run sequentially to keep local GPU and API load predictable. A failed model does not discard successful outputs from the same batch.

Metadata-only mode is supported:

```toml
[llm]
provider = "none"
model = ""
analyze_pdfs = false
```

## Full-PDF processing and context windows

Paperwatch never labels silently truncated text as a full-paper analysis.

The Web UI offers three full-text modes:

- **Direct**: send the complete extracted text only when the strict prompt and text fit the selected context and input limits.
- **Automatic**: use direct processing when it fits; otherwise use chunked processing.
- **Chunked**: split by paragraph with overlap, summarize each chunk into structured notes, recursively reduce oversized note collections, and create the final strict Research Paper Summary.

For Ollama, the selected context value also sets `num_ctx`; it is not merely an internal chunk-size hint. Chunk outputs are cached by paper, model, prompt, context, and content identity so repeat jobs reuse valid work.

The daily pipeline automatically switches full-text work to chunking when the primary profile cannot fit the direct prompt.

For chunked processing, Paperwatch:

1. extracts the complete PDF text up to `fetch.max_pdf_chars`;
2. divides it into overlapping paragraph-aware chunks sized from `context_window_tokens`;
3. asks the selected model for structured notes for every chunk;
4. recursively merges note groups when they do not fit in one context;
5. produces and stores one final strict Research Paper Summary.

The paper page displays the strategy, context size, and number of chunks used.

## Discovery and fetch window

Defaults:

```toml
[fetch]
initial_lookback_days = 365
overlap_days = 30
max_papers_per_author = 500
download_pdfs = true # legacy compatibility key; available paper PDFs are always archived
max_pdf_chars = 2000000
```

- First successful check: previous 365 publication days.
- Later checks: the last successful author check minus a 30-day overlap.
- Up to 500 works per author per run by default, with cursor pagination.
- Open PDFs are downloaded by default.

The overlap handles delayed indexing and metadata changes. It is configurable because very late indexing of an old publication can otherwise be missed.

### Cursor-independent one-year PDF backfill

Changing `initial_lookback_days` does not rewind authors that already have a
`last_checked_at` cursor. Use the explicit download-only backfill instead:

```bash
uv run paperwatch --config config.toml backfill --days 365
```

The command ignores existing author cursors, requests the previous 365 calendar
days, paginates up to 5000 works per author by default, and downloads every
available open PDF. It does not fetch blogs, run saved queries, or call an LLM. Papers imported by the backfill are
marked as historical so routine overlap runs do not send the backlog to automatic
queries. If a historical paper's PDF becomes available during a later normal run,
the file is downloaded; it is processed only when a selected query targets it.

The same operation is available on the recent-papers page through **Backfill 1
year PDFs**. Local files are linked as **Downloaded PDF** and remote origins as
**Source PDF**.

## Abstract and PDF behavior

- OpenAlex is the primary metadata/discovery source.
- arXiv is queried only when the abstract or direct PDF is missing.
- arXiv title matches are verified with author evidence.
- PDFs are restricted to known open locations; paywalls are never bypassed.
- Downloads are size-limited, signature-checked, and atomically written.
- Extracted text and model outputs are cached where their content identity matches.

`max_pdf_chars` is a safety ceiling, not a chunk size. It must be high enough to extract the complete paper before chunking. The default is two million characters.

## Ollama troubleshooting

Confirm that the service is running and that the configured model exists:

```bash
curl http://localhost:11434/api/version
ollama list
grep -A10 '^\[llm\]' config.toml
```

If the model is missing:

```bash
ollama pull qwen3:8b
```

Or use Llama:

```bash
ollama pull llama3.1:8b
sed -i 's/model = "qwen3:8b"/model = "llama3.1:8b"/' config.toml
```

Test the same endpoint Paperwatch uses:

```bash
curl http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"Return JSON: {\\"ok\\": true}"}],"stream":false,"format":"json"}'
```

Paperwatch performs one model-readiness check at the start of a daily run. If the model is unavailable, it still discovers papers and downloads PDFs, records one actionable error, and leaves the initial author cursor unchanged so the full one-year window can be analyzed after the model is installed. Native Ollama `/api/chat` is preferred; an OpenAI-compatible `/v1/chat/completions` endpoint is used as a fallback for compatible local servers such as llama.cpp.

## OpenAI Codex

Codex support uses the OpenAI Responses API and strict JSON Schema output. Configure it
as an additional model profile so the same paper can be summarized by Ollama and Codex
side by side:

```toml
[[llm_profiles]]
name = "codex-review"
provider = "codex"
model = "gpt-5.3-codex"
base_url = "https://api.openai.com/v1"
api_key = ""
reasoning_effort = "high"
timeout_seconds = 600
max_input_chars = 1000000
context_window_tokens = 400000
analyze_pdfs = true
prompt_version = "v2"
```

Then start Paperwatch with the API key available:

```bash
export OPENAI_API_KEY='...'
uv run paperwatch --config config.toml web
```

On a paper page, select `codex-review` in **Summarize or rerun**. For a full-paper
analysis, **Direct** and **Auto** upload the original PDF to OpenAI with file purpose
`user_data`, attach the returned file ID to the Responses request, and delete the remote
file after the response. The PDF is not replaced with locally extracted text in this
path, so figures, page layout, tables, appendices, and bibliography remain available to
the model. **Chunk + merge** remains an explicit alternative that extracts and processes
text locally in bounded chunks. The reasoning effort and PDF content hash are part of
the analysis cache identity.

## Claude Code CLI

Set `provider = "claude-code"` in an LLM profile to run the locally installed Claude Code CLI in headless, schema-constrained mode. Paperwatch uses the existing Claude Code login, disables tools for the analysis request, and does not require an Anthropic API key. Set `command` when the executable is not named `claude`.

## Exports

The UI supports:

- filtered paper CSV;
- filtered paper JSON;
- filtered paper JSONL;
- individual paper JSON;
- individual analysis JSON;
- source and local PDF access where available.

Exports preserve structured nested analysis data through canonical JSON fields.

## Database tables

- `authors`: stable identity, tracking state, profile metadata, and fetch cursor
- `papers`: identifiers, abstract, publication/PDF links, provenance, fetch dates, and historical auto-analysis state
- `paper_authors`: complete ordered authorship, including coauthors not currently tracked
- `topics`: OpenAlex topic labels and their domain/field/subfield hierarchy
- `paper_topics`: ranked paper-topic links, relevance scores, and primary-topic state
- `analyses`: named Research Paper Summary fields, canonical JSON, provider/model identity, and usage
- `analysis_tasks`: persistent on-demand multi-model job state
- `analysis_chunks`: cached map/reduce chunk notes
- `author_tags`: normalized tag vocabulary
- `author_tag_links`: many-to-many author tags
- `saved_queries`: prompt-only definitions, including the protected built-in Summarize prompt
- `query_runs`: per-model saved-query outputs and failures
- `runs`: daily/manual fetch totals and status
- `run_errors`: isolated pipeline failures with stage context

Schema migrations are automatic and preserve historical rows.

## CLI

```bash
paperwatch --config config.toml author search 'Yoshua Bengio'
paperwatch --config config.toml author add A123456789
paperwatch --config config.toml author list
paperwatch --config config.toml author disable A123456789
paperwatch --config config.toml author enable A123456789
paperwatch --config config.toml author remove A123456789
paperwatch --config config.toml blog add A123456789 https://example.com/blog/ --label 'Personal blog'
paperwatch --config config.toml blog import researcher-blogs.csv --create-missing
paperwatch --config config.toml blog list
paperwatch --config config.toml run
paperwatch --config config.toml backfill --days 365
paperwatch --config config.toml papers
paperwatch --config config.toml runs
paperwatch --config config.toml web --host 127.0.0.1 --port 8765
```

## Daily systemd timer

The supplied units expect the project at `~/.local/share/paperwatch`:

```bash
mkdir -p ~/.local/share ~/.config/paperwatch ~/.config/systemd/user
cp -a . ~/.local/share/paperwatch
cd ~/.local/share/paperwatch
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/paperwatch init ~/.config/paperwatch/config.toml
cp systemd/paperwatch.service systemd/paperwatch.timer systemd/paperwatch-web.service ~/.config/systemd/user/
printf '%s\n' 'OPENALEX_API_KEY=...' > ~/.config/paperwatch/env
chmod 600 ~/.config/paperwatch/env
systemctl --user daemon-reload
systemctl --user enable --now paperwatch.timer paperwatch-web.service
```

The timer runs daily, catches missed executions, and randomizes startup by up to 30 minutes.

## Verification

```bash
python -m compileall -q src tests
pytest
```

The release suite is offline and deterministic, includes branch coverage, and fails below 95%. See [TESTING.md](TESTING.md) and [AUDIT.md](AUDIT.md).
