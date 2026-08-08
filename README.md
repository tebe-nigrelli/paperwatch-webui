# paperwatch

Paperwatch is a local, daily research-paper tracker for a curated set of authors. It discovers papers through OpenAlex, fills missing abstract and open-PDF metadata from arXiv, stores everything in SQLite, and produces strict structured summaries with Ollama, OpenAI, OpenAI Codex, or Claude Code models.

Version 0.10.2 makes the Content page memory-bounded. Table queries no longer load stored article bodies, rendered HTML, diffs, abstracts, or metadata; the default page and selected-content preview are limited to 25 records; and large searchable dropdowns render only a small filtered window instead of duplicating every option in the DOM. It also includes the 0.10.1 select-all, notes-filtering, author-table, model-profile, and native PDF fixes.

## Main flow

```text
tracked OpenAlex authors
  -> daily recent-work query with rolling overlap
  -> identifier-aware deduplication
  -> conditional arXiv abstract/PDF enrichment
  -> publisher/repository and Google Scholar direct-PDF fallback
  -> SQLite metadata, ordered authorship, and ranked OpenAlex topics
  -> local archival of every exposed paper PDF
  -> optional saved-query processing selected in the Web UI
  -> recent-paper feed
  -> optional model reruns, collection prompts, tags, and exports

tracked author blog links
  -> fetched in parallel with papers by the same daily run
  -> source-aware RSS/Atom discovery with author/category scope validation
  -> RSS records enriched with the fetched primary article page and clickable media/links
  -> primary arXiv RSS targets imported into the shared paper library
  -> article-index discovery and bounded child-page monitoring
  -> documentation, publication, GitHub, and video source modes
  -> semantic text and meaningful-link comparison
  -> new items and substantial edits stored in one blog-update feed
```

Failures are isolated per author, paper, PDF, model, or saved query. Cross-process locks prevent the timer, manual fetch, and on-demand analysis jobs from modifying the same database concurrently.

## Requirements

- Python 3.11+
- An OpenAlex API key
- Ollama, an OpenAI API key, or metadata-only mode
- Optional locally authenticated Codex CLI or Claude Code model access

## Install

On Arch Linux:

```bash
sudo pacman -S python
cd paperwatch
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/paperwatch init config.toml
export OPENALEX_API_KEY='...'
```

On macOS (with [Homebrew](https://brew.sh/)):

```bash
brew install python
cd paperwatch
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/paperwatch init config.toml
export OPENALEX_API_KEY='...'
```

The application code supports Linux and macOS. The optional unit files in
`systemd/` are Linux-only; on macOS, run the CLI manually or configure the same
`paperwatch run` and `paperwatch web` commands as `launchd` jobs.

For Ollama:

```bash
ollama serve
ollama pull qwen3:8b
ollama list
```

Ollama is the local model runtime; `qwen3:8b`, `llama3.1:8b`, and similar names are models downloaded into it. The model named in `config.toml` must appear in `ollama list`.

For OpenAI API profiles:

```bash
export OPENAI_API_KEY='...'
```

For Codex CLI profiles, install `codex` and sign in once with the account you want
Paperwatch to reuse. `provider = "codex"` invokes `codex exec` in read-only mode and
uses the CLI's existing authentication; Paperwatch does not require `OPENAI_API_KEY`
for that provider.

## Start the Web UI

```bash
.venv/bin/paperwatch --config config.toml web
```

Open `http://127.0.0.1:8765`.

The server binds to localhost by default and has no login system. Do not expose it directly to the internet.

## UI structure

### Downloaded content

The home page is a newest-first unified table for papers and blog entries. Paper rows and blog rows use different backgrounds while sharing search, author filters, date filters, tags, saved queries, and pagination. The table includes:

- publication date and journal;
- wrapping paper title with direct source links;
- tracked authors, plus the number of other authors;
- the leading OpenAlex topic;
- local/source PDF state, with missing PDFs highlighted;
- the latest structured summary or blog change;
- shared paper/blog tags;
- direct links to local PDFs or original blog pages.

Tracked author names link directly to the same table filtered to that author. Every native select is enhanced with a text field: typing filters option labels and values, and Enter inserts the highlighted choice. Search, exact author, searchable multi-select author tags, shared content tags, topic, date, PDF, and analysis filters stay on this page. Checkboxes select exact paper or blog subsets for saved prompts. The table supports page sizes up to 1000 records. Local PDF links use the operating system's desktop opener (`xdg-open` on Linux and `open` on macOS), so they launch the configured PDF application.

Blog detail pages preserve sanitized article structure and show remote HTTP(S) images or GIFs only while the page is open. Paperwatch stores their URLs in article text but never downloads the media into its data directory. Back, original-source, and icon-only copy controls stay on one line.

The Runs page contains guarded data-management actions for deleting authors, downloaded papers/PDFs, or downloaded blog entries; exporting/importing the author-plus-blog catalog as JSON; and exporting/importing the complete SQLite database and PDF store as a ZIP archive.

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

The Authors page shows tracked authors in a searchable table. The table includes each tracked author’s name, editable OpenAlex ID, institution, local paper count, tags, external links, and compact icon controls for pausing, resuming, or non-destructively removing tracking. Edit an OpenAlex ID in place and press Enter to save it. OpenAlex author discovery is on the separate **Discover** page.

- Search matches name, institution/employment, ORCID, OpenAlex ID, and tags.
- The small **Add tag** field submits on Enter and clears after the redirect.
- Multiple tags are supported and existing tags are suggested.
- Discover results can be tracked with one action.

Removing an author sets `tracked = 0` and `enabled = 0`. It never deletes the author, papers, tags, analyses, or paper-author relationships.

Authors can also be created manually. Supplying an OpenAlex author ID enables paper fetching; omitting it creates a paused, blog-only author. Each author has a blog-management page for linking personal, lab, or company sources. The Authors page can import the research-source CSV and match links by exact author name, optionally creating missing blog-only authors.

### Blog updates

Blogs are fetched during every normal paper run, so scheduled and manual `paperwatch run` commands use one shared interval. Each source can use `auto`, `feed`, `article_index`, `article`, `docs`, `publications`, `github`, or `video` mode. Auto-discovered feeds are scored against the configured author/category URL to avoid silently selecting a company-wide feed. Duplicate or homepage-only feed IDs are replaced with stable article identities. Article indexes retain discovered child pages and recheck older pages after they leave the front page. CSS selectors and article-URL patterns can narrow dynamic sites. RSS entries and their fetched website pages are stored separately. Fetched pages retain a sanitized HTML representation for readable headings, paragraphs, lists, tables, code, and inline links, while normalized text remains available for searching and model input. The detail page can copy the complete normalized text to the clipboard. A manual import form creates the complete tracked-author → source → entry relationship. The first synchronization remains a hidden baseline; later meaningful item, body, or outbound-link changes become updates, while counters, avatars, navigation, social links, and tracking parameters are suppressed. Shared URLs are fetched once even when linked to several authors.

### Discover and import

The Discover page contains the network/import operations that are separate from the local library. Exactly one **Paper** or **Author** search mode is active at a time, and only that mode's results are shown:

- search papers by title on OpenAlex and review up to 100 matches before choosing one to download;
- search OpenAlex authors before adding one to tracking;
- import a local PDF with title, authors, source link, and optional abstract/date/venue/DOI/arXiv metadata;
- run the one-year PDF backfill.

### Saved collection queries

Saved queries work on both research papers and blog entries. **Summarize** is created automatically, remains first, and cannot be deleted. Query creation has its own page, and each query stores the complete execution policy: one or more configured model profiles, context-window size, paper input (`Auto`, `Abstract`, or `Full paper PDF`), and PDF strategy (`Auto`, `Direct`, or `Chunk + merge`). An optional regexp is applied to each record’s complete selected text before model execution. Only matching records proceed, and the radius setting keeps the requested number of surrounding words around every match. The query can send those excerpts to the configured models or return the excerpts locally without invoking an LLM. Blog runs use the stored feed item or normalized snapshot text and retain any version diff as additional context.

The Queries table has independent **New papers** and **New blog entries** columns. Saving those checkboxes selects the prompts that run after every normal timer or manual fetch. Each selected query runs only on records created by that fetch and uses its own stored models and settings. Initial blog baselines are excluded. When all boxes are clear, a fetch performs no model preflight and no model inference.

To run a saved prompt manually:

1. Create or edit the query and choose its models, context, paper source, and PDF strategy.
2. Filter Recent or open Blogs and optionally select specific rows.
3. Choose the saved prompt and run it.
4. Inspect persistent per-model output on Queries or the individual record page.

The Runs page shows active saved prompts with links to explicitly selected papers or blog entries. Every completed query run records its runtime selection, models, context, output, and per-model errors.

### Shared content tags

Papers and blog entries use one content-tag namespace. Tags can be added from either detail page, searched as normal text, and selected as filters on Recent papers and Blog updates. Removing the final paper/blog association removes the unused tag while author tags remain a separate author-classification system.

### Fetch history

Each run records paper and blog outcomes separately. The history shows configured author/source links, links successfully read, records with downloadable data, and records actually downloaded. The detailed run page retains the existing error list alongside these counters.

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

Ollama normally receives the Pydantic JSON schema through its structured-output `format` field. Some Ollama/model combinations fail while compiling that schema with `failed to parse grammar`. Paperwatch detects that HTTP 400 response and retries in plain JSON mode, placing the schema in the prompt and validating the returned JSON locally.

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
command = "codex"
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
- When OpenAlex exposes no working PDF, publisher and repository pages are inspected for standard PDF metadata and visible PDF/download links. Common arXiv, Google Drive, ACM, Wiley, and IEEE URL forms are normalized to direct download candidates.
- If those sources fail, Google Scholar is checked for an explicitly exposed direct PDF link. Scholar blocking or a page without a direct link is treated as no result.
- Paywalls and authentication are never bypassed. Every fallback candidate must begin with a PDF signature before it is stored, then the complete download is size-limited, EOF-checked, and atomically written.
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

## Codex CLI

Set `provider = "codex"` to run the installed Codex CLI in headless mode with its
existing local authentication. Paperwatch invokes `codex exec`, supplies the requested
JSON Schema through `--output-schema`, runs the agent in the read-only sandbox, and
captures the final structured response. No OpenAI API key is read for this provider.

```toml
[[llm_profiles]]
name = "codex-review"
provider = "codex"
model = "gpt-5.3-codex"
command = "codex"
reasoning_effort = "high"
timeout_seconds = 600
max_input_chars = 1000000
context_window_tokens = 400000
analyze_pdfs = true
prompt_version = "v2"
```

Run `codex` once and complete its normal login before starting Paperwatch. To switch
the primary `[llm]` section of an existing config without hand-editing TOML:

```bash
paperwatch --config config.toml config provider codex
```

That command preserves the rest of `[llm]`, changes `provider` to `codex`, and sets
`command = "codex"`. Existing `base_url` or `api_key` lines are left in place so they
are available if you switch back to `openai`, but Codex CLI mode ignores them.

For whole-PDF runs, Paperwatch gives the Codex CLI the local PDF path and keeps the
agent read-only. Chunk + merge remains available as the extracted-text alternative.

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
- `blogs`: source configuration, selected feed state, diagnostics, and article scan cursor
- `known_blog_articles`: durable child-article identities and last-fetch state
- `blog_entries`: feed items, discovered articles, snapshots, diffs, and analysis state

Schema migrations are automatic and preserve historical rows.

## CLI

```bash
paperwatch --config config.toml author search 'Yoshua Bengio'
paperwatch --config config.toml author add A123456789
paperwatch --config config.toml author list
paperwatch --config config.toml author disable A123456789
paperwatch --config config.toml author enable A123456789
paperwatch --config config.toml author remove A123456789
paperwatch --config config.toml blog add A123456789 https://example.com/blog/ --label 'Personal blog' --mode article_index --selector main --article-url-pattern '/blog/'
paperwatch --config config.toml blog import researcher-blogs.csv --create-missing
paperwatch --config config.toml blog list
paperwatch --config config.toml run
paperwatch --config config.toml backfill --days 365
paperwatch --config config.toml papers
paperwatch --config config.toml runs
paperwatch --config config.toml web --host 127.0.0.1 --port 8765
```

## Daily systemd timer (Linux)

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
