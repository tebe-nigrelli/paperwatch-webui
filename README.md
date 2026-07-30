# paperwatch

Paperwatch is a local, daily research-paper tracker for a curated set of authors. It discovers papers through OpenAlex, fills missing abstract and open-PDF metadata from arXiv, stores everything in SQLite, and produces strict structured summaries with Ollama or OpenAI.

Version 0.5.3 is organized around one immediate workflow: open the Web UI, scroll the newest papers, open one, and act from that page. Paper links, author links, tracking controls, model reruns, and exports are kept next to the paper rather than hidden behind nested menus.

## Main flow

```text
tracked OpenAlex authors
  -> daily recent-work query with rolling overlap
  -> identifier-aware deduplication
  -> conditional arXiv abstract/PDF enrichment
  -> SQLite metadata, ordered authorship, and ranked OpenAlex topics
  -> primary-model abstract/full-text analysis
  -> recent-paper feed
  -> optional model reruns, collection prompts, tags, and exports
```

Failures are isolated per author, paper, PDF, model, or saved query. Cross-process locks prevent the timer, manual fetch, and on-demand analysis jobs from modifying the same database concurrently.

## Requirements

- Python 3.11+
- An OpenAlex API key
- Ollama, an OpenAI API key, or metadata-only mode

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

For OpenAI:

```bash
export OPENAI_API_KEY='...'
```

## Start the Web UI

```bash
.venv/bin/paperwatch --config config.toml web
```

Open `http://127.0.0.1:8765`.

The server binds to localhost by default and has no login system. Do not expose it directly to the internet.

## UI structure

### Recent papers

The home page is the paper library, sorted newest first. It provides:

- a continuous, paginated recent-paper feed;
- one-click paper detail, arXiv, Google Scholar, and OpenAlex links when available;
- compact search plus author and OpenAlex-topic filters without a separate search workflow;
- clickable OpenAlex topic tags on every paper with topic-preserving exports;
- direct access to authors, saved queries, fetch history, and manual fetching.

### Paper detail

A paper page centralizes all paper actions:

- complete structured summaries grouped by model and source scope;
- abstract, publication metadata, local PDF, source PDF, JSON export, arXiv, Scholar, and OpenAlex links;
- an author picker for every listed coauthor;
- local paper search for the selected author;
- OpenAlex, Scholar, and arXiv author links adjacent to the author name;
- add, restore, or re-enable tracked authors;
- useful external searches even when a coauthor lacks an OpenAlex identifier;
- rerun one or several configured models against the abstract or full PDF;
- direct, automatic, or explicitly chunked full-text processing;
- a user-selectable context-window size for local-model jobs.

There is no multi-page author drill-down required to perform these actions.

### Authors and tags

The author page supports:

- OpenAlex author search and stable-ID tracking;
- enable, disable, or non-destructively remove from the visible tracking list;
- multiple tags per author;
- existing-tag suggestions while typing;
- filtering the tracked-author list by tag;
- direct OpenAlex, Google Scholar, arXiv, and local-paper links.

`Remove from list` sets `tracked = 0` and `enabled = 0`. It never deletes the author, papers, tags, analyses, or paper-author relationships. An archived author can be restored from a paper page or a new OpenAlex search.

### Saved collection queries

A saved query combines:

- a name;
- a source set: matching stored papers or matching stored summaries;
- a search expression;
- a reusable prompt.

Each query can be edited, rerun with one or several configured models and a chosen context size, or deleted. Outputs and per-model failures are saved in `query_runs` and displayed under the query. Large collections are grouped and recursively reduced so the model receives bounded inputs.

Examples:

- Compare the assumptions used by papers matching `diffusion`.
- Identify common evidence gaps among summaries matching `alignment`.
- Build a reading order for all papers matching an author, method, or title term.

Deleting a saved query removes that query and its query-run outputs; it does not delete papers or paper analyses.

## Light and dark themes

The fixed corner button switches between:

- a high-contrast light theme;
- a high-contrast dark theme.

The choice is stored locally in the browser. If browser storage is unavailable, the UI still works and falls back safely.

## LLM configuration

The primary `[llm]` profile is used by the daily pipeline and appears in Web UI model selectors.

```toml
[llm]
provider = "ollama" # ollama | openai | none
model = "qwen3:8b"
base_url = "http://localhost:11434"
timeout_seconds = 180
max_input_chars = 100000
context_window_tokens = 8192
analyze_pdfs = true
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
```

Profile names must be unique and contain only letters, numbers, dots, dashes, or underscores. Selected models run sequentially to keep local GPU and API load predictable. A failed model does not discard successful outputs from the same batch.

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
download_pdfs = true
max_pdf_chars = 2000000
```

- First successful check: previous 365 publication days.
- Later checks: the last successful author check minus a 30-day overlap.
- Up to 500 works per author per run by default, with cursor pagination.
- Open PDFs are downloaded by default.

The overlap handles delayed indexing and metadata changes. It is configurable because very late indexing of an old publication can otherwise be missed.

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
- `papers`: identifiers, abstract, publication/PDF links, provenance, and fetch dates
- `paper_authors`: complete ordered authorship, including coauthors not currently tracked
- `topics`: OpenAlex topic labels and their domain/field/subfield hierarchy
- `paper_topics`: ranked paper-topic links, relevance scores, and primary-topic state
- `analyses`: named Research Paper Summary fields, canonical JSON, provider/model identity, and usage
- `analysis_tasks`: persistent on-demand multi-model job state
- `analysis_chunks`: cached map/reduce chunk notes
- `author_tags`: normalized tag vocabulary
- `author_tag_links`: many-to-many author tags
- `saved_queries`: editable collection searches and prompts
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
paperwatch --config config.toml run
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
