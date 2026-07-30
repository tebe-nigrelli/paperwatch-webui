# Release audit — paperwatch 0.6.2

## Result

Version 0.6.2 adds in-place paper jobs to the 0.6.1 interface. Summary and saved-prompt runs preserve scroll position, stream live Ollama output, support cancellation, refresh only the result section, and hide abandoned or failed attempts after restart. Existing fetch, analysis, PDF, topic, tag, export, Codex, and author behavior is retained.


### In-place paper jobs

- Summary and saved-prompt forms submit asynchronously from the paper page.
- The paper page is not reloaded when a job starts or finishes.
- Only the results fragment is refreshed, and the previous vertical scroll position is restored.
- Ollama JSON fragments and chunk/reduction progress are shown live.
- Active jobs can be stopped; queued and active rows are finalized as cancelled.
- Startup marks abandoned jobs inconclusive, while paper pages show only active or successful attempts.

## Usability review

The primary routes require no nested search workflow:

```text
Recent table -> paper detail
Recent filters + saved prompt + model -> query result
Authors table -> filter, tag, pause/resume, archive
```

- Recent opens directly to a wrapping, horizontally resilient paper table.
- Tracked author links immediately re-filter Recent.
- PDF availability and missing state are visible without opening a paper.
- Author management uses one table and compact icon actions.
- Query definitions are separate from paper selection; selection happens once on Recent.
- Light and high-contrast dark themes remain available from the corner button.
- Diagnostics remain separate from the normal reading workflow.

## Added behavior

### Recent-paper interface

- Newest-first table is the default route.
- Columns combine date/journal, wrapping title, tracked-author links, the leading OpenAlex topic, PDF state, and latest summary.
- Current filters can be passed directly to a saved query and selected model.

### Centralized author controls

- A picker exposes every coauthor on the paper page and defaults to the first tracked author in paper order, falling back to the first author.
- Tracked coauthors are highlighted in orange; selected authors can be searched locally, opened on OpenAlex/Scholar/arXiv, or added/restored to tracking.
- Authors without an OpenAlex ID still receive useful name-based local and external searches.
- Removing from the author list only changes tracking flags; no historical data is deleted.

### Themes

- Fixed corner switch for high-contrast light/dark modes.
- Browser persistence with safe operation when local storage is unavailable.
- Explicit `[hidden]` CSS prevents generic button styles from exposing unavailable controls.

### Local-model full-text processing

- Direct, automatic, and chunked modes.
- Paragraph-aware chunks with overlap.
- Structured chunk notes, recursive reduction, and strict final Research Paper Summary.
- User-controlled context size.
- Ollama `num_ctx` receives the selected context value.
- Chunk cache keyed by content/model/prompt/context identity.
- No silent full-paper truncation.
- Complete PDF extraction defaults to a two-million-character safety ceiling before chunking.
- Daily runs preflight the configured local model once instead of producing one identical failure per paper.
- Missing models leave the initial author cursor untouched so the full lookback can be analyzed after installation.
- Native Ollama and OpenAI-compatible local chat endpoints are supported.

### Model reruns

- One or several configured profiles can analyze the same source.
- Stored results remain distinct by provider, model, prompt, scope, and content hash.
- Partial failures preserve successful model outputs.
- Identical work reuses valid cached rows.

### Saved collection prompts

- Editable named queries over matching papers or matching stored summary content.
- Reusable prompt, selected models, and context size.
- Persistent per-model output and error history.
- Recursive map/reduce for larger result sets.
- Oversized nested records are reduced to identity plus bounded content excerpts.
- Query deletion does not affect papers or analyses.

### Author tags

- Many-to-many normalized tags.
- Suggestions from existing visible author tags.
- Tag filtering on the author page.
- Archived authors retain tags without polluting the active suggestion list.


### Codex whole-PDF input

- Codex full-text **Direct** and **Auto** analyses upload the original PDF through `/v1/files` with purpose `user_data`.
- The Responses request references the uploaded `file_id` as an `input_file` beside the strict analysis prompt.
- The temporary OpenAI file is deleted after success or failure.
- No local text extraction or chunking occurs on this path; explicit **Chunk + merge** still uses the local map/reduce pipeline.
- Cache identity includes the original PDF byte hash and reasoning effort.

## Simplicity and debt review

- Existing pipeline, LLM, validation, and persistence logic is reused by the Web UI.
- SQLite remains the only database and queue/status store.
- Background work uses bounded in-process controllers plus cross-process file locks; there is no broker to administer.
- External links are deterministic ID/name searches rather than fragile scraped profile inference.
- Multi-model work is sequential to avoid surprise local GPU or API contention.
- Saved-query processing shares the existing provider abstraction and context-budget logic.
- Schema migrations are incremental through version 5 and retain prior analyses and relationships.
- No `TODO`, `FIXME`, `HACK`, or `XXX` markers remain in source or tests.

## Verification

| Check | Result |
|---|---|
| Deterministic tests | 172 passed |
| Branch-aware coverage | 95.39% |
| Enforced minimum | 95% |
| Python compilation | Passed |
| Generated default config | Parsed successfully |
| Recent-paper-first navigation | Passed |
| Direct paper and author links | Passed |
| Unindexed coauthor behavior | Passed |
| Non-destructive author removal/restoration | Passed |
| Tags and suggestions | Passed |
| High-contrast light/dark UI | Passed |
| Direct/auto/chunked summary modes | Passed |
| Ollama context propagation | Passed |
| Multi-model partial failure/cache behavior | Passed |
| Saved-query CRUD, rerun, and output persistence | Passed |
| Oversized collection-record bounding | Passed |
| Migrations through schema v5 | Passed |
| Cross-process locking | Passed |
| Rendered desktop/dark/mobile visual inspection | Passed |
| Built-wheel contents and generated config | Passed |
| Installed CLI and real local HTTP server | Passed |
| Installed chunk/map/reduce execution and cache reuse | Passed |
| One-year defaults and automatic PDF download | Passed |
| Missing Ollama model preflight and actionable pull command | Passed |
| OpenAI-compatible local-server fallback | Passed |
| Sparse OpenAlex work records skipped without aborting author discovery | Passed |
| SQLite integrity | `ok` |
| systemd unit verification | Passed |

## Operational boundaries

- The Web UI has no authentication and is intended for localhost use.
- Background jobs are process-local after they start; persistent task rows preserve status, but this is not a distributed worker system.
- Models run sequentially by design.
- Google Scholar links are search links; Paperwatch does not scrape Scholar.
- arXiv and OpenAlex links depend on available identifiers.
- Open PDFs are used when known; access controls are never bypassed.
- Model quality and effective context limits still depend on the selected model/runtime.
- Live production credentials were not used in the deterministic release suite.

## OpenAI Codex support

- `provider = "codex"` is a first-class LLM profile type.
- Codex calls use the OpenAI Responses API with the same strict JSON Schema contract as other remote OpenAI models.
- `reasoning_effort` supports `low`, `medium`, `high`, and `xhigh`; omitted Codex effort defaults to `medium`.
- Codex requires `OPENAI_API_KEY` or a profile `api_key`; the application does not invoke the Codex CLI or access its local login state.
- Direct, chunked, multi-model, rerun, and saved-query paths all use the common LLM protocol.
- Reasoning effort is included in direct and chunked cache identities, preventing stale cross-effort reuse.
- Provider-specific authentication, model-access, and HTTP errors are actionable.


## 0.5.6 corrective audit

- The patch is based directly on the uploaded 0.5.5 source tree and preserves its Codex changes.
- `paperwatch backfill --days 365` ignores existing `last_checked_at` cursors and defaults to a 5000-work-per-author safety cap.
- Backfill runs download open PDFs without invoking any LLM.
- Historical imports use `papers.auto_analyze = 0`; routine runs do not summarize the initial backlog.
- A PDF first downloaded during a later normal run is analyzed once.
- Recent-paper cards and paper details expose both downloaded and source PDF links.
- Local PDFs are served inline and remain confined to the configured PDF directory.
- PDF filters and dashboard totals include downloaded files even if the remote URL is absent.
- The missing packaged `defaults/config.toml` is restored, so `paperwatch init` works again.

Verification: 190 tests passed with 95.08% branch-aware coverage.
