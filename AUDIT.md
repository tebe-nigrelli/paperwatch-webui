# Release audit — paperwatch 0.5.5

## Result

Version 0.5.5 is release-ready. It keeps the recent-paper-first interface and adds a one-year initial discovery window, default open-PDF downloads, larger complete-PDF extraction limits, one-shot local-model readiness checks, actionable missing-model errors, OpenAI-compatible local-server fallback, and resilient skipping of sparse OpenAlex work records.

## Usability review

The primary route is intentionally shallow:

```text
Recent papers -> Paper detail -> source, author, model, and export actions
```

- The home page is the newest-paper feed rather than an analytics dashboard.
- Each feed card exposes immediate paper/source links.
- The paper page holds the complete summary, source links, author picker, author tracking, external author links, model selection, context size, and rerun controls.
- Authors and saved queries are one top-level navigation step away.
- Diagnostics remain available under Runs and Dashboard but do not interrupt normal reading.
- No essential operation requires a multi-level drill-down.

The UI remains server-rendered Jinja with one CSS file and one small JavaScript file. It has no CDN, bundler, client router, or JavaScript data framework.

## Added behavior

### Recent-paper interface

- Newest-first scrolling feed is the default route.
- Search and optional filters reuse the same feed.
- Paper cards link directly to detail, arXiv, Google Scholar, and OpenAlex where identifiers allow.

### Centralized author controls

- A picker exposes every coauthor on the paper page.
- Selected authors can be searched locally, opened on OpenAlex/Scholar/arXiv, or added/restored to tracking.
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
