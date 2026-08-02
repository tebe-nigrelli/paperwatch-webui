# Release audit — paperwatch 0.9.7

## Result

This revision makes RSS-originated arXiv papers reliable and directly navigable, gives every configured blog its own fetch worker, links query outputs back to their stored source, shortens blog source labels to the site root, removes duplicate fetch-error diagnostics from Content, and restructures destructive controls. Dropdowns with at most five choices cycle and apply immediately; larger dropdowns support typed filtering, keyboard selection, Enter-to-apply, and clearing the current text.

## Verified behavior

### Navigation and layout

- The top bar defaults to the system light/dark preference and exposes a top-right control that cycles through system, light, and dark modes.
- Diagnostics is the final navigation item.
- Runs no longer contains an Operations section.
- OpenAlex paper search, OpenAlex author search, one-year backfill, and manual PDF import are grouped on Discover.
- Recent keeps local-library filtering, result counts, JSON/CSV exports, saved-prompt controls, and paper selection in compact panels.
- Recent JSON/CSV links export the complete filtered result set, independent of the visible page.
- Paper detail groups identifiers, external references, and paper actions in one compact metadata panel while keeping the link groups visually distinct.

### Saved prompts

- A built-in **Summarize** prompt is created automatically, sorted first, editable only for prompt text, and cannot be deleted.
- User-created saved queries store only a name and prompt at the application level; legacy schema columns remain inert for migration compatibility.
- Source, PDF strategy, context-window size, one or more models, filters, and exact paper IDs are selected at run time.
- The former separate “Run selected models” UI is removed; paper detail and Recent use the same saved-prompt execution model.
- Abstract runs require every selected paper to contain an abstract. Full-PDF runs require every selected paper to have a usable local PDF or source URL.
- PDF strategy is accepted only for full-PDF runs; Direct, Auto, and Chunk + merge remain available with bounded map/reduce processing.

### Paper selection and discovery

- Recent-page checkboxes persist selected paper IDs in browser local storage across searches and pagination.
- Visible selected rows sort above unselected rows, and the runner sends the exact retained selection.
- Recent page sizes now include 25, 100, 250, 500, and 1000.
- Discover always has exactly one active Paper or Author mode and renders only that mode's results.
- Discover reviews OpenAlex matches before download and requests at most the API-supported 100 results per search.
- Exact OpenAlex downloads use `OpenAlexClient.get_work()` and validate work identifiers.

### PDFs and desktop integration

- Local PDF links call `xdg-open` without a shell, allowing KDE/system file associations to choose the viewer.
- The existing inline PDF endpoint remains available for compatibility and exports.
- Local paths remain constrained to the configured PDF directory.

### Claude Code

- `provider = "claude-code"` invokes a locally authenticated Claude Code executable in headless, JSON-schema-constrained mode.
- Custom executable paths, model selection, effort, timeout, usage metadata, cancellation, and process cleanup are supported.
- Missing executables, launch failures, nonzero exits, malformed envelopes, missing structured output, schema failures, and hung processes produce actionable errors.

## Compatibility review

- Existing SQLite data migrates incrementally through schema version 20. Hidden initial RSS, GitHub, video, and child-article records from schema 16 are promoted to usable entries.
- Legacy saved-query arguments and `/papers/{id}/pdf` remain accepted for API/backward compatibility.
- Daily fetch, backfill, manual import, summaries, saved prompts, multi-model runs, exports, author tracking, tags, active-run links, PDF validation/redownload, Ollama recovery, OpenAI, and Codex paths remain covered.
- New templates and static assets are included by the existing package-data rules.

## Verification

| Check | Result |
|---|---|
| Deterministic tests | 374 passed |
| Branch-aware coverage | 95.03% |
| Enforced minimum | 95% |
| Python compilation | Passed |
| Wheel build | Passed |
| Clean patch application | Passed |
| Exact OpenAlex work lookup | Passed |
| Protected built-in Summarize prompt | Passed |
| Runtime source/PDF strategy/context/model selection | Passed |
| Persistent exact paper selection | Passed |
| System/light/dark three-state theme control | Passed |
| Mixed-source saved-prompt rejection | Passed |
| Full-filter JSON/CSV export | Passed |
| KDE/system PDF opener path | Passed |
| Claude Code success/error/timeout paths | Passed |

## Operational boundaries

- The Web UI is intended for localhost and has no authentication.
- `xdg-open` acts on the machine running the paperwatch server; this is correct for the local desktop deployment model.
- OpenAlex limits individual search responses to 100 records, while the local Recent table can display up to 1000 records per page.
- Background jobs remain bounded in-process workers protected by cross-process locks, not a distributed queue.
- LLM quality, supported model names, and practical context limits depend on the configured runtime.
