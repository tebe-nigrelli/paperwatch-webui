# Automated testing strategy — paperwatch 0.5.3

The default suite is deterministic and requires no live OpenAlex index, arXiv service, Ollama server, OpenAI account, PDF host, or external browser assets.

```bash
pytest
```

`pyproject.toml` enables branch coverage and fails below 95%.

Release result:

- 172 tests passed
- 95.39% branch-aware coverage
- Python 3.13.5 release environment

## Feature coverage

| Area | Automated checks |
|---|---|
| Configuration | TOML parsing, generated defaults, relative paths, environment keys, ranges, metadata-only mode, multiple profiles, profile names, context windows |
| Author lifecycle | Add/update/re-enable, pause, non-destructive archive, archived-row retention, restoration, coauthor relinking |
| Author tags | Multiple tags, normalization, suggestions, duplicate handling, removal, filtering, archived-author visibility rules |
| Paper author controls | Author picker, local searches, tracked/untracked states, OpenAlex/Scholar/arXiv links, missing OpenAlex IDs, failed author lookup notices |
| OpenAlex | Authentication, author search/get, work filtering/sorting, cursor pagination, retries, malformed responses, sparse title-less work skipping, metadata normalization |
| Deduplication | DOI, arXiv, OpenAlex, title/year, and same-run coauthor cases |
| arXiv enrichment | ID/title lookup, author verification, polite throttling, retries, malformed XML, missing fields, conditional calls |
| PDFs | Safe paths, redirect handling, limits, PDF signatures, atomic writes, cache reuse, extraction failures |
| LLM providers | Ollama/OpenAI strict-schema requests, Ollama `num_ctx`, model readiness, actionable missing-model errors, native/OpenAI-compatible local endpoints, context override, output variants, usage, malformed output, HTTP failures |
| Structured summaries | Exact keys/enums, narrative limits, `Setting: ` requirement, trusted metadata normalization, named-column and JSON round trips |
| Chunked analysis | Paragraph splitting, overlap, chunk-size budgets, direct-fit checks, chunk-note cache, recursive reduction, final strict schema, context failures |
| Multi-model analysis | Direct/auto/chunked modes, same-source runs, partial failures, cache reuse, context overrides, source validation |
| Saved queries | Paper/summary search, summary-content matching, map/reduce grouping, oversized nested record compaction, model success/failure, CRUD, status recovery |
| Pipeline | One-year initial/overlap windows, model preflight, metadata/PDF continuation when a model is missing, automatic PDF chunking, cache behavior, error isolation, counters |
| Concurrency | Separate cross-process fetch, paper-analysis, and saved-query locks; clean release behavior |
| SQLite | Transactions, WAL, foreign keys, schema guards, migrations through schema v5, tag/query/chunk persistence, integrity |
| Recent-paper UI | Newest-first home feed, search/filter redirect, direct paper links, pagination, high-contrast theme control |
| Paper UI | Centralized source/author/model actions, author picker state, model reruns, context input, analysis rendering, exports |
| Author/query UI | Tag suggestions/filtering, non-destructive removal, saved-query edit/rerun/delete, persisted outputs and errors |
| Static resilience | Theme storage fallback, hidden-state CSS precedence, no CDN dependency, responsive templates |
| Web security | CSP, frame denial, MIME-sniff prevention, same-origin referrer policy, protected local-PDF paths, localhost default |
| CLI/systemd | CLI commands, web invocation, daily timer, web service files, archive wording |
| Packaging | Prompt/config/templates/assets/new modules included in wheel; installed command/server smoke tests |

## Test layers

1. **Unit tests** isolate parsing, validation, chunk budgets, schema conversion, URL generation, and database methods.
2. **Service tests** use deterministic HTTP transports and fake LLMs to exercise retries, malformed responses, and partial failures.
3. **Pipeline tests** execute full author-to-paper-to-analysis flows against temporary SQLite databases.
4. **Web integration tests** use FastAPI's test client to submit forms, follow redirects, inspect rendered HTML, poll status APIs, and download exports.
5. **Release smoke tests** install the built wheel, start the actual Uvicorn application, and send real local HTTP requests.
6. **Visual checks** render seeded recent-paper, detail, authors, queries, dark-theme, and mobile pages to inspect hierarchy and responsive behavior.

## Release procedure

```bash
python -m compileall -q src tests
pytest -q --no-cov
pytest -q
python -m build --wheel
# Offline fallback when build isolation cannot download its requirements:
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

The release audit then verifies:

1. wheel contents include templates, static assets, defaults, prompt, and all runtime modules;
2. installation into an isolated environment;
3. generated `config.toml` parses without duplicate keys;
4. installed CLI and Web server start successfully;
5. local HTTP responses for `/`, paper detail, authors, queries, status APIs, exports, CSS, and JavaScript;
6. model selectors, context controls, author actions, tags, and saved-query controls render from the wheel;
7. an installed-package chunk/map/reduce run persists expected rows and reuses its chunk/final caches;
8. a repeated identical run uses caches where applicable;
9. `PRAGMA integrity_check` returns `ok`;
10. `systemd-analyze verify` accepts all supplied units;
11. release SHA-256 checksums match.

## Deliberately excluded live tests

The deterministic suite does not call production OpenAlex, arXiv, Ollama, or OpenAI endpoints. This avoids nondeterminism, credentials, costs, and upstream rate limits. Their current request/response contracts and error paths are represented by controlled fixtures and mock servers. A deployment should still perform one manual credentialed smoke run with its chosen model and network environment.
