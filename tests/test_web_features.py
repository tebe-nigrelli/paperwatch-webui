from __future__ import annotations

from dataclasses import asdict

from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, create_app


class RunStub:
    def snapshot(self):
        return RunSnapshot()

    def start(self):
        return True


class AnalysisStub:
    def __init__(self):
        self.calls = []
        self.state = AnalysisBatchSnapshot()

    def snapshot(self):
        return AnalysisBatchSnapshot(**asdict(self.state))

    def start(
        self,
        paper_id,
        scope,
        profiles,
        analysis_strategy="direct",
        context_window_tokens=None,
    ):
        self.calls.append(
            (paper_id, scope, profiles, analysis_strategy, context_window_tokens)
        )
        return "analysis-batch"


class QueryStub:
    def __init__(self):
        self.calls = []
        self.state = QueryBatchSnapshot()

    def snapshot(self):
        return QueryBatchSnapshot(**asdict(self.state))

    def start(
        self, query_id, profiles, context, *, selection=None,
        source_scope="abstract", pdf_strategy="auto",
    ):
        self.calls.append((query_id, profiles, context, selection, source_scope, pdf_strategy))
        return "query-batch"


class AuthorClient:
    def get_author(self, openalex_id):
        from paperwatch.models import AuthorCandidate

        return AuthorCandidate(openalex_id=openalex_id, name="Added Author")

    def search_authors(self, name, limit=10):
        return []

    def close(self):
        pass


def make_app(config_path, analysis=None, query=None):
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=analysis or AnalysisStub(),
            query_controller=query or QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def seed(config_path, author_candidate, paper_record, analysis_result):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        db.save_analysis(
            paper_id=paper_id,
            scope="fulltext",
            provider="ollama",
            model="test-model",
            prompt_version="v1",
            content_sha256="analysis",
            input_chars=10,
            result=analysis_result,
            usage={},
        )
    return config, paper_id


def test_recent_feed_theme_and_direct_links(
    config_path, author_candidate, paper_record, analysis_result
):
    _, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    client = make_app(config_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Recent papers" in response.text
    assert "Research feed" not in response.text
    assert "Tracked author(s)" in response.text
    assert "OpenAlex topics" in response.text
    assert "File present" in response.text
    assert 'id="theme-toggle"' in response.text
    assert 'data-default-theme="system"' in response.text
    assert "Queries" in response.text
    assert f'/papers/{paper_id}' in response.text
    assert "https://arxiv.org/abs/2607.12345" in response.text
    assert "scholar.google.com" in response.text
    assert "openalex.org/works/w456" in response.text
    js = client.get("/static/app.js").text
    assert "paperwatch-theme" in js
    assert 'dataset.defaultTheme || "system"' in js
    assert 'system: "◐"' in js
    assert 'light: "☀"' in js
    assert 'dark: "☾"' in js
    assert 'storageRemove(themeKey)' in js
    assert "author-picker" in js
    assert "localStorage" in js
    assert "PDF strategy only applies" not in js
    assert "data-paper-processing" in js
    assert 'const wrapCycle = cycle && select.hasAttribute("data-wrap-cycle")' in js
    assert 'targetValue.value = targetValue.value === "selected" ? "results" : "selected"' in js
    assert 'control.setAttribute("aria-disabled", String(locked))' in js
    assert 'select.hasAttribute("data-float-options-up")' in js
    assert 'select.matches("[data-model-picker]") && select.closest(".content-results-runner-row")' in js
    assert 'document.body.appendChild(list)' in js
    assert 'await refreshContentPage(window.location.href, false)' in js
    assert 'response.url || url' in js
    css = client.get("/static/app.css").text
    assert "@media(prefers-color-scheme:dark)" in css
    assert "--bg:#05070a" in css
    assert "[hidden]{display:none!important}" in css
    assert ".discover-mode-option.active" in css
    assert "grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(250px,.9fr)" in css
    assert ".collection-runner-form>.content-results-runner-row{display:flex;flex-direction:row;align-items:flex-start;flex-wrap:nowrap" in css
    assert ".content-results-runner-row>.collection-button-field{display:flex" in css
    assert '.content-results-runner-row .collection-button-field>.button,.content-results-runner-row .search-select-input,.content-results-runner-row input[type="number"]{height:52px' in css
    assert '.content-results-runner-row>.collection-context-window{flex:0 0 205px;min-width:205px}' in css
    assert ".search-select-wrap-button{display:flex" in css
    assert ".search-select-options-floating{position:fixed;z-index:1000}" in css


def test_author_tags_filter_and_non_destructive_removal(
    config_path, author_candidate, paper_record, analysis_result
):
    config, _ = seed(config_path, author_candidate, paper_record, analysis_result)
    client = make_app(config_path)
    response = client.post(
        "/authors/A123/tags", data={"tag": "theory"}, follow_redirects=False
    )
    assert response.status_code == 303
    page = client.get("/authors?tag=theory")
    assert "Ada Lovelace" in page.text
    assert "theory" in page.text
    assert 'list="tag-suggestions"' in page.text
    assert "last checked" not in page.text
    assert "Add tag</button>" not in page.text
    assert 'data-filter-name="enabled"' in page.text
    client.post("/authors/A123/remove", follow_redirects=False)
    with Database(config.paths.database) as db:
        row = db.get_author("A123")
        assert row is not None
        assert row["tracked"] == 0
        assert db.get_paper(1) is not None
        assert db.list_tags() == []


def test_recent_papers_use_searchable_multi_select_author_tags_instead_of_kind(
    config_path, author_candidate, paper_record, analysis_result
):
    config, _ = seed(config_path, author_candidate, paper_record, analysis_result)
    with Database(config.paths.database) as db:
        assert db.add_author_tag("A123", "theory")
        assert db.add_author_tag("A123", "ML")

    client = make_app(config_path)
    response = client.get("/", params=[("author_tag", "theory")])
    assert response.status_code == 200
    assert 'name="author_tag" value="theory"' in response.text
    assert 'name="author" form="content-column-filters" multiple' in response.text
    assert 'aria-label="Filter authors" data-float-options' in response.text
    assert 'aria-label="Filter topics" data-float-options' in response.text
    assert 'name="tag" form="content-column-filters" multiple' in response.text
    assert 'aria-label="Filter tags" data-float-options' in response.text
    assert '<span>Kind</span>' not in response.text
    assert 'name="kind"' not in response.text
    assert "1 matching record" in response.text
    js = client.get("/static/app.js").text
    assert 'select.hasAttribute("data-float-options")' in js
    assert 'details[data-persist-open]' in js
    assert 'storageSet(key, String(details.open))' in js
    assert '[data-author-tag-combobox]' in js
    assert 'event.key === "Enter"' in js
    assert 'addTag(activeOption.dataset.value || "")' in js
    assert '[data-remove-author-tag]' in js


def test_paper_author_picker_and_chunked_multi_model_request(
    config_path, author_candidate, paper_record, analysis_result
):
    _, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    analysis = AnalysisStub()
    client = make_app(config_path, analysis=analysis)
    page = client.get(f"/papers/{paper_id}")
    assert 'id="author-picker"' in page.text
    assert 'id="author-local-papers"' in page.text
    assert 'id="author-local-blogs"' in page.text
    assert 'id="author-track-form"' in page.text
    assert "Run saved prompt" not in page.text
    assert 'id="saved-prompt-form"' not in page.text
    results_page = client.get("/")
    assert "Run saved prompt on all 1 search result" in results_page.text
    assert "Override Defaults" in results_page.text
    runner = results_page.text.split('id="collection-runner-form"', 1)[1].split("</form>", 1)[0]
    assert runner.index("Saved query") < runner.index("Override Defaults")
    assert runner.index("Override Defaults") < runner.index("Models")
    assert "Run selected models" not in page.text
    response = client.post(
        f"/papers/{paper_id}/analyze",
        data={
            "scope": "fulltext",
            "strategy": "chunked",
            "context_window_tokens": "4096",
            "model": ["default", "second"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert analysis.calls == [
        (paper_id, "fulltext", ("default", "second"), "chunked", 4096)
    ]


def test_saved_query_create_edit_run_output_and_delete(
    config_path, author_candidate, paper_record, analysis_result
):
    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    query = QueryStub()
    client = make_app(config_path, query=query)
    response = client.post(
        "/queries",
        data={
            "name": "Compare evidence",
            "prompt": "Compare the strongest evidence.",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    with Database(config.paths.database) as db:
        query_id = int(next(row["id"] for row in db.list_saved_queries() if row["name"] == "Compare evidence"))
        run_id = db.create_query_runs(
            batch_id="existing",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={
                "query": "Minimal",
                "date_from": "2026-01-01",
                "date_to": "2026-12-31",
                "resolved_paper_ids": [paper_id],
            },
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=30,
            output={
                "title": "Comparison",
                "summary": "One paper matched.",
                "items": [
                    {
                        "paper_id": paper_id,
                        "title": paper_record.title,
                        "output": "Evidence is limited to benchmarks.",
                    }
                ],
            },
        )
    page = client.get("/queries")
    assert "Compare evidence" in page.text
    assert "One paper matched" not in page.text
    history = client.get("/queries?mode=history")
    assert "Open result" in history.text
    assert "One paper matched" not in history.text
    result = client.get(f"/queries/runs/{run_id}")
    assert "One paper matched" in result.text
    assert "Search “Minimal”" in history.text
    assert "From 2026-01-01" in history.text
    assert "Select papers and run" in page.text
    assert "Paper search" not in page.text
    recent = client.get("/")
    assert "Run saved prompt on all 1 search result" in recent.text
    assert "Saved query" in recent.text
    assert 'name="override_query_defaults"' in recent.text
    assert "data-query-defaults-form" in recent.text
    assert 'data-query-defaults data-float-options-up' in recent.text
    assert 'class="collection-runner-row content-results-runner-row"' in recent.text
    assert 'class="advanced-filters collection-runner-details content-results-runner-details"' in recent.text
    assert 'content-results-runner-details" open' not in recent.text
    assert 'name="selection_mode"' in recent.text
    assert 'name="selection_mode" value="results" data-run-target-value' in recent.text
    assert 'name="return_to" value="/"' in recent.text
    assert 'type="button" data-run-target-toggle' in recent.text
    assert 'data-selected-count="0"' in recent.text
    assert 'name="paper_processing"' in recent.text
    assert 'name="paper_ids"' not in recent.text
    response = client.post(
        "/queries/run-filtered",
        data={
            "query_id": str(query_id),
            "model": "ignored-model",
            "source_scope": "fulltext",
            "pdf_strategy": "direct",
            "context_window_tokens": "4096",
            "q": "Minimal",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert query.calls[0][:3] == (query_id, (), 16384)
    assert query.calls[0][3]["query"] == "Minimal"
    assert "paper_ids" not in query.calls[0][3]
    assert query.calls[0][4:] == ("auto", "chunked")
    query.calls.clear()
    with Database(config.paths.database) as db:
        assert db.set_content_selected("paper", paper_id, True)
    selected_page = client.get("/")
    assert "Run saved prompt on all 1 search result" in selected_page.text
    assert 'data-selected-count="1"' in selected_page.text
    assert 'name="paper_ids"' not in selected_page.text
    response = client.post(
        "/queries/run-filtered",
        data={
            "query_id": str(query_id),
            "selection_mode": "selected",
            "q": "Does not affect selected mode",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert query.calls[0][3]["paper_ids"] == [paper_id]
    assert query.calls[0][3]["labels"] == ["Selected papers: 1"]
    assert "query" not in query.calls[0][3]
    query.calls.clear()
    with Database(config.paths.database) as db:
        assert db.set_content_selected("paper", paper_id, False)
        assert db.add_author_tag("A123", "theory")
        assert db.add_author_tag("A123", "ML")
    response = client.post(
        "/queries/run-filtered",
        data={
            "query_id": str(query_id),
            "override_query_defaults": "1",
            "model": "default",
            "paper_processing": "chunked",
            "context_window_tokens": "4096",
            "paper_ids": "1",
            "q": "Minimal",
            "author": "A123",
            "author_tag": ["theory", "ML"],
            "topic": "T100",
            "kind": "machine_learning",
            "scope": "fulltext",
            "has_pdf": "yes",
            "analyzed": "yes",
            "date_from": "2026-01-01",
            "date_to": "2026-12-31",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/?notice=Started+saved+query")
    assert "/queries?focus=" not in response.headers["location"]
    assert query.calls[0][:3] == (query_id, ("default",), 4096)
    selection = query.calls[0][3]
    assert "paper_ids" not in selection
    assert query.calls[0][4:] == ("auto", "chunked")
    assert selection["query"] == "Minimal"
    assert selection["author_id"] == "A123"
    assert selection["author_tags"] == ["theory", "ML"]
    assert selection["topic_id"] == "T100"
    assert selection["has_pdf"] is True
    assert selection["labels"] == [
        "Search “Minimal”",
        "Author: Ada Lovelace",
        "Authors: theory, ML",
        "Topic: Machine Learning",
        "Kind: machine learning",
        "Scope: fulltext",
        "PDF available",
        "Analyzed",
        "From 2026-01-01",
        "To 2026-12-31",
    ]
    client.post(
        f"/queries/{query_id}/edit",
        data={
            "name": "Updated query",
            "source_type": "papers",
            "prompt": "Rank methods.",
        },
        follow_redirects=False,
    )
    assert "Updated query" in client.get("/queries").text
    client.post(f"/queries/{query_id}/delete", follow_redirects=False)
    with Database(config.paths.database) as db:
        assert db.get_saved_query(query_id) is None


def test_dashboard_tag_errors_removal_and_json_fallbacks(
    config_path, author_candidate, paper_record, analysis_result
):
    from paperwatch.web import _json_value

    seed(config_path, author_candidate, paper_record, analysis_result)
    client = make_app(config_path)
    assert client.get('/dashboard').status_code == 404
    assert _json_value('', {'fallback': True}) == {'fallback': True}
    assert _json_value('{bad', []) == []

    response = client.post(
        '/authors/A123/tags', data={'tag': '   '}, follow_redirects=False
    )
    assert response.status_code == 303
    assert 'Tag+cannot+be+empty' in response.headers['location']
    assert client.post('/authors/A404/tags', data={'tag': 'theory'}).status_code == 404

    client.post('/authors/A123/tags', data={'tag': 'theory'}, follow_redirects=False)
    response = client.post(
        '/authors/A123/tags/remove', data={'tag': 'theory'}, follow_redirects=False
    )
    assert response.status_code == 303
    with Database(load_config(config_path).paths.database) as db:
        assert db.list_tags() == []


def test_saved_query_validation_missing_routes_busy_and_status(
    config_path, author_candidate, paper_record, analysis_result
):
    config, _ = seed(config_path, author_candidate, paper_record, analysis_result)

    class QueryErrorStub(QueryStub):
        def __init__(self, *, error=None, busy=False):
            super().__init__()
            self.error = error
            self.busy = busy
            self.state = QueryBatchSnapshot(running=busy, batch_id='busy' if busy else None)

        def start(
            self, query_id, profiles, context, *, selection=None,
            source_scope="abstract", pdf_strategy="auto",
        ):
            if self.error:
                raise self.error
            if self.busy:
                return None
            return super().start(
                query_id, profiles, context, selection=selection,
                source_scope=source_scope, pdf_strategy=pdf_strategy,
            )

    client = make_app(config_path)
    response = client.post(
        '/queries',
        data={'name': '', 'source_type': 'bad', 'search_query': '', 'prompt': ''},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert 'Query+name+and+prompt+are+required' in response.headers['location']

    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name='Q', source_type='papers', search_query='', prompt='Compare.'
        )

    response = client.post(
        f'/queries/{query_id}/edit',
        data={'name': '', 'source_type': 'papers', 'search_query': '', 'prompt': ''},
        follow_redirects=False,
    )
    assert 'Query+name+and+prompt+are+required' in response.headers['location']
    assert client.post(
        '/queries/999/edit',
        data={'name': 'Q', 'source_type': 'papers', 'search_query': '', 'prompt': 'P'},
    ).status_code == 404
    assert client.post('/queries/999/delete').status_code == 404

    error_client = make_app(
        config_path, query=QueryErrorStub(error=ValueError('Select at least one configured model'))
    )
    response = error_client.post(
        f'/queries/{query_id}/run', data={}, follow_redirects=False
    )
    assert 'Select+at+least+one+configured+model' in response.headers['location']

    busy_client = make_app(config_path, query=QueryErrorStub(busy=True))
    response = busy_client.post(
        f'/queries/{query_id}/run',
        data={'model': 'default', 'context_window_tokens': '2048'},
        follow_redirects=False,
    )
    assert 'already+running' in response.headers['location']
    status = busy_client.get('/api/query-status').json()
    assert status['running'] is True
    assert status['batch_id'] == 'busy'

    response = client.post('/queries/run-filtered', data={}, follow_redirects=False)
    assert 'Select+a+saved+query' in response.headers['location']
    response = client.post(
        '/queries/run-filtered',
        data={'query_id': str(query_id), 'override_query_defaults': '1', 'model': 'missing'},
        follow_redirects=False,
    )
    assert 'Select+a+configured+model' in response.headers['location']

    response = error_client.post(
        '/queries/run-filtered',
        data={'query_id': str(query_id), 'override_query_defaults': '1', 'model': 'default'},
        follow_redirects=False,
    )
    assert 'Select+at+least+one+configured+model' in response.headers['location']

    response = busy_client.post(
        '/queries/run-filtered',
        data={
            'query_id': str(query_id),
            'override_query_defaults': '1',
            'model': 'default',
            'q': 'Minimal',
        },
        follow_redirects=False,
    )
    assert 'already+running' in response.headers['location']


def test_recent_feed_shows_only_first_openalex_topic(
    config_path, author_candidate, paper_record, analysis_result
):
    seed(config_path, author_candidate, paper_record, analysis_result)
    page = make_app(config_path).get("/")
    assert page.status_code == 200
    topics_cell = page.text.split('<td class="paper-topics-cell">', 1)[1].split('</td>', 1)[0]
    assert "Machine Learning" in topics_cell
    assert "Autonomous Agents" not in topics_cell


def test_paper_author_picker_defaults_to_first_tracked_author_and_marks_it(
    config_path, author_candidate, paper_record, analysis_result
):
    from paperwatch.models import AuthorCandidate

    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    with Database(config.paths.database) as db:
        assert db.remove_author("A123")
        db.add_author(
            AuthorCandidate(
                openalex_id="A999",
                name="Alan Turing",
                institution="Bletchley Park",
            )
        )

    page = make_app(config_path).get(f"/papers/{paper_id}")
    assert page.status_code == 200
    assert '<option value="1" selected class="tracked-author-option"' in page.text
    assert 'href="/?content=papers&amp;author=A999"' in page.text
    assert 'href="/?content=blogs&amp;author=A999"' in page.text
    assert 'class="tag author-chip author-chip-tracked"' in page.text
    assert 'data-author-index="1" aria-pressed="true">Alan Turing</button>' in page.text
    assert 'data-author-index="0" aria-pressed="false">Ada Lovelace</button>' in page.text


def test_paper_author_picker_falls_back_to_first_paper_author(
    config_path, author_candidate, paper_record, analysis_result
):
    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    with Database(config.paths.database) as db:
        assert db.remove_author("A123")

    page = make_app(config_path).get(f"/papers/{paper_id}")
    assert page.status_code == 200
    assert '<option value="0" selected' in page.text
    assert 'data-author-index="0" aria-pressed="true">Ada Lovelace</button>' in page.text


def test_content_summary_change_tracks_summarize_run_and_links_status(
    config_path, author_candidate, paper_record, analysis_result
):
    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    client = make_app(config_path)
    with Database(config.paths.database) as db:
        summary = next(row for row in db.list_saved_queries() if row["name"] == "Summarize")
        run_id = db.create_query_runs(
            batch_id="summary-live",
            query_id=int(summary["id"]),
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={
                "resolved_paper_ids": [paper_id],
                "source_scope": "abstract",
                "pdf_strategy": "auto",
            },
        )[0]

    page = client.get("/")
    assert page.status_code == 200
    assert "Processing…" in page.text
    assert "Summarize" in page.text
    assert f'href="/queries/runs/{run_id}"' in page.text

    detail = client.get(f"/queries/runs/{run_id}")
    assert 'data-query-run-status="queued"' in detail.text
    assert f'action="/queries/runs/{run_id}/cancel"' in detail.text

    with Database(config.paths.database) as db:
        db.start_query_run(run_id)
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=10,
            output={
                "title": "Summary",
                "summary": "Done",
                "items": [
                    {
                        "paper_id": paper_id,
                        "content_type": "paper",
                        "title": paper_record.title,
                        "output": "Done",
                    }
                ],
            },
        )
    page = client.get("/")
    assert "Processing…" not in page.text
    assert "Summary available" in page.text
    assert f'href="/queries/runs/{run_id}"' in page.text


def test_filtered_query_stays_on_content_page_and_shows_skip_warning(
    config_path, author_candidate, paper_record, analysis_result
):
    config, _ = seed(config_path, author_candidate, paper_record, analysis_result)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(name="Filtered warning", prompt="Review")

    class WarningQueryStub(QueryStub):
        def start(self, *args, **kwargs):
            result = super().start(*args, **kwargs)
            self.state.warning = "Skipped 2 incompatible papers without an abstract or PDF."
            return result

    client = make_app(config_path, query=WarningQueryStub())
    response = client.post(
        "/queries/run-filtered",
        data={
            "query_id": str(query_id),
            "return_to": "/?q=Minimal&content=papers",
            "q": "Minimal",
            "content": "papers",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/?q=Minimal&content=papers&notice=")
    assert "level=warning" in location
    assert "/queries" not in location
    page = client.get(location)
    assert 'class="notice notice-warning"' in page.text
    assert "Skipped 2 incompatible papers" in page.text


def test_recent_outputs_can_stop_running_query(
    config_path, author_candidate, paper_record, analysis_result
):
    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)

    class CancellableQueryStub(QueryStub):
        def __init__(self):
            super().__init__()
            self.cancelled = False

        def cancel(self):
            self.cancelled = True
            return True

    query = CancellableQueryStub()
    client = make_app(config_path, query=query)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(name="Stop me", prompt="Review")
        run_id = db.create_query_runs(
            batch_id="stop-me",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"resolved_paper_ids": [paper_id]},
        )[0]
        db.start_query_run(run_id)

    page = client.get("/queries?mode=history")
    assert f'action="/queries/runs/{run_id}/cancel"' in page.text
    response = client.post(f"/queries/runs/{run_id}/cancel", follow_redirects=False)
    assert response.status_code == 303
    assert query.cancelled is True
    assert response.headers["location"].startswith("/queries?mode=history&")
    assert "Stopping+Stop+me" in response.headers["location"]


def test_regexp_print_results_highlight_exact_matches(
    config_path, author_candidate, paper_record, analysis_result
):
    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    client = make_app(config_path)
    pattern = r"(?i)military|defen[cs]e"
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Defense refs",
            prompt="Print regexp matches.",
            regexp_pattern=pattern,
            regexp_radius=2,
            regexp_mode="print",
        )
        run_id = db.create_query_runs(
            batch_id="regexp-highlight",
            query_id=query_id,
            profiles=[("regexp", "local", "regexp")],
            context_window_tokens=2048,
            selection={"resolved_paper_ids": [paper_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=40,
            output={
                "title": f"Regexp matches: {pattern}",
                "summary": "3 match(es) across 1 record(s).",
                "items": [
                    {
                        "paper_id": paper_id,
                        "content_type": "paper",
                        "title": paper_record.title,
                        "output": "Military research and defence work; <script> stays text.",
                    }
                ],
            },
        )

    detail = client.get(f"/queries/runs/{run_id}")
    assert detail.status_code == 200
    assert '<mark class="regexp-match">Military</mark>' in detail.text
    assert '<mark class="regexp-match">defence</mark>' in detail.text
    assert "&lt;script&gt; stays text." in detail.text
    assert "<script> stays text." not in detail.text

    content_detail = client.get(f"/papers/{paper_id}")
    assert content_detail.status_code == 200
    assert "Defense refs" in content_detail.text
    assert "Regexp matches:" not in content_detail.text
    assert pattern not in content_detail.text
    assert "1 matched record" not in content_detail.text
    assert f'href="/queries/{query_id}/edit"' in content_detail.text
    assert "Edit query" in content_detail.text
    assert '<mark class="regexp-match">Military</mark>' in content_detail.text


def test_regexp_print_generated_tag_is_visible_in_content_row_and_tag_picker(
    config_path, author_candidate, paper_record, analysis_result
):
    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Compute and hardware requirements",
            prompt="Print matches.",
            regexp_pattern=r"H100",
            regexp_radius=3,
            regexp_mode="print",
            content_scope="papers",
        )
        run_id = db.create_query_runs(
            batch_id="regexp-picker-backfill",
            query_id=query_id,
            profiles=[("regexp", "local", "regexp")],
            context_window_tokens=2048,
            selection={"resolved_paper_ids": [paper_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=30,
            output={
                "title": "Regexp matches: H100",
                "summary": "1 match(es) across 1 record(s).",
                "items": [
                    {
                        "paper_id": paper_id,
                        "content_type": "paper",
                        "title": paper_record.title,
                        "output": "inference reaches 30 tokens/s on a single H100.",
                    }
                ],
            },
        )

    page = make_app(config_path).get("/")
    assert page.status_code == 200
    tag = "H100"
    assert tag in page.text
    assert f'<option value="{tag}"' in page.text
    assert 'name="tag" form="content-column-filters" multiple' in page.text
    assert 'aria-label="Filter tags" data-float-options' in page.text

    filtered = make_app(config_path).get("/", params={"tag": tag})
    assert filtered.status_code == 200
    assert paper_record.title in filtered.text


def test_paper_saved_query_results_are_localized_to_current_record(
    config_path, author_candidate, paper_record, analysis_result
):
    from dataclasses import replace

    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    sibling_record = replace(
        paper_record,
        openalex_id="W-SIBLING-QUERY",
        doi=None,
        arxiv_id=None,
        title="Sibling batch-query paper",
        publication_url="https://example.test/sibling-query-paper",
        pdf_url=None,
    )
    with Database(config.paths.database) as db:
        sibling_id, _ = db.upsert_paper(sibling_record)
        query_id = db.create_saved_query(name="Batch summary", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="localized-paper-output",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=4096,
            selection={"resolved_paper_ids": [paper_id, sibling_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=2,
            input_chars=100,
            output={
                "title": "Batch result",
                "summary": "COLLECTION-WIDE SUMMARY MUST NOT APPEAR",
                "items": [
                    {
                        "paper_id": paper_id,
                        "blog_entry_id": None,
                        "content_type": "paper",
                        "title": paper_record.title,
                        "output": "CURRENT PAPER RESULT ONLY",
                    },
                    {
                        "paper_id": sibling_id,
                        "blog_entry_id": None,
                        "content_type": "paper",
                        "title": sibling_record.title,
                        "output": "SIBLING PAPER RESULT MUST NOT APPEAR",
                    },
                ],
            },
        )

    client = make_app(config_path)
    page = client.get(f"/papers/{paper_id}")
    assert page.status_code == 200
    assert "CURRENT PAPER RESULT ONLY" in page.text
    assert "SIBLING PAPER RESULT MUST NOT APPEAR" not in page.text
    assert "COLLECTION-WIDE SUMMARY MUST NOT APPEAR" not in page.text


def test_blog_saved_query_results_are_localized_to_current_record(
    config_path, author_candidate, paper_record, analysis_result
):
    config, _ = seed(config_path, author_candidate, paper_record, analysis_result)
    with Database(config.paths.database) as db:
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/localized-blog",
            label="Localized blog",
        )
        current_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Current blog entry",
            content_sha256="current-blog-hash",
            normalized_text="Current text",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
            source_url="https://example.test/localized-blog/current",
        )
        sibling_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Sibling blog entry",
            content_sha256="sibling-blog-hash",
            normalized_text="Sibling text",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
            source_url="https://example.test/localized-blog/sibling",
        )
        query_id = db.create_saved_query(name="Batch blog summary", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="localized-blog-output",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=4096,
            selection={"resolved_blog_entry_ids": [current_id, sibling_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=2,
            input_chars=100,
            output={
                "title": "Batch blog result",
                "summary": "BLOG COLLECTION SUMMARY MUST NOT APPEAR",
                "items": [
                    {
                        "paper_id": None,
                        "blog_entry_id": current_id,
                        "content_type": "blog",
                        "title": "Current blog entry",
                        "output": "CURRENT BLOG RESULT ONLY",
                    },
                    {
                        "paper_id": None,
                        "blog_entry_id": sibling_id,
                        "content_type": "blog",
                        "title": "Sibling blog entry",
                        "output": "SIBLING BLOG RESULT MUST NOT APPEAR",
                    },
                ],
            },
        )

    client = make_app(config_path)
    page = client.get(f"/blogs/{current_id}")
    assert page.status_code == 200
    assert "CURRENT BLOG RESULT ONLY" in page.text
    assert "SIBLING BLOG RESULT MUST NOT APPEAR" not in page.text
    assert "BLOG COLLECTION SUMMARY MUST NOT APPEAR" not in page.text


def test_paper_saved_query_results_fall_back_to_exact_title_when_model_omits_id(
    config_path, author_candidate, paper_record, analysis_result
):
    from dataclasses import replace

    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    sibling_record = replace(
        paper_record,
        openalex_id="W-SIBLING-TITLE-FALLBACK",
        doi=None,
        arxiv_id=None,
        title="Sibling title fallback paper",
        publication_url="https://example.test/sibling-title-fallback",
        pdf_url=None,
    )
    with Database(config.paths.database) as db:
        sibling_id, _ = db.upsert_paper(sibling_record)
        query_id = db.create_saved_query(name="Title fallback summary", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="title-fallback-paper-output",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=4096,
            selection={"resolved_paper_ids": [paper_id, sibling_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=2,
            input_chars=100,
            output={
                "title": "Batch result",
                "summary": "COLLECTION SUMMARY HIDDEN",
                "items": [
                    {
                        "paper_id": None,
                        "blog_entry_id": None,
                        "content_type": "paper",
                        "title": paper_record.title,
                        "output": "RECOVERED BY EXACT TITLE",
                    },
                    {
                        "paper_id": sibling_id,
                        "blog_entry_id": None,
                        "content_type": "paper",
                        "title": sibling_record.title,
                        "output": "SIBLING MUST STAY HIDDEN",
                    },
                ],
            },
        )

    page = make_app(config_path).get(f"/papers/{paper_id}")
    assert page.status_code == 200
    assert "Query results" in page.text
    assert "RECOVERED BY EXACT TITLE" in page.text
    assert "SIBLING MUST STAY HIDDEN" not in page.text
    assert "COLLECTION SUMMARY HIDDEN" not in page.text


def test_paper_saved_query_run_remains_visible_when_model_omits_current_item(
    config_path, author_candidate, paper_record, analysis_result
):
    from dataclasses import replace

    config, paper_id = seed(config_path, author_candidate, paper_record, analysis_result)
    sibling_record = replace(
        paper_record,
        openalex_id="W-SIBLING-MISSING-ITEM",
        doi=None,
        arxiv_id=None,
        title="Sibling missing-item paper",
        publication_url="https://example.test/sibling-missing-item",
        pdf_url=None,
    )
    with Database(config.paths.database) as db:
        sibling_id, _ = db.upsert_paper(sibling_record)
        query_id = db.create_saved_query(name="Missing item summary", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="missing-current-paper-output",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=4096,
            selection={"resolved_paper_ids": [paper_id, sibling_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=2,
            input_chars=100,
            output={
                "title": "Batch result",
                "summary": "COLLECTION SUMMARY HIDDEN",
                "items": [
                    {
                        "paper_id": sibling_id,
                        "blog_entry_id": None,
                        "content_type": "paper",
                        "title": sibling_record.title,
                        "output": "SIBLING MUST STAY HIDDEN",
                    }
                ],
            },
        )

    page = make_app(config_path).get(f"/papers/{paper_id}")
    assert page.status_code == 200
    assert "Missing item summary" in page.text
    assert "did not return a per-record result" not in page.text
    assert "SIBLING MUST STAY HIDDEN" not in page.text
    assert "COLLECTION SUMMARY HIDDEN" not in page.text


def test_blog_saved_query_results_fall_back_to_exact_title_when_model_omits_id(
    config_path, author_candidate, paper_record, analysis_result
):
    config, _ = seed(config_path, author_candidate, paper_record, analysis_result)
    with Database(config.paths.database) as db:
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog-title-fallback",
            label="Blog title fallback",
        )
        current_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Current title fallback blog",
            content_sha256="current-title-fallback-blog",
            normalized_text="Current text",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
            source_url="https://example.test/blog-title-fallback/current",
        )
        sibling_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Sibling title fallback blog",
            content_sha256="sibling-title-fallback-blog",
            normalized_text="Sibling text",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
            source_url="https://example.test/blog-title-fallback/sibling",
        )
        query_id = db.create_saved_query(name="Blog title fallback", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="title-fallback-blog-output",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=4096,
            selection={"resolved_blog_entry_ids": [current_id, sibling_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=2,
            input_chars=100,
            output={
                "title": "Regexp matches: (?i)current",
                "summary": "BLOG COLLECTION SUMMARY HIDDEN",
                "items": [
                    {
                        "paper_id": None,
                        "blog_entry_id": None,
                        "content_type": "blog",
                        "title": "Current title fallback blog",
                        "output": "BLOG RESULT RECOVERED BY TITLE",
                    },
                    {
                        "paper_id": None,
                        "blog_entry_id": sibling_id,
                        "content_type": "blog",
                        "title": "Sibling title fallback blog",
                        "output": "BLOG SIBLING MUST STAY HIDDEN",
                    },
                ],
            },
        )

    page = make_app(config_path).get(f"/blogs/{current_id}")
    assert page.status_code == 200
    assert "Saved query results" in page.text
    assert "BLOG RESULT RECOVERED BY TITLE" in page.text
    assert "BLOG SIBLING MUST STAY HIDDEN" not in page.text
    assert "BLOG COLLECTION SUMMARY HIDDEN" not in page.text
    assert "Regexp matches:" not in page.text
    assert "(?i)current" not in page.text
    assert "2 matched records" not in page.text
    assert f'href="/queries/{query_id}/edit"' in page.text
    assert "Edit query" in page.text


def test_query_page_tables_filter_each_column_and_runner_uses_query_chips(
    config_path, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.upsert_paper(paper_record)
        query_id = db.create_saved_query(name="Filterable query", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="filterable-history",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"paper_ids": [1]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=10,
            output={
                "title": "Done",
                "summary": "Done",
                "items": [{"paper_id": 1, "title": paper_record.title, "output": "Done"}],
            },
        )

    client = make_app(config_path)
    page = client.get("/queries")
    assert page.status_code == 200
    assert 'class="discover-mode-switch query-mode-switch"' in page.text
    assert 'class="discover-mode-option active" href="/queries?mode=saved"' in page.text
    assert "<h2>Saved queries</h2>" in page.text
    assert "<h2>Query history</h2>" not in page.text
    assert page.text.count("data-query-filter-table") == 1
    for key in (
        "name", "type", "supports", "model", "processing", "new-papers",
        "new-blogs", "runs", "latest",
    ):
        assert f'data-table-filter="{key}"' in page.text

    history = client.get("/queries?mode=history")
    assert history.status_code == 200
    assert 'class="discover-mode-option active" href="/queries?mode=history"' in history.text
    assert "<h2>Saved queries</h2>" not in history.text
    assert "<h2>Query history</h2>" in history.text
    assert history.text.count("data-query-filter-table") == 1
    for key in ("query", "model", "status", "records", "created", "output"):
        assert f'data-table-filter="{key}"' in history.text
    assert f'/queries?mode=saved#query-{query_id}' in history.text

    content = client.get("/")
    runner = content.text.split('id="collection-runner-form"', 1)[1].split("</form>", 1)[0]
    assert 'name="query_id" multiple' in runner
    assert 'data-search-placeholder="Search saved queries"' in runner
    assert "selected queries appear as chips below" in runner.lower()
    script = client.get("/static/app.js").text
    assert "search-select-chips" in script
    assert "start_many_on_results" not in script


def test_query_views_are_limited_to_100_rows(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(name="History limit query", prompt="Summarize")
        for index in range(105):
            db.create_saved_query(name=f"Saved limit {index:03d}", prompt="Summarize")
            db.create_query_runs(
                batch_id=f"history-limit-{index:03d}",
                query_id=query_id,
                profiles=[("default", "ollama", "test-model")],
                context_window_tokens=2048,
            )

    client = make_app(config_path)
    saved = client.get("/queries?mode=saved")
    history = client.get("/queries?mode=history")
    assert saved.text.count("data-query-filter-row") == 100
    assert history.text.count("data-query-filter-row") == 100


def test_content_runner_submits_multiple_saved_queries(config_path, paper_record):
    class MultiQueryStub(QueryStub):
        def start_many_on_results(self, *args, **kwargs):
            self.calls.append(("many", args, kwargs))
            return ("batch-one", "batch-two")

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.upsert_paper(paper_record)
        first = db.create_saved_query(name="First multi query", prompt="First")
        second = db.create_saved_query(name="Second multi query", prompt="Second")

    query = MultiQueryStub()
    client = make_app(config_path, query=query)
    response = client.post(
        "/queries/run-filtered",
        data={"query_id": [str(first), str(second)], "q": "Minimal"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "Started+2+saved+queries" in response.headers["location"]
    kind, args, kwargs = query.calls[-1]
    assert kind == "many"
    assert args[0] == (first, second)
    assert args[1] == ()
    assert kwargs["targets"][0][0] == "papers"
    assert kwargs["override_defaults"] is False
