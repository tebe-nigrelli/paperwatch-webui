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
    assert "PDF file" in response.text
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
    assert "strategy.disabled = !usesPdf" in js
    css = client.get("/static/app.css").text
    assert "@media(prefers-color-scheme:dark)" in css
    assert "--bg:#05070a" in css
    assert "[hidden]{display:none!important}" in css
    assert ".discover-mode-option.active" in css
    assert "grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(250px,.9fr)" in css


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
    assert "Pause / resume" in page.text
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
    assert "Author tags" in response.text
    assert "Search existing tags, then press Enter to add one." in response.text
    assert 'data-author-tag-combobox' in response.text
    assert 'data-author-tag-input role="combobox"' in response.text
    assert 'data-author-tag-chip="theory"' in response.text
    assert 'data-remove-author-tag="theory"' in response.text
    assert 'name="author_tag" value="theory"' in response.text
    assert 'data-value="ML"' in response.text
    assert 'role="listbox"' in response.text
    assert 'data-author-tags-action=' not in response.text
    assert 'type="checkbox" name="author_tag"' not in response.text
    assert 'data-persist-open="paperwatch-more-filters-open" open' in response.text
    assert '<span>Kind</span>' not in response.text
    assert 'name="kind"' not in response.text
    assert "1 matching record" in response.text
    js = client.get("/static/app.js").text
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
    assert 'id="author-local"' in page.text
    assert 'id="author-track-form"' in page.text
    assert "Context window tokens" in page.text
    assert 'name="context_window_tokens" min="512" max="1000000" value="16384"' in page.text
    assert "Chunk + merge" in page.text
    assert "Run saved prompt" in page.text
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
    config, _ = seed(config_path, author_candidate, paper_record, analysis_result)
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
                        "paper_id": 1,
                        "title": paper_record.title,
                        "output": "Evidence is limited to benchmarks.",
                    }
                ],
            },
        )
    page = client.get("/queries")
    assert "Compare evidence" in page.text
    assert "One paper matched" in page.text
    assert "Search “Minimal”" in page.text
    assert "From 2026-01-01" in page.text
    assert "Select papers and run" in page.text
    assert "Paper search" not in page.text
    recent = client.get("/")
    assert "Run on these" in recent.text
    assert "Saved query" in recent.text
    response = client.post(
        "/queries/run-filtered",
        data={
            "query_id": str(query_id),
            "model": "default",
            "source_scope": "fulltext",
            "pdf_strategy": "chunked",
            "context_window_tokens": "4096",
            "paper_ids": "1",
            "q": "Minimal",
            "author": "A123",
            "author_tag": ["theory", "ML"],
            "topic": "T100",
            "kind": "machine_learning",
            "scope": "fulltext",
            "has_pdf": "yes",
            "analyzed": "no",
            "date_from": "2026-01-01",
            "date_to": "2026-12-31",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/queries?focus=")
    assert f"#query-{query_id}" in response.headers["location"]
    assert "notice=Started+saved+query" in response.headers["location"]
    assert query.calls[0][:3] == (query_id, ("default",), 4096)
    selection = query.calls[0][3]
    assert selection["paper_ids"] == [1]
    assert query.calls[0][4:] == ("fulltext", "chunked")
    assert selection["query"] == "Minimal"
    assert selection["author_id"] == "A123"
    assert selection["author_tags"] == ["theory", "ML"]
    assert selection["topic_id"] == "T100"
    assert selection["has_pdf"] is True
    assert selection["labels"] == ["Selected papers: 1"]
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
        data={'query_id': str(query_id), 'model': 'missing'},
        follow_redirects=False,
    )
    assert 'Select+a+configured+model' in response.headers['location']

    response = error_client.post(
        '/queries/run-filtered',
        data={'query_id': str(query_id), 'model': 'default'},
        follow_redirects=False,
    )
    assert 'Select+at+least+one+configured+model' in response.headers['location']

    response = busy_client.post(
        '/queries/run-filtered',
        data={
            'query_id': str(query_id),
            'model': 'default',
            'has_pdf': 'no',
            'analyzed': 'yes',
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
    assert 'href="/?author=A999"' in page.text
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
