from __future__ import annotations

import json
from dataclasses import replace

from fastapi.testclient import TestClient

from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.llm import StructuredJSONResponse
from paperwatch.models import BatchQueryResult
from paperwatch.query_jobs import QueryController, run_query_for_model
from paperwatch.web import create_app


class _NoopRun:
    def snapshot(self):
        from paperwatch.web import RunSnapshot

        return RunSnapshot()


class _NoopAnalysis:
    def snapshot(self):
        from paperwatch.analysis_jobs import AnalysisBatchSnapshot

        return AnalysisBatchSnapshot()


class _NoopQuery:
    def snapshot(self):
        from paperwatch.query_jobs import QueryBatchSnapshot

        return QueryBatchSnapshot()


class _AuthorClient:
    def close(self):
        pass


class _ChunkLLM:
    provider = "ollama"
    model = "local-test"

    def __init__(self):
        self.calls: list[str] = []

    def generate_json(self, prompt, *, schema, schema_name):
        self.calls.append(prompt)
        result = BatchQueryResult(
            title="Chunk result",
            summary="Processed chunk",
            items=[{"paper_id": 1, "title": "Paper", "output": "chunk output"}],
        )
        return StructuredJSONResponse(data=result.model_dump(), usage={}, raw={})


def test_saved_query_uses_current_config_default_model(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(replace(paper_record, pdf_url=None))
        query_id = db.create_saved_query(name="Runtime default", prompt="Summarize")
        query = db.get_saved_query(query_id)
        assert "model_profile" not in query.keys()
        assert "model_profiles_json" not in query.keys()

    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace('model = "test-model"', 'model = "changed-model"'), encoding="utf-8")

    controller = QueryController(
        config_path, runner=lambda *args, **kwargs: {"completed": 1, "failed": 0}
    )
    batch_id = controller.start(
        query_id,
        (),
        2048,
        selection={"paper_ids": [paper_id]},
        source_scope="auto",
        pdf_strategy="chunked",
    )
    assert batch_id is not None
    controller.wait()
    with Database(config.paths.database) as db:
        run = db.query_runs_for_batch(batch_id)[0]
        assert run["profile"] == "default"
        assert run["model"] == "changed-model"


def test_query_runs_are_created_per_content_record(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        first_id, _ = db.upsert_paper(replace(paper_record, pdf_url=None))
        second_id, _ = db.upsert_paper(
            replace(
                paper_record,
                openalex_id="W-PER-CONTENT-2",
                doi=None,
                arxiv_id=None,
                title="Second per-content paper",
                publication_url="https://example.test/per-content-2",
                pdf_url=None,
            )
        )
        query_id = db.create_saved_query(name="Per content", prompt="Summarize")

    controller = QueryController(
        config_path, runner=lambda *args, **kwargs: {"completed": 2, "failed": 0}
    )
    batch_id = controller.start(
        query_id,
        (),
        2048,
        selection={"paper_ids": [first_id, second_id]},
        source_scope="auto",
        pdf_strategy="chunked",
    )
    assert batch_id is not None
    controller.wait()

    with Database(config.paths.database) as db:
        runs = db.query_runs_for_batch(batch_id)
        assert [run["paper_id"] for run in runs] == [first_id, second_id]
        assert all(run["blog_entry_id"] is None for run in runs)
        selections = [json.loads(run["selection_json"]) for run in runs]
        assert [selection["resolved_paper_ids"] for selection in selections] == [
            [first_id],
            [second_id],
        ]


def test_non_codex_text_always_chunks_before_merging():
    llm = _ChunkLLM()
    result, _usage, _chars = run_query_for_model(
        llm,
        user_prompt="Summarize",
        documents=[{"paper_id": 1, "title": "Paper", "fulltext": "x" * 20000}],
        context_window_tokens=512,
        max_input_chars=2000,
        pdf_strategy="direct",  # ignored for non-Codex providers
    )
    assert len(llm.calls) > 1
    assert result.items
    assert {item.paper_id for item in result.items} == {1}


def test_query_editor_has_only_three_paper_processing_modes(config_path):
    client = TestClient(
        create_app(
            config_path,
            run_controller=_NoopRun(),
            analysis_controller=_NoopAnalysis(),
            query_controller=_NoopQuery(),
            author_client_factory=lambda config: _AuthorClient(),
        )
    )
    page = client.get("/queries/new")
    assert page.status_code == 200
    assert 'name="paper_processing"' in page.text
    assert page.text.count('name="paper_processing"') == 1
    assert 'value="chunked"' in page.text
    assert 'value="whole_text"' in page.text
    assert 'value="pdf"' in page.text
    assert 'name="paper_source_scope"' not in page.text
    assert 'name="pdf_strategy"' not in page.text
    assert 'name="model"' not in page.text
    assert "Context window tokens" not in page.text


def test_regexp_query_uses_match_output_as_editable_query_tag(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(replace(paper_record, pdf_url=None))
        query_id = db.create_saved_query(
            name="Compact match",
            prompt="",
            query_kind="regexp",
            content_scope="papers",
            regexp_pattern=r"(?i)compact\s+agent",
        )

    controller = QueryController(config_path)
    batch_id = controller.start(
        query_id,
        (),
        2048,
        selection={"paper_ids": [paper_id]},
        source_scope="auto",
        pdf_strategy="chunked",
    )
    assert batch_id is not None
    controller.wait()

    with Database(config.paths.database) as db:
        tags = db.classification_results_for_content("paper", paper_id)
        assert len(tags) == 1
        assert tags[0]["freeform"] is True
        assert tags[0]["label_class"] == "Compact match"
        assert tags[0]["value"].casefold() == "compact agent"
        assert tags[0]["tag_name"].casefold() == "compact match: compact agent"

        assert db.set_content_classification(query_id, "paper", paper_id, "Manual value")
        assert db.classification_results_for_content("paper", paper_id)[0]["value"] == "Manual value"
        assert db.clear_query_classifications(query_id) == 1
        assert db.classification_results_for_content("paper", paper_id) == []


def test_stub_classification_value_marks_blog_entry_as_stub(config_path, author_candidate):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Substantive entry",
            content_sha256="substantive-entry",
            normalized_text="This is substantive content. " * 100,
            rendered_html=None,
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="substantive-entry",
            source_url="https://example.test/blog/substantive-entry",
        )
        query_id = db.create_saved_query(
            name="Stub classifier",
            prompt="Classify this entry.",
            query_kind="classification",
            content_scope="blogs",
            label_class="Entry kind",
            label_values=[
                {"value": "Relevant", "pattern": r"(?im)^\s*RESULT:\s*Relevant\s*$"},
                {"value": "Not relevant", "pattern": r"(?im)^\s*RESULT:\s*Not relevant\s*$"},
            ],
        )
        query = db.get_saved_query(query_id)
        values = Database._decoded_label_values(query["label_values_json"])
        assert [value["value"] for value in values][-1] == "STUB"

        run_id = db.create_query_runs(
            batch_id="stub-classification",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={
                "content_type": "blogs",
                "resolved_blog_entry_ids": [entry_id],
                "source_scope": "blog",
                "pdf_strategy": "auto",
            },
        )[0]
        assert db.apply_query_classifications(
            query_id,
            run_id,
            {
                "items": [
                    {
                        "blog_entry_id": entry_id,
                        "content_type": "blog",
                        "title": "Substantive entry",
                        "output": "RESULT: STUB",
                    }
                ]
            },
        ) == 1
        assert db.get_blog_entry(entry_id)["is_stub"] == 1
        assert "Entry kind: STUB" in db.get_blog_entry(entry_id)["content_tags"]


def test_new_paper_fetch_runs_automatic_regexp_query(
    app_config, author_candidate, paper_record
):
    from datetime import date

    import httpx

    from paperwatch.pipeline import PaperWatch

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    class OpenAlex:
        def recent_works(self, author_id, *, since, max_results):
            return [replace(paper_record, pdf_url=None, pdf_source=None)]

        def close(self):
            pass

    config = replace(
        app_config,
        fetch=replace(app_config.fetch, download_pdfs=False),
        llm=replace(app_config.llm, provider="none", analyze_pdfs=False),
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        query_id = db.create_saved_query(
            name="Auto compact",
            prompt="",
            auto_run_papers=True,
            query_kind="regexp",
            content_scope="papers",
            regexp_pattern=r"(?i)compact\s+agent",
            paper_source_scope="abstract",
        )
        app = PaperWatch(
            config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )

        first = app.run(today=date(2026, 8, 8))
        assert len(first.new_paper_ids) == 1
        paper_id = first.new_paper_ids[0]
        runs = [row for row in db.list_query_runs() if int(row["query_id"]) == query_id]
        assert len(runs) == 1
        assert runs[0]["status"] == "success"
        assert runs[0]["paper_id"] == paper_id
        tags = db.classification_results_for_content("paper", paper_id)
        assert [tag["tag_name"].casefold() for tag in tags] == [
            "auto compact: compact agent"
        ]

        second = app.run(today=date(2026, 8, 8))
        assert second.new_paper_ids == []
        assert len(
            [row for row in db.list_query_runs() if int(row["query_id"]) == query_id]
        ) == 1
        app.close()


def test_new_blog_feed_item_runs_automatic_regexp_query(
    app_config, author_candidate, monkeypatch
):
    import hashlib
    from datetime import date

    import httpx

    from paperwatch.blogs import BlogFetch, FeedItem
    from paperwatch.pipeline import PaperWatch

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    class OpenAlex:
        def recent_works(self, author_id, *, since, max_results):
            return []

        def close(self):
            pass

    def feed_item(external_id: str, title: str, text: str) -> FeedItem:
        return FeedItem(
            external_id=external_id,
            title=title,
            url=f"https://example.test/blog/{external_id}",
            published_at="2026-08-08",
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    baseline = feed_item("old", "Old post", "ordinary baseline text")
    new_item = feed_item("new", "New post", "A launch marker appears here")
    fetch_count = 0

    def fake_fetch_blog(client, row):
        nonlocal fetch_count
        fetch_count += 1
        items = (baseline,) if fetch_count == 1 else (new_item, baseline)
        return BlogFetch(
            method="rss",
            title="Blog",
            status_code=200,
            feed_url="https://example.test/blog/feed.xml",
            feed_items=items,
            source_mode="feed",
        )

    monkeypatch.setattr("paperwatch.pipeline.fetch_blog", fake_fetch_blog)
    config = replace(
        app_config,
        fetch=replace(app_config.fetch, download_pdfs=False),
        llm=replace(app_config.llm, provider="none", analyze_pdfs=False),
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        query_id = db.create_saved_query(
            name="Auto launch",
            prompt="",
            auto_run_blogs=True,
            query_kind="regexp",
            content_scope="blogs",
            regexp_pattern=r"(?i)launch\s+marker",
        )
        app = PaperWatch(
            config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )

        first = app.run(today=date(2026, 8, 8))
        assert first.new_blog_entry_ids == []
        assert [
            row for row in db.list_query_runs() if int(row["query_id"]) == query_id
        ] == []

        second = app.run(today=date(2026, 8, 8))
        assert len(second.new_blog_entry_ids) == 1
        entry_id = second.new_blog_entry_ids[0]
        runs = [row for row in db.list_query_runs() if int(row["query_id"]) == query_id]
        assert len(runs) == 1
        assert runs[0]["status"] == "success"
        assert runs[0]["blog_entry_id"] == entry_id
        tags = db.classification_results_for_content("blog", entry_id)
        assert [tag["tag_name"].casefold() for tag in tags] == [
            "auto launch: launch marker"
        ]
        app.close()


def test_new_paper_fetch_runs_automatic_default_model_query(
    app_config, author_candidate, paper_record, monkeypatch
):
    from datetime import date

    import httpx

    from paperwatch.pipeline import PaperWatch

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    class OpenAlex:
        def recent_works(self, author_id, *, since, max_results):
            return [replace(paper_record, pdf_url=None, pdf_source=None)]

        def close(self):
            pass

    calls: list[str] = []

    class QueryLLM:
        provider = "ollama"
        model = "test-model"

        def generate_json(self, prompt, *, schema, schema_name):
            calls.append(prompt)
            result = BatchQueryResult(
                title="Auto summary",
                summary="done",
                items=[
                    {
                        "paper_id": 999,
                        "title": "wrong identity",
                        "output": "automatic output",
                    }
                ],
            )
            return StructuredJSONResponse(data=result.model_dump(), usage={}, raw={})

        def close(self):
            pass

    monkeypatch.setattr("paperwatch.query_jobs.create_llm", lambda profile: QueryLLM())
    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        query_id = db.create_saved_query(
            name="Auto summary",
            prompt="Summarize automatically",
            auto_run_papers=True,
            content_scope="papers",
            paper_source_scope="abstract",
        )
        app = PaperWatch(
            config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )

        stats = app.run(today=date(2026, 8, 8))
        assert len(stats.new_paper_ids) == 1
        paper_id = stats.new_paper_ids[0]
        runs = [row for row in db.list_query_runs() if int(row["query_id"]) == query_id]
        assert len(runs) == 1
        assert runs[0]["status"] == "success"
        assert runs[0]["profile"] == "default"
        assert runs[0]["model"] == "test-model"
        assert runs[0]["paper_id"] == paper_id
        assert calls
        app.close()
