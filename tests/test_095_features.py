from __future__ import annotations

from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.models import stable_hash
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, create_app


class RunStub:
    def __init__(self, *, running: bool = False):
        self.running = running

    def snapshot(self):
        return RunSnapshot(running=self.running)

    def start(self, *args, **kwargs):
        return True


class AnalysisStub:
    def __init__(self, *, running: bool = False):
        self.running = running

    def snapshot(self):
        return AnalysisBatchSnapshot(running=self.running)


class QueryStub:
    def __init__(self, *, running: bool = False):
        self.running = running

    def snapshot(self):
        return QueryBatchSnapshot(running=self.running)

    def start(self, *args, **kwargs):
        return "batch"


class AuthorClient:
    def close(self):
        return None


def _client(config_path, *, running: bool = False) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(running=running),
            analysis_controller=AnalysisStub(running=running),
            query_controller=QueryStub(running=running),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def test_content_type_filter_and_version_brand(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog/",
        )
        db.save_blog_entry(
            blog_id=blog_id,
            title="Only in blogs",
            content_sha256="blog-filter",
            normalized_text="Title: Only in blogs\nURL: https://example.test/blog/post\nBody",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="snapshot",
            external_id="filter-entry",
            source_url="https://example.test/blog/post",
        )

    client = _client(config_path)
    page = client.get("/")
    assert page.status_code == 200
    assert 'data-filter-name="content" data-filter-values="all|papers|blogs"' in page.text
    assert 'data-filter-labels="Both|Papers|Blogs"' in page.text
    assert "paperwatch</span><small>v0.10.2</small>" in page.text
    assert "/static/app.css?v=0.10.2-" in page.text
    assert "/static/app.js?v=0.10.2-" in page.text

    papers = client.get("/?content=papers")
    assert paper_record.title in papers.text
    assert "Only in blogs" not in papers.text
    blogs = client.get("/?content=blogs")
    assert "Only in blogs" in blogs.text
    assert paper_record.title not in blogs.text


def test_delete_summaries_and_fetch_history(
    config_path, paper_record, analysis_result
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        digest = stable_hash("summary")
        analysis_id = db.save_analysis(
            paper_id=paper_id,
            scope="abstract",
            provider="ollama",
            model="test",
            prompt_version="v1",
            content_sha256=digest,
            input_chars=10,
            result=analysis_result,
            usage={},
        )
        task_id = db.create_analysis_tasks(
            batch_id="summary-batch",
            paper_id=paper_id,
            scope="abstract",
            profiles=[("local", "ollama", "test")],
        )[0]
        db.finish_analysis_task(task_id, analysis_id=analysis_id)
        db.save_analysis_chunk(
            paper_id=paper_id,
            provider="ollama",
            model="test",
            prompt_version="v1",
            content_sha256=digest,
            chunk_index=0,
            context_window_tokens=4096,
            summary={"summary": "cached"},
            usage={},
        )
        run_id = db.start_run()
        db.record_error(run_id, stage="fetch", message="example")
        db.finish_run(run_id, "partial", {"errors_count": 1})

    client = _client(config_path)
    runs = client.get("/data")
    assert "Delete all summaries" in runs.text
    assert "Delete fetch history" in runs.text

    response = client.post("/data/delete/summaries", follow_redirects=False)
    assert response.status_code == 303
    assert "Deleted+1+generated+summary" in response.headers["location"]
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id) is not None
        assert db._conn.execute("SELECT count(*) FROM analyses").fetchone()[0] == 0
        assert db._conn.execute("SELECT count(*) FROM analysis_tasks").fetchone()[0] == 0
        assert db._conn.execute("SELECT count(*) FROM analysis_chunks").fetchone()[0] == 0
        assert db.get_run(run_id) is not None

    response = client.post("/data/delete/fetch-history", follow_redirects=False)
    assert response.status_code == 303
    assert "Deleted+1+fetch-history+record" in response.headers["location"]
    with Database(config.paths.database) as db:
        assert db.list_runs() == []
        assert db._conn.execute("SELECT count(*) FROM run_errors").fetchone()[0] == 0
        assert db.get_paper(paper_id) is not None


def test_new_destructive_actions_are_blocked_while_jobs_run(config_path):
    client = _client(config_path, running=True)
    for endpoint in ("/data/delete/summaries", "/data/delete/fetch-history"):
        response = client.post(endpoint, follow_redirects=False)
        assert response.status_code == 303
        assert "Stop+active+jobs" in response.headers["location"]
