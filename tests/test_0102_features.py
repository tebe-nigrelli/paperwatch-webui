from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, create_app


class RunStub:
    def snapshot(self):
        return RunSnapshot()


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryStub:
    def snapshot(self):
        return QueryBatchSnapshot()


class AuthorClient:
    def close(self):
        return None


def web_client(config_path: Path) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def test_content_list_uses_lightweight_rows_and_bounded_selected_preview(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    huge_text = "article body " * 10000
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        for index in range(30):
            entry_id = db.save_blog_entry(
                blog_id=blog_id,
                title=f"Large entry {index:02d}",
                content_sha256=f"large-{index}",
                normalized_text=huge_text,
                rendered_html=f"<p>{huge_text}</p>",
                diff_text=huge_text,
                change_kind="changed",
                previous_entry_id=None,
                entry_kind="rss",
                external_id=f"large-{index}",
                source_url=f"https://example.test/blog/{index}",
            )
            db.set_content_selected("blog", entry_id, True)

        blog_rows = db.list_blog_entries(limit=1, table_view=True)
        assert "normalized_text" not in blog_rows[0].keys()
        assert "rendered_html" not in blog_rows[0].keys()
        assert "diff_text" not in blog_rows[0].keys()

        paper_rows, total = db.search_papers(limit=1, table_view=True)
        assert total == 1
        assert paper_rows[0]["id"] == paper_id
        assert "abstract" not in paper_rows[0].keys()
        assert "metadata_json" not in paper_rows[0].keys()

    response = web_client(config_path).get("/")
    assert response.status_code == 200
    assert 'name="page_size" value="25"' in response.text
    assert "Showing the newest 25 selected records" in response.text
    assert response.text.count('class="content-row content-row-blog') == 25


def test_large_search_selects_are_rendered_lazily():
    script = Path("src/paperwatch/web_static/app.js").read_text(encoding="utf-8")
    assert "const maxRenderedOptions = 80" in script
    assert "Keep typing to narrow the list" in script
    assert "list.replaceChildren(fragment)" in script
