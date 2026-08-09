from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import Database, SCHEMA_VERSION
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, create_app


class RunStub:
    def snapshot(self):
        return RunSnapshot()

    def start(self, *args, **kwargs):
        return True


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryStub:
    def snapshot(self):
        return QueryBatchSnapshot()

    def start(self, *args, **kwargs):
        return "batch"


class AuthorClient:
    def close(self):
        return None


def client(config_path: Path) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def seed(config_path, author_candidate, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog/",
            label="Example blog",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Selected blog entry",
            content_sha256="selected-entry",
            normalized_text="Title: Selected blog entry\nURL: https://example.test/blog/item\nBody",
            rendered_html="<p>Body</p>",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="selected-entry",
            source_url="https://example.test/blog/item",
        )
    return paper_id, entry_id


def test_schema_content_state_author_star_and_bulk_selection(
    config_path, author_candidate, paper_record
):
    paper_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        assert SCHEMA_VERSION == 32
        assert db.get_paper(paper_id)["read_at"] is None
        assert db.get_blog_entry(entry_id)["read_at"] is None
        assert db.set_author_starred(author_candidate.openalex_id, True)
        assert [row["openalex_id"] for row in db.list_authors_with_tags(starred=True)] == [
            author_candidate.openalex_id
        ]
        assert db.set_content_starred("paper", paper_id, True)
        assert db.set_content_read("blog", entry_id, True)
        assert db.set_content_selected("paper", paper_id, True)
        assert db.set_content_selected("blog", entry_id, True)
        assert db.search_papers(starred=True, selected=True)[1] == 1
        assert len(db.list_blog_entries(read=True, selected=True)) == 1
        assert db.bulk_update_selected_content("star") == 2
        assert db.bulk_update_selected_content("read") == 2
        assert db.bulk_update_selected_content("unselect") == 2
        assert db.search_papers(selected=True)[1] == 0
        assert db.list_blog_entries(selected=True) == []


def test_classifier_query_assigns_shared_tags_and_allows_controlled_edits(
    config_path, author_candidate, paper_record
):
    paper_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Priority classifier",
            prompt="Return a priority for every record.",
            query_kind="classification",
            label_class="Priority",
            label_values=[
                {"value": "High", "pattern": "(?i)high"},
                {"value": "Low", "pattern": "(?i)low"},
            ],
        )
        paper_run = db.create_query_runs(
            batch_id="classification-paper",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )[0]
        blog_run = db.create_query_runs(
            batch_id="classification-blog",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=2048,
            selection={"content_type": "blogs", "blog_entry_ids": [entry_id]},
        )[0]
        changed = db.apply_query_classifications(
            query_id,
            paper_run,
            {"items": [{"paper_id": paper_id, "output": "Priority: high"}]},
        )
        changed += db.apply_query_classifications(
            query_id,
            blog_run,
            {"items": [{"blog_entry_id": entry_id, "output": "This is low priority"}]},
        )
        assert changed == 2
        assert db.search_papers(content_tags=["Priority: High"])[1] == 1
        assert len(db.list_blog_entries(tags=["Priority: Low"])) == 1
        paper_label = db.classification_results_for_content("paper", paper_id)[0]
        assert paper_label["choices"] == ["High", "Low", "STUB"]
        assert db.set_content_classification(query_id, "paper", paper_id, "Low")
        assert db.classification_results_for_content("paper", paper_id)[0]["value"] == "Low"
        with pytest.raises(ValueError, match="not allowed"):
            db.set_content_classification(query_id, "paper", paper_id, "Medium")
        assert db.clear_query_classifications(query_id) == 2
        assert db.classification_results_for_content("paper", paper_id) == []


def test_content_table_colors_tags_by_provenance(
    config_path, author_candidate, paper_record
):
    paper_id, _entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        assert db.add_paper_tag(paper_id, "Written")
        assert db.add_paper_tag(paper_id, "Imported", source="import")
        query_id = db.create_saved_query(
            name="Topic classifier",
            prompt="Choose a topic.",
            query_kind="classification",
            label_class="Topic",
            label_values=[
                {"value": "AI", "pattern": "ai"},
                {"value": "Other", "pattern": "other"},
            ],
        )
        run_id = db.create_query_runs(
            batch_id="tag-provenance",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )[0]
        db.apply_query_classifications(
            query_id, run_id, {"items": [{"paper_id": paper_id, "output": "ai"}]}
        )

    home = client(config_path).get("/")
    assert home.status_code == 200
    assert 'class="tag content-tag-source-user" data-tag-source="user" title="User tag"' in home.text
    assert 'class="tag content-tag-source-import" data-tag-source="import" title="Imported tag"' in home.text
    assert 'class="tag content-tag-source-automated" data-tag-source="automated" title="Automated classification tag"' in home.text


def test_web_star_selection_reading_classifier_and_author_csv_ui(
    config_path, author_candidate, paper_record
):
    paper_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.set_content_selected("paper", paper_id, True)
        query_id = db.create_saved_query(
            name="Kind classifier",
            prompt="Choose a kind.",
            query_kind="classification",
            label_class="Kind",
            label_values=[
                {"value": "Theory", "pattern": "theory"},
                {"value": "Systems", "pattern": "systems"},
            ],
        )
        run_id = db.create_query_runs(
            batch_id="classification-web",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )[0]
        db.apply_query_classifications(
            query_id,
            run_id,
            {"items": [{"paper_id": paper_id, "output": "theory"}]},
        )

    web = client(config_path)
    home = web.get("/")
    assert home.status_code == 200
    assert "Selected content" in home.text
    assert "content-star-cell" in home.text
    assert "content-kind-paper" in home.text and "content-kind-blog" in home.text
    assert "Local research-paper monitor" not in home.text

    paper = web.get(f"/papers/{paper_id}")
    assert 'data-auto-read data-content-type="paper"' in paper.text
    assert "Query tags" in paper.text
    blog = web.get(f"/blogs/{entry_id}")
    assert 'data-auto-read data-content-type="blog"' in blog.text
    assert "Local papers" in blog.text
    assert "Local blogs" in blog.text

    query_edit = web.get(f"/queries/{query_id}/edit")
    assert 'name="query_kind"' in query_edit.text
    assert "Classification tags" in query_edit.text
    query_list = web.get("/queries")
    assert "Classifier" in query_list.text
    assert f'/queries/{query_id}/classifications/delete' in query_list.text

    data_page = web.get("/data")
    assert "Author CSV import" in data_page.text
    assert "Author CSV schema" in data_page.text
    authors = web.get("/authors")
    assert 'name="starred"' in authors.text
    imported = web.post(
        "/authors/import",
        files={
            "csv_file": (
                "authors.csv",
                "Author Name,Institution,OpenAlex ID,Starred\nGrace Hopper,Navy,A456,yes\n",
                "text/csv",
            )
        },
        follow_redirects=False,
    )
    assert imported.status_code == 303
    with Database(config.paths.database) as db:
        grace = db.get_author("A456")
        assert grace is not None and grace["starred"] == 1


def test_new_javascript_behaviour(config_path):
    script = client(config_path).get("/static/app.js").text
    assert "data-auto-read" in script
    assert "5000" in script
    assert "data-classification-only" in script
    assert "data-regexp-only" in script
    assert "data-dialog-open" in script
