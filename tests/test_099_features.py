from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter

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


def seed(config_path: Path, author_candidate, paper_record) -> tuple[int, int, int]:
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog/",
            label="Example blog",
            relationship="Direct author",
            source_mode="article_index",
            content_selector="main",
            article_url_pattern=r"/blog/",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="A stored blog entry",
            content_sha256="stored-blog-entry",
            normalized_text=(
                "Title: A stored blog entry\n"
                "URL: https://example.test/blog/item\n"
                "Stored body"
            ),
            rendered_html="<p>Stored body</p>",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="stored-blog-entry",
            source_url="https://example.test/blog/item",
        )
    return paper_id, blog_id, entry_id


def test_schema_notes_shared_tags_mass_actions_and_catalog_roundtrip(
    tmp_path, config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        assert SCHEMA_VERSION == 32
        author = db.get_author(author_candidate.openalex_id)
        assert author is not None
        assert db.set_object_notes("author", int(author["id"]), "**Author note**")
        assert db.set_object_notes("paper", paper_id, "Paper note")
        assert db.set_object_notes("blog_source", blog_id, "Source note")
        assert db.set_object_notes("blog", entry_id, "Entry note")
        assert db.add_author_tag(author_candidate.openalex_id, "Systems")
        assert db.add_blog_source_tag(blog_id, "Official")
        assert db.add_paper_tag(paper_id, "Keep")
        assert db.add_blog_entry_tag(entry_id, "Keep")
        assert db.set_author_starred(author_candidate.openalex_id, True)
        assert db.set_content_selected("paper", paper_id, True)
        assert db.set_content_selected("blog", entry_id, True)
        changed, action_id = db.bulk_update_selected_tags("add", "Review")
        assert changed == 2 and action_id is not None
        removed, remove_action = db.remove_tag_globally("Review", "content")
        assert removed == 2 and remove_action is not None
        assert db.undo_last_tag_action() == 2
        assert db.search_papers(content_tags=["Review"])[1] == 1
        assert len(db.list_blog_entries(tags=["Review"])) == 1
        exported = db.export_author_blog_catalog()

    target = tmp_path / "catalog.db"
    with Database(target) as db:
        assert db.import_author_blog_catalog(exported) == (1, 1)
        author = db.get_author(author_candidate.openalex_id)
        assert author is not None
        assert author["notes"] == "**Author note**"
        assert author["starred"] == 1
        assert db.list_authors_with_tags(tag="Systems")[0]["openalex_id"] == author_candidate.openalex_id
        imported_blog = db.author_blogs(author_candidate.openalex_id)[0]
        assert imported_blog["source_tags"] == "Official"
        assert imported_blog["notes"] == "Source note"
        assert imported_blog["source_mode"] == "article_index"
        assert imported_blog["content_selector"] == "main"


def test_content_ajax_filters_icons_selection_notes_and_tags(
    config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    web = web_client(config_path)

    initial = web.get("/")
    assert initial.status_code == 200
    assert 'class="panel selected-content-dock"' in initial.text and ' hidden' in initial.text
    assert "content-kind-paper" in initial.text
    assert "content-kind-blog" in initial.text
    for text in (
        "Star filter",
        "Filter matching title text",
        "Filter authors",
        "Filter topics",
        "Filter tags",
        "Filter file presence",
        "Filter summary presence",
    ):
        assert text in initial.text

    response = web.post(
        f"/content/paper/{paper_id}/star",
        data={"starred": "1"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200 and response.json() == {"ok": True, "starred": True}
    response = web.post(
        f"/content/blog/{entry_id}/select",
        data={"selected": "1"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200 and response.json()["selected"] is True
    response = web.post(
        f"/blogs/{entry_id}/tags/add",
        data={"tag": "Important"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200 and response.json()["ok"] is True
    response = web.post(
        f"/notes/blog/{entry_id}",
        data={"notes": "## Remember\nReview this."},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200 and response.json()["ok"] is True

    updated = web.get("/")
    assert "Selected content" in updated.text
    assert "Important" in updated.text
    assert "notes-present" in updated.text
    assert ">Notes<" in updated.text
    assert 'data-read="false"' in updated.text
    assert "content-row[data-read=\"true\"]" in web.get("/static/app.css").text


def test_content_filter_semantics_include_blog_query_summaries(
    tmp_path, config_path, author_candidate, paper_record
):
    paper_id, _blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nbody\n%%EOF\n")
    with Database(config.paths.database) as db:
        db.set_local_pdf(paper_id, str(pdf), hashlib.sha256(pdf.read_bytes()).hexdigest())
        query_id = db.create_saved_query(name="Blog summary", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="blog-summary",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=2048,
            selection={"content_type": "blogs", "blog_entry_ids": [entry_id]},
        )[0]
        db.start_query_run(run_id)
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=10,
            output={"summary": "A summary"},
        )
        assert len(db.list_blog_entries(analyzed=True)) == 1
        assert db.list_blog_entries()[0]["has_summary"] == 1

    web = web_client(config_path)
    summary = web.get("/?content=blogs&analyzed=yes")
    assert "A stored blog entry" in summary.text
    no_summary = web.get("/?content=blogs&analyzed=no")
    assert "A stored blog entry" not in no_summary.text
    file_only = web.get("/?has_pdf=yes")
    assert paper_record.title in file_only.text
    assert "A stored blog entry" not in file_only.text
    topic_only = web.get("/?topic=T100")
    assert paper_record.title in topic_only.text
    assert "A stored blog entry" not in topic_only.text


def test_pdf_hash_modification_marker_and_manual_overwrite(
    tmp_path, config_path, author_candidate, paper_record
):
    paper_id, _blog_id, _entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    existing = tmp_path / "existing.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with existing.open("wb") as handle:
        writer.write(handle)
    original_hash = hashlib.sha256(existing.read_bytes()).hexdigest()
    with Database(config.paths.database) as db:
        db.set_local_pdf(paper_id, str(existing), original_hash)
    existing.write_bytes(existing.read_bytes() + b"modified")

    web = web_client(config_path)
    page = web.get(f"/papers/{paper_id}")
    assert "Modified on disk" in page.text
    assert "Restore PDF" in page.text
    content = web.get("/")
    assert "pdf-modified-tag" in content.text

    replacement = tmp_path / "replacement.pdf"
    replacement_writer = PdfWriter()
    replacement_writer.add_blank_page(width=200, height=200)
    with replacement.open("wb") as handle:
        replacement_writer.write(handle)
    response = web.post(
        f"/papers/{paper_id}/pdf/upload",
        files={"pdf": ("replacement.pdf", replacement.read_bytes(), "application/pdf")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    with Database(config.paths.database) as db:
        paper = db.get_paper(paper_id)
        assert paper is not None
        stored = Path(str(paper["local_pdf_path"]))
        assert stored.is_file()
        assert paper["pdf_sha256"] == hashlib.sha256(stored.read_bytes()).hexdigest()


def test_author_and_blog_csv_roundtrip_preserves_metadata(
    tmp_path, config_path, author_candidate, paper_record
):
    _paper_id, blog_id, _entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        author = db.get_author(author_candidate.openalex_id)
        assert author is not None
        db.set_author_starred(author_candidate.openalex_id, True)
        db.set_author_enabled(author_candidate.openalex_id, False)
        db.add_author_tag(author_candidate.openalex_id, "Tracked")
        db.set_object_notes("author", int(author["id"]), "Author CSV note")
        db.add_blog_source_tag(blog_id, "First-party")
        db.set_object_notes("blog_source", blog_id, "Blog CSV note")
        # Re-enable the author so the source remains visible for export, but keep the
        # source disabled to test that field independently.
        db.set_author_enabled(author_candidate.openalex_id, True)
        db.set_author_blog_broken(author_candidate.openalex_id, blog_id, True)

    web = web_client(config_path)
    authors_csv = web.get("/exports/authors.csv").text
    blogs_csv = web.get("/exports/blogs.csv").text
    author_row = next(csv.DictReader(io.StringIO(authors_csv)))
    assert author_row["Starred"] == "yes"
    assert author_row["Tags"] == "Tracked"
    assert author_row["Notes"] == "Author CSV note"
    blog_row = next(csv.DictReader(io.StringIO(blogs_csv)))
    assert blog_row["Primary tags"] == "First-party"
    assert blog_row["Notes"] == "Blog CSV note"
    assert blog_row["Enabled"] == "no"
    assert blog_row["Primary source mode"] == "article_index"

    target_config = tmp_path / "roundtrip.toml"
    text = config_path.read_text()
    text = text.replace(str(config.paths.database), str(tmp_path / "roundtrip.db"))
    text = text.replace(str(config.paths.pdf_dir), str(tmp_path / "roundtrip-pdfs"))
    target_config.write_text(text)
    target_web = web_client(target_config)
    assert target_web.post(
        "/authors/import",
        files={"csv_file": ("authors.csv", authors_csv, "text/csv")},
        data={"import_tags": "Imported"},
        follow_redirects=False,
    ).status_code == 303
    assert target_web.post(
        "/authors/blogs/import",
        files={"csv_file": ("blogs.csv", blogs_csv, "text/csv")},
        data={"import_tags": "Imported", "create_missing": "1"},
        follow_redirects=False,
    ).status_code == 303
    target = load_config(target_config)
    with Database(target.paths.database) as db:
        imported_author = db.get_author(author_candidate.openalex_id)
        assert imported_author is not None
        assert imported_author["starred"] == 1
        assert imported_author["notes"] == "Author CSV note"
        assert {row["name"] for row in db.list_tags()} == {"Imported", "Tracked"}
        imported_blog = db.author_blogs(author_candidate.openalex_id)[0]
        assert imported_blog["enabled"] == 0
        assert imported_blog["notes"] == "Blog CSV note"
        assert set(imported_blog["source_tags"].split(", ")) == {"First-party", "Imported"}


def test_regexp_debug_dialog_and_ajax_javascript(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Classifier",
            prompt="Classify",
            query_kind="classification",
            label_class="Priority",
            label_values=[
                {"value": "High", "pattern": "high"},
                {"value": "Low", "pattern": "low"},
            ],
        )
    web = web_client(config_path)
    query = web.get(f"/queries/{query_id}/edit")
    assert 'data-regexp-debug="preprocess"' in query.text
    assert 'data-regexp-debug="classification"' in query.text
    assert "Debug answer text" in query.text
    data_page = web.get("/data")
    assert "Blogs CSV schema" in data_page.text
    assert "Author CSV schema" in data_page.text
    script = web.get("/static/app.js").text
    assert '"X-Paperwatch-Async": "1"' in script
    assert "refreshContentPage" in script
    assert "highlightMatches" in script
    css = web.get("/static/app.css").text
    assert "body.dialog-open" in css
    assert "overflow:hidden" in css


def test_async_state_tag_label_and_notes_routes_cover_success_and_errors(
    config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.set_content_selected("paper", paper_id, True)
        db.set_content_selected("blog", entry_id, True)
        query_id = db.create_saved_query(
            name="Route classifier",
            prompt="Classify",
            query_kind="classification",
            label_class="Route",
            label_values=[
                {"value": "One", "pattern": "one"},
                {"value": "Two", "pattern": "two"},
            ],
        )
        run_id = db.create_query_runs(
            batch_id="route-classifier",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=1024,
        )[0]
        db.apply_query_classifications(
            query_id,
            run_id,
            {"items": [{"paper_id": paper_id, "output": "one"}]},
        )

    web = web_client(config_path)
    async_headers = {"X-Paperwatch-Async": "1"}
    for action in ("star", "read", "unread", "unstar"):
        result = web.post(f"/content/selected/{action}", headers=async_headers)
        assert result.status_code == 200 and result.json()["ok"] is True
    invalid = web.post("/content/selected/invalid", headers=async_headers)
    assert invalid.status_code == 400

    added = web.post(
        "/content/selected/tags/add",
        data={"tag": "Bulk"},
        headers=async_headers,
    )
    assert added.status_code == 200 and added.json()["changed"] == 2
    removed = web.post(
        "/content/selected/tags/remove",
        data={"tag": "Bulk"},
        headers=async_headers,
    )
    assert removed.status_code == 200 and removed.json()["changed"] == 2
    empty = web.post(
        "/content/selected/tags/add",
        data={"tag": ""},
        headers=async_headers,
    )
    assert empty.status_code == 400
    web.post(
        "/content/selected/tags/add", data={"tag": "Clear me"}, headers=async_headers
    )
    cleared = web.post("/content/selected/tags/clear", headers=async_headers)
    assert cleared.status_code == 200 and cleared.json()["changed"] == 2

    for path, data in (
        (f"/authors/{author_candidate.openalex_id}/tags", {"tag": "Person"}),
        (f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/tags/add", {"tag": "Source"}),
        (f"/papers/{paper_id}/tags/add", {"tag": "Paper"}),
        (f"/blogs/{entry_id}/tags/add", {"tag": "Blog"}),
    ):
        response = web.post(path, data=data, headers=async_headers)
        assert response.status_code == 200 and response.json()["ok"] is True

    globally_removed = web.post(
        "/tags/remove-global",
        data={"tag": "Person", "scope": "authors"},
        headers=async_headers,
    )
    assert globally_removed.status_code == 200 and globally_removed.json()["changed"] == 1
    invalid_scope = web.post(
        "/tags/remove-global",
        data={"tag": "Source", "scope": "invalid"},
        headers=async_headers,
    )
    assert invalid_scope.status_code == 400
    undone = web.post("/tags/undo", headers=async_headers)
    assert undone.status_code == 200 and undone.json()["changed"] == 1

    for path, data in (
        (f"/authors/{author_candidate.openalex_id}/tags/remove", {"tag": "Person"}),
        (f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/tags/remove", {"tag": "Source"}),
        (f"/papers/{paper_id}/tags/remove", {"tag": "Paper"}),
        (f"/blogs/{entry_id}/tags/remove", {"tag": "Blog"}),
    ):
        response = web.post(path, data=data, headers=async_headers)
        assert response.status_code == 200 and response.json()["ok"] is True

    label = web.post(
        f"/content/paper/{paper_id}/classifications/{query_id}",
        data={"value": "Two"},
        headers=async_headers,
    )
    assert label.status_code == 200 and label.json()["value"] == "Two"
    bad_label = web.post(
        f"/content/paper/{paper_id}/classifications/{query_id}",
        data={"value": "Three"},
        headers=async_headers,
    )
    assert bad_label.status_code == 400
    bad_notes = web.post(
        f"/notes/unknown/{paper_id}", data={"notes": "x"}, headers=async_headers
    )
    assert bad_notes.status_code == 400

    # Cover redirect fallbacks used when JavaScript is unavailable.
    assert web.post(
        f"/content/paper/{paper_id}/read",
        data={"read": "1", "return_to": "/"},
        follow_redirects=False,
    ).status_code == 303
    assert web.post(
        f"/authors/{author_candidate.openalex_id}/star",
        data={"starred": "1", "return_to": "/authors"},
        follow_redirects=False,
    ).status_code == 303


def test_tag_database_edge_cases_and_local_catalog(tmp_path, config_path, author_candidate, paper_record):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        assert db.blog_source_tags(blog_id) == []
        assert not db.add_paper_tag(999999, "Missing")
        assert not db.remove_paper_tag(paper_id, "Missing")
        assert not db.remove_blog_entry_tag(entry_id, "Missing")
        assert not db.remove_blog_source_tag(blog_id, "Missing")
        assert db.undo_last_tag_action() == 0
        try:
            db.set_object_notes("unknown", paper_id, "x")
        except ValueError as exc:
            assert "Unknown" in str(exc)
        else:
            raise AssertionError("unknown notes type should fail")
        try:
            db.set_object_notes("paper", paper_id, "x" * 200_001)
        except ValueError as exc:
            assert "too long" in str(exc)
        else:
            raise AssertionError("oversized notes should fail")
        try:
            db.bulk_update_selected_tags("invalid", "Tag")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid bulk tag action should fail")
        try:
            db.remove_tag_globally("Tag", "invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid global scope should fail")

        db.set_content_selected("paper", paper_id, True)
        db.set_content_selected("blog", entry_id, True)
        db.add_paper_tag(paper_id, "Existing")
        db.add_blog_entry_tag(entry_id, "Existing")
        changed, _ = db.bulk_update_selected_tags("add", "Existing")
        assert changed == 0
        changed, _ = db.bulk_update_selected_tags("remove", "Existing")
        assert changed == 2
        assert db.undo_last_tag_action() == 2
        changed, _ = db.bulk_update_selected_tags("clear")
        assert changed == 2
        assert db.undo_last_tag_action() == 2

        local_id = db.add_manual_author("Local Person", "Local Lab")
        local = db.find_authors_by_name("Local Person")[0]
        db.add_author_tag(str(local["openalex_id"]), "Local")
        db.set_object_notes("author", local_id, "Local note")
        local_blog = db.add_author_blog(str(local["openalex_id"]), "https://local.example/blog")
        db.add_blog_source_tag(local_blog, "Local source")
        catalog = db.export_author_blog_catalog()

    with Database(tmp_path / "local-catalog.db") as db:
        authors, links = db.import_author_blog_catalog(catalog)
        assert authors == 2 and links == 2
        imported = db.find_authors_by_name("Local Person")
        assert len(imported) == 1
        assert imported[0]["notes"] == "Local note"
        assert db.list_authors_with_tags(tag="Local")[0]["name"] == "Local Person"


def test_manual_import_and_pdf_restore_paths(
    monkeypatch, tmp_path, config_path, author_candidate, paper_record
):
    paper_id, _blog_id, _entry_id = seed(config_path, author_candidate, paper_record)
    web = web_client(config_path)

    missing_name = web.post(
        "/authors/manual", data={"name": ""}, follow_redirects=False
    )
    assert missing_name.status_code == 303 and "level=error" in missing_name.headers["location"]
    manual = web.post(
        "/authors/manual",
        data={
            "name": "Manual Person",
            "institution": "Manual Lab",
            "tags": "Imported, Manual",
            "notes": "Manual note",
        },
        follow_redirects=False,
    )
    assert manual.status_code == 303
    missing_author = web.post(
        "/authors/blogs/manual", data={"url": "https://example.org"}, follow_redirects=False
    )
    assert missing_author.status_code == 303 and "level=error" in missing_author.headers["location"]
    linked = web.post(
        "/authors/blogs/manual",
        data={"openalex_id": author_candidate.openalex_id, "url": "https://second.example/blog", "tags": "New"},
        follow_redirects=False,
    )
    assert linked.status_code == 303

    no_file = web.post(f"/papers/{paper_id}/pdf/upload", follow_redirects=False)
    assert no_file.status_code == 303 and "level=error" in no_file.headers["location"]

    def fake_download(_url: str, destination: Path, **_kwargs):
        writer = PdfWriter()
        writer.add_blank_page(width=80, height=80)
        with Path(destination).open("wb") as handle:
            writer.write(handle)

    monkeypatch.setattr("paperwatch.web.download_pdf", fake_download)
    restored = web.post(f"/papers/{paper_id}/pdf/restore", follow_redirects=False)
    assert restored.status_code == 303 and "PDF+restored" in restored.headers["location"]
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        stored = db.get_paper(paper_id)
        assert stored is not None and Path(str(stored["local_pdf_path"])).is_file()
        assert stored["pdf_sha256"]

    from paperwatch.pdfs import PDFError

    def failing_download(_url: str, destination: Path, **_kwargs):
        Path(destination).write_bytes(b"broken")
        raise PDFError("download failed")

    monkeypatch.setattr("paperwatch.web.download_pdf", failing_download)
    failed = web.post(f"/papers/{paper_id}/pdf/restore", follow_redirects=False)
    assert failed.status_code == 303 and "level=error" in failed.headers["location"]


def test_catalog_and_full_data_web_exports_imports(tmp_path, config_path, author_candidate, paper_record):
    paper_id, _blog_id, _entry_id = seed(config_path, author_candidate, paper_record)
    web = web_client(config_path)
    catalog = web.get("/data/export/authors-blogs")
    assert catalog.status_code == 200
    assert catalog.json()["format"] == "paperwatch-authors-blogs"
    imported = web.post(
        "/data/import/authors-blogs",
        files={"file": ("catalog.json", catalog.content, "application/json")},
        follow_redirects=False,
    )
    assert imported.status_code == 303
    invalid = web.post(
        "/data/import/authors-blogs",
        files={"file": ("bad.json", b"[]", "application/json")},
        follow_redirects=False,
    )
    assert invalid.status_code == 303 and "level=error" in invalid.headers["location"]

    archive = web.get("/data/export/all")
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.set_object_notes("paper", paper_id, "changed after export")
    restored = web.post(
        "/data/import/all",
        files={"file": ("backup.zip", archive.content, "application/zip")},
        follow_redirects=False,
    )
    assert restored.status_code == 303
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id)["notes"] == ""
    bad_archive = web.post(
        "/data/import/all",
        files={"file": ("bad.zip", b"not a zip", "application/zip")},
        follow_redirects=False,
    )
    assert bad_archive.status_code == 303 and "level=error" in bad_archive.headers["location"]


def test_nonjavascript_and_error_branches_for_new_routes(
    config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Error classifier",
            prompt="Classify",
            query_kind="classification",
            label_class="Error",
            label_values=[
                {"value": "Yes", "pattern": "yes"},
                {"value": "No", "pattern": "no"},
            ],
        )
        run_id = db.create_query_runs(
            batch_id="error-classifier",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=1024,
        )[0]
        db.apply_query_classifications(
            query_id, run_id, {"items": [{"blog_entry_id": entry_id, "output": "yes"}]}
        )

    web = web_client(config_path)
    async_headers = {"X-Paperwatch-Async": "1"}
    for state, data in (
        ("star", {"starred": "1"}),
        ("read", {"read": "1"}),
        ("select", {"selected": "1"}),
    ):
        assert web.post(
            f"/content/paper/999999/{state}", data=data, headers=async_headers
        ).status_code == 404
        assert web.post(
            f"/content/blog/{entry_id}/{state}", data=data, follow_redirects=False
        ).status_code == 303
    assert web.post(
        "/content/selected/not-an-action", follow_redirects=False
    ).status_code == 400
    assert web.post(
        "/content/selected/tags/add", data={"tag": ""}, follow_redirects=False
    ).status_code == 303
    assert web.post(
        "/tags/remove-global",
        data={"tag": "x", "scope": "bad"},
        follow_redirects=False,
    ).status_code == 303
    assert web.post("/tags/undo", follow_redirects=False).status_code == 303
    assert web.post(
        f"/notes/paper/{paper_id}", data={"notes": "plain"}, follow_redirects=False
    ).status_code == 303
    assert web.post(
        f"/notes/unknown/{paper_id}", data={"notes": "plain"}, follow_redirects=False
    ).status_code == 303
    assert web.post(
        "/notes/paper/999999", data={"notes": "plain"}, headers=async_headers
    ).status_code == 404

    # Tag endpoints: validation, missing records, and non-JavaScript redirects.
    for path in (
        f"/papers/{paper_id}/tags/add",
        f"/blogs/{entry_id}/tags/add",
        f"/authors/{author_candidate.openalex_id}/tags",
        f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/tags/add",
    ):
        assert web.post(path, data={"tag": ""}, headers=async_headers).status_code == 400
    assert web.post(
        "/papers/999999/tags/add", data={"tag": "x"}, headers=async_headers
    ).status_code == 404
    assert web.post(
        "/blogs/999999/tags/add", data={"tag": "x"}, headers=async_headers
    ).status_code == 404
    assert web.post(
        "/authors/A999/tags", data={"tag": "x"}, headers=async_headers
    ).status_code == 404
    assert web.post(
        f"/authors/{author_candidate.openalex_id}/blogs/999999/tags/add",
        data={"tag": "x"}, headers=async_headers,
    ).status_code == 404
    for path in (
        f"/papers/{paper_id}/tags/add",
        f"/blogs/{entry_id}/tags/add",
        f"/authors/{author_candidate.openalex_id}/tags",
        f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/tags/add",
    ):
        assert web.post(path, data={"tag": "Redirect"}, follow_redirects=False).status_code == 303
    for path in (
        f"/papers/{paper_id}/tags/remove",
        f"/blogs/{entry_id}/tags/remove",
        f"/authors/{author_candidate.openalex_id}/tags/remove",
        f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/tags/remove",
    ):
        assert web.post(path, data={"tag": "Redirect"}, follow_redirects=False).status_code == 303

    assert web.post(
        f"/content/blog/{entry_id}/classifications/{query_id}",
        data={"value": "No"},
        follow_redirects=False,
    ).status_code == 303
    assert web.post(
        f"/content/blog/999999/classifications/{query_id}",
        data={"value": "No"}, headers=async_headers,
    ).status_code == 404
    assert web.post(
        f"/authors/{author_candidate.openalex_id}/star",
        data={"starred": "1"}, headers=async_headers,
    ).status_code == 200
    assert web.post(
        "/authors/A999/star", data={"starred": "1"}, headers=async_headers
    ).status_code == 404

    # Import validation branches and complete metadata columns.
    assert web.post("/authors/import", follow_redirects=False).status_code == 303
    assert web.post("/authors/blogs/import", follow_redirects=False).status_code == 303
    malformed_author = web.post(
        "/authors/import",
        files={"csv_file": ("authors.csv", "Institution\nOnly Lab\n", "text/csv")},
        follow_redirects=False,
    )
    assert malformed_author.status_code == 303 and "level=error" in malformed_author.headers["location"]
    complete_author = web.post(
        "/authors/import",
        files={"csv_file": (
            "authors.csv",
            "Author Name,Institution,OpenAlex ID,Starred,Enabled,Tags,Notes\n"
            "CSV Person,CSV Lab,A777,yes,no,CSV Tag,CSV Note\n",
            "text/csv",
        )},
        data={"import_tags": "All imports"},
        follow_redirects=False,
    )
    assert complete_author.status_code == 303
    malformed_blog = web.post(
        "/authors/blogs/import",
        files={"csv_file": ("blogs.csv", "Name,URL\nX,https://x.test\n", "text/csv")},
        follow_redirects=False,
    )
    assert malformed_blog.status_code == 303 and "level=error" in malformed_blog.headers["location"]


def test_more_tag_undo_and_pdf_missing_branches(tmp_path, config_path, author_candidate, paper_record):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.set_content_selected("paper", paper_id, True)
        db.set_content_selected("blog", entry_id, True)
        changed, _ = db.bulk_update_selected_tags("add", "Undo added")
        assert changed == 2
        assert db.undo_last_tag_action() == 2
        assert db.undo_last_tag_action() == 0
        db.add_author_tag(author_candidate.openalex_id, "Everywhere")
        db.add_blog_source_tag(blog_id, "Everywhere")
        db.add_paper_tag(paper_id, "Everywhere")
        db.add_blog_entry_tag(entry_id, "Everywhere")
        changed, _ = db.remove_tag_globally("Everywhere", "all")
        assert changed == 4
        assert db.undo_last_tag_action() == 4
        # Invalid stored history must be reported rather than silently discarded.
        db._conn.execute(
            "INSERT INTO tag_action_history(action, payload_json, created_at) VALUES ('bad', '{', 'now')"
        )
        db._conn.commit()
        try:
            db.undo_last_tag_action()
        except Exception as exc:
            assert "invalid" in str(exc).lower()
        else:
            raise AssertionError("invalid tag history should fail")
        try:
            db._add_content_tag("unknown", "id", 1, "x")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown tag table should fail")
        missing = tmp_path / "missing.pdf"
        db.set_local_pdf(paper_id, str(missing), "abc")

    page = web_client(config_path).get(f"/papers/{paper_id}")
    assert "Missing on disk" in page.text
    assert "Restore PDF" in page.text
    assert web_client(config_path).post(
        "/papers/999999/pdf/restore", follow_redirects=False
    ).status_code == 404


def test_private_filter_and_regexp_helpers_cover_edge_inputs(config_path):
    from starlette.datastructures import QueryParams
    from starlette.requests import Request
    from paperwatch.web import (
        _classification_form_values,
        _content_tag_values,
        _query_profile_names,
        _query_view,
        _request_return_path,
        _split_tags,
    )

    assert _content_tag_values(QueryParams("tag=One&tag=one&tag=Two")) == ("One", "Two")
    assert _content_tag_values({"tag": " Single  tag "}) == ("Single tag",)
    assert len(_split_tags(",".join(f"tag-{index}" for index in range(110)))) == 100
    assert _query_profile_names("not-json") == ("not-json",)
    assert _query_profile_names('{"profile":"bad"}') == ()
    config = load_config(config_path)
    query = _query_view(
        {
            "model_profiles_json": "[]",
            "label_values_json": "not-json",
            "regexp_pattern": "x",
            "regexp_mode": "print",
        },
        config,
    )
    assert query["local_regexp"] is True and query["label_values"] == []
    assert query["display_query_kind"] == "regexp"
    assert _classification_form_values("# comment\nHigh = high\n\nLow = low") == [
        {"value": "High", "pattern": "high"},
        {"value": "Low", "pattern": "low"},
    ]
    try:
        _classification_form_values("invalid")
    except ValueError as exc:
        assert "Label = regexp" in str(exc)
    else:
        raise AssertionError("invalid classifier syntax should fail")
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"referer", b"http://testserver/path?q=1")]})
    assert _request_return_path(request) == "/path?q=1"
    external = Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"referer", b"https://example.com")]})
    assert _request_return_path(external, "/fallback") == "/fallback"


def test_data_cleanup_routes_and_pdf_no_source(
    tmp_path, config_path, author_candidate, paper_record
):
    paper_id, _blog_id, _entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    stored_pdf = config.paths.pdf_dir / "stored.pdf"
    stored_pdf.parent.mkdir(parents=True, exist_ok=True)
    stored_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with Database(config.paths.database) as db:
        db.set_local_pdf(paper_id, str(stored_pdf), hashlib.sha256(stored_pdf.read_bytes()).hexdigest())
        query_id = db.create_saved_query(name="Temporary", prompt="Prompt")
        run_id = db.create_query_runs(
            batch_id="history",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=1024,
        )[0]
        db.start_query_run(run_id)
        db.finish_query_run(run_id, matched_count=0, input_chars=0, output={"items": []})
        db._conn.execute("UPDATE papers SET pdf_url=NULL WHERE id=?", (paper_id,))
        db._conn.commit()

    web = web_client(config_path)
    no_source = web.post(f"/papers/{paper_id}/pdf/restore", follow_redirects=False)
    assert no_source.status_code == 303 and "No+remote+PDF" in no_source.headers["location"]
    for path in (
        "/data/delete/query-history",
        "/data/delete/queries",
        "/data/delete/summaries",
        "/data/delete/fetch-history",
        "/data/delete/blogs",
        "/data/delete/papers",
        "/data/delete/authors",
    ):
        response = web.post(path, follow_redirects=False)
        assert response.status_code == 303
    assert not stored_pdf.exists()


def test_099_filter_catalog_and_tag_history_edge_branches(
    tmp_path, config_path, author_candidate, paper_record
):
    from paperwatch.db import DatabaseError

    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_blog_source_tag(blog_id, "Official")
        assert len(db.author_blogs(author_candidate.openalex_id, tag=" Official ")) == 1
        assert db.author_blogs(author_candidate.openalex_id, tag="Missing") == []
        assert db.search_papers(title_query=paper_record.title[:12])[1] == 1
        assert db.search_papers(title_query="definitely absent")[1] == 0
        assert len(db.list_blog_entries(title_query="stored blog")) == 1
        assert db.list_blog_entries(title_query="definitely absent") == []
        try:
            db._tag_spec("unknown")
        except ValueError as exc:
            assert "Unknown" in str(exc)
        else:
            raise AssertionError("unknown tag objects must fail")

        # Non-dict entries in otherwise valid history are skipped safely.
        db._conn.execute(
            "INSERT INTO tag_action_history(action, payload_json, created_at) "
            "VALUES ('skip', '[1, null]', 'now')"
        )
        db._conn.commit()
        assert db.undo_last_tag_action() == 0

        db._conn.execute(
            "INSERT INTO tag_action_history(action, payload_json, created_at) "
            "VALUES ('bad-json', '{', 'now')"
        )
        db._conn.commit()
        try:
            db.undo_last_tag_action()
        except DatabaseError as exc:
            assert "invalid" in str(exc).lower()
        else:
            raise AssertionError("malformed tag history must fail")

        payload = db.export_author_blog_catalog()
        payload["authors"][0]["blogs"][0]["enabled"] = 0

    with Database(tmp_path / "disabled-catalog.db") as target:
        assert target.import_author_blog_catalog(payload) == (1, 1)
        imported = target.author_blogs(author_candidate.openalex_id)[0]
        assert imported["enabled"] == 0


def test_099_non_ajax_routes_simple_filter_values_and_pdf_hash_errors(
    monkeypatch, tmp_path, config_path, author_candidate, paper_record
):
    from paperwatch.pdfs import PDFError, pdf_sha256
    from paperwatch.web import _paper_pdf_state, _parse_filter_values

    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    parsed = _parse_filter_values(
        {
            "topic": "https://openalex.org/T100",
            "author": author_candidate.openalex_id,
            "author_tag": " Systems ",
            "tag": " Review ",
            "title": "stored",
        }
    )
    assert parsed.topic_ids == ("T100",)
    assert parsed.author_ids == (author_candidate.openalex_id,)
    assert parsed.author_tags == ("Systems",)
    assert parsed.content_tags == ("Review",)
    assert parsed.title_query == "stored"

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"x")

    class DummyDB:
        def set_paper_pdf_hash(self, *_args):
            raise AssertionError("must not save a hash after a hash failure")

    monkeypatch.setattr("paperwatch.web.pdf_sha256", lambda _path: (_ for _ in ()).throw(PDFError("hash failed")))
    state = _paper_pdf_state(
        DummyDB(),
        {"id": paper_id, "local_pdf_path": str(broken), "pdf_sha256": "old"},
    )
    assert state["pdf_modified"] is True

    try:
        pdf_sha256(tmp_path / "missing.pdf")
    except PDFError as exc:
        assert "Could not hash PDF" in str(exc)
    else:
        raise AssertionError("hashing a missing PDF must fail")

    web = web_client(config_path)
    # Exercise non-JavaScript redirect branches as well as their validation paths.
    for path, data in (
        ("/content/selected/tags/add", {"tag": "Redirect tag"}),
        ("/tags/remove-global", {"tag": "Redirect tag", "scope": "content"}),
        (f"/papers/{paper_id}/tags/add", {"tag": "Paper redirect"}),
        (f"/papers/{paper_id}/tags/remove", {"tag": "Paper redirect"}),
        (f"/blogs/{entry_id}/tags/add", {"tag": "Blog redirect"}),
        (f"/blogs/{entry_id}/tags/remove", {"tag": "Blog redirect"}),
        (f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/tags/add", {"tag": "Source redirect"}),
        (f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/tags/remove", {"tag": "Source redirect"}),
    ):
        response = web.post(path, data=data, follow_redirects=False)
        assert response.status_code == 303

    assert web.post(
        f"/papers/{paper_id}/tags/add", data={"tag": ""}, follow_redirects=False
    ).status_code == 303
    assert web.post(
        f"/papers/{paper_id}/tags/remove", data={"tag": ""}, follow_redirects=False
    ).status_code == 303
    assert web.post(
        f"/blogs/{entry_id}/tags/remove", data={"tag": ""}, follow_redirects=False
    ).status_code == 303


def test_099_pdf_import_metadata_replacement_and_rollback(
    monkeypatch, tmp_path, config_path, author_candidate, paper_record
):
    from paperwatch.pdfs import PDFError, safe_pdf_name

    paper_id, _blog_id, _entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)

    def pdf_bytes(width: int) -> bytes:
        output = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=width, height=100)
        writer.write(output)
        return output.getvalue()

    web = web_client(config_path)
    imported = web.post(
        "/papers/import",
        data={
            "title": "Imported metadata paper",
            "authors": "Import Author",
            "source_link": "https://example.test/imported-paper",
            "tags": "Imported, Review",
            "notes": "## Imported note",
        },
        files={"pdf": ("imported.pdf", pdf_bytes(100), "application/pdf")},
        follow_redirects=False,
    )
    assert imported.status_code == 303
    with Database(config.paths.database) as db:
        rows, total = db.search_papers(title_query="Imported metadata paper")
        assert total == 1
        imported_row = rows[0]
        assert imported_row["notes"] == "## Imported note"
        assert "Imported" in str(imported_row["content_tags"])

        paper = db.get_paper(paper_id)
        assert paper is not None
        destination = config.paths.pdf_dir / safe_pdf_name(str(paper["identity_key"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(pdf_bytes(120))
        original = destination.read_bytes()
        db.set_local_pdf(paper_id, str(destination), hashlib.sha256(original).hexdigest())

    replaced = web.post(
        f"/papers/{paper_id}/pdf/upload",
        files={"pdf": ("replacement.pdf", pdf_bytes(160), "application/pdf")},
        follow_redirects=False,
    )
    assert replaced.status_code == 303
    assert destination.read_bytes() != original

    before_failure = destination.read_bytes()
    monkeypatch.setattr(
        "paperwatch.web.pdf_sha256",
        lambda _path: (_ for _ in ()).throw(PDFError("forced hash failure")),
    )
    failed = web.post(
        f"/papers/{paper_id}/pdf/upload",
        files={"pdf": ("replacement.pdf", pdf_bytes(180), "application/pdf")},
        follow_redirects=False,
    )
    assert failed.status_code == 303 and "level=error" in failed.headers["location"]
    assert destination.read_bytes() == before_failure
    assert web.post(
        "/papers/999999/pdf/upload",
        files={"pdf": ("replacement.pdf", pdf_bytes(180), "application/pdf")},
        follow_redirects=False,
    ).status_code == 404


def _roundtrip_config(config_path: Path, target_root: Path, name: str) -> Path:
    config = load_config(config_path)
    target = target_root / f"{name}.toml"
    text = config_path.read_text()
    text = text.replace(str(config.paths.database), str(target_root / f"{name}.db"))
    text = text.replace(str(config.paths.pdf_dir), str(target_root / f"{name}-pdfs"))
    target.write_text(text)
    return target


def _csv_rows(text: str, *keys: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    return sorted(rows, key=lambda row: tuple(row.get(key, "") for key in keys))


def _portable_catalog(payload: dict) -> dict:
    copied = json.loads(json.dumps(payload))
    for author in copied.get("authors", []):
        for blog in author.get("blogs", []):
            blog.pop("id", None)
    return copied


def _sqlite_snapshot(path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(str(path))
    try:
        tables = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        snapshot: dict[str, list[tuple]] = {}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            rows = [tuple(row) for row in conn.execute(f"SELECT * FROM {quoted}").fetchall()]
            snapshot[table] = sorted(rows, key=repr)
        return snapshot
    finally:
        conn.close()


def test_data_author_csv_roundtrip_is_lossless(tmp_path, config_path, author_candidate):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.import_author_record(
            openalex_id=author_candidate.openalex_id,
            name=author_candidate.name,
            orcid=author_candidate.orcid,
            institution=author_candidate.institution,
            works_count=author_candidate.works_count,
            cited_by_count=author_candidate.cited_by_count,
            enabled=False,
            starred=True,
            download_papers=False,
            notes="OpenAlex note, with commas\nand a second line",
            added_at="2025-01-02T03:04:05+00:00",
            last_checked_at="2026-08-08T09:10:11+00:00",
            metadata={"source": "roundtrip", "nested": {"ok": True}},
        )
        db.add_author_tag(author_candidate.openalex_id, "Alpha")
        db.add_author_tag(author_candidate.openalex_id, "Beta")
        local_id = "LOCAL-ROUNDTRIP-ABC123"
        db.import_author_record(
            openalex_id=local_id,
            name="Same Name Local",
            institution="Independent Lab",
            enabled=True,
            starred=True,
            download_papers=True,
            notes="Local note",
            added_at="2024-02-03T04:05:06+00:00",
            last_checked_at="2026-07-01T00:00:00+00:00",
            metadata={"manual": True, "papers_enabled": True, "custom": "kept"},
        )
        db.add_author_tag(local_id, "Local")

    source_web = web_client(config_path)
    exported = source_web.get("/exports/authors.csv")
    assert exported.status_code == 200
    source_rows = _csv_rows(exported.text, "Paperwatch ID")
    assert {row["Paperwatch ID"] for row in source_rows} == {
        author_candidate.openalex_id,
        local_id,
    }
    assert next(row for row in source_rows if row["Paperwatch ID"] == local_id)["OpenAlex ID"] == ""

    target_config = _roundtrip_config(config_path, tmp_path, "authors-lossless")
    target_web = web_client(target_config)
    imported = target_web.post(
        "/authors/import",
        files={"csv_file": ("authors.csv", exported.content, "text/csv")},
        follow_redirects=False,
    )
    assert imported.status_code == 303
    reexported = target_web.get("/exports/authors.csv")
    assert _csv_rows(reexported.text, "Paperwatch ID") == source_rows


def test_data_blog_csv_roundtrip_is_lossless_and_uses_author_id(
    tmp_path, config_path, author_candidate
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        first_id = author_candidate.openalex_id
        second_id = "A124"
        db.import_author_record(openalex_id=first_id, name="Duplicate Name")
        db.import_author_record(openalex_id=second_id, name="Duplicate Name")
        first_blog = db.import_author_blog_record(
            first_id,
            "https://example.test/first/",
            title="Fetched first title",
            label="",
            relationship="Author",
            source_mode="article_index",
            content_selector="main article",
            article_url_pattern=r"/posts/\\d+",
            enabled=False,
            dead=False,
            notes="First source note",
            item_stream_mode="article_index",
        )
        db.add_blog_source_tag(first_blog, "Official")
        second_blog = db.import_author_blog_record(
            second_id,
            "https://example.test/second/",
            title="Fetched second title",
            label="Second label",
            relationship="Lab",
            source_mode="feed",
            enabled=False,
            dead=True,
            notes="Second source note",
            item_stream_mode="feed",
        )
        db.add_blog_source_tag(second_blog, "Dead source")

    source_web = web_client(config_path)
    exported = source_web.get("/exports/blogs.csv")
    assert exported.status_code == 200
    source_rows = _csv_rows(exported.text, "Author ID", "Primary source URL")
    assert {row["Author ID"] for row in source_rows} == {first_id, second_id}
    first = next(row for row in source_rows if row["Author ID"] == first_id)
    assert first["Primary applied-ML source"] == ""
    assert first["Primary source title"] == "Fetched first title"
    assert first["Primary source type"] == "article_index"
    assert first["Enabled"] == "no"

    target_config = _roundtrip_config(config_path, tmp_path, "blogs-lossless")
    target = load_config(target_config)
    with Database(target.paths.database) as db:
        # Duplicate names deliberately make name-only matching ambiguous.
        db.import_author_record(openalex_id=first_id, name="Duplicate Name")
        db.import_author_record(openalex_id=second_id, name="Duplicate Name")
    target_web = web_client(target_config)
    imported = target_web.post(
        "/authors/blogs/import",
        files={"csv_file": ("blogs.csv", exported.content, "text/csv")},
        follow_redirects=False,
    )
    assert imported.status_code == 303
    reexported = target_web.get("/exports/blogs.csv")
    assert _csv_rows(reexported.text, "Author ID", "Primary source URL") == source_rows


def test_data_author_blog_json_roundtrip_is_lossless(tmp_path, config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        local_id = "LOCAL-JSON-ROUNDTRIP"
        db.import_author_record(
            openalex_id=local_id,
            name="Portable Local",
            institution="Portable Lab",
            works_count=7,
            cited_by_count=11,
            enabled=True,
            starred=True,
            download_papers=True,
            notes="Portable author note",
            added_at="2023-04-05T06:07:08+00:00",
            metadata={"manual": True, "custom": [1, 2, 3]},
        )
        db.add_author_tag(local_id, "Portable")
        blog_id = db.import_author_blog_record(
            local_id,
            "https://portable.example/blog/",
            title="Fetched title must remain a title",
            label="",
            relationship="Direct",
            source_mode="article_index",
            content_selector="article.post",
            article_url_pattern=r"/entry/",
            enabled=False,
            dead=False,
            notes="Portable source note",
            item_stream_mode="article_index",
        )
        db.add_blog_source_tag(blog_id, "First party")

    source_web = web_client(config_path)
    exported = source_web.get("/data/export/authors-blogs")
    assert exported.status_code == 200
    source = exported.json()
    target_config = _roundtrip_config(config_path, tmp_path, "catalog-lossless")
    target_web = web_client(target_config)
    imported = target_web.post(
        "/data/import/authors-blogs",
        files={"file": ("authors-blogs.json", exported.content, "application/json")},
        follow_redirects=False,
    )
    assert imported.status_code == 303
    restored = target_web.get("/data/export/authors-blogs").json()
    assert _portable_catalog(restored) == _portable_catalog(source)


def test_data_full_zip_roundtrip_preserves_database_and_pdf_store(
    tmp_path, config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    config.paths.pdf_dir.mkdir(parents=True, exist_ok=True)
    stored_pdf = config.paths.pdf_dir / "nested" / "stored.pdf"
    stored_pdf.parent.mkdir(parents=True, exist_ok=True)
    stored_pdf.write_bytes(b"%PDF-1.4\nlossless-pdf-bytes\n%%EOF")
    orphan = config.paths.pdf_dir / "orphan.bin"
    orphan.write_bytes(b"orphan bytes are part of the store")
    with Database(config.paths.database) as db:
        db._conn.execute(
            "UPDATE papers SET local_pdf_path=?, pdf_sha256=?, pdf_size=? WHERE id=?",
            (
                str(stored_pdf.resolve()),
                hashlib.sha256(stored_pdf.read_bytes()).hexdigest(),
                stored_pdf.stat().st_size,
                paper_id,
            ),
        )
        db._conn.commit()
        db.add_author_tag(author_candidate.openalex_id, "Whole backup")
        db.add_blog_source_tag(blog_id, "Whole backup")
        db.add_blog_entry_tag(entry_id, "Whole backup")
        db.set_download_pdfs_enabled(False)
        query_id = db.create_saved_query(name="Backup query", prompt="Keep me")
        run_id = db.create_query_runs(
            batch_id="backup-batch",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=12,
            output={"title": "Backup result", "summary": "Kept", "items": []},
        )

    web = web_client(config_path)
    exported = web.get("/data/export/all")
    assert exported.status_code == 200
    archive_bytes = exported.content
    archive_db = tmp_path / "archive.sqlite3"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive_db.write_bytes(archive.read("database.sqlite3"))
        archived_files = {
            name.removeprefix("pdfs/"): archive.read(name)
            for name in archive.namelist()
            if name.startswith("pdfs/") and not name.endswith("/")
        }
    expected_db = _sqlite_snapshot(archive_db)
    assert archived_files == {
        "nested/stored.pdf": stored_pdf.read_bytes(),
        "orphan.bin": orphan.read_bytes(),
    }

    with Database(config.paths.database) as db:
        db.delete_all_blog_entries()
        db.delete_all_summaries()
        db._conn.execute("DELETE FROM saved_queries WHERE id=?", (query_id,))
        db._conn.execute("UPDATE authors SET notes='mutated after export'")
        db._conn.commit()
    stored_pdf.write_bytes(b"mutated")
    orphan.unlink()
    (config.paths.pdf_dir / "extra.bin").write_bytes(b"extra")

    imported = web.post(
        "/data/import/all",
        files={"file": ("paperwatch-full-data.zip", archive_bytes, "application/zip")},
        follow_redirects=False,
    )
    assert imported.status_code == 303
    assert _sqlite_snapshot(config.paths.database) == expected_db
    restored_files = {
        file.relative_to(config.paths.pdf_dir).as_posix(): file.read_bytes()
        for file in config.paths.pdf_dir.rglob("*")
        if file.is_file()
    }
    assert restored_files == archived_files
