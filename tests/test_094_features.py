from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.blogs import _extract_blog_document_full
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.query_jobs import QueryBatchSnapshot, preprocess_regexp_documents
from paperwatch.web import (
    RunSnapshot,
    _render_stored_markdown,
    _safe_markdown_url,
    create_app,
)


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


def _client(config_path: Path, *, running: bool = False) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(running=running),
            analysis_controller=AnalysisStub(),
            query_controller=QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def _seed_content(config_path, author_candidate, paper_record):
    config = load_config(config_path)
    pdf = config.paths.pdf_dir / "stored.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_tag(author_candidate.openalex_id, "systems")
        paper_id, _ = db.upsert_paper(paper_record)
        db.set_local_pdf(paper_id, str(pdf))
        db.add_paper_tag(paper_id, "shared")
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog/",
            label="Example writing",
            relationship="Direct author",
            source_mode="article_index",
            content_selector="main",
            article_url_pattern=r"/blog/",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="A fetched website article",
            content_sha256="website-article",
            normalized_text=(
                "Title: A fetched website article\n"
                "URL: https://example.test/blog/article\n"
                "Body with ![diagram](https://cdn.example.test/diagram.gif) and "
                "[reference](https://example.test/reference)."
            ),
            rendered_html=(
                '<p>Body with <a href="https://cdn.example.test/diagram.gif" '
                'target="_blank" rel="noreferrer noopener"><img '
                'src="https://cdn.example.test/diagram.gif" alt="diagram" '
                'loading="lazy" decoding="async" referrerpolicy="no-referrer"></a></p>'
            ),
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="snapshot",
            external_id="article-1",
            source_url="https://example.test/blog/article",
            published_at="2026-08-01T12:00:00+00:00",
        )
        db.add_blog_entry_tag(entry_id, "shared")
    return paper_id, blog_id, entry_id, pdf


def test_remote_images_are_linked_rendered_and_never_embedded():
    source = """
    <html><head><title>Illustrated</title></head><body><main>
      <p>Before <img data-lazy-src="/assets/chart.gif?utm_source=x" alt="Animated chart"> after.</p>
      <figure><img src="https://cdn.example.test/photo.png#section" title="Photo"><figcaption>Caption</figcaption></figure>
      <img data-original="/standalone.webp" alt="Standalone">
      <img src="data:image/png;base64,AAAA" alt="Embedded">
      <img src="javascript:alert(1)" alt="Unsafe">
    </main></body></html>
    """
    title, text, _links, rendered = _extract_blog_document_full(
        source, "Fallback", "https://example.test/blog/post"
    )
    assert title == "Illustrated"
    assert "![Animated chart](https://example.test/assets/chart.gif)" in text
    assert "![Photo](https://cdn.example.test/photo.png)" in text
    assert "![Standalone](https://example.test/standalone.webp)" in text
    assert "data:image" not in text and "javascript:" not in text
    assert 'src="https://example.test/assets/chart.gif"' in rendered
    assert 'loading="lazy"' in rendered
    assert 'referrerpolicy="no-referrer"' in rendered
    assert 'title="Open image"' in rendered
    assert "base64" not in rendered

    markdown = (
        "Paragraph with ![Chart](https://cdn.example.test/chart.gif) and "
        "[source](https://example.test/source).\n\n"
        "![bad](javascript:alert(1))"
    )
    display = _render_stored_markdown(markdown)
    assert '<img src="https://cdn.example.test/chart.gif"' in display
    assert "Open image" in display
    assert '<a href="https://example.test/source"' in display
    assert "javascript:" not in display
    assert "data:image" not in display


def test_blank_regexp_is_exact_noop_and_dropdown_script_supports_typing_enter(monkeypatch):
    documents = [
        {"title": "one", "content": "alpha beta"},
        {"title": "two", "content": "gamma delta"},
    ]
    processed, matches = preprocess_regexp_documents(documents, "", 50)
    assert processed is documents
    assert matches == 0

    script = Path("src/paperwatch/web_static/app.js").read_text(encoding="utf-8")
    assert 'document.querySelectorAll("select")' in script
    assert 'input.type = "search"' in script
    assert 'event.key === "Enter"' in script
    assert "normalized(option?.textContent).includes(term)" in script
    assert "normalized(option?.value).includes(term)" in script
    assert 'select.dispatchEvent(new Event("change", { bubbles: true }))' in script
    assert "search-select-multiple" in script

    # The renderer also fails closed if URL validation rejects a nominal HTTP match.
    monkeypatch.setattr("paperwatch.web._safe_markdown_url", lambda value: None)
    rejected = _render_stored_markdown(
        "![image](https://example.test/image.png) [link](https://example.test/)"
    )
    assert "<img" not in rejected and "<a " not in rejected


def test_unified_download_page_blog_tags_detail_toolbar_and_remote_image_csp(
    config_path, author_candidate, paper_record, tmp_path, monkeypatch
):
    paper_id, _blog_id, entry_id, _pdf = _seed_content(
        config_path, author_candidate, paper_record
    )
    client = _client(config_path)

    response = client.get("/")
    assert response.status_code == 200
    assert paper_record.title in response.text
    assert "A fetched website article" in response.text
    assert "content-row-paper" in response.text
    assert "content-row-blog" in response.text
    assert "Papers and blogs" in response.text
    assert ">Tags<" in response.text
    assert "shared" in response.text
    assert "img-src &#x27;self&#x27;" not in response.text
    assert "img-src 'self' data: https: http:" in response.headers["content-security-policy"]

    papers = client.get("/papers")
    assert paper_record.title in papers.text
    assert "A fetched website article" not in papers.text
    blogs = client.get("/blogs")
    assert "A fetched website article" in blogs.text
    assert paper_record.title not in blogs.text
    tagged = client.get("/?tag=shared")
    assert paper_record.title in tagged.text and "A fetched website article" in tagged.text

    detail = client.get(f"/blogs/{entry_id}")
    assert detail.status_code == 200
    source = detail.text.index("View source")
    copy = detail.text.index("Copy text")
    delete = detail.text.index("Delete post")
    assert source < copy < delete
    assert 'class="content-detail-toolbar blog-reference-panel"' in detail.text
    assert 'class="content-detail-toolbar-facts" aria-label="Blog details"' in detail.text
    assert f'action="/content/blog/{entry_id}/export"' in detail.text
    assert 'content-detail-toolbar-source-links' in detail.text
    assert 'class="content-detail-author-sources"' not in detail.text
    assert "Source ↗" in detail.text
    assert 'src="https://cdn.example.test/diagram.gif"' in detail.text
    assert 'referrerpolicy="no-referrer"' in detail.text
    assert "shared" in detail.text

    monkeypatch.setattr("paperwatch.web._choose_export_directory", lambda: tmp_path)
    blog_export = client.post(
        f"/content/blog/{entry_id}/export",
        headers={"referer": f"http://testserver/blogs/{entry_id}"},
        follow_redirects=False,
    )
    assert blog_export.status_code == 303
    blog_file = next(tmp_path.glob(f"*-{entry_id}.json"))
    blog_payload = json.loads(blog_file.read_text())
    assert blog_payload["title"] == "A fetched website article"
    assert blog_payload["authors"]
    assert any(tag["name"] == "shared" for tag in blog_payload["tags"])

    paper_export = client.post(
        f"/content/paper/{paper_id}/export",
        headers={"referer": f"http://testserver/papers/{paper_id}"},
        follow_redirects=False,
    )
    assert paper_export.status_code == 303
    paper_file = next(
        path for path in tmp_path.glob(f"*-{paper_id}.json") if path != blog_file
    )
    assert json.loads(paper_file.read_text())["title"] == paper_record.title
    assert client.post("/content/blog/999999/export").status_code == 404
    assert client.post("/content/paper/999999/export").status_code == 404
    assert client.post("/content/unknown/1/export").status_code == 400

    paper_detail = client.get(f"/papers/{paper_id}")
    assert paper_detail.status_code == 200 and "shared" in paper_detail.text


def test_catalog_and_full_data_management_round_trip(
    config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id, pdf = _seed_content(
        config_path, author_candidate, paper_record
    )
    config = load_config(config_path)
    client = _client(config_path)

    runs = client.get("/data")
    assert runs.status_code == 200
    for label in (
        "Delete all authors",
        "Delete all downloaded papers",
        "Delete all downloaded blogs",
        "Export whole data",
        "Import whole data",
        "Export authors + blogs",
        "Import authors + blogs",
    ):
        assert label in runs.text

    catalog_response = client.get("/data/export/authors-blogs")
    assert catalog_response.status_code == 200
    catalog = catalog_response.json()
    assert catalog["format"] == "paperwatch-authors-blogs"
    assert catalog["authors"][0]["tags"] == ["systems"]
    assert catalog["authors"][0]["blogs"][0]["source_mode"] == "article_index"

    full_response = client.get("/data/export/all")
    assert full_response.status_code == 200
    archive_bytes = full_response.content
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert {"database.sqlite3", "manifest.json", "pdfs/stored.pdf"} <= set(
            archive.namelist()
        )
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "paperwatch-full-data"
        copied = sqlite3.connect(":memory:")
        copied.close()

    delete_blogs = client.post("/data/delete/blogs", follow_redirects=False)
    assert delete_blogs.status_code == 303
    assert delete_blogs.headers["location"].startswith("/data?")
    with Database(config.paths.database) as db:
        assert db.get_blog_entry(entry_id) is None
        assert any(int(row["id"]) == blog_id for row in db.list_enabled_blogs())
        assert db.known_blog_article_urls(blog_id) == ()

    delete_papers = client.post("/data/delete/papers", follow_redirects=False)
    assert delete_papers.status_code == 303
    assert not pdf.exists()
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id) is None

    delete_authors = client.post("/data/delete/authors", follow_redirects=False)
    assert delete_authors.status_code == 303
    with Database(config.paths.database) as db:
        assert db.list_authors(tracked_only=True) == []

    import_catalog = client.post(
        "/data/import/authors-blogs",
        files={
            "file": (
                "paperwatch-authors-blogs.json",
                json.dumps(catalog).encode("utf-8"),
                "application/json",
            )
        },
        follow_redirects=False,
    )
    assert import_catalog.status_code == 303
    assert import_catalog.headers["location"].startswith("/data?")
    with Database(config.paths.database) as db:
        authors = db.list_authors(tracked_only=True)
        assert len(authors) == 1
        links = db.author_blogs(author_candidate.openalex_id)
        assert len(links) == 1 and links[0]["source_mode"] == "article_index"

    import_full = client.post(
        "/data/import/all",
        files={"file": ("paperwatch-full-data.zip", archive_bytes, "application/zip")},
        follow_redirects=False,
    )
    assert import_full.status_code == 303
    assert import_full.headers["location"].startswith("/data?")
    assert "Imported+the+full+Paperwatch" in import_full.headers["location"]
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id) is not None
        assert db.get_blog_entry(entry_id) is not None
        assert len(db.list_authors(tracked_only=True)) == 1
        restored_pdf = Path(db.get_paper(paper_id)["local_pdf_path"])
        assert restored_pdf.is_file()


def test_data_management_rejects_missing_files_and_active_jobs(config_path):
    client = _client(config_path)
    missing_catalog = client.post("/data/import/authors-blogs", data={}, follow_redirects=False)
    assert missing_catalog.status_code == 303
    assert "Choose+an+authors+and+blogs+JSON" in missing_catalog.headers["location"]
    invalid_catalog = client.post(
        "/data/import/authors-blogs",
        files={"file": ("bad.json", b"[]", "application/json")},
        follow_redirects=False,
    )
    assert invalid_catalog.status_code == 303
    assert "must+be+a+JSON+object" in invalid_catalog.headers["location"]
    missing_full = client.post("/data/import/all", data={}, follow_redirects=False)
    assert missing_full.status_code == 303
    assert "Choose+a+Paperwatch+full-data+ZIP" in missing_full.headers["location"]
    invalid_full = client.post(
        "/data/import/all",
        files={"file": ("bad.zip", b"not a zip", "application/zip")},
        follow_redirects=False,
    )
    assert invalid_full.status_code == 303
    assert "not+a+ZIP+archive" in invalid_full.headers["location"]

    busy = _client(config_path, running=True)
    for endpoint in (
        "/data/delete/authors",
        "/data/delete/papers",
        "/data/delete/blogs",
        "/data/import/authors-blogs",
        "/data/import/all",
    ):
        response = busy.post(endpoint, data={}, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"].startswith("/data?")
        assert "Stop+active+jobs" in response.headers["location"]


def _zip_bytes(files: dict[str, bytes | str]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return stream.getvalue()


def test_catalog_validation_local_authors_and_skipped_records(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        for payload, message in (
            ({}, "Unsupported"),
            ({"format": "paperwatch-authors-blogs", "version": 1, "authors": {}}, "invalid"),
            ({"format": "paperwatch-authors-blogs", "version": 1, "authors": ["bad"]}, "Author record"),
            (
                {
                    "format": "paperwatch-authors-blogs",
                    "version": 1,
                    "authors": [{"name": "", "openalex_id": ""}],
                },
                "requires a name",
            ),
        ):
            try:
                db.import_author_blog_catalog(payload)
            except ValueError as exc:
                assert message in str(exc)
            else:
                raise AssertionError("invalid catalog unexpectedly imported")

        payload = {
            "format": "paperwatch-authors-blogs",
            "version": 1,
            "authors": [
                {
                    "name": "Local Writer",
                    "openalex_id": "LOCAL-exported-value",
                    "institution": "Independent",
                    "enabled": 0,
                    "tags": ["local"],
                    "blogs": [
                        "ignored",
                        {
                            "url": "https://local.example/blog/",
                            "label": "Local blog",
                            "relationship": "Direct",
                            "source_mode": "docs",
                            "content_selector": "main",
                            "article_url_pattern": "/notes/",
                        },
                    ],
                },
                {
                    "name": "No Blog Author",
                    "openalex_id": "A-NO-BLOG",
                    "metadata": "not-a-dict",
                    "blogs": "not-a-list",
                },
            ],
        }
        assert db.import_author_blog_catalog(payload) == (2, 1)
        local = db.find_authors_by_name("Local Writer")[0]
        assert str(local["openalex_id"]).startswith("LOCAL-")
        assert local["enabled"] == 0
        assert db.author_blogs(str(local["openalex_id"]))[0]["source_mode"] == "docs"

        # Invalid stored metadata is exported as an empty object rather than breaking backups.
        db._conn.execute(
            "UPDATE authors SET metadata_json='not-json' WHERE openalex_id='A-NO-BLOG'"
        )
        db._conn.commit()
        exported = db.export_author_blog_catalog()
        no_blog = next(item for item in exported["authors"] if item["name"] == "No Blog Author")
        assert no_blog["metadata"] == {}


def test_full_import_validation_variants(config_path, tmp_path):
    client = _client(config_path)

    invalid_archives = [
        (
            _zip_bytes({"database.sqlite3": b"not sqlite"}),
            "missing+its+database+or+manifest",
        ),
        (
            _zip_bytes(
                {
                    "database.sqlite3": b"not sqlite",
                    "manifest.json": json.dumps({"format": "other"}),
                }
            ),
            "Unsupported+full-data+archive",
        ),
        (
            _zip_bytes(
                {
                    "database.sqlite3": b"not sqlite",
                    "manifest.json": json.dumps({"format": "paperwatch-full-data"}),
                    "pdfs/../escape.pdf": b"bad",
                }
            ),
            "unsafe+path",
        ),
    ]
    for archive, message in invalid_archives:
        response = client.post(
            "/data/import/all",
            files={"file": ("invalid.zip", archive, "application/zip")},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert message in response.headers["location"]

    empty_db = tmp_path / "empty.sqlite3"
    sqlite3.connect(empty_db).close()
    invalid_db_archive = _zip_bytes(
        {
            "database.sqlite3": empty_db.read_bytes(),
            "manifest.json": json.dumps({"format": "paperwatch-full-data"}),
        }
    )
    response = client.post(
        "/data/import/all",
        files={"file": ("invalid-db.zip", invalid_db_archive, "application/zip")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "database+is+invalid" in response.headers["location"]


def test_unified_sort_modes_invalid_content_and_html_block_variants(
    config_path, author_candidate, paper_record
):
    _paper_id, _blog_id, entry_id, _pdf = _seed_content(
        config_path, author_candidate, paper_record
    )
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        for sort in ("newest", "oldest", "discovered", "title", "invalid"):
            rows = db.list_blog_entries(
                author_id=author_candidate.openalex_id,
                author_tags=["systems", "systems", ""],
                date_from="2026-08-01",
                date_to="2026-08-01",
                sort=sort,
            )
            assert [int(row["id"]) for row in rows] == [entry_id]
    client = _client(config_path)
    for query in (
        "?sort=title",
        "?sort=oldest",
        "?sort=discovered",
        "?content=invalid",
        "?content=blogs&sort=title",
    ):
        response = client.get("/" + query)
        assert response.status_code == 200

    source = """
    <main><blockquote>Quoted<br>line</blockquote><ul><li>First</li><li><span></span></li></ul>
    <table><tr><td>Cell</td></tr></table><div class="unknown">Unwrapped</div></main>
    """
    _title, text, _links, rendered = _extract_blog_document_full(
        source, "Blocks", "https://example.test/"
    )
    assert "> Quoted" in text
    assert "- First" in text
    assert "Cell" in text
    assert "class=" not in rendered

    image_only = '<main><img src="/only.png"></main>'
    _title, text, _links, _rendered = _extract_blog_document_full(
        image_only, "Image", "https://example.test/"
    )
    assert text == "![Image](https://example.test/only.png)"
    assert _render_stored_markdown("\n\n") == ""
    image_display = _render_stored_markdown(
        "![Standalone](https://cdn.example.test/standalone.gif)"
    )
    assert image_display.startswith('<figure class="remote-blog-image">')
    assert _safe_markdown_url("mailto:test@example.test") is None
