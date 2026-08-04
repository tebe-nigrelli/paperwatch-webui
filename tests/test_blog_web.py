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

    def start(self, *args, **kwargs):
        return True


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryStub:
    def snapshot(self):
        return QueryBatchSnapshot()


class AuthorClient:
    def close(self):
        pass


def client_for(config_path):
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def test_manual_author_and_blog_management_ui(config_path):
    config = load_config(config_path)
    client = client_for(config_path)
    page = client.get("/data")
    assert page.status_code == 200
    assert "Add author manually" in page.text
    assert "Import blog links" in page.text
    assert "Manual blog import" in page.text
    assert "Create missing authors" in page.text
    manual_form = page.text.split('class="inline-blog-import-form"', 1)[1].split(
        "</form>", 1
    )[0]
    assert manual_form.count("<label") == 3
    csv_row = page.text.split('class="author-csv-import-row blog-csv-import-row"', 1)[1].split(
        "</div>", 1
    )[0]
    assert 'class="inline-blog-csv-form"' in csv_row
    assert "Create missing authors" in csv_row
    assert "Fetch new papers, download available PDFs" in page.text
    assert 'href="/" data-content-home-link>Content</a>' in page.text

    response = client.post(
        "/authors/manual",
        data={"name": "Manual Person", "institution": "Local Lab", "openalex_id": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303
    with Database(config.paths.database) as db:
        manual = db.find_authors_by_name("Manual Person")[0]
        manual_id = str(manual["openalex_id"])
        assert manual["enabled"] == 0

    response = client.post(
        "/authors/manual",
        data={"name": "Ada Lovelace", "institution": "Institute", "openalex_id": "A123"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    with Database(config.paths.database) as db:
        assert db.get_author("A123")["enabled"] == 1

    linked_from_authors = client.post(
        "/authors/blogs/manual",
        data={"openalex_id": "A123", "url": "https://ada.example/blog#updates"},
        follow_redirects=False,
    )
    assert linked_from_authors.status_code == 303
    with Database(config.paths.database) as db:
        assert db.author_blogs("A123")[0]["url"] == "https://ada.example/blog"
    missing_manual_author = client.post(
        "/authors/blogs/manual",
        data={"openalex_id": "", "url": "https://example.com/blog"},
        follow_redirects=False,
    )
    assert "level=error" in missing_manual_author.headers["location"]
    invalid_manual_url = client.post(
        "/authors/blogs/manual",
        data={"openalex_id": "A123", "url": "http://127.0.0.1/blog"},
        follow_redirects=False,
    )
    assert "level=error" in invalid_manual_url.headers["location"]

    invalid = client.post(
        "/authors/manual",
        data={"name": "Bad", "openalex_id": "not-an-id"},
        follow_redirects=False,
    )
    assert invalid.status_code == 303 and "level=error" in invalid.headers["location"]
    blank = client.post(
        "/authors/manual", data={"name": "", "openalex_id": ""}, follow_redirects=False
    )
    assert "level=error" in blank.headers["location"]

    added = client.post(
        f"/authors/{manual_id}/blogs",
        data={
            "url": "https://example.com/blog#feed",
            "label": "Manual blog",
            "relationship": "Direct author",
        },
        follow_redirects=False,
    )
    assert added.status_code == 303
    manage = client.get(f"/authors/{manual_id}/blogs")
    assert manage.status_code == 200
    assert "Manual blog" in manage.text
    assert "Direct author" in manage.text
    assert f'action="/authors/{manual_id}/blogs/' in manage.text
    assert 'title="Edit the URL and press Enter"' in manage.text
    assert "Mark broken" in manage.text
    authors = client.get("/authors")
    assert "1 · Manage" in authors.text
    assert "Blog-only" in authors.text
    assert 'id="toggle-author-ids"' not in authors.text
    assert 'id="tracked-authors-table"' in authors.text
    assert 'class="author-openalex-column"' in authors.text
    assert 'data-author-openalex-column hidden' not in authors.text
    assert 'title="Edit the ID and press Enter"' in authors.text
    assert '<button class="sr-only" type="submit">Save OpenAlex ID</button>' in authors.text

    invalid_blog = client.post(
        f"/authors/{manual_id}/blogs",
        data={"url": "http://127.0.0.1/blog"},
        follow_redirects=False,
    )
    assert "level=error" in invalid_blog.headers["location"]

    with Database(config.paths.database) as db:
        blog_id = int(db.author_blogs(manual_id)[0]["id"])

    broken = client.post(
        f"/authors/{manual_id}/blogs/{blog_id}/availability",
        data={"broken": "1"},
        follow_redirects=False,
    )
    assert broken.status_code == 303
    with Database(config.paths.database) as db:
        blog = db.author_blogs(manual_id)[0]
        assert blog["enabled"] == 0
        assert "Marked broken" in blog["last_error"]
        assert all(int(row["id"]) != blog_id for row in db.list_enabled_blogs())
    broken_page = client.get(f"/authors/{manual_id}/blogs")
    assert "blog-source-broken" in broken_page.text
    assert "Restore" in broken_page.text

    restored = client.post(
        f"/authors/{manual_id}/blogs/{blog_id}/availability",
        data={"broken": "0"},
        follow_redirects=False,
    )
    assert restored.status_code == 303
    edited = client.post(
        f"/authors/{manual_id}/blogs/{blog_id}/edit",
        data={"url": "https://example.com/new-blog#updates"},
        follow_redirects=False,
    )
    assert edited.status_code == 303
    with Database(config.paths.database) as db:
        blog = db.author_blogs(manual_id)[0]
        assert blog["url"] == "https://example.com/new-blog"
        assert blog["enabled"] == 1
        assert blog["last_error"] is None

    invalid_edit = client.post(
        f"/authors/{manual_id}/blogs/{blog_id}/edit",
        data={"url": "http://127.0.0.1/broken"},
        follow_redirects=False,
    )
    assert "level=error" in invalid_edit.headers["location"]
    missing_edit = client.post(
        f"/authors/{manual_id}/blogs/9999/edit",
        data={"url": "https://example.com/missing"},
        follow_redirects=False,
    )
    assert missing_edit.status_code == 404
    missing_availability = client.post(
        f"/authors/{manual_id}/blogs/9999/availability",
        data={"broken": "1"},
        follow_redirects=False,
    )
    assert missing_availability.status_code == 404

    removed = client.post(
        f"/authors/{manual_id}/blogs/{blog_id}/remove", follow_redirects=False
    )
    assert removed.status_code == 303
    assert "No blogs linked" in client.get(f"/authors/{manual_id}/blogs").text
    assert client.get("/authors/A404/blogs").status_code == 404


def test_blog_csv_import_and_feed_ui(config_path):
    config = load_config(config_path)
    client = client_for(config_path)
    csv_data = b"Researcher / author,Current lab / organization,Primary applied-ML source,Primary source URL,Relationship to author\nNew Person,Lab,New Blog,https://example.com/new,Direct author\n"
    response = client.post(
        "/authors/blogs/import",
        files={"csv_file": ("blogs.csv", csv_data, "text/csv")},
        data={"create_missing": "1"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "Imported+1+blog+links" in response.headers["location"]

    missing = client.post("/authors/blogs/import", data={}, follow_redirects=False)
    assert missing.status_code == 303 and "level=error" in missing.headers["location"]
    bad = client.post(
        "/authors/blogs/import",
        files={"csv_file": ("bad.csv", b"wrong,columns\nx,y\n", "text/csv")},
        follow_redirects=False,
    )
    assert bad.status_code == 303 and "level=error" in bad.headers["location"]

    with Database(config.paths.database) as db:
        author = db.find_authors_by_name("New Person")[0]
        blog = db.author_blogs(str(author["openalex_id"]))[0]
        baseline = db.save_blog_entry(
            blog_id=int(blog["id"]),
            title="New Blog",
            content_sha256="one",
            normalized_text="old",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
        )
        changed = db.save_blog_entry(
            blog_id=int(blog["id"]),
            title="New Blog",
            content_sha256="two",
            normalized_text="new",
            diff_text="--- previous\n+++ current\n-old\n+new",
            change_kind="changed",
            previous_entry_id=baseline,
        )

    feed = client.get("/blogs")
    assert feed.status_code == 200
    assert "Blog updates" in feed.text
    assert 'class="panel paper-table-panel unified-content-panel"' in feed.text
    assert "New Person" in feed.text
    assert str(changed) in feed.text
    detail = client.get(f"/blogs/{changed}")
    assert detail.status_code == 200
    assert "Version difference" in detail.text
    assert "+new" in detail.text
    assert client.get("/blogs/9999").status_code == 404


class RecordingQueryStub:
    def __init__(self, *, result="batch-blogs", error=None):
        self.result = result
        self.error = error
        self.calls = []

    def snapshot(self):
        return QueryBatchSnapshot()

    def start(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result


def _seed_query_blog(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        from paperwatch.models import AuthorCandidate

        db.add_author(AuthorCandidate(openalex_id="A-BLOG", name="Blog Author"))
        blog_id = db.add_author_blog(
            "A-BLOG", "https://example.test/blog", label="Author blog"
        )
        baseline = db.save_blog_entry(
            blog_id=blog_id,
            title="Baseline",
            content_sha256="base",
            normalized_text="old",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="New blog entry",
            content_sha256="new",
            normalized_text="complete new text",
            diff_text="-old\n+complete new text",
            change_kind="changed",
            previous_entry_id=baseline,
            source_url="https://example.test/blog/new",
        )
        query_id = db.create_saved_query(
            name="Watch implications",
            prompt="Summarize the practical implications.",
            auto_run_blogs=True,
        )
    return config, entry_id, query_id


def test_blog_saved_prompt_routes_and_collection_runner(config_path):
    config, entry_id, query_id = _seed_query_blog(config_path)
    with Database(config.paths.database) as db:
        db.set_content_selected("blog", entry_id, True)
    controller = RecordingQueryStub()
    client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=controller,
            author_client_factory=lambda config: AuthorClient(),
        )
    )

    feed = client.get("/blogs")
    assert "Run saved prompt" in feed.text
    assert "Watch implications" in feed.text
    detail = client.get(f"/blogs/{entry_id}")
    assert "Use the same reusable prompts" in detail.text

    response = client.post(
        f"/blogs/{entry_id}/prompt",
        data={
            "query_id": str(query_id),
            "model": "default",
            "context_window_tokens": "2048",
        },
        headers={"x-paperwatch-async": "1"},
    )
    assert response.status_code == 200
    assert response.json()["batch_id"] == "batch-blogs"
    args, kwargs = controller.calls[-1]
    assert args[:3] == (query_id, ("default",), 2048)
    assert kwargs["content_type"] == "blogs"
    assert kwargs["selection"]["blog_entry_ids"] == [entry_id]
    assert kwargs["source_scope"] == "blog"

    selected = client.post(
        "/queries/run-blogs",
        data={
            "query_id": str(query_id),
            "model": "default",
            "context_window_tokens": "4096",
            "content": "blogs",
            "q": "Stored",
        },
        follow_redirects=False,
    )
    assert selected.status_code == 303
    assert "Started+1+saved+query" in selected.headers["location"]
    args, kwargs = controller.calls[-1]
    assert args[:3] == (
        query_id,
        ("default",),
        config.llm.context_window_tokens,
    )
    selection = kwargs["selection"]
    assert selection["content_type"] == "blogs"
    assert selection["blog_entry_ids"] == [entry_id]
    assert "query" not in selection

    with Database(config.paths.database) as db:
        db.set_content_selected("blog", entry_id, False)
    no_selection = client.post(
        "/queries/run-blogs",
        data={"query_id": str(query_id), "model": "default"},
        follow_redirects=False,
    )
    assert no_selection.status_code == 303
    assert "Select+at+least+one+blog+entry" in no_selection.headers["location"]

    with Database(config.paths.database) as db:
        db.set_content_selected("blog", entry_id, True)

    missing_query = client.post("/queries/run-blogs", data={}, follow_redirects=False)
    assert "Select+a+saved+query" in missing_query.headers["location"]
    missing_entry = client.post(
        "/blogs/999999/prompt", data={"query_id": query_id}, follow_redirects=False
    )
    assert missing_entry.status_code == 404
    bad_query = client.post(
        f"/blogs/{entry_id}/prompt",
        data={"query_id": "999999", "model": "default"},
        headers={"x-paperwatch-async": "1"},
    )
    assert bad_query.status_code == 400
    assert "Select a saved prompt" in bad_query.json()["message"]

    sync_success = client.post(
        f"/blogs/{entry_id}/prompt",
        data={"query_id": query_id, "model": "default"},
        follow_redirects=False,
    )
    assert sync_success.status_code == 303
    assert "Started+Watch+implications" in sync_success.headers["location"]

    sync_bad_query = client.post(
        f"/blogs/{entry_id}/prompt",
        data={"query_id": "999999", "model": "default"},
        follow_redirects=False,
    )
    assert sync_bad_query.status_code == 303
    assert "Select+a+saved+prompt" in sync_bad_query.headers["location"]

    busy_controller = RecordingQueryStub(result=None)
    busy_client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=busy_controller,
            author_client_factory=lambda config: AuthorClient(),
        )
    )
    busy = busy_client.post(
        f"/blogs/{entry_id}/prompt",
        data={"query_id": query_id, "model": "default"},
        headers={"x-paperwatch-async": "1"},
    )
    assert busy.status_code == 409
    assert "already running" in busy.json()["message"]

    busy_collection = busy_client.post(
        "/queries/run-blogs",
        data={"query_id": query_id, "model": "default"},
        follow_redirects=False,
    )
    assert busy_collection.status_code == 303
    assert "already+running" in busy_collection.headers["location"]

    broken_controller = RecordingQueryStub(error=ValueError("bad blog selection"))
    broken_client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=broken_controller,
            author_client_factory=lambda config: AuthorClient(),
        )
    )
    broken = broken_client.post(
        "/queries/run-blogs",
        data={"query_id": query_id, "model": "default"},
        follow_redirects=False,
    )
    assert "bad+blog+selection" in broken.headers["location"]

    broken_prompt = broken_client.post(
        f"/blogs/{entry_id}/prompt",
        data={"query_id": query_id, "model": "default"},
        follow_redirects=False,
    )
    assert broken_prompt.status_code == 303
    assert "bad+blog+selection" in broken_prompt.headers["location"]

    with Database(config.paths.database) as db:
        runs = db.list_query_runs(query_id)
        assert runs == []


def test_saved_query_auto_targets_and_backward_routes(config_path):
    _config, entry_id, existing_query_id = _seed_query_blog(config_path)
    controller = RecordingQueryStub()
    client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=controller,
            author_client_factory=lambda config: AuthorClient(),
        )
    )

    created = client.post(
        "/queries",
        data={
            "name": "Automatic review",
            "prompt": "Review every new record.",
            "auto_run_papers": "on",
            "auto_run_blogs": "on",
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query = next(row for row in db.list_saved_queries() if row["name"] == "Automatic review")
        query_id = int(query["id"])
        assert query["auto_run_papers"] == 1
        assert query["auto_run_blogs"] == 1
        assert [row["name"] for row in db.list_automatic_queries("papers")] == [
            "Automatic review"
        ]
        assert {row["name"] for row in db.list_automatic_queries("blogs")} == {
            "Watch implications",
            "Automatic review",
        }
        with __import__("pytest").raises(ValueError, match="Content type"):
            db.list_automatic_queries("videos")

    edited = client.post(
        f"/queries/{query_id}/edit",
        data={
            "name": "Paper-only review",
            "prompt": "Review new papers.",
            "auto_run_papers": "on",
        },
        follow_redirects=False,
    )
    assert edited.status_code == 303
    with Database(config.paths.database) as db:
        query = db.get_saved_query(query_id)
        assert query["name"] == "Paper-only review"
        assert query["auto_run_papers"] == 1
        assert query["auto_run_blogs"] == 0

    page = client.get("/queries")
    assert "Run backward" not in page.text
    assert "Save selections" not in page.text
    assert "Changes save automatically" in page.text
    assert 'data-auto-save="query-selections"' in page.text
    assert "New papers" in page.text and "New blog entries" in page.text

    papers = client.post(
        f"/queries/{query_id}/run-existing/papers",
        data={"model": "default", "context_window_tokens": "3000"},
        follow_redirects=False,
    )
    assert papers.status_code == 303
    assert controller.calls[-1][1]["content_type"] == "papers"
    assert controller.calls[-1][1]["source_scope"] == "auto"

    blogs = client.post(
        f"/queries/{existing_query_id}/run-existing/blogs",
        data={"model": "default"},
        follow_redirects=False,
    )
    assert blogs.status_code == 303
    assert controller.calls[-1][1]["content_type"] == "blogs"
    assert controller.calls[-1][1]["source_scope"] == "blog"

    unknown = client.post(
        f"/queries/{query_id}/run-existing/videos", follow_redirects=False
    )
    assert unknown.status_code == 404

    busy_controller = RecordingQueryStub(result=None)
    busy_client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=busy_controller,
            author_client_factory=lambda config: AuthorClient(),
        )
    )
    busy = busy_client.post(
        f"/queries/{query_id}/run-existing/blogs",
        data={"model": "default"},
        follow_redirects=False,
    )
    assert "already+running" in busy.headers["location"]

    invalid_controller = RecordingQueryStub(error=ValueError("invalid backward run"))
    invalid_client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=invalid_controller,
            author_client_factory=lambda config: AuthorClient(),
        )
    )
    invalid = invalid_client.post(
        f"/queries/{query_id}/run-existing/papers",
        data={"model": "default"},
        follow_redirects=False,
    )
    assert "invalid+backward+run" in invalid.headers["location"]


def test_author_openalex_id_is_editable_inline(config_path):
    config = load_config(config_path)
    client = client_for(config_path)
    client.post(
        "/authors/manual",
        data={"name": "Editable Person", "institution": "Local Lab", "openalex_id": ""},
    )
    client.post(
        "/authors/manual",
        data={"name": "Existing Person", "institution": "Other Lab", "openalex_id": "A123"},
    )
    with Database(config.paths.database) as db:
        local_id = str(db.find_authors_by_name("Editable Person")[0]["openalex_id"])

    page = client.get("/authors")
    assert 'class="author-openalex-form"' in page.text
    assert f'action="/authors/{local_id}/openalex-id"' in page.text
    assert 'id="tracked-authors-table"' in page.text
    assert 'class="author-openalex-column"' in page.text
    assert 'data-author-openalex-column hidden' not in page.text
    assert 'id="toggle-author-ids"' not in page.text
    assert 'title="Edit the ID and press Enter"' in page.text
    assert page.text.index('href="/" data-content-home-link>Content</a>') < page.text.index('href="/authors"')

    blank = client.post(
        f"/authors/{local_id}/openalex-id",
        data={"openalex_id": ""},
        follow_redirects=False,
    )
    assert blank.status_code == 303
    assert "level=error" in blank.headers["location"]

    invalid = client.post(
        f"/authors/{local_id}/openalex-id",
        data={"openalex_id": "bad"},
        follow_redirects=False,
    )
    assert invalid.status_code == 303
    assert "level=error" in invalid.headers["location"]

    duplicate = client.post(
        f"/authors/{local_id}/openalex-id",
        data={"openalex_id": "A123"},
        follow_redirects=False,
    )
    assert duplicate.status_code == 303
    assert "level=error" in duplicate.headers["location"]

    updated = client.post(
        f"/authors/{local_id}/openalex-id",
        data={"openalex_id": "a987654"},
        follow_redirects=False,
    )
    assert updated.status_code == 303
    assert "A987654" in updated.headers["location"]
    with Database(config.paths.database) as db:
        author = db.get_author("A987654")
        assert author is not None and author["enabled"] == 1
        assert db.get_author(local_id) is None

    missing = client.post(
        "/authors/A404/openalex-id",
        data={"openalex_id": "A555"},
        follow_redirects=False,
    )
    assert missing.status_code == 404

    css = client.get("/static/app.css")
    assert ".shell{width:min(1240px" in css.text
    assert ".author-blogs-column" in css.text
    assert ".author-blog-table" in css.text
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css.text
    assert '.blog-source-settings-form > input[type="url"] { grid-column: 1 / -1; }' in css.text
    javascript = client.get("/static/app.js")
    assert 'getElementById("toggle-author-ids")' not in javascript.text
    assert '.author-table.author-ids-visible' not in css.text
    assert '/static/app.css?v=0.10.2' in page.text
    assert '/static/app.js?v=0.10.2' in page.text
