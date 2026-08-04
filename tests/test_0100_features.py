from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.blogs import ArticleItem
from paperwatch.config import load_config
from paperwatch.db import Database, SCHEMA_VERSION
from paperwatch.pipeline import PaperWatch
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, _paper_pdf_state, create_app


class RunStub:
    def snapshot(self):
        return RunSnapshot()

    def start(self, *args, **kwargs):
        return True


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryCapture:
    def __init__(self):
        self.calls = []

    def snapshot(self):
        return QueryBatchSnapshot()

    def start(self, *args, **kwargs):
        self.calls.append(("one", args, kwargs))
        return "batch"

    def start_many(self, *args, **kwargs):
        self.calls.append(("many", args, kwargs))
        return "batch-many"


class AuthorClient:
    def close(self):
        return None


def client(config_path: Path, query_controller=None) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=query_controller or QueryCapture(),
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
            "https://example.test/writing/",
            label="Writing",
            source_mode="article_index",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Stored systems note",
            content_sha256="systems-note",
            normalized_text="Title: Stored systems note\nURL: https://example.test/writing/note\nNeedle body",
            rendered_html="<p>Needle body</p>",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="systems-note",
            source_url="https://example.test/writing/note",
        )
    return paper_id, blog_id, entry_id


def test_schema_25_pdf_fingerprint_persists_modified_state(
    tmp_path, config_path, author_candidate, paper_record
):
    paper_id, _, _ = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    pdf = tmp_path / "paper.pdf"
    original = b"%PDF-1.4\noriginal\n%%EOF\n"
    changed = b"%PDF-1.4\nchanged content\n%%EOF\n"
    pdf.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    with Database(config.paths.database) as db:
        db.set_local_pdf(paper_id, str(pdf), expected)
        row = dict(db.get_paper(paper_id))
        assert SCHEMA_VERSION == 25
        assert "pdf_modified_on_disk" in row
        assert _paper_pdf_state(db, row)["pdf_modified"] is False

    pdf.write_bytes(changed)
    with Database(config.paths.database) as db:
        verified = _paper_pdf_state(db, dict(db.get_paper(paper_id)))
        assert verified["pdf_modified"] is True
        persisted = dict(db.get_paper(paper_id))
        assert persisted["pdf_modified_on_disk"] == 1
        # The Content list uses the cheap fingerprint and persisted state.
        assert _paper_pdf_state(db, persisted, verify_hash=False)["pdf_modified"] is True

    pdf.write_bytes(original)
    with Database(config.paths.database) as db:
        restored = _paper_pdf_state(db, dict(db.get_paper(paper_id)))
        assert restored["pdf_modified"] is False
        assert db.get_paper(paper_id)["pdf_modified_on_disk"] == 0


def test_content_detail_data_and_query_layouts(
    config_path, author_candidate, paper_record
):
    paper_id, _, entry_id = seed(config_path, author_candidate, paper_record)
    web = client(config_path)
    content = web.get("/")
    assert content.status_code == 200
    for value in (
        "Search all stored text",
        "Reset search options",
        "Star · Type",
        "content-star-type",
        "content-global-search-row",
    ):
        assert value in content.text
    assert 'class="panel selected-content-dock"' in content.text and ' hidden' in content.text

    data = web.get("/data")
    assert "Add author manually" in data.text
    assert "Import blog links" in data.text
    assert "Delete stored records" in data.text
    assert "Author CSV schema" in data.text and "Blogs CSV schema" in data.text
    authors_page = web.get("/authors").text
    assert "Add author manually" not in authors_page
    assert "Import blog links" not in authors_page
    assert "Delete stored records" not in web.get("/runs").text

    paper = web.get(f"/papers/{paper_id}")
    assert "content-tags-panel" not in paper.text
    assert "data-pdf-drop-target" in paper.text
    assert "data-pdf-upload-form" in paper.text
    assert "data-auto-upload" in paper.text
    assert "inline-notes-panel" in paper.text
    assert "Type and press Enter" in paper.text

    blog = web.get(f"/blogs/{entry_id}")
    assert "content-tags-panel" not in blog.text
    assert "inline-notes-panel" in blog.text
    assert "blog-heading-actions" in blog.text

    query = web.get("/queries/new")
    assert "regexp-editor-grid" in query.text
    assert 'data-regexp-debug="preprocess"' in query.text
    assert 'data-regexp-debug="classification"' in query.text
    assert "Relevant = (?i)" in query.text


def test_global_search_uses_notes_tags_authors_and_blog_body(
    config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.set_object_notes("paper", paper_id, "quasar-paper-note")
        db.set_object_notes("blog_source", blog_id, "nebula-source-note")
        db.add_author_tag(author_candidate.openalex_id, "Cosmology")
        db.add_blog_source_tag(blog_id, "Observatory")

        assert db.search_papers(query="quasar-paper-note")[1] == 1
        assert db.search_papers(query="Cosmology")[1] == 1
        assert db.count_blog_entries(query="Needle body") == 1
        assert db.count_blog_entries(query="nebula-source-note") == 1
        assert db.count_blog_entries(query="Observatory") == 1

    web = client(config_path)
    assert "Minimal Agents" in web.get("/?q=quasar-paper-note").text
    assert "Stored systems note" in web.get("/?q=nebula-source-note").text


def test_arxiv_fallback_date_uses_original_submission_date():
    article = ArticleItem(
        external_id="x",
        title="Paper",
        url="https://arxiv.org/abs/1902.04546",
        text=(
            "Title: Paper\nURL: https://arxiv.org/abs/1902.04546\n"
            "[Submitted on 12 Feb 2019 (v1), last revised 4 Mar 2020 (this version, v2)]"
        ),
        sha256="x",
    )
    assert PaperWatch._arxiv_fallback_date(article) == "2019-02-12"


def test_blog_results_runner_accepts_multiple_queries_and_one_model(
    config_path, author_candidate, paper_record
):
    _, _, entry_id = seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.set_content_selected("blog", entry_id, True)
        first = db.create_saved_query(name="First", prompt="First prompt")
        second = db.create_saved_query(name="Second", prompt="Second prompt")
    capture = QueryCapture()
    web = client(config_path, capture)
    content = web.get("/")
    assert "Run saved prompt on selected blogs (1)" in content.text
    assert ">Run on selected<" in content.text
    assert 'name="context_window_tokens"' not in content.text.split(
        'class="blog-results-runner"', 1
    )[1].split("</form>", 1)[0]
    response = web.post(
        "/queries/run-blogs",
        data={
            "query_id": [str(first), str(second)],
            "model": "default",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    kind, args, kwargs = capture.calls[-1]
    assert kind == "many"
    assert args[:3] == (
        (first, second),
        "default",
        config.llm.context_window_tokens,
    )
    assert kwargs["content_type"] == "blogs"
    assert kwargs["selection"]["blog_entry_ids"] == [entry_id]
    assert "query" not in kwargs["selection"]

    selected_head = content.text.split("selected-content-table", 1)[1].split(
        "</thead>", 1
    )[0]
    headers = re.findall(r"<th.*?</th>", selected_head, flags=re.DOTALL)
    assert 'action="/content/selected/unselect"' in headers[0]
    assert re.search(r">☑</button>", headers[0])
    assert 'action="/content/selected/star"' in headers[1]
    assert re.search(r">☆</button>", headers[1])
    assert headers[1].count("<form") == 1
    assert 'action="/content/selected/read"' in headers[2]
    assert re.search(r">○</button>", headers[2])
    assert headers[2].count("<form") == 1
    assert "Remove all tags" in headers[6]
    assert "Undo tag action" not in headers[6]
    selected_tools = content.text.split('class="selected-content-tools"', 1)[1].split("</div>", 1)[0]
    assert selected_tools.index("Add tag and press Enter") < selected_tools.index("Remove tag and press Enter")
    assert selected_tools.index("Remove tag and press Enter") < selected_tools.index("Undo tag action")

    with Database(config.paths.database) as db:
        db.set_content_starred("blog", entry_id, True)
        db.set_content_read("blog", entry_id, True)
    selected_head = web.get("/").text.split("selected-content-table", 1)[1].split(
        "</thead>", 1
    )[0]
    headers = re.findall(r"<th.*?</th>", selected_head, flags=re.DOTALL)
    assert 'action="/content/selected/unstar"' in headers[1]
    assert re.search(r">★</button>", headers[1])
    assert 'action="/content/selected/unread"' in headers[2]
    assert re.search(r">✓</button>", headers[2])


def test_authors_table_restores_columns_and_places_notes_below_row(
    config_path, author_candidate, paper_record
):
    seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        author = db.get_author(author_candidate.openalex_id)
        db.set_object_notes("author", int(author["id"]), "Visible author note")

    response = client(config_path).get("/authors")
    assert response.status_code == 200
    assert "Add author manually" not in response.text
    assert "Import blog links" not in response.text
    row = response.text.split('data-author-search="Ada Lovelace', 1)[1].split(
        "</tr>", 1
    )[0]
    for cell_class in (
        "author-identity-cell",
        "author-star-cell",
        "author-openalex-cell",
        "author-blogs-cell",
        "tag-entry-cell",
        "icon-action-cell",
    ):
        assert cell_class in row
    assert "row-notes-editor" not in row
    notes_row = response.text.split('class="author-notes-row"', 1)[1].split(
        "</tr>", 1
    )[0]
    assert 'colspan="9"' in notes_row
    assert "Visible author note" in notes_row
    assert 'data-filter-name="starred"' in response.text
    assert 'data-filter-name="notes"' in response.text
    assert 'data-filter-name="enabled"' in response.text
    assert 'name="tag" form="author-filter-form" multiple' in response.text
    assert "<span>Star</span>" not in response.text


def test_compact_content_and_author_header_filters(
    config_path, author_candidate, paper_record
):
    seed(config_path, author_candidate, paper_record)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        assert db.add_author_tag("A123", "theory")
        assert db.add_author_tag("A123", "ML")
        author = db.get_author("A123")
        assert author is not None
        db.set_object_notes("author", int(author["id"]), "Filter me")
        db.set_author_starred("A123", True)
        db.set_author_enabled("A123", False)

    web = client(config_path)
    content = web.get("/")
    assert "Any reading state" not in content.text
    assert "Any notes state" not in content.text
    assert 'data-filter-name="starred" data-filter-values="any|yes|no"' in content.text
    assert 'data-filter-name="content" data-filter-values="all|papers|blogs"' in content.text
    assert 'data-filter-name="read" data-filter-values="any|yes|no"' in content.text
    assert 'data-filter-name="notes" data-filter-values="any|yes|no"' in content.text

    authors = web.get(
        "/authors",
        params=[
            ("tag", "theory"),
            ("tag", "ML"),
            ("starred", "yes"),
            ("notes", "yes"),
            ("enabled", "no"),
        ],
    )
    assert authors.status_code == 200
    assert "Ada Lovelace" in authors.text
    assert 'value="theory" selected' in authors.text
    assert 'value="ML" selected' in authors.text
    assert 'data-filter-name="starred"' in authors.text
    assert 'data-filter-name="notes"' in authors.text
    assert 'data-filter-name="enabled"' in authors.text
    assert 'name="tag" form="author-filter-form" multiple' in authors.text


def test_static_assets_use_partial_updates_and_requested_layout():
    root = Path(__file__).parents[1] / "src" / "paperwatch"
    script = (root / "web_static" / "app.js").read_text()
    css = (root / "web_static" / "app.css").read_text()
    assert "paperwatch-content-url" in script
    assert "#content-results-root" in script
    assert "syncSelectionRow" in script
    assert 'form.setAttribute("action", `/content/selected/${starred ? "unstar" : "star"}`)' in script
    assert 'form.setAttribute("action", `/content/selected/${read ? "unread" : "read"}`)' in script
    assert "pdfDropTarget" in script
    assert "data-regexp-debug" in script
    assert "control.disabled = locked" in script
    assert '["codex", "claude-code"]' in script
    assert ".unified-content-table{min-width:1540px" in css
    assert '.content-row[data-read="false"]' in css
    assert ".notes-present" in css
    assert '[data-filter-cycle]' in script
    assert 'button.dataset.filterState = nextState' in script
    assert '.filter-word-button[data-filter-state="yes"]' in css
    assert ".selected-content-tools{display:grid;grid-template-columns:220px 220px auto" in css
    assert ".regexp-editor-grid" in css
    assert ".table-wrap>.author-table{width:100%;min-width:1020px;table-layout:fixed}" in css
