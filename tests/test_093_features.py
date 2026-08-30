from __future__ import annotations

from dataclasses import asdict
import json
import threading
import time

import httpx
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.blogs import _extract_blog_document_full
from paperwatch.config import load_config
from paperwatch.db import Database, SCHEMA_VERSION
from paperwatch.llm import OllamaClient
from paperwatch.query_jobs import (
    QueryBatchSnapshot,
    QueryController,
    _regexp_excerpts,
    preprocess_regexp_documents,
)
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
    def __init__(self):
        self.calls = []

    def snapshot(self):
        return QueryBatchSnapshot()

    def start(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "batch"


class AuthorClient:
    def close(self):
        return None


def _client(config_path, query=None):
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=query or QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def test_ollama_grammar_error_retries_in_plain_json_mode(analysis_result):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": 400,
                        "message": "Failed to initialize samplers: failed to parse grammar",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json={"message": {"content": analysis_result.model_dump_json()}},
        )

    progress = []
    llm = OllamaClient(
        model="test-model",
        base_url="http://ollama.test",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    llm.set_runtime(lambda kind, value: progress.append((kind, value)), None)
    result = llm.analyze("paper")
    assert result.result.name == analysis_result.name
    assert isinstance(requests[0]["format"], dict)
    assert requests[1]["format"] == "json"
    assert "JSON Schema" in requests[1]["messages"][0]["content"]
    assert any("retrying in JSON mode" in value for kind, value in progress if kind == "message")


def test_blog_html_rendering_preserves_inline_links_and_sanitizes_markup():
    html = """
    <html><head><title>Architecture</title></head><body><main>
      <h1>What happened?</h1>
      <p>Encoder-only models (<a href="/bert?utm_source=x" onclick="bad()">e.g., BERT</a>)
      and decoder-only models (<a href="https://example.test/gpt#part">GPT series</a>) are related.</p>
      <script>alert(1)</script><div><p>Next paragraph.</p></div>
      <a href="javascript:alert(1)">unsafe</a>
    </main></body></html>
    """
    title, text, links, rendered = _extract_blog_document_full(
        html, "Fallback", "https://example.test/blog/post"
    )
    assert title == "Architecture"
    assert "models ([e.g., BERT](https://example.test/bert)) and decoder" in text
    assert "GPT series](https://example.test/gpt)" in text
    assert "\ne.g., BERT\n" not in text
    assert links == ("https://example.test/bert", "https://example.test/gpt")
    assert "onclick" not in rendered and "script" not in rendered and "javascript:" not in rendered
    assert 'target="_blank"' in rendered


def test_regexp_preprocessing_radius_merging_and_local_query(
    config_path, author_candidate
):
    excerpt, count = _regexp_excerpts(
        "zero one two target three four five target six seven", r"target", 2
    )
    assert count == 2
    assert excerpt == "one two target three four five target six seven"
    assert _regexp_excerpts("nothing", r"target", 2) == ("", 0)
    docs, matches = preprocess_regexp_documents(
        [
            {"title": "keep", "content": "alpha beta TARGET gamma delta"},
            {"title": "drop", "content": "unrelated"},
            {"title": "metadata only"},
        ],
        r"(?i)target",
        1,
    )
    assert matches == 1 and len(docs) == 1
    assert docs[0]["content"] == "beta TARGET gamma"

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Regexp entry",
            content_sha256="regexp",
            normalized_text="alpha beta TARGET gamma delta",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
        )
        query_id = db.create_saved_query(
            name="Local matches",
            prompt="",
            regexp_pattern=r"(?i)target",
            regexp_radius=1,
            regexp_mode="print",
        )
        stored = db.get_saved_query(query_id)
        assert stored["prompt"] == "Print regexp matches."
        assert stored["regexp_radius"] == 1

    controller = QueryController(config_path)
    batch = controller.start(
        query_id,
        (),
        2048,
        selection={"content_type": "blogs", "blog_entry_ids": [entry_id]},
        content_type="blogs",
    )
    assert batch
    controller.wait(10)
    with Database(config.paths.database) as db:
        run = db.query_runs_for_batch(batch)[0]
        output = json.loads(run["output_json"])
        assert run["provider"] == "local"
        assert run["model"] == "regexp"
        assert output["items"][0]["output"] == "beta TARGET gamma"


def test_regexp_preprocessing_parallelizes_content_records(monkeypatch):
    import paperwatch.query_jobs as query_jobs

    original = query_jobs._regexp_document_match
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def observed(document, expression, radius):
        with lock:
            thread_ids.add(threading.get_ident())
        time.sleep(0.005)
        return original(document, expression, radius)

    monkeypatch.setattr(query_jobs, "_regexp_document_match", observed)
    documents = [
        {"paper_id": index, "title": f"Paper {index}", "abstract": f"alpha target {index}"}
        for index in range(24)
    ]
    processed, matches = query_jobs.preprocess_regexp_documents(
        documents, r"target", 1
    )
    assert matches == len(documents)
    assert [item["paper_id"] for item in processed] == list(range(24))
    assert len(thread_ids) > 1


def test_shared_tags_direct_blog_import_rendering_and_fetch_history(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        run_id = db.start_run()
        db.finish_run(
            run_id,
            "success",
            {
                "authors_total": 3,
                "authors_checked": 2,
                "papers_found": 4,
                "papers_downloadable": 3,
                "papers_downloaded": 2,
                "blogs_total": 5,
                "blogs_checked": 4,
                "blogs_with_content": 3,
                "blog_items_available": 7,
                "blog_items_downloaded": 6,
                "blog_updates_found": 1,
            },
        )
    client = _client(config_path)
    assert client.get("/blogs/import").status_code == 200
    invalid = client.post(
        "/blogs/import",
        data={"author_id": "", "entry_title": "x", "content": "x"},
        follow_redirects=False,
    )
    assert "Choose+a+tracked+author" in invalid.headers["location"]
    response = client.post(
        "/blogs/import",
        data={
            "author_id": author_candidate.openalex_id,
            "blog_label": "Manual source",
            "blog_url": "https://example.test/writing/",
            "relationship": "Direct author",
            "entry_title": "Imported entry",
            "entry_url": "https://example.test/writing/imported",
            "published_at": "2026-08-02T08:00:00+02:00",
            "content": "A complete imported article about retrieval systems.",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    entry_id = int(response.headers["location"].split("/blogs/")[1].split("?")[0])
    detail = client.get(f"/blogs/{entry_id}")
    assert 'data-copy-source="blog-copy-source"' in detail.text
    assert "Import" not in detail.text or detail.status_code == 200

    assert client.post(
        f"/blogs/{entry_id}/tags/add", data={"tag": "retrieval"}, follow_redirects=False
    ).status_code == 303
    assert client.post(
        f"/papers/{paper_id}/tags/add", data={"tag": "retrieval"}, follow_redirects=False
    ).status_code == 303
    with Database(config.paths.database) as db:
        tags = db.list_content_tags()
        assert len(tags) == 1
        assert tags[0]["paper_count"] == 1 and tags[0]["blog_count"] == 1
        papers, total = db.search_papers(query="retrieval")
        assert total == 1 and papers[0]["id"] == paper_id
        blogs = db.list_blog_entries(query="retrieval")
        assert [row["id"] for row in blogs] == [entry_id]
        assert db.search_papers(content_tags=["retrieval"])[1] == 1
        assert len(db.list_blog_entries(tags=["retrieval"])) == 1

    assert "Imported entry" in client.get("/blogs?tag=retrieval").text
    assert paper_record.title in client.get("/?tag=retrieval").text
    assert "retrieval" in client.get(f"/papers/{paper_id}").text
    assert "regexp_pattern" in client.get("/queries/new").text
    history = client.get("/runs")
    assert "3</strong> author links" in history.text
    assert "5</strong> source links" in history.text
    run_detail = client.get(f"/runs/{run_id}")
    assert "Papers with PDF data" in run_detail.text
    assert "Sources with downloadable data" in run_detail.text

    assert client.post(
        f"/blogs/{entry_id}/tags/remove", data={"tag": "retrieval"}, follow_redirects=False
    ).status_code == 303
    assert client.post(
        f"/papers/{paper_id}/tags/remove", data={"tag": "retrieval"}, follow_redirects=False
    ).status_code == 303
    with Database(config.paths.database) as db:
        assert db.list_content_tags() == []
        assert SCHEMA_VERSION == 34


def test_regexp_and_tag_validation_edges(config_path):
    import pytest

    assert preprocess_regexp_documents([{"content": "unchanged"}], "", 4) == (
        [{"content": "unchanged"}],
        0,
    )
    assert _regexp_excerpts("", r"^", 0) == ("", 1)

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        for pattern, radius, mode, message in (
            ("x" * 2001, 0, "query", "at most 2000"),
            ("(", 0, "query", "Invalid regexp"),
            ("x", -1, "query", "between 0 and 5000"),
            ("x", 5001, "query", "between 0 and 5000"),
            ("x", 0, "other", "query or print"),
            ("", 0, "print", "requires a regexp"),
        ):
            with pytest.raises(ValueError, match=message):
                db.create_saved_query(
                    name="Validation",
                    prompt="Prompt",
                    regexp_pattern=pattern,
                    regexp_radius=radius,
                    regexp_mode=mode,
                )
        with pytest.raises(ValueError, match="Query name and prompt"):
            db.create_saved_query(name="", prompt="")
        with pytest.raises(ValueError, match="Query name is required"):
            db.create_saved_query(
                name="", prompt="", regexp_pattern="x", regexp_mode="print"
            )
        with pytest.raises(ValueError, match="Prompt is required"):
            db.create_saved_query(name="Named", prompt="")

        query_id = db.create_saved_query(
            name="Printable",
            prompt="",
            regexp_pattern="target",
            regexp_mode="print",
        )
        with pytest.raises(ValueError, match="Query name and prompt"):
            db.update_saved_query(query_id, name="", prompt="")
        with pytest.raises(ValueError, match="Query name is required"):
            db.update_saved_query(
                query_id,
                name="",
                prompt="",
                regexp_pattern="target",
                regexp_mode="print",
            )
        with pytest.raises(ValueError, match="Prompt is required"):
            db.update_saved_query(query_id, name="Named", prompt="")
        assert db.update_saved_query(
            query_id,
            name="Printable",
            prompt="",
            regexp_pattern="target",
            regexp_mode="print",
        )
        assert db.get_saved_query(query_id)["prompt"] == "Print regexp matches."
        assert db.add_paper_tag(999999, "missing") is False
        assert db.add_blog_entry_tag(999999, "missing") is False


def test_direct_import_validation_duplicate_versions_and_tag_errors(
    config_path, author_candidate
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
    client = _client(config_path)

    base = {
        "author_id": author_candidate.openalex_id,
        "blog_url": "https://example.test/blog/",
        "entry_url": "https://example.test/blog/post",
        "entry_title": "Post",
        "content": "Body",
    }
    invalid_cases = (
        ({**base, "entry_title": ""}, "Entry+title+is+required"),
        ({**base, "content": ""}, "Entry+text+is+required"),
        ({**base, "entry_title": "x" * 1001}, "Entry+title+is+too+long"),
        ({**base, "content": "x" * 5_000_001}, "Entry+text+is+too+long"),
        ({**base, "published_at": "not-a-date"}, "Published+time+must+be+ISO+8601"),
        ({**base, "author_id": "https://openalex.org/A999999"}, "Tracked+author+was+not+found"),
    )
    for data, expected in invalid_cases:
        response = client.post("/blogs/import", data=data, follow_redirects=False)
        assert response.status_code == 303
        assert expected in response.headers["location"]

    first = client.post("/blogs/import", data=base, follow_redirects=False)
    first_id = int(first.headers["location"].split("/blogs/")[1].split("?")[0])
    duplicate = client.post("/blogs/import", data=base, follow_redirects=False)
    duplicate_id = int(duplicate.headers["location"].split("/blogs/")[1].split("?")[0])
    assert duplicate_id == first_id
    changed = client.post(
        "/blogs/import", data={**base, "content": "Changed body"}, follow_redirects=False
    )
    changed_id = int(changed.headers["location"].split("/blogs/")[1].split("?")[0])
    assert changed_id == first_id
    with Database(config.paths.database) as db:
        changed_row = db.get_blog_entry(changed_id)
        assert changed_row["previous_entry_id"] is None
        assert "Changed body" in changed_row["normalized_text"]
        assert changed_row["diff_text"]

    blank_blog_tag = client.post(
        f"/blogs/{changed_id}/tags/add", data={"tag": ""}, follow_redirects=False
    )
    assert blank_blog_tag.status_code == 303 and "Tag+cannot+be+empty" in blank_blog_tag.headers["location"]
    blank_blog_remove = client.post(
        f"/blogs/{changed_id}/tags/remove", data={"tag": ""}, follow_redirects=False
    )
    assert blank_blog_remove.status_code == 303 and "Tag+cannot+be+empty" in blank_blog_remove.headers["location"]
    assert client.post(
        "/blogs/999999/tags/add", data={"tag": "x"}, follow_redirects=False
    ).status_code == 404
