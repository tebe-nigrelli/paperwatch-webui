from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.blog_import import import_blog_links
from paperwatch.config import load_config
from paperwatch.db import Database, DeadSourceError
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


def seed(config_path, author_candidate, paper_record) -> tuple[int, int, int]:
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog/",
            label="Example blog",
            relationship="Direct author",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Stored post",
            content_sha256="stored-post",
            normalized_text="Title: Stored post\nURL: https://example.test/blog/post\n\nBody",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="stored-post",
            source_url="https://example.test/blog/post",
        )
    return paper_id, blog_id, entry_id


def test_delete_buttons_create_permanent_paper_and_blog_tombstones(
    config_path, author_candidate, paper_record
):
    paper_id, blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    client = web_client(config_path)

    assert f'action="/papers/{paper_id}/delete"' in client.get(
        f"/papers/{paper_id}"
    ).text
    assert f'action="/blogs/{entry_id}/delete"' in client.get(
        f"/blogs/{entry_id}"
    ).text

    paper_delete = client.post(f"/papers/{paper_id}/delete", follow_redirects=False)
    blog_delete = client.post(f"/blogs/{entry_id}/delete", follow_redirects=False)
    assert paper_delete.status_code == 303
    assert blog_delete.status_code == 303

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id) is None
        paper = db.get_paper(paper_id, include_dead=True)
        assert paper is not None and paper["dead"] == 1
        assert db.search_papers(limit=None)[1] == 0
        assert db.search_papers(limit=None, include_dead=True)[1] == 1
        assert db.get_blog_entry(entry_id) is None
        assert db.list_enabled_blogs() == []
        blog = db.author_blogs(author_candidate.openalex_id)[0]
        assert blog["id"] == blog_id and blog["dead"] == 1 and blog["enabled"] == 0
        with pytest.raises(DeadSourceError, match="marked dead"):
            db.upsert_paper(paper_record)
        with pytest.raises(DeadSourceError, match="marked dead"):
            db.add_author_blog(
                author_candidate.openalex_id, "https://example.test/blog/"
            )

    assert client.get(f"/papers/{paper_id}").status_code == 404
    assert client.get(f"/blogs/{entry_id}").status_code == 404


def test_dead_markers_round_trip_through_exports_and_imports(
    tmp_path, config_path, author_candidate, paper_record
):
    paper_id, _blog_id, entry_id = seed(config_path, author_candidate, paper_record)
    client = web_client(config_path)
    client.post(f"/papers/{paper_id}/delete", follow_redirects=False)
    client.post(f"/blogs/{entry_id}/delete", follow_redirects=False)

    paper_json = client.get(f"/papers/{paper_id}/export.json")
    assert paper_json.status_code == 200 and paper_json.json()["dead"] == 1
    bulk_json = client.get("/exports/papers.json")
    assert bulk_json.status_code == 200
    assert [(row["id"], row["dead"]) for row in bulk_json.json()] == [(paper_id, 1)]
    paper_csv = list(csv.DictReader(io.StringIO(client.get("/exports/papers.csv").text)))
    assert paper_csv[0]["dead"] == "1"

    blog_csv_response = client.get("/exports/blogs.csv")
    blog_rows = list(csv.DictReader(io.StringIO(blog_csv_response.text)))
    assert blog_rows[0]["Dead"] == "yes"

    target_csv = tmp_path / "csv-import.db"
    with Database(target_csv) as target:
        target.add_author(author_candidate)
        stats = import_blog_links(target, io.StringIO(blog_csv_response.text))
        assert stats.links == 1
        imported = target.author_blogs(author_candidate.openalex_id)[0]
        assert imported["dead"] == 1
        assert target.list_enabled_blogs() == []

    config = load_config(config_path)
    with Database(config.paths.database) as source:
        catalog = source.export_author_blog_catalog()
    assert catalog["authors"][0]["blogs"][0]["dead"] == 1
    with Database(tmp_path / "catalog-import.db") as target:
        assert target.import_author_blog_catalog(catalog) == (1, 1)
        imported = target.author_blogs(author_candidate.openalex_id)[0]
        assert imported["dead"] == 1

    legacy_catalog = catalog.copy()
    legacy_catalog["authors"] = [dict(catalog["authors"][0])]
    legacy_catalog["authors"][0]["blogs"] = [
        {
            key: value
            for key, value in catalog["authors"][0]["blogs"][0].items()
            if key != "dead"
        }
    ]
    with Database(tmp_path / "legacy-catalog.db") as target:
        assert target.import_author_blog_catalog(legacy_catalog) == (1, 1)
        assert target.author_blogs(author_candidate.openalex_id)[0]["dead"] == 0

    archive = client.get("/data/export/all")
    assert archive.status_code == 200
    with Database(config.paths.database) as db:
        db._conn.execute("UPDATE papers SET dead=0")
        db._conn.execute("UPDATE blogs SET dead=0, enabled=1")
        db._conn.commit()
    restored = client.post(
        "/data/import/all",
        files={"file": ("backup.zip", archive.content, "application/zip")},
        follow_redirects=False,
    )
    assert restored.status_code == 303
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id, include_dead=True)["dead"] == 1
        assert db.author_blogs(author_candidate.openalex_id)[0]["dead"] == 1
