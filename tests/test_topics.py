from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from paperwatch.db import Database, SCHEMA_VERSION
from paperwatch.models import PaperTopic, parse_openalex_topics


def test_parse_openalex_topics_ranks_primary_and_ignores_malformed_items():
    work = {
        "topics": [
            {
                "id": "https://openalex.org/T10",
                "display_name": "Learning Theory",
                "score": "0.8",
                "subfield": "bad",
                "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
                "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
            },
            "bad",
            {"id": "W1", "display_name": "Not a topic"},
            {"id": "T20", "display_name": "  ", "score": 0.2},
            {"id": "T30", "display_name": "Robust Learning", "score": "bad"},
        ],
        "primary_topic": {
            "id": "https://openalex.org/T10",
            "display_name": "Learning Theory",
            "score": 0.95,
            "subfield": {
                "id": "https://openalex.org/subfields/1702",
                "display_name": "Artificial Intelligence",
            },
            "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
        },
    }

    topics = parse_openalex_topics(work)
    assert [topic.openalex_id for topic in topics] == ["T10", "T30"]
    assert topics[0].is_primary is True
    assert topics[0].score == 0.95
    assert topics[0].subfield_name == "Artificial Intelligence"
    assert topics[1].score is None
    assert parse_openalex_topics({"topics": {}, "primary_topic": "bad"}) == []


def test_topics_are_persisted_updated_filtered_and_reconstructed(tmp_path, paper_record):
    path = tmp_path / "topics.db"
    with Database(path) as db:
        paper_id, created = db.upsert_paper(paper_record)
        assert created is True
        rows = db.paper_topics(paper_id)
        assert [row["openalex_id"] for row in rows] == ["T100", "T200"]
        assert rows[0]["is_primary"] == 1
        assert rows[0]["field_name"] == "Computer Science"

        listed = db.list_topics()
        assert [(row["display_name"], row["paper_count"]) for row in listed] == [
            ("Autonomous Agents", 1),
            ("Machine Learning", 1),
        ]

        matching, total = db.search_papers(topic_id="T100")
        assert total == 1
        assert matching[0]["title"] == paper_record.title
        assert json.loads(matching[0]["topics_json"])[0]["name"] == "Machine Learning"
        assert db.search_papers(topic_id="T999")[1] == 0
        with pytest.raises(ValueError, match="topic id"):
            db.search_papers(topic_id="A123")

        reconstructed = db.get_paper_record(paper_id)
        assert reconstructed is not None
        assert reconstructed.topics[0].name == "Machine Learning"
        assert reconstructed.topics[0].is_primary is True

        updated = replace(
            paper_record,
            topics=[
                PaperTopic(
                    openalex_id="T100",
                    name="Machine Learning and AI",
                    score=0.99,
                    is_primary=True,
                    field_name="Computer Science",
                    metadata={"display_name": "Machine Learning and AI"},
                )
            ],
        )
        same_id, created = db.upsert_paper(updated)
        assert same_id == paper_id and created is False
        assert [row["display_name"] for row in db.paper_topics(paper_id)] == [
            "Machine Learning and AI"
        ]


def test_version_five_database_backfills_topics_from_metadata(tmp_path, paper_record):
    path = tmp_path / "migration.db"
    with Database(path) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db._conn.execute("DELETE FROM paper_topics")
        db._conn.execute("DELETE FROM topics")
        db._conn.execute("PRAGMA user_version = 5")
        db._conn.commit()

    with Database(path) as db:
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert [row["openalex_id"] for row in db.paper_topics(paper_id)] == ["T100", "T200"]

    conn = sqlite3.connect(path)
    conn.execute("UPDATE papers SET metadata_json='not-json'")
    conn.execute("DELETE FROM paper_topics")
    conn.execute("DELETE FROM topics")
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()
    with Database(path) as db:
        assert db.paper_topics(paper_id) == []
