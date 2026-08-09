from __future__ import annotations

import re

from paperwatch.config import load_config
from paperwatch.default_queries import (
    DEFAULT_SAVED_QUERIES,
    LEGACY_REGEXP_PATTERNS,
    add_default_queries,
    refresh_default_regexp_patterns,
)
from paperwatch.db import Database
from paperwatch.query_jobs import regexp_tag_result
from paperwatch.web import _render_regexp_excerpt


def _regexp_defaults() -> dict[str, str]:
    return {
        str(item["name"]): str(item["regexp_pattern"])
        for item in DEFAULT_SAVED_QUERIES
        if item.get("regexp_mode") == "print"
    }


def test_regexp_defaults_include_specific_low_noise_terms():
    patterns = _regexp_defaults()
    positives = {
        "Funding and grants": "This work was supported by the NSF under grant agreement 1234.",
        "Defense and military references": "Experiments were funded by the Air Force Research Laboratory.",
        "Source code and repositories": "Code is at https://github.com/example/research-system",
        "Data, weights, and research artifacts": "Pretrained weights have been publicly released.",
        "Compute and hardware requirements": "Inference used an MI300X with 192 GB of GPU memory.",
        "Limitations and impact sections": "7. Ethical Considerations",
        "Real-world deployment evidence": "The service was evaluated under live traffic.",
        "Statistical uncertainty and rigor": "We report Cohen's d with a 95% CI.",
        "Strong novelty and performance claims": "The method improves over the baseline by 8%.",
        "Failures, negative results, and impossibility": "The larger model underperforms on the held-out split.",
    }
    for name, text in positives.items():
        assert re.search(patterns[name], text, re.MULTILINE), name

    repository = re.search(
        patterns["Source code and repositories"],
        "See https://github.com/example/research-system for the implementation.",
        re.MULTILINE,
    )
    assert repository is not None
    assert repository.group(0) == "github.com/example/research-system"
    artifact = re.search(
        patterns["Data, weights, and research artifacts"],
        "Weights: https://huggingface.co/example/model",
        re.MULTILINE,
    )
    assert artifact is not None
    assert artifact.group(0) == "huggingface.co/example/model"

    negatives = {
        "Funding and grants": "The grant proposal deadline is Friday.",
        "Source code and repositories": "The source material was coded by two annotators.",
        "Compute and hardware requirements": "Training improves the final accuracy.",
        "Real-world deployment evidence": "The production value was estimated analytically.",
        "Strong novelty and performance claims": "The first author collected the data.",
        "Failures, negative results, and impossibility": "The lower panel shows the result.",
    }
    for name, text in negatives.items():
        assert re.search(patterns[name], text, re.MULTILINE) is None, name


def test_regexp_generated_url_tag_drops_scheme(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Repository link",
            prompt="",
            query_kind="regexp",
            content_scope="papers",
            regexp_pattern=r"https?://github\.com/[^\s]+",
            paper_source_scope="abstract",
        )
        run_id = db.create_query_runs(
            batch_id="url-regexp-tag",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"content_type": "papers", "paper_ids": [paper_id]},
        )[0]
        changed = db.apply_query_classifications(
            query_id,
            run_id,
            {
                "items": [
                    {
                        "paper_id": paper_id,
                        "content_type": "paper",
                        "output": "https://github.com/example/research-system",
                    }
                ]
            },
        )
        assert changed == 1
        tag = db.classification_results_for_content("paper", paper_id)[0]
        assert tag["value"] == "github.com/example/research-system"
        assert tag["tag_name"] == "github.com/example/research-system"

    result = regexp_tag_result(
        [
            {
                "paper_id": paper_id,
                "title": "Repository",
                "abstract": "See https://github.com/example/research-system",
            }
        ],
        r"https?://github\.com/[^\s]+",
    )
    assert result.items[0].output == "github.com/example/research-system"
    rendered = _render_regexp_excerpt(
        "See https://github.com/example/research-system",
        r"https?://github\.com/[^\s]+",
    )
    assert "https://<mark" in rendered
    assert '<mark class="regexp-match">github.com/example/research-system</mark>' in rendered


def test_existing_untouched_regexp_defaults_are_refreshed_but_edits_are_preserved(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        add_default_queries(db)
        source = next(
            row for row in db.list_saved_queries()
            if row["name"] == "Source code and repositories"
        )
        compute = next(
            row for row in db.list_saved_queries()
            if row["name"] == "Compute and hardware requirements"
        )
        with db.transaction() as conn:
            conn.execute(
                "UPDATE saved_queries SET regexp_pattern=? WHERE id=?",
                (LEGACY_REGEXP_PATTERNS["Source code and repositories"], int(source["id"])),
            )
            conn.execute(
                "UPDATE saved_queries SET regexp_pattern='CUSTOM-HARDWARE' WHERE id=?",
                (int(compute["id"]),),
            )

        assert refresh_default_regexp_patterns(db) == 1
        refreshed = db.get_saved_query(int(source["id"]))
        edited = db.get_saved_query(int(compute["id"]))
        assert "github\\.com" in refreshed["regexp_pattern"]
        assert "https?://" not in refreshed["regexp_pattern"]
        assert edited["regexp_pattern"] == "CUSTOM-HARDWARE"
