from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .blogs import normalize_blog_url
from .db import Database


@dataclass(slots=True)
class BlogImportStats:
    rows: int = 0
    links: int = 0
    authors_created: int = 0
    authors_missing: int = 0
    authors_ambiguous: int = 0
    invalid_urls: int = 0


def import_blog_links(
    db: Database,
    stream: TextIO,
    *,
    create_missing: bool = False,
) -> BlogImportStats:
    reader = csv.DictReader(stream)
    required = {"Researcher / author", "Primary source URL"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("CSV must contain Researcher / author and Primary source URL columns")
    stats = BlogImportStats()
    for row in reader:
        stats.rows += 1
        name = " ".join((row.get("Researcher / author") or "").strip().split())
        if not name:
            stats.authors_missing += 1
            continue
        matches = db.find_authors_by_name(name)
        if not matches and create_missing:
            db.add_manual_author(name, row.get("Current lab / organization"))
            matches = db.find_authors_by_name(name)
            stats.authors_created += 1
        if not matches:
            stats.authors_missing += 1
            continue
        if len(matches) != 1:
            stats.authors_ambiguous += 1
            continue
        author_id = str(matches[0]["openalex_id"])
        sources = (
            (
                row.get("Primary source URL") or "",
                row.get("Primary applied-ML source") or "",
                row.get("Relationship to author") or "",
            ),
            (
                row.get("Secondary source URL") or "",
                row.get("Secondary nearby source") or "",
                row.get("Secondary relationship") or "",
            ),
        )
        for raw_url, label, relationship in sources:
            if not raw_url.strip():
                continue
            try:
                url = normalize_blog_url(raw_url)
            except ValueError:
                stats.invalid_urls += 1
                continue
            db.add_author_blog(
                author_id,
                url,
                label=label,
                relationship=relationship,
            )
            stats.links += 1
    return stats


def import_blog_links_path(
    db: Database,
    path: str | Path,
    *,
    create_missing: bool = False,
) -> BlogImportStats:
    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        return import_blog_links(db, handle, create_missing=create_missing)
