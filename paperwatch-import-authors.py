#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.models import AuthorCandidate
from paperwatch.sources.openalex import OpenAlexClient, OpenAlexError


STOPWORDS = {
    "and", "ai", "center", "centre", "company", "corporation", "department",
    "for", "foundation", "group", "inc", "institute", "lab", "laboratory",
    "labs", "of", "research", "school", "the", "university",
}
NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class Resolution:
    csv_order: str
    requested_name: str
    organization: str
    status: str
    selected: AuthorCandidate | None
    name_similarity: float
    organization_overlap: int
    alternatives: str
    error: str = ""


def ascii_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return NON_ALNUM.sub(" ", without_marks.casefold()).strip()


def organization_tokens(value: str) -> set[str]:
    return {
        token
        for token in ascii_key(value).split()
        if len(token) > 2 and token not in STOPWORDS
    }


def name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, ascii_key(left), ascii_key(right)).ratio()


def candidate_rank(candidate: AuthorCandidate, requested_org: str) -> tuple[int, int, int]:
    overlap = len(organization_tokens(requested_org) & organization_tokens(candidate.institution or ""))
    return overlap, int(math.log10(max(1, candidate.cited_by_count)) * 100), candidate.works_count


def format_candidate(candidate: AuthorCandidate) -> str:
    return (
        f"{candidate.name} [{candidate.openalex_id}]"
        f" ({candidate.institution or 'unknown institution'};"
        f" works={candidate.works_count}; citations={candidate.cited_by_count})"
    )


def resolve_author(
    client: OpenAlexClient,
    *,
    csv_order: str,
    requested_name: str,
    organization: str,
    limit: int,
    choose_best: bool,
) -> Resolution:
    try:
        candidates = client.search_authors(requested_name, limit=limit)
    except Exception as exc:
        return Resolution(
            csv_order, requested_name, organization, "error", None, 0.0, 0, "", str(exc)
        )

    if not candidates:
        return Resolution(csv_order, requested_name, organization, "not_found", None, 0.0, 0, "")

    requested_key = ascii_key(requested_name)
    exact = [candidate for candidate in candidates if ascii_key(candidate.name) == requested_key]
    pool = exact
    match_kind = "exact"

    if not pool:
        fuzzy = [
            candidate
            for candidate in candidates
            if name_similarity(requested_name, candidate.name) >= 0.94
        ]
        pool = fuzzy
        match_kind = "fuzzy"

    if not pool:
        alternatives = " | ".join(format_candidate(item) for item in candidates[:5])
        return Resolution(
            csv_order, requested_name, organization, "not_found", None, 0.0, 0, alternatives
        )

    ranked = sorted(pool, key=lambda item: candidate_rank(item, organization), reverse=True)
    selected = ranked[0]
    selected_rank = candidate_rank(selected, organization)
    similarity = name_similarity(requested_name, selected.name)
    alternatives = " | ".join(format_candidate(item) for item in ranked[1:5])

    if len(ranked) == 1:
        status = match_kind
    else:
        second_rank = candidate_rank(ranked[1], organization)
        organization_decides = selected_rank[0] > second_rank[0]
        strong_fuzzy = match_kind == "fuzzy" and similarity >= 0.98 and selected_rank[0] > 0
        if organization_decides or strong_fuzzy:
            status = f"{match_kind}_ranked"
        elif choose_best:
            status = f"{match_kind}_forced"
        else:
            return Resolution(
                csv_order,
                requested_name,
                organization,
                "ambiguous",
                None,
                similarity,
                selected_rank[0],
                " | ".join(format_candidate(item) for item in ranked[:5]),
            )

    return Resolution(
        csv_order,
        requested_name,
        organization,
        status,
        selected,
        similarity,
        selected_rank[0],
        alternatives,
    )


def read_rows(path: Path, name_column: str, organization_column: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or name_column not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain the column {name_column!r}; found {reader.fieldnames or []}"
            )
        rows = []
        seen: set[str] = set()
        for row in reader:
            name = (row.get(name_column) or "").strip()
            if not name:
                continue
            key = ascii_key(name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "order": (row.get("Curated order") or "").strip(),
                    "name": name,
                    "organization": (row.get(organization_column) or "").strip(),
                }
            )
    return rows


def write_report(path: Path, resolutions: Iterable[Resolution]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "curated_order",
                "requested_name",
                "requested_organization",
                "status",
                "openalex_id",
                "matched_name",
                "matched_institution",
                "name_similarity",
                "organization_overlap",
                "works_count",
                "cited_by_count",
                "alternatives",
                "error",
            ],
        )
        writer.writeheader()
        for item in resolutions:
            selected = item.selected
            writer.writerow(
                {
                    "curated_order": item.csv_order,
                    "requested_name": item.requested_name,
                    "requested_organization": item.organization,
                    "status": item.status,
                    "openalex_id": selected.openalex_id if selected else "",
                    "matched_name": selected.name if selected else "",
                    "matched_institution": selected.institution if selected else "",
                    "name_similarity": f"{item.name_similarity:.3f}",
                    "organization_overlap": item.organization_overlap,
                    "works_count": selected.works_count if selected else "",
                    "cited_by_count": selected.cited_by_count if selected else "",
                    "alternatives": item.alternatives,
                    "error": item.error,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve researcher names through OpenAlex and add them to Paperwatch."
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--name-column", default="Researcher")
    parser.add_argument("--organization-column", default="Current lab / organization")
    parser.add_argument("--limit", type=int, default=25, help="OpenAlex candidates per name")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paperwatch-author-import-report.csv"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write resolved authors to the database; otherwise only generate the report.",
    )
    parser.add_argument(
        "--choose-best",
        action="store_true",
        help="For duplicate exact names, select the highest-ranked candidate instead of stopping.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Import resolved authors even if some rows are unresolved.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 1:
        print("--limit must be positive", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        rows = read_rows(args.csv_file, args.name_column, args.organization_column)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    client = OpenAlexClient(
        api_key=config.openalex.api_key,
        base_url=config.openalex.base_url,
        timeout_seconds=config.openalex.timeout_seconds,
    )
    try:
        resolutions = [
            resolve_author(
                client,
                csv_order=row["order"],
                requested_name=row["name"],
                organization=row["organization"],
                limit=args.limit,
                choose_best=args.choose_best,
            )
            for row in rows
        ]
    finally:
        client.close()

    write_report(args.report, resolutions)
    resolved = [item for item in resolutions if item.selected is not None]
    unresolved = [item for item in resolutions if item.selected is None]

    print(
        f"Resolved {len(resolved)}/{len(resolutions)} authors; "
        f"report: {args.report}"
    )
    for item in unresolved:
        detail = item.error or item.alternatives or "no matching OpenAlex author"
        print(f"UNRESOLVED: {item.requested_name}: {item.status}: {detail}", file=sys.stderr)

    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the report.")
        return 0 if not unresolved else 2

    if unresolved and not args.allow_partial:
        print(
            "No database changes made. Resolve the ambiguous rows or add --allow-partial.",
            file=sys.stderr,
        )
        return 2

    db = Database(config.paths.database)
    imported = 0
    try:
        for item in resolved:
            assert item.selected is not None
            db.add_author(item.selected)
            imported += 1
    finally:
        db.close()

    print(f"Imported or restored {imported} authors in {config.paths.database}")
    return 0 if not unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
