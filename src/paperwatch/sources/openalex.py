from __future__ import annotations

import time
from typing import Any, Callable

import httpx

from .. import USER_AGENT

from ..models import (
    AuthorCandidate,
    PaperAuthor,
    PaperRecord,
    clean_text,
    extract_arxiv_id,
    parse_openalex_topics,
    reconstruct_abstract,
    short_openalex_id,
    strip_doi,
)


class OpenAlexError(RuntimeError):
    pass


class OpenAlexClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openalex.org",
        timeout_seconds: float = 30,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self._sleeper = sleeper
        self.max_retries = max(0, max_retries)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            raw = response.headers.get("retry-after")
            if raw:
                try:
                    return max(0.0, float(raw))
                except ValueError:
                    pass
        return float(2**attempt)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise OpenAlexError(
                "OpenAlex API key missing. Set OPENALEX_API_KEY or openalex.api_key."
            )
        full_params = {**params, "api_key": self.api_key}
        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(f"{self.base_url}{path}", params=full_params)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.max_retries:
                        self._sleeper(self._retry_delay(response, attempt))
                        continue
                response.raise_for_status()
                break
            except httpx.TransportError as exc:
                if attempt < self.max_retries:
                    self._sleeper(self._retry_delay(None, attempt))
                    continue
                raise OpenAlexError(f"OpenAlex request failed: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                raise OpenAlexError(f"OpenAlex request failed: {exc}") from exc
        assert response is not None
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenAlexError("OpenAlex returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise OpenAlexError("OpenAlex returned an unexpected payload")
        return payload

    @staticmethod
    def _institution(item: dict[str, Any]) -> str | None:
        institutions = item.get("last_known_institutions") or []
        if institutions and isinstance(institutions[0], dict):
            return clean_text(institutions[0].get("display_name"))
        affiliations = item.get("affiliations") or []
        if affiliations and isinstance(affiliations[0], dict):
            institution = affiliations[0].get("institution") or {}
            if isinstance(institution, dict):
                return clean_text(institution.get("display_name"))
        return None

    @classmethod
    def parse_author(cls, item: dict[str, Any]) -> AuthorCandidate:
        identifier = short_openalex_id(item.get("id"))
        name = clean_text(item.get("display_name"))
        if not identifier or not name:
            raise OpenAlexError("Author record is missing id or name")
        return AuthorCandidate(
            openalex_id=identifier,
            name=name,
            orcid=item.get("orcid"),
            institution=cls._institution(item),
            works_count=int(item.get("works_count") or 0),
            cited_by_count=int(item.get("cited_by_count") or 0),
            metadata=item,
        )

    def search_authors(self, name: str, limit: int = 10) -> list[AuthorCandidate]:
        query = clean_text(name)
        if not query:
            raise OpenAlexError("Author search name cannot be empty")
        if limit < 1:
            raise OpenAlexError("Author search limit must be positive")
        payload = self._get(
            "/authors",
            {
                "search": query,
                "per_page": min(limit, 100),
            },
        )
        results = payload.get("results", [])
        if results is None:
            results = []
        if not isinstance(results, list):
            raise OpenAlexError("OpenAlex author results are malformed")
        return [self.parse_author(item) for item in results]

    def get_author(self, author_id: str) -> AuthorCandidate:
        identifier = short_openalex_id(author_id)
        if not identifier:
            raise OpenAlexError("Invalid OpenAlex author id")
        return self.parse_author(self._get(f"/authors/{identifier}", {}))

    def search_works(self, title: str, limit: int = 10) -> list[PaperRecord]:
        query = clean_text(title)
        if not query:
            raise OpenAlexError("Paper title cannot be empty")
        if limit < 1:
            raise OpenAlexError("Paper search limit must be positive")
        payload = self._get(
            "/works",
            {
                "search": query,
                "per_page": min(limit, 100),
            },
        )
        results = payload.get("results", [])
        if results is None:
            results = []
        if not isinstance(results, list):
            raise OpenAlexError("OpenAlex work results are malformed")
        papers: list[PaperRecord] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                papers.append(self.parse_work(item))
            except OpenAlexError as exc:
                if str(exc) != "Work record is missing title":
                    raise
        return papers

    def get_work(self, work_id: str) -> PaperRecord:
        identifier = short_openalex_id(clean_text(work_id))
        if not identifier or not identifier.startswith("W"):
            raise OpenAlexError("Invalid OpenAlex work id")
        return self.parse_work(self._get(f"/works/{identifier}", {}))

    @staticmethod
    def _location_urls(work: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("primary_location", "best_oa_location"):
            location = work.get(key) or {}
            if not isinstance(location, dict):
                continue
            for field in ("landing_page_url", "pdf_url"):
                value = location.get(field)
                if isinstance(value, str) and value:
                    values.append(value)
        for location in work.get("locations") or []:
            if not isinstance(location, dict):
                continue
            for field in ("landing_page_url", "pdf_url"):
                value = location.get(field)
                if isinstance(value, str) and value:
                    values.append(value)
        return values

    @staticmethod
    def _best_pdf(work: dict[str, Any]) -> tuple[str | None, str | None]:
        best = work.get("best_oa_location") or {}
        if isinstance(best, dict) and best.get("pdf_url"):
            return best["pdf_url"], "openalex_best_oa"
        primary = work.get("primary_location") or {}
        if isinstance(primary, dict) and primary.get("pdf_url"):
            source = ((primary.get("source") or {}).get("display_name") or "openalex_primary")
            return primary["pdf_url"], source
        for location in work.get("locations") or []:
            if isinstance(location, dict) and location.get("pdf_url"):
                source = ((location.get("source") or {}).get("display_name") or "openalex_location")
                return location["pdf_url"], source
        return None, None

    @classmethod
    def parse_work(cls, work: dict[str, Any]) -> PaperRecord:
        title = clean_text(work.get("display_name") or work.get("title"))
        if not title:
            raise OpenAlexError("Work record is missing title")

        authors: list[PaperAuthor] = []
        for authorship in work.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") or {}
            if not isinstance(author, dict):
                continue
            name = clean_text(author.get("display_name"))
            if not name:
                continue
            authors.append(
                PaperAuthor(
                    openalex_id=short_openalex_id(author.get("id")),
                    name=name,
                    position=authorship.get("author_position"),
                    is_corresponding=bool(authorship.get("is_corresponding", False)),
                )
            )

        primary = work.get("primary_location") or {}
        if not isinstance(primary, dict):
            primary = {}
        best_oa = work.get("best_oa_location") or {}
        if not isinstance(best_oa, dict):
            best_oa = {}
        source = primary.get("source") or {}
        if not isinstance(source, dict):
            source = {}
        doi = strip_doi(work.get("doi") or (work.get("ids") or {}).get("doi"))
        urls = cls._location_urls(work)
        arxiv_id = extract_arxiv_id(*urls)
        pdf_url, pdf_source = cls._best_pdf(work)
        publication_url = (
            primary.get("landing_page_url")
            or best_oa.get("landing_page_url")
            or (f"https://doi.org/{doi}" if doi else None)
            or work.get("id")
        )
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

        return PaperRecord(
            openalex_id=short_openalex_id(work.get("id")),
            doi=doi,
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            publication_date=work.get("publication_date"),
            updated_date=work.get("updated_date"),
            venue=clean_text(source.get("display_name")),
            publication_url=publication_url,
            pdf_url=pdf_url,
            abstract_source="openalex" if abstract else None,
            pdf_source=pdf_source,
            authors=authors,
            topics=parse_openalex_topics(work),
            metadata=work,
        )

    def recent_works(
        self,
        author_id: str,
        *,
        since: str,
        max_results: int = 100,
    ) -> list[PaperRecord]:
        identifier = short_openalex_id(author_id)
        if not identifier:
            raise OpenAlexError("Invalid OpenAlex author id")
        if max_results < 1:
            raise OpenAlexError("max_results must be positive")

        records: list[PaperRecord] = []
        cursor = "*"
        while len(records) < max_results:
            per_page = min(100, max_results - len(records))
            payload = self._get(
                "/works",
                {
                    "filter": f"author.id:{identifier},from_publication_date:{since}",
                    "sort": "publication_date:desc",
                    "per_page": per_page,
                    "cursor": cursor,
                },
            )
            batch = payload.get("results", [])
            if batch is None:
                batch = []
            if not isinstance(batch, list):
                raise OpenAlexError("OpenAlex work results are malformed")
            for item in batch:
                if not isinstance(item, dict):
                    continue
                try:
                    records.append(self.parse_work(item))
                except OpenAlexError as exc:
                    # OpenAlex occasionally emits sparse placeholder records. A record
                    # without a title cannot be displayed, identified, or analyzed, but
                    # it should not prevent the remaining works for the author from being
                    # discovered.
                    if str(exc) == "Work record is missing title":
                        continue
                    raise
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not batch or not cursor:
                break
        return records[:max_results]
