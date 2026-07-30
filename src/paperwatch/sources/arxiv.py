from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

import httpx

from .. import USER_AGENT

from ..models import clean_text, extract_arxiv_id, normalize_title, strip_doi


ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
NS = {"atom": ATOM, "arxiv": ARXIV}


class ArxivError(RuntimeError):
    pass


@dataclass(slots=True)
class ArxivEntry:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str | None
    updated: str | None
    abstract_url: str
    pdf_url: str | None
    doi: str | None = None


class ArxivClient:
    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/api/query",
        timeout_seconds: float = 30,
        title_match_threshold: float = 0.90,
        min_interval_seconds: float = 3.0,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_retries: int = 1,
    ):
        self.base_url = base_url
        self.title_match_threshold = title_match_threshold
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._sleeper = sleeper
        self._clock = clock
        self._last_request_at: float | None = None
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self.max_retries = max(0, max_retries)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _wait_for_slot(self) -> None:
        now = self._clock()
        if self._last_request_at is None:
            return
        remaining = self.min_interval_seconds - (now - self._last_request_at)
        if remaining > 0:
            self._sleeper(remaining)

    def _query(self, params: dict[str, Any]) -> list[ArxivEntry]:
        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                response = self.client.get(self.base_url, params=params)
                self._last_request_at = self._clock()
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.max_retries:
                        retry_after = response.headers.get("retry-after")
                        try:
                            delay = float(retry_after) if retry_after else float(2**attempt)
                        except ValueError:
                            delay = float(2**attempt)
                        self._sleeper(max(0.0, delay))
                        continue
                response.raise_for_status()
                break
            except httpx.TransportError as exc:
                self._last_request_at = self._clock()
                if attempt < self.max_retries:
                    self._sleeper(float(2**attempt))
                    continue
                raise ArxivError(f"arXiv request failed: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                raise ArxivError(f"arXiv request failed: {exc}") from exc
        assert response is not None
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise ArxivError("arXiv returned invalid Atom XML") from exc
        return [self._parse_entry(entry) for entry in root.findall("atom:entry", NS)]

    @staticmethod
    def _text(entry: ET.Element, path: str) -> str | None:
        node = entry.find(path, NS)
        return clean_text(node.text if node is not None else None)

    @classmethod
    def _parse_entry(cls, entry: ET.Element) -> ArxivEntry:
        identifier_url = cls._text(entry, "atom:id")
        arxiv_id = extract_arxiv_id(identifier_url)
        title = cls._text(entry, "atom:title")
        abstract = cls._text(entry, "atom:summary")
        if not arxiv_id or not title or not abstract or not identifier_url:
            raise ArxivError("arXiv entry is missing required metadata")

        authors = [
            name
            for node in entry.findall("atom:author/atom:name", NS)
            if (name := clean_text(node.text))
        ]
        pdf_url = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break
        return ArxivEntry(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=cls._text(entry, "atom:published"),
            updated=cls._text(entry, "atom:updated"),
            abstract_url=identifier_url,
            pdf_url=pdf_url,
            doi=strip_doi(cls._text(entry, "arxiv:doi")),
        )

    def by_id(self, arxiv_id: str) -> ArxivEntry | None:
        expected = extract_arxiv_id(f"arxiv:{arxiv_id}")
        if not expected:
            raise ArxivError("Invalid arXiv id")
        entries = self._query({"id_list": expected, "max_results": 1})
        if not entries:
            return None
        return entries[0] if entries[0].arxiv_id.casefold() == expected.casefold() else None

    @staticmethod
    def _quoted_title(title: str) -> str:
        return re.sub(r"[\"\\]", " ", title).strip()

    @staticmethod
    def _author_matches(first_author: str | None, authors: list[str]) -> bool:
        if not first_author:
            return True
        expected = normalize_title(first_author).split()
        if not expected:
            return True
        surname = expected[-1]
        return any(surname in normalize_title(author).split() for author in authors)

    def by_title(self, title: str, first_author: str | None = None) -> ArxivEntry | None:
        query_title = clean_text(title)
        if not query_title:
            raise ArxivError("Paper title cannot be empty")
        entries = self._query(
            {
                "search_query": f'ti:"{self._quoted_title(query_title)}"',
                "start": 0,
                "max_results": 5,
                "sortBy": "lastUpdatedDate",
                "sortOrder": "descending",
            }
        )
        target = normalize_title(query_title)
        best: tuple[float, ArxivEntry] | None = None
        for entry in entries:
            if not self._author_matches(first_author, entry.authors):
                continue
            score = SequenceMatcher(None, target, normalize_title(entry.title)).ratio()
            if best is None or score > best[0]:
                best = (score, entry)
        if best and best[0] >= self.title_match_threshold:
            return best[1]
        return None
