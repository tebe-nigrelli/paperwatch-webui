from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from . import USER_AGENT

from .analysis import analyze_text
from .chunking import analyze_chunked_text, direct_prompt_fit
from .config import AppConfig
from .db import Database
from .llm import LLMClient
from .locking import run_lock
from .models import AnalysisResult, PaperRecord
from .pdfs import download_pdf, extract_pdf_text, safe_pdf_name
from .sources.arxiv import ArxivClient
from .sources.openalex import OpenAlexClient


@dataclass(slots=True)
class RunStats:
    authors_checked: int = 0
    papers_found: int = 0
    papers_analyzed: int = 0
    pdfs_downloaded: int = 0
    errors_count: int = 0
    run_id: int | None = None
    status: str = "running"

    def as_dict(self) -> dict[str, int]:
        return {
            "authors_checked": self.authors_checked,
            "papers_found": self.papers_found,
            "papers_analyzed": self.papers_analyzed,
            "pdfs_downloaded": self.pdfs_downloaded,
            "errors_count": self.errors_count,
        }


class PaperWatch:
    def __init__(
        self,
        config: AppConfig,
        *,
        db: Database,
        openalex: OpenAlexClient,
        arxiv: ArxivClient | None,
        llm: LLMClient,
        pdf_client: httpx.Client | None = None,
    ):
        self.config = config
        self.db = db
        self.openalex = openalex
        self.arxiv = arxiv
        self.llm = llm
        self._owns_pdf_client = pdf_client is None
        self._llm_available = llm.provider != "none"
        self._llm_preflight_failed = False
        self.pdf_client = pdf_client or httpx.Client(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    def close(self) -> None:
        if self._owns_pdf_client:
            self.pdf_client.close()

    def __enter__(self) -> PaperWatch:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _since(self, author_row: Any, today: date) -> date:
        last_checked = author_row["last_checked_at"]
        if last_checked:
            base = date.fromisoformat(last_checked[:10])
            return base - timedelta(days=self.config.fetch.overlap_days)
        return today - timedelta(days=self.config.fetch.initial_lookback_days)

    def _record_error(
        self,
        stats: RunStats,
        *,
        stage: str,
        exc: Exception,
        author_id: str | None = None,
        paper: PaperRecord | None = None,
    ) -> None:
        stats.errors_count += 1
        assert stats.run_id is not None
        self.db.record_error(
            stats.run_id,
            stage=stage,
            message=str(exc),
            author_openalex_id=author_id,
            paper_identity_key=paper.identity_key if paper else None,
        )

    def _enrich_from_arxiv(self, paper: PaperRecord) -> PaperRecord:
        if not self.arxiv or (paper.abstract and paper.pdf_url):
            return paper
        first_author = paper.authors[0].name if paper.authors else None
        entry = self.arxiv.by_id(paper.arxiv_id) if paper.arxiv_id else None
        if entry is None:
            entry = self.arxiv.by_title(paper.title, first_author)
        if entry is None:
            return paper

        paper.arxiv_id = paper.arxiv_id or entry.arxiv_id
        paper.abstract = paper.abstract or entry.abstract
        paper.abstract_source = paper.abstract_source or "arxiv"
        paper.pdf_url = paper.pdf_url or entry.pdf_url
        paper.pdf_source = paper.pdf_source or ("arxiv" if entry.pdf_url else None)
        paper.publication_url = paper.publication_url or entry.abstract_url
        paper.updated_date = paper.updated_date or entry.updated
        paper.doi = paper.doi or entry.doi
        paper.metadata = {**paper.metadata, "arxiv_match": asdict(entry)}
        return paper

    def _analyze(
        self,
        stats: RunStats,
        *,
        paper_id: int,
        paper: PaperRecord,
        text: str,
        scope: str,
    ) -> AnalysisResult | None:
        if self.llm.provider == "none" or not self._llm_available:
            return None
        direct_fits, prompt_chars, prompt_budget = direct_prompt_fit(
            prompt_path=self.config.paths.prompt,
            max_input_chars=self.config.llm.max_input_chars,
            paper=paper,
            text=text,
            scope=scope,
            model_name=self.llm.model,
            context_window_tokens=self.config.llm.context_window_tokens,
        )
        if scope == "fulltext" and not direct_fits:
            chunked = analyze_chunked_text(
                db=self.db,
                prompt_path=self.config.paths.prompt,
                llm_config=self.config.llm,
                llm=self.llm,
                paper_id=paper_id,
                paper=paper,
                text=text,
                context_window_tokens=self.config.llm.context_window_tokens,
            )
            outcome = chunked.outcome
        else:
            if not direct_fits:
                detail = (
                    f"The prompt needs {prompt_chars} characters but the configured context safely "
                    f"allows about {prompt_budget}; increase llm.context_window_tokens"
                    if prompt_chars
                    else "The source exceeds llm.max_input_chars"
                )
                raise ValueError(detail)
            outcome = analyze_text(
                db=self.db,
                prompt_path=self.config.paths.prompt,
                llm_config=self.config.llm,
                llm=self.llm,
                paper_id=paper_id,
                paper=paper,
                text=text,
                scope=scope,
            )
        if outcome.created:
            stats.papers_analyzed += 1
        return outcome.result

    def _process_pdf(
        self,
        stats: RunStats,
        *,
        paper_id: int,
        paper: PaperRecord,
    ) -> None:
        if not paper.pdf_url or not self.config.fetch.download_pdfs:
            return

        destination = self.config.paths.pdf_dir / safe_pdf_name(paper.identity_key)
        downloaded = download_pdf(
            paper.pdf_url,
            destination,
            max_bytes=self.config.fetch.max_pdf_bytes,
            client=self.pdf_client,
        )
        if downloaded:
            stats.pdfs_downloaded += 1
        self.db.set_local_pdf(paper_id, str(destination))

        if self.config.llm.analyze_pdfs and self.llm.provider != "none":
            text = extract_pdf_text(destination, max_chars=self.config.fetch.max_pdf_chars)
            self._analyze(stats, paper_id=paper_id, paper=paper, text=text, scope="fulltext")

    def _process_paper(self, stats: RunStats, author_id: str, paper: PaperRecord) -> None:
        try:
            paper = self._enrich_from_arxiv(paper)
        except Exception as exc:
            self._record_error(
                stats,
                stage="arxiv_enrichment",
                exc=exc,
                author_id=author_id,
                paper=paper,
            )

        paper_id, _ = self.db.upsert_paper(paper)
        if paper.abstract:
            try:
                self._analyze(
                    stats,
                    paper_id=paper_id,
                    paper=paper,
                    text=paper.abstract,
                    scope="abstract",
                )
            except Exception as exc:
                self._record_error(
                    stats,
                    stage="abstract_analysis",
                    exc=exc,
                    author_id=author_id,
                    paper=paper,
                )

        try:
            self._process_pdf(
                stats,
                paper_id=paper_id,
                paper=paper,
            )
        except Exception as exc:
            self._record_error(
                stats,
                stage="pdf_processing",
                exc=exc,
                author_id=author_id,
                paper=paper,
            )

    @staticmethod
    def _run_identity(paper: PaperRecord) -> str:
        return paper.openalex_id or paper.doi or paper.arxiv_id or paper.identity_key

    def run(self, *, today: date | None = None) -> RunStats:
        with run_lock(self.config.paths.database):
            return self._run_locked(today=today)

    def _run_locked(self, *, today: date | None = None) -> RunStats:
        stats = RunStats()
        stats.run_id = self.db.start_run()
        current_day = today or datetime.now(timezone.utc).date()
        seen_papers: set[str] = set()
        self._llm_available = self.llm.provider != "none"
        self._llm_preflight_failed = False

        try:
            if self._llm_available:
                try:
                    check_ready = getattr(self.llm, "check_ready", None)
                    if check_ready is not None:
                        check_ready()
                except Exception as exc:
                    self._llm_available = False
                    self._llm_preflight_failed = True
                    self._record_error(stats, stage="llm_preflight", exc=exc)

            for author in self.db.list_authors(enabled_only=True):
                author_id = author["openalex_id"]
                try:
                    papers = self.openalex.recent_works(
                        author_id,
                        since=self._since(author, current_day).isoformat(),
                        max_results=self.config.fetch.max_papers_per_author,
                    )
                except Exception as exc:
                    self._record_error(
                        stats, stage="paper_discovery", exc=exc, author_id=author_id
                    )
                    continue

                stats.authors_checked += 1
                for paper in papers:
                    run_identity = self._run_identity(paper)
                    if run_identity in seen_papers:
                        continue
                    seen_papers.add(run_identity)
                    stats.papers_found += 1
                    try:
                        self._process_paper(stats, author_id, paper)
                    except Exception as exc:
                        self._record_error(
                            stats,
                            stage="paper_processing",
                            exc=exc,
                            author_id=author_id,
                            paper=paper,
                        )
                # If the configured model is unavailable, leave the initial-fetch
                # cursor untouched so a later run can analyze the entire lookback
                # window after the model is installed.
                if not self._llm_preflight_failed:
                    self.db.mark_author_checked(author_id)

            stats.status = "partial" if stats.errors_count else "success"
            self.db.finish_run(stats.run_id, stats.status, stats.as_dict())
            return stats
        except Exception as exc:
            stats.status = "failed"
            self.db.finish_run(
                stats.run_id, "failed", stats.as_dict(), error_text=str(exc)
            )
            raise
