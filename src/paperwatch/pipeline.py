from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import re
import threading
from typing import Any

import httpx

from . import USER_AGENT

from .analysis import analyze_pdf_file, analyze_text
from .blogs import ArticleItem, BlogFetch, blog_diff, fetch_blog
from .chunking import analyze_chunked_text, direct_prompt_fit
from .config import AppConfig
from .db import Database, DeadSourceError
from .llm import (
    LLMClient, configure_context_window, configure_job_runtime, create_llm
)
from .locking import run_lock
from .job_runtime import JobCancelled, ProgressCallback, check_cancelled, publish
from .models import AnalysisResult, PaperAuthor, PaperRecord, extract_arxiv_id, normalize_title
from .pdfs import (
    PDFError,
    ResolvedPDF,
    download_pdf,
    extract_pdf_text,
    is_complete_pdf,
    resolve_pdf_url,
    safe_pdf_name,
    pdf_sha256,
)
from .sources.arxiv import ArxivClient
from .sources.openalex import OpenAlexClient


MAX_BLOG_FETCH_WORKERS = 8


@dataclass(slots=True)
class RunStats:
    authors_total: int = 0
    authors_checked: int = 0
    papers_found: int = 0
    papers_downloadable: int = 0
    papers_downloaded: int = 0
    papers_analyzed: int = 0
    pdfs_downloaded: int = 0
    blogs_total: int = 0
    blogs_checked: int = 0
    blogs_with_content: int = 0
    blog_items_available: int = 0
    blog_items_downloaded: int = 0
    blog_updates_found: int = 0
    errors_count: int = 0
    run_id: int | None = None
    status: str = "running"
    new_paper_ids: list[int] = field(default_factory=list)
    new_blog_entry_ids: list[int] = field(default_factory=list)
    _downloadable_paper_ids: set[int] = field(default_factory=set, repr=False)
    _downloaded_paper_ids: set[int] = field(default_factory=set, repr=False)

    def as_dict(self) -> dict[str, int]:
        return {
            "authors_total": self.authors_total,
            "authors_checked": self.authors_checked,
            "papers_found": self.papers_found,
            "papers_downloadable": self.papers_downloadable,
            "papers_downloaded": self.papers_downloaded,
            "papers_analyzed": self.papers_analyzed,
            "pdfs_downloaded": self.pdfs_downloaded,
            "blogs_total": self.blogs_total,
            "blogs_checked": self.blogs_checked,
            "blogs_with_content": self.blogs_with_content,
            "blog_items_available": self.blog_items_available,
            "blog_items_downloaded": self.blog_items_downloaded,
            "blog_updates_found": self.blog_updates_found,
            "errors_count": self.errors_count,
        }


class _SelectedPaperDownloadError(RuntimeError):
    def __init__(self, message: str, paper_id: int):
        super().__init__(message)
        self.paper_id = paper_id


RSS_WEBSITE_MARKER = "\n\nWebsite content:\n"
RSS_WEBSITE_HTML = re.compile(
    r'<hr><section class="rss-linked-page">.*</section>\s*$',
    re.DOTALL,
)


def _rss_with_article(
    feed_text: str,
    feed_html: str | None,
    article: ArticleItem,
) -> tuple[str, str | None, str]:
    """Combine one feed record with its primary webpage, replacing old enrichment."""
    base_text = feed_text.split(RSS_WEBSITE_MARKER, 1)[0].rstrip()
    combined_text = f"{base_text}{RSS_WEBSITE_MARKER}{article.text}"[:1_000_000]
    base_html = RSS_WEBSITE_HTML.sub("", feed_html or "").rstrip()
    article_html = article.rendered_html or ""
    combined_html = (
        f'{base_html}{"<hr>" if base_html else ""}<section class="rss-linked-page">'
        f"<h2>Website content</h2>{article_html}</section>"
        if article_html
        else (base_html or None)
    )
    return (
        combined_text,
        combined_html,
        hashlib.sha256(combined_text.encode("utf-8")).hexdigest(),
    )


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
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
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
        self._progress = progress
        self._cancel_event = cancel_event
        self._progress_current = 0
        self._progress_total = 0
        configure_job_runtime(self.llm, self._progress, self._cancel_event)

    def _start_progress(self, total: int, stage: str) -> None:
        self._progress_current = 0
        self._progress_total = max(0, total)
        publish(self._progress, "stage", stage)
        publish(self._progress, "progress", f"0/{self._progress_total}")

    def _advance_progress(self, message: str) -> None:
        self._progress_current = min(
            self._progress_total,
            self._progress_current + 1,
        )
        publish(
            self._progress,
            "progress",
            f"{self._progress_current}/{self._progress_total}",
        )
        publish(self._progress, "message", message)

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

    def _fetch_blog_batch(
        self, blogs: list[Any]
    ) -> list[tuple[Any, BlogFetch | None, Exception | None]]:
        """Fetch blog sources without touching SQLite.

        The batch can run in a worker while the main thread discovers papers and
        downloads PDFs. Results are applied on the main thread so the existing
        SQLite connection remains single-threaded.
        """
        if not blogs:
            return []
        # Bound concurrency because each source may perform several child-page
        # requests. This prevents a large watchlist from creating one thread and
        # up to MAX_ARTICLE_FETCHES outbound requests per source simultaneously.
        worker_count = min(len(blogs), MAX_BLOG_FETCH_WORKERS)
        if worker_count == 1:
            blog = blogs[0]
            check_cancelled(self._cancel_event)
            try:
                return [(blog, fetch_blog(self.pdf_client, blog), None)]
            except JobCancelled:
                raise
            except Exception as exc:
                return [(blog, None, exc)]

        results: list[tuple[Any, BlogFetch | None, Exception | None] | None] = [
            None
        ] * len(blogs)
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="paperwatch-blog-source",
        ) as executor:
            future_indexes = {
                executor.submit(fetch_blog, self.pdf_client, blog): index
                for index, blog in enumerate(blogs)
            }
            for future in as_completed(future_indexes):
                check_cancelled(self._cancel_event)
                index = future_indexes[future]
                blog = blogs[index]
                try:
                    results[index] = (blog, future.result(), None)
                except JobCancelled:
                    raise
                except Exception as exc:
                    results[index] = (blog, None, exc)
        return [result for result in results if result is not None]

    def _prepare_blog_fetch_rows(self, blogs: list[Any]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for blog in blogs:
            row = dict(blog)
            blog_id = row.get("id")
            row["_known_article_urls"] = (
                self.db.known_blog_article_urls(int(blog_id)) if blog_id is not None else ()
            )
            prepared.append(row)
        return prepared

    @staticmethod
    def _arxiv_fallback_title(
        arxiv_id: str,
        feed_title: str | None,
        article: ArticleItem | None,
    ) -> str:
        candidates = [article.title if article else None, feed_title]
        for value in candidates:
            title = str(value or "").strip()
            if title and title.casefold() not in {"arxiv", "arxiv.org", "paper"}:
                return title
        return f"arXiv {arxiv_id}"

    @staticmethod
    def _arxiv_fallback_abstract(article: ArticleItem | None) -> str | None:
        if article is None:
            return None
        lines = [
            line
            for line in article.text.splitlines()
            if line.strip()
            and not line.startswith(("Title:", "URL:", "Outbound links:", "- "))
        ]
        text = "\n".join(lines).strip()
        return text[:100_000] or None

    @staticmethod
    def _arxiv_fallback_date(article: ArticleItem | None) -> str | None:
        if article is None:
            return None
        match = re.search(
            r"\bSubmitted\s+on\s+(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})\b",
            article.text,
        )
        if match is None:
            match = re.search(
                r"\[v1\]\s+\w{3},\s+(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})\b",
                article.text,
            )
        if match is None:
            return None
        try:
            return datetime.strptime(match.group(1), "%d %b %Y").date().isoformat()
        except ValueError:
            return None

    def _import_arxiv_feed_item(
        self,
        stats: RunStats,
        url: str | None,
        *,
        feed_title: str | None = None,
        article: ArticleItem | None = None,
    ) -> int | None:
        """Import and return the paper linked by a primary arXiv RSS URL.

        The arXiv API enriches the record when enabled. A minimal paper is still
        created from the RSS/article data when the API is disabled or unavailable,
        so the RSS entry can always link to an in-app paper page.
        """
        if not url:
            return None
        try:
            parsed = httpx.URL(url)
        except (TypeError, ValueError):
            return None
        if parsed.host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            return None
        arxiv_id = extract_arxiv_id(url)
        if not arxiv_id:
            return None

        entry = None
        if self.arxiv is not None:
            try:
                entry = self.arxiv.by_id(arxiv_id)
            except JobCancelled:
                raise
            except Exception as exc:
                self._record_error(
                    stats,
                    stage="rss_arxiv_enrichment",
                    exc=RuntimeError(f"{url}: {exc}"),
                )

        if entry is not None:
            paper = PaperRecord(
                openalex_id=None,
                doi=entry.doi,
                arxiv_id=entry.arxiv_id,
                title=entry.title,
                abstract=entry.abstract,
                publication_date=entry.published,
                updated_date=entry.updated,
                venue="arXiv",
                publication_url=entry.abstract_url,
                pdf_url=entry.pdf_url,
                abstract_source="arxiv",
                pdf_source="arxiv" if entry.pdf_url else None,
                authors=[PaperAuthor(openalex_id=None, name=name) for name in entry.authors],
                metadata={"rss_arxiv_source": url},
            )
        else:
            abstract = self._arxiv_fallback_abstract(article)
            paper = PaperRecord(
                openalex_id=None,
                doi=None,
                arxiv_id=arxiv_id,
                title=self._arxiv_fallback_title(arxiv_id, feed_title, article),
                abstract=abstract,
                publication_date=self._arxiv_fallback_date(article),
                updated_date=None,
                venue="arXiv",
                publication_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                abstract_source="rss-linked-page" if abstract else "rss",
                pdf_source="arxiv",
                authors=[],
                metadata={"rss_arxiv_source": url, "rss_fallback": True},
            )

        try:
            paper_id, is_new = self.db.upsert_paper(paper, auto_analyze=bool(paper.abstract))
            if is_new:
                stats.papers_found += 1
                stats.new_paper_ids.append(paper_id)
        except DeadSourceError:
            return None
        except JobCancelled:
            raise
        except Exception as exc:
            self._record_error(
                stats,
                stage="rss_arxiv_import",
                exc=RuntimeError(f"{url}: {exc}"),
            )
            return None

        # A fallback record has a canonical PDF link, but it should not turn a
        # successful RSS import into an error merely because the optional arXiv
        # API was disabled. API-enriched records keep the existing eager download.
        if entry is not None and paper.pdf_url:
            try:
                self._process_pdf(
                    stats,
                    paper_id=paper_id,
                    paper=paper,
                    analyze=False,
                    analyze_new_downloads=False,
                )
            except JobCancelled:
                raise
            except Exception as exc:
                self._record_error(
                    stats,
                    stage="rss_arxiv_pdf",
                    exc=RuntimeError(f"{url}: {exc}"),
                    paper=paper,
                )
        return paper_id

    def _apply_blog_fetch(
        self,
        stats: RunStats,
        blog: Any,
        fetched: BlogFetch | None,
        error: Exception | None,
    ) -> None:
        blog_id = int(blog["id"])
        try:
            if error is not None:
                raise error
            assert fetched is not None
            stats.blogs_checked += 1
            available = (
                len(fetched.feed_items)
                + len(fetched.discovered_articles)
                + (1 if fetched.snapshot_text else 0)
            )
            downloaded = (
                len(fetched.feed_items)
                + len(fetched.article_items)
                + (1 if fetched.snapshot_text else 0)
            )
            if available or downloaded:
                stats.blogs_with_content += 1
            stats.blog_items_available += available
            stats.blog_items_downloaded += downloaded
            if fetched.not_modified:
                self.db.mark_blog_checked(
                    blog_id,
                    status=fetched.status_code,
                    error=None,
                    fetch_method=fetched.method,
                    feed_url=fetched.feed_url,
                    feed_discovery_checked=fetched.discovery_checked,
                    diagnostics=json.dumps(fetched.diagnostics),
                    item_stream_mode=fetched.source_mode,
                )
                return

            if fetched.method == "rss":
                initial_sync = not bool(blog["feed_initialized"])
                article_by_url = {item.url: item for item in fetched.article_items}
                paper_by_url: dict[str, int] = {}
                for item in reversed(fetched.feed_items):
                    article = article_by_url.get(item.url or "")
                    item_text = item.text
                    item_html = item.rendered_html
                    item_sha256 = item.sha256
                    if article is not None:
                        item_text, item_html, item_sha256 = _rss_with_article(
                            item.text,
                            item.rendered_html,
                            article,
                        )
                    previous = self.db.latest_blog_entry(
                        blog_id,
                        entry_kind="rss",
                        external_id=item.external_id,
                    )
                    # Import primary arXiv targets even when the RSS record is
                    # unchanged, and retain an explicit in-app paper relationship.
                    linked_paper_id = self._import_arxiv_feed_item(
                        stats,
                        item.url,
                        feed_title=item.title,
                        article=article,
                    )
                    if item.url and linked_paper_id is not None:
                        paper_by_url[item.url] = linked_paper_id
                    if previous is not None and previous["content_sha256"] == item_sha256:
                        if linked_paper_id is not None:
                            self.db.link_blog_entry_to_paper(int(previous["id"]), linked_paper_id)
                        self.db.set_blog_entry_published_at(
                            int(previous["id"]), item.published_at
                        )
                        continue
                    initial_item = previous is None and initial_sync
                    if previous is None:
                        kind, diff = "changed", ""
                    else:
                        kind, diff = blog_diff(
                            str(previous["normalized_text"]),
                            item_text,
                            # Feed entries are already scoped records. Any semantic
                            # body/title change is meaningful; webpage noise filtering
                            # is only appropriate for snapshots.
                            source_mode="feed",
                        )
                    entry_id = self.db.save_blog_entry(
                        blog_id=blog_id,
                        title=item.title,
                        content_sha256=item_sha256,
                        normalized_text=item_text,
                        diff_text=diff,
                        change_kind=kind,
                        previous_entry_id=(
                            int(previous["id"]) if previous is not None else None
                        ),
                        entry_kind="rss",
                        external_id=item.external_id,
                        source_url=item.url,
                        published_at=item.published_at,
                        rendered_html=item_html,
                        linked_paper_id=linked_paper_id,
                    )
                    if kind != "baseline" and not initial_item:
                        stats.blog_updates_found += 1
                        stats.new_blog_entry_ids.append(entry_id)
                feed_urls = {item.url for item in fetched.feed_items if item.url}
                for article in fetched.article_items:
                    # A 304 feed response has no item payload, but known article pages
                    # are still rechecked. Enrich the latest RSS record in place so an
                    # older teaser-only entry gains the webpage body without producing
                    # a false new feed update.
                    linked_paper_id = paper_by_url.get(article.url)
                    if linked_paper_id is None:
                        linked_paper_id = self._import_arxiv_feed_item(
                            stats,
                            article.url,
                            feed_title=article.title,
                            article=article,
                        )
                    rss_entry = self.db.latest_blog_entry_for_source(
                        blog_id,
                        entry_kind="rss",
                        source_url=article.url,
                    )
                    if rss_entry is None:
                        continue
                    if linked_paper_id is not None:
                        self.db.link_blog_entry_to_paper(int(rss_entry["id"]), linked_paper_id)
                    combined_text, combined_html, combined_sha256 = _rss_with_article(
                        str(rss_entry["normalized_text"] or ""),
                        str(rss_entry["rendered_html"] or "") or None,
                        article,
                    )
                    if str(rss_entry["content_sha256"]) != combined_sha256:
                        self.db.update_blog_entry_content(
                            int(rss_entry["id"]),
                            content_sha256=combined_sha256,
                            normalized_text=combined_text,
                            rendered_html=combined_html,
                            published_at=article.published_at,
                        )
                    else:
                        self.db.set_blog_entry_published_at(
                            int(rss_entry["id"]), article.published_at
                        )
                discovered = tuple(fetched.discovered_articles)
                fetched_article_urls = {item.url for item in fetched.article_items}
                if discovered or fetched.article_items:
                    self.db.sync_known_blog_articles(
                        blog_id,
                        ((item.url, item.title) for item in discovered),
                        fetched_urls=fetched_article_urls,
                    )
                for article in fetched.article_items:
                    previous_article = self.db.latest_blog_entry(
                        blog_id,
                        entry_kind="snapshot",
                        external_id=article.external_id,
                    )
                    if (
                        previous_article is not None
                        and previous_article["content_sha256"] == article.sha256
                    ):
                        self.db.set_blog_entry_published_at(
                            int(previous_article["id"]), article.published_at
                        )
                        continue
                    first_page_fetch = previous_article is None
                    if previous_article is None:
                        article_kind, article_diff = "changed", ""
                    else:
                        article_kind, article_diff = blog_diff(
                            str(previous_article["normalized_text"]),
                            article.text,
                            source_mode="article",
                        )
                    article_entry_id = self.db.save_blog_entry(
                        blog_id=blog_id,
                        title=article.title,
                        content_sha256=article.sha256,
                        normalized_text=article.text,
                        diff_text=article_diff,
                        change_kind=article_kind,
                        previous_entry_id=(
                            int(previous_article["id"])
                            if previous_article is not None
                            else None
                        ),
                        entry_kind="snapshot",
                        external_id=article.external_id,
                        source_url=article.url,
                        rendered_html=article.rendered_html,
                        published_at=article.published_at,
                    )
                    # The feed item itself is the new-content notification. The
                    # first HTML fetch is a usable website snapshot, not a
                    # second notification for the same post.
                    if article_kind != "baseline" and not first_page_fetch:
                        stats.blog_updates_found += 1
                        stats.new_blog_entry_ids.append(article_entry_id)

                self.db.mark_blog_checked(
                    blog_id,
                    status=fetched.status_code,
                    etag=fetched.etag,
                    last_modified=fetched.last_modified,
                    title=fetched.title,
                    error=None,
                    fetch_method="rss",
                    feed_url=fetched.feed_url,
                    feed_initialized=True,
                    feed_discovery_checked=fetched.discovery_checked,
                    article_initialized=(
                        True
                        if fetched.source_mode
                        in {"auto", "feed", "article_index", "publications"}
                        else None
                    ),
                    article_cursor=fetched.article_cursor_next,
                    diagnostics=json.dumps(fetched.diagnostics),
                    item_stream_mode=fetched.source_mode,
                )
                return

            assert fetched.snapshot_text is not None
            assert fetched.snapshot_sha256 is not None
            initial_article_sync = not bool(blog.get("article_initialized", 0))
            existing_article_urls = {
                str(value) for value in (blog.get("_known_article_urls", ()) or ())
            }
            discovered = tuple(fetched.discovered_articles)
            new_discoveries = [
                item for item in discovered if item.url not in existing_article_urls
            ]
            fetched_article_urls = {item.url for item in fetched.article_items}
            if discovered or fetched.article_items:
                self.db.sync_known_blog_articles(
                    blog_id,
                    ((item.url, item.title) for item in discovered),
                    fetched_urls=fetched_article_urls,
                )
            # Article-index pages are navigation, not content. Store only pages
            # that were actually fetched; publication lists retain their useful
            # discovery records for external paper targets.
            discovery_candidates = (
                discovered if fetched.source_mode == "publications" else ()
            )
            for article in discovery_candidates:
                # A successfully fetched child page is stored below as the
                # actual entry; do not also create a teaser-only discovery row.
                if article.url in fetched_article_urls:
                    continue
                normalized = "\n".join(
                    part
                    for part in (
                        f"Title: {article.title}",
                        f"URL: {article.url}",
                        article.summary,
                    )
                    if part
                )
                external_id = hashlib.sha256(
                    f"article-discovery\0{article.url}".encode("utf-8")
                ).hexdigest()
                previous_discovery = self.db.latest_blog_entry(
                    blog_id,
                    entry_kind="rss",
                    external_id=external_id,
                )
                if previous_discovery is not None:
                    continue
                linked_paper_id = self._import_arxiv_feed_item(
                    stats,
                    article.url,
                    feed_title=article.title,
                )
                entry_id = self.db.save_blog_entry(
                    blog_id=blog_id,
                    title=article.title,
                    content_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    normalized_text=normalized,
                    diff_text="",
                    change_kind="changed",
                    previous_entry_id=None,
                    entry_kind="rss",
                    external_id=external_id,
                    source_url=article.url,
                    published_at=article.published_at,
                    linked_paper_id=linked_paper_id,
                )
                if not initial_article_sync and article.url not in existing_article_urls:
                    stats.blog_updates_found += 1
                    stats.new_blog_entry_ids.append(entry_id)

            for article in fetched.article_items:
                linked_paper_id = self._import_arxiv_feed_item(
                    stats,
                    article.url,
                    feed_title=article.title,
                    article=article,
                )
                previous_article = self.db.latest_blog_entry(
                    blog_id,
                    entry_kind="rss",
                    external_id=article.external_id,
                )
                reuse_source_entry = previous_article is None
                if reuse_source_entry:
                    previous_article = self.db.latest_blog_entry_for_source(
                        blog_id,
                        entry_kind="rss",
                        source_url=article.url,
                    )
                if previous_article is not None and linked_paper_id is not None:
                    self.db.link_blog_entry_to_paper(
                        int(previous_article["id"]), linked_paper_id
                    )
                if previous_article is not None and previous_article["content_sha256"] == article.sha256:
                    self.db.set_blog_entry_published_at(
                        int(previous_article["id"]), article.published_at
                    )
                    continue
                if reuse_source_entry and previous_article is not None:
                    self.db.update_blog_entry_content(
                        int(previous_article["id"]),
                        content_sha256=article.sha256,
                        normalized_text=article.text,
                        rendered_html=article.rendered_html,
                        published_at=article.published_at,
                    )
                    continue
                initial_article = previous_article is None and (
                    initial_article_sync or article.url in existing_article_urls
                )
                if previous_article is None:
                    article_kind, article_diff = "changed", ""
                else:
                    article_kind, article_diff = blog_diff(
                        str(previous_article["normalized_text"]),
                        article.text,
                        source_mode="article",
                    )
                article_entry_id = self.db.save_blog_entry(
                    blog_id=blog_id,
                    title=article.title,
                    content_sha256=article.sha256,
                    normalized_text=article.text,
                    diff_text=article_diff,
                    change_kind=article_kind,
                    previous_entry_id=(
                        int(previous_article["id"])
                        if previous_article is not None
                        else None
                    ),
                    entry_kind="rss",
                    external_id=article.external_id,
                    source_url=article.url,
                    rendered_html=article.rendered_html,
                    published_at=article.published_at,
                    linked_paper_id=linked_paper_id,
                )
                if article_kind != "baseline" and not initial_article:
                    stats.blog_updates_found += 1
                    stats.new_blog_entry_ids.append(article_entry_id)

            if fetched.source_mode in {"article_index", "publications"}:
                self.db.suppress_blog_source_page_entries(blog_id)
                self.db.mark_blog_checked(
                    blog_id,
                    status=fetched.status_code,
                    etag=fetched.etag,
                    last_modified=fetched.last_modified,
                    title=fetched.title,
                    error=None,
                    fetch_method="snapshot",
                    feed_url=fetched.feed_url,
                    feed_discovery_checked=fetched.discovery_checked,
                    article_initialized=True,
                    article_cursor=fetched.article_cursor_next,
                    diagnostics=json.dumps(fetched.diagnostics),
                    item_stream_mode=fetched.source_mode,
                )
                return

            previous = self.db.latest_blog_entry(
                blog_id, entry_kind="snapshot", external_id="source-page"
            )
            if (
                previous is not None
                and previous["content_sha256"] == fetched.snapshot_sha256
            ):
                self.db.set_blog_entry_published_at(
                    int(previous["id"]), fetched.snapshot_published_at
                )
                self.db.mark_blog_checked(
                    blog_id,
                    status=fetched.status_code,
                    etag=fetched.etag,
                    last_modified=fetched.last_modified,
                    title=fetched.title,
                    error=None,
                    fetch_method="snapshot",
                    feed_url=fetched.feed_url,
                    feed_discovery_checked=fetched.discovery_checked,
                    article_initialized=(
                        True
                        if fetched.source_mode in {"article_index", "publications"}
                        else None
                    ),
                    article_cursor=fetched.article_cursor_next,
                    diagnostics=json.dumps(fetched.diagnostics),
                    item_stream_mode=fetched.source_mode,
                )
                return
            kind, diff = blog_diff(
                str(previous["normalized_text"]) if previous is not None else None,
                fetched.snapshot_text,
                source_mode=fetched.source_mode,
            )
            if new_discoveries and fetched.source_mode in {"article_index", "publications"}:
                kind, diff = "baseline", ""
            entry_id = self.db.save_blog_entry(
                blog_id=blog_id,
                title=fetched.title,
                content_sha256=fetched.snapshot_sha256,
                normalized_text=fetched.snapshot_text,
                diff_text=diff,
                change_kind=kind,
                previous_entry_id=int(previous["id"]) if previous is not None else None,
                entry_kind="snapshot",
                external_id="source-page",
                source_url=str(blog["url"]),
                rendered_html=fetched.snapshot_rendered_html,
                published_at=fetched.snapshot_published_at,
            )
            if kind != "baseline":
                stats.blog_updates_found += 1
                stats.new_blog_entry_ids.append(entry_id)
            self.db.mark_blog_checked(
                blog_id,
                status=fetched.status_code,
                etag=fetched.etag,
                last_modified=fetched.last_modified,
                title=fetched.title,
                error=None,
                fetch_method="snapshot",
                feed_url=fetched.feed_url,
                feed_discovery_checked=fetched.discovery_checked,
                article_initialized=(
                    True
                    if fetched.source_mode in {"article_index", "publications"}
                    else None
                ),
                article_cursor=fetched.article_cursor_next,
                diagnostics=json.dumps(fetched.diagnostics),
                item_stream_mode=fetched.source_mode,
            )
        except JobCancelled:
            raise
        except Exception as exc:
            self.db.mark_blog_checked(blog_id, status=None, error=str(exc))
            self._record_error(
                stats,
                stage="blog_fetch",
                exc=RuntimeError(f"{blog['url']}: {exc}"),
            )
        finally:
            self._advance_progress(f"Checked blog {blog['url']}")

    def _apply_blog_batch(
        self,
        stats: RunStats,
        results: list[tuple[Any, BlogFetch | None, Exception | None]],
    ) -> None:
        for blog, fetched, error in results:
            check_cancelled(self._cancel_event)
            self._apply_blog_fetch(stats, blog, fetched, error)

    def _fetch_blogs(self, stats: RunStats, blogs: list[Any] | None = None) -> None:
        selected = blogs if blogs is not None else self.db.list_enabled_blogs()
        prepared = self._prepare_blog_fetch_rows(selected)
        self._apply_blog_batch(stats, self._fetch_blog_batch(prepared))

    def _start_blog_fetches(
        self, blogs: list[Any]
    ) -> tuple[
        ThreadPoolExecutor | None,
        Future[list[tuple[Any, BlogFetch | None, Exception | None]]] | None,
    ]:
        if not blogs:
            return None, None
        # sqlite3.Row instances belong to the main database connection. Copy the
        # values before handing work to a background thread.
        safe_blogs = self._prepare_blog_fetch_rows(blogs)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paperwatch-blogs")
        return executor, executor.submit(self._fetch_blog_batch, safe_blogs)

    @staticmethod
    def _stop_blog_fetches(executor: ThreadPoolExecutor | None) -> None:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

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
        strategy: str = "auto",
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
        use_chunked = scope == "fulltext" and (
            strategy == "chunked" or (strategy == "auto" and not direct_fits)
        )
        # Lightweight/testing clients may implement only the direct structured
        # analysis API. Preserve that compatibility when the complete prompt fits;
        # real chunk-capable clients still use chunk + merge by default.
        if use_chunked and direct_fits and not hasattr(self.llm, "generate_json"):
            use_chunked = False
        if use_chunked:
            chunked = analyze_chunked_text(
                db=self.db,
                prompt_path=self.config.paths.prompt,
                llm_config=self.config.llm,
                llm=self.llm,
                paper_id=paper_id,
                paper=paper,
                text=text,
                context_window_tokens=self.config.llm.context_window_tokens,
                progress=self._progress,
                cancel_event=self._cancel_event,
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

    @staticmethod
    def _paper_pdf_candidates(paper: PaperRecord) -> list[str]:
        candidates: list[str] = []

        def add(value: Any) -> None:
            url = str(value or "").strip()
            if url.startswith(("http://", "https://")) and url not in candidates:
                candidates.append(url)

        add(paper.pdf_url)
        add(paper.publication_url)
        metadata = paper.metadata if isinstance(paper.metadata, dict) else {}
        for key in ("best_oa_location", "primary_location"):
            location = metadata.get(key)
            if not isinstance(location, dict):
                continue
            add(location.get("pdf_url"))
            add(location.get("landing_page_url"))
        locations = metadata.get("locations")
        if isinstance(locations, list):
            for location in locations:
                if not isinstance(location, dict):
                    continue
                add(location.get("pdf_url"))
                add(location.get("landing_page_url"))
        return candidates

    def _resolve_paper_pdf(self, paper: PaperRecord) -> ResolvedPDF | None:
        return resolve_pdf_url(
            title=paper.title,
            doi=paper.doi,
            candidates=self._paper_pdf_candidates(paper),
            client=self.pdf_client,
            cancel_event=self._cancel_event,
        )

    def _store_resolved_pdf(
        self,
        *,
        paper_id: int,
        paper: PaperRecord,
        resolved: ResolvedPDF,
    ) -> None:
        paper.pdf_url = resolved.url
        paper.pdf_source = resolved.source
        self.db.set_paper_pdf_source(paper_id, resolved.url, resolved.source)

    def _process_pdf(
        self,
        stats: RunStats,
        *,
        paper_id: int,
        paper: PaperRecord,
        analyze: bool,
        analyze_new_downloads: bool,
    ) -> None:
        # Fetches are archival: once a paper exposes a valid PDF, keep a local copy.
        # Landing pages and Scholar result links are resolved only when the direct
        # OpenAlex/arXiv URL is absent or fails PDF validation.
        resolution_attempted = False
        if not paper.pdf_url:
            resolved = self._resolve_paper_pdf(paper)
            resolution_attempted = True
            if resolved is None:
                return
            self._store_resolved_pdf(paper_id=paper_id, paper=paper, resolved=resolved)

        stats._downloadable_paper_ids.add(paper_id)
        stats.papers_downloadable = len(stats._downloadable_paper_ids)

        destination = self.config.paths.pdf_dir / safe_pdf_name(paper.identity_key)
        try:
            downloaded = download_pdf(
                paper.pdf_url,
                destination,
                max_bytes=self.config.fetch.max_pdf_bytes,
                client=self.pdf_client,
                cancel_event=self._cancel_event,
            )
        except PDFError as direct_error:
            if resolution_attempted:
                raise
            resolved = self._resolve_paper_pdf(paper)
            if resolved is None or resolved.url == paper.pdf_url:
                raise direct_error
            self._store_resolved_pdf(paper_id=paper_id, paper=paper, resolved=resolved)
            downloaded = download_pdf(
                paper.pdf_url,
                destination,
                max_bytes=self.config.fetch.max_pdf_bytes,
                client=self.pdf_client,
                cancel_event=self._cancel_event,
            )
        if downloaded:
            stats.pdfs_downloaded += 1
        self.db.set_local_pdf(paper_id, str(destination), pdf_sha256(destination))
        stats._downloaded_paper_ids.add(paper_id)
        stats.papers_downloaded = len(stats._downloaded_paper_ids)

        should_analyze = analyze or (downloaded and analyze_new_downloads)
        if (
            not should_analyze
            or not self.config.llm.analyze_pdfs
            or self.llm.provider == "none"
            or not self._llm_available
        ):
            return
        strategy = self.config.llm.pdf_analysis_strategy
        if self.llm.provider in {"codex", "claude-code"}:
            outcome = analyze_pdf_file(
                db=self.db,
                prompt_path=self.config.paths.prompt,
                llm_config=self.config.llm,
                llm=self.llm,
                paper_id=paper_id,
                paper=paper,
                pdf_path=destination,
            )
            if outcome.created:
                stats.papers_analyzed += 1
            return
        text = extract_pdf_text(
            destination, max_chars=self.config.fetch.max_pdf_chars
        )
        self._analyze(
            stats,
            paper_id=paper_id,
            paper=paper,
            text=text,
            scope="fulltext",
            strategy=strategy,
        )

    def _process_paper(
        self,
        stats: RunStats,
        author_id: str,
        paper: PaperRecord,
        *,
        analyze: bool,
        auto_analyze: bool,
    ) -> None:
        try:
            paper = self._enrich_from_arxiv(paper)
        except JobCancelled:
            raise
        except Exception as exc:
            self._record_error(
                stats,
                stage="arxiv_enrichment",
                exc=exc,
                author_id=author_id,
                paper=paper,
            )

        try:
            paper_id, is_new = self.db.upsert_paper(
                paper, auto_analyze=auto_analyze
            )
        except DeadSourceError:
            return
        if is_new:
            stats.new_paper_ids.append(paper_id)
        stored = self.db.get_paper(paper_id)
        analysis_allowed = analyze and bool(stored and stored["auto_analyze"])
        if analysis_allowed and self.config.llm.analyze_abstracts and paper.abstract:
            try:
                self._analyze(
                    stats,
                    paper_id=paper_id,
                    paper=paper,
                    text=paper.abstract,
                    scope="abstract",
                )
            except JobCancelled:
                raise
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
                analyze=analysis_allowed,
                analyze_new_downloads=analyze,
            )
        except JobCancelled:
            raise
        except Exception as exc:
            self._record_error(
                stats,
                stage="pdf_processing",
                exc=exc,
                author_id=author_id,
                paper=paper,
            )

    @staticmethod
    def _local_pdf_missing(row: Any) -> bool:
        value = str(row["local_pdf_path"] or "").strip()
        if not value:
            return True
        path = Path(value).expanduser()
        return not path.is_file() or not is_complete_pdf(path)

    def _retry_missing_pdfs(
        self,
        stats: RunStats,
        rows: list[Any],
        *,
        analyze: bool,
    ) -> None:
        for row in rows:
            check_cancelled(self._cancel_event)
            paper_id = int(row["id"])
            title = str(row["title"])
            try:
                paper = self.db.get_paper_record(paper_id)
                if paper is None:
                    continue
                if paper.openalex_id:
                    try:
                        refreshed = self.openalex.get_work(paper.openalex_id)
                        paper = refreshed
                    except Exception:
                        # A stored PDF URL or arXiv match may still be usable when
                        # OpenAlex is temporarily unavailable.
                        pass
                paper = self._enrich_from_arxiv(paper)
                paper_id, _ = self.db.upsert_paper(paper, auto_analyze=True)
                stored = self.db.get_paper(paper_id)
                self._process_pdf(
                    stats,
                    paper_id=paper_id,
                    paper=paper,
                    analyze=analyze and bool(stored and stored["auto_analyze"]),
                    analyze_new_downloads=analyze,
                )
            except JobCancelled:
                raise
            except Exception as exc:
                self._record_error(
                    stats,
                    stage="pdf_retry",
                    exc=RuntimeError(f"{title}: {exc}"),
                )
            finally:
                self._advance_progress(f"Retried PDF for {title}")

    @staticmethod
    def _run_identity(paper: PaperRecord) -> str:
        return paper.openalex_id or paper.doi or paper.arxiv_id or paper.identity_key

    @staticmethod
    def _best_title_match(title: str, papers: list[PaperRecord]) -> PaperRecord | None:
        if not papers:
            return None
        normalized = normalize_title(title)
        exact = next((paper for paper in papers if paper.normalized_title == normalized), None)
        if exact is not None:
            return exact
        return max(
            papers,
            key=lambda paper: SequenceMatcher(
                None, normalized, paper.normalized_title
            ).ratio(),
        )

    def fetch_title(self, title: str) -> tuple[RunStats, int | None]:
        query = title.strip()
        if not query:
            raise ValueError("Paper title cannot be empty")
        with run_lock(self.config.paths.database):
            return self._fetch_title_locked(query)

    def fetch_work(self, work_id: str) -> tuple[RunStats, int | None]:
        identifier = work_id.strip()
        if not identifier:
            raise ValueError("OpenAlex work id cannot be empty")
        with run_lock(self.config.paths.database):
            return self._fetch_work_locked(identifier)

    def _fetch_title_locked(self, title: str) -> tuple[RunStats, int | None]:
        stats = RunStats()
        stats.run_id = self.db.start_run()
        paper_id: int | None = None
        try:
            matches = self.openalex.search_works(title, limit=10)
            paper = self._best_title_match(title, matches)
            if paper is None:
                raise ValueError(f"No OpenAlex paper matched {title!r}")
            return self._download_selected_paper(stats, paper)
        except JobCancelled:
            stats.status = "cancelled"
            self.db.finish_run(stats.run_id, "cancelled", stats.as_dict())
            return stats, paper_id
        except Exception as exc:
            stats.status = "failed"
            self._record_error(stats, stage="paper_search_download", exc=exc)
            self.db.finish_run(
                stats.run_id, "failed", stats.as_dict(), error_text=str(exc)
            )
            return stats, getattr(exc, "paper_id", paper_id)

    def _fetch_work_locked(self, work_id: str) -> tuple[RunStats, int | None]:
        stats = RunStats()
        stats.run_id = self.db.start_run()
        try:
            paper = self.openalex.get_work(work_id)
            return self._download_selected_paper(stats, paper)
        except JobCancelled:
            stats.status = "cancelled"
            self.db.finish_run(stats.run_id, "cancelled", stats.as_dict())
            return stats, None
        except Exception as exc:
            stats.status = "failed"
            self._record_error(stats, stage="paper_download", exc=exc)
            self.db.finish_run(
                stats.run_id, "failed", stats.as_dict(), error_text=str(exc)
            )
            return stats, getattr(exc, "paper_id", None)

    def _download_selected_paper(
        self, stats: RunStats, paper: PaperRecord
    ) -> tuple[RunStats, int | None]:
        blogs = self.db.list_enabled_blogs()
        self._start_progress(1 + len(blogs), "Fetching selected paper and blogs")
        blog_executor, blog_future = self._start_blog_fetches(blogs)
        try:
            check_cancelled(self._cancel_event)
            stats.papers_found = 1
            try:
                paper = self._enrich_from_arxiv(paper)
            except Exception as exc:
                self._record_error(stats, stage="arxiv_enrichment", exc=exc, paper=paper)
            paper_id, is_new = self.db.upsert_paper(paper, auto_analyze=True)
            if is_new:
                stats.new_paper_ids.append(paper_id)
            paper_error: _SelectedPaperDownloadError | None = None
            try:
                self._process_pdf(
                    stats,
                    paper_id=paper_id,
                    paper=paper,
                    analyze=False,
                    analyze_new_downloads=False,
                )
                stored = self.db.get_paper(paper_id)
                local_path = Path(str(stored["local_pdf_path"] or "")) if stored else None
                if local_path is None or not is_complete_pdf(local_path):
                    raise PDFError("The selected paper has no downloadable PDF")
            except JobCancelled:
                raise
            except Exception as exc:
                paper_error = _SelectedPaperDownloadError(str(exc), paper_id)
            finally:
                self._advance_progress(f"Fetched paper {paper.title}")

            # A manual paper fetch is a complete fetch cycle too. Blog network
            # requests start before paper enrichment/download and are committed
            # after the paper work finishes, keeping SQLite single-threaded.
            if blog_future is not None:
                self._apply_blog_batch(stats, blog_future.result())
            check_cancelled(self._cancel_event)
            self._run_automatic_queries(stats)
            if paper_error is not None:
                raise paper_error
            stats.status = "partial" if stats.errors_count else "success"
            self.db.finish_run(stats.run_id, stats.status, stats.as_dict())
            return stats, paper_id
        finally:
            self._stop_blog_fetches(blog_executor)

    def run(
        self,
        *,
        today: date | None = None,
        since_override: date | None = None,
        download_only: bool = False,
        historical: bool = False,
        analyze_discovered_papers: bool = False,
        max_papers_per_author_override: int | None = None,
    ) -> RunStats:
        with run_lock(self.config.paths.database):
            return self._run_locked(
                today=today,
                since_override=since_override,
                download_only=download_only,
                historical=historical,
                analyze_discovered_papers=analyze_discovered_papers,
                max_papers_per_author_override=max_papers_per_author_override,
            )

    def _run_locked(
        self,
        *,
        today: date | None = None,
        since_override: date | None = None,
        download_only: bool = False,
        historical: bool = False,
        analyze_discovered_papers: bool = False,
        max_papers_per_author_override: int | None = None,
    ) -> RunStats:
        if max_papers_per_author_override is not None and max_papers_per_author_override < 1:
            raise ValueError("max_papers_per_author_override must be at least 1")
        stats = RunStats()
        stats.run_id = self.db.start_run()
        current_day = today or datetime.now(timezone.utc).date()
        seen_papers: set[str] = set()
        implicit_analysis = (
            analyze_discovered_papers and not download_only and not historical
        )
        self._llm_available = self.llm.provider != "none" and not download_only
        self._llm_preflight_failed = False
        blog_executor: ThreadPoolExecutor | None = None

        try:
            authors = self.db.list_authors(enabled_only=True)
            retry_since = (
                current_day - timedelta(days=self.config.fetch.retry_missing_pdf_days)
            ).isoformat()
            retry_rows = [
                row
                for row in self.db.list_recent_papers_for_pdf_retry(retry_since)
                if self._local_pdf_missing(row)
            ]
            blogs = [] if historical else self.db.list_enabled_blogs()
            stats.authors_total = len(authors)
            stats.blogs_total = len(blogs)
            self._start_progress(
                len(authors) + len(retry_rows) + len(blogs),
                (
                    "Backfilling papers and PDFs"
                    if historical
                    else "Fetching papers, PDFs, and blogs"
                ),
            )
            blog_executor, blog_future = self._start_blog_fetches(blogs)

            # Fetch-time model execution is controlled by the saved-query table.
            # This optional path exists only for explicit callers of the legacy
            # structured paper-analysis workflow.
            if implicit_analysis and self._llm_available:
                try:
                    check_ready = getattr(self.llm, "check_ready", None)
                    if check_ready is not None:
                        check_ready()
                except Exception as exc:
                    self._llm_available = False
                    self._llm_preflight_failed = True
                    self._record_error(stats, stage="llm_preflight", exc=exc)

            for author in authors:
                check_cancelled(self._cancel_event)
                author_id = author["openalex_id"]
                try:
                    papers = self.openalex.recent_works(
                        author_id,
                        since=(since_override or self._since(author, current_day)).isoformat(),
                        max_results=(
                            max_papers_per_author_override
                            or self.config.fetch.max_papers_per_author
                        ),
                    )
                except JobCancelled:
                    raise
                except Exception as exc:
                    self._record_error(
                        stats, stage="paper_discovery", exc=exc, author_id=author_id
                    )
                else:
                    stats.authors_checked += 1
                    for paper in papers:
                        check_cancelled(self._cancel_event)
                        run_identity = self._run_identity(paper)
                        if run_identity in seen_papers:
                            continue
                        seen_papers.add(run_identity)
                        stats.papers_found += 1
                        try:
                            self._process_paper(
                                stats,
                                author_id,
                                paper,
                                analyze=implicit_analysis,
                                auto_analyze=not historical,
                            )
                        except JobCancelled:
                            raise
                        except Exception as exc:
                            self._record_error(
                                stats,
                                stage="paper_processing",
                                exc=exc,
                                author_id=author_id,
                                paper=paper,
                            )
                    # Normal fetches do not depend on an implicit model. Only an
                    # explicit legacy-analysis caller needs to preserve the old
                    # retry-on-model-failure cursor behavior.
                    if not implicit_analysis or not self._llm_preflight_failed:
                        self.db.mark_author_checked(author_id)
                finally:
                    self._advance_progress(f"Checked author {author['name']}")

            self._retry_missing_pdfs(
                stats,
                retry_rows,
                analyze=implicit_analysis,
            )
            if blog_future is not None:
                self._apply_blog_batch(stats, blog_future.result())

            if not download_only and not historical:
                check_cancelled(self._cancel_event)
                self._run_automatic_queries(stats)

            stats.status = "partial" if stats.errors_count else "success"
            self.db.finish_run(stats.run_id, stats.status, stats.as_dict())
            return stats
        except JobCancelled:
            stats.status = "cancelled"
            self.db.finish_run(stats.run_id, "cancelled", stats.as_dict())
            return stats
        except Exception as exc:
            stats.status = "failed"
            self.db.finish_run(
                stats.run_id, "failed", stats.as_dict(), error_text=str(exc)
            )
            raise
        finally:
            self._stop_blog_fetches(blog_executor)

    @staticmethod
    def _stored_query_profiles(query: Any) -> tuple[str, ...]:
        try:
            profile = str(query["model_profile"] or "").strip()
        except (KeyError, IndexError):
            profile = ""
        if profile:
            return (profile,)
        try:
            values = json.loads(str(query["model_profiles_json"] or "[]"))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            values = []
        if not isinstance(values, list):
            return ()
        return tuple(dict.fromkeys(
            str(value).strip() for value in values if str(value).strip()
        ))[:1]

    def _run_automatic_queries(self, stats: RunStats) -> None:
        from .query_jobs import collect_query_documents, run_query_for_model

        available = {item.profile: item for item in self.config.available_llms}
        if not available:
            return

        jobs: list[tuple[str, list[int], str, Any, tuple[str, ...]]] = []
        targets = (
            ("papers", stats.new_paper_ids),
            ("blogs", stats.new_blog_entry_ids),
        )
        for content_type, raw_ids in targets:
            ids = list(dict.fromkeys(raw_ids))
            if not ids:
                continue
            for query in self.db.list_automatic_queries(content_type):
                selected = self._stored_query_profiles(query)
                if not selected:
                    selected = (next(iter(available)),)
                valid = tuple(name for name in selected if name in available)
                missing = [name for name in selected if name not in available]
                if missing:
                    self._record_error(
                        stats,
                        stage=f"automatic_{content_type}_query",
                        exc=RuntimeError(
                            f"{query['name']}: configured model profile(s) are unavailable: "
                            f"{', '.join(missing)}"
                        ),
                    )
                if valid:
                    jobs.append((content_type, ids, str(query["name"]), query, valid))

        if not jobs:
            return
        self._start_progress(
            sum(len(profiles) for _, _, _, _, profiles in jobs),
            "Running automatic prompts",
        )

        for content_type, ids, query_name, query, profile_names in jobs:
            check_cancelled(self._cancel_event)
            source_scope = (
                "blog" if content_type == "blogs"
                else str(query["paper_source_scope"] or "auto")
            )
            pdf_strategy = (
                "auto" if content_type == "blogs"
                else str(query["pdf_strategy"] or "chunked")
            )
            context_window_tokens = available[profile_names[0]].context_window_tokens
            selection: dict[str, Any] = {
                "content_type": content_type,
                "source_scope": source_scope,
                "pdf_strategy": pdf_strategy,
                "labels": [f"New {content_type}: {len(ids)}"],
            }
            selection[
                "paper_ids" if content_type == "papers" else "blog_entry_ids"
            ] = ids
            batch_id = f"auto-{stats.run_id}-{content_type}-{query['id']}"
            run_ids = self.db.create_query_runs(
                batch_id=batch_id,
                query_id=int(query["id"]),
                profiles=[
                    (name, available[name].provider, available[name].model)
                    for name in profile_names
                ],
                context_window_tokens=context_window_tokens,
                selection=selection,
            )
            try:
                documents = collect_query_documents(
                    self.db,
                    source_scope,
                    selection=selection,
                    config=self.config,
                    pdf_client=self.pdf_client,
                )
            except JobCancelled:
                for run_id in run_ids:
                    self.db.finish_query_run(
                        run_id, matched_count=0, input_chars=0, error="Cancelled by user"
                    )
                raise
            except Exception as exc:
                for run_id in run_ids:
                    self.db.finish_query_run(
                        run_id, matched_count=0, input_chars=0, error=str(exc)
                    )
                    self._advance_progress(f"{query_name} failed before model execution")
                self._record_error(
                    stats,
                    stage=f"automatic_{content_type}_query",
                    exc=RuntimeError(f"{query_name}: {exc}"),
                )
                continue

            for profile_name, run_id in zip(profile_names, run_ids, strict=True):
                check_cancelled(self._cancel_event)
                profile = available[profile_name]
                self.db.start_query_run(run_id)
                llm: LLMClient | None = None
                owns_llm = False
                try:
                    if profile_name == self.config.llm.profile:
                        llm = self.llm
                    else:
                        llm = create_llm(profile)
                        owns_llm = True
                    configure_job_runtime(llm, self._progress, self._cancel_event)
                    configure_context_window(llm, context_window_tokens)
                    check_ready = getattr(llm, "check_ready", None)
                    if check_ready is not None:
                        check_ready()
                    result, _usage, input_chars = run_query_for_model(
                        llm,
                        user_prompt=str(query["prompt"]),
                        documents=documents,
                        context_window_tokens=context_window_tokens,
                        max_input_chars=profile.max_input_chars,
                        pdf_strategy=pdf_strategy,
                        content_type=content_type,
                        progress=self._progress,
                        cancel_event=self._cancel_event,
                    )
                    self.db.finish_query_run(
                        run_id,
                        matched_count=len(documents),
                        input_chars=input_chars,
                        output=result.model_dump(),
                    )
                except JobCancelled:
                    self.db.finish_query_run(
                        run_id,
                        matched_count=len(documents),
                        input_chars=0,
                        error="Cancelled by user",
                    )
                    raise
                except Exception as exc:
                    self.db.finish_query_run(
                        run_id,
                        matched_count=len(documents),
                        input_chars=0,
                        error=str(exc),
                    )
                    self._record_error(
                        stats,
                        stage=f"automatic_{content_type}_query",
                        exc=RuntimeError(
                            f"{query_name} with {profile_name}: {exc}"
                        ),
                    )
                finally:
                    if llm is not None and owns_llm:
                        llm.close()
                    self._advance_progress(
                        f"Ran {query_name} with {profile_name} on new {content_type}"
                    )
