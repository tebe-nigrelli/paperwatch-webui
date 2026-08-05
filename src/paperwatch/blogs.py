from __future__ import annotations

import hashlib
import html
import importlib.util
import ipaddress
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher, unified_diff
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from soupsieve.util import SelectorSyntaxError


MAX_RESPONSE_BYTES = 5_000_000
MAX_TEXT_CHARS = 1_000_000
MAX_DIFF_CHARS = 200_000
MAX_FEED_ITEMS = 200
MAX_FEED_CANDIDATES = 8
MAX_ARTICLE_FETCHES = 8
BLOG_REQUEST_TIMEOUT = 20.0
YOUTUBE_PLAYLIST_LIMIT = 20
YOUTUBE_REQUEST_TIMEOUT = 90.0
FEED_CONTENT_TYPES = {
    "application/atom+xml",
    "application/rdf+xml",
    "application/rss+xml",
}
SOURCE_MODES = {
    "auto",
    "feed",
    "article_index",
    "article",
    "docs",
    "publications",
    "github",
    "video",
}
OUTBOUND_LINKS_HEADER = "Outbound links:"
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref_src",
}
VOLATILE_COUNT_WORDS = {
    "comments",
    "followers",
    "following",
    "forks",
    "likes",
    "reactions",
    "shares",
    "stars",
    "subscribers",
    "views",
}
SOCIAL_HOSTS = {
    "bsky.app",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "mastodon.social",
    "threads.net",
    "twitter.com",
    "x.com",
}
IGNORED_LINK_LABELS = {
    "about",
    "back",
    "contact",
    "cookie policy",
    "home",
    "log in",
    "login",
    "menu",
    "next",
    "privacy",
    "privacy policy",
    "previous",
    "search",
    "share",
    "sign in",
    "sign up",
    "subscribe",
    "terms",
    "terms of service",
}
SCOPED_PATH_MARKERS = {"author", "authors", "category", "categories", "tag", "tags"}
SCOPED_QUERY_KEYS = {"author", "category", "tag"}
ARTICLE_PATH_WORDS = {
    "article",
    "articles",
    "blog",
    "blogs",
    "insight",
    "insights",
    "news",
    "post",
    "posts",
    "publication",
    "publications",
    "research",
    "story",
    "stories",
    "writing",
}
INDEX_PATH_WORDS = {
    "archive",
    "archives",
    "articles",
    "blog",
    "blogs",
    "category",
    "categories",
    "insight",
    "insights",
    "news",
    "newsroom",
    "posts",
    "publication",
    "publications",
    "research",
    "stories",
    "tag",
    "tags",
}
NON_ARTICLE_PATH_WORDS = {
    "about",
    "author",
    "authors",
    "category",
    "categories",
    "contact",
    "download",
    "downloads",
    "event",
    "events",
    "feed",
    "login",
    "privacy",
    "product",
    "products",
    "search",
    "support",
    "signin",
    "signup",
    "tag",
    "tags",
    "terms",
}


@dataclass(frozen=True, slots=True)
class FeedItem:
    external_id: str
    title: str
    url: str | None
    published_at: str | None
    text: str
    sha256: str
    outbound_links: tuple[str, ...] = ()
    rendered_html: str | None = None


@dataclass(frozen=True, slots=True)
class ArticleLink:
    title: str
    url: str
    summary: str = ""
    published_at: str | None = None


@dataclass(frozen=True, slots=True)
class ArticleItem:
    external_id: str
    title: str
    url: str
    text: str
    sha256: str
    outbound_links: tuple[str, ...] = ()
    rendered_html: str | None = None
    published_at: str | None = None


@dataclass(frozen=True, slots=True)
class BlogFetch:
    method: str
    title: str
    status_code: int
    not_modified: bool = False
    etag: str | None = None
    last_modified: str | None = None
    feed_url: str | None = None
    discovery_checked: bool = False
    snapshot_text: str | None = None
    snapshot_sha256: str | None = None
    snapshot_links: tuple[str, ...] = ()
    snapshot_rendered_html: str | None = None
    snapshot_published_at: str | None = None
    feed_items: tuple[FeedItem, ...] = ()
    discovered_articles: tuple[ArticleLink, ...] = ()
    article_items: tuple[ArticleItem, ...] = ()
    source_mode: str = "auto"
    diagnostics: tuple[str, ...] = ()
    article_cursor_next: int | None = None


def normalize_blog_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("Blog URL is required")
    if len(raw) > 2000:
        raise ValueError("Blog URL is too long")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Blog URL must be a valid http or https URL")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Local blog URLs are not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Private or local blog URLs are not allowed")
    port = parsed.port
    netloc = hostname
    if port and not (parsed.scheme.lower() == "http" and port == 80) and not (
        parsed.scheme.lower() == "https" and port == 443
    ):
        netloc = f"{hostname}:{port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def normalize_source_mode(value: object) -> str:
    mode = str(value or "auto").strip().lower().replace("-", "_")
    if mode not in SOURCE_MODES:
        raise ValueError(f"Unknown blog source mode: {mode}")
    return mode


def validate_content_selector(value: object) -> str | None:
    selector = str(value or "").strip()
    if not selector:
        return None
    if len(selector) > 500:
        raise ValueError("Content selector is too long")
    try:
        BeautifulSoup("<html><body></body></html>", "html.parser").select(selector)
    except SelectorSyntaxError as exc:
        raise ValueError("Content selector is invalid") from exc
    return selector


def validate_article_url_pattern(value: object) -> str | None:
    pattern = str(value or "").strip()
    if not pattern:
        return None
    if len(pattern) > 500:
        raise ValueError("Article URL pattern is too long")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError("Article URL pattern is invalid") from exc
    return pattern


def infer_source_mode(url: str, source_type: str = "", label: str = "") -> str:
    host = (urlsplit(url).hostname or "").lower()
    context = f"{source_type} {label}".casefold()
    if host == "github.com" or "github" in context:
        return "github"
    if "youtube.com" in host or "youtu.be" in host or "video" in context:
        return "video"
    if any(word in context for word in ("documentation", "online textbook", "living technical", "technical guides")):
        return "docs"
    if "publication" in context or "research / project feed" in context:
        return "publications"
    if any(word in context for word in ("rss", "atom", "newsletter")):
        return "feed"
    return "auto"


def _row_value(row: object, key: str, default: object = None) -> object:
    try:
        value = row[key]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return default
    return value


def _clean_lines(value: str) -> str:
    lines: list[str] = []
    for raw in value.splitlines():
        cleaned = re.sub(r"\s+", " ", raw).strip()
        if cleaned and (not lines or lines[-1] != cleaned):
            lines.append(cleaned)
    text = "\n".join(lines)
    if not text:
        raise ValueError("Blog page did not contain readable text")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return text


def _clean_readable_text(value: str) -> str:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", value):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        cleaned = "\n".join(line for line in lines if line).strip()
        if cleaned and (not paragraphs or paragraphs[-1] != cleaned):
            paragraphs.append(cleaned)
    text = "\n".join(paragraphs)
    if not text:
        raise ValueError("Blog page did not contain readable text")
    return text[:MAX_TEXT_CHARS]


def _normalize_link_url(value: str, base_url: str) -> str | None:
    try:
        parsed = urlsplit(urljoin(base_url, value.strip()))
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return None
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            return None
        port = parsed.port
    except ValueError:
        return None
    netloc = hostname
    if port and not (parsed.scheme.lower() == "http" and port == 80) and not (
        parsed.scheme.lower() == "https" and port == 443
    ):
        netloc = f"{hostname}:{port}"
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
            and key.lower() not in TRACKING_QUERY_KEYS
        ],
        doseq=True,
    )
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", query, ""))


def _link_label(element: object) -> str:
    try:
        value = element.get_text(" ", strip=True)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return ""
    label = re.sub(r"\s+", " ", value).strip()
    if not label:
        try:
            label = str(element.get("aria-label") or element.get("title") or "").strip()  # type: ignore[attr-defined]
        except AttributeError:
            return ""
    if not label or len(label) > 300:
        return ""
    if re.search(r"\b(?:profile picture|avatar|logo|icon)\b", label, re.I):
        return ""
    return label


def _ignored_link(label: str, url: str, base_url: str) -> bool:
    normalized_label = re.sub(r"\s+", " ", label).strip().casefold()
    if normalized_label in IGNORED_LINK_LABELS:
        return True
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host.removeprefix("www.") in SOCIAL_HOSTS:
        return True
    suffix = parsed.path.lower().rsplit("/", 1)[-1]
    if re.search(r"\.(?:avif|css|gif|ico|jpe?g|js|pdf|png|svg|webp|zip)$", suffix):
        return True
    if parsed.path in {"", "/"} and host == (urlsplit(base_url).hostname or "").lower():
        return True
    if any(token in parsed.path.lower().split("/") for token in ("share", "login", "signin", "signup")):
        return True
    return False


def _outbound_link_pairs(root: object, base_url: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    try:
        elements = root.find_all("a", href=True)  # type: ignore[attr-defined]
    except AttributeError:
        return ()
    for element in elements:
        label = _link_label(element)
        if not label:
            continue
        url = _normalize_link_url(str(element.get("href") or ""), base_url)
        if url is None or _ignored_link(label, url, base_url):
            continue
        item = (label, url)
        if item not in pairs:
            pairs.append(item)
    return tuple(pairs)


def _text_with_links(text: str, links: tuple[tuple[str, str], ...]) -> str:
    if not links:
        return text
    link_lines = [OUTBOUND_LINKS_HEADER]
    link_lines.extend(f"- {label} -> {url}" for label, url in links)
    combined = f"{text}\n" + "\n".join(link_lines)
    return combined[:MAX_TEXT_CHARS]


_SAFE_CONTENT_TAGS = {
    "a", "b", "blockquote", "br", "code", "em", "figcaption", "figure",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "li", "ol",
    "p", "pre", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul",
}
_BLOCK_TAGS = {
    "blockquote", "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "p", "pre", "tr",
}


def _sanitize_content_html(root: object, base_url: str) -> str:
    fragment = BeautifulSoup(str(root), "html.parser")
    for element in fragment(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
        element.decompose()
    for tag in list(fragment.find_all(True)):
        if tag.name not in _SAFE_CONTENT_TAGS:
            tag.unwrap()
            continue
        if tag.name == "a":
            href = _normalize_link_url(str(tag.get("href") or ""), base_url)
            tag.attrs = {}
            if href is None:
                tag.unwrap()
            else:
                tag["href"] = href
                tag["target"] = "_blank"
                tag["rel"] = "noreferrer noopener"
        elif tag.name == "img":
            raw_src = next(
                (
                    str(tag.get(name) or "").strip()
                    for name in ("src", "data-src", "data-original", "data-lazy-src")
                    if str(tag.get(name) or "").strip()
                ),
                "",
            )
            src = _normalize_link_url(raw_src, base_url) if raw_src else None
            alt = re.sub(r"\s+", " ", str(tag.get("alt") or tag.get("title") or "Image")).strip()
            tag.attrs = {}
            if src is None:
                tag.decompose()
                continue
            tag["src"] = src
            tag["alt"] = alt[:300] or "Image"
            tag["loading"] = "lazy"
            tag["decoding"] = "async"
            tag["referrerpolicy"] = "no-referrer"
            if not (isinstance(tag.parent, Tag) and tag.parent.name == "a"):
                wrapper = fragment.new_tag("a")
                wrapper["href"] = src
                wrapper["target"] = "_blank"
                wrapper["rel"] = "noreferrer noopener"
                wrapper["title"] = "Open image"
                tag.wrap(wrapper)
        else:
            tag.attrs = {}
    return "".join(str(child) for child in fragment.contents).strip()


def _inline_markdown(tag: Tag) -> str:
    clone = BeautifulSoup(str(tag), "html.parser")
    for nested in clone.find_all(["ul", "ol", "table"]):
        nested.decompose()
    for anchor in list(clone.find_all("a")):
        image = anchor.find("img")
        if image is None:
            continue
        src = str(image.get("src") or anchor.get("href") or "").strip()
        alt = re.sub(r"\s+", " ", str(image.get("alt") or "Image")).strip()
        anchor.replace_with(NavigableString(f"![{alt}]({src})" if src else alt))
    for image in list(clone.find_all("img")):
        src = str(image.get("src") or "").strip()
        alt = re.sub(r"\s+", " ", str(image.get("alt") or "Image")).strip()
        image.replace_with(NavigableString(f"![{alt}]({src})" if src else alt))
    for anchor in clone.find_all("a"):
        label = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
        href = str(anchor.get("href") or "").strip()
        anchor.replace_with(NavigableString(f"[{label}]({href})" if label and href else label))
    break_marker = "\ue000"
    for br in clone.find_all("br"):
        br.replace_with(NavigableString(break_marker))
    text = clone.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(f" {break_marker} ", "\n").replace(break_marker, "\n")
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"([\(\[\{]) +", r"\1", text)
    text = re.sub(r" +([\)\]\}\.,;:!?])", r"\1", text)
    return text.strip()


def _readable_text_from_html(rendered_html: str) -> str:
    soup = BeautifulSoup(rendered_html, "html.parser")
    blocks: list[str] = []
    for tag in soup.find_all(_BLOCK_TAGS):
        if any(
            isinstance(parent, Tag) and parent.name in _BLOCK_TAGS
            for parent in tag.parents
            if parent is not soup
        ):
            continue
        text = _inline_markdown(tag)
        if not text:
            continue
        if tag.name == "li":
            text = f"- {text}"
        elif tag.name == "blockquote":
            text = "\n".join(f"> {line}" for line in text.splitlines())
        blocks.append(text)
    for image in soup.find_all("img"):
        if any(
            isinstance(parent, Tag) and parent.name in _BLOCK_TAGS
            for parent in image.parents
            if parent is not soup
        ):
            continue
        src = str(image.get("src") or "").strip()
        alt = re.sub(r"\s+", " ", str(image.get("alt") or "Image")).strip()
        if src:
            blocks.append(f"![{alt or 'Image'}]({src})")
    if not blocks:
        fallback = soup.get_text(" ", strip=True)
        return _clean_readable_text(fallback)
    return _clean_readable_text("\n\n".join(blocks))


def _content_sha256(text: str, links: tuple[str, ...] = ()) -> str:
    payload = text
    if links and OUTBOUND_LINKS_HEADER not in text:
        payload += "\n" + "\n".join(sorted(links))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decode_outbound_links(value: object) -> tuple[str, ...] | None:
    if not value:
        return None
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        return None
    return tuple(dict.fromkeys(payload))


def _document_root(soup: BeautifulSoup, selector: str | None) -> object:
    if selector:
        root = soup.select_one(selector)
        if root is None:
            raise ValueError(f"Content selector did not match: {selector}")
        return root
    return soup.find("main") or soup.find("article") or soup.body or soup


def _extract_blog_document_full(
    html: str,
    fallback_title: str,
    base_url: str,
    selector: str | None = None,
) -> tuple[str, str, tuple[str, ...], str]:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
        element.decompose()
    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
    root = _document_root(soup, selector)
    rendered_html = _sanitize_content_html(root, base_url)
    text = _readable_text_from_html(rendered_html)
    pairs = _outbound_link_pairs(root, base_url)
    normalized = _text_with_links(text, pairs)
    return (
        title.strip() or fallback_title,
        normalized,
        tuple(url for _, url in pairs),
        rendered_html,
    )


def _extract_blog_document(
    html: str,
    fallback_title: str,
    base_url: str,
    selector: str | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    title, text, links, _rendered_html = _extract_blog_document_full(
        html, fallback_title, base_url, selector
    )
    return title, text, links


def extract_blog_text(
    html: str,
    fallback_title: str,
    base_url: str | None = None,
    selector: str | None = None,
) -> tuple[str, str]:
    title, text, _, _ = _extract_blog_document_full(
        html,
        fallback_title,
        base_url or "https://invalid.example/",
        selector,
    )
    return title, text


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ElementTree.Element, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in element:
        if _local_name(child.tag) in wanted:
            return "".join(child.itertext()).strip()
    return ""


def _html_text(
    value: str, *, base_url: str | None = None
) -> tuple[str, tuple[str, ...], str]:
    soup = BeautifulSoup(value, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
        element.decompose()
    rendered_html = _sanitize_content_html(soup, base_url or "https://invalid.example/")
    text = _readable_text_from_html(rendered_html)
    if base_url is None:
        return text, (), rendered_html
    pairs = _outbound_link_pairs(soup, base_url)
    return _text_with_links(text, pairs), tuple(url for _, url in pairs), rendered_html


def _published_at(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw[:100]
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


_MONTH_NUMBERS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_PATTERN = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)


def _date_value(year: int, month: int, day: int | None) -> str | None:
    try:
        if day is None:
            date(year, month, 1)
            return f"{year:04d}-{month:02d}"
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def extract_first_date(value: str) -> str | None:
    """Return the first valid date-like string as ISO, preserving month precision.

    Numeric dates use US month/day/year order when ambiguous. Patterns do not
    require surrounding whitespace so dates embedded in minified text remain
    detectable, for example ``ertApr 14, 20265554Share``.
    """
    candidates: list[tuple[int, int, str]] = []

    def add(match: re.Match[str], parsed: str | None, precision: int) -> None:
        if parsed is not None:
            candidates.append((match.start(), -precision, parsed))

    for match in re.finditer(
        rf"(?P<month>{_MONTH_PATTERN})\s*,?\s*(?P<day>\d{{1,2}})"
        rf"(?:st|nd|rd|th)?\s*,?\s*(?P<year>(?:19|20)\d{{2}})",
        value,
        re.IGNORECASE,
    ):
        month = _MONTH_NUMBERS[match.group("month").casefold()]
        add(
            match,
            _date_value(int(match.group("year")), month, int(match.group("day"))),
            3,
        )

    for match in re.finditer(
        rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*(?P<month>{_MONTH_PATTERN})"
        rf"\s*,?\s*(?P<year>(?:19|20)\d{{2}})",
        value,
        re.IGNORECASE,
    ):
        month = _MONTH_NUMBERS[match.group("month").casefold()]
        add(
            match,
            _date_value(int(match.group("year")), month, int(match.group("day"))),
            3,
        )

    for match in re.finditer(
        r"(?P<year>(?:19|20)\d{2})\s*[-/.]\s*(?P<month>\d{1,2})"
        r"\s*[-/.]\s*(?P<day>\d{1,2})",
        value,
    ):
        add(
            match,
            _date_value(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ),
            3,
        )

    for match in re.finditer(
        r"(?P<month>\d{1,2})\s*[/.-]\s*(?P<day>\d{1,2})"
        r"\s*[/.-]\s*(?P<year>(?:19|20)\d{2})",
        value,
    ):
        add(
            match,
            _date_value(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ),
            3,
        )

    for match in re.finditer(
        rf"(?P<month>{_MONTH_PATTERN})\s*,?\s*(?P<year>(?:19|20)\d{{2}})",
        value,
        re.IGNORECASE,
    ):
        month = _MONTH_NUMBERS[match.group("month").casefold()]
        add(match, _date_value(int(match.group("year")), month, None), 2)

    if not candidates:
        return None
    return min(candidates)[2]


def _html_first_date(value: str) -> str | None:
    """Extract a publication date from structured HTML, then visible text."""
    soup = BeautifulSoup(value, "html.parser")

    def parsed_date(raw: object) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        if parsed := extract_first_date(text):
            return parsed
        parsed = _published_at(text)
        return parsed if parsed and re.match(r"^\d{4}-\d{2}", parsed) else None

    metadata_keys = {
        "article:published_time",
        "date",
        "datecreated",
        "datepublished",
        "dc.date",
        "dc.date.issued",
        "og:published_time",
        "publish-date",
        "pubdate",
    }
    for element in soup.find_all("meta"):
        key = str(
            element.get("property")
            or element.get("name")
            or element.get("itemprop")
            or ""
        ).strip().casefold()
        if key not in metadata_keys:
            continue
        if parsed := parsed_date(element.get("content")):
            return parsed

    for element in soup.find_all("time"):
        if parsed := parsed_date(
            element.get("datetime") or element.get_text(" ", strip=True)
        ):
            return parsed

    def json_dates(payload: object) -> list[str]:
        found: list[str] = []
        if isinstance(payload, dict):
            for key, item in payload.items():
                if str(key).casefold() in {"datepublished", "datecreated"}:
                    if parsed := parsed_date(item):
                        found.append(parsed)
                else:
                    found.extend(json_dates(item))
        elif isinstance(payload, list):
            for item in payload:
                found.extend(json_dates(item))
        return found

    for element in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(element.string or element.get_text())
        except (TypeError, ValueError):
            continue
        dates = json_dates(payload)
        if dates:
            return dates[0]

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()
    return extract_first_date(soup.get_text(" ", strip=True))


def _youtube_url(value: str) -> bool:
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return host in {"youtube.com", "m.youtube.com", "youtu.be"}


def _description_links(value: str, base_url: str) -> tuple[str, ...]:
    links: list[str] = []
    for raw in re.findall(r"https?://[^\s<>\[\](){}\"']+", value):
        candidate = raw.rstrip(".,;:!?")
        normalized = _normalize_link_url(candidate, base_url)
        if normalized and normalized not in links:
            links.append(normalized)
    return tuple(links)


def _plain_text_html(value: str) -> str | None:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]
    if not paragraphs:
        return None
    return "".join(
        "<p>" + html.escape(part).replace("\n", "<br>") + "</p>"
        for part in paragraphs
    )


def _youtube_published_at(payload: dict[str, object], description: str) -> str | None:
    for key in ("release_timestamp", "timestamp"):
        raw = payload.get(key)
        if isinstance(raw, (int, float)):
            try:
                return datetime.fromtimestamp(raw, timezone.utc).replace(microsecond=0).isoformat()
            except (OverflowError, OSError, ValueError):
                pass
    for key in ("release_date", "upload_date"):
        raw = str(payload.get(key) or "").strip()
        if re.fullmatch(r"\d{8}", raw):
            try:
                return datetime.strptime(raw, "%Y%m%d").date().isoformat()
            except ValueError:
                pass
        parsed = extract_first_date(raw)
        if parsed:
            return parsed
    return extract_first_date(description)


def _youtube_feed_item(payload: dict[str, object], fallback_url: str) -> FeedItem | None:
    url = str(
        payload.get("webpage_url")
        or payload.get("original_url")
        or payload.get("url")
        or fallback_url
    ).strip()
    if not _youtube_url(url):
        url = fallback_url
    try:
        normalized_url = normalize_blog_url(url)
    except ValueError:
        return None
    title = str(payload.get("title") or payload.get("fulltitle") or "YouTube video").strip()
    description = str(payload.get("description") or "").strip()
    published_at = _youtube_published_at(payload, description)
    links = _description_links(description, normalized_url)
    text = _clean_lines(
        "\n".join(
            part
            for part in (
                f"Title: {title}",
                f"Published: {published_at}" if published_at else "",
                f"URL: {normalized_url}",
                description,
            )
            if part
        )
    )
    html_parts = [
        '<p class="rss-source-link"><a href="'
        + html.escape(normalized_url, quote=True)
        + '" target="_blank" rel="noreferrer">Open source</a></p>'
    ]
    if published_at:
        html_parts.append(
            '<p class="rss-published"><time datetime="'
            + html.escape(published_at, quote=True)
            + '">'
            + html.escape(published_at)
            + "</time></p>"
        )
    description_html = _plain_text_html(description)
    if description_html:
        html_parts.append(description_html)
    identity = str(payload.get("id") or normalized_url)
    return FeedItem(
        external_id=hashlib.sha256(f"youtube\0{identity}".encode("utf-8")).hexdigest(),
        title=title,
        url=normalized_url,
        published_at=published_at,
        text=text,
        sha256=_content_sha256(text, links),
        outbound_links=links,
        rendered_html="".join(html_parts),
    )


def _youtube_fetch(row: object) -> BlogFetch:
    url = str(_row_value(row, "url", ""))
    if importlib.util.find_spec("yt_dlp") is None:
        raise ValueError("yt-dlp is not installed")
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--dump-single-json",
                "--skip-download",
                "--no-warnings",
                "--ignore-errors",
                "--playlist-end",
                str(YOUTUBE_PLAYLIST_LIMIT),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=YOUTUBE_REQUEST_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"yt-dlp failed: {exc}") from exc
    if process.returncode != 0:
        detail = process.stderr.strip().splitlines()[-1:] or ["unknown error"]
        raise ValueError(f"yt-dlp failed: {detail[0][:300]}")
    try:
        payload = json.loads(process.stdout)
    except (TypeError, ValueError) as exc:
        raise ValueError("yt-dlp returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("yt-dlp returned an invalid result")
    raw_entries = payload.get("entries")
    entries = raw_entries if isinstance(raw_entries, list) else [payload]
    items = tuple(
        item
        for raw in entries[:YOUTUBE_PLAYLIST_LIMIT]
        if isinstance(raw, dict)
        and (item := _youtube_feed_item(raw, url)) is not None
    )
    if not items:
        raise ValueError("yt-dlp returned no readable videos")
    title = str(payload.get("title") or _row_value(row, "title", "") or url).strip()
    return BlogFetch(
        method="rss",
        title=title,
        status_code=200,
        feed_items=items,
        source_mode="video",
        discovery_checked=True,
        diagnostics=(f"Loaded {len(items)} YouTube video descriptions with yt-dlp",),
    )


@dataclass(frozen=True, slots=True)
class _RawFeedItem:
    title: str
    url: str | None
    identity: str
    published_at: str | None
    body_text: str
    outbound_links: tuple[str, ...]
    rendered_html: str | None


def _raw_feed_item(
    element: ElementTree.Element,
    *,
    base_url: str,
    atom: bool,
) -> _RawFeedItem:
    title = _child_text(element, "title") or "Untitled post"
    if atom:
        link = ""
        for child in element:
            if _local_name(child.tag) != "link":
                continue
            rel = child.attrib.get("rel", "alternate").lower()
            href = child.attrib.get("href", "").strip()
            if href and rel in {"", "alternate"}:
                link = href
                break
        identity = _child_text(element, "id")
        published = _child_text(element, "published", "updated")
        body = _child_text(element, "content", "summary")
    else:
        link = _child_text(element, "link")
        identity = _child_text(element, "guid", "id")
        published = _child_text(element, "pubdate", "date", "published", "updated")
        body = _child_text(element, "encoded", "description", "content", "summary")

    item_url = _normalize_link_url(link, base_url) if link else None
    if item_url:
        parsed_item_url = urlsplit(item_url)
        if (parsed_item_url.hostname or "").lower() in {
            "arxiv.org",
            "www.arxiv.org",
            "export.arxiv.org",
        }:
            item_url = urlunsplit(
                (
                    "https",
                    parsed_item_url.netloc,
                    parsed_item_url.path,
                    parsed_item_url.query,
                    "",
                )
            )
    published_at = _published_at(published)
    try:
        body_text, outbound_links, rendered_html = (
            _html_text(body, base_url=item_url or base_url)
            if body
            else (_clean_lines(title), (), None)
        )
    except ValueError:
        body_text, outbound_links, rendered_html = _clean_lines(title), (), None
    return _RawFeedItem(
        title=title.strip(),
        url=item_url,
        identity=identity.strip(),
        published_at=published_at,
        body_text=body_text,
        outbound_links=outbound_links,
        rendered_html=rendered_html,
    )


def _feed_external_id(
    item: _RawFeedItem,
    *,
    base_url: str,
    duplicated_urls: set[str],
    duplicated_ids: set[str],
) -> tuple[str, str | None]:
    item_url = item.url
    if item_url and item_url not in duplicated_urls and item_url != normalize_blog_url(base_url):
        raw = f"url\0{item_url}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest(), item_url
    identity = item.identity
    if identity and identity not in duplicated_ids:
        normalized_identity = _normalize_link_url(identity, base_url)
        raw_identity = normalized_identity or identity
        raw = f"guid\0{raw_identity}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest(), None if item_url in duplicated_urls else item_url
    raw = f"fallback\0{normalize_blog_url(base_url)}\0{item.title}\0{item.published_at or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), None if item_url in duplicated_urls else item_url


def _feed_item_html(
    item: _RawFeedItem,
    item_url: str | None,
    *,
    published_at: str | None = None,
) -> str | None:
    """Render feed metadata with a real source link before the sanitized body."""
    parts: list[str] = []
    if item_url:
        parts.append(
            '<p class="rss-source-link"><a href="'
            + html.escape(item_url, quote=True)
            + '" target="_blank" rel="noreferrer">Open source</a></p>'
        )
    displayed_date = published_at or item.published_at
    if displayed_date:
        parts.append(
            '<p class="rss-published"><time datetime="'
            + html.escape(displayed_date, quote=True)
            + '">'
            + html.escape(displayed_date)
            + '</time></p>'
        )
    if item.rendered_html:
        parts.append(item.rendered_html)
    return "".join(parts) or None


def parse_feed(xml: str, base_url: str, fallback_title: str) -> tuple[str, tuple[FeedItem, ...]]:
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise ValueError("Invalid RSS or Atom feed") from exc

    root_name = _local_name(root.tag)
    if root_name == "feed":
        atom = True
        title = _child_text(root, "title") or fallback_title
        elements = [item for item in root if _local_name(item.tag) == "entry"]
    elif root_name in {"rss", "rdf"}:
        atom = False
        channel = next(
            (item for item in root.iter() if _local_name(item.tag) == "channel"),
            root,
        )
        title = _child_text(channel, "title") or fallback_title
        elements = [item for item in root.iter() if _local_name(item.tag) == "item"]
    else:
        raise ValueError("Document is not an RSS or Atom feed")

    raw_items = [
        _raw_feed_item(element, base_url=base_url, atom=atom)
        for element in elements[:MAX_FEED_ITEMS]
    ]
    url_counts = Counter(item.url for item in raw_items if item.url)
    identity_counts = Counter(item.identity for item in raw_items if item.identity)
    duplicated_urls = {url for url, count in url_counts.items() if count > 1}
    duplicated_ids = {identity for identity, count in identity_counts.items() if count > 1}
    parsed: list[FeedItem] = []
    seen_external_ids: set[str] = set()
    for index, item in enumerate(raw_items):
        external_id, item_url = _feed_external_id(
            item,
            base_url=base_url,
            duplicated_urls=duplicated_urls,
            duplicated_ids=duplicated_ids,
        )
        if external_id in seen_external_ids:
            external_id = hashlib.sha256(f"{external_id}\0{index}".encode("utf-8")).hexdigest()
        seen_external_ids.add(external_id)
        published_at = item.published_at or extract_first_date(item.body_text)
        normalized = _clean_lines(
            "\n".join(
                part
                for part in (
                    f"Title: {item.title}",
                    f"Published: {published_at}" if published_at else "",
                    f"URL: {item_url}" if item_url else "",
                    item.body_text,
                )
                if part
            )
        )
        parsed.append(
            FeedItem(
                external_id=external_id,
                title=item.title,
                url=item_url,
                published_at=published_at,
                text=normalized,
                sha256=_content_sha256(normalized, item.outbound_links),
                outbound_links=item.outbound_links,
                rendered_html=_feed_item_html(
                    item, item_url, published_at=published_at
                ),
            )
        )
    return title.strip() or fallback_title, tuple(parsed)


def discover_feed_urls(html: str, base_url: str) -> tuple[str, ...]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []
    for link in soup.find_all("link", href=True):
        if len(found) >= MAX_FEED_CANDIDATES:
            break
        rel = {str(value).lower() for value in (link.get("rel") or [])}
        content_type = str(link.get("type") or "").split(";", 1)[0].strip().lower()
        href = str(link.get("href") or "")
        is_feed_type = content_type in FEED_CONTENT_TYPES or "rss" in content_type or "atom" in content_type
        if "alternate" not in rel or not is_feed_type:
            continue
        try:
            candidate = normalize_blog_url(urljoin(base_url, href))
        except ValueError:
            continue
        if candidate not in found:
            found.append(candidate)
    return tuple(found)


def _scope_tokens(url: str) -> tuple[str, ...]:
    parsed = urlsplit(url)
    parts = [part.casefold() for part in parsed.path.split("/") if part]
    tokens: list[str] = []
    for index, part in enumerate(parts[:-1]):
        if part in SCOPED_PATH_MARKERS:
            tokens.append(parts[index + 1])
    for key, value in parse_qsl(parsed.query):
        if key.casefold() in SCOPED_QUERY_KEYS and value.strip():
            tokens.append(value.strip().casefold())
    return tuple(dict.fromkeys(tokens))


def _is_scoped_page(url: str) -> bool:
    return bool(_scope_tokens(url))


def feed_scope_score(page_url: str, feed_url: str) -> int:
    page = urlsplit(page_url)
    feed = urlsplit(feed_url)
    score = 20 if (page.hostname or "").lower() == (feed.hostname or "").lower() else -10
    page_tokens = _scope_tokens(page_url)
    haystack = f"{feed.path}?{feed.query}".casefold()
    if page_tokens:
        score += 35 * sum(token in haystack for token in page_tokens)
        if not any(token in haystack for token in page_tokens):
            score -= 20
    page_parts = [part for part in page.path.split("/") if part]
    feed_parts = [part for part in feed.path.split("/") if part]
    common = 0
    for left, right in zip(page_parts, feed_parts):
        if left.casefold() != right.casefold():
            break
        common += 1
    score += min(common * 5, 15)
    if feed.path.rstrip("/").casefold() in {"/feed", "/rss", "/atom", "/index.xml", "/feed.xml"}:
        score -= 10
    return score


def _conventional_feed_urls(page_url: str) -> tuple[str, ...]:
    parsed = urlsplit(page_url)
    base = urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
    path = parsed.path.rstrip("/")
    candidates = [
        urljoin(base, f"{path}/feed/") if path else urljoin(base, "/feed/"),
        urljoin(base, f"{path}/feed.xml") if path else urljoin(base, "/feed.xml"),
        urljoin(base, f"{path}/index.xml") if path else urljoin(base, "/index.xml"),
        urljoin(base, "/feed/"),
        urljoin(base, "/feed.xml"),
        urljoin(base, "/index.xml"),
        urljoin(base, "/rss.xml"),
    ]
    normalized: list[str] = []
    for value in candidates:
        try:
            candidate = normalize_blog_url(value)
        except ValueError:
            continue
        if candidate != normalize_blog_url(page_url) and candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def _headers(etag: object, last_modified: object) -> dict[str, str]:
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = str(etag)
    if last_modified:
        headers["If-Modified-Since"] = str(last_modified)
    return headers


def _checked_response(response: httpx.Response) -> httpx.Response:
    if response.status_code == 304:
        return response
    response.raise_for_status()
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError(f"Blog response exceeds {MAX_RESPONSE_BYTES} bytes")
    return response


def _article_score(element: object, *, url: str, base_url: str) -> int:
    parsed = urlsplit(url)
    base = urlsplit(base_url)
    score = 0
    if (parsed.hostname or "").lower() == (base.hostname or "").lower():
        score += 20
    path_parts = [part.casefold() for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2:
        score += 10
    if any(part in ARTICLE_PATH_WORDS for part in path_parts):
        score += 20
    if any(part in NON_ARTICLE_PATH_WORDS for part in path_parts):
        score -= 50
    try:
        if element.find_parent("article") is not None:  # type: ignore[attr-defined]
            score += 30
        elif element.find_parent("li") is not None:  # type: ignore[attr-defined]
            score += 5
        classes = " ".join(
            str(value)
            for ancestor in element.parents  # type: ignore[attr-defined]
            for value in (ancestor.get("class") or [])
        ).casefold()
        if re.search(r"\b(?:article|blog|entry|news|post|publication|story)\b", classes):
            score += 20
    except AttributeError:
        pass
    return score


def _index_url_hint(value: str) -> bool:
    path_parts = [part.casefold() for part in urlsplit(value).path.split("/") if part]
    if not path_parts:
        return True
    if any(part in SCOPED_PATH_MARKERS for part in path_parts):
        return True
    return path_parts[-1] in INDEX_PATH_WORDS


def _should_crawl_article_index(
    *,
    configured_mode: str,
    page_url: str,
    pattern: str | None,
    candidates: tuple[ArticleLink, ...],
) -> bool:
    if configured_mode in {"article_index", "publications"}:
        return True
    if not candidates:
        return False
    if configured_mode in {"feed", "video"}:
        return True
    if configured_mode != "auto":
        return False
    # Auto mode must be conservative: a normal article often contains related
    # links. Treat it as an index only when the URL itself looks like a listing
    # or the user supplied an explicit article URL pattern.
    return bool(pattern) or _index_url_hint(page_url)


def _article_candidates(
    root: object,
    *,
    base_url: str,
    pattern: str | None = None,
    allow_external: bool = False,
) -> tuple[ArticleLink, ...]:
    compiled = re.compile(pattern) if pattern else None
    base_host = (urlsplit(base_url).hostname or "").lower()
    candidates: list[tuple[int, ArticleLink]] = []
    seen: set[str] = set()
    try:
        elements = root.find_all("a", href=True)  # type: ignore[attr-defined]
    except AttributeError:
        return ()
    for element in elements:
        label = _link_label(element)
        if not label or label.casefold() in IGNORED_LINK_LABELS:
            continue
        url = _normalize_link_url(str(element.get("href") or ""), base_url)
        if url is None or url == normalize_blog_url(base_url) or _ignored_link(label, url, base_url):
            continue
        parsed = urlsplit(url)
        same_host = (parsed.hostname or "").lower() == base_host
        if not allow_external and not same_host:
            continue
        if compiled and not compiled.search(url):
            continue
        if not compiled and not same_host and not (
            "doi.org" in (parsed.hostname or "") or "arxiv.org" in (parsed.hostname or "")
        ):
            continue
        score = _article_score(element, url=url, base_url=base_url)
        if compiled:
            score += 50
        if not compiled and score < 40:
            continue
        if url in seen:
            continue
        seen.add(url)
        try:
            container = element.find_parent(["article", "li", "section", "div"])  # type: ignore[attr-defined]
            summary = re.sub(r"\s+", " ", container.get_text(" ", strip=True)).strip() if container else label
        except AttributeError:
            summary = label
        if len(summary) > 500:
            summary = summary[:500]
        candidates.append(
            (
                score,
                ArticleLink(
                    label,
                    url,
                    summary,
                    published_at=extract_first_date(summary),
                ),
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return tuple(item for _, item in candidates[:MAX_FEED_ITEMS])


def _fetch_feed_url(
    client: httpx.Client,
    *,
    feed_url: str,
    title: str,
    etag: object = None,
    last_modified: object = None,
    discovery_checked: bool = True,
    page_url: str | None = None,
    source_mode: str = "auto",
    diagnostics: tuple[str, ...] = (),
    row: object | None = None,
) -> BlogFetch:
    score = feed_scope_score(page_url, feed_url) if page_url else None
    if page_url and _is_scoped_page(page_url) and score is not None and score < 25:
        raise ValueError(f"Rejected out-of-scope feed {feed_url} (score {score})")
    response = _checked_response(
        client.get(
            feed_url,
            headers=_headers(etag, last_modified),
            timeout=BLOG_REQUEST_TIMEOUT,
        )
    )
    selected = diagnostics + (
        f"Selected feed {feed_url}" + (f" with scope score {score}" if score is not None else ""),
    )
    if response.status_code == 304:
        discovered, article_items, cursor_next, article_diagnostics = _follow_feed_articles(
            client,
            items=(),
            feed_url=feed_url,
            page_url=page_url,
            row=row,
            source_mode=source_mode,
        )
        return BlogFetch(
            method="rss",
            title=title,
            status_code=304,
            not_modified=not bool(article_items),
            etag=str(etag) if etag else None,
            last_modified=str(last_modified) if last_modified else None,
            feed_url=feed_url,
            discovery_checked=discovery_checked,
            discovered_articles=discovered,
            article_items=article_items,
            source_mode=source_mode,
            diagnostics=selected + article_diagnostics,
            article_cursor_next=cursor_next,
        )
    feed_title, items = parse_feed(response.text, feed_url, title)
    discovered, article_items, cursor_next, article_diagnostics = _follow_feed_articles(
        client,
        items=items,
        feed_url=feed_url,
        page_url=page_url,
        row=row,
        source_mode=source_mode,
    )
    return BlogFetch(
        method="rss",
        title=feed_title,
        status_code=response.status_code,
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        feed_url=feed_url,
        discovery_checked=discovery_checked,
        feed_items=items,
        discovered_articles=discovered,
        article_items=article_items,
        source_mode=source_mode,
        diagnostics=selected + article_diagnostics,
        article_cursor_next=cursor_next,
    )


def _fetch_article_pages(
    client: httpx.Client,
    *,
    candidates: tuple[ArticleLink, ...],
    known_urls: tuple[str, ...],
    cursor: int,
    selector: str | None,
) -> tuple[tuple[ArticleItem, ...], int, tuple[str, ...]]:
    current = {item.url: item for item in candidates}
    new_urls = [url for url in current if url not in known_urls]
    rotating = [url for url in known_urls if url not in new_urls]
    selected = new_urls[:MAX_ARTICLE_FETCHES]
    remaining = max(0, MAX_ARTICLE_FETCHES - len(selected))
    if rotating and remaining:
        start = cursor % len(rotating)
        selected.extend((rotating * 2)[start : start + remaining])
    selected = list(dict.fromkeys(selected))[:MAX_ARTICLE_FETCHES]
    next_cursor = (cursor + max(1, remaining)) % max(len(rotating), 1)
    items: list[ArticleItem] = []
    diagnostics: list[str] = []
    for url in selected:
        fallback = current.get(url)
        fallback_title = fallback.title if fallback else url
        try:
            response = _checked_response(
                client.get(url, timeout=BLOG_REQUEST_TIMEOUT)
            )
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            leading = response.text.lstrip().lower()
            if content_type in FEED_CONTENT_TYPES or leading.startswith(("<?xml", "<rss", "<feed", "<rdf")):
                raise ValueError("Article URL returned a feed instead of a web page")
            if content_type and not any(token in content_type for token in ("html", "text", "xml")):
                raise ValueError(f"Unsupported article content type: {content_type}")
            title, body, links, rendered_html = _extract_blog_document_full(
                response.text,
                fallback_title,
                url,
                selector=None,
            )
            published_at = _html_first_date(response.text) or extract_first_date(body) or (
                fallback.published_at if fallback is not None else None
            )
            text = f"Title: {title}\nURL: {url}\n\n{body}"[:MAX_TEXT_CHARS]
            items.append(
                ArticleItem(
                    external_id=hashlib.sha256(f"article-body\0{url}".encode("utf-8")).hexdigest(),
                    title=title,
                    url=url,
                    text=text,
                    sha256=_content_sha256(text, links),
                    outbound_links=links,
                    rendered_html=rendered_html,
                    published_at=published_at,
                )
            )
        except (httpx.HTTPError, ValueError) as exc:
            diagnostics.append(f"Could not fetch child article {url}: {exc}")
    return tuple(items), next_cursor, tuple(diagnostics)



def _feed_article_links(
    items: tuple[FeedItem, ...],
    *,
    feed_url: str,
    page_url: str | None,
) -> tuple[ArticleLink, ...]:
    """Return canonical article pages advertised by a feed.

    Only the item's primary URL is followed. Links embedded in the article body
    remain change metadata; recursively crawling references would turn a blog
    subscription into an unbounded web crawler.
    """
    excluded = {normalize_blog_url(feed_url)}
    if page_url:
        excluded.add(normalize_blog_url(page_url))
    found: list[ArticleLink] = []
    seen: set[str] = set()
    for item in items:
        if not item.url or item.url in excluded or item.url in seen:
            continue
        seen.add(item.url)
        summary_lines = [
            line
            for line in item.text.splitlines()
            if not line.startswith(("Title:", "Published:", "URL:", OUTBOUND_LINKS_HEADER, "- "))
        ]
        summary = re.sub(r"\s+", " ", " ".join(summary_lines)).strip()[:500]
        found.append(
            ArticleLink(
                item.title,
                item.url,
                summary,
                published_at=item.published_at or extract_first_date(summary),
            )
        )
    return tuple(found)


def _follow_feed_articles(
    client: httpx.Client,
    *,
    items: tuple[FeedItem, ...],
    feed_url: str,
    page_url: str | None,
    row: object | None,
    source_mode: str,
) -> tuple[tuple[ArticleLink, ...], tuple[ArticleItem, ...], int | None, tuple[str, ...]]:
    # A source configured as an article index or publication list may still
    # expose RSS.  Once a feed is selected, its primary item URL is the same
    # article target we would have followed from the HTML listing.
    if source_mode not in {"auto", "feed", "article_index", "publications"} or row is None:
        return (), (), None, ()
    discovered = _feed_article_links(items, feed_url=feed_url, page_url=page_url)
    known_urls = tuple(
        str(value) for value in (_row_value(row, "_known_article_urls", ()) or ())
    )
    article_items, cursor_next, diagnostics = _fetch_article_pages(
        client,
        candidates=discovered,
        known_urls=known_urls,
        cursor=int(_row_value(row, "article_cursor", 0) or 0),
        selector=None,
    )
    return discovered, article_items, cursor_next, diagnostics


def _github_fetch(client: httpx.Client, row: object) -> BlogFetch:
    url = str(_row_value(row, "url", ""))
    title = str(_row_value(row, "title", "") or url)
    parts = [part for part in urlsplit(url).path.split("/") if part]
    if len(parts) >= 2:
        for stream, feed_url in (
            ("releases", f"https://github.com/{parts[0]}/{parts[1]}/releases.atom"),
            ("commits", f"https://github.com/{parts[0]}/{parts[1]}/commits.atom"),
        ):
            try:
                fetched = _fetch_feed_url(
                    client,
                    feed_url=feed_url,
                    title=title,
                    source_mode="github",
                    diagnostics=(f"GitHub repository mode: {stream}",),
                    row=None,
                )
            except (httpx.HTTPError, ValueError):
                continue
            if fetched.feed_items or stream == "commits":
                return fetched
    if not parts:
        raise ValueError("GitHub source must identify an organization, user, or repository")
    name = parts[0]
    endpoints = (
        f"https://api.github.com/orgs/{name}/repos?per_page=100&type=public&sort=created&direction=desc",
        f"https://api.github.com/users/{name}/repos?per_page=100&type=owner&sort=created&direction=desc",
    )
    response: httpx.Response | None = None
    for endpoint in endpoints:
        candidate = client.get(
            endpoint,
            headers={"Accept": "application/vnd.github+json"},
            timeout=BLOG_REQUEST_TIMEOUT,
        )
        if candidate.status_code == 404:
            continue
        response = _checked_response(candidate)
        break
    if response is None:
        raise ValueError("GitHub organization or user was not found")
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("GitHub repository response was invalid")
    items: list[FeedItem] = []
    for repo in payload[:MAX_FEED_ITEMS]:
        if not isinstance(repo, dict) or not repo.get("html_url"):
            continue
        repo_url = _normalize_link_url(str(repo["html_url"]), url)
        if repo_url is None:
            continue
        repo_title = str(repo.get("full_name") or repo.get("name") or repo_url)
        text = _clean_lines(
            "\n".join(
                part
                for part in (
                    f"Title: {repo_title}",
                    f"URL: {repo_url}",
                    str(repo.get("description") or "No description"),
                    f"Language: {repo.get('language')}" if repo.get("language") else "",
                    f"Archived: {bool(repo.get('archived'))}",
                )
                if part
            )
        )
        links = tuple(
            link
            for value in (repo.get("homepage"),)
            if value and (link := _normalize_link_url(str(value), repo_url)) is not None
        )
        items.append(
            FeedItem(
                external_id=hashlib.sha256(f"github\0{repo_url}".encode("utf-8")).hexdigest(),
                title=repo_title,
                url=repo_url,
                text=text,
                sha256=_content_sha256(text, links),
                outbound_links=links,
                published_at=_published_at(str(repo.get("created_at") or "")),
            )
        )
    return BlogFetch(
        method="rss",
        title=title,
        status_code=response.status_code,
        feed_items=tuple(items),
        source_mode="github",
        discovery_checked=True,
        diagnostics=(f"GitHub repository API selected for {name}",),
    )


def _youtube_feed_url(html: str, page_url: str) -> str | None:
    for candidate in discover_feed_urls(html, page_url):
        if "youtube.com/feeds/videos.xml" in candidate:
            return candidate
    match = re.search(r'"channelId"\s*:\s*"(UC[\w-]+)"', html)
    if match:
        return f"https://www.youtube.com/feeds/videos.xml?channel_id={match.group(1)}"
    return None


def _fetch_page(
    client: httpx.Client,
    row: object,
    *,
    use_conditionals: bool,
    skip_feed_url: str | None = None,
) -> BlogFetch:
    url = str(_row_value(row, "url", ""))
    fallback = str(_row_value(row, "title", "") or url)
    mode = normalize_source_mode(_row_value(row, "source_mode", "auto"))
    selector = validate_content_selector(_row_value(row, "content_selector"))
    pattern = validate_article_url_pattern(_row_value(row, "article_url_pattern"))
    previous_stream = str(_row_value(row, "item_stream_mode", "") or "")
    conditional = (
        use_conditionals
        and mode in {"auto", "feed", "article", "docs"}
        and previous_stream not in {"article_index", "publications"}
    )
    response = _checked_response(
        client.get(
            url,
            headers=(
                _headers(_row_value(row, "etag"), _row_value(row, "last_modified"))
                if conditional
                else {}
            ),
            timeout=BLOG_REQUEST_TIMEOUT,
        )
    )
    if response.status_code == 304:
        return BlogFetch(
            method="snapshot",
            title=fallback,
            status_code=304,
            not_modified=True,
            discovery_checked=True,
            source_mode=mode,
        )

    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in FEED_CONTENT_TYPES and not any(
        token in content_type for token in ("html", "xml", "text")
    ):
        raise ValueError(f"Unsupported blog content type: {content_type or 'unknown'}")

    leading = response.text.lstrip().lower()
    if (
        content_type in FEED_CONTENT_TYPES
        or "xml" in content_type
        or leading.startswith(("<?xml", "<rss", "<feed", "<rdf"))
    ):
        try:
            feed_title, items = parse_feed(response.text, url, fallback)
        except ValueError:
            pass
        else:
            discovered, article_items, cursor_next, article_diagnostics = _follow_feed_articles(
                client,
                items=items,
                feed_url=url,
                page_url=None,
                row=row,
                source_mode=mode,
            )
            return BlogFetch(
                method="rss",
                title=feed_title,
                status_code=response.status_code,
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
                feed_url=url,
                discovery_checked=True,
                feed_items=items,
                discovered_articles=discovered,
                article_items=article_items,
                source_mode=mode,
                diagnostics=("Configured URL is a direct feed",) + article_diagnostics,
                article_cursor_next=cursor_next,
            )

    diagnostics: list[str] = []
    feed_candidates = list(discover_feed_urls(response.text, url))
    if mode in {"feed", "video"}:
        for candidate in _conventional_feed_urls(url):
            if candidate not in feed_candidates:
                feed_candidates.append(candidate)
    if mode == "video":
        youtube = _youtube_feed_url(response.text, url)
        if youtube and youtube not in feed_candidates:
            feed_candidates.insert(0, youtube)
    if mode in {"auto", "feed", "video"}:
        if len(feed_candidates) > MAX_FEED_CANDIDATES:
            diagnostics.append(
                f"Limited feed discovery to {MAX_FEED_CANDIDATES} candidates"
            )
        for candidate in feed_candidates[:MAX_FEED_CANDIDATES]:
            if candidate == skip_feed_url:
                continue
            score = feed_scope_score(url, candidate)
            if _is_scoped_page(url) and score < 25:
                diagnostics.append(f"Rejected out-of-scope feed {candidate} (score {score})")
                continue
            try:
                return _fetch_feed_url(
                    client,
                    feed_url=candidate,
                    title=fallback,
                    discovery_checked=True,
                    page_url=url,
                    source_mode=mode,
                    diagnostics=tuple(diagnostics),
                    row=row,
                )
            except (httpx.HTTPError, ValueError) as exc:
                diagnostics.append(f"Feed candidate failed {candidate}: {exc}")
        if mode in {"feed", "video"}:
            diagnostics.append("No valid scoped feed was available; using article-index fallback")

    title, text, outbound_links, rendered_html = _extract_blog_document_full(
        response.text, fallback, url, selector
    )
    effective_mode = mode
    discovered: tuple[ArticleLink, ...] = ()
    article_items: tuple[ArticleItem, ...] = ()
    cursor_next: int | None = None
    if mode in {"auto", "feed", "video", "article_index", "publications"}:
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
            element.decompose()
        root = _document_root(soup, selector)
        candidates = _article_candidates(
            root,
            base_url=url,
            pattern=pattern,
            allow_external=mode == "publications",
        )
        diagnostics.append(f"Discovered {len(candidates)} candidate content links")
        crawl_index = _should_crawl_article_index(
            configured_mode=mode,
            page_url=url,
            pattern=pattern,
            candidates=candidates,
        )
        if crawl_index:
            effective_mode = "publications" if mode == "publications" else "article_index"
            discovered = candidates
            known_urls = tuple(
                str(value)
                for value in (_row_value(row, "_known_article_urls", ()) or ())
            )
            article_items, cursor_next, child_diagnostics = _fetch_article_pages(
                client,
                candidates=discovered,
                known_urls=known_urls,
                cursor=int(_row_value(row, "article_cursor", 0) or 0),
                selector=selector,
            )
            diagnostics.extend(child_diagnostics)
        elif mode in {"auto", "feed", "video"}:
            effective_mode = "article"
            if candidates:
                diagnostics.append(
                    "Kept source as a direct article; candidate links lacked index-page evidence"
                )
    return BlogFetch(
        method="snapshot",
        title=title,
        status_code=response.status_code,
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        feed_url=feed_candidates[0] if feed_candidates else None,
        discovery_checked=True,
        snapshot_text=text,
        snapshot_sha256=_content_sha256(text, outbound_links),
        snapshot_links=outbound_links,
        snapshot_rendered_html=rendered_html,
        snapshot_published_at=_html_first_date(response.text) or extract_first_date(text),
        discovered_articles=discovered,
        article_items=article_items,
        source_mode=effective_mode,
        diagnostics=tuple(diagnostics),
        article_cursor_next=cursor_next,
    )


def fetch_blog(client: httpx.Client, row: object) -> BlogFetch:
    mode = normalize_source_mode(_row_value(row, "source_mode", "auto"))
    url = str(_row_value(row, "url", ""))
    if mode == "video" and _youtube_url(url):
        # Never fall back to the YouTube HTML page: unauthenticated requests can
        # return a consent document whose rotating URLs look like content changes.
        return _youtube_fetch(row)
    if mode == "github" or (mode == "auto" and urlsplit(url).hostname == "github.com"):
        return _github_fetch(client, row)
    feed_url = str(_row_value(row, "feed_url", "") or "").strip()
    title = str(_row_value(row, "title", "") or url)
    if feed_url and mode not in {"article_index", "article", "docs", "publications"}:
        try:
            return _fetch_feed_url(
                client,
                feed_url=feed_url,
                title=title,
                etag=_row_value(row, "feed_etag"),
                last_modified=_row_value(row, "feed_last_modified"),
                page_url=url,
                source_mode=mode,
                row=row,
            )
        except (httpx.HTTPError, ValueError):
            return _fetch_page(
                client,
                row,
                use_conditionals=True,
                skip_feed_url=feed_url,
            )

    return _fetch_page(
        client,
        row,
        use_conditionals=bool(_row_value(row, "feed_discovery_checked", False)),
    )


def _split_snapshot(value: str) -> tuple[list[str], dict[str, frozenset[str]] | None]:
    lines = value.splitlines()
    try:
        marker = lines.index(OUTBOUND_LINKS_HEADER)
    except ValueError:
        content = lines
        links: dict[str, frozenset[str]] | None = None
    else:
        content = lines[:marker]
        mapping: dict[str, set[str]] = {}
        for line in lines[marker + 1 :]:
            if not line.startswith("- ") or " -> " not in line:
                continue
            label, url = line[2:].rsplit(" -> ", 1)
            if url.strip():
                mapping.setdefault(label.strip().casefold(), set()).add(url.strip())
        links = {label: frozenset(urls) for label, urls in mapping.items()}

    direct_urls = {
        line.removeprefix("URL:").strip()
        for line in content
        if line.startswith("URL:") and line.removeprefix("URL:").strip()
    }
    if links is not None and direct_urls:
        links = {**links, "__direct__": frozenset(direct_urls)}
    elif links is None and direct_urls:
        links = {"__direct__": frozenset(direct_urls)}
    return content, links


def _semantic_line(value: str) -> str:
    line = re.sub(r"\s+", " ", value).strip().casefold()
    if not line:
        return ""
    if re.search(r"\b(?:profile picture|avatar)\b", line):
        return ""
    if re.fullmatch(r"[\d\s.,+\-–—%]+", line):
        return ""
    words = set(re.findall(r"[a-z]+", line))
    if words & VOLATILE_COUNT_WORDS:
        line = re.sub(r"\b\d[\d,._]*\b", "#", line)
    return " ".join(re.findall(r"[\w#]+", line, flags=re.UNICODE))


def _semantic_lines(lines: list[str]) -> list[str]:
    return [normalized for line in lines if (normalized := _semantic_line(line))]


def _changed_line_counts(
    old_lines: list[str], new_lines: list[str]
) -> tuple[int, int, int]:
    old_changed = 0
    new_changed = 0
    changed_chars = 0
    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_changed += old_end - old_start
        new_changed += new_end - new_start
        changed_chars += sum(len(line) for line in old_lines[old_start:old_end])
        changed_chars += sum(len(line) for line in new_lines[new_start:new_end])
    return old_changed, new_changed, changed_chars


def _link_sets(mapping: dict[str, frozenset[str]] | None) -> frozenset[str] | None:
    if mapping is None:
        return None
    return frozenset(url for urls in mapping.values() for url in urls)


def _meaningful_link_change(
    old_links: dict[str, frozenset[str]] | None,
    new_links: dict[str, frozenset[str]] | None,
    *,
    source_mode: str,
    content_changed: bool,
) -> bool:
    if old_links is None and new_links is not None:
        return False
    if old_links == new_links:
        return False
    old_set = _link_sets(old_links) or frozenset()
    new_set = _link_sets(new_links) or frozenset()
    if old_set == new_set:
        return False
    if old_links and new_links:
        for label in old_links.keys() & new_links.keys():
            if label != "__direct__" and old_links[label] != new_links[label]:
                return True
    changed_count = len(old_set ^ new_set)
    if source_mode == "article":
        return changed_count >= 1
    if source_mode == "docs":
        return content_changed or changed_count >= 3
    if source_mode in {"article_index", "publications"}:
        return content_changed and changed_count >= 1
    return content_changed or changed_count >= 2


def _is_meaningful_change(
    previous: str,
    current: str,
    *,
    source_mode: str,
) -> tuple[bool, bool]:
    old_content, old_links = _split_snapshot(previous)
    new_content, new_links = _split_snapshot(current)

    old_semantic = _semantic_lines(old_content)
    new_semantic = _semantic_lines(new_content)
    content_equal = old_semantic == new_semantic
    if not content_equal and Counter(old_semantic) == Counter(new_semantic):
        content_equal = True
    if not content_equal:
        old_tokens = Counter(token for line in old_semantic for token in line.split())
        new_tokens = Counter(token for line in new_semantic for token in line.split())
        if old_tokens == new_tokens:
            content_equal = True

    line_ratio = SequenceMatcher(None, old_semantic, new_semantic, autojunk=False).ratio()
    largest = max(len(old_semantic), len(new_semantic), 1)
    old_changed, new_changed, changed_chars = _changed_line_counts(
        old_semantic, new_semantic
    )
    changed_lines = max(old_changed, new_changed)
    changed_fraction = changed_lines / largest
    reset = line_ratio < 0.15 and largest > 20
    if content_equal:
        content_meaningful = False
    elif source_mode in {"feed", "video", "github"}:
        content_meaningful = True
    elif source_mode == "docs":
        content_meaningful = (
            reset
            or (changed_lines >= 4 and changed_fraction >= 0.20)
            or changed_lines >= 12
            or changed_chars >= 800
        )
    else:
        content_meaningful = (
            reset
            or (largest <= 5 and changed_lines >= 1)
            or (changed_lines >= 2 and changed_fraction >= 0.30)
            or changed_lines >= 8
            or changed_chars >= 400
        )
    link_meaningful = _meaningful_link_change(
        old_links,
        new_links,
        source_mode=source_mode,
        content_changed=not content_equal,
    )
    return content_meaningful or link_meaningful, reset


def compact_blog_diff(value: str) -> str:
    """Remove unchanged context and redundant file headers from a unified diff."""
    lines = value.splitlines()
    if not any(line.startswith("@@") for line in lines):
        return value
    compact = [
        line
        for line in lines
        if line.startswith("@@")
        or (line.startswith("-") and not line.startswith("---"))
        or (line.startswith("+") and not line.startswith("+++"))
    ]
    return "\n".join(compact)


def blog_diff(
    previous: str | None,
    current: str,
    *,
    source_mode: str = "auto",
) -> tuple[str, str]:
    if previous is None:
        return "baseline", ""
    mode = normalize_source_mode(source_mode)
    meaningful, semantic_reset = _is_meaningful_change(
        previous,
        current,
        source_mode=mode,
    )
    if not meaningful:
        return "baseline", ""
    old_lines = previous.splitlines()
    new_lines = current.splitlines()
    ratio = SequenceMatcher(None, old_lines, new_lines, autojunk=False).ratio()
    kind = (
        "reset"
        if semantic_reset
        or (ratio < 0.15 and max(len(old_lines), len(new_lines)) > 20)
        else "changed"
    )
    diff = compact_blog_diff(
        "\n".join(
            unified_diff(
                old_lines,
                new_lines,
                fromfile="previous",
                tofile="current",
                lineterm="",
                n=0,
            )
        )
    )
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... diff truncated ..."
        kind = "reset"
    return kind, diff
