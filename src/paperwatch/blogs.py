from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher, unified_diff
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup


MAX_RESPONSE_BYTES = 5_000_000
MAX_TEXT_CHARS = 1_000_000
MAX_DIFF_CHARS = 200_000
MAX_FEED_ITEMS = 200
FEED_CONTENT_TYPES = {
    "application/atom+xml",
    "application/rdf+xml",
    "application/rss+xml",
}


@dataclass(frozen=True, slots=True)
class FeedItem:
    external_id: str
    title: str
    url: str | None
    published_at: str | None
    text: str
    sha256: str


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
    feed_items: tuple[FeedItem, ...] = ()


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


def extract_blog_text(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
        element.decompose()
    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = _clean_lines(root.get_text("\n", strip=True))
    return (title.strip() or fallback_title, text)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ElementTree.Element, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in element:
        if _local_name(child.tag) in wanted:
            return "".join(child.itertext()).strip()
    return ""


def _html_text(value: str) -> str:
    soup = BeautifulSoup(value, "html.parser")
    return _clean_lines(soup.get_text(" ", strip=True))


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


def _feed_item(
    element: ElementTree.Element,
    *,
    base_url: str,
    atom: bool,
) -> FeedItem | None:
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

    item_url = urljoin(base_url, link) if link else None
    published_at = _published_at(published)
    try:
        body_text = _html_text(body) if body else _clean_lines(title)
    except ValueError:
        body_text = _clean_lines(title)
    normalized = _clean_lines(
        "\n".join(
            part
            for part in (
                f"Title: {title}",
                f"Published: {published_at}" if published_at else "",
                f"URL: {item_url}" if item_url else "",
                body_text,
            )
            if part
        )
    )
    raw_identity = identity.strip() or item_url or f"{title}\n{published_at or ''}\n{body_text}"
    external_id = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()
    return FeedItem(
        external_id=external_id,
        title=title.strip(),
        url=item_url,
        published_at=published_at,
        text=normalized,
        sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


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

    parsed = [
        item
        for element in elements[:MAX_FEED_ITEMS]
        if (item := _feed_item(element, base_url=base_url, atom=atom)) is not None
    ]
    return title.strip() or fallback_title, tuple(parsed)


def discover_feed_urls(html: str, base_url: str) -> tuple[str, ...]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []
    for link in soup.find_all("link", href=True):
        rel = {str(value).lower() for value in (link.get("rel") or [])}
        content_type = str(link.get("type") or "").split(";", 1)[0].strip().lower()
        if "alternate" not in rel or content_type not in FEED_CONTENT_TYPES:
            continue
        try:
            candidate = normalize_blog_url(urljoin(base_url, str(link["href"])))
        except ValueError:
            continue
        if candidate not in found:
            found.append(candidate)
    return tuple(found)


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


def _fetch_feed_url(
    client: httpx.Client,
    *,
    feed_url: str,
    title: str,
    etag: object = None,
    last_modified: object = None,
    discovery_checked: bool = True,
) -> BlogFetch:
    response = _checked_response(
        client.get(feed_url, headers=_headers(etag, last_modified))
    )
    if response.status_code == 304:
        return BlogFetch(
            method="rss",
            title=title,
            status_code=304,
            not_modified=True,
            feed_url=feed_url,
            discovery_checked=discovery_checked,
        )
    feed_title, items = parse_feed(response.text, feed_url, title)
    return BlogFetch(
        method="rss",
        title=feed_title,
        status_code=response.status_code,
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        feed_url=feed_url,
        discovery_checked=discovery_checked,
        feed_items=items,
    )


def _fetch_page(
    client: httpx.Client,
    row: object,
    *,
    use_conditionals: bool,
    skip_feed_url: str | None = None,
) -> BlogFetch:
    url = str(_row_value(row, "url", ""))
    fallback = str(_row_value(row, "title", "") or url)
    response = _checked_response(
        client.get(
            url,
            headers=(
                _headers(_row_value(row, "etag"), _row_value(row, "last_modified"))
                if use_conditionals
                else {}
            ),
        )
    )
    if response.status_code == 304:
        return BlogFetch(
            method="snapshot",
            title=fallback,
            status_code=304,
            not_modified=True,
            discovery_checked=True,
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
            return BlogFetch(
                method="rss",
                title=feed_title,
                status_code=response.status_code,
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
                feed_url=url,
                discovery_checked=True,
                feed_items=items,
            )

    discovered_feed_url: str | None = None
    for candidate in discover_feed_urls(response.text, url):
        if candidate == skip_feed_url:
            continue
        discovered_feed_url = discovered_feed_url or candidate
        try:
            return _fetch_feed_url(
                client,
                feed_url=candidate,
                title=fallback,
                discovery_checked=True,
            )
        except (httpx.HTTPError, ValueError):
            continue

    title, text = extract_blog_text(response.text, fallback)
    return BlogFetch(
        method="snapshot",
        title=title,
        status_code=response.status_code,
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        feed_url=discovered_feed_url,
        discovery_checked=True,
        snapshot_text=text,
        snapshot_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def fetch_blog(client: httpx.Client, row: object) -> BlogFetch:
    feed_url = str(_row_value(row, "feed_url", "") or "").strip()
    title = str(_row_value(row, "title", "") or _row_value(row, "url", ""))
    if feed_url:
        try:
            return _fetch_feed_url(
                client,
                feed_url=feed_url,
                title=title,
                etag=_row_value(row, "feed_etag"),
                last_modified=_row_value(row, "feed_last_modified"),
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


def blog_diff(previous: str | None, current: str) -> tuple[str, str]:
    if previous is None:
        return "baseline", ""
    old_lines = previous.splitlines()
    new_lines = current.splitlines()
    ratio = SequenceMatcher(None, old_lines, new_lines, autojunk=False).ratio()
    kind = "reset" if ratio < 0.15 and max(len(old_lines), len(new_lines)) > 20 else "changed"
    diff = "\n".join(
        unified_diff(old_lines, new_lines, fromfile="previous", tofile="current", lineterm="")
    )
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... diff truncated ..."
        kind = "reset"
    return kind, diff
