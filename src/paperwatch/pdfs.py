from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from . import USER_AGENT
from .job_runtime import check_cancelled


class PDFError(RuntimeError):
    pass


SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
PDF_LINK_RE = re.compile(
    r"(?:\.pdf(?:$|[?#])|/pdf(?:direct)?(?:/|$)|/pdfft(?:/|$)|/download(?:/|$)|"
    r"/bitstream(?:/|$)|stamppdf|getpdf|article-pdf)",
    re.I,
)
PDF_LABEL_RE = re.compile(r"\b(?:source|download|view|full(?:\s+text)?|article)?\s*pdf\b", re.I)
PDF_META_KEYS = {"citation_pdf_url", "eprints.document_url", "pdf_url", "wkhealth_pdf_url"}


@dataclass(frozen=True, slots=True)
class ResolvedPDF:
    url: str
    source: str


def safe_pdf_name(identity_key: str) -> str:
    cleaned = SAFE_RE.sub("_", identity_key).strip("._") or "paper"
    digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:100]}-{digest}.pdf"


def is_complete_pdf(path: Path) -> bool:
    """Return whether a file has both a PDF header and a final EOF marker."""
    try:
        if not path.exists() or path.stat().st_size < 10:
            return False
        with path.open("rb") as handle:
            header = handle.read(1024)
            handle.seek(max(0, path.stat().st_size - 65536))
            trailer = handle.read()
        return b"%PDF-" in header and b"%%EOF" in trailer
    except OSError:
        return False


def _looks_like_pdf(path: Path) -> bool:
    return is_complete_pdf(path)


def validate_pdf_file(path: Path) -> None:
    """Raise ``PDFError`` when a local file is not a readable, unencrypted PDF."""
    if not is_complete_pdf(path):
        raise PDFError("Uploaded file is not a complete PDF")
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PDFError("Uploaded PDF is encrypted")
        # Force pypdf to parse the page tree now rather than failing during analysis.
        len(reader.pages)
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError(f"Uploaded file is not a readable PDF: {exc}") from exc


def _known_pdf_variants(url: str) -> tuple[str, ...]:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path
    variants: list[str] = []
    if host in {"arxiv.org", "www.arxiv.org", "arxiv.com", "www.arxiv.com"}:
        match = re.match(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?$", path, flags=re.I)
        if match:
            variants.append(f"https://arxiv.org/pdf/{match.group(1)}.pdf")
    elif host == "drive.google.com":
        match = re.search(r"/file/d/([^/]+)", path)
        file_id = match.group(1) if match else parse_qs(parsed.query).get("id", [None])[0]
        if file_id:
            variants.append(
                f"https://drive.google.com/uc?{urlencode({'export': 'download', 'id': file_id})}"
            )
    elif host == "dl.acm.org" and path.startswith("/doi/") and not path.startswith("/doi/pdf"):
        variants.append(f"https://dl.acm.org/doi/pdf/{path.removeprefix('/doi/').lstrip('/')}")
    elif host.endswith("wiley.com") and path.startswith("/doi/") and "/pdf" not in path:
        variants.append(
            f"{parsed.scheme}://{parsed.netloc}/doi/pdfdirect/{path.removeprefix('/doi/').lstrip('/')}"
        )
    elif host.endswith("ieee.org") and (match := re.search(r"/document/(\d+)", path)):
        variants.append(
            "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?"
            + urlencode({"tp": "", "arnumber": match.group(1)})
        )
    return tuple(candidate for candidate in variants if candidate != url)


def _pdf_links(html: str, base_url: str, *, scholar: bool = False) -> tuple[str, ...]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []

    def add(raw: str | None) -> None:
        if not raw:
            return
        candidate = urljoin(base_url, raw.strip())
        parsed = urlsplit(candidate)
        if parsed.hostname and parsed.hostname.endswith("google.com") and parsed.path == "/scholar_url":
            candidate = parse_qs(parsed.query).get("url", [candidate])[0]
            parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return
        if scholar and parsed.hostname in {"scholar.google.com", "www.google.com"}:
            return
        if candidate not in found:
            found.append(candidate)

    for meta in soup.find_all("meta"):
        key = str(meta.get("name") or meta.get("property") or "").lower()
        content = str(meta.get("content") or "")
        if key in PDF_META_KEYS or ("pdf" in key and PDF_LINK_RE.search(content)):
            add(content)
    for link in soup.find_all("link", href=True):
        if str(link.get("type") or "").split(";", 1)[0].lower() == "application/pdf":
            add(str(link["href"]))
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        label = " ".join(
            [anchor.get_text(" ", strip=True), str(anchor.get("title") or ""), str(anchor.get("aria-label") or "")]
        )
        parent_classes = " ".join(str(value) for value in (anchor.parent.get("class", []) if anchor.parent else []))
        if PDF_LINK_RE.search(href) or PDF_LABEL_RE.search(label) or (
            scholar and ("gs_or_ggsm" in parent_classes or "[pdf]" in label.lower())
        ):
            add(href)
    return tuple(found)


def _preview(
    client: httpx.Client,
    url: str,
    *,
    cancel_event: threading.Event | None,
) -> tuple[str, str, bytes]:
    check_cancelled(cancel_event)
    with client.stream(
        "GET",
        url,
        headers={"Accept": "application/pdf,text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"},
    ) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            check_cancelled(cancel_event)
            if chunk:
                chunks.append(chunk[: max(0, 2_000_000 - size)])
                size += len(chunks[-1])
            if size >= 2_000_000:
                break
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        return str(response.url), content_type, b"".join(chunks)


def resolve_pdf_url(
    *,
    title: str,
    doi: str | None,
    candidates: list[str] | tuple[str, ...],
    client: httpx.Client | None = None,
    timeout_seconds: float = 30,
    cancel_event: threading.Event | None = None,
) -> ResolvedPDF | None:
    """Resolve publisher and Scholar links to a URL that starts with a PDF header."""
    owns_client = client is None
    http = client or httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    queue: list[tuple[str, str, bool]] = []
    for candidate in candidates:
        if not str(candidate).startswith(("http://", "https://")):
            continue
        queue.extend((variant, "source_page", False) for variant in _known_pdf_variants(candidate))
        queue.append((candidate, "source_page", False))
    seen: set[str] = set()

    def inspect(url: str, source: str, nested: bool) -> ResolvedPDF | None:
        if url in seen:
            return None
        seen.add(url)
        try:
            final_url, content_type, body = _preview(http, url, cancel_event=cancel_event)
        except (httpx.HTTPError, ValueError):
            return None
        if b"%PDF-" in body[:1024]:
            return ResolvedPDF(final_url, source)
        if not nested and ("html" in content_type or body.lstrip().startswith(b"<")):
            for link in _pdf_links(body.decode("utf-8", errors="replace"), final_url):
                queue.append((link, "publisher_page", True))
        return None

    try:
        while queue and len(seen) < 24:
            url, source, nested = queue.pop(0)
            if resolved := inspect(url, source, nested):
                return resolved
        if not title.strip():
            return None
        query = f'"{title.strip()}"' + (f' "{doi.strip()}"' if doi else "")
        scholar_url = "https://scholar.google.com/scholar?" + urlencode({"q": query})
        try:
            final_url, content_type, body = _preview(http, scholar_url, cancel_event=cancel_event)
        except (httpx.HTTPError, ValueError):
            return None
        if "html" not in content_type and not body.lstrip().startswith(b"<"):
            return None
        for link in _pdf_links(body.decode("utf-8", errors="replace"), final_url, scholar=True):
            for candidate in (*_known_pdf_variants(link), link):
                if resolved := inspect(candidate, "google_scholar", True):
                    return resolved
        return None
    finally:
        if owns_client:
            http.close()


def download_pdf(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    client: httpx.Client | None = None,
    timeout_seconds: float = 60,
    cancel_event: threading.Event | None = None,
) -> bool:
    if not url.lower().startswith(("http://", "https://")):
        raise PDFError("PDF URL must use http or https")
    if max_bytes <= 0:
        raise PDFError("PDF maximum size must be greater than zero")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _looks_like_pdf(destination):
        return False

    owns_client = client is None
    http = client or httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    total = 0
    try:
        check_cancelled(cancel_event)
        with http.stream("GET", url) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length and int(length) > max_bytes:
                raise PDFError(f"PDF exceeds maximum size of {max_bytes} bytes")
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
                    check_cancelled(cancel_event)
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise PDFError(f"PDF exceeds maximum size of {max_bytes} bytes")
                    handle.write(chunk)
        if not _looks_like_pdf(temporary):
            raise PDFError("Downloaded content is not a PDF")
        temporary.replace(destination)
        return True
    except PDFError:
        raise
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise PDFError(f"PDF download failed: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        if owns_client:
            http.close()


def extract_pdf_text(path: Path, *, max_chars: int) -> str:
    if max_chars <= 0:
        raise PDFError("PDF text limit must be greater than zero")
    if not _looks_like_pdf(path):
        raise PDFError(f"Not a readable PDF: {path}")
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PDFError("PDF is encrypted")
        parts: list[str] = []
        count = 0
        for page in reader.pages:
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if not text:
                continue
            added = len(text) + (2 if parts else 0)
            if count + added > max_chars:
                raise PDFError(
                    f"PDF text exceeds fetch.max_pdf_chars={max_chars}; "
                    "increase the limit to avoid partial analysis"
                )
            parts.append(text)
            count += added
        output = "\n\n".join(parts).strip()
        if not output:
            raise PDFError("PDF contains no extractable text")
        return output
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError(f"PDF text extraction failed: {exc}") from exc
