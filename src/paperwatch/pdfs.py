from __future__ import annotations

import hashlib
import re
from pathlib import Path

import httpx

from . import USER_AGENT
from pypdf import PdfReader


class PDFError(RuntimeError):
    pass


SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_pdf_name(identity_key: str) -> str:
    cleaned = SAFE_RE.sub("_", identity_key).strip("._") or "paper"
    digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:100]}-{digest}.pdf"


def _looks_like_pdf(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 5:
            return False
        with path.open("rb") as handle:
            return b"%PDF-" in handle.read(1024)
    except OSError:
        return False


def download_pdf(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    client: httpx.Client | None = None,
    timeout_seconds: float = 60,
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
        with http.stream("GET", url) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length and int(length) > max_bytes:
                raise PDFError(f"PDF exceeds maximum size of {max_bytes} bytes")
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes():
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
