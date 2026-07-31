from __future__ import annotations

import httpx
import pytest

from paperwatch.pdfs import PDFError, download_pdf, extract_pdf_text, safe_pdf_name


PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


def test_safe_pdf_name():
    name = safe_pdf_name("doi:10.1000/a/b")
    assert name.startswith("doi_10.1000_a_b-")
    assert name.endswith(".pdf")


def test_download_pdf_validates_and_skips_existing(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=PDF_BYTES, headers={"content-type": "application/pdf"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    destination = tmp_path / "paper.pdf"
    assert download_pdf("https://example.test/paper.pdf", destination, max_bytes=1000, client=client)
    assert destination.read_bytes().startswith(b"%PDF-")
    assert not download_pdf("https://example.test/paper.pdf", destination, max_bytes=1000, client=client)
    assert calls == 1


def test_download_pdf_rejects_non_pdf_and_oversize(tmp_path):
    non_pdf = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"<html>no</html>"))
    )
    with pytest.raises(PDFError, match="not a PDF"):
        download_pdf("https://example.test/file", tmp_path / "a.pdf", max_bytes=1000, client=non_pdf)

    oversized = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=PDF_BYTES, headers={"content-length": "9999"})
        )
    )
    with pytest.raises(PDFError, match="exceeds maximum"):
        download_pdf("https://example.test/file", tmp_path / "b.pdf", max_bytes=100, client=oversized)


def test_extract_pdf_text_normalizes_and_rejects_truncation(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF_BYTES)

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        is_encrypted = False
        pages = [Page("hello   world"), Page("second page")]

    monkeypatch.setattr("paperwatch.pdfs.PdfReader", lambda *args, **kwargs: Reader())
    assert extract_pdf_text(path, max_chars=100) == "hello world\n\nsecond page"
    with pytest.raises(PDFError, match="partial analysis"):
        extract_pdf_text(path, max_chars=15)


def test_extract_pdf_text_rejects_empty(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF_BYTES)

    class Reader:
        is_encrypted = False
        pages = []

    monkeypatch.setattr("paperwatch.pdfs.PdfReader", lambda *args, **kwargs: Reader())
    with pytest.raises(PDFError, match="no extractable text"):
        extract_pdf_text(path, max_chars=100)


def test_safe_pdf_names_are_bounded_and_collision_resistant():
    first = safe_pdf_name("doi:10/x:y")
    second = safe_pdf_name("doi:10/x?y")
    assert first != second
    assert len(safe_pdf_name("title:" + "x" * 1000)) < 130


def test_download_pdf_validates_url_limit_and_stream_size(tmp_path):
    with pytest.raises(PDFError, match="http or https"):
        download_pdf("file:///tmp/a.pdf", tmp_path / "a.pdf", max_bytes=10)
    with pytest.raises(PDFError, match="greater than zero"):
        download_pdf("https://example.test/a.pdf", tmp_path / "a.pdf", max_bytes=0)

    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES * 10))
    )
    destination = tmp_path / "large.pdf"
    with pytest.raises(PDFError, match="exceeds maximum"):
        download_pdf("https://example.test/a.pdf", destination, max_bytes=20, client=client)
    assert not destination.exists()
    assert not destination.with_suffix(".pdf.part").exists()


def test_download_pdf_wraps_http_and_invalid_content_length(tmp_path):
    for response, message in [
        (httpx.Response(500), "download failed"),
        (httpx.Response(200, content=PDF_BYTES, headers={"content-length": "bad"}), "download failed"),
    ]:
        client = httpx.Client(transport=httpx.MockTransport(lambda request, r=response: r))
        with pytest.raises(PDFError, match=message):
            download_pdf("https://example.test/a.pdf", tmp_path / "a.pdf", max_bytes=1000, client=client)


def test_pdf_header_may_appear_within_first_kilobyte(tmp_path):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"\n\n" + PDF_BYTES)
        )
    )
    path = tmp_path / "prefixed.pdf"
    assert download_pdf("https://example.test/a.pdf", path, max_bytes=1000, client=client)


def test_extract_pdf_text_validates_limits_file_and_encryption(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF_BYTES)
    with pytest.raises(PDFError, match="limit"):
        extract_pdf_text(path, max_chars=0)
    with pytest.raises(PDFError, match="Not a readable"):
        extract_pdf_text(tmp_path / "missing.pdf", max_chars=10)

    class Encrypted:
        is_encrypted = True
        pages = []

        def decrypt(self, password):
            return 0

    monkeypatch.setattr("paperwatch.pdfs.PdfReader", lambda *args, **kwargs: Encrypted())
    with pytest.raises(PDFError, match="encrypted"):
        extract_pdf_text(path, max_chars=10)


def test_extract_pdf_text_skips_empty_pages_and_wraps_reader_errors(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF_BYTES)

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        is_encrypted = False
        pages = [Page(None), Page(" useful ")]

    monkeypatch.setattr("paperwatch.pdfs.PdfReader", lambda *args, **kwargs: Reader())
    assert extract_pdf_text(path, max_chars=100) == "useful"

    monkeypatch.setattr("paperwatch.pdfs.PdfReader", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("bad")))
    with pytest.raises(PDFError, match="extraction failed"):
        extract_pdf_text(path, max_chars=100)


def test_download_pdf_owned_client_is_closed(tmp_path, monkeypatch):
    created = []

    class StreamResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield PDF_BYTES

    class FakeClient:
        def __init__(self, **kwargs):
            self.closed = False
            created.append(self)

        def stream(self, *args, **kwargs):
            return StreamResponse()

        def close(self):
            self.closed = True

    monkeypatch.setattr("paperwatch.pdfs.httpx.Client", FakeClient)
    assert download_pdf("https://example.test/a.pdf", tmp_path / "a.pdf", max_bytes=1000)
    assert created[0].closed


def test_incomplete_pdf_is_not_cached(tmp_path):
    from paperwatch.pdfs import is_complete_pdf

    path = tmp_path / "partial.pdf"
    path.write_bytes(b"%PDF-1.7\nmissing trailer")
    assert is_complete_pdf(path) is False
    path.write_bytes(PDF_BYTES)
    assert is_complete_pdf(path) is True


def test_pdf_completeness_handles_filesystem_errors(tmp_path, monkeypatch):
    from pathlib import Path
    from paperwatch.pdfs import is_complete_pdf

    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF_BYTES)
    original_stat = Path.stat

    def broken_stat(self, *args, **kwargs):
        if self == path:
            raise OSError("disk error")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", broken_stat)
    assert is_complete_pdf(path) is False


def test_download_pdf_skips_empty_chunks_and_enforces_incremental_limit(tmp_path):
    class StreamResponse:
        headers = {}

        def __init__(self, chunks):
            self._chunks = chunks

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield from self._chunks

    class FakeClient:
        def __init__(self, chunks):
            self._chunks = chunks

        def stream(self, *args, **kwargs):
            return StreamResponse(self._chunks)

    destination = tmp_path / "paper.pdf"
    assert download_pdf(
        "https://example.test/a.pdf",
        destination,
        max_bytes=1000,
        client=FakeClient([b"", PDF_BYTES]),
    )

    with pytest.raises(PDFError, match="exceeds maximum size"):
        download_pdf(
            "https://example.test/b.pdf",
            tmp_path / "too-large.pdf",
            max_bytes=10,
            client=FakeClient([b"%PDF-", b"x" * 20]),
        )
