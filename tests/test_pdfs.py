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


def test_validate_uploaded_pdf_accepts_readable_file_and_rejects_bad_files(tmp_path, monkeypatch):
    from paperwatch.pdfs import validate_pdf_file

    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF_BYTES)

    class Reader:
        is_encrypted = False
        pages = [object()]

    monkeypatch.setattr("paperwatch.pdfs.PdfReader", lambda *args, **kwargs: Reader())
    validate_pdf_file(path)

    path.write_bytes(b"not a complete pdf")
    with pytest.raises(PDFError, match="not a complete PDF"):
        validate_pdf_file(path)


def test_validate_uploaded_pdf_rejects_encryption_and_wraps_reader_errors(tmp_path, monkeypatch):
    from paperwatch.pdfs import validate_pdf_file

    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF_BYTES)

    class Encrypted:
        is_encrypted = True
        pages = []

        def decrypt(self, password):
            return 0

    monkeypatch.setattr("paperwatch.pdfs.PdfReader", lambda *args, **kwargs: Encrypted())
    with pytest.raises(PDFError, match="encrypted"):
        validate_pdf_file(path)

    monkeypatch.setattr(
        "paperwatch.pdfs.PdfReader",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("broken reader")),
    )
    with pytest.raises(PDFError, match="not a readable PDF"):
        validate_pdf_file(path)


def test_resolve_pdf_url_from_publisher_metadata():
    from paperwatch.pdfs import resolve_pdf_url

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/article":
            return httpx.Response(
                200,
                text='''<html><head><meta name="citation_pdf_url" content="/article.pdf"></head></html>''',
                headers={"content-type": "text/html"},
            )
        if request.url.path == "/article.pdf":
            return httpx.Response(200, content=PDF_BYTES, headers={"content-type": "application/pdf"})
        raise AssertionError(str(request.url))

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    resolved = resolve_pdf_url(
        title="A paper",
        doi=None,
        candidates=["https://academic.oup.com/article"],
        client=client,
    )
    assert resolved is not None
    assert resolved.url == "https://academic.oup.com/article.pdf"
    assert resolved.source == "publisher_page"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("https://arxiv.com/abs/2607.12345", "https://arxiv.org/pdf/2607.12345.pdf"),
        (
            "https://drive.google.com/file/d/abc123/view?usp=sharing",
            "https://drive.google.com/uc?export=download&id=abc123",
        ),
        ("https://dl.acm.org/doi/10.1145/123.456", "https://dl.acm.org/doi/pdf/10.1145/123.456"),
        (
            "https://onlinelibrary.wiley.com/doi/10.1002/example",
            "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/example",
        ),
        (
            "https://ieeexplore.ieee.org/document/1234567",
            "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=1234567",
        ),
    ],
)
def test_resolve_pdf_url_uses_known_direct_variants(source, expected):
    from paperwatch.pdfs import resolve_pdf_url

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url) == expected:
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(200, text="<html>No PDF</html>", headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    resolved = resolve_pdf_url(title="Known paper", doi=None, candidates=[source], client=client)
    assert resolved is not None
    assert resolved.url == expected
    assert seen[0] == expected


def test_resolve_pdf_url_uses_google_scholar_direct_link():
    from paperwatch.pdfs import resolve_pdf_url

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "scholar.google.com":
            assert '"Scholar paper"' in request.url.params["q"]
            return httpx.Response(
                200,
                text='''<html><div class="gs_or_ggsm"><a href="https://drive.google.com/file/d/file123/view">[PDF] drive.google.com</a></div></html>''',
                headers={"content-type": "text/html"},
            )
        if request.url.path == "/uc" and request.url.params["id"] == "file123":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    resolved = resolve_pdf_url(
        title="Scholar paper",
        doi="10.1000/scholar",
        candidates=[],
        client=client,
    )
    assert resolved is not None
    assert resolved.source == "google_scholar"
    assert resolved.url.startswith("https://drive.google.com/uc?")


def test_resolve_pdf_url_accepts_source_pdf_and_repository_links():
    from paperwatch.pdfs import resolve_pdf_url

    pages = {
        "www.sciencedirect.com": '<a href="/science/article/pii/S123/pdfft">Download PDF</a>',
        "irep.mbzuai.ac.ae": '<a href="/bitstream/123/1/paper">View PDF</a>',
        "philarchive.org": '<a href="/archive/author-paper.pdf">PDF</a>',
        "www.surgjournal.com": '<a aria-label="Source PDF" href="/article-pdf">Article</a>',
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path in {"/article", "/record/123", "/rec/ABC", "/paper"}:
            return httpx.Response(
                200,
                text=pages[request.url.host],
                headers={"content-type": "text/html"},
            )
        return httpx.Response(200, content=PDF_BYTES)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    for url in (
        "https://www.sciencedirect.com/article",
        "https://irep.mbzuai.ac.ae/record/123",
        "https://philarchive.org/rec/ABC",
        "https://www.surgjournal.com/paper",
    ):
        resolved = resolve_pdf_url(title="Repository paper", doi=None, candidates=[url], client=client)
        assert resolved is not None
        assert resolved.source == "publisher_page"


def test_resolve_pdf_url_returns_none_for_unusable_sources_and_blank_title():
    from paperwatch.pdfs import resolve_pdf_url

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b"not html and not pdf",
                headers={"content-type": "text/plain"},
            )
        )
    )
    assert resolve_pdf_url(title="", doi=None, candidates=["https://example.test/item"], client=client) is None


def test_known_pdf_variants_cover_query_and_unmatched_paths():
    from paperwatch.pdfs import _known_pdf_variants

    assert _known_pdf_variants("https://arxiv.org/help") == ()
    assert _known_pdf_variants("https://drive.google.com/open?id=query-id") == (
        "https://drive.google.com/uc?export=download&id=query-id",
    )
    assert _known_pdf_variants("https://drive.google.com/open") == ()
    assert _known_pdf_variants("https://example.test/article") == ()


def test_pdf_link_extraction_handles_metadata_links_redirects_and_invalid_values():
    from paperwatch.pdfs import _pdf_links

    html = '''
    <meta name="citation_pdf_url">
    <meta name="custom_pdf" content="/meta.pdf">
    <link type="application/pdf; charset=binary" href="/linked.pdf">
    <a href="mailto:test@example.com">PDF</a>
    <a href="/linked.pdf">Duplicate PDF</a>
    <a href="https://scholar.google.com/scholar_url?url=https%3A%2F%2Frepo.test%2Fpaper.pdf">[PDF]</a>
    <a href="https://scholar.google.com/other">[PDF]</a>
    '''
    links = _pdf_links(html, "https://publisher.test/article", scholar=True)
    assert links == (
        "https://publisher.test/meta.pdf",
        "https://publisher.test/linked.pdf",
        "https://repo.test/paper.pdf",
    )


def test_pdf_preview_skips_empty_chunks_and_caps_landing_pages():
    from paperwatch.pdfs import _preview

    class Response:
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://example.test/final"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b""
            yield b"x" * 2_000_100
            yield b"ignored"

    class Client:
        def stream(self, *args, **kwargs):
            return Response()

    final, content_type, body = _preview(Client(), "https://example.test/start", cancel_event=None)
    assert final.endswith("/final")
    assert content_type == "text/html"
    assert len(body) == 2_000_000


def test_resolve_pdf_url_handles_duplicate_sources_http_errors_and_non_html_scholar():
    from paperwatch.pdfs import resolve_pdf_url

    calls = 0

    def duplicate_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="plain", headers={"content-type": "text/plain"})

    duplicate_client = httpx.Client(transport=httpx.MockTransport(duplicate_handler))
    assert resolve_pdf_url(
        title="",
        doi=None,
        candidates=["https://example.test/a", "https://example.test/a", "file:///bad"],
        client=duplicate_client,
    ) is None
    assert calls == 1

    error_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    )
    assert resolve_pdf_url(
        title="Error paper",
        doi=None,
        candidates=["https://example.test/a"],
        client=error_client,
    ) is None

    plain_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"plain", headers={"content-type": "text/plain"})
        )
    )
    assert resolve_pdf_url(title="Plain paper", doi=None, candidates=[], client=plain_client) is None


def test_resolve_pdf_url_closes_owned_client(monkeypatch):
    from paperwatch.pdfs import resolve_pdf_url

    created = []

    class Response:
        headers = {"content-type": "application/pdf"}
        url = "https://example.test/paper.pdf"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield PDF_BYTES

    class Client:
        def __init__(self, **kwargs):
            self.closed = False
            created.append(self)

        def stream(self, *args, **kwargs):
            return Response()

        def close(self):
            self.closed = True

    monkeypatch.setattr("paperwatch.pdfs.httpx.Client", Client)
    resolved = resolve_pdf_url(
        title="Owned client",
        doi=None,
        candidates=["https://example.test/paper.pdf"],
    )
    assert resolved is not None
    assert created[0].closed
