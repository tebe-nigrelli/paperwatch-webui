from __future__ import annotations

import httpx
import pytest

from paperwatch.sources.arxiv import ArxivClient, ArxivError


def test_arxiv_by_id_parses_atom(fixture_dir):
    xml = (fixture_dir / "arxiv_feed.xml").read_bytes()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=xml, headers={"content-type": "application/atom+xml"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    api = ArxivClient(base_url="https://arxiv.test/query", client=client)
    entry = api.by_id("2607.12345")
    assert entry is not None
    assert entry.arxiv_id == "2607.12345"
    assert entry.abstract.startswith("We introduce")
    assert entry.pdf_url.endswith("2607.12345v2")
    assert entry.doi == "10.1000/example"
    assert seen[0].url.params["id_list"] == "2607.12345"


def test_arxiv_title_matching_uses_similarity_and_author(fixture_dir):
    xml = (fixture_dir / "arxiv_feed.xml").read_bytes()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=xml)))
    api = ArxivClient(
        base_url="https://arxiv.test/query",
        title_match_threshold=0.9,
        min_interval_seconds=0,
        client=client,
    )
    assert api.by_title("Minimal Agents for Reliable Machine Learning", "Ada Lovelace") is not None
    assert api.by_title("A Completely Different Paper", "Ada Lovelace") is None


def test_arxiv_invalid_xml_is_reported():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not xml"))
    )
    api = ArxivClient(base_url="https://arxiv.test/query", client=client)
    with pytest.raises(ArxivError, match="invalid Atom"):
        api.by_id("2607.12345")


def test_arxiv_rate_limiter_waits_between_requests(fixture_dir):
    xml = (fixture_dir / "arxiv_feed.xml").read_bytes()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=xml)))
    values = iter([0.0, 0.0, 1.0, 4.0])
    waits = []
    api = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=3,
        client=client,
        clock=lambda: next(values),
        sleeper=waits.append,
    )
    api.by_id("2607.12345")
    api.by_id("2607.12345")
    assert waits == [2.0]


def test_arxiv_exact_title_with_wrong_author_is_rejected(fixture_dir):
    xml = (fixture_dir / "arxiv_feed.xml").read_bytes()
    api = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=xml))),
    )
    assert api.by_title("Minimal Agents for Reliable Machine Learning", "Different Surname") is None


def test_arxiv_by_id_validation_empty_and_mismatch(fixture_dir):
    with pytest.raises(ArxivError, match="Invalid arXiv id"):
        ArxivClient(client=httpx.Client()).by_id("bad")

    empty = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>',
                )
            )
        ),
    )
    assert empty.by_id("2607.12345") is None

    xml = (fixture_dir / "arxiv_feed.xml").read_bytes()
    mismatch = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=xml))),
    )
    assert mismatch.by_id("2607.99999") is None


def test_arxiv_retries_status_and_transport(fixture_dir):
    xml = (fixture_dir / "arxiv_feed.xml").read_bytes()
    calls = 0
    waits = []

    def unavailable_once(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"retry-after": "0.2"})
        return httpx.Response(200, content=xml)

    api = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(unavailable_once)),
        sleeper=waits.append,
    )
    assert api.by_id("2607.12345")
    assert waits == [0.2]

    calls = 0
    waits.clear()

    def offline_once(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, content=xml)

    api = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(offline_once)),
        sleeper=waits.append,
    )
    assert api.by_id("2607.12345")
    assert waits == [1.0]


def test_arxiv_reports_exhausted_http_and_transport_errors():
    api = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(400))),
    )
    with pytest.raises(ArxivError, match="request failed"):
        api.by_id("2607.12345")

    def offline(request):
        raise httpx.ReadError("offline", request=request)

    api = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(offline)),
    )
    with pytest.raises(ArxivError, match="offline"):
        api.by_id("2607.12345")


def test_arxiv_missing_entry_metadata_and_title_validation():
    broken = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2607.12345</id></entry></feed>'''
    api = ArxivClient(
        base_url="https://arxiv.test/query",
        min_interval_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=broken))),
    )
    with pytest.raises(ArxivError, match="missing required metadata"):
        api.by_id("2607.12345")

    with pytest.raises(ArxivError, match="title cannot be empty"):
        api.by_title("  ")


def test_arxiv_helpers_and_owned_client_close():
    assert ArxivClient._quoted_title('A "quoted" \\ title') == "A  quoted    title"
    assert ArxivClient._author_matches(None, [])
    assert ArxivClient._author_matches("   ", [])
    assert not ArxivClient._author_matches("Ada Lovelace", ["Alan Turing"])

    api = ArxivClient()
    assert not api.client.is_closed
    api.close()
    assert api.client.is_closed
