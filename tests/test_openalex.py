from __future__ import annotations

import httpx
import pytest

from paperwatch.sources.openalex import OpenAlexClient, OpenAlexError


def test_parse_author_and_work(openalex_author_json, openalex_work_json):
    author = OpenAlexClient.parse_author(openalex_author_json)
    paper = OpenAlexClient.parse_work(openalex_work_json)
    assert author.openalex_id == "A123"
    assert author.institution == "Analytical Engine Institute"
    assert paper.openalex_id == "W456"
    assert paper.doi == "10.1000/example"
    assert paper.arxiv_id == "2607.12345"
    assert paper.abstract == "We introduce a compact agent architecture."
    assert paper.pdf_url.endswith("2607.12345.pdf")
    assert paper.venue == "Journal of Minimal ML"
    assert len(paper.authors) == 2
    assert [topic.name for topic in paper.topics] == ["Machine Learning", "Autonomous Agents"]
    assert paper.topics[0].is_primary is True
    assert paper.topics[0].field_name == "Computer Science"


def test_search_and_get_author_use_api_key(openalex_author_json):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/authors/A123"):
            return httpx.Response(200, json=openalex_author_json)
        return httpx.Response(200, json={"results": [openalex_author_json]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    api = OpenAlexClient(api_key="secret", base_url="https://openalex.test", client=client)
    results = api.search_authors("Ada Lovelace", limit=3)
    author = api.get_author("https://openalex.org/A123")
    assert results[0].name == "Ada Lovelace"
    assert author.openalex_id == "A123"
    assert all(request.url.params["api_key"] == "secret" for request in seen)
    assert seen[0].url.params["search"] == "Ada Lovelace"


def test_recent_works_pages_and_passes_date(openalex_work_json):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={"results": [openalex_work_json], "meta": {"next_cursor": "next"}},
            )
        return httpx.Response(200, json={"results": [], "meta": {"next_cursor": None}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    api = OpenAlexClient(api_key="secret", base_url="https://openalex.test", client=client)
    papers = api.recent_works("A123", since="2026-07-01", max_results=5)
    assert len(papers) == 1
    assert "author.id:A123" in requests[0].url.params["filter"]
    assert "from_publication_date:2026-07-01" in requests[0].url.params["filter"]
    assert requests[0].url.params["cursor"] == "*"
    assert requests[1].url.params["cursor"] == "next"


def test_recent_works_skips_sparse_records_without_titles(openalex_work_json):
    sparse = {"id": "https://openalex.org/WEMPTY", "display_name": None}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [sparse, openalex_work_json], "meta": {}},
        )

    api = OpenAlexClient(
        api_key="secret",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    papers = api.recent_works("A123", since="2026-01-01", max_results=10)
    assert [paper.openalex_id for paper in papers] == ["W456"]


def test_missing_api_key_fails_before_network():
    api = OpenAlexClient(api_key="", base_url="https://openalex.test", client=httpx.Client())
    with pytest.raises(OpenAlexError, match="API key missing"):
        api.search_authors("Ada")


def test_openalex_current_sort_syntax(openalex_work_json):
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json={"results": [openalex_work_json], "meta": {}})

    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    api.recent_works("a123", since="2026-01-01", max_results=1)
    assert seen["sort"] == "publication_date:desc"
    assert "author.id:A123" in seen["filter"]


def test_openalex_retries_transient_status_and_transport(openalex_author_json):
    calls = 0
    waits = []

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0.25"})
        return httpx.Response(200, json={"results": [openalex_author_json]})

    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=waits.append,
    )
    assert api.search_authors("Ada")[0].name == "Ada Lovelace"
    assert waits == [0.25]

    calls = 0
    waits.clear()

    def transport_then_ok(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json={"results": [openalex_author_json]})

    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(transport_then_ok)),
        sleeper=waits.append,
    )
    assert api.search_authors("Ada")
    assert waits == [1.0]


def test_openalex_reports_http_json_and_payload_errors():
    cases = [
        (lambda request: httpx.Response(400), "request failed"),
        (lambda request: httpx.Response(200, content=b"not-json"), "invalid JSON"),
        (lambda request: httpx.Response(200, json=[]), "unexpected payload"),
    ]
    for handler, message in cases:
        api = OpenAlexClient(
            api_key="key",
            base_url="https://openalex.test",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        with pytest.raises(OpenAlexError, match=message):
            api.search_authors("Ada")


def test_openalex_exhausted_transport_and_retry_after_fallback():
    waits = []

    def unavailable(request):
        return httpx.Response(503, headers={"retry-after": "not-a-number"})

    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(unavailable)),
        sleeper=waits.append,
        max_retries=1,
    )
    with pytest.raises(OpenAlexError, match="request failed"):
        api.search_authors("Ada")
    assert waits == [1.0]

    def offline(request):
        raise httpx.ReadError("offline", request=request)

    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(offline)),
        sleeper=lambda value: None,
        max_retries=0,
    )
    with pytest.raises(OpenAlexError, match="offline"):
        api.search_authors("Ada")


def test_author_parsing_fallbacks_and_validation(openalex_author_json):
    item = dict(openalex_author_json)
    item["last_known_institutions"] = []
    item["affiliations"] = [{"institution": {"display_name": "Fallback University"}}]
    assert OpenAlexClient.parse_author(item).institution == "Fallback University"

    item["affiliations"] = ["bad"]
    assert OpenAlexClient.parse_author(item).institution is None
    with pytest.raises(OpenAlexError, match="missing id or name"):
        OpenAlexClient.parse_author({})


def test_author_search_and_work_payload_validation(openalex_author_json):
    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": {}}))),
    )
    with pytest.raises(OpenAlexError, match="cannot be empty"):
        api.search_authors("  ")
    with pytest.raises(OpenAlexError, match="limit must be positive"):
        api.search_authors("Ada", limit=0)
    with pytest.raises(OpenAlexError, match="malformed"):
        api.search_authors("Ada")
    with pytest.raises(OpenAlexError, match="Invalid OpenAlex"):
        api.get_author("")
    with pytest.raises(OpenAlexError, match="max_results"):
        api.recent_works("A123", since="2026-01-01", max_results=0)


def test_work_parsing_location_fallbacks_and_malformed_items(openalex_work_json):
    from copy import deepcopy

    work = deepcopy(openalex_work_json)
    work["best_oa_location"] = None
    work["primary_location"]["pdf_url"] = "https://publisher.test/paper.pdf"
    paper = OpenAlexClient.parse_work(work)
    assert paper.pdf_url == "https://publisher.test/paper.pdf"
    assert paper.pdf_source == "Journal of Minimal ML"

    work = deepcopy(openalex_work_json)
    work["best_oa_location"] = None
    work["primary_location"]["pdf_url"] = None
    work["locations"] = ["bad", {"pdf_url": "https://repo.test/p.pdf", "source": {}}]
    assert OpenAlexClient.parse_work(work).pdf_source == "openalex_location"

    work = deepcopy(openalex_work_json)
    work["authorships"] = ["bad", {"author": "bad"}, {"author": {"display_name": "  "}}]
    work["primary_location"] = "bad"
    work["best_oa_location"] = "bad"
    parsed = OpenAlexClient.parse_work(work)
    assert parsed.authors == []

    with pytest.raises(OpenAlexError, match="missing title"):
        OpenAlexClient.parse_work({})


def test_recent_works_rejects_malformed_results():
    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"results": {}, "meta": {}})
            )
        ),
    )
    with pytest.raises(OpenAlexError, match="work results are malformed"):
        api.recent_works("A123", since="2026-01-01")


def test_owned_client_can_be_closed():
    api = OpenAlexClient(api_key="key", base_url="https://openalex.test")
    assert not api.client.is_closed
    api.close()
    assert api.client.is_closed


def test_search_works_by_title(openalex_work_json):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json={"results": [None, openalex_work_json]})

    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    papers = api.search_works("Minimal Agents", limit=3)
    assert [paper.openalex_id for paper in papers] == ["W456"]
    assert seen["search"] == "Minimal Agents"
    assert seen["per_page"] == "3"
    with pytest.raises(OpenAlexError, match="cannot be empty"):
        api.search_works(" ")
    with pytest.raises(OpenAlexError, match="limit must be positive"):
        api.search_works("Paper", limit=0)

    malformed = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"results": {}})
            )
        ),
    )
    with pytest.raises(OpenAlexError, match="malformed"):
        malformed.search_works("Paper")


def test_search_works_skips_sparse_title_records(openalex_work_json):
    sparse = {"id": "https://openalex.org/WEMPTY", "display_name": None}
    api = OpenAlexClient(
        api_key="key",
        base_url="https://openalex.test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"results": [sparse, openalex_work_json]}
                )
            )
        ),
    )
    assert [paper.openalex_id for paper in api.search_works("Minimal")] == ["W456"]
