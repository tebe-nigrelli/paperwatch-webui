from __future__ import annotations

import json

import httpx
import pytest

from paperwatch.llm import (
    CodexClient,
    LLMError,
    OllamaClient,
    OpenAIClient,
    PromptBuilder,
    create_llm,
)


def test_prompt_builder_includes_metadata_and_rejects_truncation(tmp_path, paper_record):
    template = tmp_path / "prompt.txt"
    template.write_text("{title}|{authors}|{scope}|{text}")
    prompt = PromptBuilder(template, max_input_chars=20).build(paper_record, "abcdefghij", "abstract", "test-model")
    assert paper_record.title in prompt
    assert "Ada Lovelace" in prompt
    assert prompt.endswith("abcdefghij")
    with pytest.raises(LLMError, match="partial analysis"):
        PromptBuilder(template, max_input_chars=5).build(
            paper_record, "abcdefghij", "abstract", "test-model"
        )


def test_ollama_structured_request_and_response(analysis_result):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": analysis_result.model_dump_json()},
                "prompt_eval_count": 100,
                "eval_count": 50,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OllamaClient(
        model="qwen", base_url="http://ollama.test", timeout_seconds=1,
        context_window_tokens=4096, client=client
    )
    response = llm.analyze("analyze this")
    assert response.result.one_sentence_summary == analysis_result.one_sentence_summary
    assert response.result.generated_by_model == "qwen (version not exposed)"
    assert response.usage["eval_count"] == 50
    assert seen["format"]["required"]
    assert seen["stream"] is False
    assert seen["options"]["num_ctx"] == 4096


def test_ollama_missing_model_error_is_actionable_and_does_not_fallback():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            404,
            json={"error": "model 'llama3.1:8b' not found, try pulling it first"},
        )

    llm = OllamaClient(
        model="llama3.1:8b",
        base_url="http://ollama.test/api",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError, match=r"ollama pull llama3\.1:8b"):
        llm.analyze("x")
    assert paths == ["/api/chat"]


def test_ollama_falls_back_to_openai_compatible_chat(analysis_result):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/chat":
            return httpx.Response(404, text="not found")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": analysis_result.model_dump_json()}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    llm = OllamaClient(
        model="local-model",
        base_url="http://llama.test/v1",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = llm.analyze("x")
    assert result.result.name == analysis_result.name
    assert result.usage["completion_tokens"] == 20
    assert [item[0] for item in requests] == ["/api/chat", "/v1/chat/completions"]
    assert requests[1][1]["response_format"]["json_schema"]["strict"] is True


def test_ollama_readiness_checks_native_and_compatible_model_lists():
    native = OllamaClient(
        model="qwen3:8b",
        base_url="http://ollama.test",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"models": [{"name": "qwen3:8b"}]}
                )
            )
        ),
    )
    native.check_ready()

    def compatible_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "llama-local"}]})

    compatible = OllamaClient(
        model="llama-local",
        base_url="http://llama.test",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(compatible_handler)),
    )
    compatible.check_ready()

    missing = OllamaClient(
        model="missing:8b",
        base_url="http://ollama.test",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"models": [{"name": "qwen3:8b"}]}
                )
            )
        ),
    )
    with pytest.raises(LLMError, match=r"ollama pull missing:8b"):
        missing.check_ready()


def test_openai_responses_structured_request_and_response(analysis_result):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": analysis_result.model_dump_json()}
                        ]
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAIClient(
        model="gpt-test",
        base_url="https://api.openai.test",
        api_key="secret",
        timeout_seconds=1,
        client=client,
    )
    response = llm.analyze("analyze this")
    assert response.result.paper_kind == "machine_learning"
    assert response.result.generated_by_model == "gpt-test (version not exposed)"
    assert seen["headers"]["authorization"] == "Bearer secret"
    assert seen["json"]["store"] is False
    assert seen["json"]["text"]["format"]["strict"] is True



def test_codex_responses_structured_request_and_reasoning(analysis_result):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = request.headers
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output_text": analysis_result.model_dump_json(),
                "usage": {"input_tokens": 12, "output_tokens": 34},
            },
        )

    llm = CodexClient(
        model="gpt-5.3-codex",
        base_url="https://api.openai.test/v1",
        api_key="secret",
        timeout_seconds=1,
        reasoning_effort="high",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = llm.analyze("analyze this")
    assert llm.provider == "codex"
    assert response.result.generated_by_model == "gpt-5.3-codex (version not exposed)"
    assert response.usage["output_tokens"] == 34
    assert seen["path"] == "/v1/responses"
    assert seen["headers"]["authorization"] == "Bearer secret"
    assert seen["json"]["reasoning"] == {"effort": "high"}
    assert seen["json"]["text"]["format"]["strict"] is True


def test_codex_missing_key_has_specific_error():
    llm = CodexClient(
        model="gpt-5.3-codex",
        base_url="https://api.openai.test",
        api_key="",
        timeout_seconds=1,
        client=httpx.Client(),
    )
    with pytest.raises(LLMError, match="OpenAI Codex API key is missing"):
        llm.check_ready()
    with pytest.raises(LLMError, match="OpenAI Codex API key is missing"):
        llm.analyze("x")

def test_llm_rejects_invalid_model_json():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"message": {"content": '{"summary":"x"}'}})
        )
    )
    llm = OllamaClient(model="qwen", base_url="http://ollama.test", timeout_seconds=1, client=client)
    with pytest.raises(LLMError, match="analysis failed"):
        llm.analyze("test")


def test_create_none_provider(app_config):
    object.__setattr__(app_config.llm, "provider", "none")
    llm = create_llm(app_config.llm)
    assert llm.provider == "none"


def test_no_llm_and_provider_factory_branches(app_config):
    from dataclasses import replace
    from paperwatch.llm import NoLLMClient

    no_llm = NoLLMClient()
    with pytest.raises(LLMError, match="disabled"):
        no_llm.analyze("x")
    no_llm.close()

    ollama = create_llm(replace(app_config.llm, provider="ollama"), client=httpx.Client())
    assert isinstance(ollama, OllamaClient)
    assert ollama.context_window_tokens == app_config.llm.context_window_tokens
    openai = create_llm(
        replace(app_config.llm, provider="openai", api_key="key", base_url="https://api.test/v1"),
        client=httpx.Client(),
    )
    assert isinstance(openai, OpenAIClient)
    assert openai.base_url == "https://api.test/v1"
    codex = create_llm(
        replace(
            app_config.llm,
            provider="codex",
            model="gpt-5.3-codex",
            api_key="key",
            base_url="https://api.test/v1",
            reasoning_effort="xhigh",
        ),
        client=httpx.Client(),
    )
    assert isinstance(codex, CodexClient)
    assert codex.provider == "codex"
    assert codex.reasoning_effort == "xhigh"
    with pytest.raises(LLMError, match="Unsupported"):
        create_llm(replace(app_config.llm, provider="bad"))


def test_ollama_response_error_branches():
    cases = [
        (httpx.Response(200, json=[]), "unexpected payload"),
        (httpx.Response(200, json={}), "no message content"),
        (httpx.Response(500), "analysis failed"),
        (httpx.Response(200, content=b"bad-json"), "analysis failed"),
    ]
    for response, message in cases:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request, r=response: r)
        )
        llm = OllamaClient(model="m", base_url="http://ollama.test", timeout_seconds=1, client=client)
        with pytest.raises(LLMError, match=message):
            llm.analyze("x")


def test_openai_output_shapes_and_errors(analysis_result):
    assert OpenAIClient._output_text({"output_text": "direct"}) == "direct"
    with pytest.raises(LLMError, match="malformed output"):
        OpenAIClient._output_text({"output": {}})
    with pytest.raises(LLMError, match="no output text"):
        OpenAIClient._output_text({"output": [None, {"content": [None, {"type": "refusal"}]}]})

    missing_key = OpenAIClient(
        model="m", base_url="https://api.test", api_key="", timeout_seconds=1, client=httpx.Client()
    )
    with pytest.raises(LLMError, match="key is missing"):
        missing_key.analyze("x")

    responses = [
        (httpx.Response(200, json=[]), "unexpected payload"),
        (httpx.Response(200, json={"output": []}), "no output text"),
        (httpx.Response(500), "analysis failed"),
    ]
    for response, message in responses:
        client = httpx.Client(transport=httpx.MockTransport(lambda request, r=response: r))
        llm = OpenAIClient(
            model="m", base_url="https://api.test", api_key="key", timeout_seconds=1, client=client
        )
        with pytest.raises(LLMError, match=message):
            llm.analyze("x")

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"output_text": analysis_result.model_dump_json(), "usage": "bad"},
            )
        )
    )
    llm = OpenAIClient(
        model="m", base_url="https://api.test", api_key="key", timeout_seconds=1, client=client
    )
    assert llm.analyze("x").usage == {}


def test_prompt_builder_validation_and_unknown_authors(tmp_path, paper_record):
    missing = tmp_path / "missing.txt"
    with pytest.raises(LLMError, match="Cannot read"):
        PromptBuilder(missing, 10)

    bad_limit = tmp_path / "bad-limit.txt"
    bad_limit.write_text("{text}")
    with pytest.raises(LLMError, match="greater than zero"):
        PromptBuilder(bad_limit, 0)

    literal_json = tmp_path / "literal-json.txt"
    literal_json.write_text('{"key": "value"} {text}')
    assert PromptBuilder(literal_json, 10).build(paper_record, "abc", "abstract", "m").startswith('{"key"')

    no_text = tmp_path / "no-text.txt"
    no_text.write_text("{title}")
    with pytest.raises(LLMError, match=r"\{text\}"):
        PromptBuilder(no_text, 10)

    template = tmp_path / "valid.txt"
    template.write_text("{authors}|{publication_date}|{venue}|{scope}|{text}")
    paper_record.authors = []
    paper_record.publication_date = None
    paper_record.venue = None
    prompt = PromptBuilder(template, 10).build(paper_record, "abc", "abstract", "m")
    assert prompt.startswith("Unknown|Unknown|Unknown")


def test_owned_llm_clients_close():
    ollama = OllamaClient(model="m", base_url="http://localhost:11434", timeout_seconds=1)
    ollama.close()
    assert ollama.client.is_closed
    openai = OpenAIClient(
        model="m", base_url="https://api.openai.com", api_key="key", timeout_seconds=1
    )
    openai.close()
    assert openai.client.is_closed


def test_generic_structured_generation_and_disabled_branch(analysis_result):
    from paperwatch.llm import NoLLMClient
    from paperwatch.models import BATCH_QUERY_JSON_SCHEMA

    no_llm = NoLLMClient()
    with pytest.raises(LLMError, match="disabled"):
        no_llm.generate_json("x", schema={}, schema_name="x")

    payload = {"title": "T", "summary": "S", "items": []}
    ollama = OllamaClient(
        model="m",
        base_url="http://test",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"message": {"content": json.dumps(payload)}}
                )
            )
        ),
    )
    assert ollama.generate_json(
        "x", schema=BATCH_QUERY_JSON_SCHEMA, schema_name="collection"
    ).data == payload

    openai = OpenAIClient(
        model="m",
        base_url="https://test",
        api_key="key",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"output_text": json.dumps(payload)})
            )
        ),
    )
    assert openai.generate_json(
        "x", schema=BATCH_QUERY_JSON_SCHEMA, schema_name="collection"
    ).data == payload


def test_generic_structured_generation_rejects_non_objects():
    ollama = OllamaClient(
        model="m",
        base_url="http://test",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"message": {"content": "[]"}})
            )
        ),
    )
    with pytest.raises(LLMError, match="JSON object"):
        ollama.generate_json("x", schema={}, schema_name="x")

    openai = OpenAIClient(
        model="m",
        base_url="https://test",
        api_key="key",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"output_text": "[]"})
            )
        ),
    )
    with pytest.raises(LLMError, match="JSON object"):
        openai.generate_json("x", schema={}, schema_name="x")


def test_context_override_is_applied_only_to_ollama():
    from paperwatch.llm import configure_context_window

    ollama = OllamaClient(
        model="m", base_url="http://test", timeout_seconds=1,
        context_window_tokens=1024, client=httpx.Client()
    )
    configure_context_window(ollama, 16384)
    assert ollama.context_window_tokens == 16384

    openai = OpenAIClient(
        model="m", base_url="https://test", api_key="key",
        timeout_seconds=1, client=httpx.Client()
    )
    configure_context_window(openai, 16384)
    assert not hasattr(openai, "context_window_tokens")


def test_ollama_readiness_error_and_normalization_branches():
    from paperwatch.llm import NoLLMClient

    NoLLMClient().check_ready()
    assert OllamaClient._model_matches("model", {"model:latest"})
    assert OllamaClient._model_matches("model:latest", {"model"})
    assert not OllamaClient._model_matches("missing", {"other"})

    cases = [
        (
            lambda request: httpx.Response(500, json={"message": "server failed"}),
            "HTTP 500",
        ),
        (lambda request: httpx.Response(200, json=[]), "malformed"),
        (
            lambda request: httpx.Response(
                200,
                json={"models": [None, {"model": "alternate:latest"}]},
            ),
            r"ollama pull wanted",
        ),
    ]
    for handler, message in cases:
        llm = OllamaClient(
            model="wanted",
            base_url="http://ollama.test",
            timeout_seconds=1,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMError, match=message):
            llm.check_ready()

    compatible_cases = [
        (
            lambda request: httpx.Response(404)
            if request.url.path == "/api/tags"
            else httpx.Response(404),
            "endpoints were not found",
        ),
        (
            lambda request: httpx.Response(404)
            if request.url.path == "/api/tags"
            else httpx.Response(200, json=[]),
            "model-list response is malformed",
        ),
    ]
    for handler, message in compatible_cases:
        llm = OllamaClient(
            model="wanted",
            base_url="http://ollama.test",
            timeout_seconds=1,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMError, match=message):
            llm.check_ready()

    def disconnected(request):
        raise httpx.ConnectError("offline", request=request)

    llm = OllamaClient(
        model="wanted",
        base_url="http://ollama.test",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(disconnected)),
    )
    with pytest.raises(LLMError, match="Cannot connect"):
        llm.check_ready()


def test_ollama_openai_compatible_error_shapes_and_openai_readiness(analysis_result):
    from paperwatch.llm import OpenAIClient

    responses = [
        (httpx.Response(500, json={"detail": "bad compatible server"}), "HTTP 500"),
        (httpx.Response(200, json=[]), "unexpected OpenAI-compatible"),
        (httpx.Response(200, json={"choices": []}), "no choices"),
        (
            httpx.Response(200, json={"choices": [{"message": {}}]}),
            "no message content",
        ),
        (
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": analysis_result.model_dump_json()}}
                    ],
                    "usage": "bad",
                },
            ),
            None,
        ),
    ]
    for compatible_response, message in responses:
        def handler(request, response=compatible_response):
            if request.url.path == "/api/chat":
                return httpx.Response(404)
            return response

        llm = OllamaClient(
            model="m",
            base_url="http://llama.test",
            timeout_seconds=1,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        if message:
            with pytest.raises(LLMError, match=message):
                llm.analyze("x")
        else:
            assert llm.analyze("x").usage == {}

    ready = OpenAIClient(
        model="m", base_url="https://api.test", api_key="key", timeout_seconds=1,
        client=httpx.Client()
    )
    ready.check_ready()
    missing = OpenAIClient(
        model="m", base_url="https://api.test", api_key="", timeout_seconds=1,
        client=httpx.Client()
    )
    with pytest.raises(LLMError, match="key is missing"):
        missing.check_ready()


def test_openai_and_codex_http_error_messages(analysis_result):
    detail_cases = [
        (httpx.Response(400, json={"error": {"message": "model is unavailable"}}), "model .* unavailable"),
        (httpx.Response(401, json={"message": "bad key"}), "API key was rejected"),
        (httpx.Response(500, json={"detail": "server broke"}), "HTTP 500: server broke"),
        (httpx.Response(503, text="plain failure"), "plain failure"),
        (httpx.Response(502), "HTTP 502"),
    ]
    for response, message in detail_cases:
        llm = CodexClient(
            model="gpt-5.3-codex",
            base_url="https://api.openai.test",
            api_key="key",
            timeout_seconds=1,
            client=httpx.Client(
                transport=httpx.MockTransport(lambda request, item=response: item)
            ),
        )
        with pytest.raises(LLMError, match=message):
            llm.analyze("x")

    assert OpenAIClient._error_detail(httpx.Response(400, json={"message": " top "})) == "top"
    assert OpenAIClient._error_detail(httpx.Response(400, json={"detail": " detail "})) == "detail"


def test_codex_rejects_invalid_structured_result():
    llm = CodexClient(
        model="gpt-5.3-codex",
        base_url="https://api.openai.test",
        api_key="key",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"output_text": '{"summary":"x"}'})
            )
        ),
    )
    with pytest.raises(LLMError, match="OpenAI Codex analysis failed"):
        llm.analyze("x")


def test_codex_pdf_upload_attaches_original_file_and_deletes_remote_copy(
    analysis_result, tmp_path
):
    pdf = tmp_path / "whole-paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\noriginal-complete-pdf\n%%EOF")
    requests: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        requests.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/v1/files":
            assert b'purpose' in body
            assert b'user_data' in body
            assert b'whole-paper.pdf' in body
            assert b'original-complete-pdf' in body
            return httpx.Response(200, json={"id": "file-paper-123"})
        if request.method == "POST" and request.url.path == "/v1/responses":
            payload = json.loads(body)
            content = payload["input"][0]["content"]
            assert content[0] == {"type": "input_file", "file_id": "file-paper-123"}
            assert content[1]["type"] == "input_text"
            assert "attached pdf" in content[1]["text"].lower().replace(" as a", "")
            return httpx.Response(
                200,
                json={
                    "output_text": analysis_result.model_dump_json(),
                    "usage": {"input_tokens": 123, "output_tokens": 45},
                },
            )
        if request.method == "DELETE" and request.url.path == "/v1/files/file-paper-123":
            return httpx.Response(200, json={"id": "file-paper-123", "deleted": True})
        return httpx.Response(404)

    llm = CodexClient(
        model="gpt-5.3-codex",
        base_url="https://api.openai.test/v1",
        api_key="secret",
        timeout_seconds=1,
        reasoning_effort="high",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = llm.analyze_pdf("Read the attached PDF in full", pdf)
    assert response.result.name == analysis_result.name
    assert response.usage["input_file_bytes"] == pdf.stat().st_size
    assert [(method, path) for method, path, _ in requests] == [
        ("POST", "/v1/files"),
        ("POST", "/v1/responses"),
        ("DELETE", "/v1/files/file-paper-123"),
    ]


def test_codex_pdf_upload_failure_is_actionable_and_does_not_call_responses(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(400, json={"error": {"message": "invalid PDF"}})

    llm = CodexClient(
        model="gpt-5.3-codex",
        base_url="https://api.openai.test/v1",
        api_key="secret",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError, match="invalid PDF"):
        llm.analyze_pdf("x", pdf)
    assert paths == ["/v1/files"]
