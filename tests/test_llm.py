from __future__ import annotations

import json
from pathlib import Path
import subprocess

import httpx
import pytest

from paperwatch.llm import (
    ClaudeCodeClient,
    CodexClient,
    LLMError,
    OllamaClient,
    OpenAIClient,
    PromptBuilder,
    create_llm,
)


class FakeClaudeProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.prompts: list[tuple[str | None, float | int | None]] = []
        self.terminated = False
        self.killed = False
        self.running = False

    def communicate(self, prompt=None, timeout=None):
        self.prompts.append((prompt, timeout))
        return self.stdout, self.stderr

    def poll(self):
        return None if self.running else self.returncode

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False


def test_claude_code_structured_request_response_and_factory(
    app_config, analysis_result, monkeypatch
):
    envelope = {
        "structured_output": analysis_result.model_dump(),
        "usage": {"input_tokens": 12, "output_tokens": 34},
        "total_cost_usd": 0.01,
        "duration_ms": 1250,
        "num_turns": 1,
    }
    process = FakeClaudeProcess(json.dumps(envelope))
    seen = {}

    def popen(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return process

    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/claude")
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", popen)
    output = []
    llm = ClaudeCodeClient(
        model="claude-sonnet-4-5", command="claude", timeout_seconds=9,
        reasoning_effort="high",
    )
    llm.set_runtime(lambda kind, value: output.append((kind, value)), None)
    response = llm.analyze("analyze this paper")

    assert response.result.name == analysis_result.name
    assert response.result.generated_by_model == "claude-sonnet-4-5 (version not exposed)"
    assert response.usage["output_tokens"] == 34
    assert response.usage["total_cost_usd"] == 0.01
    assert process.prompts == [("analyze this paper", 9)]
    assert seen["args"][0:3] == ["claude", "-p", seen["args"][2]]
    assert "--json-schema" in seen["args"]
    assert seen["args"][seen["args"].index("--model") + 1] == "claude-sonnet-4-5"
    assert seen["args"][seen["args"].index("--tools") + 1] == ""
    assert seen["args"][seen["args"].index("--effort") + 1] == "high"
    assert seen["kwargs"]["stdin"] is subprocess.PIPE
    assert output and output[0][0] == "output"

    from dataclasses import replace

    created = create_llm(
        replace(
            app_config.llm, provider="claude-code", model="claude-haiku-4-5",
            command="custom-claude", reasoning_effort=None,
        )
    )
    assert isinstance(created, ClaudeCodeClient)
    assert created.command == "custom-claude"


def test_claude_code_native_pdf_uses_read_tool(
    tmp_path, analysis_result, monkeypatch
):
    pdf = tmp_path / "complete paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nCOMPLETE PAPER\n%%EOF")
    envelope = {
        "structured_output": analysis_result.model_dump(),
        "usage": {"input_tokens": 42},
    }
    process = FakeClaudeProcess(json.dumps(envelope))
    seen = {}

    def popen(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return process

    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/claude")
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", popen)
    response = ClaudeCodeClient(model="claude-sonnet-4-5").analyze_pdf(
        "Summarize the paper", pdf
    )

    assert seen["args"][seen["args"].index("--tools") + 1] == "Read"
    assert seen["kwargs"]["cwd"] == str(tmp_path)
    assert "complete paper.pdf" in process.prompts[0][0]
    assert "read the PDF directly and in full" in process.prompts[0][0]
    assert response.usage["input_file_bytes"] == pdf.stat().st_size

    with pytest.raises(LLMError, match="PDF file does not exist"):
        ClaudeCodeClient(model="claude-sonnet-4-5").analyze_pdf(
            "Summarize the paper", tmp_path / "missing.pdf"
        )


def test_claude_code_readiness_close_and_response_errors(
    tmp_path, analysis_result, monkeypatch
):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: None)
    missing = ClaudeCodeClient(model="m", command="not-installed", timeout_seconds=1)
    with pytest.raises(LLMError, match="was not found"):
        missing.check_ready()

    local_command = tmp_path / "claude-local"
    local_command.write_text("stub")
    local = ClaudeCodeClient(model="m", command=str(local_command), timeout_seconds=1)
    local.check_ready()
    running = FakeClaudeProcess()
    running.running = True
    local._process = running
    local.close()
    assert running.terminated

    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/claude")
    cases = [
        (FakeClaudeProcess("", "authentication failed", 1), "authentication failed"),
        (FakeClaudeProcess("not-json"), "invalid JSON envelope"),
        (FakeClaudeProcess("[]"), "unexpected response envelope"),
        (FakeClaudeProcess(json.dumps({"usage": {}})), "no structured_output"),
    ]
    for process, message in cases:
        monkeypatch.setattr("paperwatch.llm.subprocess.Popen", lambda *args, p=process, **kwargs: p)
        with pytest.raises(LLMError, match=message):
            ClaudeCodeClient(model="m", timeout_seconds=1).analyze("x")

    malformed_result = dict(analysis_result.model_dump())
    malformed_result.pop("name")
    process = FakeClaudeProcess(
        json.dumps({"structured_output": malformed_result, "usage": "invalid"})
    )
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", lambda *args, **kwargs: process)
    with pytest.raises(LLMError, match="Claude Code analysis failed"):
        ClaudeCodeClient(model="m", timeout_seconds=1).analyze("x")

    def fail_start(*args, **kwargs):
        raise OSError("cannot execute")

    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", fail_start)
    with pytest.raises(LLMError, match="could not start"):
        ClaudeCodeClient(model="m", timeout_seconds=1).analyze("x")


def test_claude_code_timeout_terminates_then_kills(monkeypatch):
    class TimeoutProcess(FakeClaudeProcess):
        def communicate(self, prompt=None, timeout=None):
            self.prompts.append((prompt, timeout))
            if len(self.prompts) < 3:
                raise subprocess.TimeoutExpired("claude", timeout)
            return "", ""

    process = TimeoutProcess()
    process.running = True
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/claude")
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", lambda *args, **kwargs: process)
    with pytest.raises(LLMError, match="timed out after 0.5 seconds"):
        ClaudeCodeClient(model="m", timeout_seconds=0.5).generate_json(
            "x", schema={}, schema_name="ignored"
        )
    assert process.terminated
    assert process.killed
    assert [timeout for _prompt, timeout in process.prompts] == [0.5, 5, None]


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



def test_codex_cli_structured_request_and_reasoning(analysis_result, monkeypatch):
    process = FakeClaudeProcess(analysis_result.model_dump_json())
    seen = {}

    def popen(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return process

    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", popen)
    output = []
    llm = CodexClient(
        model="gpt-5.3-codex", command="codex", timeout_seconds=9,
        reasoning_effort="high",
    )
    llm.set_runtime(lambda kind, value: output.append((kind, value)), None)
    response = llm.analyze("analyze this")
    assert llm.provider == "codex"
    assert response.result.generated_by_model == "gpt-5.3-codex (version not exposed)"
    assert response.usage == {"provider": "codex", "transport": "cli"}
    assert len(process.prompts) == 1
    assert process.prompts[0][1] == 9
    assert process.prompts[0][0].endswith("\n\nanalyze this")
    assert "Do not inspect files" in process.prompts[0][0]
    assert seen["args"][:2] == ["codex", "exec"]
    assert "--ephemeral" in seen["args"]
    assert seen["args"][seen["args"].index("--sandbox") + 1] == "read-only"
    assert "--skip-git-repo-check" in seen["args"]
    assert "--output-schema" in seen["args"]
    assert "--output-last-message" in seen["args"]
    assert seen["args"][seen["args"].index("--model") + 1] == "gpt-5.3-codex"
    assert 'model_reasoning_effort="high"' in seen["args"]
    assert seen["args"][-1] == "-"
    assert seen["kwargs"]["stdin"] is subprocess.PIPE
    assert Path(seen["kwargs"]["cwd"]).name.startswith("paperwatch-codex-")
    assert output and output[0][0] == "output"


def test_codex_cli_readiness_uses_installed_executable(tmp_path, monkeypatch):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: None)
    missing = CodexClient(model="gpt-5.3-codex", command="not-installed", timeout_seconds=1)
    with pytest.raises(LLMError, match="Codex CLI executable .* was not found"):
        missing.check_ready()
    local_command = tmp_path / "codex-local"
    local_command.write_text("stub")
    CodexClient(model="gpt-5.3-codex", command=str(local_command), timeout_seconds=1).check_ready()


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
            command="custom-codex",
            api_key="",
            reasoning_effort="xhigh",
        ),
        client=httpx.Client(),
    )
    assert isinstance(codex, CodexClient)
    assert codex.provider == "codex"
    assert codex.command == "custom-codex"
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


def test_openai_http_error_messages():
    detail_cases = [
        (httpx.Response(400, json={"error": {"message": "model is unavailable"}}), "model .* unavailable"),
        (httpx.Response(401, json={"message": "bad key"}), "API key was rejected"),
        (httpx.Response(500, json={"detail": "server broke"}), "HTTP 500: server broke"),
        (httpx.Response(503, text="plain failure"), "plain failure"),
        (httpx.Response(502), "HTTP 502"),
    ]
    for response, message in detail_cases:
        llm = OpenAIClient(
            model="gpt-test",
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


def test_codex_cli_rejects_invalid_structured_result(monkeypatch):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    process = FakeClaudeProcess('{"summary":"x"}')
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", lambda *args, **kwargs: process)
    with pytest.raises(LLMError, match="Codex CLI analysis failed"):
        CodexClient(model="gpt-5.3-codex", timeout_seconds=1).analyze("x")


def test_codex_cli_pdf_reads_local_original_in_read_only_mode(analysis_result, tmp_path, monkeypatch):
    pdf = tmp_path / "whole-paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\noriginal-complete-pdf\n%%EOF")
    process = FakeClaudeProcess(analysis_result.model_dump_json())
    seen = {}

    def popen(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return process

    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", popen)
    response = CodexClient(
        model="gpt-5.3-codex", command="codex", timeout_seconds=1, reasoning_effort="high"
    ).analyze_pdf("Read the complete paper", pdf)
    assert response.result.name == analysis_result.name
    assert response.usage["input_file_bytes"] == pdf.stat().st_size
    assert seen["kwargs"]["cwd"] == str(tmp_path)
    assert seen["args"][seen["args"].index("--sandbox") + 1] == "read-only"
    assert str(pdf) in process.prompts[0][0]
    assert "complete document" in process.prompts[0][0]

    with pytest.raises(LLMError, match="PDF file does not exist"):
        CodexClient(model="gpt-5.3-codex").analyze_pdf("x", tmp_path / "missing.pdf")


def test_codex_cli_failure_is_actionable(monkeypatch):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    process = FakeClaudeProcess("", "not logged in; run codex login", 1)
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", lambda *args, **kwargs: process)
    with pytest.raises(LLMError, match="not logged in; run codex login"):
        CodexClient(model="gpt-5.3-codex", timeout_seconds=1).analyze("x")


def test_codex_cli_failure_reports_tail_not_banner(monkeypatch):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    stderr = "\n".join(
        ["OpenAI Codex v0.147.0", "--------", "user", "very long prompt"]
        + [f"prompt line {index}" for index in range(100)]
        + ["ERROR: request failed: context length exceeded"]
    )
    process = FakeClaudeProcess("", stderr, 1)
    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", lambda *args, **kwargs: process)
    with pytest.raises(LLMError) as excinfo:
        CodexClient(model="gpt-5.3-codex", timeout_seconds=1).analyze("x")
    message = str(excinfo.value)
    assert "context length exceeded" in message
    assert "OpenAI Codex v0.147.0" not in message


def test_codex_cli_retries_chatgpt_unsupported_model_with_cli_default(
    analysis_result, monkeypatch
):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    processes = [
        FakeClaudeProcess(
            "",
            "ERROR: {\"status\":400,\"error\":{\"message\":\"The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account.\"}}",
            1,
        ),
        FakeClaudeProcess(
            analysis_result.model_dump_json(),
            "OpenAI Codex v0.147.0\nmodel: gpt-5.1-codex-max\nprovider: openai",
        ),
    ]
    calls = []

    def popen(args, **kwargs):
        calls.append(args)
        return processes.pop(0)

    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", popen)
    response = CodexClient(model="gpt-5.3-codex", timeout_seconds=1).analyze("x")
    assert len(calls) == 2
    assert calls[0][calls[0].index("--model") + 1] == "gpt-5.3-codex"
    assert "--model" not in calls[1]
    assert response.result.generated_by_model == "gpt-5.1-codex-max (version not exposed)"
    assert response.usage["model_fallback"] is True
    assert response.usage["requested_model"] == "gpt-5.3-codex"
    assert response.usage["effective_model"] == "gpt-5.1-codex-max"


def test_codex_cli_empty_model_uses_cli_default(analysis_result, monkeypatch):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    process = FakeClaudeProcess(
        analysis_result.model_dump_json(),
        "OpenAI Codex v0.147.0\nmodel: account-default-codex\nprovider: openai",
    )
    calls = []
    monkeypatch.setattr(
        "paperwatch.llm.subprocess.Popen",
        lambda args, **kwargs: calls.append(args) or process,
    )
    response = CodexClient(model="", timeout_seconds=1).analyze("x")
    assert "--model" not in calls[0]
    assert response.result.generated_by_model == "account-default-codex (version not exposed)"
    assert response.usage["effective_model"] == "account-default-codex"


def test_codex_cli_retries_schema_specific_failure_without_output_schema(
    analysis_result, monkeypatch
):
    monkeypatch.setattr("paperwatch.llm.shutil.which", lambda command: "/usr/bin/codex")
    processes = [
        FakeClaudeProcess(
            "",
            "ERROR invalid_json_schema: Invalid schema for response_format codex_output_schema",
            1,
        ),
        FakeClaudeProcess(analysis_result.model_dump_json()),
    ]
    calls = []

    def popen(args, **kwargs):
        calls.append(args)
        return processes.pop(0)

    monkeypatch.setattr("paperwatch.llm.subprocess.Popen", popen)
    response = CodexClient(model="gpt-5.3-codex", timeout_seconds=1).analyze("x")
    assert response.result.name == analysis_result.name
    assert len(calls) == 2
    assert "--output-schema" in calls[0]
    assert "--output-schema" not in calls[1]


def test_ollama_retries_empty_and_incomplete_structured_output(analysis_result):
    responses = [
        httpx.Response(200, content=b""),
        httpx.Response(200, json={"message": {"content": analysis_result.model_dump_json()}}),
    ]
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return responses.pop(0)

    llm = OllamaClient(
        model="deepseek-r1:8b",
        base_url="http://ollama.test",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.analyze("paper").result.name == analysis_result.name
    assert len(requests) == 2
    assert all(item["think"] is False for item in requests)
    assert requests[1]["stream"] is False
    assert "complete JSON object" in requests[1]["messages"][0]["content"]

    partial = analysis_result.model_dump_json()[:80]
    responses.extend(
        [
            httpx.Response(200, json={"message": {"content": partial}}),
            httpx.Response(200, json={"message": {"content": analysis_result.model_dump_json()}}),
        ]
    )
    assert llm.analyze("paper").result.name == analysis_result.name


def test_ollama_json_decoder_accepts_fences_and_extracts_object():
    assert OllamaClient._decode_json_content('```json\n{"ok": true}\n```') == {"ok": True}
    assert OllamaClient._decode_json_content('prefix {"ok": true} suffix') == {"ok": True}
    with pytest.raises(LLMError, match="JSON object"):
        OllamaClient._decode_json_content('[1, 2]')
    with pytest.raises(LLMError, match="incomplete JSON"):
        OllamaClient._decode_json_content('{"value": "unfinished')


def test_ollama_json_decoder_reports_empty_and_invalid_output():
    with pytest.raises(LLMError, match="no message content"):
        OllamaClient._decode_json_content("   ")
    with pytest.raises(LLMError, match="invalid JSON"):
        OllamaClient._decode_json_content("not-json")


def test_ollama_streaming_missing_model_is_actionable():
    llm = OllamaClient(
        model="missing:8b",
        base_url="http://ollama.test",
        timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    404, json={"error": "model missing:8b not found, pull it first"}
                )
            )
        ),
    )
    llm.set_runtime(lambda kind, value: None, __import__("threading").Event())
    with pytest.raises(LLMError, match=r"ollama pull missing:8b"):
        llm.analyze("paper")


def test_ollama_structured_output_retries_twice_with_larger_output_budget(analysis_result):
    partial = analysis_result.model_dump_json()[:100]
    responses = [
        httpx.Response(200, json={"message": {"content": partial}}),
        httpx.Response(200, json={"message": {"content": partial}}),
        httpx.Response(200, json={"message": {"content": analysis_result.model_dump_json()}}),
    ]
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return responses.pop(0)

    progress = []
    llm = OllamaClient(
        model="deepseek-r1:8b",
        base_url="http://ollama.test",
        timeout_seconds=1,
        context_window_tokens=12_288,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    llm.set_runtime(lambda kind, value: progress.append((kind, value)), __import__("threading").Event())
    response = llm.analyze("paper")
    assert response.result.name == analysis_result.name
    assert len(requests) == 3
    limits = [item["options"]["num_predict"] for item in requests]
    assert limits == sorted(limits)
    assert limits[0] < limits[-1]
    assert sum("larger output budget" in value for kind, value in progress if kind == "message") == 2


def test_openai_strict_schema_requires_every_object_property():
    from paperwatch.models import BATCH_QUERY_JSON_SCHEMA

    llm = OpenAIClient(
        model="m",
        base_url="https://api.test",
        api_key="key",
        timeout_seconds=1,
        client=httpx.Client(),
    )
    payload = llm._request_payload(
        "x", schema=BATCH_QUERY_JSON_SCHEMA, schema_name="paper_collection_query"
    )
    strict_schema = payload["text"]["format"]["schema"]

    def assert_strict(value):
        if isinstance(value, dict):
            assert "default" not in value
            properties = value.get("properties")
            if isinstance(properties, dict):
                assert value["required"] == list(properties)
                assert value["additionalProperties"] is False
            for child in value.values():
                assert_strict(child)
        elif isinstance(value, list):
            for child in value:
                assert_strict(child)

    assert_strict(strict_schema)
    assert "blog_entry_id" in strict_schema["$defs"]["BatchQueryItem"]["required"]
    llm.close()
