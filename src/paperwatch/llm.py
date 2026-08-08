from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Protocol, TypeVar
import threading

import httpx
from pydantic import BaseModel, ValidationError

from .config import LLMConfig
from .job_runtime import JobCancelled, ProgressCallback, check_cancelled
from .models import ANALYSIS_JSON_SCHEMA, AnalysisResult, LLMResponse, PaperRecord, model_attribution


class LLMError(RuntimeError):
    pass


class _OllamaOutputError(LLMError):
    """A successful Ollama HTTP response with unusable model output."""


class _OllamaGrammarError(LLMError):
    """The server cannot compile Ollama's JSON-schema grammar."""


TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(slots=True)
class StructuredJSONResponse:
    data: dict[str, Any]
    usage: dict[str, Any]
    raw: dict[str, Any]


class LLMClient(Protocol):
    provider: str
    model: str

    def analyze(self, prompt: str) -> LLMResponse:
        ...

    def analyze_pdf(self, prompt: str, pdf_path: Path) -> LLMResponse:
        ...

    def check_ready(self) -> None:
        ...

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        ...

    def generate_json_with_pdf(
        self,
        prompt: str,
        pdf_path: Path,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        ...

    def set_runtime(
        self, progress: ProgressCallback | None, cancel_event: threading.Event | None
    ) -> None:
        ...

    def close(self) -> None:
        ...


class NoLLMClient:
    provider = "none"
    model = "none"

    def analyze(self, prompt: str) -> LLMResponse:
        raise LLMError("LLM provider is disabled")

    def check_ready(self) -> None:
        return None

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], schema_name: str
    ) -> StructuredJSONResponse:
        raise LLMError("LLM provider is disabled")

    def set_runtime(
        self, progress: ProgressCallback | None, cancel_event: threading.Event | None
    ) -> None:
        return None

    def close(self) -> None:
        return None


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_seconds: float,
        context_window_tokens: int | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model
        normalized = base_url.rstrip("/")
        if normalized.endswith("/api"):
            normalized = normalized[:-4]
        elif normalized.endswith("/v1"):
            normalized = normalized[:-3]
        self.base_url = normalized.rstrip("/")
        self.context_window_tokens = context_window_tokens
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._progress: ProgressCallback | None = None
        self._cancel_event: threading.Event | None = None

    def set_runtime(
        self, progress: ProgressCallback | None, cancel_event: threading.Event | None
    ) -> None:
        self._progress = progress
        self._cancel_event = cancel_event

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    @staticmethod
    def _model_matches(requested: str, installed: set[str]) -> bool:
        if requested in installed:
            return True
        if ":" not in requested and f"{requested}:latest" in installed:
            return True
        if requested.endswith(":latest") and requested[:-7] in installed:
            return True
        return False

    def check_ready(self) -> None:
        """Verify that an Ollama-compatible server and the configured model are available."""
        try:
            native_endpoint = f"{self.base_url}/api/tags"
            response = self.client.get(native_endpoint)
            installed: set[str] = set()
            if response.status_code != 404:
                if response.is_error:
                    raise self._http_error(response, endpoint=native_endpoint)
                payload = response.json()
                if not isinstance(payload, dict):
                    raise LLMError("Ollama model-list response is malformed")
                for item in payload.get("models") or []:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("model")
                    if isinstance(name, str) and name:
                        installed.add(name)
            else:
                compatible_endpoint = f"{self.base_url}/v1/models"
                response = self.client.get(compatible_endpoint)
                if response.is_error:
                    raise LLMError(
                        "Ollama API endpoints were not found. Confirm that the service at "
                        f"{self.base_url} is Ollama or an OpenAI-compatible llama.cpp server."
                    )
                payload = response.json()
                if not isinstance(payload, dict):
                    raise LLMError("OpenAI-compatible model-list response is malformed")
                for item in payload.get("data") or []:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        installed.add(item["id"])
            if not self._model_matches(self.model, installed):
                available = ", ".join(sorted(installed)[:10]) or "none"
                raise LLMError(
                    f"Ollama model {self.model!r} is not installed. Run `ollama pull {self.model}`. "
                    f"Available models: {available}"
                )
        except LLMError:
            raise
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot connect to Ollama at {self.base_url}. Start it with `ollama serve` "
                "or your operating system's service manager, then retry."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Ollama readiness check failed: {exc}") from exc

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            for key in ("error", "message", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested = value.get("message") or value.get("detail")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
        text = response.text.strip()
        return text[:500] if text else f"HTTP {response.status_code}"

    def _http_error(self, response: httpx.Response, *, endpoint: str) -> LLMError:
        detail = self._error_detail(response)
        lowered = detail.casefold()
        if response.status_code == 404 and "model" in lowered and any(
            phrase in lowered for phrase in ("not found", "does not exist", "pull")
        ):
            return LLMError(
                f"Ollama analysis failed: model {self.model!r} is not installed. "
                f"Run `ollama pull {self.model}` and retry. Server response: {detail}"
            )
        if response.status_code == 400 and (
            "failed to parse grammar" in lowered
            or "failed to initialize samplers" in lowered
            or ("grammar" in lowered and "parse" in lowered)
        ):
            return _OllamaGrammarError(detail)
        return LLMError(
            f"Ollama analysis failed: request to {endpoint} returned "
            f"HTTP {response.status_code}: {detail}"
        )

    @staticmethod
    def _decode_json_content(content: str) -> dict[str, Any]:
        import json

        cleaned = content.strip()
        if not cleaned:
            raise _OllamaOutputError("Ollama response has no message content")
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            last_fence = cleaned.rfind("```")
            if first_newline >= 0 and last_fence > first_newline:
                cleaned = cleaned[first_newline + 1:last_fence].strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            start = cleaned.find("{")
            if start >= 0:
                try:
                    data, _ = json.JSONDecoder().raw_decode(cleaned[start:])
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    return data
            detail = (
                "incomplete JSON"
                if exc.pos >= max(0, len(cleaned) - 2) or "Unterminated" in exc.msg
                else "invalid JSON"
            )
            raise _OllamaOutputError(
                f"Ollama returned {detail}: {exc.msg} at character {exc.pos}"
            ) from exc
        if not isinstance(data, dict):
            raise _OllamaOutputError("Ollama structured output must be a JSON object")
        return data

    def _native_generate_once(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        force_stream: bool | None = None,
        output_tokens: int | None = None,
        json_mode_only: bool = False,
    ) -> StructuredJSONResponse | None:
        endpoint = f"{self.base_url}/api/chat"
        effective_prompt = prompt
        if json_mode_only:
            effective_prompt = (
                prompt
                + "\n\nThe server cannot enforce the schema as a grammar. Return exactly one "
                "JSON object matching this JSON Schema, with no markdown or commentary:\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
        request_json = {
            "model": self.model,
            "messages": [{"role": "user", "content": effective_prompt}],
            "stream": (
                self._progress is not None or self._cancel_event is not None
                if force_stream is None
                else force_stream
            ),
            "format": "json" if json_mode_only else schema,
            "think": False,
            "options": {
                "temperature": 0,
                **({"num_predict": output_tokens} if output_tokens else {}),
                **(
                    {"num_ctx": self.context_window_tokens}
                    if self.context_window_tokens is not None
                    else {}
                ),
            },
        }
        check_cancelled(self._cancel_event)
        if not request_json["stream"]:
            response = self.client.post(endpoint, json=request_json)
            if response.status_code == 404:
                detail = self._error_detail(response).casefold()
                if "model" in detail and any(
                    phrase in detail for phrase in ("not found", "does not exist", "pull")
                ):
                    raise self._http_error(response, endpoint=endpoint)
                return None
            if response.is_error:
                raise self._http_error(response, endpoint=endpoint)
            try:
                payload = response.json()
            except ValueError as exc:
                raise _OllamaOutputError(
                    "Ollama returned an empty or invalid JSON response envelope"
                ) from exc
            if not isinstance(payload, dict):
                raise _OllamaOutputError("Ollama returned an unexpected payload")
            message = payload.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise _OllamaOutputError("Ollama response has no message content")
            usage = {
                key: payload[key]
                for key in ("prompt_eval_count", "eval_count", "total_duration")
                if key in payload
            }
            return StructuredJSONResponse(
                data=self._decode_json_content(content), usage=usage, raw=payload
            )

        content_parts: list[str] = []
        final_payload: dict[str, Any] = {}
        try:
            with self.client.stream("POST", endpoint, json=request_json) as response:
                if response.status_code == 404:
                    body = response.read()
                    fallback = httpx.Response(
                        404, headers=response.headers, content=body, request=response.request
                    )
                    detail = self._error_detail(fallback).casefold()
                    if "model" in detail and any(
                        phrase in detail for phrase in ("not found", "does not exist", "pull")
                    ):
                        raise self._http_error(fallback, endpoint=endpoint)
                    return None
                if response.is_error:
                    body = response.read()
                    failed = httpx.Response(
                        response.status_code,
                        headers=response.headers,
                        content=body,
                        request=response.request,
                    )
                    raise self._http_error(failed, endpoint=endpoint)
                for line in response.iter_lines():
                    check_cancelled(self._cancel_event)
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    final_payload = payload
                    message = payload.get("message") or {}
                    fragment = message.get("content") if isinstance(message, dict) else None
                    if isinstance(fragment, str) and fragment:
                        content_parts.append(fragment)
                        if self._progress is not None:
                            self._progress("output", fragment)
        except JobCancelled:
            raise
        except httpx.HTTPError as exc:
            check_cancelled(self._cancel_event)
            raise LLMError(f"Ollama streaming response failed: {exc}") from exc
        except ValueError as exc:
            check_cancelled(self._cancel_event)
            raise _OllamaOutputError(f"Ollama streaming response failed: {exc}") from exc

        content = "".join(content_parts)
        if not content:
            raise _OllamaOutputError("Ollama response has no message content")
        usage = {
            key: final_payload[key]
            for key in ("prompt_eval_count", "eval_count", "total_duration")
            if key in final_payload
        }
        return StructuredJSONResponse(
            data=self._decode_json_content(content), usage=usage, raw=final_payload
        )

    def _native_generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
    ) -> StructuredJSONResponse | None:
        first_error: _OllamaOutputError | None = None
        last_error: _OllamaOutputError | None = None
        context = self.context_window_tokens or 16_384
        output_limits = (
            max(1_024, min(4_096, context // 4)),
            max(2_048, min(8_192, context // 3)),
            max(3_072, min(12_288, context // 2)),
        )
        json_mode_only = False
        for attempt in range(3):
            retry_prompt = prompt
            if attempt:
                retry_prompt = (
                    prompt
                    + "\n\nReturn one complete JSON object only. Do not include reasoning, "
                    "markdown fences, commentary, or text outside the JSON object. "
                    "Keep strings concise so every required field is completed before stopping."
                )
            try:
                return self._native_generate_once(
                    retry_prompt,
                    schema=schema,
                    force_stream=False if attempt else None,
                    output_tokens=output_limits[attempt],
                    json_mode_only=json_mode_only,
                )
            except _OllamaGrammarError:
                if json_mode_only:
                    raise
                json_mode_only = True
                if self._progress is not None:
                    self._progress(
                        "message",
                        "Ollama could not compile the JSON schema; retrying in JSON mode",
                    )
                try:
                    return self._native_generate_once(
                        retry_prompt,
                        schema=schema,
                        force_stream=False,
                        output_tokens=output_limits[attempt],
                        json_mode_only=True,
                    )
                except _OllamaOutputError as exc:
                    last_error = exc
                    if first_error is None:
                        first_error = exc
            except _OllamaOutputError as exc:
                last_error = exc
                if attempt == 0:
                    first_error = exc
                if attempt < 2 and self._progress is not None:
                    self._progress(
                        "message",
                        "Ollama returned incomplete structured output; retrying with a larger output budget",
                    )
        assert last_error is not None
        detail = str(last_error)
        if first_error is not None and str(first_error) != detail:
            detail = f"{detail}; initial response: {first_error}"
        raise LLMError(f"Ollama analysis failed: {detail}") from last_error

    def _openai_compatible_generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        endpoint = f"{self.base_url}/v1/chat/completions"
        check_cancelled(self._cancel_event)
        response = self.client.post(
            endpoint,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
        )
        if response.is_error:
            raise self._http_error(response, endpoint=endpoint)
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(
                "Ollama analysis failed: OpenAI-compatible endpoint returned an empty or invalid JSON envelope"
            ) from exc
        if not isinstance(payload, dict):
            raise LLMError("Ollama returned an unexpected OpenAI-compatible payload")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMError("Ollama OpenAI-compatible response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMError("Ollama OpenAI-compatible response has no message content")
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        check_cancelled(self._cancel_event)
        if self._progress is not None:
            self._progress("output", content)
        try:
            data = self._decode_json_content(content)
        except _OllamaOutputError as exc:
            raise LLMError(f"Ollama analysis failed: {exc}") from exc
        return StructuredJSONResponse(data=data, usage=usage, raw=payload)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        try:
            native = self._native_generate(prompt, schema=schema)
            if native is not None:
                return native
            return self._openai_compatible_generate(
                prompt, schema=schema, schema_name=schema_name
            )
        except LLMError:
            raise
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot connect to Ollama at {self.base_url}. Start it with `ollama serve` "
                "or your operating system's service manager, then retry."
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            check_cancelled(self._cancel_event)
            raise LLMError(f"Ollama analysis failed: {exc}") from exc

    def analyze(self, prompt: str) -> LLMResponse:
        try:
            response = self.generate_json(
                prompt, schema=ANALYSIS_JSON_SCHEMA, schema_name="paper_analysis"
            )
            result = AnalysisResult.model_validate(response.data).model_copy(
                update={"generated_by_model": model_attribution(self.model)}
            )
            return LLMResponse(result=result, usage=response.usage, raw=response.raw)
        except LLMError:
            raise
        except ValidationError as exc:
            raise LLMError(f"Ollama analysis failed: {exc}") from exc


class OpenAIClient:
    provider = "openai"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        reasoning_effort: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model
        normalized = base_url.rstrip("/")
        self.base_url = normalized if normalized.endswith("/v1") else f"{normalized}/v1"
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self._progress: ProgressCallback | None = None
        self._cancel_event: threading.Event | None = None
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def set_runtime(
        self, progress: ProgressCallback | None, cancel_event: threading.Event | None
    ) -> None:
        self._progress = progress
        self._cancel_event = cancel_event

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def check_ready(self) -> None:
        if not self.api_key:
            raise LLMError(f"{self.provider_name} API key is missing")

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return error["message"].strip()
            for key in ("message", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        text = response.text.strip()
        return text[:500] if text else f"HTTP {response.status_code}"

    def _http_error(self, response: httpx.Response) -> LLMError:
        detail = self._error_detail(response)
        if response.status_code in {400, 404} and "model" in detail.casefold():
            return LLMError(
                f"{self.provider_name} analysis failed: model {self.model!r} is unavailable. "
                "Verify the model ID and that your OpenAI API project has access. "
                f"Server response: {detail}"
            )
        if response.status_code in {401, 403}:
            return LLMError(
                f"{self.provider_name} analysis failed: the API key was rejected or lacks access. "
                f"Server response: {detail}"
            )
        return LLMError(
            f"{self.provider_name} analysis failed: HTTP {response.status_code}: {detail}"
        )

    def _request_payload(
        self, prompt: str, *, schema: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        strict_schema = copy.deepcopy(schema)

        def normalize(value: Any) -> None:
            if isinstance(value, dict):
                value.pop("default", None)
                properties = value.get("properties")
                if isinstance(properties, dict):
                    value["required"] = list(properties)
                    value["additionalProperties"] = False
                for child in value.values():
                    normalize(child)
            elif isinstance(value, list):
                for child in value:
                    normalize(child)

        normalize(strict_schema)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "description": "Structured JSON response",
                    "schema": strict_schema,
                    "strict": True,
                }
            },
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    @staticmethod
    def _output_text(payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text
        output = payload.get("output", [])
        if output is None:
            output = []
        if not isinstance(output, list):
            raise LLMError("OpenAI response has malformed output")
        for item in output:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content") or []
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"]
        raise LLMError("OpenAI response has no output text")

    def _structured_response(
        self, payload: Any
    ) -> StructuredJSONResponse:
        if not isinstance(payload, dict):
            raise LLMError(f"{self.provider_name} returned an unexpected payload")
        import json

        data = json.loads(self._output_text(payload))
        if not isinstance(data, dict):
            raise LLMError(f"{self.provider_name} structured output must be a JSON object")
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return StructuredJSONResponse(data=data, usage=usage, raw=payload)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        if not self.api_key:
            raise LLMError(f"{self.provider_name} API key is missing")
        check_cancelled(self._cancel_event)
        try:
            response = self.client.post(
                f"{self.base_url}/responses",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._request_payload(prompt, schema=schema, schema_name=schema_name),
            )
            if response.is_error:
                raise self._http_error(response)
            check_cancelled(self._cancel_event)
            structured = self._structured_response(response.json())
            if self._progress is not None:
                import json
                self._progress("output", json.dumps(structured.data, ensure_ascii=False))
            return structured
        except LLMError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            check_cancelled(self._cancel_event)
            raise LLMError(f"{self.provider_name} analysis failed: {exc}") from exc

    def analyze(self, prompt: str) -> LLMResponse:
        try:
            response = self.generate_json(
                prompt, schema=ANALYSIS_JSON_SCHEMA, schema_name="paper_analysis"
            )
            result = AnalysisResult.model_validate(response.data).model_copy(
                update={"generated_by_model": model_attribution(self.model)}
            )
            return LLMResponse(result=result, usage=response.usage, raw=response.raw)
        except LLMError:
            raise
        except ValidationError as exc:
            raise LLMError(f"{self.provider_name} analysis failed: {exc}") from exc



class CodexClient:
    """Codex CLI headless mode using the user's existing local authentication."""

    provider = "codex"

    def __init__(
        self,
        *,
        model: str,
        command: str = "codex",
        timeout_seconds: float = 600,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self._progress: ProgressCallback | None = None
        self._cancel_event: threading.Event | None = None
        self._process: subprocess.Popen[str] | None = None

    def set_runtime(
        self, progress: ProgressCallback | None, cancel_event: threading.Event | None
    ) -> None:
        self._progress = progress
        self._cancel_event = cancel_event

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def check_ready(self) -> None:
        executable = shutil.which(self.command)
        if executable is None and not Path(self.command).is_file():
            raise LLMError(
                f"Codex CLI executable {self.command!r} was not found; install Codex or set command"
            )

    @staticmethod
    def _normalized_schema(schema: dict[str, Any]) -> dict[str, Any]:
        strict_schema = copy.deepcopy(schema)

        def normalize(value: Any) -> None:
            if isinstance(value, dict):
                value.pop("default", None)
                properties = value.get("properties")
                if isinstance(properties, dict):
                    value["required"] = list(properties)
                    value["additionalProperties"] = False
                for child in value.values():
                    normalize(child)
            elif isinstance(value, list):
                for child in value:
                    normalize(child)

        normalize(strict_schema)
        return strict_schema

    def _command(
        self,
        schema_path: Path | None,
        output_path: Path,
        *,
        include_model: bool = True,
    ) -> list[str]:
        command = [
            self.command,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
        ]
        if schema_path is not None:
            command.extend(("--output-schema", str(schema_path)))
        command.extend(("--output-last-message", str(output_path)))
        if include_model and self.model:
            command.extend(("--model", self.model))
        if self.reasoning_effort:
            command.extend(
                ("--config", f'model_reasoning_effort="{self.reasoning_effort}"')
            )
        command.append("-")
        return command

    @staticmethod
    def _failure_detail(stdout: str, stderr: str) -> str:
        ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        text = ansi.sub("", stderr or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        markers = (
            "error", "failed", "failure", "invalid", "unsupported", "denied",
            "authentication", "not logged", "login", "rate limit", "too large",
            "context length", "response_format", "output schema", "output-schema",
            "text.format.schema", "timed out", "timeout",
        )
        useful = [line for line in lines if any(marker in line.lower() for marker in markers)]
        if useful:
            detail = "\n".join(useful[-12:])
        elif lines:
            # Codex prints its banner and prompt first; the useful failure is normally at the end.
            detail = "\n".join(lines[-20:])
        else:
            out_lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
            detail = "\n".join(out_lines[-20:]) if out_lines else "Codex CLI exited unsuccessfully"
        return detail[-4000:]

    @staticmethod
    def _schema_failure(detail: str) -> bool:
        lowered = detail.lower()
        return (
            "invalid_json_schema" in lowered
            or "response_format" in lowered
            or "text.format.schema" in lowered
            or "output schema" in lowered
            or "output-schema" in lowered
            or ("schema" in lowered and "invalid" in lowered)
        )

    @staticmethod
    def _chatgpt_model_failure(detail: str) -> bool:
        lowered = detail.lower()
        return (
            "model is not supported when using codex with a chatgpt account" in lowered
            or (
                "not supported" in lowered
                and "codex" in lowered
                and "chatgpt account" in lowered
            )
        )

    @staticmethod
    def _reported_model(stdout: str, stderr: str) -> str | None:
        for text in (stderr, stdout):
            match = re.search(r"(?mi)^model:\s*([^\r\n]+?)\s*$", text or "")
            if match:
                model = match.group(1).strip()
                if model:
                    return model
        return None

    @staticmethod
    def _single_shot_prompt(prompt: str, *, allow_local_pdf: bool) -> str:
        if allow_local_pdf:
            prefix = (
                "You are being used as a single-shot analysis backend. Inspect only the exact "
                "local PDF named in this prompt when needed. Do not inspect the surrounding "
                "repository or unrelated files, and do not modify anything. "
            )
        else:
            prefix = (
                "You are being used as a single-shot analysis backend. Use only the material "
                "contained in this prompt. Do not inspect files, run shell commands, call tools "
                "or MCP servers, or modify anything. "
            )
        return prefix + "Return only the requested JSON object, with no Markdown fences or prose.\n\n" + prompt

    def _generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        pdf_path: Path | None = None,
    ) -> StructuredJSONResponse:
        self.check_ready()
        check_cancelled(self._cancel_event)
        process_prompt = prompt
        cwd: str | None = None
        path: Path | None = None
        if pdf_path is not None:
            path = Path(pdf_path).expanduser().resolve()
            if not path.is_file():
                raise LLMError(f"PDF file does not exist: {path}")
            cwd = str(path.parent)
            process_prompt = (
                f"The complete research-paper PDF is the local file {json.dumps(str(path))}. "
                "Use read-only local tools as needed to inspect that exact PDF and analyze the "
                "complete document, including figures, tables, appendices, footnotes, and "
                "bibliography. Do not substitute an extracted-text summary.\n\n"
                f"{prompt}"
            )
        process_prompt = self._single_shot_prompt(
            process_prompt, allow_local_pdf=path is not None
        )

        with tempfile.TemporaryDirectory(prefix="paperwatch-codex-") as temp_dir:
            temp = Path(temp_dir)
            if cwd is None:
                # Do not inherit Paperwatch's repository as the Codex working directory. That
                # can load project instructions, MCP servers, skills, and unrelated repo context.
                cwd = str(temp)
            schema_path = temp / "schema.json"
            output_path = temp / "last-message.json"
            schema_path.write_text(
                json.dumps(self._normalized_schema(schema), ensure_ascii=False),
                encoding="utf-8",
            )
            popen_kwargs: dict[str, Any] = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
            }
            if cwd is not None:
                popen_kwargs["cwd"] = cwd
            def run(
                schema: Path | None, *, include_model: bool
            ) -> tuple[subprocess.Popen[str], str, str]:
                try:
                    current = subprocess.Popen(
                        self._command(schema, output_path, include_model=include_model),
                        **popen_kwargs,
                    )
                    self._process = current
                    run_stdout, run_stderr = current.communicate(
                        process_prompt, timeout=self.timeout_seconds
                    )
                    return current, run_stdout, run_stderr
                except subprocess.TimeoutExpired as exc:
                    current.terminate()
                    try:
                        current.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        current.kill()
                        current.communicate()
                    raise LLMError(
                        f"Codex CLI analysis timed out after {self.timeout_seconds:g} seconds"
                    ) from exc
                except OSError as exc:
                    raise LLMError(f"Codex CLI could not start: {exc}") from exc
                finally:
                    self._process = None

            active_schema: Path | None = schema_path
            include_model = bool(self.model)
            model_fallback = False
            while True:
                process, stdout, stderr = run(active_schema, include_model=include_model)
                check_cancelled(self._cancel_event)
                if process.returncode == 0:
                    break
                detail = self._failure_detail(stdout, stderr)
                retry = False
                if include_model and self._chatgpt_model_failure(detail):
                    # ChatGPT-authenticated Codex exposes a different model set from the API.
                    # Let the installed CLI choose an account-compatible default rather than
                    # hard-coding a model name whose availability can change over time.
                    include_model = False
                    model_fallback = True
                    retry = True
                elif active_schema is not None and self._schema_failure(detail):
                    # Structured-output support has changed across CLI releases. Retry without
                    # the CLI schema and validate the returned JSON locally.
                    active_schema = None
                    retry = True
                if not retry:
                    raise LLMError(f"Codex CLI analysis failed: {detail}")
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    pass
            output_text = ""
            try:
                if output_path.is_file():
                    output_text = output_path.read_text(encoding="utf-8").strip()
            except OSError:
                output_text = ""
            if not output_text:
                output_text = stdout.strip()
            try:
                data = json.loads(output_text)
            except json.JSONDecodeError as exc:
                raise LLMError("Codex CLI returned invalid structured JSON") from exc
            if not isinstance(data, dict):
                raise LLMError("Codex CLI structured output must be a JSON object")
            usage: dict[str, Any] = {"provider": "codex", "transport": "cli"}
            if model_fallback or not self.model:
                effective_model = self._reported_model(stdout, stderr) or "Codex CLI default"
                usage["effective_model"] = effective_model
                if model_fallback:
                    usage["requested_model"] = self.model
                    usage["model_fallback"] = True
            if path is not None:
                usage["input_file_bytes"] = path.stat().st_size
            if self._progress is not None:
                self._progress("output", json.dumps(data, ensure_ascii=False))
            return StructuredJSONResponse(
                data=data,
                usage=usage,
                raw={"output": output_text},
            )

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        del schema_name
        return self._generate_json(prompt, schema=schema)

    def generate_json_with_pdf(
        self,
        prompt: str,
        pdf_path: Path,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        del schema_name
        return self._generate_json(prompt, schema=schema, pdf_path=Path(pdf_path))

    def analyze(self, prompt: str) -> LLMResponse:
        try:
            response = self.generate_json(
                prompt, schema=ANALYSIS_JSON_SCHEMA, schema_name="paper_analysis"
            )
            effective_model = str(
                response.usage.get("effective_model") or self.model or "Codex CLI default"
            )
            result = AnalysisResult.model_validate(response.data).model_copy(
                update={"generated_by_model": model_attribution(effective_model)}
            )
            return LLMResponse(result=result, usage=response.usage, raw=response.raw)
        except LLMError:
            raise
        except ValidationError as exc:
            raise LLMError(f"Codex CLI analysis failed: {exc}") from exc

    def analyze_pdf(self, prompt: str, pdf_path: Path) -> LLMResponse:
        try:
            response = self.generate_json_with_pdf(
                prompt,
                pdf_path,
                schema=ANALYSIS_JSON_SCHEMA,
                schema_name="paper_analysis",
            )
            effective_model = str(
                response.usage.get("effective_model") or self.model or "Codex CLI default"
            )
            result = AnalysisResult.model_validate(response.data).model_copy(
                update={"generated_by_model": model_attribution(effective_model)}
            )
            return LLMResponse(result=result, usage=response.usage, raw=response.raw)
        except LLMError:
            raise
        except ValidationError as exc:
            raise LLMError(f"Codex CLI PDF analysis failed: {exc}") from exc


class ClaudeCodeClient:
    """Claude Code headless mode using the user's existing local authentication."""

    provider = "claude-code"

    def __init__(
        self,
        *,
        model: str,
        command: str = "claude",
        timeout_seconds: float = 600,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self._progress: ProgressCallback | None = None
        self._cancel_event: threading.Event | None = None
        self._process: subprocess.Popen[str] | None = None

    def set_runtime(
        self, progress: ProgressCallback | None, cancel_event: threading.Event | None
    ) -> None:
        self._progress = progress
        self._cancel_event = cancel_event

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def check_ready(self) -> None:
        executable = shutil.which(self.command)
        if executable is None and not Path(self.command).is_file():
            raise LLMError(
                f"Claude Code executable {self.command!r} was not found; install Claude Code or set command"
            )

    def _command(
        self, schema: dict[str, Any], *, allow_pdf_read: bool = False
    ) -> list[str]:
        tools = "Read" if allow_pdf_read else ""
        system_prompt = (
            "You are a careful research-paper analyst. Use only the Read tool to read "
            "the PDF path supplied by the user, read the complete document, and follow "
            "the user's output schema exactly."
            if allow_pdf_read
            else "You are a careful research-paper analyst. Do not use tools. Follow the "
            "user's source and output schema exactly."
        )
        command = [
            self.command,
            "-p",
            "Analyze the research-paper request supplied on stdin and return only the requested structured output.",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            "--model",
            self.model,
            "--safe-mode",
            "--tools",
            tools,
            "--strict-mcp-config",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--system-prompt",
            system_prompt,
        ]
        if self.reasoning_effort:
            command.extend(("--effort", self.reasoning_effort))
        return command

    def _generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        pdf_path: Path | None = None,
    ) -> StructuredJSONResponse:
        self.check_ready()
        check_cancelled(self._cancel_event)
        process_prompt = prompt
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
        }
        if pdf_path is not None:
            path = Path(pdf_path).resolve()
            if not path.is_file():
                raise LLMError(f"PDF file does not exist: {path}")
            process_prompt = (
                f"The complete PDF is the local file {json.dumps(str(path))}. Use the Read "
                "tool to read the PDF directly and in full, "
                "including figures, tables, appendices, footnotes, and bibliography. Do not "
                f"use an extracted-text substitute.\n\n{prompt}"
            )
            popen_kwargs["cwd"] = str(path.parent)
        try:
            process = subprocess.Popen(
                self._command(schema, allow_pdf_read=pdf_path is not None),
                **popen_kwargs,
            )
            self._process = process
            stdout, stderr = process.communicate(
                process_prompt, timeout=self.timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise LLMError(
                f"Claude Code analysis timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise LLMError(f"Claude Code could not start: {exc}") from exc
        finally:
            self._process = None
        check_cancelled(self._cancel_event)
        if process.returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"exit status {process.returncode}"
            raise LLMError(f"Claude Code analysis failed: {detail[:1000]}")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LLMError("Claude Code returned an invalid JSON envelope") from exc
        if not isinstance(payload, dict):
            raise LLMError("Claude Code returned an unexpected response envelope")
        data = payload.get("structured_output")
        if not isinstance(data, dict):
            raise LLMError("Claude Code returned no structured_output object")
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        for key in ("total_cost_usd", "duration_ms", "duration_api_ms", "num_turns"):
            if key in payload:
                usage[key] = payload[key]
        if pdf_path is not None:
            usage["input_file_bytes"] = Path(pdf_path).stat().st_size
        if self._progress is not None:
            self._progress("output", json.dumps(data, ensure_ascii=False))
        return StructuredJSONResponse(data=data, usage=usage, raw=payload)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        del schema_name
        return self._generate_json(prompt, schema=schema)

    def generate_json_with_pdf(
        self,
        prompt: str,
        pdf_path: Path,
        *,
        schema: dict[str, Any],
        schema_name: str,
    ) -> StructuredJSONResponse:
        del schema_name
        return self._generate_json(prompt, schema=schema, pdf_path=Path(pdf_path))

    def analyze(self, prompt: str) -> LLMResponse:
        try:
            response = self.generate_json(
                prompt, schema=ANALYSIS_JSON_SCHEMA, schema_name="paper_analysis"
            )
            result = AnalysisResult.model_validate(response.data).model_copy(
                update={"generated_by_model": model_attribution(self.model)}
            )
            return LLMResponse(result=result, usage=response.usage, raw=response.raw)
        except LLMError:
            raise
        except ValidationError as exc:
            raise LLMError(f"Claude Code analysis failed: {exc}") from exc

    def analyze_pdf(self, prompt: str, pdf_path: Path) -> LLMResponse:
        try:
            response = self._generate_json(
                prompt, schema=ANALYSIS_JSON_SCHEMA, pdf_path=Path(pdf_path)
            )
            result = AnalysisResult.model_validate(response.data).model_copy(
                update={"generated_by_model": model_attribution(self.model)}
            )
            return LLMResponse(result=result, usage=response.usage, raw=response.raw)
        except LLMError:
            raise
        except (OSError, ValidationError) as exc:
            raise LLMError(f"Claude Code PDF analysis failed: {exc}") from exc


def configure_job_runtime(
    llm: LLMClient,
    progress: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> None:
    setter = getattr(llm, "set_runtime", None)
    if callable(setter):
        setter(progress, cancel_event)


def configure_context_window(llm: LLMClient, context_window_tokens: int) -> None:
    """Apply a per-job context override when the provider exposes one."""
    if isinstance(llm, OllamaClient):
        llm.context_window_tokens = context_window_tokens


def create_llm(config: LLMConfig, *, client: httpx.Client | None = None) -> LLMClient:
    if config.provider == "none":
        return NoLLMClient()
    if config.provider == "ollama":
        return OllamaClient(
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            context_window_tokens=config.context_window_tokens,
            client=client,
        )
    if config.provider == "openai":
        return OpenAIClient(
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            reasoning_effort=config.reasoning_effort,
            client=client,
        )
    if config.provider == "codex":
        return CodexClient(
            model=config.model,
            command=config.command or "codex",
            timeout_seconds=config.timeout_seconds,
            reasoning_effort=config.reasoning_effort or "medium",
        )
    if config.provider == "claude-code":
        return ClaudeCodeClient(
            model=config.model,
            command=config.command,
            timeout_seconds=config.timeout_seconds,
            reasoning_effort=config.reasoning_effort,
        )
    raise LLMError(f"Unsupported LLM provider: {config.provider}")


class PromptBuilder:
    PLACEHOLDERS = (
        "title",
        "authors",
        "publication_date",
        "venue",
        "source",
        "scope",
        "model_name",
        "text",
    )

    def __init__(self, template_path: Path, max_input_chars: int):
        try:
            self.template = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMError(f"Cannot read prompt template: {exc}") from exc
        if max_input_chars <= 0:
            raise LLMError("max_input_chars must be greater than zero")
        if "{text}" not in self.template:
            raise LLMError("Prompt template must contain the {text} placeholder")
        self.max_input_chars = max_input_chars

    def build(self, paper: PaperRecord, text: str, scope: str, model_name: str) -> str:
        if len(text) > self.max_input_chars:
            raise LLMError(
                f"Input contains {len(text)} characters, exceeding llm.max_input_chars="
                f"{self.max_input_chars}; use chunked analysis or increase the limit to avoid partial analysis"
            )
        authors = ", ".join(author.name for author in paper.authors) or "Unknown"
        source = paper.doi or paper.arxiv_id or paper.openalex_id or "Unknown"
        values = {
            "title": paper.title,
            "authors": authors,
            "publication_date": paper.publication_date or "Unknown",
            "venue": paper.venue or "Unknown",
            "source": source,
            "scope": scope,
            "model_name": model_name,
            "text": text,
        }
        prompt = self.template
        for key in self.PLACEHOLDERS:
            prompt = prompt.replace("{" + key + "}", values[key])
        return prompt
