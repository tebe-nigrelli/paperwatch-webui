from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PathsConfig:
    database: Path
    pdf_dir: Path
    prompt: Path


@dataclass(frozen=True)
class OpenAlexConfig:
    api_key: str
    base_url: str
    timeout_seconds: float


@dataclass(frozen=True)
class ArxivConfig:
    enabled: bool
    base_url: str
    timeout_seconds: float
    min_interval_seconds: float
    title_match_threshold: float


@dataclass(frozen=True)
class FetchConfig:
    overlap_days: int
    initial_lookback_days: int
    max_papers_per_author: int
    download_pdfs: bool
    max_pdf_bytes: int
    max_pdf_chars: int


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float
    max_input_chars: int
    context_window_tokens: int
    analyze_pdfs: bool
    prompt_version: str
    reasoning_effort: str | None = None
    profile: str = "default"
    command: str = "claude"


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    paths: PathsConfig
    openalex: OpenAlexConfig
    arxiv: ArxivConfig
    fetch: FetchConfig
    llm: LLMConfig
    llm_profiles: tuple[LLMConfig, ...] = ()

    @property
    def available_llms(self) -> tuple[LLMConfig, ...]:
        return tuple(
            item for item in (self.llm, *self.llm_profiles) if item.provider != "none"
        )

    def llm_profile(self, name: str) -> LLMConfig | None:
        return next((item for item in self.available_llms if item.profile == name), None)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _bool(section: dict[str, Any], key: str, default: bool) -> bool:
    raw_key = key.rsplit(".", 1)[-1]
    value = section.get(raw_key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false")
    return value


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")



PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _llm_config(
    section: dict[str, Any],
    *,
    profile: str,
    defaults: dict[str, Any] | None = None,
) -> LLMConfig:
    inherited = defaults or {}
    provider = str(section.get("provider", inherited.get("provider", "ollama"))).strip().lower()
    if provider not in {"ollama", "openai", "codex", "claude-code", "none"}:
        raise ConfigError(
            f"LLM profile {profile!r} provider must be ollama, openai, codex, claude-code, or none"
        )
    if not PROFILE_RE.fullmatch(profile):
        raise ConfigError(
            f"LLM profile name {profile!r} must use letters, numbers, dots, dashes, or underscores"
        )
    default_url = (
        "https://api.openai.com/v1"
        if provider in {"openai", "codex"}
        else "http://localhost:11434"
    )
    same_provider = provider == str(inherited.get("provider", provider)).strip().lower()
    inherited_url = inherited.get("base_url") if same_provider else None
    api_key = str(section.get("api_key", inherited.get("api_key", "") if same_provider else "")).strip()
    if provider in {"openai", "codex"} and not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return LLMConfig(
        provider=provider,
        model=str(section.get("model", inherited.get("model", "qwen3:8b"))).strip(),
        base_url=str(section.get("base_url", inherited_url or default_url)).rstrip("/"),
        api_key=api_key,
        timeout_seconds=float(section.get("timeout_seconds", inherited.get("timeout_seconds", 180))),
        max_input_chars=int(section.get("max_input_chars", inherited.get("max_input_chars", 100_000))),
        context_window_tokens=int(section.get("context_window_tokens", inherited.get("context_window_tokens", 16384))),
        analyze_pdfs=_bool(section, "analyze_pdfs", bool(inherited.get("analyze_pdfs", True))),
        prompt_version=str(section.get("prompt_version", inherited.get("prompt_version", "v1"))).strip(),
        reasoning_effort=(
            str(
                section.get(
                    "reasoning_effort",
                    inherited.get("reasoning_effort", "") if same_provider else "",
                )
            ).strip().lower()
            or ("medium" if provider == "codex" else None)
        ),
        profile=profile,
        command=str(section.get("command", inherited.get("command", "claude"))).strip() or "claude",
    )


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    base = config_path.parent
    paths = _section(data, "paths")
    openalex = _section(data, "openalex")
    arxiv = _section(data, "arxiv")
    fetch = _section(data, "fetch")
    llm = _section(data, "llm")

    primary_values = dict(llm)
    if not str(primary_values.get("api_key", "")).strip():
        primary_values["api_key"] = os.getenv("OPENAI_API_KEY", "").strip()
    primary_llm = _llm_config(primary_values, profile="default")

    raw_profiles = data.get("llm_profiles", [])
    if not isinstance(raw_profiles, list):
        raise ConfigError("[[llm_profiles]] entries must be an array of tables")
    profiles: list[LLMConfig] = []
    profile_names = {"default"}
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            raise ConfigError("Each [[llm_profiles]] entry must be a table")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ConfigError("Each [[llm_profiles]] entry requires a name")
        if name in profile_names:
            raise ConfigError(f"Duplicate LLM profile name: {name}")
        if not str(raw.get("model", "")).strip():
            raise ConfigError(f"LLM profile {name!r} requires a model")
        profile_names.add(name)
        profiles.append(_llm_config(raw, profile=name, defaults=primary_values))

    openalex_key = os.getenv("OPENALEX_API_KEY", str(openalex.get("api_key", ""))).strip()

    try:
        cfg = AppConfig(
            config_path=config_path,
            paths=PathsConfig(
                database=_resolve(base, str(paths.get("database", "paperwatch.db"))),
                pdf_dir=_resolve(base, str(paths.get("pdf_dir", "data/pdfs"))),
                prompt=_resolve(base, str(paths.get("prompt", "prompts/analyze.txt"))),
            ),
            openalex=OpenAlexConfig(
                api_key=openalex_key,
                base_url=str(openalex.get("base_url", "https://api.openalex.org")).rstrip("/"),
                timeout_seconds=float(openalex.get("timeout_seconds", 30)),
            ),
            arxiv=ArxivConfig(
                enabled=_bool(arxiv, "arxiv.enabled", True),
                base_url=str(arxiv.get("base_url", "https://export.arxiv.org/api/query")),
                timeout_seconds=float(arxiv.get("timeout_seconds", 30)),
                min_interval_seconds=float(arxiv.get("min_interval_seconds", 3)),
                title_match_threshold=float(arxiv.get("title_match_threshold", 0.90)),
            ),
            fetch=FetchConfig(
                overlap_days=int(fetch.get("overlap_days", 30)),
                initial_lookback_days=int(fetch.get("initial_lookback_days", 365)),
                max_papers_per_author=int(fetch.get("max_papers_per_author", 500)),
                download_pdfs=_bool(fetch, "fetch.download_pdfs", True),
                max_pdf_bytes=int(fetch.get("max_pdf_bytes", 50_000_000)),
                max_pdf_chars=int(fetch.get("max_pdf_chars", 2_000_000)),
            ),
            llm=primary_llm,
            llm_profiles=tuple(profiles),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid configuration value: {exc}") from exc

    if cfg.fetch.overlap_days < 0 or cfg.fetch.initial_lookback_days < 1:
        raise ConfigError("Fetch day windows are invalid")
    _positive(cfg.openalex.timeout_seconds, "openalex.timeout_seconds")
    _positive(cfg.arxiv.timeout_seconds, "arxiv.timeout_seconds")
    if cfg.arxiv.min_interval_seconds < 0:
        raise ConfigError("arxiv.min_interval_seconds cannot be negative")
    _positive(cfg.fetch.max_papers_per_author, "fetch.max_papers_per_author")
    _positive(cfg.fetch.max_pdf_bytes, "fetch.max_pdf_bytes")
    _positive(cfg.fetch.max_pdf_chars, "fetch.max_pdf_chars")
    if not 0 <= cfg.arxiv.title_match_threshold <= 1:
        raise ConfigError("title_match_threshold must be between 0 and 1")
    for item in (cfg.llm, *cfg.llm_profiles):
        _positive(item.timeout_seconds, f"llm profile {item.profile} timeout_seconds")
        _positive(item.max_input_chars, f"llm profile {item.profile} max_input_chars")
        _positive(item.context_window_tokens, f"llm profile {item.profile} context_window_tokens")
        if item.provider != "none" and not item.model:
            raise ConfigError(f"LLM profile {item.profile!r} model cannot be empty")
        if item.provider not in {"openai", "codex", "claude-code"} and item.reasoning_effort is not None:
            raise ConfigError(
                f"LLM profile {item.profile!r} reasoning_effort is only supported for openai, codex, or claude-code"
            )
        if item.reasoning_effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ConfigError(
                f"LLM profile {item.profile!r} reasoning_effort must be none, minimal, low, "
                "medium, high, xhigh, or max"
            )
        if item.provider == "codex" and item.reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ConfigError(
                f"Codex profile {item.profile!r} reasoning_effort must be low, medium, high, or xhigh"
            )
        if item.provider == "claude-code" and item.reasoning_effort not in {
            None,
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ConfigError(
                f"Claude Code profile {item.profile!r} reasoning_effort must be low, medium, high, xhigh, or max"
            )
        if item.provider in {"openai", "codex"} and not item.api_key:
            provider_name = "OpenAI Codex" if item.provider == "codex" else "OpenAI"
            raise ConfigError(
                f"OPENAI_API_KEY or api_key is required for {provider_name} profile {item.profile!r}"
            )

    cfg.paths.pdf_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.database.parent.mkdir(parents=True, exist_ok=True)
    return cfg
