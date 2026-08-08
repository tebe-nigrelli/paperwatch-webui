from __future__ import annotations

from pathlib import Path

import pytest

from paperwatch.config import ConfigError, load_config


def test_load_config_resolves_paths_and_creates_pdf_dir(config_path: Path):
    cfg = load_config(config_path)
    assert cfg.paths.database.is_absolute()
    assert cfg.paths.pdf_dir.exists()
    assert cfg.llm.provider == "ollama"


def test_environment_keys_override_file(config_path: Path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "env-openalex")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
    text = config_path.read_text().replace('provider = "ollama"', 'provider = "openai"')
    text = text.replace('base_url = "http://ollama.test"', 'base_url = "https://api.openai.com"')
    config_path.write_text(text)
    cfg = load_config(config_path)
    assert cfg.openalex.api_key == "env-openalex"
    assert cfg.llm.api_key == "env-openai"


def test_invalid_provider_is_rejected(config_path: Path):
    config_path.write_text(config_path.read_text().replace('provider = "ollama"', 'provider = "bogus"'))
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_missing_config_is_rejected(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.toml")


def test_invalid_toml_and_non_table_are_rejected(tmp_path: Path):
    invalid = tmp_path / "invalid.toml"
    invalid.write_text("[paths\n")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(invalid)

    non_table = tmp_path / "non-table.toml"
    non_table.write_text('paths = "wrong"')
    with pytest.raises(ConfigError, match=r"\[paths\] must be a table"):
        load_config(non_table)


def test_boolean_and_numeric_values_are_validated(config_path: Path):
    text = config_path.read_text().replace("download_pdfs = true", 'download_pdfs = "false"')
    config_path.write_text(text)
    with pytest.raises(ConfigError, match="fetch.download_pdfs"):
        load_config(config_path)

    text = text.replace('download_pdfs = "false"', "download_pdfs = true")
    text = text.replace("max_pdf_bytes = 1000000", "max_pdf_bytes = 0")
    config_path.write_text(text)
    with pytest.raises(ConfigError, match="max_pdf_bytes"):
        load_config(config_path)


def test_all_range_and_required_value_checks(config_path: Path, monkeypatch):
    replacements = [
        ("overlap_days = 7", "overlap_days = -1", "day windows"),
        ("min_interval_seconds = 0", "min_interval_seconds = -1", "cannot be negative"),
        ("title_match_threshold = 0.9", "title_match_threshold = 1.1", "between 0 and 1"),
        ('model = "test-model"', 'model = ""', "cannot be empty"),
    ]
    original = config_path.read_text()
    for old, new, message in replacements:
        config_path.write_text(original.replace(old, new))
        with pytest.raises(ConfigError, match=message):
            load_config(config_path)

    openai = original.replace('provider = "ollama"', 'provider = "openai"')
    openai = openai.replace('base_url = "http://ollama.test"', 'base_url = "https://api.openai.com"')
    config_path.write_text(openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_config(config_path)


def test_none_provider_allows_empty_model(config_path: Path):
    text = config_path.read_text().replace('provider = "ollama"', 'provider = "none"')
    text = text.replace('model = "test-model"', 'model = ""')
    config_path.write_text(text)
    assert load_config(config_path).llm.provider == "none"


def test_invalid_scalar_conversion_is_config_error(config_path: Path):
    config_path.write_text(config_path.read_text().replace("timeout_seconds = 1", 'timeout_seconds = "x"', 1))
    with pytest.raises(ConfigError, match="Invalid configuration value"):
        load_config(config_path)


def test_multiple_llm_profiles_are_loaded_and_selectable(config_path: Path):
    config_path.write_text(
        config_path.read_text()
        + """

[[llm_profiles]]
name = "local-large"
provider = "ollama"
model = "qwen3:30b"
base_url = "http://second-ollama.test"
timeout_seconds = 9
max_input_chars = 9000
analyze_pdfs = false
prompt_version = "profile-v1"
"""
    )
    cfg = load_config(config_path)
    assert [item.profile for item in cfg.available_llms] == ["default", "local-large"]
    selected = cfg.llm_profile("local-large")
    assert selected.model == "qwen3:30b"
    assert selected.base_url == "http://second-ollama.test"
    assert selected.analyze_pdfs is False
    assert cfg.llm_profile("missing") is None


def test_llm_profile_validation(config_path: Path):
    original = config_path.read_text()
    cases = [
        (
            """
[[llm_profiles]]
provider = "ollama"
model = "m"
""",
            "requires a name",
        ),
        (
            """
[[llm_profiles]]
name = "default"
provider = "ollama"
model = "m"
""",
            "Duplicate",
        ),
        (
            """
[[llm_profiles]]
name = "bad name"
provider = "ollama"
model = "m"
""",
            "profile name",
        ),
        (
            """
[[llm_profiles]]
name = "missing-model"
provider = "ollama"
""",
            "requires a model",
        ),
        (
            """
[[llm_profiles]]
name = "bad-provider"
provider = "other"
model = "m"
""",
            "provider",
        ),
    ]
    for extra, message in cases:
        config_path.write_text(original + extra)
        with pytest.raises(ConfigError, match=message):
            load_config(config_path)

    config_path.write_text("llm_profiles = \"wrong\"\n" + original)
    with pytest.raises(ConfigError, match="array of tables"):
        load_config(config_path)


def test_openai_profile_uses_v1_default_and_environment_key(config_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    config_path.write_text(
        config_path.read_text()
        + """

[[llm_profiles]]
name = "remote"
provider = "openai"
model = "gpt-test"
"""
    )
    profile = load_config(config_path).llm_profile("remote")
    assert profile.base_url == "https://api.openai.com/v1"
    assert profile.api_key == "secret"



def test_codex_profile_uses_cli_defaults_without_api_key(config_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path.write_text(
        config_path.read_text()
        + """

[[llm_profiles]]
name = "codex-review"
provider = "codex"
model = "gpt-5.3-codex"
context_window_tokens = 400000
"""
    )
    profile = load_config(config_path).llm_profile("codex-review")
    assert profile.provider == "codex"
    assert profile.base_url == ""
    assert profile.api_key == ""
    assert profile.command == "codex"
    assert profile.reasoning_effort == "medium"
    assert profile.context_window_tokens == 400000


def test_codex_reasoning_effort_is_validated(config_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "codex-secret")
    original = config_path.read_text()
    config_path.write_text(
        original
        + """

[[llm_profiles]]
name = "codex-bad"
provider = "codex"
model = "gpt-5.3-codex"
reasoning_effort = "max"
"""
    )
    with pytest.raises(ConfigError, match="Codex profile.*low, medium, high, or xhigh"):
        load_config(config_path)

    config_path.write_text(
        original
        + """

[[llm_profiles]]
name = "openai-bad"
provider = "openai"
model = "gpt-test"
reasoning_effort = "turbo"
"""
    )
    with pytest.raises(ConfigError, match="reasoning_effort"):
        load_config(config_path)

def test_fetch_defaults_cover_one_year_and_large_complete_pdfs(tmp_path: Path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{text}")
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[paths]
database = "{(tmp_path / 'paperwatch.db').as_posix()}"
pdf_dir = "{(tmp_path / 'pdfs').as_posix()}"
prompt = "{prompt.as_posix()}"

[openalex]
api_key = "key"

[fetch]

[llm]
provider = "none"
model = ""
"""
    )
    loaded = load_config(config)
    assert loaded.fetch.initial_lookback_days == 365
    assert loaded.fetch.download_pdfs is True
    assert loaded.fetch.max_pdf_chars == 2_000_000
    assert loaded.fetch.max_papers_per_author == 500


def test_reasoning_effort_is_rejected_for_ollama(config_path: Path):
    config_path.write_text(
        config_path.read_text().replace(
            'prompt_version = "test-v1"',
            'prompt_version = "test-v1"\nreasoning_effort = "high"',
            1,
        )
    )
    with pytest.raises(ConfigError, match="only supported for openai, codex, or claude-code"):
        load_config(config_path)


def test_claude_code_profile_uses_local_command_without_api_key(config_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path.write_text(
        config_path.read_text()
        + """

[[llm_profiles]]
name = "claude-review"
provider = "claude-code"
model = "claude-sonnet-4-5"
command = "/opt/claude/bin/claude"
reasoning_effort = "high"
"""
    )
    profile = load_config(config_path).llm_profile("claude-review")
    assert profile.provider == "claude-code"
    assert profile.command == "/opt/claude/bin/claude"
    assert profile.api_key == ""
    assert profile.reasoning_effort == "high"


def test_claude_code_reasoning_effort_is_validated(config_path: Path):
    config_path.write_text(
        config_path.read_text()
        + """

[[llm_profiles]]
name = "claude-bad"
provider = "claude-code"
model = "claude-sonnet-4-5"
reasoning_effort = "minimal"
"""
    )
    with pytest.raises(ConfigError, match="Claude Code profile.*low, medium, high, xhigh, or max"):
        load_config(config_path)


def test_pdf_retry_and_analysis_defaults_and_overrides(config_path: Path):
    loaded = load_config(config_path)
    assert loaded.fetch.retry_missing_pdf_days == 90
    assert loaded.llm.analyze_abstracts is True
    assert loaded.llm.pdf_analysis_strategy == "chunked"

    text = config_path.read_text()
    text = text.replace(
        "max_pdf_chars = 10000",
        "max_pdf_chars = 10000\nretry_missing_pdf_days = 14",
    )
    text = text.replace(
        'analyze_pdfs = true',
        'analyze_abstracts = false\nanalyze_pdfs = true\npdf_analysis_strategy = "direct"',
    )
    config_path.write_text(text)
    loaded = load_config(config_path)
    assert loaded.fetch.retry_missing_pdf_days == 14
    assert loaded.llm.analyze_abstracts is False
    assert loaded.llm.pdf_analysis_strategy == "direct"


def test_new_pdf_retry_and_strategy_values_are_validated(config_path: Path):
    original = config_path.read_text()
    config_path.write_text(
        original.replace(
            "max_pdf_chars = 10000",
            "max_pdf_chars = 10000\nretry_missing_pdf_days = 0",
        )
    )
    with pytest.raises(ConfigError, match="retry_missing_pdf_days"):
        load_config(config_path)

    config_path.write_text(
        original.replace(
            'analyze_pdfs = true',
            'analyze_pdfs = true\npdf_analysis_strategy = "sometimes"',
        )
    )
    with pytest.raises(ConfigError, match="pdf_analysis_strategy"):
        load_config(config_path)
def test_codex_profile_without_model_does_not_inherit_other_provider_model(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[paths]
database = "paperwatch.db"
pdf_dir = "pdfs"
prompt = "prompts/analyze.txt"

[llm]
provider = "openai"
model = "gpt-api-only"
api_key = "test"

[[llm_profiles]]
name = "codex-review"
provider = "codex"
command = "codex"
"""
    )
    config = load_config(path)
    profile = config.llm_profile("codex-review")
    assert profile is not None
    assert profile.model == ""
