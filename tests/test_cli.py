from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from paperwatch import cli
from paperwatch.db import Database
from paperwatch.models import AuthorCandidate


class FakeOpenAlex:
    def search_authors(self, name, limit=10):
        return [
            AuthorCandidate(
                openalex_id="A123",
                name="Ada Lovelace",
                institution="Analytical Engine Institute",
                works_count=42,
                cited_by_count=9001,
            )
        ]

    def get_author(self, author_id):
        return AuthorCandidate(
            openalex_id="A123",
            name="Ada Lovelace",
            institution="Analytical Engine Institute",
        )

    def close(self):
        pass


class FakeLLM:
    provider = "none"
    model = "none"

    def close(self):
        pass


def service_factory(config):
    return cli.Services(
        config=config,
        db=Database(config.paths.database),
        openalex=FakeOpenAlex(),
        arxiv=None,
        llm=FakeLLM(),
    )


def test_init_creates_config_and_prompt(tmp_path, capsys):
    destination = tmp_path / "config.toml"
    assert cli.main(["init", str(destination)]) == 0
    assert destination.exists()
    assert (tmp_path / "prompts" / "analyze.txt").exists()
    from paperwatch.config import load_config
    assert load_config(destination).llm.context_window_tokens == 8192
    assert str(destination) in capsys.readouterr().out
    assert cli.main(["init", str(destination)]) == 2


def test_author_search_add_list_disable_remove(config_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_services", service_factory)

    assert cli.main(["--config", str(config_path), "author", "search", "Ada"]) == 0
    assert "Ada Lovelace" in capsys.readouterr().out

    assert cli.main(["--config", str(config_path), "author", "add", "A123"]) == 0
    assert "Added Ada Lovelace" in capsys.readouterr().out

    assert cli.main(["--config", str(config_path), "author", "list"]) == 0
    assert "enabled" in capsys.readouterr().out

    assert cli.main(["--config", str(config_path), "author", "disable", "A123"]) == 0
    assert "disabled" in capsys.readouterr().out

    assert cli.main(["--config", str(config_path), "author", "remove", "A123"]) == 0
    assert "Removed A123" in capsys.readouterr().out


def test_run_command_prints_stats(config_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_services", service_factory)

    class FakePdfClient:
        def close(self):
            pass

    class FakePaperWatch:
        def __init__(self, *args, **kwargs):
            self._owns_pdf_client = True
            self.pdf_client = FakePdfClient()

        def run(self, today=None):
            return SimpleNamespace(
                run_id=7,
                status="success",
                authors_checked=2,
                papers_found=3,
                papers_analyzed=4,
                pdfs_downloaded=1,
                errors_count=0,
            )

        def close(self):
            self.pdf_client.close()

    monkeypatch.setattr(cli, "PaperWatch", FakePaperWatch)
    assert cli.main(["--config", str(config_path), "run", "--today", "2026-07-24"]) == 0
    output = capsys.readouterr().out
    assert "run=7" in output
    assert "papers=3" in output


def test_papers_and_runs_commands(config_path, monkeypatch, paper_record, capsys):
    monkeypatch.setattr(cli, "build_services", service_factory)
    with Database(config_path.parent / "paperwatch.db") as db:
        db.upsert_paper(paper_record)
        run_id = db.start_run()
        db.finish_run(run_id, "success", {})

    assert cli.main(["--config", str(config_path), "papers"]) == 0
    assert paper_record.title in capsys.readouterr().out
    assert cli.main(["--config", str(config_path), "runs"]) == 0
    assert "success" in capsys.readouterr().out


def test_missing_config_returns_usage_error(tmp_path, capsys):
    assert cli.main(["--config", str(tmp_path / "missing.toml"), "author", "list"]) == 2
    assert "Config file not found" in capsys.readouterr().err


def test_services_close_and_arxiv_disabled_build(app_config):
    from dataclasses import replace

    closed = []

    class Closable:
        def close(self):
            closed.append(self)

    services = cli.Services(
        config=app_config,
        db=Closable(),
        openalex=Closable(),
        arxiv=Closable(),
        llm=Closable(),
    )
    services.close()
    assert len(closed) == 4

    config = replace(
        app_config,
        arxiv=replace(app_config.arxiv, enabled=False),
        llm=replace(app_config.llm, provider="none"),
    )
    built = cli.build_services(config)
    try:
        assert built.arxiv is None
        assert built.llm.provider == "none"
    finally:
        built.close()


def test_author_not_found_and_full_url_commands(config_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_services", service_factory)
    assert cli.main(["--config", str(config_path), "author", "enable", "A404"]) == 1
    assert "not found" in capsys.readouterr().err.lower()
    assert cli.main(["--config", str(config_path), "author", "remove", "A404"]) == 1


def test_cli_argument_validation():
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["papers", "--limit", "0"])
    with pytest.raises(SystemExit):
        cli.main(["run", "--today", "24-07-2026"])


def test_main_reports_service_and_command_errors(config_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_services", lambda config: (_ for _ in ()).throw(ValueError("bad service")))
    assert cli.main(["--config", str(config_path), "papers"]) == 2
    assert "bad service" in capsys.readouterr().err

    monkeypatch.setattr(cli, "build_services", service_factory)
    monkeypatch.setattr(cli, "_list_papers", lambda args, services: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main(["--config", str(config_path), "papers"]) == 1
    assert "boom" in capsys.readouterr().err


def test_init_preserves_existing_prompt(tmp_path):
    prompt = tmp_path / "prompts" / "analyze.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("custom {text}")
    assert cli.main(["init", str(tmp_path / "config.toml")]) == 0
    assert prompt.read_text() == "custom {text}"


def test_python_module_entrypoint_displays_help(monkeypatch, capsys):
    import runpy
    import sys
    import pytest

    monkeypatch.setattr(sys, "argv", ["paperwatch", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("paperwatch.__main__", run_name="__main__")
    assert exc.value.code == 0
    assert "Fetch and analyze recent papers" in capsys.readouterr().out


def test_web_command_runs_uvicorn(config_path, monkeypatch):
    calls = {}
    marker = object()
    monkeypatch.setattr("paperwatch.web.create_app", lambda path: marker)

    def fake_run(app, **kwargs):
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert cli.main(["--config", str(config_path), "web", "--host", "127.0.0.2", "--port", "9000", "--log-level", "debug"]) == 0
    assert calls == {"app": marker, "host": "127.0.0.2", "port": 9000, "log_level": "debug"}


def test_web_command_reports_config_error(config_path, monkeypatch, capsys):
    monkeypatch.setattr("paperwatch.web.create_app", lambda path: (_ for _ in ()).throw(ValueError("bad web config")))
    assert cli.main(["--config", str(config_path), "web"]) == 2
    assert "bad web config" in capsys.readouterr().err
