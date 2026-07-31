from __future__ import annotations

import inspect
import secrets
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from . import USER_AGENT
from .analysis import analyze_pdf_file, analyze_text
from .chunking import analyze_chunked_text, direct_prompt_fit
from .config import AppConfig, load_config
from .db import Database
from .job_runtime import JobCancelled, ProgressCallback, check_cancelled, publish
from .llm import (
    LLMClient,
    LLMError,
    configure_context_window,
    configure_job_runtime,
    create_llm,
)
from .locking import analysis_lock
from .pdfs import PDFError, download_pdf, extract_pdf_text, is_complete_pdf, safe_pdf_name


@dataclass(slots=True)
class AnalysisBatchSnapshot:
    running: bool = False
    batch_id: str | None = None
    paper_id: int | None = None
    scope: str | None = None
    profiles: tuple[str, ...] = ()
    analysis_strategy: str = "direct"
    context_window_tokens: int | None = None
    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    cancel_requested: bool = False
    stage: str = ""
    message: str = ""
    live_output: str = ""
    error: str | None = None


def _paper_pdf_path(
    config: AppConfig,
    db: Database,
    paper_id: int,
    pdf_client: httpx.Client,
) -> tuple[Any, Path]:
    paper = db.get_paper_record(paper_id)
    row = db.get_paper(paper_id)
    if paper is None or row is None:
        raise ValueError("Paper not found")
    root = config.paths.pdf_dir.resolve()
    path: Path | None = None
    if row["local_pdf_path"]:
        candidate = Path(row["local_pdf_path"]).expanduser().resolve()
        if (
            candidate.is_file()
            and (candidate == root or root in candidate.parents)
            and is_complete_pdf(candidate)
        ):
            path = candidate
        elif candidate.is_file() and (candidate == root or root in candidate.parents):
            candidate.unlink(missing_ok=True)
    if path is None:
        if not paper.pdf_url:
            raise ValueError("This paper has no PDF available or the cached PDF is incomplete")
        path = root / safe_pdf_name(paper.identity_key)
        download_pdf(
            paper.pdf_url,
            path,
            max_bytes=config.fetch.max_pdf_bytes,
            client=pdf_client,
        )
        db.set_local_pdf(paper_id, str(path))
    return paper, path


def _paper_text(
    config: AppConfig,
    db: Database,
    paper_id: int,
    scope: str,
    pdf_client: httpx.Client,
) -> tuple[Any, str]:
    paper = db.get_paper_record(paper_id)
    if paper is None:
        raise ValueError("Paper not found")
    if scope == "abstract":
        if not paper.abstract:
            raise ValueError("This paper has no abstract to analyze")
        return paper, paper.abstract
    if scope != "fulltext":
        raise ValueError("Analysis scope must be abstract or fulltext")
    paper, path = _paper_pdf_path(config, db, paper_id, pdf_client)
    try:
        text = extract_pdf_text(path, max_chars=config.fetch.max_pdf_chars)
    except PDFError as exc:
        if not paper.pdf_url or not str(exc).startswith("PDF text extraction failed:"):
            raise
        path.unlink(missing_ok=True)
        download_pdf(
            paper.pdf_url,
            path,
            max_bytes=config.fetch.max_pdf_bytes,
            client=pdf_client,
        )
        db.set_local_pdf(paper_id, str(path))
        text = extract_pdf_text(path, max_chars=config.fetch.max_pdf_chars)
    return paper, text


def _cancel_pending_tasks(
    db: Database, tasks: dict[str, Any], *, current_task_id: int | None = None
) -> int:
    cancelled = 0
    for task in tasks.values():
        task_id = int(task["id"])
        if task_id == current_task_id or task["status"] in {"queued", "running"}:
            db.finish_analysis_task(task_id, error="Cancelled by user")
            cancelled += 1
    return cancelled


def run_analysis_batch(
    config_path: Path,
    batch_id: str,
    paper_id: int,
    scope: str,
    profile_names: tuple[str, ...],
    analysis_strategy: str = "direct",
    context_window_tokens: int | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    active_client: Callable[[LLMClient | None], None] | None = None,
) -> dict[str, int | bool]:
    config = load_config(config_path)
    profiles = {item.profile: item for item in config.available_llms}
    selected = [profiles[name] for name in profile_names if name in profiles]
    completed = 0
    failed = 0
    publish(progress, "stage", "Preparing paper")
    with analysis_lock(config.paths.database):
        with Database(config.paths.database) as db, httpx.Client(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as pdf_client:
            tasks = {row["profile"]: row for row in db.analysis_tasks_for_batch(batch_id)}
            selected_names = {item.profile for item in selected}
            for name, task in tasks.items():
                if name not in selected_names:
                    db.finish_analysis_task(
                        int(task["id"]),
                        error=f"Configured LLM profile {name!r} is no longer available",
                    )
                    failed += 1
            try:
                check_cancelled(cancel_event)
                pdf_path: Path | None = None
                text: str | None = None
                needs_codex_file = (
                    scope == "fulltext"
                    and analysis_strategy != "chunked"
                    and any(item.provider == "codex" for item in selected)
                )
                if needs_codex_file:
                    publish(progress, "message", "Loading the source PDF")
                    paper, pdf_path = _paper_pdf_path(config, db, paper_id, pdf_client)
                else:
                    publish(
                        progress,
                        "message",
                        "Extracting PDF text" if scope == "fulltext" else "Loading abstract",
                    )
                    paper, text = _paper_text(config, db, paper_id, scope, pdf_client)
                check_cancelled(cancel_event)
            except JobCancelled:
                _cancel_pending_tasks(db, tasks)
                return {"completed": completed, "failed": failed, "cancelled": True}
            except Exception as exc:
                for task in tasks.values():
                    db.finish_analysis_task(int(task["id"]), error=str(exc))
                    failed += 1
                return {"completed": completed, "failed": failed}

            for profile in selected:
                task = tasks.get(profile.profile)
                if task is None:
                    continue
                task_id = int(task["id"])
                try:
                    check_cancelled(cancel_event)
                except JobCancelled:
                    _cancel_pending_tasks(db, tasks)
                    return {"completed": completed, "failed": failed, "cancelled": True}
                db.start_analysis_task(task_id)
                llm: LLMClient | None = None
                try:
                    publish(progress, "reset_output", "")
                    publish(progress, "stage", f"Summarizing with {profile.model}")
                    publish(progress, "message", f"Using profile {profile.profile}")
                    llm = create_llm(profile)
                    configure_job_runtime(llm, progress, cancel_event)
                    if active_client is not None:
                        active_client(llm)
                    effective_context = context_window_tokens or profile.context_window_tokens
                    configure_context_window(llm, effective_context)
                    effective_strategy = analysis_strategy
                    if scope == "abstract":
                        effective_strategy = "direct"
                    codex_pdf_upload = (
                        scope == "fulltext"
                        and profile.provider == "codex"
                        and effective_strategy != "chunked"
                    )
                    if codex_pdf_upload:
                        assert pdf_path is not None
                        publish(progress, "message", "Uploading and reading the complete PDF")
                        outcome = analyze_pdf_file(
                            db=db,
                            prompt_path=config.paths.prompt,
                            llm_config=profile,
                            llm=llm,
                            paper_id=paper_id,
                            paper=paper,
                            pdf_path=pdf_path,
                        )
                        chunk_count = 0
                    else:
                        if text is None:
                            assert pdf_path is not None
                            text = extract_pdf_text(
                                pdf_path, max_chars=config.fetch.max_pdf_chars
                            )
                        direct_fits, prompt_chars, prompt_budget = direct_prompt_fit(
                            prompt_path=config.paths.prompt,
                            max_input_chars=profile.max_input_chars,
                            paper=paper,
                            text=text,
                            scope=scope,
                            model_name=llm.model,
                            context_window_tokens=effective_context,
                        )
                        if effective_strategy == "auto":
                            effective_strategy = "direct" if direct_fits else "chunked"
                        if effective_strategy == "direct" and not direct_fits:
                            detail = (
                                f"The direct prompt needs {prompt_chars} characters but the selected "
                                f"context safely allows about {prompt_budget}; use chunked analysis or "
                                "choose a larger context window"
                                if prompt_chars
                                else "The source exceeds the configured direct-input limit; use chunked analysis"
                            )
                            raise LLMError(detail)
                        if effective_strategy == "chunked":
                            chunked = analyze_chunked_text(
                                db=db,
                                prompt_path=config.paths.prompt,
                                llm_config=profile,
                                llm=llm,
                                paper_id=paper_id,
                                paper=paper,
                                text=text,
                                context_window_tokens=effective_context,
                                progress=progress,
                                cancel_event=cancel_event,
                            )
                            outcome = chunked.outcome
                            chunk_count = chunked.chunk_count
                        else:
                            publish(progress, "message", "Generating the structured summary")
                            outcome = analyze_text(
                                db=db,
                                prompt_path=config.paths.prompt,
                                llm_config=profile,
                                llm=llm,
                                paper_id=paper_id,
                                paper=paper,
                                text=text,
                                scope=scope,
                            )
                            chunk_count = 0
                    check_cancelled(cancel_event)
                    db.finish_analysis_task(
                        task_id, analysis_id=outcome.analysis_id, chunk_count=chunk_count
                    )
                    completed += 1
                    publish(progress, "message", f"Completed {profile.model}")
                except JobCancelled:
                    _cancel_pending_tasks(db, tasks, current_task_id=task_id)
                    return {"completed": completed, "failed": failed, "cancelled": True}
                except Exception as exc:
                    db.finish_analysis_task(task_id, error=str(exc))
                    failed += 1
                    publish(progress, "message", f"{profile.model} failed: {exc}")
                finally:
                    if active_client is not None:
                        active_client(None)
                    if llm is not None:
                        llm.close()
    publish(progress, "stage", "Complete")
    return {"completed": completed, "failed": failed}


class AnalysisController:
    """Run one persistent, multi-model analysis batch at a time."""

    def __init__(
        self,
        config_path: str | Path,
        runner: Callable[..., dict[str, int | bool]] | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self._runner = runner or run_analysis_batch
        self._lock = threading.Lock()
        self._snapshot = AnalysisBatchSnapshot()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._active_llm: LLMClient | None = None

    def start(
        self,
        paper_id: int,
        scope: str,
        profiles: tuple[str, ...],
        analysis_strategy: str = "direct",
        context_window_tokens: int | None = None,
    ) -> str | None:
        with self._lock:
            if self._snapshot.running:
                return None
            if scope not in {"abstract", "fulltext"}:
                raise ValueError("Analysis scope must be abstract or fulltext")
            if analysis_strategy not in {"direct", "chunked", "auto"}:
                raise ValueError("Analysis strategy must be direct, chunked, or auto")
            if scope == "abstract" and analysis_strategy == "chunked":
                raise ValueError("Chunked analysis is only available for full text")
            if context_window_tokens is not None and not 512 <= context_window_tokens <= 1_000_000:
                raise ValueError("Context window must be between 512 and 1000000 tokens")
            config = load_config(self.config_path)
            available = {item.profile: item for item in config.available_llms}
            requested = tuple(dict.fromkeys(profiles))
            unknown = [name for name in requested if name not in available]
            if unknown:
                raise ValueError(f"Unknown configured model profile(s): {', '.join(unknown)}")
            if not requested:
                raise ValueError("Select at least one configured model")
            if len(requested) > 12:
                raise ValueError("Select at most 12 models per batch")
            batch_id = secrets.token_hex(10)
            with Database(config.paths.database) as db:
                if db.get_paper(paper_id) is None:
                    raise ValueError("Paper not found")
                db.create_analysis_tasks(
                    batch_id=batch_id,
                    paper_id=paper_id,
                    scope=scope,
                    profiles=[
                        (name, available[name].provider, available[name].model)
                        for name in requested
                    ],
                    analysis_strategy=analysis_strategy,
                    context_window_tokens=context_window_tokens,
                )
            self._cancel_event = threading.Event()
            self._active_llm = None
            self._snapshot = AnalysisBatchSnapshot(
                running=True,
                batch_id=batch_id,
                paper_id=paper_id,
                scope=scope,
                profiles=requested,
                analysis_strategy=analysis_strategy,
                context_window_tokens=context_window_tokens,
                stage="Queued",
                message="Preparing analysis",
            )
            self._thread = threading.Thread(
                target=self._work,
                args=(
                    batch_id,
                    paper_id,
                    scope,
                    requested,
                    analysis_strategy,
                    context_window_tokens,
                ),
                name="paperwatch-model-batch",
                daemon=True,
            )
            self._thread.start()
            return batch_id

    def _progress(self, kind: str, value: str) -> None:
        with self._lock:
            if kind == "stage":
                self._snapshot.stage = value[:200]
            elif kind == "message":
                self._snapshot.message = value[:1000]
            elif kind == "reset_output":
                self._snapshot.live_output = ""
            elif kind == "output":
                self._snapshot.live_output = (self._snapshot.live_output + value)[-100_000:]

    def _set_active_llm(self, llm: LLMClient | None) -> None:
        with self._lock:
            self._active_llm = llm

    def cancel(self) -> bool:
        with self._lock:
            if not self._snapshot.running:
                return False
            self._snapshot.cancel_requested = True
            self._snapshot.message = "Stopping after the current response boundary"
            self._cancel_event.set()
            llm = self._active_llm
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        return True

    def _work(
        self,
        batch_id: str,
        paper_id: int,
        scope: str,
        profiles: tuple[str, ...],
        analysis_strategy: str,
        context_window_tokens: int | None,
    ) -> None:
        try:
            parameter_count = len(inspect.signature(self._runner).parameters)
            if parameter_count <= 5:
                result = self._runner(self.config_path, batch_id, paper_id, scope, profiles)
            elif parameter_count <= 7:
                result = self._runner(
                    self.config_path,
                    batch_id,
                    paper_id,
                    scope,
                    profiles,
                    analysis_strategy,
                    context_window_tokens,
                )
            else:
                result = self._runner(
                    self.config_path,
                    batch_id,
                    paper_id,
                    scope,
                    profiles,
                    analysis_strategy,
                    context_window_tokens,
                    self._progress,
                    self._cancel_event,
                    self._set_active_llm,
                )
            with self._lock:
                self._snapshot.running = False
                self._snapshot.completed = int(result.get("completed", 0))
                self._snapshot.failed = int(result.get("failed", 0))
                self._snapshot.cancelled = bool(result.get("cancelled", False))
                self._snapshot.cancel_requested = False
                self._snapshot.stage = "Cancelled" if self._snapshot.cancelled else "Complete"
                if self._snapshot.cancelled:
                    self._snapshot.message = "Analysis cancelled"
                elif not self._snapshot.message:
                    self._snapshot.message = "Analysis complete"
        except Exception as exc:
            try:
                config = load_config(self.config_path)
                with Database(config.paths.database) as db:
                    for task in db.analysis_tasks_for_batch(batch_id):
                        if task["status"] in {"queued", "running"}:
                            db.finish_analysis_task(int(task["id"]), error=str(exc))
            except Exception:
                pass
            finally:
                with self._lock:
                    self._snapshot.running = False
                    self._snapshot.error = str(exc)
                    self._snapshot.stage = "Failed"
                    self._snapshot.message = str(exc)
        finally:
            with self._lock:
                self._active_llm = None

    def snapshot(self) -> AnalysisBatchSnapshot:
        with self._lock:
            return AnalysisBatchSnapshot(**asdict(self._snapshot))

    def wait(self, timeout: float = 10) -> None:
        thread = self._thread
        if thread:
            thread.join(timeout)
