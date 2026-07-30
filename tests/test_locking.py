from __future__ import annotations

import pytest

from paperwatch.locking import RunAlreadyInProgress, run_lock


def test_run_lock_prevents_overlap_and_releases(tmp_path):
    database = tmp_path / "paperwatch.db"
    with run_lock(database):
        lock_path = tmp_path / "paperwatch.db.run.lock"
        assert lock_path.read_text().strip().isdigit()
        with pytest.raises(RunAlreadyInProgress, match="already using"):
            with run_lock(database):
                pass
    with run_lock(database):
        pass


def test_analysis_lock_is_independent_and_prevents_overlap(tmp_path):
    from paperwatch.locking import AnalysisAlreadyInProgress, analysis_lock

    database = tmp_path / "paperwatch.db"
    with analysis_lock(database):
        with run_lock(database):
            pass
        with pytest.raises(AnalysisAlreadyInProgress, match="analysis batch"):
            with analysis_lock(database):
                pass
    with analysis_lock(database):
        pass
