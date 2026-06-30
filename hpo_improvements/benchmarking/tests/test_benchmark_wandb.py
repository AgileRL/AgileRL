"""Tests for the Weights & Biases helpers in ``benchmark.py``.

Every W&B / subprocess interaction is mocked: a fake ``wandb`` module, fake
``Run`` objects, and a patched ``subprocess.run`` / ``time``. No network.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import benchmark


# --------------------------------------------------------------------------- #
# _find_wandb_run                                                             #
# --------------------------------------------------------------------------- #
class _FakeApi:
    def __init__(self, *, filter_runs=None, scan_runs=None, filter_raises=False):
        self.default_entity = "ent"
        self._filter_runs = filter_runs if filter_runs is not None else []
        self._scan_runs = scan_runs if scan_runs is not None else []
        self._filter_raises = filter_raises
        self.calls = []

    def runs(self, path, filters=None, order=None):
        self.calls.append((path, filters, order))
        if filters is not None:
            if self._filter_raises:
                raise RuntimeError("boom")
            return self._filter_runs
        return self._scan_runs


def _install_fake_wandb(monkeypatch, api):
    fake = SimpleNamespace(Api=lambda: api)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


class TestFindWandbRun:
    def test_filter_hit(self, monkeypatch):
        run = SimpleNamespace(name="r", state="finished")
        api = _FakeApi(filter_runs=[run])
        _install_fake_wandb(monkeypatch, api)
        assert benchmark._find_wandb_run("proj", "r") is run
        # path uses entity/project.
        assert api.calls[0][0] == "ent/proj"

    def test_filter_empty_then_scan_hit(self, monkeypatch):
        run = SimpleNamespace(name="r", state="finished")
        api = _FakeApi(filter_runs=[], scan_runs=[SimpleNamespace(name="other"), run])
        _install_fake_wandb(monkeypatch, api)
        assert benchmark._find_wandb_run("proj", "r") is run

    def test_filter_raises_falls_back_to_scan(self, monkeypatch):
        run = SimpleNamespace(name="r")
        api = _FakeApi(filter_raises=True, scan_runs=[run])
        _install_fake_wandb(monkeypatch, api)
        assert benchmark._find_wandb_run("proj", "r") is run

    def test_not_found_returns_none(self, monkeypatch):
        api = _FakeApi(filter_runs=[], scan_runs=[SimpleNamespace(name="x")])
        _install_fake_wandb(monkeypatch, api)
        assert benchmark._find_wandb_run("proj", "r") is None


# --------------------------------------------------------------------------- #
# wandb_run_succeeded                                                         #
# --------------------------------------------------------------------------- #
class TestWandbRunSucceeded:
    def test_finished_true(self, monkeypatch):
        monkeypatch.setattr(
            benchmark, "_find_wandb_run", lambda p, r: SimpleNamespace(state="finished")
        )
        assert benchmark.wandb_run_succeeded("p", "r") is True

    @pytest.mark.parametrize("state", ["running", "crashed", "failed", "killed"])
    def test_non_finished_false(self, monkeypatch, state):
        monkeypatch.setattr(
            benchmark, "_find_wandb_run", lambda p, r: SimpleNamespace(state=state)
        )
        assert benchmark.wandb_run_succeeded("p", "r") is False

    def test_none_false(self, monkeypatch):
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: None)
        assert benchmark.wandb_run_succeeded("p", "r") is False

    def test_exception_false(self, monkeypatch):
        def boom(p, r):
            raise RuntimeError("x")

        monkeypatch.setattr(benchmark, "_find_wandb_run", boom)
        assert benchmark.wandb_run_succeeded("p", "r") is False


# --------------------------------------------------------------------------- #
# fetch_wandb_history                                                         #
# --------------------------------------------------------------------------- #
class TestFetchWandbHistory:
    def test_no_run_returns_empty(self, monkeypatch):
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: None)
        df = benchmark.fetch_wandb_history("p", "r")
        assert df.empty

    def test_returns_df_when_complete(self, monkeypatch):
        rows = [{"train/global_step": 1}, {"train/global_step": 2}]
        run = SimpleNamespace(scan_history=lambda: rows, state="finished", summary={})
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: run)
        monkeypatch.setattr(benchmark, "_history_is_complete", lambda run, df: True)
        df = benchmark.fetch_wandb_history("p", "r")
        assert list(df["train/global_step"]) == [1, 2]

    def test_timeout_returns_best(self, monkeypatch):
        rows = [{"train/global_step": 1}]
        run = SimpleNamespace(scan_history=lambda: rows, state="finished", summary={})
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: run)
        monkeypatch.setattr(benchmark, "_history_is_complete", lambda run, df: False)
        # max_wait=0 so the deadline is immediately reached, returning best.
        df = benchmark.fetch_wandb_history("p", "r", max_wait=0.0)
        assert len(df) == 1


# --------------------------------------------------------------------------- #
# upload_run_files / download_run_files                                       #
# --------------------------------------------------------------------------- #
class _UploadRun:
    def __init__(self):
        self.uploaded = []

    def upload_file(self, path, root=None):
        self.uploaded.append((path, root))


class TestUploadRunFiles:
    def test_uploads_existing_only(self, monkeypatch, tmp_path):
        present = tmp_path / "elite.pt"
        present.write_text("x")
        missing = tmp_path / "nope.pt"
        run = _UploadRun()
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: run)
        benchmark.upload_run_files("p", "r", [present, missing], tmp_path)
        assert len(run.uploaded) == 1
        assert run.uploaded[0][0] == str(present)

    def test_no_files_is_noop(self, monkeypatch, tmp_path):
        called = []
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: called.append(1))
        benchmark.upload_run_files("p", "r", [tmp_path / "absent"], tmp_path)
        assert called == []  # never even looks up the run

    def test_missing_run_warns_not_raise(self, monkeypatch, tmp_path):
        f = tmp_path / "f.pt"
        f.write_text("x")
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: None)
        benchmark.upload_run_files("p", "r", [f], tmp_path)  # no raise

    def test_upload_exception_swallowed(self, monkeypatch, tmp_path):
        f = tmp_path / "f.pt"
        f.write_text("x")

        class Boom:
            def upload_file(self, *a, **k):
                raise RuntimeError("nope")

        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: Boom())
        benchmark.upload_run_files("p", "r", [f], tmp_path)  # no raise


class _DownloadRun:
    def __init__(self, downloaded):
        self._downloaded = downloaded

    def file(self, name):
        downloaded = self._downloaded

        class _F:
            def download(self, root=None, replace=False):
                downloaded.append((name, root, replace))

        return _F()


class TestDownloadRunFiles:
    def test_downloads_missing_only(self, monkeypatch, tmp_path):
        (tmp_path / "have.pt").write_text("x")
        downloaded = []
        monkeypatch.setattr(
            benchmark, "_find_wandb_run", lambda p, r: _DownloadRun(downloaded)
        )
        benchmark.download_run_files("p", "r", tmp_path, ["have.pt", "want.pt"])
        assert [d[0] for d in downloaded] == ["want.pt"]

    def test_all_present_no_lookup(self, monkeypatch, tmp_path):
        (tmp_path / "a.pt").write_text("x")
        called = []
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: called.append(1))
        benchmark.download_run_files("p", "r", tmp_path, ["a.pt"])
        assert called == []

    def test_run_none_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(benchmark, "_find_wandb_run", lambda p, r: None)
        benchmark.download_run_files("p", "r", tmp_path, ["x.pt"])  # no raise


# --------------------------------------------------------------------------- #
# _finish_active_wandb_run                                                    #
# --------------------------------------------------------------------------- #
class TestFinishActiveWandbRun:
    def test_noop_when_no_run(self, monkeypatch):
        finished = []
        fake = SimpleNamespace(run=None, finish=lambda: finished.append(1))
        monkeypatch.setitem(sys.modules, "wandb", fake)
        benchmark._finish_active_wandb_run()
        assert finished == []

    def test_finishes_active_run(self, monkeypatch):
        finished = []
        fake = SimpleNamespace(run=object(), finish=lambda: finished.append(1))
        monkeypatch.setitem(sys.modules, "wandb", fake)
        benchmark._finish_active_wandb_run()
        assert finished == [1]

    def test_finish_exception_swallowed(self, monkeypatch):
        def boom():
            raise RuntimeError("x")

        fake = SimpleNamespace(run=object(), finish=boom)
        monkeypatch.setitem(sys.modules, "wandb", fake)
        benchmark._finish_active_wandb_run()  # no raise


# --------------------------------------------------------------------------- #
# _silence_wandb_service_teardown                                             #
# --------------------------------------------------------------------------- #
class TestSilenceTeardown:
    def _patch_service(self, monkeypatch, teardown):
        import wandb.sdk.lib.service as svc_pkg

        class FakeSC:
            pass

        FakeSC.teardown = teardown
        fake_mod = SimpleNamespace(ServiceConnection=FakeSC)
        monkeypatch.setattr(svc_pkg, "service_connection", fake_mod, raising=False)
        return FakeSC

    def test_wraps_and_is_idempotent(self, monkeypatch):
        def orig(self):
            return "orig"

        FakeSC = self._patch_service(monkeypatch, orig)
        benchmark._silence_wandb_service_teardown()
        assert getattr(FakeSC.teardown, "_hpo_wrapped", False) is True
        wrapped = FakeSC.teardown
        benchmark._silence_wandb_service_teardown()  # second call: no re-wrap
        assert FakeSC.teardown is wrapped

    def test_swallows_teardown_error(self, monkeypatch):
        def orig(self):
            raise RuntimeError("broken pipe")

        FakeSC = self._patch_service(monkeypatch, orig)
        benchmark._silence_wandb_service_teardown()
        inst = FakeSC()
        inst._torn_down = False
        assert FakeSC.teardown(inst) is None  # swallowed

    def test_short_circuit_when_torn_down(self, monkeypatch):
        calls = []

        def orig(self):
            calls.append(1)

        FakeSC = self._patch_service(monkeypatch, orig)
        benchmark._silence_wandb_service_teardown()
        inst = FakeSC()
        inst._torn_down = True
        FakeSC.teardown(inst)
        assert calls == []  # orig not called


# --------------------------------------------------------------------------- #
# _sync_offline_wandb_run                                                     #
# --------------------------------------------------------------------------- #
class TestSyncOfflineWandbRun:
    def _make_run_dir(self, tmp_path):
        runs_root = tmp_path / "wandb"
        run_dir = runs_root / "offline-run-20240101_000000-abcd"
        run_dir.mkdir(parents=True)
        return run_dir

    def test_no_run_dir_returns_false(self, tmp_path):
        (tmp_path / "wandb").mkdir()
        assert benchmark._sync_offline_wandb_run(tmp_path, wandb_api_key=None) is False

    def test_success(self, monkeypatch, tmp_path):
        self._make_run_dir(tmp_path)
        captured = {}

        def fake_run(cmd, env=None, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = env
            return SimpleNamespace(returncode=0, stderr="", stdout="ok")

        import subprocess

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok = benchmark._sync_offline_wandb_run(tmp_path, wandb_api_key="KEY", retries=3)
        assert ok is True
        assert "sync" in captured["cmd"]
        # WANDB_MODE popped, key injected.
        assert "WANDB_MODE" not in captured["env"]
        assert captured["env"]["WANDB_API_KEY"] == "KEY"

    def test_retries_then_fails(self, monkeypatch, tmp_path):
        self._make_run_dir(tmp_path)
        attempts = []

        def fake_run(cmd, env=None, **kwargs):
            attempts.append(1)
            return SimpleNamespace(returncode=1, stderr="err", stdout="")

        slept = []
        import subprocess
        import time

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        ok = benchmark._sync_offline_wandb_run(tmp_path, wandb_api_key=None, retries=3)
        assert ok is False
        assert len(attempts) == 3
