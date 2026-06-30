"""Tests for the orchestration glue in ``benchmark.py``.

All heavy seams (real training, EnvPool, W&B, rendering, plotting) are
monkeypatched, so these assert the wiring — manifest stamping, dispatch between
the single-/multi-agent paths, the curve maths in ``fetch_and_plot``, the
aggregator fan-out, ``main``'s argv flow and per-env error isolation, and the
render early-return guards — without doing any real compute.
"""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import benchmark


# --------------------------------------------------------------------------- #
# _make_envpool_env                                                           #
# --------------------------------------------------------------------------- #
class TestMakeEnvpoolEnv:
    def _install_fake_envpool(self, monkeypatch, obs_shape):
        recorded = {}

        def make_gymnasium(env_id, num_envs, seed):
            recorded["args"] = (env_id, num_envs, seed)
            return SimpleNamespace(
                single_observation_space=SimpleNamespace(shape=obs_shape),
                observation_space=SimpleNamespace(shape=obs_shape),
            )

        fake = SimpleNamespace(make_gymnasium=make_gymnasium)
        monkeypatch.setitem(sys.modules, "envpool", fake)
        return recorded

    def test_seed_scaled_by_num_envs(self, monkeypatch):
        recorded = self._install_fake_envpool(monkeypatch, (4,))
        monkeypatch.setattr(
            benchmark.gym.wrappers.vector,
            "FlattenObservation",
            lambda env: ("flat", env),
        )
        benchmark._make_envpool_env("Ant-v4", num_envs=8, seed=3)
        assert recorded["args"] == ("Ant-v4", 8, 24)  # 3 * 8

    def test_flat_obs_is_flattened(self, monkeypatch):
        self._install_fake_envpool(monkeypatch, (17,))
        monkeypatch.setattr(
            benchmark.gym.wrappers.vector,
            "FlattenObservation",
            lambda env: ("flat", env),
        )
        out = benchmark._make_envpool_env("Ant-v4", 2, 1)
        assert out[0] == "flat"

    def test_image_obs_not_flattened(self, monkeypatch):
        self._install_fake_envpool(monkeypatch, (4, 84, 84))
        monkeypatch.setattr(
            benchmark.gym.wrappers.vector,
            "FlattenObservation",
            lambda env: pytest.fail("should not flatten image obs"),
        )
        out = benchmark._make_envpool_env("Pong-v5", 2, 1)
        assert out.single_observation_space.shape == (4, 84, 84)


# --------------------------------------------------------------------------- #
# _build_trainer dispatch                                                     #
# --------------------------------------------------------------------------- #
class TestBuildTrainer:
    def test_multi_agent_uses_local_trainer_from_manifest(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(
            benchmark.LocalTrainer,
            "from_manifest",
            classmethod(
                lambda cls, m, device: (calls.__setitem__("m", (m, device)), "T")[1]
            ),
        )
        out = benchmark._build_trainer({}, "simple_spread_v3", "IPPO", 1, "cpu")
        assert out == "T"
        assert calls["m"][1] == "cpu"

    def test_single_agent_builds_envpool(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            benchmark,
            "_make_envpool_env",
            lambda e, n, s: (seen.__setitem__("ep", (e, n, s)), "ENV")[1],
        )
        monkeypatch.setattr(
            benchmark._EnvPoolLocalTrainer,
            "from_manifest_with_env",
            classmethod(
                lambda cls, m, env, device: (
                    seen.__setitem__("ft", (env, device)),
                    "T",
                )[1]
            ),
        )
        manifest = {"environment": {"num_envs": 8}}
        out = benchmark._build_trainer(manifest, "Ant-v4", "PPO", 3, "cpu")
        assert out == "T"
        assert seen["ep"] == ("Ant-v4", 8, 3)
        assert seen["ft"][1] == "cpu"


# --------------------------------------------------------------------------- #
# _EnvPoolLocalTrainer                                                        #
# --------------------------------------------------------------------------- #
class TestEnvPoolLocalTrainer:
    def test_make_env_returns_prebuilt(self):
        inst = object.__new__(benchmark._EnvPoolLocalTrainer)
        inst._prebuilt_env = "PREBUILT"
        assert inst._make_env() == "PREBUILT"

    def test_from_manifest_with_env_filters_none_and_resets(self, monkeypatch):
        captured = {}
        validated = SimpleNamespace(
            environment={"name": "Ant-v4", "num_envs": 8, "drop": None},
            algorithm="A",
            training="T",
            mutation="M",
            tournament_selection="Tour",
            replay_buffer="RB",
        )
        monkeypatch.setattr(
            benchmark.TrainingManifest,
            "get_validated",
            classmethod(lambda cls, m, mode: validated),
        )
        monkeypatch.setattr(benchmark, "GymEnvSpec", lambda **kw: ("envspec", kw))

        def fake_init(self, **kw):
            captured.update(kw)
            captured["prebuilt_at_init"] = type(self)._prebuilt_env

        monkeypatch.setattr(benchmark._EnvPoolLocalTrainer, "__init__", fake_init)
        benchmark._EnvPoolLocalTrainer.from_manifest_with_env({}, "VEC", device="cpu")
        # None env fields dropped before building the GymEnvSpec.
        assert captured["environment"] == ("envspec", {"name": "Ant-v4", "num_envs": 8})
        # the prebuilt env was visible during construction...
        assert captured["prebuilt_at_init"] == "VEC"
        # ...and reset to None afterwards.
        assert benchmark._EnvPoolLocalTrainer._prebuilt_env is None


# --------------------------------------------------------------------------- #
# run_training                                                                #
# --------------------------------------------------------------------------- #
class _Agent:
    def __init__(self, fitness):
        self.fitness = fitness
        self.saved_to = None

    def save_checkpoint(self, path):
        self.saved_to = path


class _FakeTrainer:
    def __init__(self, population):
        self.population = list(population)
        self._returned = list(population)
        self.env = SimpleNamespace(close=lambda: None, reset=lambda seed=None: None)
        self.memory = None
        self.n_step_memory = None
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return self._returned, None


@pytest.fixture
def run_training_seams(monkeypatch):
    """Patch every heavy seam in run_training; return the recorder dict."""
    rec = {"render": [], "rsnorm": 0, "move_buffer": 0, "synced": [], "uploaded": []}

    agents = [_Agent([1.0]), _Agent([2.0])]
    trainer = _FakeTrainer(agents)
    rec["trainer"] = trainer
    rec["agents"] = agents

    monkeypatch.setattr(benchmark, "_silence_wandb_service_teardown", lambda: None)
    monkeypatch.setattr(benchmark, "seed_everything", lambda s: None)
    monkeypatch.setattr(benchmark, "_build_trainer", lambda *a, **k: trainer)
    monkeypatch.setattr(
        benchmark,
        "_move_replay_buffer_to_cpu",
        lambda t: rec.__setitem__("move_buffer", rec["move_buffer"] + 1),
    )

    def fake_rsnorm(agent):
        rec["rsnorm"] += 1
        return agent

    monkeypatch.setattr(benchmark, "RSNorm", fake_rsnorm)
    monkeypatch.setattr(benchmark, "tee_to_file", lambda p: contextlib.nullcontext())
    monkeypatch.setattr(benchmark, "_finish_active_wandb_run", lambda: None)
    monkeypatch.setattr(
        benchmark,
        "render_best_agent",
        lambda *a, **k: rec["render"].append(("single", a)),
    )
    monkeypatch.setattr(
        benchmark,
        "render_best_multi_agent",
        lambda *a, **k: rec["render"].append(("multi", a)),
    )
    monkeypatch.setattr(
        benchmark,
        "_sync_offline_wandb_run",
        lambda *a, **k: rec["synced"].append(a) or True,
    )
    monkeypatch.setattr(
        benchmark,
        "upload_run_files",
        lambda *a, **k: rec["uploaded"].append(a),
    )
    return rec


class TestRunTraining:
    def _manifest(self):
        return {
            "environment": {"name": "placeholder", "num_envs": 8},
            "training": {"pop_size": 2},
            "mutation": {"rl_hp_selection": {"lr": {}}},
        }

    def test_ppo_path(self, run_training_seams, tmp_path):
        rec = run_training_seams
        base = self._manifest()
        out = benchmark.run_training(
            base_manifest=base,
            env_name="Ant-v4",
            algo="PPO",
            seed=5,
            device="cpu",
            project="p",
            run_name="run1",
            out_dir=tmp_path,
            wandb_api_key=None,
        )
        assert out == {
            "run_name": "run1",
            "pop_size": 2,
            "hp_names": ["lr"],
            "elite_path": str(tmp_path / "elite_PPO.pt"),
        }
        # base_manifest must be untouched (deep-copied internally).
        assert base["environment"]["name"] == "placeholder"
        # PPO applies RSNorm to each population member.
        assert rec["rsnorm"] == 2
        # best agent (fitness 2.0) checkpointed to elite path.
        assert rec["agents"][1].saved_to == str(tmp_path / "elite_PPO.pt")
        # single-agent render dispatched.
        assert rec["render"][0][0] == "single"
        # offline W&B env vars set.
        import os

        assert os.environ["WANDB_MODE"] == "offline"
        assert os.environ["WANDB_DIR"] == str(tmp_path)

    def test_move_replay_buffer_flag(self, run_training_seams, tmp_path):
        rec = run_training_seams
        benchmark.run_training(
            base_manifest=self._manifest(),
            env_name="Ant-v4",
            algo="DQN",
            seed=1,
            device="cpu",
            project="p",
            run_name="r",
            out_dir=tmp_path,
            wandb_api_key=None,
            move_replay_buffer_to_cpu=True,
        )
        assert rec["move_buffer"] == 1
        # DQN is not PPO -> no RSNorm.
        assert rec["rsnorm"] == 0

    def test_ippo_path_dispatches_multi_render(self, run_training_seams, tmp_path):
        rec = run_training_seams
        base = {
            "environment": {"name": "x", "num_envs": 1},
            "training": {"pop_size": 2},
        }
        benchmark.run_training(
            base_manifest=base,
            env_name="simple_spread_v3",
            algo="IPPO",
            seed=2,
            device="cpu",
            project="p",
            run_name="r",
            out_dir=tmp_path,
            wandb_api_key=None,
        )
        assert rec["render"][0][0] == "multi"
        # IPPO is not PPO -> no RSNorm.
        assert rec["rsnorm"] == 0

    def test_no_render_when_disabled(self, run_training_seams, tmp_path):
        rec = run_training_seams
        benchmark.run_training(
            base_manifest=self._manifest(),
            env_name="Ant-v4",
            algo="PPO",
            seed=1,
            device="cpu",
            project="p",
            run_name="r",
            out_dir=tmp_path,
            wandb_api_key=None,
            render=False,
        )
        assert rec["render"] == []


# --------------------------------------------------------------------------- #
# run_environment                                                             #
# --------------------------------------------------------------------------- #
class TestRunEnvironment:
    def test_constructs_run_name_and_env_dir(self, monkeypatch, tmp_path):
        seen = {}

        def fake_training(**kw):
            seen["train"] = kw
            return {"pop_size": 4, "hp_names": ["lr"]}

        def fake_fetch(**kw):
            seen["fetch"] = kw
            return ("X",)

        monkeypatch.setattr(benchmark, "run_training", fake_training)
        monkeypatch.setattr(benchmark, "fetch_and_plot", fake_fetch)
        out = benchmark.run_environment(
            base_manifest={},
            env_name="Ant-v4",
            algo="PPO",
            seed=7,
            device="cpu",
            project="p",
            base_name="bench",
            stamp="STAMP",
            bench_dir=tmp_path,
            wandb_api_key=None,
        )
        assert out == ("X",)
        assert seen["train"]["run_name"] == "bench-PPO-Ant-v4-s7-STAMP"
        assert seen["train"]["out_dir"] == tmp_path / "Ant-v4" / "s7"
        # pop_size / hp_names threaded from training into fetch_and_plot.
        assert seen["fetch"]["pop_size"] == 4
        assert seen["fetch"]["hp_names"] == ["lr"]


# --------------------------------------------------------------------------- #
# fetch_and_plot                                                              #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fetch_and_plot_seams(monkeypatch):
    """Patch the W&B fetch/download and all plotting in fetch_and_plot."""
    monkeypatch.setattr(benchmark, "download_run_files", lambda *a, **k: None)
    monkeypatch.setattr(
        benchmark,
        "normalization_scores",
        lambda algo, env: SimpleNamespace(normalize=lambda f: f / 10.0),
    )
    for name in [
        "plot_fitness",
        "plot_mutation_schedule",
        "plot_dormant_fraction",
        "plot_diversity",
        "plot_population_hp_trajectory",
        "plot_hp_fitness",
    ]:
        monkeypatch.setattr(benchmark.plotting, name, lambda *a, **k: None)


class TestFetchAndPlot:
    def test_empty_history_returns_none(
        self, monkeypatch, fetch_and_plot_seams, tmp_path
    ):
        monkeypatch.setattr(
            benchmark, "fetch_wandb_history", lambda p, r: pd.DataFrame()
        )
        out = benchmark.fetch_and_plot(
            project="p",
            run_name="r",
            algo="PPO",
            env_name="Ant-v4",
            pop_size=2,
            env_dir=tmp_path,
        )
        assert out is None

    def test_returns_curve_math(self, monkeypatch, fetch_and_plot_seams, tmp_path):
        hist = pd.DataFrame(
            {"train/global_step": [4, 8], "eval/best_fitness": [10.0, 20.0]}
        )
        monkeypatch.setattr(benchmark, "fetch_wandb_history", lambda p, r: hist)
        out = benchmark.fetch_and_plot(
            project="p",
            run_name="r",
            algo="PPO",
            env_name="Ant-v4",
            pop_size=2,
            env_dir=tmp_path,
        )
        x, best, y, diversity = out
        np.testing.assert_allclose(x, [2.0, 4.0])  # step / pop_size
        np.testing.assert_allclose(best, [10.0, 20.0])
        np.testing.assert_allclose(y, [1.0, 2.0])  # normalize = f/10
        assert diversity is None
        # the history CSV is written out.
        assert (tmp_path / "wandb_history.csv").is_file()

    def test_diversity_extracted(self, monkeypatch, fetch_and_plot_seams, tmp_path):
        col = benchmark.plotting.DIVERSITY_SPECS[0][0]
        hist = pd.DataFrame(
            {"train/global_step": [4], "eval/best_fitness": [10.0], col: [0.3]}
        )
        monkeypatch.setattr(benchmark, "fetch_wandb_history", lambda p, r: hist)
        out = benchmark.fetch_and_plot(
            project="p",
            run_name="r",
            algo="PPO",
            env_name="Ant-v4",
            pop_size=1,
            env_dir=tmp_path,
        )
        _, _, _, diversity = out
        assert col in diversity


# --------------------------------------------------------------------------- #
# plot_seed_aggregates / plot_diversity_aggregates                            #
# --------------------------------------------------------------------------- #
class TestSeedAggregates:
    def test_calls_aggregate_and_profile(self, monkeypatch, tmp_path):
        calls = {"agg": 0, "prof": 0}
        monkeypatch.setattr(
            benchmark.plotting,
            "plot_fitness_over_seeds",
            lambda *a, **k: (np.array([0.0, 1.0]), np.ones((2, 2))),
        )
        monkeypatch.setattr(
            benchmark.plotting,
            "plot_aggregate",
            lambda *a, **k: calls.__setitem__("agg", calls["agg"] + 1),
        )
        monkeypatch.setattr(
            benchmark.plotting,
            "plot_performance_profile",
            lambda *a, **k: calls.__setitem__("prof", calls["prof"] + 1),
        )
        curves = {"Ant-v4": [(np.array([0.0, 1.0]),) * 3]}
        benchmark.plot_seed_aggregates(curves, tmp_path, "L", "S")
        assert calls == {"agg": 1, "prof": 1}

    def test_skips_aggregate_when_nothing_plottable(self, monkeypatch, tmp_path):
        calls = {"agg": 0}
        monkeypatch.setattr(
            benchmark.plotting, "plot_fitness_over_seeds", lambda *a, **k: None
        )
        monkeypatch.setattr(
            benchmark.plotting,
            "plot_aggregate",
            lambda *a, **k: calls.__setitem__("agg", 1),
        )
        monkeypatch.setattr(
            benchmark.plotting, "plot_performance_profile", lambda *a, **k: None
        )
        benchmark.plot_seed_aggregates(
            {"Ant-v4": [(np.array([0.0]),) * 3]}, tmp_path, "L", "S"
        )
        assert calls["agg"] == 0


class TestDiversityAggregates:
    def test_regroups_per_metric(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(
            benchmark.plotting,
            "plot_diversity_over_seeds",
            lambda *a, **k: {
                "eval/hp_diversity": (np.array([0.0, 1.0]), np.ones((2, 2)))
            },
        )
        monkeypatch.setattr(
            benchmark.plotting,
            "plot_diversity_aggregate",
            lambda per_metric, *a, **k: seen.setdefault("pm", per_metric),
        )
        curves = {
            "Ant-v4": [
                (np.array([0.0, 1.0]), {"eval/hp_diversity": np.array([0.1, 0.2])})
            ]
        }
        benchmark.plot_diversity_aggregates(curves, tmp_path, "L", "S")
        assert "eval/hp_diversity" in seen["pm"]
        assert "Ant-v4" in seen["pm"]["eval/hp_diversity"]


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
class TestMain:
    def _write_manifest(self, tmp_path, extra_training=None):
        import yaml

        manifest = {"algorithm": {"name": "PPO"}, "training": {"evo_steps": 100}}
        if extra_training:
            manifest["training"].update(extra_training)
        p = tmp_path / "manifest.yaml"
        p.write_text(yaml.safe_dump(manifest))
        return p

    def _argv(self, config, **over):
        args = {
            "--project": "p",
            "--name": "n",
            "--config": str(config),
            "--device": "cpu",
            "--envs": "Ant-v4",
            "--seeds": "1",
        }
        args.update(over)
        out = []
        for k, v in args.items():
            out += [k, v]
        return out

    def test_writes_config_and_aggregates(self, monkeypatch, tmp_path):
        monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(
            benchmark,
            "run_environment",
            lambda **k: (
                np.array([0.0, 1.0]),
                np.array([1.0, 2.0]),
                np.array([0.0, 1.0]),
                None,
            ),
        )
        called = {"seed": 0}
        monkeypatch.setattr(
            benchmark,
            "plot_seed_aggregates",
            lambda *a, **k: called.__setitem__("seed", 1),
        )
        monkeypatch.setattr(
            benchmark, "plot_diversity_aggregates", lambda *a, **k: None
        )
        cfg = self._write_manifest(tmp_path)
        benchmark.main(self._argv(cfg))
        assert called == {"seed": 1}
        # the persisted config.yaml carries a benchmark metadata block.
        import yaml

        bench_dirs = list((tmp_path / "results").iterdir())
        assert len(bench_dirs) == 1
        saved = yaml.safe_load((bench_dirs[0] / "config.yaml").read_text())
        assert saved["benchmark"]["seeds"] == [1]
        assert saved["benchmark"]["environments"] == ["Ant-v4"]

    def test_max_steps_clamps_evo_steps(self, monkeypatch, tmp_path):
        seen = {}
        monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(
            benchmark,
            "run_environment",
            lambda **k: (
                seen.setdefault("manifest", k["base_manifest"]) and None or None
            ),
        )
        monkeypatch.setattr(benchmark, "plot_seed_aggregates", lambda *a, **k: None)
        cfg = self._write_manifest(tmp_path, extra_training={"evo_steps": 100})
        benchmark.main(self._argv(cfg, **{"--max-steps": "50"}))
        training = seen["manifest"]["training"]
        assert training["max_steps"] == 50
        assert training["evo_steps"] == 50  # clamped down to max_steps

    def test_bad_config_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path / "results")
        with pytest.raises(FileNotFoundError):
            benchmark.main(self._argv("/definitely/not/a/real/config.yaml"))

    def test_per_env_failure_is_isolated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path / "results")

        def boom(**k):
            raise RuntimeError("seed blew up")

        monkeypatch.setattr(benchmark, "run_environment", boom)
        agg = {"called": 0}
        monkeypatch.setattr(
            benchmark,
            "plot_seed_aggregates",
            lambda *a, **k: agg.__setitem__("called", 1),
        )
        cfg = self._write_manifest(tmp_path)
        # the exception inside the loop must not propagate.
        benchmark.main(self._argv(cfg))
        # all runs failed -> no curves -> aggregate skipped (but no crash).
        assert agg["called"] == 0


# --------------------------------------------------------------------------- #
# render guards                                                               #
# --------------------------------------------------------------------------- #
class TestRenderGuards:
    def test_pusher_skipped_before_loader(self, monkeypatch, tmp_path):
        loaded = []
        monkeypatch.setitem(
            benchmark._RENDER_LOADERS, "PPO", lambda *a, **k: loaded.append(1)
        )
        benchmark.render_best_agent(
            tmp_path / "e.pt", "Pusher-v4", 0, "cpu", tmp_path / "r.mp4", "PPO"
        )
        assert loaded == []
        assert not (tmp_path / "r.mp4").exists()

    def test_unsupported_algo_skipped(self, monkeypatch, tmp_path):
        # Make the MuJoCo headless guard pass so we reach the loader lookup.
        monkeypatch.setenv("MUJOCO_GL", "osmesa")
        benchmark.render_best_agent(
            tmp_path / "e.pt", "Ant-v4", 0, "cpu", tmp_path / "r.mp4", "SAC"
        )
        assert not (tmp_path / "r.mp4").exists()

    def test_headless_mujoco_skip_on_linux(self, monkeypatch, tmp_path):
        import platform

        loaded = []
        monkeypatch.setitem(
            benchmark._RENDER_LOADERS, "PPO", lambda *a, **k: loaded.append(1)
        )
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        benchmark.render_best_agent(
            tmp_path / "e.pt", "Ant-v4", 0, "cpu", tmp_path / "r.mp4", "PPO"
        )
        assert loaded == []
