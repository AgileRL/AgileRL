"""Benchmark PPO across environment suites under different HPO regimes.

Trains a fresh PPO agent (no HPO / AgileRL HPO / HPO improvements -- the regime
is whatever the YAML's ``mutation`` block encodes) on each selected environment,
logging to Weights & Biases, then pulls the run history back from W&B to produce
per-environment and aggregate plots plus a rendered video of the best agent.

Run directly from this folder, e.g.::

    python benchmark.py --config configs/ppo_relu_no_hpo.yaml \
        --project my_project --name run0 --seed 42 --device cpu --envs all

Any omitted argument is prompted for interactively. ``--config`` is interpreted
relative to this script's directory.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Must be set before PyTorch's C++ extension is loaded (i.e. before `import torch`).
# On MPS, some ops (e.g. aten::linalg_qr) are not yet implemented; this flag makes
# them silently fall back to CPU rather than raising a NotImplementedError.
# It is a no-op on CUDA and CPU devices.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Allow running as a plain script: make sibling modules importable.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotting  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from registry import (  # noqa: E402
    allowed_envs,
    normalization_scores,
    resolve_env_selection,
)
from reproducibility import seed_everything  # noqa: E402

from agilerl.algorithms import PPO  # noqa: E402
from agilerl.models.env import GymEnvSpec  # noqa: E402
from agilerl.models.manifest import TrainingManifest  # noqa: E402
from agilerl.training.trainer import LocalTrainer  # noqa: E402

logger = logging.getLogger("hpo_benchmark")

RESULTS_DIR = SCRIPT_DIR / "results"
MAX_RENDER_STEPS = 3000


# --------------------------------------------------------------------------- #
# EnvPool helpers
# --------------------------------------------------------------------------- #
def _make_envpool_env(env_id: str, num_envs: int, seed: int) -> Any:
    """Return an EnvPool-backed vectorized Gymnasium environment.

    EnvPool seeds at creation time; callers must **not** call
    ``env.reset(seed=...)`` afterwards.

    :param env_id: Gymnasium environment id (e.g. ``"Ant-v5"``).
    :type env_id: str
    :param num_envs: Number of parallel environments.
    :type num_envs: int
    :param seed: Seed applied at creation time for reproducibility.
    :type seed: int
    :return: Vectorized EnvPool environment with Gymnasium API.
    :rtype: Any
    """
    try:
        import envpool
    except ModuleNotFoundError as exc:
        msg = (
            "EnvPool is not installed. "
            "Install it before running this benchmark, e.g. `uv add envpool`."
        )
        raise ImportError(msg) from exc

    make_fn = (
        envpool.make_gymnasium if hasattr(envpool, "make_gymnasium") else envpool.make
    )
    env = make_fn(env_id, num_envs=num_envs, seed=seed)
    # MuJoCo v5 observations are already flat; FlattenObservation is a no-op
    # but keeps the interface consistent with any future non-flat envs.
    return gym.wrappers.vector.FlattenObservation(env)


class _EnvPoolLocalTrainer(LocalTrainer):
    """``LocalTrainer`` that uses a pre-built EnvPool vector environment.

    Based on the same pattern used in
    ``hpo_improvements/panda_robotiq_push_cube/train.py``.
    """

    _prebuilt_env: Any = None

    def _make_env(self) -> Any:
        if self._prebuilt_env is not None:
            return self._prebuilt_env
        return super()._make_env()

    @classmethod
    def from_manifest_with_env(
        cls,
        manifest: dict[str, Any] | TrainingManifest,
        vector_env: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> _EnvPoolLocalTrainer:
        """Build a trainer from a manifest dict and a pre-built EnvPool env.

        :param manifest: YAML manifest dict or validated ``TrainingManifest``.
        :param vector_env: Pre-built (and already seeded) EnvPool environment.
        :param device: Torch device string.
        """
        validated = (
            TrainingManifest.get_validated(manifest, mode="python")
            if not isinstance(manifest, TrainingManifest)
            else manifest
        )
        env_data = {
            k: v for k, v in dict(validated.environment).items() if v is not None
        }
        env_spec = GymEnvSpec(**env_data)
        cls._prebuilt_env = vector_env
        try:
            return cls(
                algorithm=validated.algorithm,
                environment=env_spec,
                training=validated.training,
                mutation=validated.mutation,
                tournament=validated.tournament_selection,
                replay_buffer=validated.replay_buffer,
                device=device,
            )
        finally:
            cls._prebuilt_env = None


# --------------------------------------------------------------------------- #
# Input stage
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Missing values are prompted for interactively."""
    p = argparse.ArgumentParser(description="AgileRL PPO HPO benchmark")
    p.add_argument("--project", help="Weights & Biases project name")
    p.add_argument("--name", help="Benchmark base name (used in run/folder names)")
    p.add_argument("--config", help="Path to YAML config, relative to this script")
    p.add_argument("--seed", type=int, help="Global seed for reproducibility")
    p.add_argument("--device", help="Torch device, e.g. 'cpu', 'cuda', or 'mps'")
    p.add_argument(
        "--envs",
        help="Comma/space separated env names, or 'all'",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps (useful for quick checks)",
    )
    p.add_argument(
        "--wandb-api-key",
        default=None,
        help="W&B API key (else WANDB_API_KEY env var is used)",
    )
    return p.parse_args(argv)


def _prompt(value: str | None, message: str) -> str:
    """Return ``value`` if set, else prompt the user."""
    if value is not None and str(value) != "":
        return str(value)
    return input(message).strip()


def resolve_device(device: str) -> str:
    """Validate *device* and return a usable torch device string.

    Checks that the requested backend is actually available at runtime and
    warns (rather than crashes) when it is not, falling back to CPU.

    * ``"cuda"`` — requires a CUDA-capable GPU (``torch.cuda.is_available()``).
    * ``"mps"``  — requires Apple Silicon / AMD on macOS with PyTorch ≥ 2.0
                   (``torch.backends.mps.is_available()``).
    * ``"cpu"``  — always available.

    :param device: Requested device string (case-insensitive).
    :type device: str
    :return: A valid torch device string (``"cpu"``, ``"cuda"``, or ``"mps"``).
    :rtype: str
    """
    device = device.strip().lower()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")
            return "cpu"
        return device  # preserve e.g. "cuda:1"
    if device == "mps":
        if (
            not getattr(torch.backends, "mps", None)
            or not torch.backends.mps.is_available()
        ):
            logger.warning(
                "MPS requested but not available on this system — falling back to CPU. "
                "MPS requires macOS 12.3+, Apple Silicon or AMD GPU, and PyTorch ≥ 2.0."
            )
            return "cpu"
        return "mps"
    if device != "cpu":
        logger.warning("Unknown device '%s' — falling back to CPU.", device)
        return "cpu"
    return "cpu"


def load_manifest_dict(config_path: Path) -> dict[str, Any]:
    """Load a YAML manifest as a plain dict.

    :param config_path: Path to the YAML config.
    :type config_path: Path
    :return: Parsed manifest dictionary.
    :rtype: dict[str, Any]
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Logging capture
# --------------------------------------------------------------------------- #
class _Tee:
    """Write to multiple streams at once (console + file)."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def tee_to_file(path: Path):
    """Mirror stdout/stderr and root logging to ``path`` for the duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    orig_out, orig_err = sys.stdout, sys.stderr
    file = open(path, "w")
    file_handler = logging.StreamHandler(file)
    file_handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(file_handler)
    try:
        sys.stdout = _Tee(orig_out, file)
        sys.stderr = _Tee(orig_err, file)
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        root.removeHandler(file_handler)
        file.close()


# --------------------------------------------------------------------------- #
# W&B history fetch
# --------------------------------------------------------------------------- #
def fetch_wandb_history(project: str, run_name: str) -> pd.DataFrame:
    """Fetch the full logged history of a finished W&B run as a DataFrame.

    :param project: W&B project name.
    :type project: str
    :param run_name: Display name of the run to fetch.
    :type run_name: str
    :return: History dataframe (includes ``_runtime`` relative time), or empty.
    :rtype: pandas.DataFrame
    """
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    path = f"{entity}/{project}" if entity else project

    run = None
    try:
        runs = api.runs(path, filters={"displayName": run_name})
        run = runs[0] if len(runs) else None
    except Exception as exc:
        logger.warning("W&B filter query failed (%s); scanning recent runs.", exc)

    if run is None:
        try:
            run = next(
                (r for r in api.runs(path, order="-created_at") if r.name == run_name),
                None,
            )
        except Exception as exc:
            logger.warning("W&B run scan failed: %s", exc)

    if run is None:
        logger.warning(
            "Could not locate W&B run '%s' in project '%s'.", run_name, project
        )
        return pd.DataFrame()

    rows = list(run.scan_history())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_best_agent(
    elite_path: Path, env_name: str, seed: int, device: str, out_path: Path
) -> None:
    """Render one greedy episode of the saved elite agent to an MP4.

    :param elite_path: Path to the saved elite checkpoint.
    :type elite_path: Path
    :param env_name: Gymnasium environment id.
    :type env_name: str
    :param seed: Seed for the render env's first reset.
    :type seed: int
    :param device: Torch device.
    :type device: str
    :param out_path: Destination .mp4 path.
    :type out_path: Path
    """
    # Headless MuJoCo rendering: export MUJOCO_GL=egl (GPU) or osmesa (CPU)
    # before running. We do not set it here because MuJoCo selects its GL
    # backend at import time and forcing EGL would break env creation on hosts
    # without an EGL library.
    #
    # When MUJOCO_GL is unset and there is no DISPLAY, MuJoCo falls back to GLFW
    # and aborts with a *native* error (libc++abi / SIGABRT) that Python cannot
    # catch -- which would kill the whole benchmark. So we skip rendering unless
    # a usable GL backend is available.
    import os
    import platform

    # On macOS, MuJoCo renders via glfw (Quartz) and needs neither DISPLAY nor
    # MUJOCO_GL.  On Linux without a display, glfw aborts at the C level (not
    # catchable), so we require either DISPLAY or a headless MUJOCO_GL backend.
    on_macos = platform.system() == "Darwin"
    if (
        not on_macos
        and not os.environ.get("MUJOCO_GL")
        and not os.environ.get("DISPLAY")
    ):
        logger.warning(
            "Skipping render for %s: no DISPLAY and MUJOCO_GL is unset. "
            "Set MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU) to enable video on Linux.",
            env_name,
        )
        return

    try:
        import imageio

        agent = PPO.load(str(elite_path), device=device)
        agent.set_training_mode(False)

        render_env = gym.wrappers.FlattenObservation(
            gym.make(env_name, render_mode="rgb_array")
        )
        obs, _ = render_env.reset(seed=seed)
        frames: list[np.ndarray] = []
        done = False
        steps = 0
        while not done and steps < MAX_RENDER_STEPS:
            frames.append(render_env.render())
            action = agent.get_action(np.expand_dims(obs, 0))[0]
            obs, _, term, trunc, _ = render_env.step(np.asarray(action[0]))
            done = bool(term) or bool(trunc)
            steps += 1
        render_env.close()

        if frames:
            imageio.mimsave(str(out_path), frames, fps=30)
            logger.info("Saved render: %s (%d frames)", out_path, len(frames))
    except Exception as exc:
        logger.exception("Rendering failed for %s: %s", env_name, exc)


# --------------------------------------------------------------------------- #
# Per-environment run
# --------------------------------------------------------------------------- #
def run_environment(
    *,
    base_manifest: dict[str, Any],
    env_name: str,
    algo: str,
    seed: int,
    device: str,
    project: str,
    base_name: str,
    stamp: str,
    bench_dir: Path,
    wandb_api_key: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Train, log, fetch, render and plot for a single environment.

    :return: ``(x, normalized_fitness)`` for the aggregate plot, or None on failure.
    :rtype: tuple[numpy.ndarray, numpy.ndarray] | None
    """
    env_dir = bench_dir / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{base_name}-{algo}-{env_name}-{stamp}"

    manifest = copy.deepcopy(base_manifest)
    manifest["environment"]["name"] = env_name
    pop_size = int(manifest.get("training", {}).get("pop_size", 1) or 1)

    # Fresh, reproducible agent per environment.
    # EnvPool is seeded at creation; do not call env.reset(seed=...) afterwards.
    seed_everything(seed)
    num_envs = int(manifest.get("environment", {}).get("num_envs", 1) or 1)
    envpool_env = _make_envpool_env(env_name, num_envs, seed)
    trainer = _EnvPoolLocalTrainer.from_manifest_with_env(
        manifest, envpool_env, device=device
    )

    elite_path = env_dir / f"elite_{algo}.pt"
    log_path = env_dir / "train.log"

    logger.info("=== Training %s on %s (run '%s') ===", algo, env_name, run_name)
    with tee_to_file(log_path):
        population, _ = trainer.train(
            wb=True,
            wandb_api_key=wandb_api_key,
            wandb_kwargs={"project": project, "addl_args": {"name": run_name}},
            verbose=True,
        )

    # Save the best agent of the final population (works with or without HPO).
    def _last_fitness(agent: Any) -> float:
        fit = getattr(agent, "fitness", None)
        return fit[-1] if fit else float("-inf")

    best_agent = max(population, key=_last_fitness)
    best_agent.save_checkpoint(str(elite_path))
    logger.info("Saved best agent: %s", elite_path)

    try:
        hp_names = list(best_agent.registry.hp_config.names())
    except Exception:
        hp_names = None

    # Close the (async) training env before spinning up a render env.
    try:
        trainer.env.close()
    except Exception:
        pass

    # Render the best agent.
    render_best_agent(elite_path, env_name, seed, device, env_dir / "render.mp4")

    # Pull logged data back from W&B for plotting.
    history = fetch_wandb_history(project, run_name)
    if history.empty:
        logger.warning("No W&B history for %s; skipping plots.", env_name)
        return None
    history.to_csv(env_dir / "wandb_history.csv", index=False)

    scores = normalization_scores(algo, env_name)
    plotting.plot_fitness(
        history, env_name, scores, pop_size, str(env_dir / "fitness.png")
    )
    plotting.plot_mutation_schedule(
        history,
        env_name,
        pop_size,
        str(env_dir / "mutation_schedule.png"),
        hp_names=hp_names,
    )

    # Build the per-env normalized curve for the aggregate plot.
    data = history.dropna(
        subset=[plotting.GLOBAL_STEP_COL, plotting.BEST_FITNESS_COL]
    ).sort_values(plotting.GLOBAL_STEP_COL)
    if data.empty:
        return None
    x = data[plotting.GLOBAL_STEP_COL].to_numpy(dtype=float) / max(pop_size, 1)
    y = np.array([scores.normalize(f) for f in data[plotting.BEST_FITNESS_COL]])
    return x, y


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    """Run the benchmark end to end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    project = _prompt(args.project, "W&B project name: ")
    base_name = _prompt(args.name, "Benchmark run name: ")
    config_rel = _prompt(args.config, "Config path (relative to script): ")
    config_path = (SCRIPT_DIR / config_rel).resolve()
    if not config_path.is_file():
        msg = f"Config not found: {config_path}"
        raise FileNotFoundError(msg)

    base_manifest = load_manifest_dict(config_path)
    algo = base_manifest["algorithm"]["name"]
    if args.max_steps is not None:
        training = base_manifest.setdefault("training", {})
        training["max_steps"] = args.max_steps
        # Keep evo_steps valid (manifest requires evo_steps <= max_steps).
        evo = training.get("evo_steps")
        if evo is not None and evo > args.max_steps:
            training["evo_steps"] = args.max_steps

    seed = int(_prompt(args.seed, "Seed (int): "))
    device = resolve_device(_prompt(args.device, "Device (e.g. cpu/cuda/mps): "))

    valid = allowed_envs(algo)
    print(f"\nAlgorithm '{algo}'. Permitted environments:")
    for name in valid:
        print(f"  - {name}")
    print("  - all")
    raw = _prompt(
        args.envs, "\nEnvironments to run (space/comma separated, or 'all'): "
    )
    selection = list(raw.replace(",", " ").split())
    env_names = resolve_env_selection(algo, selection)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bench_dir = RESULTS_DIR / f"{base_name}-{algo}-{stamp}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact config used, plus seed/device/selection metadata.
    saved = copy.deepcopy(base_manifest)
    saved["benchmark"] = {
        "seed": seed,
        "device": device,
        "project": project,
        "name": base_name,
        "datetime": stamp,
        "environments": env_names,
    }
    with open(bench_dir / "config.yaml", "w") as f:
        yaml.safe_dump(saved, f, sort_keys=False)

    logger.info("Benchmark dir: %s", bench_dir)
    logger.info("Environments: %s", env_names)

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for env_name in env_names:
        try:
            result = run_environment(
                base_manifest=base_manifest,
                env_name=env_name,
                algo=algo,
                seed=seed,
                device=device,
                project=project,
                base_name=base_name,
                stamp=stamp,
                bench_dir=bench_dir,
                wandb_api_key=args.wandb_api_key,
            )
            if result is not None:
                curves[env_name] = result
        except Exception as exc:  # noqa: PERF203
            logger.exception("Environment %s failed: %s", env_name, exc)

    if curves:
        plotting.plot_aggregate(
            curves, str(bench_dir / "aggregate_normalized_fitness.png")
        )
        logger.info("Saved aggregate plot across %d environments.", len(curves))
    else:
        logger.warning("No environment produced plottable data.")


if __name__ == "__main__":
    main()
