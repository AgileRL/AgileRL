"""Benchmark PPO across environment suites under different HPO regimes.

Trains a fresh PPO agent (no HPO / AgileRL HPO / HPO improvements -- the regime
is whatever the YAML's ``mutation`` block encodes) on each selected environment,
logging to Weights & Biases, then pulls the run history back from W&B to produce
per-environment and aggregate plots plus a rendered video of the best agent.

Run directly from this folder, e.g.::

    python benchmark.py --config configs/ppo_relu_no_hpo.yaml \
        --project my_project --name run0 --seeds 1,2,3 --device cpu --envs all

Pass ``--seeds`` (or a single ``--seed``) to train one ``(environment, seed)``
run per seed sequentially; the per-environment over-seeds and cross-environment
aggregate plots are then produced exactly as the Ray orchestrator does, so the
sequential and parallel paths yield the same results without Ray. Any omitted
argument is prompted for interactively. ``--config`` is interpreted relative to
this script's directory.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import re
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

# Pin the BLAS/OpenMP backends to a single thread before torch/numpy are
# imported (they read these only at load time). This complements
# reproducibility.seed_everything's torch.set_num_threads(1): together they make
# CPU runs bit-reproducible regardless of the host's core count, so the
# sequential and Ray paths agree on matched hardware/builds. setdefault lets an
# explicit user override win. (Also runs in the Ray training task: it imports
# this module -- run_training's home -- before torch, so the pin lands there too;
# runtime-env.yaml additionally sets them at worker startup.)
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

# Allow running as a plain script: make sibling modules importable.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gymnasium as gym  # noqa: E402
import mechanism  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotting  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from registry import (  # noqa: E402
    MPE_ENV_CONFIGS,
    allowed_envs,
    env_config,
    env_suite_name,
    normalization_scores,
    resolve_env_selection,
)
from reproducibility import seed_everything  # noqa: E402

from agilerl.algorithms import DQN, IPPO, PPO  # noqa: E402
from agilerl.models.env import GymEnvSpec  # noqa: E402
from agilerl.models.manifest import TrainingManifest  # noqa: E402
from agilerl.training.trainer import LocalTrainer  # noqa: E402
from agilerl.utils.utils import set_mutation_history_dir  # noqa: E402
from agilerl.wrappers.agent import RSNorm  # noqa: E402

logger = logging.getLogger("hpo_benchmark")

RESULTS_DIR = SCRIPT_DIR / "results"
MAX_RENDER_STEPS = 3000
# Number of greedy episodes to record for the multi-agent (MPE) render. MPE episodes
# are short (~max_cycles steps), so a few are concatenated into one watchable clip.
MAX_RENDER_EPISODES = 5

# Filename of the rendered best-agent video, both on disk and on the W&B run.
RENDER_FILENAME = "render.mp4"

# Filename of the per-run training log, both on disk and on the W&B run.
LOG_FILENAME = "train.log"

# Filename of the per-generation evolutionary mutation history, both on disk and
# on the W&B run. Must match the name written by
# :class:`agilerl.logger.MutationHistoryLogger`.
MUTATION_HISTORY_FILENAME = "mutation_history.csv"


def elite_filename(algo: str) -> str:
    """Return the elite-checkpoint filename for *algo* (e.g. ``elite_PPO.pt``).

    Centralised so the worker (which writes it), the W&B upload, and the
    orchestrator (which downloads it) all agree on one name.
    """
    return f"elite_{algo}.pt"


def mutable_hp_names(manifest: dict[str, Any]) -> list[str]:
    """Return the RL hyperparameters the manifest allows to be mutated.

    These are exactly the keys of the ``mutation.rl_hp_selection`` block (the
    per-hyperparameter search ranges); with no ``mutation`` block (the no-HPO
    regime) the list is empty. The mutation-schedule plot is driven off this, so
    only hyperparameters the run could actually mutate are charted.

    :param manifest: Parsed YAML manifest dict.
    :type manifest: dict[str, Any]
    :return: Mutable RL-hyperparameter names, in manifest order.
    :rtype: list[str]
    """
    mutation = manifest.get("mutation") or {}
    rl_hp = mutation.get("rl_hp_selection") or {}
    return list(rl_hp.keys())


# --------------------------------------------------------------------------- #
# EnvPool helpers
# --------------------------------------------------------------------------- #
def _make_envpool_env(env_id: str, num_envs: int, seed: int) -> Any:
    """Return an EnvPool-backed vectorized Gymnasium environment.

    EnvPool seeds at creation time; callers must **not** call
    ``env.reset(seed=...)`` afterwards.

    EnvPool assigns sub-environment ``i`` the seed ``base + i``, so a run with
    base seed ``s`` consumes the contiguous range ``[s, s + num_envs)``. Passing
    consecutive run seeds (e.g. ``1, 2, 3``) straight through would therefore
    make those ranges overlap by ``num_envs - 1`` (seeds ``1`` and ``2`` share
    15 of 16 sub-environment seeds), so the per-seed runs would sample nearly
    identical environment instantiations and the over-seeds variance would be
    artificially small. To keep the ranges disjoint we scale the run seed by
    ``num_envs`` before handing it to EnvPool: run seed ``s`` -> base
    ``s * num_envs`` -> range ``[s * num_envs, (s + 1) * num_envs)``.

    Works for both MuJoCo (PPO) and Atari (DQN) suites. MuJoCo v5 observations
    are already flat 1-D vectors, so they are passed through
    :class:`gymnasium.wrappers.vector.FlattenObservation`. Atari observations
    are stacked 84x84 grayscale frames (shape ``(4, 84, 84)``) and must keep
    their spatial structure for the CNN encoder, so flattening is skipped for
    any environment whose single observation is more than 1-D.

    :param env_id: EnvPool environment id (e.g. ``"Ant-v4"`` or ``"Pong-v5"``).
    :type env_id: str
    :param num_envs: Number of parallel environments.
    :type num_envs: int
    :param seed: Run seed; scaled by ``num_envs`` to derive the EnvPool base
        seed so per-seed runs use disjoint sub-environment seed ranges.
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
    # Scale by num_envs so each run seed maps to a disjoint sub-environment seed
    # range [seed * num_envs, (seed + 1) * num_envs); see the docstring above.
    base_seed = seed * num_envs
    env = make_fn(env_id, num_envs=num_envs, seed=base_seed)
    # Only flatten flat (vector) observations. Image observations such as
    # Atari's (4, 84, 84) frame stack must keep their spatial structure so the
    # CNN encoder can consume them.
    single_space = getattr(env, "single_observation_space", env.observation_space)
    if len(getattr(single_space, "shape", ())) <= 1:
        env = gym.wrappers.vector.FlattenObservation(env)
    return env


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
                mf_pbt=validated.mf_pbt,
                replay_buffer=validated.replay_buffer,
                device=device,
            )
        finally:
            cls._prebuilt_env = None


def is_multi_agent_algo(algo: str) -> bool:
    """Return whether *algo* is trained on the multi-agent (PettingZoo) path.

    Single-agent suites (PPO/MuJoCo, DQN/Atari) build an EnvPool vector env that is
    injected into the trainer; multi-agent suites (IPPO/MPE) instead let
    :class:`LocalTrainer` build a PettingZoo vector env from the manifest's
    ``PzEnvSpec``. This predicate is the single switch between those two paths.
    """
    return algo.upper() == "IPPO"


def _build_trainer(
    manifest: dict[str, Any], env_name: str, algo: str, seed: int, device: str
) -> LocalTrainer:
    """Build a ready-to-train trainer for one (environment, algorithm).

    For single-agent algorithms this constructs an EnvPool vector env (seeded at
    creation) and injects it into an :class:`_EnvPoolLocalTrainer`. For multi-agent
    algorithms (IPPO) it bypasses EnvPool entirely and lets
    :meth:`LocalTrainer.from_manifest` build the PettingZoo vector env from the
    manifest's ``PzEnvSpec`` (whose custom entrypoint is the MPE adapter); the
    manifest's ``environment.config`` / ``environment.path`` are prepared by the
    caller (see :func:`run_training`).

    :param manifest: Parsed manifest dict (already env-name-stamped).
    :param env_name: Environment id for this run.
    :param algo: Algorithm name.
    :param seed: Run seed (used for the EnvPool base seed on the single-agent path).
    :param device: Torch device string.
    :return: A constructed (untrained) trainer.
    :rtype: LocalTrainer
    """
    if is_multi_agent_algo(algo):
        # PettingZoo path: the env is built from the manifest's PzEnvSpec, so no
        # pre-built env is injected. seed_everything (called by run_training) plus
        # the multi-agent loop's own env reset cover reproducibility.
        return LocalTrainer.from_manifest(manifest, device=device)

    num_envs = int(manifest.get("environment", {}).get("num_envs", 1) or 1)
    envpool_env = _make_envpool_env(env_name, num_envs, seed)
    return _EnvPoolLocalTrainer.from_manifest_with_env(
        manifest, envpool_env, device=device
    )


def _move_replay_buffer_to_cpu(trainer: LocalTrainer) -> None:
    """Relocate the trainer's off-policy replay buffer(s) to CPU (host RAM).

    The trainer builds its buffer on the training device, but the buffer's
    backing storage is allocated lazily on the first ``add()`` (which copies the
    transition to ``buffer.device``). Re-pointing ``buffer.device`` to ``"cpu"``
    *before* training therefore makes the whole buffer live in host RAM, keeping
    a large Atari replay buffer out of VRAM so an HPO population (one set of
    networks per agent) fits on the GPU. Sampled minibatches are small and are
    moved back to the agent's device in each algorithm's ``learn()`` (see
    ``DQN.learn``). On-policy algorithms (e.g. IPPO) have no replay buffer, so
    this is a no-op for them.

    :param trainer: A constructed (untrained) trainer.
    """
    for attr in ("memory", "n_step_memory"):
        buffer = getattr(trainer, attr, None)
        if buffer is not None:
            buffer.device = "cpu"


# --------------------------------------------------------------------------- #
# Input stage
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Missing values are prompted for interactively."""
    p = argparse.ArgumentParser(description="AgileRL PPO HPO benchmark")
    p.add_argument("--project", help="Weights & Biases project name")
    p.add_argument("--name", help="Benchmark base name (used in run/folder names)")
    p.add_argument("--config", help="Path to YAML config, relative to this script")
    p.add_argument("--seed", type=int, help="Single global seed (if --seeds omitted)")
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma/space separated seeds; each (env, seed) is run sequentially",
    )
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
    p.add_argument(
        "--move-replay-buffer-to-cpu",
        action="store_true",
        help=(
            "Store the off-policy replay buffer on CPU (host RAM) instead of the "
            "training device. Keeps a large Atari buffer out of VRAM so an HPO "
            "population fits on the GPU; sampled minibatches are moved to the "
            "agent's device in learn(). No effect for on-policy algos (no buffer)."
        ),
    )
    return p.parse_args(argv)


def _prompt(value: str | None, message: str) -> str:
    """Return ``value`` if set, else prompt the user."""
    if value is not None and str(value) != "":
        return str(value)
    return input(message).strip()


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    """Resolve the seed list from ``--seeds``, falling back to ``--seed`` / a prompt.

    One ``(environment, seed)`` learning run is executed per seed, mirroring the
    Ray orchestrator's grid so the sequential and parallel paths produce the same
    per-environment over-seeds and aggregate plots.
    """
    if args.seeds:
        return [int(s) for s in str(args.seeds).replace(",", " ").split()]
    if args.seed is not None:
        return [int(args.seed)]
    raw = _prompt(None, "Seeds (space/comma separated, e.g. '1 2 3'): ")
    return [int(s) for s in raw.replace(",", " ").split()]


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


# Matches CSI escape sequences (colours, cursor moves, line clears) plus the
# lone ESC-prefixed shorthands tqdm/rich emit. Stripping these is what turns an
# unreadable, control-code-riddled train.log into plain text.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|[@-Z\\-_])")


class _LogFileWriter:
    """File wrapper that renders terminal output as plain, readable text.

    Progress bars (tqdm) repaint a single line by emitting ANSI escapes and
    carriage returns. Mirroring those raw into a file leaves it full of ``\\x1b``
    control codes and overwritten fragments. This wrapper strips the escapes and
    resolves ``\\r`` overwrites in a small line buffer, so each line lands in the
    file only in its final rendered state -- exactly what you'd see on screen.
    """

    def __init__(self, file: Any) -> None:
        self._file = file
        self._line = ""

    def write(self, data: str) -> int:
        n = len(data)
        data = _ANSI_RE.sub("", data).replace("\b", "")
        for ch in data:
            if ch == "\n":
                self._file.write(self._line + "\n")
                self._line = ""
            elif ch == "\r":
                # Carriage return: the next write overwrites this line.
                self._line = ""
            else:
                self._line += ch
        self._file.flush()
        return n

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if self._line:
            self._file.write(self._line + "\n")
            self._line = ""
        self._file.flush()


@contextmanager
def tee_to_file(path: Path):
    """Mirror stdout/stderr and root logging to ``path`` for the duration.

    Output is sanitized on the way to disk (ANSI escapes stripped, ``\\r``
    progress repaints collapsed) so the saved log is plain text, while the live
    console stream keeps its original formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    orig_out, orig_err = sys.stdout, sys.stderr
    raw = open(path, "w")
    file = _LogFileWriter(raw)
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
        raw.close()


# --------------------------------------------------------------------------- #
# W&B history fetch
# --------------------------------------------------------------------------- #
def _find_wandb_run(project: str, run_name: str) -> Any:
    """Locate a finished W&B run by display name.

    :param project: W&B project name.
    :param run_name: Display name of the run to fetch.
    :return: A ``wandb.apis.public.Run`` or ``None`` if it cannot be found.
    :rtype: Any
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
    return run


def wandb_run_succeeded(project: str, run_name: str) -> bool:
    """Whether a W&B run exists and finished cleanly (``state == "finished"``).

    Used by the Ray orchestrator to *recover* a job that Ray reports FAILED even
    though its training run actually completed and uploaded every artifact -- the
    classic case being the worker being killed by a signal (OOM, node preemption,
    or an uncatchable teardown-time exit) *after* ``wandb.finish()`` and all
    uploads have already succeeded. ``"finished"`` is only ever set by an explicit
    ``wandb.finish()``, so it cleanly separates a fully-completed run from one that
    truly died mid-training (which stays ``running`` or goes ``crashed``/``failed``/
    ``killed``). Such a finished run is durable on W&B and should be collected, not
    dropped. Best-effort: any lookup error is treated as "not recoverable".

    :param project: W&B project name.
    :param run_name: Display name of the run to check.
    :return: True only if the run is present and in the ``finished`` state.
    :rtype: bool
    """
    try:
        run = _find_wandb_run(project, run_name)
    except Exception as exc:  # noqa: BLE001 - recovery probe must never raise
        logger.warning("W&B completion check failed for %s: %s", run_name, exc)
        return False
    return run is not None and getattr(run, "state", None) == "finished"


_TERMINAL_RUN_STATES = ("finished", "crashed", "failed", "killed")


def _history_is_complete(run: Any, df: pd.DataFrame) -> bool:
    """Return whether *df* holds the run's full history.

    The final history point is flushed at ``wandb.finish()`` (e.g. the last
    evaluation logged just before training returns), *separately* from the
    points streamed during training. Right after finish the run already reports
    a terminal state — and Ray marks its job SUCCEEDED — while ``scan_history``
    has not yet ingested that final point, so an eager fetch silently drops it.

    The authoritative target is the run **summary's** final ``train/global_step``:
    the summary is synced at finish and is reliably present once the run is
    terminal, *even while the history time-series is still being ingested*. We
    therefore wait until the streamed history reaches that step.

    ``lastHistoryStep`` is **not** used as the primary signal: it is itself
    derived from ingested history, so it lags low in lockstep with the missing
    point (reporting 2 when step 3 hasn't landed), which would let a truncated
    history pass. It is only a fallback when no summary step is available.

    :param run: A ``wandb.apis.public.Run``.
    :param df: History streamed via ``scan_history`` so far.
    :return: True if *df* reaches the run's final logged step (or completeness
        cannot be verified for an already-terminal run).
    """
    if df.empty:
        return False
    # A still-running run will keep logging, so never accept early.
    state = getattr(run, "state", None)
    if state is not None and state not in _TERMINAL_RUN_STATES:
        return False

    # Primary signal: the summary's final global step (synced at finish).
    summary = getattr(run, "summary", None)
    target_step = None
    if summary is not None:
        try:
            target_step = summary.get(plotting.GLOBAL_STEP_COL)
        except Exception:
            target_step = None
    if target_step is not None and plotting.GLOBAL_STEP_COL in df.columns:
        gs = df[plotting.GLOBAL_STEP_COL].dropna()
        return not gs.empty and float(gs.max()) >= float(target_step)

    # Fallback: server-side last-history-step index vs the fetched _step.
    target = getattr(run, "lastHistoryStep", None)
    if target is None or target < 0 or "_step" not in df.columns:
        # Can't verify against a step target; accept what a terminal run gave us.
        return True
    last = df["_step"].dropna()
    return not last.empty and int(last.max()) >= int(target)


def fetch_wandb_history(
    project: str,
    run_name: str,
    *,
    max_wait: float = 180.0,
    poll_interval: float = 5.0,
) -> pd.DataFrame:
    """Fetch the full logged history of a finished W&B run as a DataFrame.

    Polls ``scan_history`` until it spans the run's authoritative last step (see
    :func:`_history_is_complete`), guarding against the read-after-finish race
    where the public history API has not yet caught up with ``wandb.finish()``
    and returns a truncated history. The longest history seen is kept, so a slow
    backend degrades to fewer points rather than to whatever the first poll
    happened to return.

    :param project: W&B project name.
    :type project: str
    :param run_name: Display name of the run to fetch.
    :type run_name: str
    :param max_wait: Seconds to keep polling for a complete history before
        giving up and returning the most complete history seen.
    :type max_wait: float
    :param poll_interval: Seconds between polls.
    :type poll_interval: float
    :return: History dataframe (includes ``_runtime`` relative time), or empty.
    :rtype: pandas.DataFrame
    """
    import time  # noqa: PLC0415

    run = _find_wandb_run(project, run_name)
    if run is None:
        return pd.DataFrame()

    deadline = time.monotonic() + max_wait
    best = pd.DataFrame()
    while True:
        df = pd.DataFrame(list(run.scan_history()))
        if len(df) > len(best):
            best = df
        if _history_is_complete(run, df):
            return df
        if time.monotonic() >= deadline:
            summary = getattr(run, "summary", None)
            target_step = summary.get(plotting.GLOBAL_STEP_COL) if summary else None
            got_step = (
                best[plotting.GLOBAL_STEP_COL].dropna().max()
                if plotting.GLOBAL_STEP_COL in best.columns
                and not best[plotting.GLOBAL_STEP_COL].dropna().empty
                else "?"
            )
            logger.warning(
                "W&B history for %s still incomplete after %.0fs (%d rows; "
                "reached global_step %s of %s) — plotting with what is available.",
                run_name,
                max_wait,
                len(best),
                got_step,
                target_step,
            )
            return best
        time.sleep(poll_interval)
        # Re-fetch so state / lastHistoryStep reflect server-side progress.
        run = _find_wandb_run(project, run_name) or run


def upload_run_files(
    project: str, run_name: str, files: list[Path], root: Path
) -> None:
    """Attach local files to a finished W&B run so the driver can fetch them.

    By the time a learning run returns from ``trainer.train(...)`` the W&B run is
    already finished (its loggers are closed), so we cannot ``wandb.log`` more.
    Instead we attach the artifacts to the existing run via the public API. Each
    file is stored under its path relative to *root* (e.g. ``elite_PPO.pt``,
    ``render.mp4``), the same names :func:`download_run_files` looks for.

    This is what makes the Ray path work: the worker writes the checkpoint/render
    to its (remote) node, uploads them here, and the orchestrator pulls them back
    from W&B — the only channel shared between cluster and driver.

    :param project: W&B project name.
    :param run_name: Display name of the finished run to attach to.
    :param files: Local files to upload (missing ones are skipped).
    :param root: Directory the stored names are taken relative to.
    """
    files = [f for f in files if f.is_file()]
    if not files:
        return
    run = _find_wandb_run(project, run_name)
    if run is None:
        logger.warning(
            "No W&B run for %s; cannot upload %d file(s).", run_name, len(files)
        )
        return
    for f in files:
        try:
            run.upload_file(str(f), root=str(root))
            logger.info("Uploaded %s to W&B run %s", f.name, run_name)
        except Exception:
            logger.exception("Failed to upload %s to W&B run %s", f.name, run_name)


def download_run_files(
    project: str, run_name: str, dest_dir: Path, filenames: list[str]
) -> None:
    """Download named files from a finished W&B run into *dest_dir*.

    Used by the parallel orchestrator to pull the elite checkpoint and render
    MP4 that a remote Ray worker produced (and uploaded via
    :func:`upload_run_files`) back into the local results tree. Files already
    present locally are skipped, so the sequential path — where the worker wrote
    them straight into *dest_dir* — does no redundant network round-trips.

    :param project: W&B project name.
    :param run_name: Display name of the finished run.
    :param dest_dir: Local directory to download into.
    :param filenames: Names of the run files to fetch (as stored on W&B).
    """
    missing = [n for n in filenames if not (dest_dir / n).is_file()]
    if not missing:
        return
    run = _find_wandb_run(project, run_name)
    if run is None:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        try:
            run.file(name).download(root=str(dest_dir), replace=True)
            logger.info("Downloaded %s from W&B run %s", name, run_name)
        except Exception:
            logger.warning(
                "W&B file '%s' not available for run %s (skipping).", name, run_name
            )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
# Loaders for the elite checkpoint, keyed by algorithm name. Rendering is only
# wired up for the algorithms with a registered environment suite (registry.py).
_RENDER_LOADERS = {"PPO": PPO.load, "DQN": DQN.load, "IPPO": IPPO.load}


def _ale_render_id(env_id: str) -> str:
    """Map an EnvPool Atari id to its Gymnasium ALE id (``Pong-v5`` -> ``ALE/Pong-v5``)."""
    return env_id if env_id.startswith("ALE/") else f"ALE/{env_id}"


def _make_atari_render_env(env_id: str) -> Any:
    """Build a Gymnasium ALE env whose observations match the EnvPool training env.

    EnvPool's Atari envs emit a 4-frame stack of 84x84 grayscale images (shape
    ``(4, 84, 84)`` uint8) produced with frame-skip 4 and a random number of
    initial no-ops (max 30). This reproduces that observation pipeline with
    Gymnasium's :class:`~gymnasium.wrappers.AtariPreprocessing` and
    :class:`~gymnasium.wrappers.FrameStackObservation` so the trained CNN sees
    the same input format it trained on, while ``env.render()`` still returns the
    full-colour ``(210, 160, 3)`` game frames for the MP4.

    :param env_id: EnvPool Atari id (e.g. ``"Pong-v5"``).
    :type env_id: str
    :return: A Gymnasium env yielding ``(4, 84, 84)`` uint8 observations.
    :rtype: Any
    """
    import ale_py  # noqa: F401  (importing registers the ALE/* namespace)

    base = gym.make(
        _ale_render_id(env_id),
        render_mode="rgb_array",
        frameskip=1,  # AtariPreprocessing applies the frame skip itself
        repeat_action_probability=0.0,  # match EnvPool (no sticky actions)
    )
    env = gym.wrappers.AtariPreprocessing(
        base,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    return gym.wrappers.FrameStackObservation(env, stack_size=4)


def render_best_agent(
    elite_path: Path,
    env_name: str,
    seed: int,
    device: str,
    out_path: Path,
    algo: str,
) -> None:
    """Render one greedy episode of the saved elite agent to an MP4.

    The render path is algorithm-aware: PPO agents play their (continuous)
    MuJoCo env, while DQN agents play the Gymnasium ALE Atari env built by
    :func:`_make_atari_render_env` and act with greedy (epsilon = 0) discrete
    actions. Failures are logged but never raised, so a missing render backend
    never aborts the benchmark.

    :param elite_path: Path to the saved elite checkpoint.
    :type elite_path: Path
    :param env_name: Environment id (EnvPool/Gymnasium, e.g. ``"Ant-v4"`` or ``"Pong-v5"``).
    :type env_name: str
    :param seed: Seed for the render env's first reset.
    :type seed: int
    :param device: Torch device.
    :type device: str
    :param out_path: Destination .mp4 path.
    :type out_path: Path
    :param algo: Algorithm name (selects the checkpoint loader and env pipeline).
    :type algo: str
    """
    is_atari = algo.upper() == "DQN"

    # Pusher is unrenderable here: ``Pusher-v4`` is only supported on
    # ``mujoco<3`` (its block is lighter than air on newer MuJoCo, so Gymnasium
    # disabled the model -- https://github.com/Farama-Foundation/Gymnasium/issues/950),
    # and this project runs mujoco>=3. EnvPool trains it fine and the elite
    # checkpoint is still saved upstream of this call; only the video is skipped.
    if not is_atari and env_name.split("-")[0] == "Pusher":
        logger.info(
            "Skipping render for %s: Pusher is only renderable on mujoco<3.",
            env_name,
        )
        return

    # Headless MuJoCo rendering: export MUJOCO_GL=egl (GPU) or osmesa (CPU)
    # before running. We do not set it here because MuJoCo selects its GL
    # backend at import time and forcing EGL would break env creation on hosts
    # without an EGL library.
    #
    # When MUJOCO_GL is unset and there is no DISPLAY, MuJoCo falls back to GLFW
    # and aborts with a *native* error (libc++abi / SIGABRT) that Python cannot
    # catch -- which would kill the whole benchmark. So we skip rendering unless
    # a usable GL backend is available. ALE/Atari renders headless via rgb_array
    # and needs neither DISPLAY nor MUJOCO_GL, so this guard is MuJoCo-only.
    if not is_atari:
        import os
        import platform

        # On macOS, MuJoCo renders via glfw (Quartz) and needs neither DISPLAY
        # nor MUJOCO_GL.  On Linux without a display, glfw aborts at the C level
        # (not catchable), so we require either DISPLAY or a headless backend.
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

    loader = _RENDER_LOADERS.get(algo.upper())
    if loader is None:
        logger.warning(
            "Rendering not supported for algorithm '%s'; skipping %s.", algo, env_name
        )
        return

    try:
        import imageio

        agent = loader(str(elite_path), device=device)
        agent.set_training_mode(False)

        if is_atari:
            render_env = _make_atari_render_env(env_name)
        else:
            render_env = gym.wrappers.FlattenObservation(
                gym.make(env_name, render_mode="rgb_array")
            )

        obs, _ = render_env.reset(seed=seed)
        frames: list[np.ndarray] = []
        done = False
        steps = 0
        while not done and steps < MAX_RENDER_STEPS:
            frames.append(render_env.render())
            if is_atari:
                # Greedy discrete action; DQN.get_action defaults to epsilon = 0.
                action = agent.get_action(np.expand_dims(obs, 0))
                obs, _, term, trunc, _ = render_env.step(
                    int(np.asarray(action).reshape(-1)[0])
                )
            else:
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


def render_best_multi_agent(
    elite_path: Path,
    env_name: str,
    seed: int,
    device: str,
    out_path: Path,
    algo: str,
) -> None:
    """Render greedy episodes of the saved IPPO elite on a cooperative MPE task.

    The multi-agent counterpart of :func:`render_best_agent`: it rebuilds a single
    (non-vectorized) ``mpe2`` parallel env with ``render_mode="rgb_array"`` using the
    exact :data:`registry.MPE_ENV_CONFIGS` kwargs the agent trained on, loads the elite
    IPPO population, and steps it with greedy actions, collecting one full-scene frame
    per step. The per-agent dict action loop is agent-agnostic, so it handles both
    homogeneous tasks (Spread/Reference) and the heterogeneous Speaker-Listener (whose
    agents have different action spaces) without special-casing. Failures are logged but
    never raised, so a missing render backend never aborts the benchmark.

    :param elite_path: Path to the saved elite IPPO checkpoint.
    :type elite_path: Path
    :param env_name: MPE task id (e.g. ``"simple_spread_v3"``).
    :type env_name: str
    :param seed: Seed for the render env's first reset.
    :type seed: int
    :param device: Torch device.
    :type device: str
    :param out_path: Destination .mp4 path.
    :type out_path: Path
    :param algo: Algorithm name (selects the checkpoint loader; ``"IPPO"``).
    :type algo: str
    """
    loader = _RENDER_LOADERS.get(algo.upper())
    if loader is None:
        logger.warning(
            "Rendering not supported for algorithm '%s'; skipping %s.", algo, env_name
        )
        return

    # MPE renders via pygame/SDL. On a display-less Linux host SDL aborts unless a
    # driver is set; the dummy driver still produces rgb_array frames. macOS renders
    # directly. Set before the first mpe2/pygame import in this process.
    import os
    import platform

    if platform.system() != "Darwin" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    try:
        import imageio
        from mpe_adapter import make_mpe_parallel_env

        agent = loader(str(elite_path), device=device)
        agent.set_training_mode(False)

        # Same construction as training (registry config), plus rgb_array rendering.
        render_env = make_mpe_parallel_env(
            render_mode="rgb_array", **MPE_ENV_CONFIGS[env_name]
        )

        frames: list[np.ndarray] = []
        steps = 0
        # A few episodes for a watchable clip (MPE episodes are short, ~max_cycles).
        for episode in range(MAX_RENDER_EPISODES):
            obs, info = render_env.reset(seed=seed + episode)
            while render_env.agents and steps < MAX_RENDER_STEPS:
                frame = render_env.render()
                if frame is not None:
                    frames.append(frame)
                # Greedy per-agent actions; squeeze the (non-vectorized) batch dim and
                # cast discrete actions to int (mirrors IPPO.test's non-vectorized path).
                action, _, _, _ = agent.get_action(obs=obs, infos=info)
                action = {
                    a: int(np.asarray(act).reshape(-1)[0]) for a, act in action.items()
                }
                obs, _, term, trunc, info = render_env.step(action)
                steps += 1
        render_env.close()

        if frames:
            imageio.mimsave(str(out_path), frames, fps=15)
            logger.info("Saved render: %s (%d frames)", out_path, len(frames))
    except Exception as exc:
        logger.exception("Rendering failed for %s: %s", env_name, exc)


# --------------------------------------------------------------------------- #
# W&B fetch + plotting (shared with the Ray orchestrator)
# --------------------------------------------------------------------------- #
def _read_mutation_history(env_dir: Path) -> pd.DataFrame | None:
    """Load a run's ``mutation_history.csv`` if present and non-empty.

    :param env_dir: The per-seed run directory.
    :return: The parsed dataframe, or None when the file is absent or unreadable
        (no-HPO and some regimes still write it; the mechanism plots degrade to a
        placeholder when it is missing).
    """
    path = env_dir / MUTATION_HISTORY_FILENAME
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    return df if not df.empty else None


def fetch_and_plot(
    *,
    project: str,
    run_name: str,
    algo: str,
    env_name: str,
    pop_size: int,
    env_dir: Path,
    hp_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None] | None:
    """Pull a finished W&B run, write the CSV, plot, and return the curve.

    This is the read-only, network-and-plot half of a learning run. It is shared
    by the local sequential benchmark (:func:`run_environment`) and the Ray
    orchestrator so both produce byte-for-byte identical plots and aggregate
    curves. It needs only W&B and the normalization registry — never the
    cluster-side training artifacts.

    :param project: W&B project name.
    :param run_name: Display name of the finished W&B run to fetch.
    :param algo: Algorithm name (used to look up normalization scores).
    :param env_name: Environment id.
    :param pop_size: Population size (per-agent step normalization for the x-axis).
    :param env_dir: Output directory for this run's CSV and plots.
    :param hp_names: Optional hyperparameter names for the mutation-schedule plot.
    :return: ``(x, best_fitness, normalized_fitness, diversity)`` — best fitness
        feeds the per-environment over-seeds plot, normalized fitness the
        aggregate, and ``diversity`` maps each logged ``eval/*_diversity`` column
        to its per-cycle values (aligned on the same rows) for the diversity
        over-seeds and aggregate plots (None when no diversity was logged). The
        whole tuple is None if there is no plottable history.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[str,
        numpy.ndarray] | None] | None
    """
    env_dir.mkdir(parents=True, exist_ok=True)

    # Pull logged data back from W&B for plotting.
    history = fetch_wandb_history(project, run_name)
    if history.empty:
        logger.warning("No W&B history for %s; skipping plots.", env_name)
        return None
    history.to_csv(env_dir / "wandb_history.csv", index=False)

    # Pull the elite checkpoint and render back from W&B. For the sequential
    # path these already sit in env_dir (skipped); for the Ray path they were
    # produced on a remote worker and this is how they reach the driver.
    download_run_files(
        project,
        run_name,
        env_dir,
        [
            elite_filename(algo),
            RENDER_FILENAME,
            LOG_FILENAME,
            MUTATION_HISTORY_FILENAME,
        ],
    )

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
    plotting.plot_dormant_fraction(
        history, env_name, pop_size, str(env_dir / "dormant_fraction.png")
    )
    plotting.plot_diversity(history, env_name, pop_size, str(env_dir / "diversity.png"))

    # Evolutionary-mechanism diagnostics. The hyperparameter trajectory/landscape
    # come from the W&B history; the efficacy and population-dynamics figures come
    # from the per-generation mutation-history CSV (downloaded above for the Ray
    # path, already present for the sequential path). All degrade to a placeholder
    # when their input is missing or degenerate.
    plotting.plot_population_hp_trajectory(
        history,
        env_name,
        pop_size,
        str(env_dir / "mechanism_hp_population.png"),
        hp_names=hp_names,
    )
    plotting.plot_hp_fitness(
        history,
        env_name,
        pop_size,
        str(env_dir / "mechanism_hp_fitness.png"),
        hp_names=hp_names,
    )
    mut_df = _read_mutation_history(env_dir)
    plotting.plot_mechanism_efficacy(
        mut_df, env_name, str(env_dir / "mechanism_efficacy.png")
    )
    plotting.plot_mechanism_efficacy_distribution(
        mut_df, env_name, str(env_dir / "mechanism_efficacy_distribution.png")
    )
    plotting.plot_mechanism_population(
        mut_df, env_name, pop_size, str(env_dir / "mechanism_population.png")
    )

    # Build the per-env normalized curve for the aggregate plot.
    data = history.dropna(
        subset=[plotting.GLOBAL_STEP_COL, plotting.BEST_FITNESS_COL]
    ).sort_values(plotting.GLOBAL_STEP_COL)
    if data.empty:
        return None
    x = data[plotting.GLOBAL_STEP_COL].to_numpy(dtype=float) / max(pop_size, 1)
    best = data[plotting.BEST_FITNESS_COL].to_numpy(dtype=float)
    y = np.array([scores.normalize(f) for f in best])

    # Per-env diversity curves (aligned on the same rows as the fitness curve),
    # one array per logged diagnostic, for the over-seeds and aggregate plots.
    diversity = {
        col: data[col].to_numpy(dtype=float)
        for col, _ in plotting.DIVERSITY_SPECS
        if col in data.columns
    }
    return x, best, y, (diversity or None)


def plot_seed_aggregates(
    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    bench_dir: Path,
    algo_label: str,
    suite_name: str,
) -> None:
    """Draw per-env over-seeds plots and the cross-env aggregate from seed curves.

    Shared by the sequential benchmark (:func:`main`) and the Ray orchestrator so
    both reduce each environment's seeds to one IQM curve (with a
    stratified-bootstrap CI) the exact same way, then feed those into the
    aggregate normalized-fitness plot and the performance profile.

    :param curves: Mapping env_name -> list of ``(x, best_fitness,
        normalized_fitness)``, one entry per finished seed.
    :param bench_dir: Benchmark output directory (plots are written under it).
    :param algo_label: Display label for the algorithm + HPO method.
    :param suite_name: Human-readable environment-suite name (aggregate and
        performance-profile titles).
    """
    env_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for env_name, seed_curves in curves.items():
        out_path = bench_dir / env_name / "fitness_over_seeds.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seed_stack = plotting.plot_fitness_over_seeds(
            seed_curves, env_name, str(out_path), algo_label=algo_label
        )
        if seed_stack is not None:
            env_curves[env_name] = seed_stack
            logger.info(
                "Saved over-seeds plot for %s (%d seeds).", env_name, len(seed_curves)
            )

    if not env_curves:
        logger.warning("No run produced plottable data; skipping aggregate.")
        return
    plotting.plot_aggregate(
        env_curves,
        str(bench_dir / "aggregate_normalized_fitness.png"),
        algo_label=algo_label,
        suite_name=suite_name,
    )
    plotting.plot_performance_profile(
        env_curves,
        str(bench_dir / "performance_profile.png"),
        algo_label=algo_label,
        suite_name=suite_name,
    )
    logger.info("Saved aggregate plot across %d environments.", len(env_curves))


def plot_diversity_aggregates(
    curves: dict[str, list[tuple[np.ndarray, dict[str, np.ndarray]]]],
    bench_dir: Path,
    algo_label: str,
    suite_name: str,
) -> None:
    """Draw per-env over-seeds and cross-env aggregate population-diversity plots.

    Mirrors :func:`plot_seed_aggregates` for the three normalised-diversity
    diagnostics: each environment's seeds are reduced to one IQM curve (with a
    stratified-bootstrap CI) per diagnostic, then fed into the suite-level
    aggregate. Shared by the sequential benchmark and the Ray orchestrator.

    :param curves: Mapping env_name -> list of ``(x, {column: values})``, one
        entry per finished seed (the ``diversity`` element of
        :func:`fetch_and_plot`).
    :param bench_dir: Benchmark output directory (plots are written under it).
    :param algo_label: Display label for the algorithm + HPO method.
    :param suite_name: Human-readable environment-suite name (aggregate titles).
    """
    per_metric_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for env_name, seed_curves in curves.items():
        out_path = bench_dir / env_name / "diversity_over_seeds.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stacks = plotting.plot_diversity_over_seeds(
            seed_curves, env_name, str(out_path), algo_label=algo_label
        )
        for col, grid_stack in stacks.items():
            per_metric_curves.setdefault(col, {})[env_name] = grid_stack

    if not per_metric_curves:
        logger.warning("No run logged diversity; skipping diversity aggregate.")
        return
    plotting.plot_diversity_aggregate(
        per_metric_curves,
        str(bench_dir / "aggregate_diversity.png"),
        algo_label=algo_label,
        suite_name=suite_name,
    )
    logger.info("Saved diversity aggregate across %d environments.", len(curves))


def plot_mechanism_aggregates(
    bench_dir: Path,
    algo_label: str,
    suite_name: str,
) -> None:
    """Draw the per-environment over-seeds and cross-environment aggregate figures.

    Self-contained: it re-reads each finished seed's ``mutation_history.csv`` from
    disk (every per-seed run already wrote one) rather than threading mechanism
    data through :func:`fetch_and_plot`'s return value, so it adds nothing to the
    contract the sequential and Ray paths share. For each environment it writes an
    ``over_seeds`` figure pooling that environment's seeds, then the suite
    aggregate over every environment -- mirroring the fitness/diversity layout:

    * ``<env>/mechanism_efficacy_over_seeds.png`` and
      ``aggregate_mechanism_efficacy.png`` -- per-category win-rate and mean
      change (rliable bootstrap CIs). The over-seeds figure uses raw episodic
      return; the suite aggregate uses the change in **expert-normalised** fitness
      (per-env, so a high-return env does not dominate the cross-env pool).
    * ``<env>/mechanism_efficacy_distribution_over_seeds.png`` and
      ``aggregate_mechanism_efficacy_distribution.png`` -- the per-category
      probability density of the per-cycle change (raw per env, normalised for the
      suite aggregate).
    * ``<env>/mechanism_population_over_seeds.png`` and
      ``aggregate_mechanism_population.png`` -- the IQM (with stratified-bootstrap
      band) of selection pressure and turnover over per-agent steps. Only
      lineage-informative seeds contribute (see
      :func:`mechanism.lineage_is_informative`); regimes that record no lineage
      (no-HPO, MF-PBT) simply yield no curve.

    :param bench_dir: Benchmark output directory (the per-env/per-seed tree).
    :param algo_label: Display label for the algorithm + HPO method.
    :param suite_name: Human-readable environment-suite name (for titles).
    """
    suite_deltas: dict[str, list[np.ndarray]] = {}
    per_metric: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    drew_any = False

    # Algorithm name (read from the persisted manifest), used to expert-normalise
    # the suite-level deltas per environment so a high-return env does not dominate
    # the cross-environment pool. The per-env over-seeds figures stay in raw units.
    algo = None
    config_path = bench_dir / "config.yaml"
    if config_path.is_file():
        try:
            algo = load_manifest_dict(config_path).get("algorithm", {}).get("name")
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Could not read algorithm from %s: %s", config_path, exc)

    for env_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        env_name = env_dir.name
        env_deltas: dict[str, list[np.ndarray]] = {}
        sel_curves: list[tuple[np.ndarray, np.ndarray]] = []
        tur_curves: list[tuple[np.ndarray, np.ndarray]] = []

        # Expert-normalisation for this env's suite-level deltas (None -> raw
        # fitness when the env has no registered random/expert baseline).
        normalize = None
        if algo is not None:
            try:
                normalize = normalization_scores(algo, env_name).normalize
            except (KeyError, ValueError):
                logger.warning(
                    "No normalisation baseline for %s/%s; aggregate uses raw deltas.",
                    algo,
                    env_name,
                )

        for seed_dir in sorted(env_dir.glob("s*")):
            mut_df = _read_mutation_history(seed_dir)
            if mut_df is None:
                continue
            for cat, d in mechanism.category_deltas(mut_df).items():
                env_deltas.setdefault(cat, []).append(d)  # raw, for over-seeds
            for cat, d in mechanism.category_deltas(
                mut_df, transform=normalize
            ).items():
                suite_deltas.setdefault(cat, []).append(d)  # normalised, for suite
            if not mechanism.lineage_is_informative(mut_df):
                continue
            pop_size = max(int(mut_df["agent_slot"].nunique()), 1)
            sx, sy = mechanism.selection_pressure(mut_df, pop_size)
            tx, ty = mechanism.turnover(mut_df, pop_size)
            if sx.size:
                sel_curves.append((sx, sy))
            if tx.size:
                tur_curves.append((tx, ty))

        if not env_deltas and not sel_curves and not tur_curves:
            continue
        drew_any = True

        # Per-environment over-seeds figures (pool this env's seeds, raw units).
        if env_deltas:
            env_pooled = {cat: np.concatenate(a) for cat, a in env_deltas.items()}
            plotting.plot_mechanism_efficacy_over_seeds(
                env_pooled,
                env_name,
                str(env_dir / "mechanism_efficacy_over_seeds.png"),
            )
            plotting.plot_mechanism_efficacy_distribution_over_seeds(
                env_pooled,
                env_name,
                str(env_dir / "mechanism_efficacy_distribution_over_seeds.png"),
            )

        stacks = plotting.plot_mechanism_population_over_seeds(
            sel_curves,
            tur_curves,
            env_name,
            str(env_dir / "mechanism_population_over_seeds.png"),
            algo_label=algo_label,
        )
        for key, grid_stack in stacks.items():
            per_metric.setdefault(key, {})[env_name] = grid_stack

    # Suite-level aggregates over every environment.
    if suite_deltas:
        pooled = {cat: np.concatenate(a) for cat, a in suite_deltas.items()}
        plotting.plot_mechanism_efficacy_aggregate(
            pooled,
            str(bench_dir / "aggregate_mechanism_efficacy.png"),
            algo_label=algo_label,
            suite_name=suite_name,
        )
        plotting.plot_mechanism_efficacy_distribution_aggregate(
            pooled,
            str(bench_dir / "aggregate_mechanism_efficacy_distribution.png"),
            algo_label=algo_label,
            suite_name=suite_name,
        )
    if per_metric:
        plotting.plot_mechanism_population_aggregate(
            per_metric,
            str(bench_dir / "aggregate_mechanism_population.png"),
            algo_label=algo_label,
            suite_name=suite_name,
        )

    if drew_any:
        logger.info(
            "Saved mechanism over-seeds and aggregate figures under %s.", bench_dir
        )
    else:
        logger.warning(
            "No run wrote usable mutation history; skipping mechanism aggregate."
        )


# --------------------------------------------------------------------------- #
# Training core (one learning run)
# --------------------------------------------------------------------------- #
def _silence_wandb_service_teardown() -> None:
    """Make wandb's service teardown tolerant of socket errors (non-fatal).

    On Python 3.13, wandb's asyncio-based service teardown can raise
    ``BrokenPipeError`` from ``StreamWriter.wait_closed()`` when the run is
    finished and the service socket is closed -- *after* the run history has
    already been logged (a known CPython 3.13 asyncio ``wait_closed`` issue).
    That error otherwise propagates out of ``trainer.train()`` and fails the
    whole run even though training and logging succeeded.

    We cannot dodge it by pinning an older wandb: the legacy (non-asyncio)
    service was removed in wandb 0.21.0, while support for the new 86-char
    ``wandb_v1_`` API keys only arrived in 0.22.3 -- so every key-compatible
    version uses the asyncio service. Instead we wrap the teardown entry point
    to log-and-swallow any teardown-time error. Idempotent; a no-op if the
    internal class is absent (older/newer wandb layouts).

    Two distinct failures are covered:

    * The Python 3.13 ``BrokenPipeError`` on the first teardown.
    * ``teardown()`` may only run once and raises ``AssertionError`` on a second
      call. wandb invokes it both at run finalization and via an ``atexit`` hook
      (``teardown_atexit``), so after the first call swallows a ``BrokenPipeError``
      the ``atexit`` call would otherwise raise "Already torn down" and fail the
      run *after* training, checkpointing and uploads have all succeeded. We
      short-circuit when already torn down and swallow every teardown exception.
    """
    try:
        from wandb.sdk.lib.service import service_connection
    except Exception:
        return

    cls = getattr(service_connection, "ServiceConnection", None)
    orig = getattr(cls, "teardown", None)
    if cls is None or orig is None or getattr(orig, "_hpo_wrapped", False):
        return

    def _safe_teardown(self: Any, *args: Any, **kwargs: Any) -> Any:
        # A second teardown (e.g. the atexit hook after an explicit finish)
        # would raise AssertionError("Already torn down."); skip it quietly.
        if getattr(self, "_torn_down", False):
            return None
        try:
            return orig(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - teardown errors are non-fatal here
            logger.warning("Suppressed wandb service teardown error: %s", exc)
            return None

    _safe_teardown._hpo_wrapped = True  # type: ignore[attr-defined]
    cls.teardown = _safe_teardown


def _finish_active_wandb_run() -> None:
    """Finish the process-global W&B run if one is still active.

    AgileRL opens a W&B run per learning run and closes it (``wandb.finish()``)
    via ``population.finish()`` only at the *normal* end of ``trainer.train()``.
    If training raises partway, that close is skipped and the run stays open.
    Because the sequential runner reuses **one process** and W&B permits only one
    active run per process, an orphaned run blocks the *next* seed's
    ``wandb.init()`` from starting its backend service -- its ``debug.log`` never
    reaches "starting backend", no ``.wandb`` history is written, and every
    ``wandb.log()`` for that seed silently goes nowhere (the run is then dropped
    from the over-seeds aggregate). Calling this in ``run_training``'s ``finally``
    guarantees the run is always finished before control returns to the seed loop.

    Idempotent: a no-op on the normal path, where ``population.finish()`` has
    already cleared ``wandb.run``. Teardown-time errors are swallowed (mirroring
    :func:`_silence_wandb_service_teardown`) so this never masks the real training
    exception propagating out of the ``finally``.
    """
    import wandb  # noqa: PLC0415

    if getattr(wandb, "run", None) is None:
        return
    try:
        wandb.finish()
    except Exception as exc:  # noqa: BLE001 - finalization must never mask the real error
        logger.warning("Failed to finish active W&B run cleanly: %s", exc)


def _sync_offline_wandb_run(
    wandb_dir: Path,
    *,
    wandb_api_key: str | None,
    retries: int = 5,
    timeout: int = 300,
) -> bool:
    """Upload a finished *offline* W&B run to the server synchronously.

    With ``WANDB_MODE=offline`` the run's full history is written durably to
    ``wandb_dir/wandb/`` during training, independent of the live service
    socket. ``wandb sync`` then uploads that on-disk run in one synchronous,
    retriable transfer, so the complete history reaches W&B even when the live
    async upload would have dropped it (e.g. the Python 3.13 asyncio teardown
    ``BrokenPipeError`` at ``wandb.finish()`` aborting the history-backlog drain).

    :param wandb_dir: The ``WANDB_DIR`` used for the run; its ``wandb/`` subdir
        holds the ``offline-run-*`` directory (and a ``latest-run`` symlink).
    :param wandb_api_key: API key for the upload (else ``WANDB_API_KEY`` env).
    :param retries: Attempts on transient failure (exponential backoff).
    :param timeout: Per-attempt timeout in seconds.
    :return: True if the sync succeeded.
    :rtype: bool
    """
    import subprocess  # noqa: PLC0415
    import time  # noqa: PLC0415

    runs_root = wandb_dir / "wandb"
    target = runs_root / "latest-run"
    if not target.exists():
        # Fall back to the newest run dir if the convenience symlink is absent.
        candidates = sorted(runs_root.glob("*run-*"))
        if not candidates:
            logger.warning("No offline W&B run found under %s to sync.", runs_root)
            return False
        target = candidates[-1]

    # The sync subprocess must reach the server, so it runs without the offline
    # WANDB_MODE this process set for training.
    env = dict(os.environ)
    env.pop("WANDB_MODE", None)
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "wandb", "sync", str(target)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode == 0:
                logger.info("Synced offline W&B run %s to the server.", target.name)
                return True
            logger.warning(
                "wandb sync attempt %d/%d failed (rc=%d): %s",
                attempt,
                retries,
                result.returncode,
                (result.stderr or result.stdout or "")[-500:],
            )
        except Exception as exc:  # noqa: BLE001 - retry transient sync failures
            logger.warning(
                "wandb sync attempt %d/%d errored: %s", attempt, retries, exc
            )
        time.sleep(min(2**attempt, 30))

    logger.error(
        "Failed to sync offline W&B run %s after %d attempts.", target, retries
    )
    return False


def derive_pop_size(manifest: dict) -> int:
    """Resolve the population size from a raw (unvalidated) manifest dict.

    HPO / no-HPO configs set ``training.pop_size`` explicitly, but MF-PBT configs
    omit it -- the manifest validator derives it as
    ``n_subpopulations * n_individuals_per_subpopulation`` and never writes it back
    into the raw dict. This mirrors that derivation so the harness (which reads the
    raw dict, not the validated manifest) gets the right pop_size for the x-axis
    normalisation. Falls back to 1 only when neither source is available.

    :param manifest: Raw manifest dict (as loaded from YAML).
    :return: Population size.
    :rtype: int
    """
    explicit = manifest.get("training", {}).get("pop_size")
    if explicit:
        return int(explicit)
    mf = manifest.get("mf_pbt")
    if mf:
        return int(mf["n_subpopulations"]) * int(mf["n_individuals_per_subpopulation"])
    return 1


def run_training(
    *,
    base_manifest: dict[str, Any],
    env_name: str,
    algo: str,
    seed: int,
    device: str,
    project: str,
    run_name: str,
    out_dir: Path,
    wandb_api_key: str | None,
    render: bool = True,
    move_replay_buffer_to_cpu: bool = False,
) -> dict[str, Any]:
    """Train a fresh agent for one (environment, seed), checkpoint and render.

    This is the compute-heavy half of a learning run: it builds the EnvPool
    environment, trains the population (with or without HPO, per the manifest),
    saves the best agent, and optionally renders it. It deliberately does **not**
    fetch W&B history or plot — that is the caller's job (see
    :func:`fetch_and_plot`). The Ray entrypoint (``ray_code/worker.py``) wraps
    this function in a remote task; the sequential benchmark calls it directly.

    :param base_manifest: Parsed YAML manifest (will be deep-copied).
    :param env_name: Gymnasium environment id (overrides the manifest's env name).
    :param algo: Algorithm name (used only for the checkpoint filename).
    :param seed: Global seed for reproducibility.
    :param device: Torch device string (``cpu``/``cuda``/``mps``).
    :param project: W&B project name.
    :param run_name: W&B run display name.
    :param out_dir: Directory for ``elite_{algo}.pt``, ``train.log`` and the render.
    :param wandb_api_key: W&B API key (else ``WANDB_API_KEY`` env var is used).
    :param render: Whether to render the best agent to an MP4.
    :param move_replay_buffer_to_cpu: Keep the off-policy replay buffer on CPU
        (host RAM) instead of the training device (see
        :func:`_move_replay_buffer_to_cpu`). No effect for on-policy algos.
    :return: ``{"run_name", "pop_size", "hp_names", "elite_path"}``.
    :rtype: dict[str, Any]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard against wandb's Python 3.13 asyncio teardown raising a (harmless)
    # BrokenPipeError that would otherwise fail this run after logging succeeds.
    _silence_wandb_service_teardown()

    multi_agent = is_multi_agent_algo(algo)

    manifest = copy.deepcopy(base_manifest)
    manifest["environment"]["name"] = env_name
    pop_size = derive_pop_size(manifest)

    # Multi-agent (IPPO): the env is built from the manifest's PzEnvSpec via a custom
    # adapter entrypoint (e.g. mpe_adapter), so point its `config` at the selected task
    # (env_config resolves the per-env factory kwargs from the registry) and make the
    # adapter importable from the spawned vec-env subprocesses (which do not inherit
    # this process's sys.path) by pinning `path` to this benchmarking dir.
    if multi_agent:
        env_cfg = manifest["environment"]
        env_cfg.setdefault("config", {})
        env_cfg["config"].update(env_config(algo, env_name))
        if not env_cfg.get("path"):
            env_cfg["path"] = str(SCRIPT_DIR)

    # Fresh, reproducible agent. On the single-agent path EnvPool is seeded at
    # creation (do not call env.reset(seed=...) afterwards).
    seed_everything(seed)
    trainer = _build_trainer(manifest, env_name, algo, seed, device)
    if move_replay_buffer_to_cpu:
        _move_replay_buffer_to_cpu(trainer)

    # Multi-agent (PettingZoo) env seeding for reproducibility. Unlike EnvPool
    # (seeded at creation), AgileRL's multi-agent on-policy loop calls env.reset()
    # WITHOUT a seed (agilerl/training/train_multi_agent_on_policy.py), so the
    # PettingZoo vector env would draw fresh OS entropy each run and the collected
    # rollouts -- hence the fitness curve itself -- would not be reproducible. We
    # therefore seed the shared vec env exactly ONCE here, before training. After
    # this single seeded reset the env's RNG advances deterministically, so the
    # loop's later unseeded resets stay reproducible across runs yet still vary per
    # episode (re-seeding on every reset would instead restart every episode from
    # the same layout and harm training -- see reproducibility.seed_everything's
    # note). The seed is scaled by num_envs so consecutive run seeds map to disjoint
    # sub-env seed ranges, mirroring the EnvPool path's rationale.
    if multi_agent:
        num_envs = int(manifest.get("environment", {}).get("num_envs", 1) or 1)
        trainer.env.reset(seed=seed * num_envs)

    # Observation normalization (PPO only). "ReLU to the Rescue" (Jesson et al.,
    # 2024) normalizes observations to zero mean / unit variance -- in AgileRL
    # this is the RSNorm agent wrapper, which keeps running mean/std statistics and
    # normalizes the observation before both action selection and learning (its
    # learn() has a dedicated PPO branch that normalizes the rollout buffer; the
    # stale "off-policy only" warning in its docstring no longer holds). The
    # wrapper is applied here, before trainer.train(), so the statistics are
    # learned online over the course of training; constructing it rewires each
    # agent's get_action/learn in place, and we also replace the population entries
    # so the elite that gets checkpointed (and reloaded for rendering) carries its
    # obs_rms stats -- PPO.load reconstructs the wrapper from the checkpoint, so the
    # rendered/evaluated agent normalizes with the exact statistics it trained with.
    #
    # Restricted to PPO/MuJoCo: this is the continuous-control recipe from the
    # paper. DQN/Atari observations are uint8 image stacks (handled by the CNN's
    # own /255 scaling), so running-mean/std normalization is not applied there.
    if algo.upper() == "PPO":
        trainer.population = [RSNorm(agent) for agent in trainer.population]

    elite_path = out_dir / elite_filename(algo)
    log_path = out_dir / LOG_FILENAME

    # Log W&B *offline*: the full run history is written durably to disk during
    # training and uploaded afterwards in one synchronous, retried `wandb sync`
    # (see _sync_offline_wandb_run). This is the actual fix for the missing-history
    # runs -- it removes the dependency on the live wandb service socket, whose
    # Python 3.13 asyncio teardown can BrokenPipe at finish and abort the drain of
    # the not-yet-uploaded history backlog, leaving a run on W&B with a summary but
    # no history at all.
    #
    # The offline run dir must be pinned per (env, seed) so we know what to sync.
    # WANDB_DIR alone is NOT enough in the sequential runner: run_training is called
    # repeatedly in one process, and wandb reads WANDB_DIR only when its session
    # first starts -- subsequent changes are ignored ("Changes to your wandb
    # environment variables will be ignored because your wandb session has already
    # started"). That made every run after the first write its offline data into the
    # first run's dir, so fetch_and_plot found no history for any later run. Passing
    # `dir` directly to wandb.init() (via addl_args) is a per-run setting that is
    # honored on every call. The env var is kept for the first run / external tools.
    os.environ["WANDB_DIR"] = str(out_dir)
    os.environ["WANDB_MODE"] = "offline"

    logger.info("=== Training %s on %s (run '%s') ===", algo, env_name, run_name)
    # Record a per-generation evolutionary mutation history alongside the other
    # run artifacts. Set process-locally so it covers every algorithm/trainer.
    set_mutation_history_dir(out_dir)
    try:
        with tee_to_file(log_path):
            population, _ = trainer.train(
                wb=True,
                wandb_api_key=wandb_api_key,
                wandb_kwargs={
                    "project": project,
                    "addl_args": {"name": run_name, "dir": str(out_dir)},
                },
                verbose=True,
            )
    finally:
        set_mutation_history_dir(None)
        # Guarantee the W&B run is closed even if trainer.train() raised: an
        # orphaned (unfinished) run blocks the *next* seed's wandb.init() in this
        # shared process, silently losing that seed's history (see
        # _finish_active_wandb_run). On the normal path this is a no-op.
        _finish_active_wandb_run()

    # Save the best agent of the final population (works with or without HPO).
    def _last_fitness(agent: Any) -> float:
        fit = getattr(agent, "fitness", None)
        return fit[-1] if fit else float("-inf")

    best_agent = max(population, key=_last_fitness)
    best_agent.save_checkpoint(str(elite_path))
    logger.info("Saved best agent: %s", elite_path)

    # The mutation-schedule plot charts exactly the hyperparameters the YAML
    # allows to be mutated (the mutation.rl_hp_selection keys).
    hp_names = mutable_hp_names(manifest)

    # Close the (async) training env before spinning up a render env.
    try:
        trainer.env.close()
    except Exception:
        pass

    # Render the best agent. The render path is split by suite: single-agent
    # (PPO/MuJoCo, DQN/Atari) plays a Gymnasium env, while multi-agent (IPPO/MPE)
    # plays a single non-vectorized PettingZoo env (render_best_multi_agent).
    render_path = out_dir / RENDER_FILENAME
    if render:
        render_fn = render_best_multi_agent if multi_agent else render_best_agent
        render_fn(elite_path, env_name, seed, device, render_path, algo)

    # Upload the offline run (full history + summary + console) to W&B as a
    # synchronous, retried transfer. After this the run exists on the server with
    # its complete history, so it is both fetchable and viewable on W&B. Done
    # before upload_run_files so the run exists when the artifacts are attached.
    _sync_offline_wandb_run(out_dir, wandb_api_key=wandb_api_key)

    # Attach the elite checkpoint, render and training log to the finished W&B
    # run so the orchestrator (which may be on a different machine than this
    # worker) can pull them back via :func:`download_run_files`. The log is
    # complete here: the tee_to_file block has already exited.
    artifacts = [elite_path, log_path]
    if render and render_path.is_file():
        artifacts.append(render_path)
    # The mutation history (written by MutationHistoryLogger into out_dir during
    # training) is uploaded too so the orchestrator can pull it back: on the Ray
    # path out_dir is a throwaway remote working dir, so without this the CSV
    # would never reach the local results tree.
    mutation_history_path = out_dir / MUTATION_HISTORY_FILENAME
    if mutation_history_path.is_file():
        artifacts.append(mutation_history_path)
    upload_run_files(project, run_name, artifacts, root=out_dir)

    return {
        "run_name": run_name,
        "pop_size": pop_size,
        "hp_names": hp_names,
        "elite_path": str(elite_path),
    }


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
    move_replay_buffer_to_cpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None] | None:
    """Train, log, fetch, render and plot for a single environment.

    The training/checkpoint/render half is :func:`run_training` (also wrapped as
    a Ray task by ``ray_code/worker.py``); the W&B-fetch + plot half is
    :func:`fetch_and_plot`. Both halves are reused by the Ray orchestrator,
    keeping a single source of truth.

    :return: ``(x, best_fitness, normalized_fitness, diversity)`` for the plots,
        or None on failure.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[str,
        numpy.ndarray] | None] | None
    """
    env_dir = bench_dir / env_name / f"s{seed}"
    run_name = f"{base_name}-{algo}-{env_name}-s{seed}-{stamp}"

    result = run_training(
        base_manifest=base_manifest,
        env_name=env_name,
        algo=algo,
        seed=seed,
        device=device,
        project=project,
        run_name=run_name,
        out_dir=env_dir,
        wandb_api_key=wandb_api_key,
        render=True,
        move_replay_buffer_to_cpu=move_replay_buffer_to_cpu,
    )

    return fetch_and_plot(
        project=project,
        run_name=run_name,
        algo=algo,
        env_name=env_name,
        pop_size=result["pop_size"],
        env_dir=env_dir,
        hp_names=result["hp_names"],
    )


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

    seeds = _parse_seeds(args)
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
        "seeds": seeds,
        "device": device,
        "project": project,
        "name": base_name,
        "datetime": stamp,
        "environments": env_names,
    }
    with open(bench_dir / "config.yaml", "w") as f:
        yaml.safe_dump(saved, f, sort_keys=False)

    logger.info("Benchmark dir: %s", bench_dir)
    logger.info("Environments: %s | seeds: %s", env_names, seeds)

    # env -> one (x, best_fitness, normalized_fitness) per finished seed. Same
    # shape the Ray orchestrator collects, so plot_seed_aggregates reduces seeds
    # to IQM curves identically on both paths.
    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    diversity_curves: dict[str, list[tuple[np.ndarray, dict[str, np.ndarray]]]] = {}
    for env_name in env_names:
        for seed in seeds:
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
                    move_replay_buffer_to_cpu=args.move_replay_buffer_to_cpu,
                )
                if result is not None:
                    x, best, norm, diversity = result
                    curves.setdefault(env_name, []).append((x, best, norm))
                    if diversity is not None:
                        diversity_curves.setdefault(env_name, []).append((x, diversity))
            except Exception as exc:  # noqa: PERF203
                logger.exception(
                    "Environment %s (seed %d) failed: %s", env_name, seed, exc
                )

    algo_label = f"{algo.upper()} ({base_name})"
    suite_name = env_suite_name(algo)
    if curves:
        plot_seed_aggregates(curves, bench_dir, algo_label, suite_name)
    else:
        logger.warning("No environment produced plottable data.")
    if diversity_curves:
        plot_diversity_aggregates(diversity_curves, bench_dir, algo_label, suite_name)
    # Mechanism aggregates re-read the per-seed mutation-history CSVs from disk,
    # so they run whenever any environment produced output.
    if curves:
        plot_mechanism_aggregates(bench_dir, algo_label, suite_name)


if __name__ == "__main__":
    main()
