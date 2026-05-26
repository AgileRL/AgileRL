# EnvPool GFootball `run_to_score` DQN benchmark

This folder contains a ready-to-run benchmark setup for training a DQN agent on
Google Research Football (`run_to_score`) through EnvPool.

## What is included

- `train.py`: Local trainer entrypoint that builds an EnvPool vector env and runs AgileRL.
- `dqn_run_to_score.yaml`: Default manifest with practical DQN hyperparameters.
- `play_rendered.py`: Render native GFootball episodes from a saved AgileRL DQN checkpoint.

## HPO status

HPO is disabled by default in this setup:

- `training.pop_size` is set to `1`
- no `mutation` block
- no `tournament_selection` block

## Run

```bash
uv run python benchmarking/gfootball_run_to_score_envpool/train.py
```

Optional:

```bash
uv run python benchmarking/gfootball_run_to_score_envpool/train.py --device cpu --wb
```

Training always saves the best (elite) DQN checkpoint at the end. Default path:

`benchmarking/gfootball_run_to_score_envpool/checkpoints/gfootball_academy_run_to_score_elite_DQN.pt`

You can override this with:

```bash
uv run python benchmarking/gfootball_run_to_score_envpool/train.py \
  --elite-path /path/to/elite_dqn.pt
```

## Render a trained agent

EnvPool GFootball is headless, so rendering uses native `gfootball` directly.

```bash
uv run python benchmarking/gfootball_run_to_score_envpool/play_rendered.py \
  --checkpoint benchmarking/gfootball_run_to_score_envpool/checkpoints/gfootball_academy_run_to_score_elite_DQN.pt \
  --episodes 3
```

## Notes

- The default EnvPool task id is `gfootball/academy_run_to_score-v1`.
- Ensure `envpool` is built with GFootball task support in your environment.
- Rendering requires native `gfootball` Python package (`import gfootball.env`).
- By default, this benchmark flattens observations (`flatten_obs: true`) and
  uses an MLP DQN to avoid channel-order issues across wrappers.
- On macOS, if EnvPool import fails with a missing Qt5 framework, install it:
  `brew install qt@5`.
- If you run on Apple Silicon with `--device mps`, `run.sh` automatically sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1` to work around unsupported MPS ops (slower).
