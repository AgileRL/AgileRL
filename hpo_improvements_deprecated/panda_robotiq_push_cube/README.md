# EnvPool MuJoCo Playground `PandaRobotiqPushCube` PPO benchmark

This folder contains a ready-to-run benchmark for training a PPO agent on the
`PandaRobotiqPushCube` manipulation task from MuJoCo Playground via EnvPool.

## Environment

| Property | Value |
|---|---|
| EnvPool task ID | `PandaRobotiqPushCube-v1` |
| Observation | 48-dim flat vector |
| Action | 7-dim continuous, Box(-1, 1) |
| Episode length | 3000 steps |

## What is included

- `train.py`: Local trainer entrypoint that builds an EnvPool vector env and runs AgileRL PPO.
- `ppo_panda_robotiq_push_cube.yaml`: Default manifest with PPO hyperparameters.

## HPO status

HPO is disabled by default in this setup:

- `training.pop_size` is set to `1`
- no `mutation` block
- no `tournament_selection` block

## Run

```bash
uv run python hpo_improvements/panda_robotiq_push_cube/train.py
```

Optional flags:

```bash
uv run python hpo_improvements/panda_robotiq_push_cube/train.py --device cpu --wb
```

Or via the shell wrapper:

```bash
bash hpo_improvements/panda_robotiq_push_cube/run.sh --device cpu
```

Training saves the best (elite) PPO checkpoint to:

```
hpo_improvements/panda_robotiq_push_cube/checkpoints/panda_robotiq_push_cube_elite_PPO.pt
```

Override the path with:

```bash
uv run python hpo_improvements/panda_robotiq_push_cube/train.py \
  --elite-path /path/to/elite_ppo.pt
```
