# agilerl-arena

`agilerl-arena` is the standalone Arena SDK + CLI package for AgileRL.

It provides:

- Python client for Arena workflows (auth, environment validation, experiment submission, deployment, inference)
- `arena` CLI for scripting and CI usage
- Lightweight manifest validation models for Arena job manifests

This package is distributed independently from core `agilerl`, but exposes modules through the shared namespace:

```python
from agilerl.arena import ArenaClient, Agent
```

## Installation

Install directly:

```bash
pip install agilerl-arena
```

Or install through core AgileRL extras:

```bash
pip install "agilerl[arena]"
```

## Quickstart

### 1) Authenticate

Preferred for CI/automation:

```bash
export ARENA_API_KEY="arena_pat_..."
```

Or interactive login:

```bash
arena login
```

### 2) Validate an environment

```bash
arena env validate --source path/to/my_env.py --name my-env
```

### 3) Submit a training manifest

```bash
arena experiments submit path/to/manifest.yaml --project my-project
```

## Python SDK example

```python
from agilerl.arena import ArenaClient

client = ArenaClient()  # uses ARENA_API_KEY if set

client.validate_environment(
    source="acrobot.py",
    name="acrobot-env",
)

result = client.submit_experiment(
    manifest="dqn.yaml",
    resource_id="arena-medium",
    project="my-project",
)

print(result)
```

## Inference example

```python
from agilerl.arena import Agent

agent = Agent("https://<deployment-id>.inference.agilerl.com", api_key="arena_pat_...")
action, _ = agent.get_action(observation)
```

## Notes on packaging and imports

- Distribution name: `agilerl-arena`
- Python import namespace: `agilerl.arena`
- CLI command: `arena`

`agilerl-arena` and `agilerl` intentionally share the `agilerl.*` namespace as separate packages.

## On-prem cluster registration

Register a customer Kubernetes cluster against Arena (enterprise on-prem v1) and
write Helm values files locally:

```bash
arena on-prem cluster register --name my-cluster --profile enterprise \
  --storage-endpoint http://s3.corp.example.com:9000 \
  --storage-bucket arena-prod \
  --storage-secret-name corp-s3
```

Lab / PoC profile (bundled MinIO values included):

```bash
arena on-prem cluster register --name lab-cluster --profile lab
```

Requires an Arena server with the enterprise on-prem cluster register API
(agilerl-platform Phase 1 / MR-a9). Run `arena on-prem enable` first, or omit
`--skip-enable` to enable the provider automatically.
