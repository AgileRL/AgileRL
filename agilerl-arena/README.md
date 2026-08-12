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

Inference uses the same credential as the rest of the CLI: a personal access token
from Profile then CLI API key, or the token `arena login` stores. `Agent` picks up
`ARENA_API_KEY` when you pass no `api_key`.

```python
from agilerl.arena import Agent

agent = Agent("https://<deployment-id>.inference.agilerl.com", api_key="arena_pat_...")
action, _ = agent.get_action(observation)
```

### Chat sessions

LLM deployments keep chat history, and the CLI keeps track of which conversation
you are in. The first prompt starts one and later prompts carry on from it:

```bash
arena agent run my-chat-model
arena agent generate --prompt "My name is Sam."
arena agent generate --prompt "What is my name?"
```

Start a fresh conversation with `--new-session`, or end the current one with
`arena agent sessions clear` so the next prompt starts anew.

To move between conversations, `arena agent sessions resume` lists what is
stored and lets you pick one with the arrow keys:

```bash
arena agent sessions list                  # what is stored, current one marked
arena agent sessions resume                # arrow keys, Enter to pick
arena agent sessions get <session-id>      # read a transcript
arena agent sessions delete <session-id>   # remove one for good
```

Who a session belongs to follows the deployment's memory scope, chosen on the
first deploy with `arena agent deploy <experiment> --memory-scope user|organization`.
It defaults to `user` and cannot be changed later.

## Notes on packaging and imports

- Distribution name: `agilerl-arena`
- Python import namespace: `agilerl.arena`
- CLI command: `arena`

`agilerl-arena` and `agilerl` intentionally share the `agilerl.*` namespace as separate packages.
