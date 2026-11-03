# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Core AgileRL must import and run without the optional LLM extras.

``pip install agilerl`` does not install the ``cpu-llm`` extra (peft,
transformers, ...), so importing the core RL algorithms and trainers must not
pull those packages in. CI installs every extra, so each check runs in a fresh
interpreter that simulates a core-only install: the LLM modules are marked
missing in ``sys.modules`` and their distributions are hidden from
``importlib.metadata``, which makes ``agilerl.HAS_LLM_DEPENDENCIES`` False.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import agilerl

# Import names of the distributions in the ``cpu-llm`` extra, plus vLLM.
_LLM_MODULES = {
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "liger-kernel": "liger_kernel",
    "openenv": "openenv",
    "peft": "peft",
    "transformers": "transformers",
    "vllm": "vllm",
}

_SIMULATE_CORE_INSTALL = textwrap.dedent(
    """
    import importlib.metadata as metadata
    import re
    import sys

    hidden_distributions = {hidden_distributions!r}
    for module_name in {hidden_modules!r}:
        sys.modules[module_name] = None  # makes `import module_name` raise

    _real_version = metadata.version

    def _version(distribution_name):
        if re.sub(r"[-_.]+", "-", distribution_name).lower() in hidden_distributions:
            raise metadata.PackageNotFoundError(distribution_name)
        return _real_version(distribution_name)

    metadata.version = _version

    import agilerl

    assert agilerl.HAS_LLM_DEPENDENCIES is False
    """
)

_CORE_IMPORTS = textwrap.dedent(
    """
    from agilerl.algorithms import (
        CQN, DDPG, DQN, IPPO, MADDPG, MATD3, PPO, TD3, NeuralTS, NeuralUCB,
        RainbowDQN,
    )
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.train_bandits import train_bandits
    from agilerl.training.train_multi_agent_off_policy import (
        train_multi_agent_off_policy,
    )
    from agilerl.training.train_multi_agent_on_policy import (
        train_multi_agent_on_policy,
    )
    from agilerl.training.train_off_policy import train_off_policy
    from agilerl.training.train_offline import train_offline
    from agilerl.training.train_on_policy import train_on_policy
    from agilerl.utils.utils import (
        create_population, make_multi_agent_vect_envs, make_vect_envs,
    )
    """
)


def _run_core_only(code: str) -> subprocess.CompletedProcess[str]:
    preamble = _SIMULATE_CORE_INSTALL.format(
        hidden_distributions=sorted(_LLM_MODULES),
        hidden_modules=sorted(_LLM_MODULES.values()),
    )
    return subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_llm_extra_modules_are_listed():
    """Keep the simulated install in sync with the ``cpu-llm`` extra."""
    assert set(agilerl.LLM_PACKAGES) <= set(_LLM_MODULES)


def test_simulated_core_install_hides_llm_packages():
    result = _run_core_only(
        """
        import importlib.util

        for name in ("peft", "transformers"):
            assert importlib.util.find_spec(name) is None, name
            try:
                __import__(name)
            except ModuleNotFoundError:
                pass
            else:
                raise AssertionError(f"{name} should be missing")
        """
    )
    assert result.returncode == 0, result.stderr


def test_core_imports_without_llm_extras():
    result = _run_core_only(_CORE_IMPORTS)
    assert result.returncode == 0, result.stderr


def test_core_population_without_llm_extras():
    result = _run_core_only(
        _CORE_IMPORTS
        + textwrap.dedent(
            """
        from gymnasium import spaces

        obs_space = spaces.Box(-1.0, 1.0, (4,))
        pop = PPO.population(
            size=2, observation_space=obs_space, action_space=spaces.Discrete(2)
        )
        assert len(pop) == 2

        agent_ids = ["agent_0", "agent_1"]
        pop = IPPO.population(
            size=2,
            observation_space=[obs_space, obs_space],
            action_space=[spaces.Discrete(2), spaces.Discrete(2)],
            agent_ids=agent_ids,
        )
        assert len(pop) == 2
        """
        )
    )
    assert result.returncode == 0, result.stderr
