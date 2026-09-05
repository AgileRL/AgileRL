# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``resume_from_checkpoint`` vs ``load_weights_from`` on ``build_algorithm``.

They are different operations and must not be confused:

* **resume** continues an interrupted run. The checkpoint is authoritative --
  optimizer moments, the lr-schedule position and the hyperparameters they were
  computed under all come back. A spec that disagrees gets a warning: checkpoint
  hyperparameters win; the spec does not describe the restored run.
* **load weights** warm-starts a *new* run from a previous one's parameters. Only
  the weights are taken; the optimizer, schedule and progress start fresh, and the
  spec keeps every hyperparameter.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import DQN
from agilerl.models.algorithms.dqn import DQNSpec
from tests.helper_functions import generate_discrete_space, generate_random_box_space

if HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig

    from agilerl.algorithms.grpo import GRPO
    from agilerl.models.algorithms.grpo import GRPOSpec
    from agilerl.utils.algo_utils import CosineLRScheduleConfig
    from tests import TINY_LLM_FIXTURE_PATH


def _lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )


def _schedule_config():
    return CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.1)


def _stub_tokenizer():
    return type(
        "Tok",
        (),
        {
            "eos_token_id": 0,
            "eos_token": "<pad>",
            "pad_token_id": 0,
            "pad_token": "<pad>",
            # Real HF tokenizers always expose this; the pad resolver reads it.
            "unk_token_id": None,
        },
    )()


@pytest.fixture(scope="module")
def grpo_checkpoint(tmp_path_factory):
    """One GRPO run, five scheduler steps in. Written once and shared read-only.

    Building a real causal-LM is the expensive part of this module, so it happens
    as few times as possible.
    """
    agent = GRPO(
        model_name=TINY_LLM_FIXTURE_PATH,
        pad_token_id=0,
        pad_token="<pad>",
        device="cpu",
        lr=5e-5,
        group_size=2,
        lora_config=_lora_config(),
        cosine_lr_schedule_config=_schedule_config(),
    )
    for _ in range(5):
        agent.lr_scheduler.step()

    path = str(tmp_path_factory.mktemp("grpo") / "run")
    agent.save_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def sft_checkpoints(tmp_path_factory):
    """One trained SFT agent, saved in both checkpoint formats."""
    from agilerl.algorithms.sft import SFT

    source = SFT(
        model_name=TINY_LLM_FIXTURE_PATH,
        pad_token_id=0,
        pad_token="<pad>",
        lora_config=_lora_config(),
        device="cpu",
        lr=5e-5,
    )
    for name, param in source.actor.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)

    base = tmp_path_factory.mktemp("sft")
    paths = {}
    for lora_only in (True, False):
        path = str(base / f"ckpt_{lora_only}")
        source.save_checkpoint(path, lora_only=lora_only)
        paths[lora_only] = path
    return paths


@pytest.fixture
def spaces():
    return generate_random_box_space(shape=(4,)), generate_discrete_space(2)


@pytest.fixture
def dqn_checkpoint(spaces, tmp_path):
    """A DQN checkpoint trained at lr=1e-3, 500 steps in."""
    observation_space, action_space = spaces
    agent = DQN(observation_space, action_space, lr=1e-3)
    for param in agent.actor.parameters():
        torch.nn.init.normal_(param, std=0.5)
    agent.steps = 500

    path = str(tmp_path / "ckpt.pt")
    agent.save_checkpoint(path)
    return agent, path


class TestRLSpecResumeVsLoad:
    def test_resume_takes_the_checkpoint_hyperparameters(self, spaces, dqn_checkpoint):
        _, path = dqn_checkpoint
        observation_space, action_space = spaces

        with pytest.warns(UserWarning, match="restored hyperparameters that differ"):
            agent = DQNSpec(lr=7e-4).build_algorithm(
                observation_space, action_space, index=0, resume_from_checkpoint=path
            )

        assert agent.lr == 1e-3
        assert agent.steps == 500

    def test_resume_is_silent_when_the_spec_agrees(self, spaces, dqn_checkpoint):
        _, path = dqn_checkpoint
        observation_space, action_space = spaces

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = DQNSpec(lr=1e-3).build_algorithm(
                observation_space, action_space, index=0, resume_from_checkpoint=path
            )

        assert agent.lr == 1e-3
        assert not [w for w in caught if "restored hyperparameters" in str(w.message)]

    def test_load_weights_keeps_the_spec_hyperparameters(self, spaces, dqn_checkpoint):
        source, path = dqn_checkpoint
        observation_space, action_space = spaces

        agent = DQNSpec(lr=7e-4).build_algorithm(
            observation_space, action_space, index=0, load_weights_from=path
        )

        assert agent.lr == 7e-4
        assert agent.steps == 0
        assert all(
            torch.allclose(before, after)
            for before, after in zip(
                source.actor.parameters(), agent.actor.parameters(), strict=True
            )
        )

    def test_the_two_options_are_mutually_exclusive(self, spaces, dqn_checkpoint):
        _, path = dqn_checkpoint
        observation_space, action_space = spaces

        with pytest.raises(ValueError, match="Provide exactly one of"):
            DQNSpec(lr=7e-4).build_algorithm(
                observation_space,
                action_space,
                index=0,
                resume_from_checkpoint=path,
                load_weights_from=path,
            )


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
class TestLLMSpecResumeVsLoad:
    """The lr schedule is the sharpest test: resuming must keep its decay point."""

    @staticmethod
    def _tokenizer():
        return _stub_tokenizer()

    def _spec(self, lr):
        return GRPOSpec(
            pretrained_model_name_or_path=TINY_LLM_FIXTURE_PATH,
            lr=lr,
            group_size=2,
            lora_config=_lora_config(),
            cosine_lr_schedule_config=_schedule_config(),
        )

    def test_resume_restores_the_schedule_position(self, grpo_checkpoint):
        agent = self._spec(5e-5).build_algorithm(
            tokenizer=self._tokenizer(),
            resume_from_checkpoint=grpo_checkpoint,
            device="cpu",
        )
        assert agent.lr_scheduler.last_epoch == 5
        assert agent.lr == 5e-5

    def test_resume_warns_when_the_spec_disagrees(self, grpo_checkpoint):
        with pytest.warns(UserWarning, match="restored hyperparameters that differ"):
            agent = self._spec(5e-6).build_algorithm(
                tokenizer=self._tokenizer(),
                resume_from_checkpoint=grpo_checkpoint,
                device="cpu",
            )
        # The checkpoint wins: its optimizer state belongs to its lr.
        assert agent.lr == 5e-5

    def test_load_weights_starts_a_fresh_schedule_and_optimizer(self, grpo_checkpoint):
        agent = self._spec(5e-6).build_algorithm(
            tokenizer=self._tokenizer(),
            load_weights_from=grpo_checkpoint,
            device="cpu",
        )
        assert agent.lr == 5e-6
        # Warm start, not resume: the schedule begins at its warmup start, exactly
        # as it would for an agent built without a checkpoint at all.
        assert agent.lr_scheduler.last_epoch == 0

        fresh = self._spec(5e-6).build_algorithm(
            tokenizer=self._tokenizer(), device="cpu"
        )
        assert [g["lr"] for g in agent.optimizer.optimizer.param_groups] == [
            g["lr"] for g in fresh.optimizer.optimizer.param_groups
        ]


class TestLoadWeightsCoversBothCheckpointFormats:
    """``load_weights`` must handle a LoRA checkpoint and a full-model one."""

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed"
    )
    @pytest.mark.parametrize("lora_only", [True, False])
    def test_llm_load_weights_takes_weights_and_nothing_else(
        self, sft_checkpoints, lora_only
    ):
        from agilerl.algorithms.sft import SFT

        target = SFT(
            model_name=TINY_LLM_FIXTURE_PATH,
            pad_token_id=0,
            pad_token="<pad>",
            lora_config=_lora_config(),
            device="cpu",
            lr=7e-6,
        )
        target.load_weights(sft_checkpoints[lora_only])

        params = dict(target.actor.named_parameters())
        adapters = [v for k, v in params.items() if "lora_B.actor" in k]
        assert adapters
        assert all(weight.abs().sum() > 0 for weight in adapters)
        assert target.lr == 7e-6

    def test_rl_load_weights_wraps_models_when_accelerated(
        self, spaces, dqn_checkpoint
    ):
        """The accelerator branch re-wraps the freshly-loaded modules."""
        _, path = dqn_checkpoint
        observation_space, action_space = spaces

        agent = DQN(observation_space, action_space, lr=7e-4)
        agent.accelerator = MagicMock()
        with patch.object(DQN, "wrap_models") as mock_wrap:
            agent.load_weights(path)

        mock_wrap.assert_called_once()

    def test_rl_load_weights_recompiles_when_compiled(self, spaces, dqn_checkpoint):
        """The torch.compile branch recompiles the freshly-loaded modules."""
        _, path = dqn_checkpoint
        observation_space, action_space = spaces

        agent = DQN(observation_space, action_space, lr=7e-4)
        agent.accelerator = None
        agent.torch_compiler = "default"
        with (
            patch.object(DQN, "recompile") as mock_recompile,
            patch("agilerl.algorithms.core.evolvable_checkpoint.configure_tf32_precision") as mock_tf32,
        ):
            agent.load_weights(path)

        mock_recompile.assert_called_once()
        mock_tf32.assert_called_once()
