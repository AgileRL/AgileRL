from unittest.mock import ANY, MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from accelerate import Accelerator

pytest.importorskip("transformers", reason="LLM dependencies not installed")
pytest.importorskip("deepspeed", reason="LLM tests require deepspeed.")
pytest.importorskip("vllm", reason="LLM tests require vllm.")

from agilerl.algorithms import DPO, GRPO, LLMPPO, LLMREINFORCE
from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.sft import SFT
from agilerl.rollouts.on_policy import collect_rollouts_llm
from agilerl.training.train_llm import (
    _format_prefixed_metrics,
    _normalize_learn_metrics,
    build_eval_wandb_dict,
    build_train_wandb_dict,
    finetune_llm_reasoning,
    train_llm_dataset,
    train_llm_rollout,
)


def _wire_phased_env_mock(mock_env):
    """Delegate BatchRolloutEnv's phased interface to a per-env mock's
    reset()/step() (mirrors ``tests.helpers.rollout_doubles.RolloutEnvDoubleMixin``)
    so mocks drive the real concurrent collector path — reset/step call counts
    and step side effects (e.g. flipping ``done``) still apply.
    """
    pending: dict = {}

    def _reset_fetch(seed=None, *, row_index=None):
        return mock_env.reset(seed=seed, row_index=row_index)

    def _step_prepare(full_completion, sampling_logps=None):
        pending["step"] = (full_completion, sampling_logps)
        return ""

    def _step_apply(_env_result):
        full_completion, sampling_logps = pending["step"]
        return mock_env.step(full_completion, sampling_logps=sampling_logps)

    mock_env._reset_fetch.side_effect = _reset_fetch
    mock_env._reset_apply.side_effect = lambda obs, info, *, row_index=None: (obs, info)
    mock_env._step_prepare.side_effect = _step_prepare
    mock_env._step_env.side_effect = lambda gen_text: None
    mock_env._step_apply.side_effect = _step_apply


def _make_multiturn_mock_env(*, turn_boundaries_len: int = 3):
    """GEM-style env: reset/step/get_episode_data + turn_boundaries for step accounting."""
    mock_env = MagicMock(
        spec=[
            "reset",
            "step",
            "get_episode_data",
            "turn_boundaries",
            "done",
            "current_prompt",
            # Phased interface BatchRolloutEnv drives (reset/step are the
            # single-env composition of these).
            "_reset_fetch",
            "_reset_apply",
            "_step_prepare",
            "_step_env",
            "_step_apply",
        ],
    )
    prompt_dict: dict = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    mock_env.reset.return_value = (prompt_dict, {})
    mock_env.step.return_value = (prompt_dict, 0.0, False, False, {})
    _wire_phased_env_mock(mock_env)
    mock_env.turn_boundaries = list(range(turn_boundaries_len))
    mock_env.done = False
    mock_env.current_prompt = prompt_dict
    L = 8
    T = 2
    mock_env.get_episode_data.return_value = (
        torch.ones(1, L, dtype=torch.long),
        torch.ones(1, L, dtype=torch.long),
        torch.zeros(1, L, dtype=torch.long),
        torch.ones(T, dtype=torch.float32),
        None,
    )
    return mock_env


def _make_pop_for_wandb_dict(size: int = 2):
    pop = []
    for idx in range(size):
        agent = MagicMock()
        agent.index = idx
        agent.registry = MagicMock()
        agent.registry.hp_config = MagicMock()
        agent.registry.hp_config.config = {"lr": 1e-4}
        agent.lr = 1e-4 + idx * 1e-5
        pop.append(agent)
    return pop


def _make_multiturn_mock_agent(*, spec=LLMPPO):
    mock_agent = MagicMock(spec=spec)
    mock_agent.fitness = [0.0]
    if spec is LLMPPO:
        mock_agent.algo = "LLMPPO"
    elif spec is LLMREINFORCE:
        mock_agent.algo = "LLMREINFORCE"
    elif spec is GRPO:
        mock_agent.algo = "GRPO"
        mock_agent.group_size = 1
    else:
        mock_agent.algo = getattr(spec, "__name__", "MOCK")

    def _mock_get_action(obs, training=True, **kwargs):
        if isinstance(obs, dict):
            input_ids = obs.get("input_ids")
            batch = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 1
        else:
            batch = len(obs)
        return ActionResult(
            [torch.ones(1, 5, dtype=torch.long) for _ in range(batch)], None
        )

    mock_agent.get_action.side_effect = _mock_get_action
    if spec is GRPO:
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_kl": 0.2}
    else:
        mock_agent.learn.return_value = {
            "mean_loss": 0.5,
            "mean_kl": 0.2,
            "mean_pg_loss": 0.1,
            "mean_vf_loss": 0.1,
            "mean_entropy": 1.0,
        }
    mock_agent.batch_size = 16
    mock_agent.batch_size_per_process = 16
    mock_agent.max_model_len = 1024
    mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    mock_agent.steps = [10]
    mock_agent.scores = [0.0]
    mock_agent.index = 0
    mock_agent.mut = 0
    mock_agent.device = torch.device("cpu")
    mock_agent.set_reference_policy = MagicMock()
    # ``agent.test`` returns ``np.array(mean_fit)`` for real LLM algos; provide a
    # tensorable default so the training loop can wrap it in ``torch.tensor``.
    mock_agent.test.return_value = np.array(0.5, dtype=np.float32)
    return mock_agent


def _make_multiturn_env_factory(*, turn_boundaries_len: int = 3):
    return lambda: _make_multiturn_mock_env(turn_boundaries_len=turn_boundaries_len)


class TestFinetuneLlmPreference:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_preference_basic_training_loop(self, use_accelerator):
        """Test the basic training loop in finetune_llm."""
        # Create mock agent
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.fitness = [0.0]
        mock_agent.local_rank = "0"  # Main process
        mock_agent.get_action = MagicMock()
        mock_agent.learn.return_value = (0.5, 0.2, 0.1)
        mock_agent.test.return_value = 0.87
        mock_agent.batch_size = 32
        mock_agent.batch_size_per_process = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        # Create mock environment - use MagicMock for special methods
        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        example_prefernce_env_return = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env.reset.return_value = example_prefernce_env_return
        mock_env.step.return_value = example_prefernce_env_return
        mock_env.data_batch_size_per_gpu = 1

        # Mock other dependencies
        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_agg.return_value = 0.5
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                accelerator=Accelerator() if use_accelerator else None,
            )
            assert mock_env.reset.call_count == 6
            assert mock_agent.get_action.call_count == 0
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            expected_agg_calls = 21
            assert mock_agg.call_count == expected_agg_calls
            if not use_accelerator:
                assert all(
                    call_args.args[0] is None for call_args in mock_agg.call_args_list
                )
            assert mock_agent.test.call_count == 3  # Should be called at step 2

    @pytest.mark.parametrize(
        "use_accelerator",
        [True, False],
    )
    def test_finetune_llm_preference_with_wandb_and_checkpoints(self, use_accelerator):
        """Test finetune_llm with wandb logging and checkpointing enabled."""
        # Create mock agent
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = (0.5, 0.2, 0.1)
        mock_agent.test.return_value = 0.87
        mock_agent.batch_size = 32
        mock_agent.batch_size_per_process = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.lr = 0.001

        # Create mock environment - use MagicMock for special methods
        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        example_prefernce_env_return = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env.reset.return_value = example_prefernce_env_return
        mock_env.step.return_value = example_prefernce_env_return
        mock_env.data_batch_size_per_gpu = 1

        # Mock dependencies
        with (
            patch("agilerl.training.train_llm.trange") as mock_trange,
            patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
            patch("agilerl.training.train_llm.wandb") as mock_wandb,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
        ):
            # Configure mocks
            mock_pbar = Mock()
            mock_trange.return_value = mock_pbar
            mock_agg.return_value = 0.5

            # Run the function with wandb and checkpointing enabled
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                save_elite=True,
                wb=True,
                wandb_api_key="fake_key",
                evaluation_interval=3,
                accelerator=Accelerator() if use_accelerator else None,
                checkpoint_steps=6,
            )

            # Verify wandb was initialized
            mock_init_wandb.assert_called_once()
            # Verify wandb logging
            assert mock_wandb.log.call_count >= 5
            # Verify checkpointing
            assert mock_save.call_count == 1

            # Verify evaluation was called at the right intervals (steps 3)
            assert mock_agent.test.call_count == 2

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_preference_evolvable_training_loop(self, use_accelerator):
        """Test the basic training loop in finetune_llm."""
        # Create mock agent
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = (0.5, 0.2, 0.1)
        mock_agent.test.return_value = 0.87
        mock_agent.batch_size = 32
        mock_agent.batch_size_per_process = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        # Create mock environment - use MagicMock for special methods
        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        example_prefernce_env_return = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env.reset.return_value = example_prefernce_env_return
        mock_env.step.return_value = example_prefernce_env_return
        mock_env.data_batch_size_per_gpu = 1

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        # Mock other dependencies
        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.safe_aggregate_metrics") as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch(
                "agilerl.training.train_llm.tournament_selection_and_mutation"
            ) as mock_tournament_selection_and_mutation,
        ):
            mock_tournament_selection_and_mutation.return_value = [mock_agent]

            mock_agg.return_value = 0.5
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=Accelerator() if use_accelerator else None,
                tournament=Mock(),
                mutation=mutation,
            )
            assert mock_env.reset.call_count == 6
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            expected_agg_calls = 21
            assert mock_agg.call_count == expected_agg_calls
            if not use_accelerator:
                assert all(
                    call_args.args[0] is None for call_args in mock_agg.call_args_list
                )
            assert mock_agent.test.call_count == 3  # Should be called at step 2
            assert (
                mock_tournament_selection_and_mutation.call_count == 6
            )  # Should be called at step 2

    def test_finetune_llm_preference_warning_num_epochs_and_max_steps(self):
        """Test that finetune_llm raises a warning if evo_steps is not set."""
        # Create mock agent
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = (0.5, 0.2, 0.1)
        mock_agent.test.return_value = 0.87
        mock_agent.batch_size = 32
        mock_agent.batch_size_per_process = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        # Create mock environment - use MagicMock for special methods
        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        example_prefernce_env_return = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env.reset.return_value = example_prefernce_env_return
        mock_env.step.return_value = example_prefernce_env_return
        mock_env.data_batch_size_per_gpu = 1

        # Mock other dependencies
        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch(
                "agilerl.training.train_llm.tournament_selection_and_mutation"
            ) as mock_tournament_selection_and_mutation,
        ):
            mock_tournament_selection_and_mutation.return_value = [mock_agent]

            mock_agg.return_value = 0.5
            with pytest.warns(
                UserWarning,
                match=r"'num_epochs' will take precedence over 'max_steps'",
            ) as num_epochs_and_max_steps_warning:
                train_llm_dataset(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    num_epochs=10,
                    max_steps=100,
                    evo_steps=None,
                )
            assert (
                "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'."
                in str(num_epochs_and_max_steps_warning[0].message)
            )

    def test_finetune_llm_preference_break_on_num_epochs(self):
        # Create mock agent
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = (0.5, 0.2, 0.1)
        mock_agent.test.return_value = 0.87
        mock_agent.batch_size = 32
        mock_agent.batch_size_per_process = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        # Create mock environment - use MagicMock for special methods
        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        example_prefernce_env_return = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env.reset.return_value = example_prefernce_env_return
        mock_env.step.return_value = example_prefernce_env_return
        mock_env.data_batch_size_per_gpu = 1

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        # Mock other dependencies
        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.init_wandb"),
            patch("agilerl.training.train_llm.wandb"),
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_env.num_epochs = 2
            mock_agg.return_value = 0.5
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=None,
                num_epochs=2,
                checkpoint_steps=3,
            )

    def test_finetune_llm_preference_value_error_if_algo_not_dpo(self):
        # Create mock agent
        mock_agent = MagicMock(spec=GRPO)
        mock_agent.algo = "GRPO"
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]
        with pytest.raises(
            ValueError,
            match=r"The algorithm must be DPO .preference. or SFT .supervised.",
        ):
            train_llm_dataset(
                pop=[mock_agent],
                env=MagicMock(),
                evaluation_interval=2,
                accelerator=None,
            )

    def test_finetune_llm_preference_env_fn_uses_distinct_env_instances(self):
        agent_a = MagicMock(spec=DPO)
        agent_a.algo = "DPO"
        agent_a.fitness = [0.0]
        agent_a.learn.return_value = (0.5, 0.2, 0.1)
        agent_a.batch_size_per_process = 1
        agent_a.batch_size = 1
        agent_a.steps = [0]
        agent_a.scores = [0.0]
        agent_a.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

        agent_b = MagicMock(spec=DPO)
        agent_b.algo = "DPO"
        agent_b.fitness = [0.0]
        agent_b.learn.return_value = (0.5, 0.2, 0.1)
        agent_b.batch_size_per_process = 1
        agent_b.batch_size = 1
        agent_b.steps = [0]
        agent_b.scores = [0.0]
        agent_b.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"

        env_a = MagicMock()
        env_a.__len__.return_value = 1
        env_a.name = "env_a"
        env_a.data_batch_size_per_gpu = 1
        env_a.num_epochs = 0
        env_a.reset.return_value = {"prompt": ["a"]}
        env_a.step.return_value = {"prompt": ["a_next"]}

        env_b = MagicMock()
        env_b.__len__.return_value = 1
        env_b.name = "env_b"
        env_b.data_batch_size_per_gpu = 1
        env_b.num_epochs = 0
        env_b.reset.return_value = {"prompt": ["b"]}
        env_b.step.return_value = {"prompt": ["b_next"]}

        env_fn = MagicMock(side_effect=[env_a, env_b])

        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus",
                return_value=0.5,
            ),
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            train_llm_dataset(
                pop=[agent_a, agent_b],
                env_fn=env_fn,
                max_steps=2,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        assert env_fn.call_count == 2
        assert env_a.reset.call_count == 1
        assert env_b.reset.call_count == 1
        assert env_a.step.call_count == 0
        assert env_b.step.call_count == 0
        assert agent_a.learn.call_args.args[0] == {"prompt": ["a"]}
        assert agent_b.learn.call_args.args[0] == {"prompt": ["b"]}

    def test_finetune_llm_preference_learns_from_current_reset_batch(self):
        # Arrange
        agent = MagicMock(spec=DPO)
        agent.algo = "DPO"
        agent.fitness = [0.0]
        agent.batch_size_per_process = 1
        agent.batch_size = 1
        agent.steps = [0]
        agent.scores = [0.0]
        agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        agent.learn.return_value = {
            "mean_chosen_reward": 0.7,
            "mean_rejected_reward": 0.2,
            "mean_kl": 0.1,
        }

        env = MagicMock()
        env.__len__.return_value = 3
        env.name = "mock_env"
        env.data_batch_size_per_gpu = 1
        env.num_epochs = 0
        prompts = [{"prompt": ["p0"]}, {"prompt": ["p1"]}, {"prompt": ["p2"]}]
        epoch_trackers = [0, 0, 1]
        state = {"idx": 0}

        def reset_side_effect():
            idx = state["idx"]
            env.num_epochs = epoch_trackers[idx]
            state["idx"] += 1
            return prompts[idx]

        env.reset.side_effect = reset_side_effect

        # Act
        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus",
                side_effect=lambda _acc, value: value,
            ),
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            train_llm_dataset(
                pop=[agent],
                env=env,
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        # Assert
        assert env.reset.call_count == 3
        assert [
            call_args.args[0] for call_args in agent.learn.call_args_list
        ] == prompts
        assert [
            call_args.args[0] for call_args in agent.set_reference_policy.call_args_list
        ] == [0, 0, 1]

    def test_finetune_llm_preference_env_fn_preserves_agent_batch_isolation(self):
        # Arrange
        def make_agent() -> MagicMock:
            built_agent = MagicMock(spec=DPO)
            built_agent.algo = "DPO"
            built_agent.fitness = [0.0]
            built_agent.batch_size_per_process = 1
            built_agent.batch_size = 1
            built_agent.steps = [0]
            built_agent.scores = [0.0]
            built_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
            built_agent.learn.return_value = {
                "mean_chosen_reward": 0.7,
                "mean_rejected_reward": 0.2,
                "mean_kl": 0.1,
            }
            return built_agent

        agent_a = make_agent()
        agent_b = make_agent()

        env_a = MagicMock()
        env_a.__len__.return_value = 1
        env_a.name = "env_a"
        env_a.data_batch_size_per_gpu = 1
        env_a.num_epochs = 0
        env_a.reset.return_value = {"prompt": ["a_only"]}

        env_b = MagicMock()
        env_b.__len__.return_value = 1
        env_b.name = "env_b"
        env_b.data_batch_size_per_gpu = 1
        env_b.num_epochs = 0
        env_b.reset.return_value = {"prompt": ["b_only"]}

        env_fn = MagicMock(side_effect=[env_a, env_b])

        # Act
        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus",
                side_effect=lambda _acc, value: value,
            ),
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            train_llm_dataset(
                pop=[agent_a, agent_b],
                env_fn=env_fn,
                max_steps=2,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        # Assert
        assert env_a.reset.call_count == 1
        assert env_b.reset.call_count == 1
        assert agent_a.learn.call_args.args[0] == {"prompt": ["a_only"]}
        assert agent_b.learn.call_args.args[0] == {"prompt": ["b_only"]}

    def test_finetune_llm_preference_csv_logging_without_wandb(self, tmp_path, capsys):
        """DPO: csv_check only path; teardown closes CSV and prints path (train_llm.py ~858-860)."""
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.fitness = [0.0]
        mock_agent.get_action = MagicMock()
        mock_agent.learn.return_value = (0.5, 0.2, 0.1)
        mock_agent.test.return_value = 0.87
        mock_agent.batch_size = 32
        mock_agent.batch_size_per_process = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]
        mock_agent.pretrained_model_name_or_path = "x"

        example = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env = MagicMock()
        mock_env.__len__.return_value = 4
        mock_env.reset.return_value = example
        mock_env.step.return_value = example
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch("agilerl.training.train_llm.wandb") as mock_wandb,
        ):
            mock_agg.return_value = 0.5
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                accelerator=None,
                elite_path=str(tmp_path),
                wb=False,
                log_csv=True,
                verbose=False,
            )
        mock_wandb.log.assert_not_called()
        csv_path = tmp_path / "metrics.csv"
        assert csv_path.is_file()
        assert "Train/Best Reward Margin" in csv_path.read_text()
        out = capsys.readouterr().out
        assert "Training metrics saved to" in out
        assert "metrics.csv" in out

    def test_finetune_llm_preference_aggregate_skips_eval_when_never_evaluates(
        self, tmp_path, capsys
    ):
        """DPO: agg_test_metrics None skips eval keys in aggregate block."""
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.fitness = [0.0]
        mock_agent.get_action = MagicMock()
        mock_agent.learn.return_value = (0.5, 0.2, 0.1)
        mock_agent.test.return_value = 0.87
        mock_agent.batch_size = 32
        mock_agent.batch_size_per_process = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        example = {
            "prompt": ["This is a mock prompt"],
            "prompt_lengths": [10],
            "chosen": ["This is a mock chosen prompt"],
            "rejected": ["This is a mock rejected prompt"],
            "chosen_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "rejected_input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        mock_env = MagicMock()
        mock_env.__len__.return_value = 4
        mock_env.reset.return_value = example
        mock_env.step.return_value = example
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch("agilerl.training.train_llm.wandb"),
        ):
            mock_agg.return_value = 0.5
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=100,
                accelerator=None,
                elite_path=str(tmp_path),
                wb=False,
                log_csv=True,
                verbose=False,
            )
        mock_agent.test.assert_not_called()
        assert "Training metrics saved to" in capsys.readouterr().out


class TestFinetuneLlmSft:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_sft_basic_training_loop(self, use_accelerator):
        """Test the basic training loop in finetune_llm_sft."""
        mock_agent = MagicMock(spec=SFT)
        mock_agent.algo = "SFT"
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.65}
        mock_agent.test.return_value = -0.4
        mock_agent.batch_size_per_process = 1
        mock_agent.batch_size = 1
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.name = "mock_sft"
        mock_env.num_epochs = 0
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_safe_agg.side_effect = lambda acc, val: (
                float(val) if not isinstance(val, float) else val
            )
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                accelerator=None if use_accelerator else Accelerator(),
            )
            assert mock_env.reset.call_count == 6
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_sft_with_wandb_and_checkpoints(self, use_accelerator):
        """Test finetune_llm_sft with wandb logging and checkpointing enabled."""
        mock_agent = MagicMock(spec=SFT)
        mock_agent.algo = "SFT"
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 32}
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.65}
        mock_agent.test.return_value = -0.4
        mock_agent.batch_size_per_process = 1
        mock_agent.batch_size = 1
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.lr = 0.001

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.name = "mock_sft"
        mock_env.num_epochs = 0
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.train_llm.trange") as mock_trange,
            patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
            patch("agilerl.training.train_llm.wandb") as mock_wandb,
            patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
        ):
            mock_pbar = Mock()
            mock_trange.return_value = mock_pbar
            mock_safe_agg.side_effect = lambda acc, val: (
                float(val) if not isinstance(val, float) else val
            )

            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                save_elite=True,
                wb=True,
                wandb_api_key="fake_key",
                evaluation_interval=3,
                accelerator=None if use_accelerator else Accelerator(),
                checkpoint_steps=6,
            )

            mock_init_wandb.assert_called_once()
            assert mock_wandb.log.call_count >= 5
            assert mock_save.call_count == 1
            assert mock_agent.test.call_count == 2

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_sft_evolvable_training_loop(self, use_accelerator):
        """Test the evolvable training loop in finetune_llm_sft."""
        mock_agent = MagicMock(spec=SFT)
        mock_agent.algo = "SFT"
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.65}
        mock_agent.test.return_value = -0.4
        mock_agent.batch_size_per_process = 1
        mock_agent.batch_size = 1
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.name = "mock_sft"
        mock_env.num_epochs = 0
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch(
                "agilerl.training.train_llm.tournament_selection_and_mutation"
            ) as mock_tournament_selection_and_mutation,
        ):
            mock_tournament_selection_and_mutation.return_value = [mock_agent]
            mock_safe_agg.side_effect = lambda acc, val: (
                float(val) if not isinstance(val, float) else val
            )

            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=None if use_accelerator else Accelerator(),
                tournament=Mock(),
                mutation=mutation,
            )
            assert mock_env.reset.call_count == 6
            assert mock_env.step.call_count == 0
            assert mock_agent.learn.call_count == 6
            assert mock_agent.test.call_count == 3
            assert mock_tournament_selection_and_mutation.call_count == 6

    def test_finetune_llm_sft_warning_num_epochs_and_max_steps(self):
        """Test that finetune_llm_sft warns when both num_epochs and max_steps are set."""
        mock_agent = MagicMock(spec=SFT)
        mock_agent.algo = "SFT"
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.65}
        mock_agent.test.return_value = -0.4
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 6
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_safe_agg.side_effect = lambda acc, val: (
                float(val) if not isinstance(val, float) else val
            )
            with pytest.warns(
                UserWarning,
                match=r"'num_epochs' will take precedence over 'max_steps'",
            ) as num_epochs_and_max_steps_warning:
                train_llm_dataset(
                    pop=[mock_agent],
                    env=mock_env,
                    evaluation_interval=2,
                    num_epochs=10,
                    max_steps=100,
                    evo_steps=None,
                )
            assert "num_epochs" in str(num_epochs_and_max_steps_warning[0].message)
            assert "max_steps" in str(num_epochs_and_max_steps_warning[0].message)

    def test_finetune_llm_sft_break_on_num_epochs(self):
        """Test that finetune_llm_sft breaks when num_epochs is reached."""
        mock_agent = MagicMock(spec=SFT)
        mock_agent.algo = "SFT"
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.65}
        mock_agent.test.return_value = -0.4
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 3
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.init_wandb"),
            patch("agilerl.training.train_llm.wandb"),
            patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_env.num_epochs = 2
            mock_safe_agg.side_effect = lambda acc, val: (
                float(val) if not isinstance(val, float) else val
            )
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                evo_steps=1,
                accelerator=None,
                num_epochs=2,
                checkpoint_steps=3,
            )

    def test_finetune_llm_sft_value_error_if_algo_not_sft(self):
        """Test that finetune_llm_sft raises ValueError if agent is not SFT."""
        mock_agent = MagicMock(spec=GRPO)
        mock_agent.algo = "GRPO"
        mock_agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]
        with pytest.raises(
            ValueError,
            match=r"The algorithm must be DPO .preference. or SFT .supervised.",
        ):
            train_llm_dataset(
                pop=[mock_agent],
                env=MagicMock(),
                evaluation_interval=2,
                accelerator=None,
            )

    def test_finetune_llm_sft_evo_steps_not_set(self):
        """Test that finetune_llm_sft raises ValueError if evo_steps not set with tournament/mutation."""
        with pytest.raises(
            ValueError,
            match="'evo_steps' must be set if 'tournament' and 'mutation' are not None",
        ):
            train_llm_dataset(
                pop=[MagicMock(spec=SFT)],
                env=MagicMock(),
                evo_steps=None,
                accelerator=None,
                tournament=MagicMock(),
                mutation=MagicMock(),
            )

    def test_finetune_llm_sft_env_fn_updates_prompts_by_agent(self):
        """SFT env_fn path initializes and updates per-agent prompts."""
        agent0 = MagicMock(spec=SFT)
        agent1 = MagicMock(spec=SFT)
        for agent in (agent0, agent1):
            agent.algo = "SFT"
            agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.2}
            agent.test.return_value = -0.3
            agent.batch_size = 1
            agent.batch_size_per_process = 1
            agent.steps = [0]
            agent.scores = [0.0]
            agent.pretrained_model_name_or_path = "x"
            agent.fitness = [0.0]
            agent.registry = MagicMock()
            agent.registry.hp_config = MagicMock()
            agent.registry.hp_config.config = {}

        def _mk_env():
            env = MagicMock()
            env.__len__.return_value = 2
            env.reset.return_value = "initial_prompts"
            env.step.return_value = "next_prompts"
            env.data_batch_size_per_gpu = 1
            return env

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch(
                "agilerl.training.train_llm.safe_aggregate_metrics",
                side_effect=lambda _a, v: float(v),
            ),
        ):
            train_llm_dataset(
                pop=[agent0, agent1],
                env_fn=_mk_env,
                accelerator=None,
                max_steps=2,
                evaluation_interval=100,
                verbose=False,
            )
        assert agent0.learn.call_count >= 1
        assert agent1.learn.call_count >= 1

    def test_finetune_llm_sft_csv_logging_without_wandb(self, tmp_path, capsys):
        """SFT: csv_check only; teardown closes CSV and prints path (train_llm.py ~1094-1096)."""
        mock_agent = MagicMock(spec=SFT)
        mock_agent.algo = "SFT"
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {}
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.65}
        mock_agent.test.return_value = -0.4
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]
        mock_agent.pretrained_model_name_or_path = "x"

        mock_env = MagicMock()
        mock_env.__len__.return_value = 4
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch("agilerl.training.train_llm.wandb") as mock_wandb,
        ):
            mock_safe_agg.side_effect = lambda acc, val: (
                float(val) if not isinstance(val, float) else val
            )
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=2,
                accelerator=None,
                elite_path=str(tmp_path),
                wb=False,
                log_csv=True,
                verbose=False,
            )
        mock_wandb.log.assert_not_called()
        metrics_csv = tmp_path / "metrics.csv"
        assert metrics_csv.is_file()
        assert "Train/Best Loss" in metrics_csv.read_text(encoding="utf-8")
        out = capsys.readouterr().out
        assert "Training metrics saved to" in out
        assert "metrics.csv" in out

    def test_finetune_llm_sft_aggregate_skips_eval_fitness_when_never_evaluates(
        self, tmp_path
    ):
        """SFT: no eval skips Eval/Best Fitness keys in aggregate block."""
        mock_agent = MagicMock(spec=SFT)
        mock_agent.algo = "SFT"
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {}
        mock_agent.fitness = [0.0]
        mock_agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.65}
        mock_agent.test.return_value = -0.4
        mock_agent.batch_size_per_process = 32
        mock_agent.batch_size = 32
        mock_agent.steps = [10]
        mock_agent.scores = [0.0]

        mock_env = MagicMock()
        mock_env.__len__.return_value = 4
        mock_env.reset.return_value = "initial_prompts"
        mock_env.step.return_value = "next_prompts"
        mock_env.data_batch_size_per_gpu = 1

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.utils.utils.safe_aggregate_metrics") as mock_safe_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch("agilerl.training.train_llm.wandb"),
        ):
            mock_safe_agg.side_effect = lambda acc, val: (
                float(val) if not isinstance(val, float) else val
            )
            train_llm_dataset(
                pop=[mock_agent],
                env=mock_env,
                evaluation_interval=100,
                accelerator=None,
                elite_path=str(tmp_path),
                wb=False,
                log_csv=True,
                verbose=False,
            )
        mock_agent.test.assert_not_called()


class TestFinetuneLlmMultiturn:
    @pytest.mark.parametrize("agent_spec", [LLMPPO, LLMREINFORCE, GRPO])
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_multiturn_basic_training_loop(
        self, agent_spec, use_accelerator
    ):
        """Smoke: episode collection, learn with turn_ids, step accounting; no agent.test."""
        mock_agent = _make_multiturn_mock_agent(spec=agent_spec)
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
        max_turns = 2
        batch_size = 1
        batch_steps_per_iter = len(mock_env.turn_boundaries)
        max_steps = 9
        num_outer = max_steps // batch_steps_per_iter

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=max_turns,
                init_hp={"BATCH_SIZE": batch_size, "ALGO": mock_agent.algo},
                max_steps=max_steps,
                evaluation_interval=100,
                verbose=False,
                accelerator=None if use_accelerator else Accelerator(),
            )

        assert mock_env.reset.call_count == num_outer * batch_size
        assert mock_agent.get_action.call_count == num_outer * batch_size * max_turns
        assert mock_env.step.call_count == num_outer * batch_size * max_turns
        assert mock_env.get_episode_data.call_count == num_outer * batch_size
        assert mock_agent.learn.call_count == num_outer
        assert mock_agent.test.call_count == 0
        n_metrics = 4 if agent_spec is GRPO else 7
        assert mock_agg.call_count == num_outer * n_metrics
        # All LLM algos (GRPO included) now receive per-turn ids in the
        # multiturn loop, so learn() is always called with turn_ids.
        mock_agent.learn.assert_called_with(ANY, turn_ids=ANY)
        assert mock_save.call_count == 1

    def test_finetune_llm_multiturn_forwards_sampling_logps_to_learn(self):
        """When the rollout captures sampling logps, they're forwarded to
        ``learn(..., sampling_logps=...)`` for GRPO/PPO/REINFORCE agents.
        """
        mock_agent = _make_multiturn_mock_agent(spec=GRPO)
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
        sampling_logps = [torch.zeros(1, 7)]
        rollout_return = (
            [torch.ones(1, 8, dtype=torch.long)],  # completion_ids_list
            [torch.ones(1, 7, dtype=torch.bool)],  # action_masks_list
            [torch.zeros(1, 7, dtype=torch.long)],  # all_turn_ids
            [torch.ones(2, dtype=torch.float32)],  # all_rewards
            len(mock_env.turn_boundaries),  # batch_steps
            123,  # group_seed
            sampling_logps,  # all_sampling_logps (non-None)
        )
        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.collect_rollouts_llm",
                return_value=rollout_return,
            ),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus",
                return_value=0.5,
            ),
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": mock_agent.algo},
                max_steps=len(mock_env.turn_boundaries),  # one outer iteration
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        _, learn_kwargs = mock_agent.learn.call_args
        assert learn_kwargs.get("sampling_logps") is sampling_logps

    def test_finetune_llm_multiturn_allows_batch_size_indivisible_by_group_size(self):
        """The batch>group case is unconstrained too: batch_size=3, group_size=2
        (three prompts, two completions each) must pass startup validation rather
        than being rejected. Patch the rollout to a sentinel and assert the call
        reaches it — i.e. it gets past the (now-removed) divisibility guard.
        """
        mock_agent = _make_multiturn_mock_agent(spec=GRPO)
        mock_agent.group_size = 2
        mock_agent.batch_size = 3
        mock_agent.batch_size_per_process = 3

        sentinel = RuntimeError("reached rollout")
        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.collect_rollouts_llm",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="reached rollout"),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                max_turns=1,
                env_factory=_make_multiturn_env_factory(turn_boundaries_len=3),
                init_hp={"BATCH_SIZE": 3, "ALGO": "GRPO"},
                max_steps=100,
                accelerator=None,
                verbose=False,
            )

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_multiturn_with_wandb_and_checkpoints(self, use_accelerator):
        mock_agent = _make_multiturn_mock_agent()
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {"lr": 0.001, "batch_size": 16}
        mock_agent.lr = 0.01
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

        with (
            patch("agilerl.training.train_llm.trange") as mock_trange,
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
            patch("agilerl.training.train_llm.wandb") as mock_wandb,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
        ):
            mock_trange.return_value = Mock()
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5

            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=18,
                evaluation_interval=100,
                verbose=False,
                wb=True,
                wandb_api_key="fake_key",
                checkpoint_steps=2,
                accelerator=None if use_accelerator else Accelerator(),
            )

        mock_init_wandb.assert_called_once()
        assert mock_wandb.log.call_count >= 2
        assert mock_save.call_count >= 1

    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_finetune_llm_multiturn_evolvable_training_loop(self, use_accelerator):
        mock_agent = _make_multiturn_mock_agent()
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
            patch(
                "agilerl.training.train_llm.tournament_selection_and_mutation"
            ) as mock_tourn,
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            mock_tourn.return_value = [mock_agent]

            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                verbose=False,
                evo_steps=1,
                tournament=Mock(),
                mutation=mutation,
                accelerator=None if use_accelerator else Accelerator(),
            )

        assert mock_tourn.call_count == 3
        assert mock_save.call_count == 0

    def test_finetune_llm_multiturn_value_error_when_evo_steps_missing_with_tournament(
        self,
    ):
        mutation = MagicMock()
        mutation.architecture_mut = 0
        mutation.new_layer_prob = 0
        mutation.parameters_mut = 0
        mutation.activation_mut = 0
        mock_agent = _make_multiturn_mock_agent()
        with pytest.raises(ValueError, match="'evo_steps' must be set"):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock,
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=1,
                evo_steps=None,
                tournament=MagicMock(),
                mutation=mutation,
                accelerator=None,
            )

    def test_finetune_llm_multiturn_warns_when_evo_steps_without_tournament(self):
        mock_agent = _make_multiturn_mock_agent()
        with pytest.warns(UserWarning, match="evo_steps"):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock,
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=0,
                evo_steps=3,
                tournament=None,
                mutation=None,
                accelerator=None,
                verbose=False,
            )

    def test_finetune_llm_multiturn_value_error_if_algo_not_supported(self):
        mock_agent = MagicMock(spec=DPO)
        mock_agent.algo = "DPO"
        mock_agent.batch_size = 16
        mock_agent.batch_size_per_process = 16
        with pytest.raises(
            ValueError,
            match="The algorithm must be LLMPPO, LLMREINFORCE, or GRPO for multi-turn finetuning",
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=MagicMock,
                max_turns=1,
                init_hp={"BATCH_SIZE": 1, "ALGO": "DPO"},
                max_steps=0,
                accelerator=None,
                verbose=False,
            )

    def test_finetune_llm_multiturn_test_interval(self):
        """``finetune_llm_multiturn`` should call ``agent.test`` on a fresh env
        from ``env_factory`` every ``evaluation_interval`` outer iterations,
        matching the API of the other LLM trainers (no separate ``eval_fn``).
        """
        mock_agent = _make_multiturn_mock_agent()
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
        mock_agent.test.return_value = np.array(0.42, dtype=np.float32)

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=1,
                verbose=False,
                accelerator=None,
            )

        assert mock_agent.test.call_count == 3
        # Every call should hit the test env that env_factory produced.
        assert all(c.args[0] is mock_env for c in mock_agent.test.call_args_list)
        n_metrics = 7
        n_eval_agg = 3
        assert mock_agg.call_count == 3 * n_metrics + n_eval_agg

    def test_finetune_llm_multiturn_max_reward_adds_accuracy_metric(self):
        mock_agent = _make_multiturn_mock_agent()
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=9,
                evaluation_interval=100,
                max_reward=1.0,
                verbose=False,
                accelerator=None,
            )

        num_outer = 3
        assert mock_agg.call_count == num_outer * 8

    def test_finetune_llm_multiturn_init_hp_none_uses_agent_fields(self):
        """Covers init_hp branch that copies BATCH_SIZE_PER_GPU and ALGO from the agent."""
        mock_agent = _make_multiturn_mock_agent()
        mock_agent.batch_size_per_process = 7
        mock_agent.algo = "LLMPPO"
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch("agilerl.training.train_llm.init_wandb") as mock_init_wandb,
            patch("agilerl.training.train_llm.wandb"),
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp=None,
                max_steps=0,
                wb=True,
                wandb_api_key="fake",
                verbose=False,
                accelerator=None,
            )

        init_kw = mock_init_wandb.call_args.kwargs["init_hyperparams"]
        assert init_kw["BATCH_SIZE_PER_GPU"] == 7
        assert init_kw["ALGO"] == "LLMPPO"

    def test_finetune_llm_multiturn_max_model_len_rollout(self):
        """Covers the multiturn rollout loop when the agent has max_model_len set."""
        mock_agent = _make_multiturn_mock_agent()
        mock_agent.max_model_len = 1024
        mock_env = MagicMock()
        prompt: dict = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        L, T = 8, 2
        mock_env.reset.return_value = (prompt, {})
        mock_env.step.return_value = (prompt, 0.0, False, False, {})
        mock_env.turn_boundaries = [0, 1, 2]
        mock_env.dataset_size = 0
        mock_env.done = False
        mock_env.current_prompt = prompt
        mock_env.get_episode_data.return_value = (
            torch.ones(1, L, dtype=torch.long),
            torch.ones(1, L, dtype=torch.long),
            torch.zeros(1, L, dtype=torch.long),
            torch.ones(T, dtype=torch.float32),
            None,
        )
        _wire_phased_env_mock(mock_env)

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

    def test_finetune_llm_multiturn_breaks_turn_loop_when_terminated(self):
        """Covers early exit from the max_turns loop when env.step sets terminated."""
        mock_agent = _make_multiturn_mock_agent()
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
        prompt: dict = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        mock_env.reset.return_value = (prompt, {})

        def _terminating_step(*_args, **_kwargs):
            mock_env.done = True  # the real wrapper sets this on terminate
            return (prompt, 1.0, True, False, {})

        mock_env.step.side_effect = _terminating_step
        max_turns = 5

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=max_turns,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=3,
                evaluation_interval=100,
                verbose=False,
                accelerator=None,
            )

        assert mock_agent.get_action.call_count == 1

    def test_finetune_llm_multiturn_wandb_accuracy_and_eval_scores_with_verbose_banner(
        self,
    ):
        """W&B max_reward keys, Eval/Best score from agent.test, HPO keys, verbose pbar.write paths."""
        mock_pbar = Mock()
        mock_agent = _make_multiturn_mock_agent()
        mock_agent.registry = MagicMock()
        mock_agent.registry.hp_config = MagicMock()
        mock_agent.registry.hp_config.config = {"lr": 0.01}
        mock_agent.lr = 0.01
        mock_agent.test.return_value = np.array(0.33, dtype=np.float32)
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

        with (
            patch("agilerl.training.train_llm.trange", return_value=mock_pbar),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch("agilerl.training.train_llm.init_wandb"),
            patch("agilerl.training.train_llm.wandb") as mock_wandb,
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO", "env_name": "gem_test"},
                max_steps=9,
                evaluation_interval=1,
                max_reward=0.5,
                wb=True,
                wandb_api_key="fake",
                verbose=True,
                accelerator=None,
            )

        assert mock_pbar.write.call_count >= 2
        eval_logged = any(
            "Eval/Best Score" in c.args[0] for c in mock_wandb.log.call_args_list
        )
        assert eval_logged
        hpo_logged = any(
            "HPO_agent_0/lr" in c.args[0] for c in mock_wandb.log.call_args_list
        )
        assert hpo_logged
        acc_logged = any(
            "Train/Best Accuracy" in c.args[0] for c in mock_wandb.log.call_args_list
        )
        assert acc_logged

    def test_finetune_llm_multiturn_accelerator_syncs_after_test(self):
        """Covers accelerator.wait_for_everyone() after distributed eval aggregation
        that follows the ``agent.test`` call.
        """
        mock_agent = _make_multiturn_mock_agent()
        mock_agent.test.return_value = np.array(0.1, dtype=np.float32)
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
        acc = MagicMock(spec=Accelerator)
        acc.is_main_process = True
        acc.wait_for_everyone = MagicMock()

        with (
            patch("agilerl.training.train_llm.trange"),
            patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
            patch(
                "agilerl.training.train_llm.aggregate_metrics_across_gpus"
            ) as mock_agg,
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
            patch(
                "agilerl.training.train_llm._distributed_world_size",
                return_value=1,
            ),
            patch(
                "agilerl.training.train_llm._distributed_rank",
                return_value=0,
            ),
        ):
            mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
            mock_agg.return_value = 0.5
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=3,
                evaluation_interval=1,
                verbose=False,
                accelerator=acc,
            )

        assert acc.wait_for_everyone.call_count >= 1
        assert mock_agent.test.call_count >= 1

    def test_finetune_llm_multiturn_allows_group_size_indivisible_by_batch_size(
        self,
    ):
        """batch_size and group_size need not divide each other for GRPO.

        The rollout vec env keeps each prompt's group whole (group-contiguous),
        so e.g. batch_size=2, group_size=3 (two prompts, three completions each)
        is valid and must pass startup validation rather than being rejected.
        We patch the rollout to raise a sentinel and assert the call reaches it
        — i.e. it gets past the (now-removed) divisibility guard.
        """
        agent = _make_multiturn_mock_agent(spec=GRPO)
        agent.group_size = 3
        agent.batch_size = 2
        agent.batch_size_per_process = 2

        sentinel = RuntimeError("reached rollout")
        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.collect_rollouts_llm",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="reached rollout"),
        ):
            train_llm_rollout(
                pop=[agent],
                max_turns=2,
                env_factory=_make_multiturn_env_factory(),
                init_hp={"BATCH_SIZE": 2, "BATCH_SIZE_PER_GPU": 2, "ALGO": "GRPO"},
                max_steps=8,
                accelerator=None,
                wb=False,
                verbose=False,
            )

    def test_finetune_llm_multiturn_wall_deadline_stops_loop(self):
        """When ``max_wall_seconds`` is set and the deadline elapses, the outer
        loop must break immediately and emit the wall-time stop message — the
        agent's ``learn`` is never called.
        """
        import builtins

        mock_agent = _make_multiturn_mock_agent()
        mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)

        # ``time.monotonic`` is read twice for the deadline: once at start
        # (to set ``wall_deadline``) and once per iteration (to compare). Force
        # the second read to be far in the future so the loop bails right away.
        monotonic_values = iter([0.0, 1_000_000.0])
        captured_prints = []

        original_print = builtins.print

        def _capture_print(*args, **kwargs):
            captured_prints.append(" ".join(str(a) for a in args))
            return original_print(*args, **kwargs)

        with (
            patch("agilerl.training.train_llm.trange"),
            patch(
                "agilerl.training.train_llm.time.monotonic",
                side_effect=lambda: next(monotonic_values),
            ),
            patch("builtins.print", side_effect=_capture_print),
            patch("agilerl.training.train_llm.save_llm_checkpoint"),
        ):
            train_llm_rollout(
                pop=[mock_agent],
                env_factory=lambda: mock_env,
                max_turns=2,
                init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
                max_steps=100,
                verbose=False,
                accelerator=None,
                max_wall_seconds=5.0,
            )

        # The loop should bail before doing any work.
        assert mock_agent.learn.call_count == 0
        assert any("wall time limit (5.0s) reached" in line for line in captured_prints)


class TestBuildTrainWandbDict:
    def test_build_train_wandb_dict_reasoning_llmppo_uses_fallback_pg_and_entropy_keys(
        self,
    ):
        pop = _make_pop_for_wandb_dict(size=2)
        agent = MagicMock(spec=LLMPPO)
        agent_metrics_dict = {
            "agent_0/train_metrics": {
                "Train/Rewards": 1.0,
                "Train/Mean Loss": 0.5,
                "Train/Mean KL": 0.1,
                "Train/Completion Length": 8.0,
                "Train/PG Loss": 0.2,
                "Train/Entropy": 1.2,
                "Train/Mean VF Loss": 0.3,
                "Train/Mean Clipfrac": 0.15,
                "Train/Accuracy": 0.5,
            },
            "agent_1/train_metrics": {
                "Train/Rewards": 3.0,
                "Train/Mean Loss": 0.7,
                "Train/Mean KL": 0.3,
                "Train/Completion Length": 10.0,
                "Train/PG Loss": 0.4,
                "Train/Entropy": 1.0,
                "Train/Mean VF Loss": 0.5,
                "Train/Mean Clipfrac": 0.35,
                "Train/Accuracy": 0.75,
            },
        }

        out = build_train_wandb_dict(
            agent_metrics_dict=agent_metrics_dict,
            pop=pop,
            agent=agent,
            max_reward=4.0,
            mode="reasoning",
        )
        assert out["Train/Best Reward"] == 3.0
        assert out["Train/Mean Population Reward"] == pytest.approx(2.0)
        assert out["Train/Mean Population PG Loss"] == pytest.approx(0.3)
        assert out["Train/Mean Population Entropy"] == pytest.approx(1.1)
        assert out["Train/Mean Population Critic Loss"] == pytest.approx(0.4)
        assert out["Train/Mean Population Clipfrac"] == pytest.approx(0.25)
        assert out["Train/Best Accuracy"] == pytest.approx(0.75)
        assert "HPO_agent_0/lr" in out


class TestBuildEvalWandbDict:
    def test_build_eval_wandb_dict_preference_and_multiturn_score_modes(self):
        pop = _make_pop_for_wandb_dict(size=2)
        pref_metrics = {
            "agent_0/test_metrics": {"Eval/Mean Reward Margin": 0.2},
            "agent_1/test_metrics": {"Eval/Mean Reward Margin": 0.6},
        }
        pref_out = build_eval_wandb_dict(pref_metrics, pop=pop, mode="preference")
        assert pref_out["Eval/Best Reward Margin"] == pytest.approx(0.6)
        assert pref_out["Eval/Mean Population Reward Margin"] == pytest.approx(0.4)

        score_metrics = {
            "agent_0/test_metrics": {"Eval/Score": 0.9},
            "agent_1/test_metrics": {"Eval/Score": 0.7},
        }
        score_out = build_eval_wandb_dict(
            score_metrics, pop=pop, mode="multiturn", eval_score_mode=True
        )
        assert score_out["Eval/Best Score"] == pytest.approx(0.9)
        assert score_out["Eval/Mean Population Score"] == pytest.approx(0.8)

    def test_build_train_and_eval_wandb_dict_sft_mode(self):
        pop = _make_pop_for_wandb_dict(size=2)
        agent = MagicMock(spec=SFT)
        train_metrics = {
            "agent_0/train_metrics": {
                "Train/Loss": 0.5,
                "Train/Perplexity": 10.0,
            },
            "agent_1/train_metrics": {
                "Train/Loss": 0.3,
                "Train/Perplexity": 12.0,
            },
        }
        train_out = build_train_wandb_dict(
            agent_metrics_dict=train_metrics,
            pop=pop,
            agent=agent,
            mode="sft",
        )
        assert train_out["Train/Best Loss"] == pytest.approx(0.3)
        assert train_out["Train/Mean Population Loss"] == pytest.approx(0.4)
        assert train_out["Train/Mean Population Perplexity"] == pytest.approx(11.0)
        assert "HPO_agent_0/lr" in train_out

        eval_metrics = {
            "agent_0/test_metrics": {"Eval/Negative loss (fitness)": -0.5},
            "agent_1/test_metrics": {"Eval/Negative loss (fitness)": -0.3},
        }
        eval_out = build_eval_wandb_dict(eval_metrics, pop=pop, mode="sft")
        assert eval_out["Eval/Best Fitness"] == pytest.approx(-0.3)
        assert eval_out["Eval/Mean Population Fitness"] == pytest.approx(-0.4)

    def test_build_train_wandb_dict_preference_mode(self):
        pop = _make_pop_for_wandb_dict(size=2)
        agent = MagicMock(spec=DPO)
        train_metrics = {
            "agent_0/train_metrics": {
                "Train/Loss": 0.5,
                "Train/Mean Reward Margin": 0.2,
            },
            "agent_1/train_metrics": {
                "Train/Loss": 0.3,
                "Train/Mean Reward Margin": 0.4,
            },
        }
        train_out = build_train_wandb_dict(
            agent_metrics_dict=train_metrics, pop=pop, agent=agent, mode="preference"
        )
        assert train_out["Train/Mean Population Loss"] == pytest.approx(0.4)
        assert train_out["Train/Mean Population Reward Margin"] == pytest.approx(0.3)

    def test_build_eval_wandb_dict_reasoning_reward_and_accuracy(self):
        """Default reasoning mode aggregates Eval/Reward, plus Eval/Accuracy when
        ``max_reward`` is set, across the population.
        """
        pop = _make_pop_for_wandb_dict(size=2)
        metrics = {
            "agent_0/test_metrics": {"Eval/Reward": 1.0, "Eval/Accuracy": 0.8},
            "agent_1/test_metrics": {"Eval/Reward": 0.6, "Eval/Accuracy": 0.4},
        }
        out = build_eval_wandb_dict(metrics, pop=pop, mode="reasoning", max_reward=1.0)
        assert out["Eval/Best Reward"] == pytest.approx(1.0)
        assert out["Eval/Mean Population Reward"] == pytest.approx(0.8)
        assert out["Eval/Best Accuracy"] == pytest.approx(0.8)
        assert out["Eval/Mean Population Accuracy"] == pytest.approx(0.6)


class TestNormalizeLearnMetrics:
    def test_train_metric_format_and_learn_output_normalization_helpers(self):
        formatted = _format_prefixed_metrics(
            {"mean_kl": 0.2, "mean_pg_loss": 0.1}, "Train"
        )
        assert formatted["Train/Mean KL"] == 0.2
        assert formatted["Train/Mean PG Loss"] == 0.1

        agent = MagicMock(spec=LLMREINFORCE)
        metrics = _normalize_learn_metrics(
            agent, (1.0, 0.5, 0.2, 0.3), mode="multiturn"
        )
        assert metrics["mean_loss"] == 1.0
        assert metrics["mean_kl"] == 0.5
        assert metrics["pg_loss"] == 0.2
        assert metrics["entropy"] == 0.3

    def test_preference_normalizes_loss_key(self):
        agent = MagicMock(spec=DPO)
        # tuple form
        metrics = _normalize_learn_metrics(agent, (0.7, 0.2, 0.1), mode="preference")
        assert metrics["loss"] == 0.7
        assert metrics["mean_chosen_reward"] == 0.2
        assert metrics["mean_rejected_reward"] == 0.1
        # dict form: DPO.learn returns "mean_loss", translated to "loss"
        dict_metrics = _normalize_learn_metrics(
            agent,
            {"mean_loss": 0.5, "mean_chosen_reward": 0.1, "mean_rejected_reward": 0.0},
            mode="preference",
        )
        assert dict_metrics["loss"] == 0.5
        assert "mean_loss" not in dict_metrics

    def test_normalize_learn_metrics_error_paths_and_multiturn_len5(self):
        agent = MagicMock(spec=LLMPPO)

        with pytest.raises(
            TypeError, match="Expected learn\\(\\) to return dict or tuple"
        ):
            _normalize_learn_metrics(agent, 1.23, mode="reasoning")

        with pytest.raises(
            ValueError, match="Preference learn\\(\\) tuple output must have 3 values"
        ):
            _normalize_learn_metrics(agent, (1.0, 2.0), mode="preference")

        mt_metrics = _normalize_learn_metrics(
            agent,
            (1.0, 0.5, 0.2, 0.3, 0.1),
            mode="multiturn",
        )
        assert mt_metrics["mean_vf_loss"] == 0.3
        assert mt_metrics["mean_entropy"] == 0.1

        with pytest.raises(
            ValueError,
            match="Reasoning/multi-turn learn\\(\\) tuple output has an unsupported shape",
        ):
            _normalize_learn_metrics(agent, (1.0, 0.5, 0.2), mode="reasoning")

    def test_reasoning_two_tuple_normalizes_loss_and_kl(self):
        """A 2-value rollout learn() tuple maps to ``mean_loss`` / ``mean_kl``."""
        agent = MagicMock(spec=LLMREINFORCE)
        metrics = _normalize_learn_metrics(agent, (0.8, 0.1), mode="reasoning")
        assert metrics == {"mean_loss": 0.8, "mean_kl": 0.1}


class TestSaveEliteCheckpoint:
    def test_save_elite_checkpoint_picks_best_agent(self, tmp_path):
        from agilerl.training.train_llm import _save_elite_checkpoint

        with patch("agilerl.training.train_llm.save_llm_checkpoint") as save:
            worse = MagicMock()
            worse.fitness = [1.0]
            better = MagicMock()
            better.fitness = [3.0]
            elite_dir = str(tmp_path / "elite")
            _save_elite_checkpoint([worse, better], True, elite_dir, None)
        save.assert_called_once_with(better, elite_dir)

    def test_save_elite_checkpoint_waits_but_skips_non_main_process(self, tmp_path):
        from agilerl.training.train_llm import _save_elite_checkpoint

        acc = MagicMock()
        acc.is_main_process = False
        with patch("agilerl.training.train_llm.save_llm_checkpoint") as save:
            agent = MagicMock()
            agent.fitness = [1.0]
            _save_elite_checkpoint([agent], True, str(tmp_path / "elite"), acc)
        acc.wait_for_everyone.assert_called_once()
        save.assert_not_called()


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (train_llm_dataset, DPO),
        (train_llm_dataset, SFT),
    ],
)
def test_finetune_llm_env_and_env_fn_mutually_exclusive(finetune_fn, agent_spec):
    agent = MagicMock(spec=agent_spec)
    if agent_spec is GRPO:
        agent.algo = "GRPO"
    elif agent_spec is DPO:
        agent.algo = "DPO"
    else:
        agent.algo = "SFT"
    agent.batch_size_per_process = 1
    agent.batch_size = 1
    agent.steps = [0]
    agent.scores = [0.0]
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    agent.fitness = [0.0]

    env = MagicMock()
    env.__len__.return_value = 1
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0
    env.reset.return_value = "prompts"
    env.step.return_value = "prompts"

    with pytest.raises(ValueError, match="Provide exactly one of 'env' or 'env_fn'"):
        finetune_fn(
            pop=[agent],
            env=env,
            env_fn=lambda: env,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


@pytest.mark.parametrize("agent_spec", [DPO, SFT])
def test_finetune_llm_requires_env_or_env_fn(agent_spec):
    with pytest.raises(ValueError, match="Either 'env' or 'env_fn' must be provided"):
        train_llm_dataset(
            pop=[MagicMock(spec=agent_spec)],
            env=None,
            env_fn=None,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (train_llm_dataset, DPO),
        (train_llm_dataset, SFT),
    ],
)
def test_finetune_llm_warns_on_shared_env_with_population(finetune_fn, agent_spec):
    agents = []
    for algo_name in ("a0", "a1"):
        agent = MagicMock(spec=agent_spec)
        if agent_spec is GRPO:
            agent.algo = "GRPO"
        elif agent_spec is DPO:
            agent.algo = "DPO"
        else:
            agent.algo = "SFT"
        agent.batch_size_per_process = 1
        agent.batch_size = 1
        agent.steps = [0]
        agent.scores = [0.0]
        agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
        agent.fitness = [0.0]
        agent.index = algo_name
        if agent_spec is SFT:
            agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.0}
            agent.test.return_value = -0.1
        agents.append(agent)

    env = MagicMock()
    env.__len__.return_value = 1
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0
    env.reset.return_value = "prompts"
    env.step.return_value = "prompts"

    with pytest.warns(UserWarning, match="fairness bias"):
        finetune_fn(
            pop=agents,
            env=env,
            max_steps=0,
            verbose=False,
            accelerator=None,
        )


@pytest.mark.parametrize("finetune_fn", [train_llm_dataset])
def test_finetune_llm_checkpoint_triggering_non_divisible_steps(finetune_fn):
    agent = MagicMock(spec=DPO)
    agent.algo = "DPO"
    agent.learn.return_value = (0.5, 0.2, 0.1)
    env = MagicMock()
    env.reset.return_value = {"prompt": ["x"]}
    env.step.return_value = {"prompt": ["y"]}

    agent.fitness = [0.0]
    agent.batch_size_per_process = 1
    agent.batch_size = 1
    agent.steps = [0]
    agent.scores = [0.0]
    agent.pretrained_model_name_or_path = "Qwen/Qwen2.5-0.5B"
    env.__len__.return_value = 10
    env.name = "mock_env"
    env.data_batch_size_per_gpu = 1
    env.num_epochs = 0

    with (
        patch("agilerl.training.train_llm.trange"),
        patch(
            "agilerl.training.train_llm.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch("agilerl.training.train_llm.save_llm_checkpoint") as mock_save,
    ):
        finetune_fn(
            pop=[agent],
            env=env,
            max_steps=5,
            checkpoint_steps=2,
            evaluation_interval=100,
            verbose=False,
            accelerator=None,
        )

    assert mock_save.call_count == 3


def test_collect_rollouts_llm_breaks_when_vector_env_has_no_active_prompts():
    mock_agent = _make_multiturn_mock_agent()
    prompt = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    mock_env = MagicMock(spec=["reset", "step", "get_trajectories"])
    mock_env.reset.return_value = [prompt]
    mock_env.step.return_value = None
    mock_env.get_trajectories.return_value = (
        [torch.ones(1, 8, dtype=torch.long)],
        [torch.ones(1, 7, dtype=torch.bool)],
        [torch.zeros(1, 7, dtype=torch.long)],
        [torch.ones(2, dtype=torch.float32)],
        1,
        None,  # all_sampling_logps (added to get_trajectories' return)
    )

    _ = collect_rollouts_llm(
        agent=mock_agent,
        env=mock_env,
        n_steps=5,
        batch_size=1,
        group_seed=123,
        group_size=1,
    )

    assert mock_agent.get_action.call_count == 1
    assert mock_env.step.call_count == 1


def test_validate_evolution_args_warns_when_checkpoint_steps_ignored():
    from agilerl.training.train_llm import _validate_llm_evolution_args

    with pytest.warns(
        UserWarning,
        match="'checkpoint_steps' is set, but evolution is active",
    ):
        _validate_llm_evolution_args(
            evo_steps=2,
            tournament=MagicMock(),
            mutation=MagicMock(),
            checkpoint_steps=10,
        )


def test_init_llm_wandb_passes_entity_and_run_name():
    from agilerl.training.train_llm import _init_llm_wandb

    agent = MagicMock()
    agent.batch_size = 8
    agent.pretrained_model_name_or_path = "mock-model"
    pop = [agent]
    init_hp = {"ALGO": "GRPO"}

    with patch("agilerl.training.train_llm.init_wandb") as mock_init:
        _init_llm_wandb(
            init_hp=init_hp,
            pop=pop,
            env_name="mock-env",
            effective_data_batch_size=8,
            wb=True,
            wandb_api_key="fake-key",
            accelerator=None,
            wandb_entity="acme",
            wandb_run_name="run-1",
        )

    assert mock_init.call_args.kwargs["addl_args"] == {
        "entity": "acme",
        "name": "run-1",
    }


@pytest.mark.parametrize(
    ("finetune_fn", "agent_spec"),
    [
        (train_llm_dataset, DPO),
        (train_llm_dataset, SFT),
    ],
)
def test_inner_loop_breaks_after_max_steps_first_agent(finetune_fn, agent_spec):
    if agent_spec is DPO:
        agent0 = MagicMock(spec=DPO)
        agent1 = MagicMock(spec=DPO)
        for agent in (agent0, agent1):
            agent.algo = "DPO"
            agent.learn.return_value = (0.5, 0.2, 0.1)
            agent.test.return_value = 0.7
            agent.batch_size = 1
            agent.batch_size_per_process = 1
            agent.steps = [0]
            agent.scores = [0.0]
            agent.pretrained_model_name_or_path = "x"
            agent.fitness = [0.0]
        example = {
            "prompt": ["p"],
            "prompt_lengths": [1],
            "chosen": ["c"],
            "rejected": ["r"],
            "chosen_input_ids": [1],
            "chosen_attention_mask": [1],
            "rejected_input_ids": [1],
            "rejected_attention_mask": [1],
        }
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = example
        env.step.return_value = example
    else:
        agent0 = MagicMock(spec=SFT)
        agent1 = MagicMock(spec=SFT)
        for agent in (agent0, agent1):
            agent.algo = "SFT"
            agent.learn.return_value = {"mean_loss": 0.5, "mean_perplexity": 1.2}
            agent.test.return_value = -0.3
            agent.batch_size = 1
            agent.batch_size_per_process = 1
            agent.steps = [0]
            agent.scores = [0.0]
            agent.pretrained_model_name_or_path = "x"
            agent.fitness = [0.0]
            agent.registry = MagicMock()
            agent.registry.hp_config = MagicMock()
            agent.registry.hp_config.config = {}
        env = MagicMock()
        env.__len__.return_value = 2
        env.reset.return_value = "initial_prompts"
        env.step.return_value = "next_prompts"

    env.data_batch_size_per_gpu = 1

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
        patch(
            "agilerl.training.train_llm.aggregate_metrics_across_gpus", return_value=0.5
        ),
        patch(
            "agilerl.training.train_llm.safe_aggregate_metrics",
            side_effect=lambda _a, v: float(v),
        ),
    ):
        finetune_fn(
            pop=[agent0, agent1],
            env=env,
            accelerator=None,
            max_steps=1,
            evaluation_interval=100,
            verbose=False,
        )
    assert agent0.learn.call_count == 1
    assert agent1.learn.call_count == 0


def test_open_csv_log_and_log_row(tmp_path):
    from agilerl.training.train_llm import _log_csv_row, _open_csv_log

    csv_file, writer = _open_csv_log(str(tmp_path), ["step"], None)
    assert csv_file is not None
    assert writer is not None
    _log_csv_row(writer, csv_file, {"step": 1}, None)
    csv_file.close()

    non_main = MagicMock()
    non_main.is_main_process = False
    csv_file_none, writer_none = _open_csv_log(str(tmp_path), ["step"], non_main)
    assert csv_file_none is None
    assert writer_none is None

    writer_mock = MagicMock()
    file_mock = MagicMock()
    _log_csv_row(writer_mock, file_mock, {"step": 2}, non_main)
    writer_mock.writerow.assert_not_called()


def test_finetune_llm_reasoning_raises_migration_pointer():
    """The deprecated entrypoint raises with the migration instruction."""
    with pytest.raises(NotImplementedError, match="train_llm_rollout instead"):
        finetune_llm_reasoning()


def test_train_llm_rollout_closes_test_env_on_teardown():
    """A lazily-built eval env is closed when the run tears down."""
    mock_agent = _make_multiturn_mock_agent()
    mock_agent.test.return_value = np.array(0.42, dtype=np.float32)
    mock_env = _make_multiturn_mock_env(turn_boundaries_len=3)
    mock_env.close = MagicMock()  # spec omits close; add it so teardown forwards

    with (
        patch("agilerl.training.train_llm.trange"),
        patch("agilerl.training.train_llm.stack_and_pad_experiences") as mock_stack,
        patch("agilerl.training.train_llm.aggregate_metrics_across_gpus") as mock_agg,
        patch("agilerl.training.train_llm.save_llm_checkpoint"),
    ):
        mock_stack.return_value = (torch.zeros(1, 8, dtype=torch.long),)
        mock_agg.return_value = 0.5
        train_llm_rollout(
            pop=[mock_agent],
            env_factory=lambda: mock_env,
            max_turns=2,
            init_hp={"BATCH_SIZE": 1, "ALGO": "LLMPPO"},
            max_steps=3,
            evaluation_interval=1,  # build + eventually close a test env
            verbose=False,
            accelerator=None,
        )

    mock_env.close.assert_called()
