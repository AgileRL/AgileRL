import copy
from unittest.mock import MagicMock, Mock, call, patch

from typing import TYPE_CHECKING
import gymnasium as gym
import numpy as np
import pytest
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from gymnasium import spaces
from peft import LoraConfig
from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    IPPO,
    MADDPG,
    MATD3,
    NeuralTS,
    NeuralUCB,
    PPO,
    TD3,
    RainbowDQN,
)
from agilerl.algorithms.core import EvolvableAlgorithm, LLMAlgorithm

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from agilerl.algorithms import GRPO, LLMPPO, LLMREINFORCE
from agilerl.typing import BatchDimension
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    aggregate_metrics_across_gpus,
    calculate_vectorized_scores,
    consolidate_mutations,
    create_population,
    default_progress_bar,
    gather_tensor,
    get_env_defined_actions,
    make_multi_agent_vect_envs,
    make_skill_vect_envs,
    make_vect_envs,
    observation_space_channels_to_first,
    plot_population_score,
    print_hyperparams,
    save_llm_checkpoint,
    save_population_checkpoint,
    suppress_verbose_logging,
    tournament_selection_and_mutation,
    init_wandb,
)
from agilerl.utils.algo_utils import CosineLRScheduleConfig
from agilerl.wrappers.learning import Skill
from tests.test_algorithms.test_llms.test_grpo import (
    create_module as create_dummy_lm_for_reinforce,
)

# Shared HP dict that can be used by any algorithm
SHARED_INIT_HP = {
    "POPULATION_SIZE": 4,
    "DOUBLE": True,
    "BATCH_SIZE": 128,
    "CUDAGRAPHS": False,
    "LR": 1e-3,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "GAMMA": 0.99,
    "LEARN_STEP": 1,
    "TAU": 1e-3,
    "BETA": 0.4,
    "PRIOR_EPS": 0.000001,
    "NUM_ATOMS": 51,
    "V_MIN": 0,
    "V_MAX": 200,
    "N_STEP": 3,
    "POLICY_FREQ": 10,
    "GAE_LAMBDA": 0.95,
    "ACTION_STD_INIT": 0.6,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "UPDATE_EPOCHS": 4,
    "AGENT_IDS": ["agent1", "agent2"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "CHANNELS_LAST": False,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}


def test_make_vect_envs_requires_env_or_make_env():
    with pytest.raises(
        ValueError, match="Either env_name or make_env must be provided"
    ):
        make_vect_envs()


def test_make_vect_envs_with_make_env():
    def make_env():
        return gym.make("CartPole-v1")

    env = make_vect_envs(make_env=make_env, num_envs=2)
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == 2


def test_make_vect_envs_sync_vector():
    env = make_vect_envs("CartPole-v1", num_envs=2, should_async_vector=False)
    assert isinstance(env, gym.vector.SyncVectorEnv)
    assert env.num_envs == 2


def test_observation_space_channels_to_first_box():
    space = spaces.Box(0, 255, shape=(32, 32, 3), dtype="uint8")
    result = observation_space_channels_to_first(space)
    assert result.shape == (3, 32, 32)


def test_observation_space_channels_to_first_dict():
    space = spaces.Dict(
        {
            "img": spaces.Box(0, 255, shape=(16, 16, 4), dtype="uint8"),
            "vec": spaces.Box(0, 1, shape=(4,), dtype="float32"),
        }
    )
    result = observation_space_channels_to_first(space)
    assert result["img"].shape == (4, 16, 16)
    assert result["vec"].shape == (4,)


def test_observation_space_channels_to_first_tuple():
    space = spaces.Tuple(
        (
            spaces.Box(0, 255, shape=(8, 8, 2), dtype="uint8"),
            spaces.Discrete(5),
        )
    )
    result = observation_space_channels_to_first(space)
    assert result.spaces[0].shape == (2, 8, 8)
    assert isinstance(result.spaces[1], spaces.Discrete)


def test_observation_space_channels_to_first_discrete_passthrough():
    space = spaces.Discrete(5)
    result = observation_space_channels_to_first(space)
    assert result == space


def test_suppress_verbose_logging():
    suppress_verbose_logging()


def test_default_progress_bar_no_accelerator():
    bar = default_progress_bar(10, accelerator=None)
    assert bar.total == 10


def test_default_progress_bar_with_accelerator():
    acc = Accelerator()
    bar = default_progress_bar(10, accelerator=acc)
    assert bar.total == 10


def test_get_env_defined_actions_all_none():
    info = {"a": {"env_defined_action": None}, "b": {"env_defined_action": None}}
    assert get_env_defined_actions(info, ["a", "b"]) is None


def test_get_env_defined_actions_some_defined():
    info = {"a": {"env_defined_action": 1}, "b": {"env_defined_action": None}}
    result = get_env_defined_actions(info, ["a", "b"])
    assert result == {"a": 1, "b": None}


def test_batch_dimension_repr():
    assert repr(BatchDimension()) == "BatchDimension"


def test_save_population_checkpoint_no_accelerator(tmp_path):
    pop = [
        MagicMock(spec=EvolvableAlgorithm),
        MagicMock(spec=EvolvableAlgorithm),
    ]
    for i, agent in enumerate(pop):
        agent.steps = [100, 200]
        agent.save_checkpoint = MagicMock()
    save_path = str(tmp_path / "ckpt")
    save_population_checkpoint(pop, save_path, overwrite_checkpoints=True)
    assert pop[0].save_checkpoint.called
    assert pop[1].save_checkpoint.called


def test_save_population_checkpoint_with_accelerator(tmp_path):
    pop = [MagicMock(spec=EvolvableAlgorithm), MagicMock(spec=EvolvableAlgorithm)]
    for agent in pop:
        agent.steps = [100, 200]
        agent.save_checkpoint = MagicMock()
        agent.unwrap_models = MagicMock()
        agent.wrap_models = MagicMock()
    accel = MagicMock(spec=Accelerator)
    accel.wait_for_everyone = MagicMock()
    accel.is_main_process = True
    save_path = str(tmp_path / "ckpt")
    save_population_checkpoint(
        pop, save_path, overwrite_checkpoints=True, accelerator=accel
    )
    assert accel.wait_for_everyone.call_count >= 3
    for agent in pop:
        agent.unwrap_models.assert_called()
        agent.wrap_models.assert_called()
        agent.save_checkpoint.assert_called_once()


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_returns_asyncvectorenv_object():
    num_envs = 3
    env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == num_envs


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_returns_asyncvectorenv_object_multiagent():
    # ``speaker_listener_like_env`` mirrors the MPE speaker/listener API but
    # imports trivially, so workers spawn far faster than with the real
    # ``simple_speaker_listener_v4.parallel_env`` (which pulls PettingZoo
    # MPE + PyGame on each subprocess startup).
    from tests.pz_vector_test_utils import speaker_listener_like_env

    num_envs = 2
    env_kwargs = {"continuous_actions": False}
    env = make_multi_agent_vect_envs(
        speaker_listener_like_env, num_envs=num_envs, **env_kwargs
    )
    env.close()
    assert env.num_envs == num_envs


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_returns_asyncvectorenv_object_skill():
    num_envs = 3
    skill = Skill
    env = make_skill_vect_envs("CartPole-v1", skill=skill, num_envs=num_envs)
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == num_envs


# Can create a population of agent for each single agent algorithm
def test_create_initial_population_single_agent():
    observation_space = spaces.Box(0, 1, shape=(4,))
    continuous_action_space = spaces.Box(0, 1, shape=(2,))
    discrete_action_space = spaces.Discrete(2)
    net_config = {"encoder_config": {"hidden_size": [8, 8]}}
    population_size = 4
    device = "cpu"
    accelerator = None

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
    }

    for algo, algo_cls in algo_classes.items():
        if algo in ["TD3", "DDPG"]:
            action_space = continuous_action_space
        else:
            action_space = discrete_action_space

        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=population_size,
            device=device,
            accelerator=accelerator,
        )
        assert len(population) == population_size
        for agent in population:
            assert isinstance(agent, algo_cls)
            assert agent.observation_space == observation_space
            assert agent.action_space == action_space
            assert agent.device == "cpu"
            assert agent.accelerator is None


# Can create a population of agent for bandit algorithms
def test_create_initial_population_bandits():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(5)
    net_config = {"encoder_config": {"hidden_size": [8]}}
    init_hp = {**SHARED_INIT_HP, "LAMBDA": 1.0, "REG": 0.000625}

    for algo, algo_cls in [("NeuralUCB", NeuralUCB), ("NeuralTS", NeuralTS)]:
        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=init_hp,
            population_size=2,
            device="cpu",
        )
        assert len(population) == 2
        for agent in population:
            assert isinstance(agent, algo_cls)


# Can create a population of agent for each multi agent algorithm
def test_create_initial_population_multi_agent():
    observation_space = [spaces.Box(0, 1, shape=(4,)) for _ in range(2)]
    action_space = [spaces.Discrete(2) for _ in range(2)]
    net_config = {"encoder_config": {"hidden_size": [8], "min_mlp_nodes": 2}}
    population_size = 4
    device = "cpu"
    accelerator = None

    algo_classes = {
        "MADDPG": MADDPG,
        "MATD3": MATD3,
        "IPPO": IPPO,
    }

    for algo, algo_cls in algo_classes.items():
        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=population_size,
            device=device,
            accelerator=accelerator,
        )
        assert len(population) == population_size
        for agent in population:
            assert isinstance(agent, algo_cls)
            assert agent.observation_spaces == observation_space
            assert agent.action_spaces == action_space
            assert agent.device == "cpu"
            assert agent.accelerator is None


@pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES,
    reason="agilerl[llm] not installed",
)
@pytest.mark.parametrize(
    "algo,expected_type",
    [
        ("GRPO", GRPO),
        ("LLMPPO", LLMPPO),
        ("llmppo", LLMPPO),
        ("LLMREINFORCE", LLMREINFORCE),
    ],
)
def test_create_population_llm_policy_gradient_algorithms(
    vector_space, algo, expected_type
):
    init_hp = {
        "PAD_TOKEN_ID": 1000 - 1,
        "PAD_TOKEN": "<pad>",
        "BATCH_SIZE": 2,
        "BETA": 0.001,
        "LR": 0.001,
        "MAX_GRAD_NORM": 0.5,
        "UPDATE_EPOCHS": 1,
        "MAX_MODEL_LEN": 100,
        "GRADIENT_CHECKPOINTING": False,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    population_size = 1

    lora_kw = {
        "r": 16,
        "lora_alpha": 64,
        "target_modules": ["linear_1"],
        "task_type": "CAUSAL_LM",
        "lora_dropout": 0.05,
    }
    actor = create_dummy_lm_for_reinforce(
        input_size=10,
        max_tokens=20,
        vocab_size=1000,
        device=device,
    )
    common_kw = dict(
        algo=algo,
        observation_space=vector_space,
        action_space=copy.deepcopy(vector_space),
        net_config=None,
        INIT_HP=init_hp,
        hp_config=None,
        population_size=population_size,
        device=device,
        accelerator=None,
        actor_network=actor,
        algo_kwargs={
            "lora_config": LoraConfig(**lora_kw),
            "pad_token_id": 1000 - 1,
            "pad_token": "<pad>",
            "use_vllm": False,
        },
    )

    if expected_type is LLMPPO:
        mock_agent = MagicMock(spec=LLMPPO)
        with patch("agilerl.utils.utils.LLMPPO", return_value=mock_agent) as mock_cls:
            population = create_population(**common_kw)
        assert len(population) == population_size
        assert population[0] is mock_agent
        mock_cls.assert_called_once()
        call_kw = mock_cls.call_args.kwargs
        assert call_kw["batch_size"] == init_hp["BATCH_SIZE"]
        assert call_kw["beta"] == init_hp["BETA"]
        assert call_kw["vf_coef"] == SHARED_INIT_HP["VF_COEF"]
        assert call_kw["lr_actor"] == init_hp["LR"]
    else:
        population = create_population(**common_kw)
        assert len(population) == population_size
        for agent in population:
            assert isinstance(agent, expected_type)
            assert agent.accelerator is None
            assert agent.batch_size == init_hp["BATCH_SIZE"]


@pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES,
    reason="agilerl[llm] not installed",
)
def test_create_population_llmppo_uses_clone_and_generation_defaults(vector_space):
    init_hp = {
        "BATCH_SIZE": 2,
        "LR": 7e-5,
        "BETA": 0.01,
        "MAX_GRAD_NORM": 0.5,
        "UPDATE_EPOCHS": 1,
        "MAX_MODEL_LEN": 96,
        "MAX_OUTPUT_TOKENS": 12,
        "USE_VLLM": True,
        "GRADIENT_CHECKPOINTING": False,
        "COSINE_lR_SCHEDULER": {"num_epochs": 10, "warmup_proportion": 0.1},
    }
    actor = MagicMock(name="actor_network")
    actor.state_dict.return_value = {"w": torch.tensor([1.0])}
    cloned_actor = MagicMock(name="cloned_actor")
    vllm_cfg = object()
    a0 = MagicMock(name="ppo_agent_0")
    a1 = MagicMock(name="ppo_agent_1")

    with (
        patch("agilerl.utils.utils.clone_llm", return_value=cloned_actor) as mock_clone,
        patch("agilerl.utils.utils.LLMPPO", side_effect=[a0, a1]) as mock_llmppo,
    ):
        population = create_population(
            algo="LLMPPO",
            observation_space=vector_space,
            action_space=copy.deepcopy(vector_space),
            net_config=None,
            INIT_HP=init_hp,
            hp_config=None,
            population_size=2,
            device="cpu",
            accelerator=None,
            actor_network=actor,
            vllm_config=vllm_cfg,
            algo_kwargs={"pad_token_id": 999, "pad_token": "<pad>"},
        )

    assert population == [a0, a1]
    mock_clone.assert_called_once()
    first_kw = mock_llmppo.call_args_list[0].kwargs
    second_kw = mock_llmppo.call_args_list[1].kwargs
    assert first_kw["actor_network"] is actor
    assert second_kw["actor_network"] is cloned_actor
    assert first_kw["use_vllm"] is True
    assert first_kw["vllm_config"] is vllm_cfg
    assert first_kw["lr_actor"] == init_hp["LR"]
    assert first_kw["cosine_lr_schedule_config"] is not None
    assert isinstance(first_kw["cosine_lr_schedule_config"], CosineLRScheduleConfig)


@pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES,
    reason="agilerl[llm] not installed",
)
def test_create_population_llmreinforce_normalized_name_and_kwargs_overrides(
    vector_space,
):
    init_hp = {
        "BATCH_SIZE": 3,
        "LR": 5e-6,
        "BETA": 0.02,
        "MAX_GRAD_NORM": 0.7,
        "UPDATE_EPOCHS": 2,
        "MAX_MODEL_LEN": 80,
        "USE_VLLM": False,
        "GRADIENT_CHECKPOINTING": False,
        "COSINE_lR_SCHEDULER": {"num_epochs": 8, "warmup_proportion": 0.2},
    }
    actor = MagicMock(name="actor_network")
    actor.state_dict.return_value = {"w": torch.tensor([2.0])}
    cloned_actor = MagicMock(name="cloned_actor")
    pop0 = MagicMock(name="reinforce_agent_0")
    pop1 = MagicMock(name="reinforce_agent_1")
    global_vllm_cfg = object()
    local_vllm_cfg = object()

    with (
        patch("agilerl.utils.utils.clone_llm", return_value=cloned_actor) as mock_clone,
        patch(
            "agilerl.utils.utils.LLMREINFORCE",
            side_effect=[pop0, pop1],
        ) as mock_reinforce,
    ):
        population = create_population(
            algo="llmreinforce",
            observation_space=vector_space,
            action_space=copy.deepcopy(vector_space),
            net_config=None,
            INIT_HP=init_hp,
            hp_config=None,
            population_size=2,
            device="cpu",
            accelerator=None,
            actor_network=actor,
            vllm_config=global_vllm_cfg,
            torch_compiler="inductor",
            algo_kwargs={
                "pad_token_id": 999,
                "pad_token": "<pad>",
                "use_vllm": True,
                "vllm_config": local_vllm_cfg,
            },
        )

    assert population == [pop0, pop1]
    mock_clone.assert_called_once()
    first_kw = mock_reinforce.call_args_list[0].kwargs
    second_kw = mock_reinforce.call_args_list[1].kwargs
    assert first_kw["actor_network"] is actor
    assert second_kw["actor_network"] is cloned_actor
    assert first_kw["use_vllm"] is True
    assert first_kw["vllm_config"] is local_vllm_cfg
    assert first_kw["torch_compiler"] == "inductor"
    assert first_kw["lr"] == init_hp["LR"]
    assert isinstance(first_kw["cosine_lr_schedule_config"], CosineLRScheduleConfig)


@pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES,
    reason="agilerl[llm] not installed",
)
def test_create_population_llmppo_uses_unique_per_agent_accelerators(vector_space):
    init_hp = {
        "BATCH_SIZE": 2,
        "LR": 7e-5,
        "BETA": 0.01,
        "MAX_GRAD_NORM": 0.5,
        "UPDATE_EPOCHS": 1,
        "MAX_MODEL_LEN": 96,
        "MAX_OUTPUT_TOKENS": 12,
        "USE_VLLM": False,
        "GRADIENT_CHECKPOINTING": False,
    }
    actor = MagicMock(name="actor_network")
    actor.state_dict.return_value = {"w": torch.tensor([1.0])}
    cloned_actor = MagicMock(name="cloned_actor")
    a0 = MagicMock(name="ppo_agent_0")
    a1 = MagicMock(name="ppo_agent_1")
    base_accelerator = MagicMock(name="base_accelerator")
    acc0 = MagicMock(name="agent_accel_0")
    acc1 = MagicMock(name="agent_accel_1")

    with (
        patch("agilerl.utils.utils.clone_llm", return_value=cloned_actor),
        patch(
            "agilerl.utils.utils.get_state_dict",
            return_value={"w": torch.tensor([1.0])},
        ),
        patch(
            "agilerl.utils.utils.get_llm_accelerator", side_effect=[acc0, acc1]
        ) as mock_get_accel,
        patch("agilerl.utils.utils.LLMPPO", side_effect=[a0, a1]) as mock_llmppo,
    ):
        population = create_population(
            algo="LLMPPO",
            observation_space=vector_space,
            action_space=copy.deepcopy(vector_space),
            net_config=None,
            INIT_HP=init_hp,
            hp_config=None,
            population_size=2,
            device="cpu",
            accelerator=base_accelerator,
            actor_network=actor,
            algo_kwargs={"pad_token_id": 999, "pad_token": "<pad>", "use_vllm": False},
        )

    assert population == [a0, a1]
    assert mock_get_accel.call_args_list == [
        call(base_accelerator, 0),
        call(base_accelerator, 1),
    ]
    first_kw = mock_llmppo.call_args_list[0].kwargs
    second_kw = mock_llmppo.call_args_list[1].kwargs
    assert first_kw["accelerator"] is acc0
    assert second_kw["accelerator"] is acc1


# The function returns a list of episode rewards from the first episode in each parallel environment.
def test_returns_list_of_episode_rewards():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
    expected_rewards = [6, 9]

    result = calculate_vectorized_scores(
        rewards,
        terminations,
        include_unterminated=False,
        only_first_episode=True,
    )

    assert result == expected_rewards


# The function returns a list of episode rewards including all episodes.
def test_returns_list_of_episode_rewards_including_unterminated():
    rewards = np.array([[1, 2, 3], [4, 5, 6]])
    terminations = np.array([[0, 0, 1], [0, 1, 0]])
    expected_rewards = [6, 9, 6]

    result = calculate_vectorized_scores(
        rewards,
        terminations,
        include_unterminated=True,
        only_first_episode=False,
    )

    assert result == expected_rewards


# The function returns a list of episode rewards including all terminated episodes.
def test_returns_list_of_episode_rewards_all_terminated_episodes():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
    expected_rewards = [6, 9, 9]

    result = calculate_vectorized_scores(
        rewards,
        terminations,
        include_unterminated=False,
        only_first_episode=False,
    )

    assert result == expected_rewards


# The function returns a list of episode rewards including all terminated episodes.
def test_returns_list_of_episode_rewards_including_all_terminated_episodes():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
    expected_rewards = [6, 9, 9]

    result = calculate_vectorized_scores(
        rewards,
        terminations,
        include_unterminated=False,
        only_first_episode=False,
    )

    assert result == expected_rewards


# The function returns a list of episode rewards containing no terminated episodes.
def test_returns_list_of_episode_rewards_with_no_terminations():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    expected_rewards = [15, 30]

    result = calculate_vectorized_scores(
        rewards,
        terminations,
        include_unterminated=True,
        only_first_episode=False,
    )

    assert result == expected_rewards


def test_print_hyperparams_empty_fitness():
    pop = create_population(
        algo="DQN",
        observation_space=spaces.Box(0, 1, shape=(4,)),
        action_space=spaces.Discrete(2),
        net_config={"encoder_config": {"hidden_size": [8]}},
        INIT_HP=SHARED_INIT_HP,
        population_size=1,
    )
    pop[0].fitness = []
    with patch("builtins.print") as mock_print:
        print_hyperparams(pop)
        call_args = mock_print.call_args[0][0]
        assert "nan" in call_args.lower() or "nan" in str(call_args)


# The function prints the hyperparameters and fitnesses of all agents in the population.
def test_prints_hyperparams():
    # Arrange
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    net_config = {"encoder_config": {"hidden_size": [8]}}
    population_size = 1
    device = "cpu"
    accelerator = None
    algo = "DQN"

    pop = create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config,
        INIT_HP=SHARED_INIT_HP,
        population_size=population_size,
        device=device,
        accelerator=accelerator,
    )

    # Manually set attributes
    pop[0].fitness = [1, 2, 3]
    pop[0].lr = 0.01
    pop[0].batch_size = 32

    agent = pop[0]
    mean_fitness = np.mean(agent.fitness[-5:]).item()
    attrs = EvolvableAlgorithm.inspect_attributes(agent)
    expected_lines = [
        f"Agent ID: {agent.index}  |  Mean 5 Fitness: {mean_fitness:.2f}",
        "Attributes:",
        *[f"  {k}: {v}" for k, v in sorted(attrs.items())],
    ]
    expected_output = "\n".join(expected_lines) + "\n"

    with patch("builtins.print") as mock_print:
        print_hyperparams(pop)
        mock_print.assert_called_once_with(expected_output)


# The function should correctly plot the fitness scores of all agents in the population.
@patch("agilerl.utils.utils.plt")
def test_plot_fitness_scores_all_agents(mock_plt):
    # Create a population of agents with fitness scores
    class Agent:
        def __init__(self, fitness):
            self.fitness = fitness
            self.steps = list(range(len(fitness) + 1))

    pop = [Agent([1, 2, 3]), Agent([4, 5, 6]), Agent([7, 8, 9])]

    # Call the function under test
    plot_population_score(pop)

    # Assert plotting functions have been called with expected args
    mock_plt.title.assert_called_once_with("Score History - Mutations")
    mock_plt.xlabel.assert_called_once_with("Steps")

    # Assert plt.figure got called
    assert mock_plt.figure.called


def test_save_with_accelerator(tmp_path):
    """Test saving checkpoint when agent has an accelerator."""
    agent = Mock()
    agent.actor = Mock()
    agent.accelerator = Mock()
    agent.accelerator.wait_for_everyone = Mock()
    agent.algo = "grpo"
    save_llm_checkpoint(agent, str(tmp_path))
    agent.save_checkpoint.assert_called_once_with(str(tmp_path))
    agent.accelerator.wait_for_everyone.assert_called()


def test_save_without_accelerator(tmp_path):
    """Test saving checkpoint when agent has no accelerator."""
    agent = Mock()
    agent.actor = Mock()
    agent.algo = "grpo"
    agent.accelerator = None
    save_llm_checkpoint(agent, str(tmp_path))
    agent.save_checkpoint.assert_called_once_with(str(tmp_path))


def test_init_wandb_addl_args():
    with patch("agilerl.utils.utils.wandb") as mock_wandb:
        mock_wandb.api = MagicMock()
        init_wandb(
            algo="DQN",
            env_name="CartPole-v1",
            addl_args={"tags": ["test"]},
        )
        mock_wandb.init.assert_called_once()
        assert mock_wandb.init.call_args[1].get("tags") == ["test"]


def test_init_wandb_no_api_warns():
    class FakeWandb:
        def init(self, **kwargs):
            pass

    wandb_no_api = FakeWandb()
    with patch("agilerl.utils.utils.wandb", wandb_no_api):
        with pytest.warns(UserWarning, match="API key"):
            init_wandb(algo="DQN", env_name="CartPole-v1")


def test_init_wandb_with_accelerator_main_process():
    with patch("agilerl.utils.utils.wandb") as mock_wandb:
        mock_wandb.api = MagicMock()
        mock_accel = MagicMock(spec=Accelerator)
        mock_accel.is_main_process = True
        mock_accel.wait_for_everyone = Mock()
        init_wandb(algo="DQN", env_name="CartPole-v1", accelerator=mock_accel)
        mock_accel.wait_for_everyone.assert_called()
        mock_wandb.init.assert_called_once()


def test_init_wandb_with_api_key():
    class NoApiWandb:
        login = Mock()
        init = Mock()

    with patch("agilerl.utils.utils.wandb", NoApiWandb):
        init_wandb(algo="DQN", env_name="CartPole-v1", wandb_api_key="test-key")
        NoApiWandb.login.assert_called_once_with(key="test-key")


def test_init_wandb_mutation_hyperparams():
    with patch("agilerl.utils.utils.wandb") as mock_wandb:
        mock_wandb.api = MagicMock()
        init_wandb(
            algo="DQN",
            env_name="CartPole-v1",
            mutation_hyperparams={"MUT_1": 0.5},
        )
        assert mock_wandb.init.call_args[1]["config"].get("MUT_1") == 0.5


def test_tournament_selection_and_mutation_no_accelerator():
    population = [MagicMock(spec=EvolvableAlgorithm) for _ in range(3)]
    for agent in population:
        agent.steps = [100]
    tournament = MagicMock(spec=TournamentSelection)
    tournament.select = Mock(return_value=(population[0], population))
    mutation = MagicMock(spec=Mutations)
    mutation.mutation = Mock(return_value=population)
    result = tournament_selection_and_mutation(
        population, tournament, mutation, "CartPole-v1", algo="DQN"
    )
    tournament.select.assert_called_once()
    mutation.mutation.assert_called_once()
    assert len(result) == 3


def test_tournament_selection_and_mutation_worker_loads_checkpoint():
    """Worker process loads checkpoints saved by main process."""
    population = [MagicMock(spec=EvolvableAlgorithm) for _ in range(2)]
    for agent in population:
        agent.steps = [100]
        agent.load_checkpoint = Mock()
        agent.unwrap_models = Mock()
        agent.wrap_models = Mock()
    tournament = MagicMock(spec=TournamentSelection)
    tournament.select = Mock(return_value=(population[0], population))
    mutation = MagicMock(spec=Mutations)
    mutation.mutation = Mock(return_value=population)
    accel = MagicMock(spec=Accelerator)
    accel.wait_for_everyone = Mock()
    accel.is_main_process = False

    with patch("agilerl.utils.utils.Path") as mock_path:
        mock_path.return_value.mkdir = Mock()
        tournament_selection_and_mutation(
            population,
            tournament,
            mutation,
            "CartPole-v1",
            algo="DQN",
            accelerator=accel,
        )
    for agent in population:
        agent.load_checkpoint.assert_called()


def test_tournament_selection_and_mutation_save_elite_with_path():
    population = [MagicMock(spec=EvolvableAlgorithm) for _ in range(2)]
    elite = population[0]
    elite.steps = [100]
    elite.save_checkpoint = Mock()
    tournament = MagicMock(spec=TournamentSelection)
    tournament.select = Mock(return_value=(elite, population))
    mutation = MagicMock(spec=Mutations)
    mutation.mutation = Mock(return_value=population)
    tournament_selection_and_mutation(
        population,
        tournament,
        mutation,
        "CartPole-v1",
        algo="DQN",
        elite_path="/tmp/elite",
        save_elite=True,
    )
    elite.save_checkpoint.assert_called_once_with("/tmp/elite.pt")


def test_save_llm_checkpoint_with_path(tmp_path):
    agent = Mock()
    agent.actor = Mock()
    agent.algo = "grpo"
    agent.accelerator = None
    path = str(tmp_path / "my_ckpt")
    save_llm_checkpoint(agent, path)
    agent.save_checkpoint.assert_called_once_with(path)


def test_gather_tensor_with_tensor_input():
    """Test gather_tensor with tensor input"""
    accelerator = Accelerator()

    input_tensor = torch.tensor([1, 2, 3], device=accelerator.device)

    gathered = gather_tensor(input_tensor, accelerator)

    assert isinstance(gathered, torch.Tensor)

    assert torch.equal(gathered, input_tensor)


def test_gather_tensor_with_non_tensor_input():
    """Test gather_tensor with non-tensor input"""
    input_list = [1, 2, 3]

    accelerator = Accelerator()

    gathered = gather_tensor(input_list, accelerator)

    assert isinstance(gathered, torch.Tensor)

    assert torch.equal(gathered, torch.tensor(input_list).to(accelerator.device))


def test_gather_tensor_device():
    """Test that tensor is moved to accelerator device"""
    input_tensor = torch.tensor([1, 2, 3])

    accelerator = Accelerator()

    gathered = gather_tensor(input_tensor, accelerator)

    assert gathered.device.type == accelerator.device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gather_tensor_distributed():
    """Test gather_tensor in distributed setting"""
    accelerator = Accelerator()

    rank = accelerator.process_index
    input_tensor = torch.tensor([rank], device=accelerator.device)

    gathered = gather_tensor(input_tensor, accelerator)

    assert len(gathered) == accelerator.num_processes
    assert torch.equal(
        gathered,
        torch.arange(accelerator.num_processes, device=accelerator.device),
    )


def test_aggregate_metrics_across_gpus_single_process():
    """Test aggregate_metrics_across_gpus with single process"""
    accelerator = Accelerator()

    metric_tensor = torch.tensor([1.0, 2.0, 3.0], device=accelerator.device)

    result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

    assert result == 2.0  # (1 + 2 + 3) / 3 = 2.0
    assert isinstance(result, float)


def test_aggregate_metrics_across_gpus_with_scalar():
    """Test aggregate_metrics_across_gpus with scalar input"""
    accelerator = Accelerator()

    metric_tensor = torch.tensor(5.0, device=accelerator.device)

    result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

    assert result == 5.0
    assert isinstance(result, float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_aggregate_metrics_across_gpus_distributed():
    """Test aggregate_metrics_across_gpus in distributed setting"""
    accelerator = Accelerator()

    rank = accelerator.process_index
    metric_tensor = torch.tensor([rank + 1.0], device=accelerator.device)

    result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

    expected_mean = (
        sum(range(1, accelerator.num_processes + 1)) / accelerator.num_processes
    )
    assert (
        abs(result - expected_mean) < 1e-6
    )  # Allow for small floating point differences


def test_aggregate_metrics_across_gpus_with_negative_values():
    """Test aggregate_metrics_across_gpus with negative values"""
    accelerator = Accelerator()

    metric_tensor = torch.tensor([-1.0, -2.0, -3.0], device=accelerator.device)

    result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

    assert result == -2.0  # (-1 + -2 + -3) / 3 = -2.0
    assert isinstance(result, float)


def test_aggregate_metrics_across_gpus_with_zero_values():
    """Test aggregate_metrics_across_gpus with zero values"""
    accelerator = Accelerator()

    metric_tensor = torch.tensor([0.0, 0.0, 0.0], device=accelerator.device)

    result = aggregate_metrics_across_gpus(accelerator, metric_tensor)

    assert result == 0.0
    assert isinstance(result, float)


def test_consolidate_mutations_warning_if_not_llm_algorithm():
    """Test consolidate_mutations"""
    population = [Mock() for _ in range(3)]
    with pytest.warns(UserWarning):
        consolidate_mutations(population)


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
def test_consolidate_mutations():
    population = [MagicMock(spec=GRPO) for _ in range(3)]
    for agent in population:
        agent.mut = "lr"
        agent.lr = 0.01
        agent.lr_critic = None
        agent.optimizer = Mock()
        agent.optimizer.param_groups = [{"lr": 0.01}]
        agent.cosine_lr_schedule_config = {"warmup_steps": 0, "total_steps": 100}
        agent.accelerator = MagicMock(spec=Accelerator)
        agent.accelerator.is_main_process = True
        agent.accelerator.wait_for_everyone = Mock()
        agent.accelerator.state = MagicMock()
        agent.accelerator.state.deepspeed_plugin = MagicMock(spec=DeepSpeedPlugin)
        agent.accelerator.state.deepspeed_plugin.deepspeed_config = {}
        agent.actor = MagicMock()
    consolidate_mutations(population)
    for agent in population:
        assert agent.mut == "lr"
        assert agent.lr == 0.01
        assert agent.optimizer.param_groups[0]["lr"] == 0.01


def test_tournament_selection_and_mutation_language_model():
    """Test tournament_selection_and_mutation with language model"""
    population = [MagicMock(spec=LLMAlgorithm) for _ in range(3)]
    for agent in population:
        agent.mut = "lr"
        agent.lr = 0.01
        agent.optimizer = Mock()
        agent.optimizer.param_groups = [{"lr": 0.01}]
        agent.accelerator = MagicMock(spec=Accelerator)
        agent.actor = MagicMock()
        agent.actor.save_checkpoint = Mock()
    tournament = MagicMock(spec=TournamentSelection)
    mutation = MagicMock(spec=Mutations)
    mutation.mutation = Mock(return_value=population)
    tournament.select = Mock(return_value=(population[0], population))
    env_name = "CartPole-v1"
    algo = None
    elite_path = None
    save_elite = True
    language_model = True
    accelerator = MagicMock(spec=Accelerator)
    accelerator.is_main_process = True
    accelerator.wait_for_everyone = Mock()

    with (
        patch("agilerl.utils.utils.save_llm_checkpoint") as mock_save_llm_checkpoint,
        patch(
            "agilerl.utils.utils.consolidate_mutations"
        ) as mock_consolidate_mutations,
    ):
        output_pop = tournament_selection_and_mutation(
            population,
            tournament,
            mutation,
            env_name,
            algo,
            elite_path,
            save_elite,
            accelerator,
            language_model,
        )
        mock_save_llm_checkpoint.assert_called_once_with(population[0], elite_path)
        mock_consolidate_mutations.assert_called_once_with(output_pop)

    tournament.select.assert_called_once_with(population)
    mutation.mutation.assert_called_once_with(population)
    accelerator.wait_for_everyone.assert_called()


def test_check_box2d_available_raises_when_box2d_missing(monkeypatch):
    """Covers the ImportError path when Box2D is required but not installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "Box2D":
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from agilerl.utils.utils import _check_box2d_available

    with pytest.raises(ImportError, match="Box2D physics engine"):
        _check_box2d_available("LunarLander-v2")


@pytest.mark.llm
def test_create_population_sft_cpu():
    """Exercise ``create_population`` SFT branch (clone after first agent)."""
    pytest.importorskip("peft")
    from peft import LoraConfig

    from agilerl.algorithms.sft import SFT
    from tests.test_algorithms.test_llms.test_grpo import create_module

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["linear_1"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    actor = create_module(5, 10, 30, "cpu")
    pop = create_population(
        algo="SFT",
        net_config=None,
        INIT_HP=SHARED_INIT_HP,
        population_size=2,
        actor_network=actor,
        algo_kwargs={
            "pad_token_id": 29,
            "pad_token": "<pad>",
            "lora_config": lora_config,
        },
    )
    assert len(pop) == 2
    assert all(isinstance(agent, SFT) for agent in pop)


@pytest.mark.llm
def test_create_population_dpo_cpu():
    """Exercise ``create_population`` DPO branch (clone after first agent)."""
    pytest.importorskip("peft")
    from peft import LoraConfig

    from agilerl.algorithms.dpo import DPO
    from tests.test_algorithms.test_llms.test_grpo import create_module

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["linear_1"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    actor = create_module(5, 10, 30, "cpu")
    pop = create_population(
        algo="DPO",
        net_config=None,
        INIT_HP=SHARED_INIT_HP,
        population_size=2,
        actor_network=actor,
        algo_kwargs={
            "pad_token_id": 29,
            "pad_token": "<pad>",
            "lora_config": lora_config,
        },
    )
    assert len(pop) == 2
    assert all(isinstance(agent, DPO) for agent in pop)
