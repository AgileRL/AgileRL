import copy
import os
from unittest.mock import MagicMock, Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from gymnasium import spaces
from peft import LoraConfig
from pettingzoo.mpe import simple_speaker_listener_v4

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    GRPO,
    IPPO,
    LLMPPO,
    LLMReinforce,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    RainbowDQN,
)
from agilerl.algorithms.core import EvolvableAlgorithm, LLMAlgorithm
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    aggregate_metrics_across_gpus,
    calculate_vectorized_scores,
    consolidate_mutations,
    create_population,
    gather_tensor,
    make_multi_agent_vect_envs,
    make_skill_vect_envs,
    make_vect_envs,
    plot_population_score,
    print_hyperparams,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
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


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_returns_asyncvectorenv_object():
    num_envs = 3
    env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == num_envs


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_returns_asyncvectorenv_object_multiagent():
    num_envs = 3
    env = simple_speaker_listener_v4.parallel_env
    env_kwargs = {"continuous_actions": False}
    env = make_multi_agent_vect_envs(env, num_envs=num_envs, **env_kwargs)
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
        ("LLMReinforce", LLMReinforce),
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
            "agilerl.utils.utils.LLMReinforce",
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


def test_save_with_accelerator():
    """Test saving checkpoint when agent has an accelerator."""
    agent = Mock()
    agent.actor = Mock()
    agent.accelerator = Mock()
    agent.accelerator.wait_for_everyone = Mock()
    agent.algo = "grpo"
    save_llm_checkpoint(agent, None)
    agent.save_checkpoint.assert_called_once_with(
        "./saved_checkpoints/grpo",
        weights_only=False,
    )
    agent.accelerator.wait_for_everyone.assert_called()
    os.rmdir("saved_checkpoints/grpo")


def test_save_without_accelerator():
    """Test saving checkpoint when agent has no accelerator."""
    agent = Mock()
    agent.actor = Mock()
    agent.algo = "grpo"
    agent.accelerator = None
    save_llm_checkpoint(agent, None)
    agent.save_checkpoint.assert_called_once_with(
        "./saved_checkpoints/grpo",
        weights_only=False,
    )
    os.rmdir("saved_checkpoints/grpo")


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
