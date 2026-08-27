# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import warnings
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
from gymnasium import spaces

from agilerl import HAS_DEEPSPEED, HAS_LLM_DEPENDENCIES, HAS_VLLM
from agilerl.algorithms import DDPG, DQN, IPPO, PPO, TD3, NeuralUCB
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

if HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig
from agilerl.hpo.mutation import (
    MutationError,
    Mutations,
    get_exp_layer,
    set_global_seed,
)
from agilerl.modules import EvolvableBERT, EvolvableModule, ModuleDict
from agilerl.utils import mutation_utils
from agilerl.utils.utils import create_population
from agilerl.wrappers.agent import AgentWrapper, AsyncAgentsWrapper, RSNorm
from tests.helper_functions import (
    assert_state_dicts_equal,
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
    grama_scores_for,
)

if HAS_DEEPSPEED and HAS_VLLM:
    from tests.test_algorithms.test_llms.test_grpo import create_module
else:
    create_module = None

if TYPE_CHECKING:
    from agilerl.algorithms.core import EvolvableAlgorithm

# Shared HP dict that can be used by any algorithm
SHARED_INIT_HP = {
    "POPULATION_SIZE": 2,
    "DOUBLE": True,
    "BATCH_SIZE": 32,
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
    "AGENT_IDS": ["agent_0", "agent_1", "other_agent_0"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}

SHARED_INIT_HP_MA = SHARED_INIT_HP.copy()


def create_bert_network(device):
    return EvolvableBERT([12], [12], device=device)


def create_bert_networks_multi_agent(device):
    return ModuleDict(
        {
            "agent_0": create_bert_network(device),
            "agent_1": create_bert_network(device),
            "other_agent_0": create_bert_network(device),
        },
    )


@pytest.fixture
def bert_network(device):
    return create_bert_network(device)


@pytest.fixture
def bert_networks_multi_agent(device):
    return create_bert_networks_multi_agent(device)


@pytest.fixture
def bert_matd3_critic_networks(device):
    return [
        create_bert_networks_multi_agent(device),
        create_bert_networks_multi_agent(device),
    ]


@pytest.fixture
def init_pop(
    algo,
    observation_space,
    action_space,
    net_config,
    INIT_HP,
    population_size,
    device,
    accelerator_flag,
    hp_config,
    torch_compiler,
    request,
    actor_network=None,
    critic_network=None,
):
    accelerator = Accelerator(device_placement=False) if accelerator_flag else None
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    if hp_config is not None:
        hp_config = request.getfixturevalue(hp_config)

    if actor_network is not None:
        actor_network = request.getfixturevalue(actor_network)
    if critic_network is not None:
        critic_network = request.getfixturevalue(critic_network)

    pop = create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        hp_config=hp_config,
        net_config=request.getfixturevalue(net_config),
        INIT_HP=INIT_HP,
        population_size=population_size,
        device=device,
        accelerator=accelerator,
        actor_network=actor_network,
        critic_network=critic_network,
        torch_compiler=torch_compiler,
    )
    yield pop
    gc.collect()


class TestMutationsInit:
    # The constructor initializes all the attributes of the Mutations class correctly.
    def test_constructor_initializes_attributes(self):
        no_mutation = 0.1
        architecture = 0.2
        new_layer_prob = 0.3
        parameters = 0.4
        activation = 0.5
        rl_hp = 0.6
        mutation_sd = 0.7
        activation_selection = ["ReLU", "Sigmoid"]
        mutate_elite = True
        rand_seed = 12345
        device = "cpu"
        accelerator = None

        mutations = Mutations(
            no_mutation,
            architecture,
            new_layer_prob,
            parameters,
            activation,
            rl_hp,
            mutation_sd=mutation_sd,
            activation_selection=activation_selection,
            mutate_elite=mutate_elite,
            rand_seed=rand_seed,
            device=device,
            accelerator=accelerator,
        )

        assert mutations.rng is not None
        assert mutations.no_mut == no_mutation
        assert mutations.architecture_mut == architecture
        assert mutations.new_layer_prob == new_layer_prob
        assert mutations.parameters_mut == parameters
        assert mutations.activation_mut == activation
        assert mutations.rl_hp_mut == rl_hp
        assert mutations.mutation_sd == mutation_sd
        assert mutations.activation_selection == activation_selection
        assert mutations.mutate_elite == mutate_elite
        assert mutations.device == device
        assert mutations.accelerator == accelerator

    def test_raises_for_negative_no_mutation(self):
        with pytest.raises(AssertionError, match="greater than or equal to zero"):
            Mutations(-0.1, 0, 0.5, 0, 0, 0, 0.1, device="cpu")

    def test_raises_for_invalid_new_layer_prob(self):
        with pytest.raises(AssertionError, match="between zero and one"):
            Mutations(0, 0, 1.5, 0, 0, 0, 0.1, device="cpu")


class TestMutationsFindAnalogousMutation:
    def test_returns_none_for_empty_sampled(self):
        mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device="cpu")
        assert (
            mutations._find_analogous_mutation(
                "", ["agent_0.vector_mlp.add_node"], "agent_0"
            )
            is None
        )

    def test_returns_exact_match(self):
        mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device="cpu")
        available = ["agent_0.vector_mlp.add_node", "agent_1.vector_mlp.add_node"]
        assert (
            mutations._find_analogous_mutation(
                "agent_0.vector_mlp.add_node", available, "agent_0"
            )
            == "agent_0.vector_mlp.add_node"
        )

    def test_returns_analogous_by_policy_agent(self):
        mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device="cpu")
        # Sampled from agent_0, look for same bottom-level method with policy_agent in path.
        # Implementation returns the first match; put agent_1 first so we get the one with policy_agent.
        available = ["agent_1.vector_mlp.add_node", "agent_0.vector_mlp.add_node"]
        result = mutations._find_analogous_mutation(
            "agent_0.encoder.add_node", available, "agent_1"
        )
        assert result == "agent_1.vector_mlp.add_node"

    def test_returns_analogous_by_vector_mlp(self):
        mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device="cpu")
        available = ["agent_0.vector_mlp.add_node"]
        result = mutations._find_analogous_mutation(
            "other_module.add_node", available, "agent_0"
        )
        assert result == "agent_0.vector_mlp.add_node"

    def test_returns_none_when_no_analogous(self):
        mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device="cpu")
        available = ["agent_0.vector_mlp.add_channel"]  # different bottom-level method
        assert (
            mutations._find_analogous_mutation(
                "agent_0.encoder.add_node", available, "agent_0"
            )
            is None
        )

    def test_returns_none_when_bottom_matches_but_no_agent_or_mlp(self):
        mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device="cpu")
        # Method has same bottom-level but neither policy_agent nor 'vector_mlp' in parts
        available = ["other.add_node"]
        assert (
            mutations._find_analogous_mutation(
                "agent_0.encoder.add_node", available, "agent_0"
            )
            is None
        )


class TestMutationsGaussianParameterMutation:
    @pytest.mark.gpu
    def test_skips_zero_sized_weights(self, device):
        class ZeroWeightModule(EvolvableModule):
            def __init__(self):
                super().__init__(device="cpu")
                self.w = torch.nn.Parameter(torch.empty(0, 2))

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

        muts = Mutations(0, 0, 0.5, 1, 0, 0, 0.1, device=device)
        mod = ZeroWeightModule()
        out = muts._gaussian_parameter_mutation(mod)
        assert out is mod


class TestMutationsArchitectureMutateSingle:
    @pytest.mark.gpu
    def test_no_methods_sets_none(self, monkeypatch, device):
        class DummyPolicy:
            mutation_methods: ClassVar[list[str]] = []

        class DummyIndividual:
            def __init__(self):
                self.mut = None

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        individual = DummyIndividual()
        monkeypatch.setattr(
            DummyIndividual,
            "get_eval_modules",
            lambda self, cloning=True: ({"actor": DummyPolicy()}, {}),
            raising=False,
        )
        with pytest.warns(
            UserWarning, match="No mutation methods found for the policy network"
        ):
            out = muts._architecture_mutate_single(individual)
        assert out.mut == "None"


class TestMutationsArchitectureMutateMulti:
    @pytest.mark.gpu
    def test_no_methods_sets_none(self, monkeypatch, device):
        class DummyPolicy:
            mutation_methods: ClassVar[list[str]] = []

        class DummyIndividual:
            def __init__(self):
                self.mut = None

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        individual = DummyIndividual()
        monkeypatch.setattr(
            DummyIndividual,
            "get_eval_modules",
            lambda self, cloning=True: ({"actors": DummyPolicy()}, {}),
            raising=False,
        )
        with pytest.warns(
            UserWarning, match="No mutation methods found for the policy network"
        ):
            out = muts._architecture_mutate_multi(individual)
        assert out.mut == "None"

    @pytest.mark.gpu
    def test_none_applied_mutation_branch(self, monkeypatch, device):
        class DummySubmodule(EvolvableModule):
            def __init__(self):
                super().__init__(device="cpu")

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

        class DummyPolicyDict(ModuleDict):
            """ModuleDict container with fixed mutation method names for the test."""

            def __init__(
                self,
                modules: dict[str, EvolvableModule],
                device: str = "cpu",
            ) -> None:
                # Empty during ModuleDict construction so add_module hasattr
                # checks do not resolve dotted mutation names prematurely.
                object.__setattr__(self, "_fixed_mutation_methods", [])
                super().__init__(modules, device=device)
                self._fixed_mutation_methods = [
                    "agent_0.add_node",
                    "agent_1.add_node",
                ]

            def sample_mutation_method(self, *_args, **_kwargs):
                return "agent_0.add_node"

            @property
            def mutation_methods(self) -> list[str]:
                return list(self._fixed_mutation_methods)

            def get_mutation_methods(self):
                return {}

        class DummyIndividual:
            def mutation_hook(self):
                return None

            def reinit_optimizers(self):
                return None

        policy = DummyPolicyDict(
            {"agent_0": DummySubmodule(), "agent_1": DummySubmodule()},
            device="cpu",
        )
        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        monkeypatch.setattr(
            DummyIndividual,
            "get_eval_modules",
            lambda self, cloning=True: ({"actors": policy}, {}),
            raising=False,
        )
        monkeypatch.setattr(
            muts,
            "_apply_arch_mutation",
            lambda *_args, **_kwargs: (None, {}),
        )
        monkeypatch.setattr(
            muts, "_to_device_and_set_individual", lambda *_args, **_kwargs: None
        )
        individual = DummyIndividual()
        out = muts._architecture_mutate_multi(individual)
        assert out.mut == "None"

    @pytest.mark.gpu
    def test_raises_when_no_analogous(self, monkeypatch, device):
        class DummyEval(EvolvableModule):
            def __init__(self):
                super().__init__(device="cpu")
                self.last_mutation_attr = None

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

        class DummyPolicy(ModuleDict):
            """ModuleDict policy double with a fixed sampled mutation method."""

            def __init__(
                self,
                modules: dict[str, EvolvableModule],
                device: str = "cpu",
            ) -> None:
                object.__setattr__(self, "_fixed_mutation_methods", [])
                super().__init__(modules, device=device)
                self._fixed_mutation_methods = ["agent_0.add_node"]

            def sample_mutation_method(self, *_args, **_kwargs):
                return "agent_0.add_node"

            @property
            def mutation_methods(self) -> list[str]:
                return list(self._fixed_mutation_methods)

            def get_mutation_methods(self):
                return {}

        class DummyIndividual:
            def mutation_hook(self):
                return None

            def reinit_optimizers(self):
                return None

        policy = DummyPolicy({"agent_0": DummyEval()}, device="cpu")
        evals = {"critics": ModuleDict({"agent_0": DummyEval()}, device="cpu")}
        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        monkeypatch.setattr(
            DummyIndividual,
            "get_eval_modules",
            lambda self, cloning=True: ({"actors": policy}, evals),
            raising=False,
        )
        monkeypatch.setattr(
            muts,
            "_apply_arch_mutation",
            lambda *_args, **_kwargs: ("agent_0.add_node", {}),
        )
        monkeypatch.setattr(
            muts, "_to_device_and_set_individual", lambda *_args, **_kwargs: None
        )
        with pytest.raises(MutationError, match="No analogous method found"):
            muts._architecture_mutate_multi(DummyIndividual())


class TestMutationsApplyArchMutation:
    @pytest.mark.gpu
    def test_error_and_none_paths(self, device):
        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)

        with pytest.raises(MutationError, match="inherits from 'EvolvableModule'"):
            muts._apply_arch_mutation(torch.nn.Linear(2, 2), "x")

        class DummyNet(EvolvableModule):
            def __init__(self):
                super().__init__(device="cpu")
                self._layer_mutation_methods = ["mut"]
                self._node_mutation_methods = []
                self.last_mutation_attr = "mut"
                self.last_mutation = lambda: None

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

            def mut(self):
                self.last_mutation_attr = "mut"
                return {"k": 1}

        net = DummyNet()
        applied, m = muts._apply_arch_mutation(net, None)
        assert applied is None
        assert m == {}
        with pytest.raises(MutationError, match="not found"):
            muts._apply_arch_mutation(net, "missing_mut")


class TestMutationsReinitBanditGrads:
    def test_grown_parameters_extend_the_matrix(self):
        """A parameter that grows contributes its extra indices to ``to_add``."""

        class DummyActor(EvolvableModule):
            def __init__(self, out_mod):
                super().__init__(device="cpu")
                self.out_mod = out_mod

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

            def get_output_dense(self):
                return self.out_mod

        class DummyBandit:
            def __init__(self):
                self.sigma_inv = torch.eye(3)  # old weight 2 + bias 1
                self.lamb = 2.0
                self.device = "cpu"
                self.accelerator = None

        old_layer = torch.nn.Linear(2, 1)  # weight 2 + bias 1 = 3
        new_layer = torch.nn.Linear(2, 2)  # weight 4 + bias 2 = 6

        bandit = DummyBandit()
        Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device="cpu")._reinit_bandit_grads(
            bandit, DummyActor(new_layer), old_layer
        )
        assert bandit.sigma_inv.shape[0] == 6

    @pytest.mark.gpu
    def test_error_and_matrix_resize_paths(self, device):
        class DummyActor(EvolvableModule):
            def __init__(self, out_mod):
                super().__init__(device="cpu")
                self.out_mod = out_mod

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

            def get_output_dense(self):
                return self.out_mod

        # get_exp_layer requires nn.Linear output layers; subclass Linear and
        # add asymmetric parameters to exercise shrink/grow/add/remove paths.
        class OldLayer(torch.nn.Linear):
            def __init__(self):
                super().__init__(2, 2)  # weight 4 + bias 2
                self.only_old = torch.nn.Parameter(torch.ones(1))  # +1 → 7

        class NewLayer(torch.nn.Linear):
            def __init__(self):
                super().__init__(2, 1)  # weight 2 + bias 1
                self.only_new = torch.nn.Parameter(torch.ones(5))  # +5 → 8

        class DummyBandit:
            def __init__(self):
                self.sigma_inv = torch.eye(7)
                self.lamb = 2.0
                self.device = "cpu"
                self.accelerator = None

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        with pytest.raises(ValueError, match="not supported"):
            muts._reinit_bandit_grads(DummyBandit(), torch.nn.Linear(2, 2), OldLayer())

        bandit = DummyBandit()
        muts._reinit_bandit_grads(bandit, DummyActor(NewLayer()), OldLayer())
        assert bandit.sigma_inv.shape[0] == 8
        assert bandit.exp_layer is not None

    def test_raises_when_output_layer_is_none(self, device):
        """_reinit_bandit_grads raises ValueError when the offspring actor has no"""

        class NoOutputActor(EvolvableModule):
            def __init__(self):
                super().__init__(device="cpu")

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

            def get_output_dense(self):
                return None

        class DummyBandit:
            def __init__(self):
                self.sigma_inv = torch.eye(2)
                self.lamb = 2.0
                self.device = "cpu"
                self.accelerator = None

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        with pytest.raises(TypeError, match="expected a linear output layer"):
            muts._reinit_bandit_grads(
                DummyBandit(), NoOutputActor(), torch.nn.Linear(2, 2)
            )


class TestMutationsParameterMutation:
    def test_raises_when_no_policy_group(self, device):
        """parameter_mutation raises MutationError when the individual has no"""

        class NoPolicyRegistry:
            def __init__(self):
                self.groups = []

            def policy(self, return_group=False):
                return None

        class NoPolicyIndividual:
            def __init__(self):
                self.registry = NoPolicyRegistry()
                self.grama_scores = None
                self.accelerator = None

            def unrolled_eval_networks(self):
                return []

            def eval_policy_network_ids(self):
                return set()

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        with pytest.raises(MutationError, match="No policy network group registered"):
            muts.parameter_mutation(NoPolicyIndividual())


class TestMutationsMutation:
    # Checks no mutations if all probabilities set to zero
    @pytest.mark.gpu
    @pytest.mark.parametrize("algo", ["DQN"])
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["discrete_space"])
    @pytest.mark.parametrize("accelerator_flag", [False])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("hp_config", ["default_hp_config"])
    def test_no_options(self, init_pop, device):
        pre_training_mut = True
        population = init_pop
        mutations = Mutations(0, 0, 0, 0, 0, 0, 0.1, device=device)

        new_population = [agent.clone() for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert_state_dicts_equal(
                old.actor.state_dict(), individual.actor.state_dict()
            )

    #### Single-agent algorithm mutations ####
    # The mutation method applies random mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "hp_config", "action_space"),
        [
            ("DQN", "default_hp_config", "discrete_space"),
            ("Rainbow DQN", "default_hp_config", "discrete_space"),
            ("DDPG", "ac_hp_config", "vector_space"),
            ("TD3", "ac_hp_config", "vector_space"),
            ("PPO", "default_hp_config", "discrete_space"),
            ("CQN", "default_hp_config", "discrete_space"),
            ("NeuralUCB", "default_hp_config", "discrete_space"),
            ("NeuralTS", "default_hp_config", "discrete_space"),
        ],
    )
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_random_mutations(self, algo, init_pop, device, accelerator_flag):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        population = init_pop
        pre_training_mut = True

        mutations = Mutations(
            0,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            mutate_elite=False,
            device=device,
            accelerator=accelerator,
        )

        # Unwrap models if using accelerator
        if accelerator is not None:
            for agent in population:
                agent.unwrap_models()

        mutated_population = mutations.mutation(population, pre_training_mut)

        assert len(mutated_population) == len(population)
        assert (
            mutated_population[0].mut == "None"
        )  # Satisfies mutate_elite=False condition
        for individual in mutated_population:
            policy = getattr(individual, individual.registry.policy())
            assert individual.mut in [
                "None",
                "batch_size",
                "lr",
                "lr_actor",
                "lr_critic",
                "learn_step",
                "act",
                "param",
                policy.last_mutation_attr,
            ]

    # The mutation method applies no mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "action_space"),
        [
            ("DQN", "discrete_space"),
            ("Rainbow DQN", "discrete_space"),
            ("DDPG", "vector_space"),
            ("TD3", "vector_space"),
            ("PPO", "discrete_space"),
            ("CQN", "discrete_space"),
            ("NeuralUCB", "discrete_space"),
            ("NeuralTS", "discrete_space"),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_no_mutations(self, init_pop, device, accelerator_flag):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False

        population = init_pop

        mutations = Mutations(
            1,
            0,
            0,
            0,
            0,
            0,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut == "None"
            assert old.index == individual.index
            assert old.actor != individual.actor
            assert_state_dicts_equal(
                old.actor.state_dict(), individual.actor.state_dict()
            )

    # The mutation method applies no mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "action_space"),
        [
            ("DQN", "discrete_space"),
            ("Rainbow DQN", "discrete_space"),
            ("DDPG", "vector_space"),
            ("TD3", "vector_space"),
            ("PPO", "discrete_space"),
            ("CQN", "discrete_space"),
            ("NeuralUCB", "discrete_space"),
            ("NeuralTS", "discrete_space"),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_no_mutations_pre_training_mut(
        self, init_pop, device, accelerator_flag
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = True
        population = init_pop

        # Set all mutation probabilities to 0
        mutations = Mutations(
            1,
            0,
            0,
            0,
            0,
            1,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut in [
                "None",
                "batch_size",
                "lr",
                "lr_actor",
                "lr_critic",
                "learn_step",
            ]
            assert old.index == individual.index
            assert old.actor != individual.actor
            assert_state_dicts_equal(
                old.actor.state_dict(), individual.actor.state_dict()
            )

    # The mutation method applies RL hyperparameter mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "hp_config", "action_space"),
        [
            ("DQN", "default_hp_config", "discrete_space"),
            ("Rainbow DQN", "default_hp_config", "discrete_space"),
            ("DDPG", "ac_hp_config", "vector_space"),
            ("TD3", "ac_hp_config", "vector_space"),
            ("PPO", "default_hp_config", "discrete_space"),
            ("CQN", "default_hp_config", "discrete_space"),
            ("NeuralUCB", "default_hp_config", "discrete_space"),
            ("NeuralTS", "default_hp_config", "discrete_space"),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_rl_hp_mutations(
        self,
        init_pop,
        device,
        accelerator_flag,
        hp_config,
        request,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop
        mutations = Mutations(
            0,
            0,
            0,
            0,
            0,
            1,
            0.1,
            device=device,
            accelerator=accelerator,
        )
        hp_config = request.getfixturevalue(hp_config)

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            available_mutations = hp_config.names()
            assert individual.mut in available_mutations

            new_value = getattr(individual, individual.mut)
            min_value = hp_config[individual.mut].min
            max_value = hp_config[individual.mut].max
            assert min_value <= new_value <= max_value
            assert old.index == individual.index

    # The mutation method applies activation mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "action_space"),
        [
            ("DQN", "discrete_space"),
            ("Rainbow DQN", "discrete_space"),
            ("DDPG", "vector_space"),
            ("TD3", "vector_space"),
            ("PPO", "discrete_space"),
            ("CQN", "discrete_space"),
            ("NeuralUCB", "discrete_space"),
            ("NeuralTS", "discrete_space"),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [
            ("vector_space", "encoder_mlp_config"),
            ("image_space", "encoder_cnn_config"),
            ("dict_space", "encoder_multi_input_config"),
        ],
    )
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_activation_mutations(
        self,
        init_pop,
        observation_space,
        device,
        accelerator_flag,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop

        if (
            isinstance(observation_space, spaces.Box)
            and len(observation_space.shape) == 3
        ):
            activation_selection = ["ReLU", "ELU", "GELU"]
        else:
            activation_selection = ["Tanh", "ReLU", "ELU", "GELU"]

        mutations = Mutations(
            0,
            0,
            0,
            0,
            1,
            0,
            0.1,
            activation_selection=activation_selection,
            device=device,
            accelerator=accelerator,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut in ["None", "act"]
            if individual.mut == "act":
                assert old.actor.activation != individual.actor.activation
                assert individual.actor.activation in activation_selection
            assert old.index == individual.index

    # The mutation method applies activation mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [
            ("vector_space", "encoder_mlp_config"),
            ("image_space", "encoder_cnn_config"),
            ("dict_space", "encoder_multi_input_config"),
            ("discrete_space", "encoder_mlp_config"),
        ],
    )
    @pytest.mark.parametrize(("algo", "action_space"), [("DDPG", "vector_space")])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_activation_mutations_no_skip(
        self, init_pop, device, accelerator_flag
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop
        mutations = Mutations(
            0,
            0,
            0,
            0,
            1,
            0,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        for individual in population:
            individual.algo = None
            individual.lr = 1e-3
        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut in ["None", "act"]
            if individual.mut == "act":
                assert old.actor.activation != individual.actor.activation
                assert individual.actor.activation in ["ReLU", "ELU", "GELU"]
            assert old.index == individual.index

    # The mutation method applies parameter mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "action_space", "wrapper_cls"),
        [
            ("DQN", "discrete_space", None),
            ("Rainbow DQN", "discrete_space", None),
            ("DDPG", "vector_space", None),
            ("DDPG", "vector_space", RSNorm),
            ("TD3", "vector_space", None),
            ("PPO", "discrete_space", None),
            ("CQN", "discrete_space", None),
            ("NeuralUCB", "discrete_space", None),
            ("NeuralTS", "discrete_space", None),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_parameter_mutations(
        self,
        algo,
        device,
        accelerator_flag,
        init_pop,
        wrapper_cls,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False

        population = init_pop

        if wrapper_cls is not None:
            population = [wrapper_cls(agent) for agent in population]

        mutations = Mutations(
            0,
            0,
            0,
            1,
            0,
            0,
            0.5,
            device=device,
            accelerator=accelerator,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut == "param"
            # Due to randomness, sometimes parameters are not different
            # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
            assert old.index == individual.index

            # Compare state dictionaries of the actor (or network)
            policy_name = old.registry.policy()
            old_policy = getattr(old, policy_name)
            new_policy = getattr(individual, policy_name)
            old_sd = old_policy.state_dict()
            new_sd = new_policy.state_dict()
            mutation_found = False
            for key in old_sd:
                if "norm" in key:  # Skip normalization layers
                    continue
                diff_norm = (old_sd[key] - new_sd[key]).norm().item()
                if diff_norm > 1e-6:
                    mutation_found = True
                    break

            assert mutation_found, f"Mutation not applied for agent index {old.index}"

    #### Multi-agent algorithm mutations ####
    # The mutation method applies random mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("ma_vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    def test_applies_random_mutations_multi_agent(
        self, init_pop, device, accelerator_flag
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop

        # Random mutations
        mutations = Mutations(
            0,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        if accelerator is not None:
            for agent in population:
                agent.unwrap_models()

        mutated_population = mutations.mutation(population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for individual in mutated_population:
            policy = getattr(individual, individual.registry.policy())
            if policy.last_mutation_attr is not None:
                sampled_mutation = ".".join(policy.last_mutation_attr.split(".")[1:])
            else:
                sampled_mutation = "None"

            assert individual.mut in [
                "None",
                "batch_size",
                "lr",
                "lr_actor",
                "lr_critic",
                "learn_step",
                "act",
                "param",
                sampled_mutation,
            ]

    # The mutation method applies no mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("ma_vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    def test_applies_no_mutations_multi_agent(self, init_pop, device, accelerator_flag):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop

        mutations = Mutations(
            1,
            0,
            0,
            0,
            0,
            0,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        if accelerator is not None:
            for agent in population:
                agent.unwrap_models()

        mutated_population = mutations.mutation(population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut == "None"
            assert old.index == individual.index
            assert old.actors == individual.actors

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "hp_config"),
        [
            ("MADDPG", "ac_hp_config"),
            ("MATD3", "ac_hp_config"),
            ("IPPO", "default_hp_config"),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("ma_vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    def test_applies_rl_hp_mutations_multi_agent(
        self,
        init_pop,
        device,
        accelerator_flag,
        hp_config,
        request,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop

        mutations = Mutations(
            0,
            0,
            0,
            0,
            0,
            1,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        hp_config = request.getfixturevalue(hp_config)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            available_mutations = hp_config.names()
            assert individual.mut in available_mutations

            new_value = getattr(individual, individual.mut)
            min_value = hp_config[individual.mut].min
            max_value = hp_config[individual.mut].max
            assert min_value <= new_value <= max_value
            assert old.index == individual.index

    # The mutation method applies activation mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [
            ("ma_vector_space", "encoder_mlp_config"),
            ("ma_image_space", "encoder_cnn_config"),
            ("ma_dict_space_small", "encoder_multi_input_config"),
        ],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    def test_applies_activation_mutations_multi_agent(
        self,
        init_pop,
        device,
        accelerator_flag,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop

        mutations = Mutations(
            0,
            0,
            0,
            0,
            1,
            0,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut in ["None", "act"]
            if individual.mut == "act":
                for old_actor, actor in zip(
                    old.actors, individual.actors, strict=False
                ):
                    assert old_actor.activation != actor.activation
                    assert individual.actors[0].activation in [
                        "ReLU",
                        "ELU",
                        "GELU",
                    ]
            assert old.index == individual.index

    # The mutation method applies activation mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize("algo", ["MADDPG", "MATD3", "IPPO"])
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("ma_vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    def test_applies_activation_mutations_multi_agent_no_skip(
        self,
        init_pop,
        device,
        accelerator_flag,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop

        mutations = Mutations(
            0,
            0,
            0,
            0,
            1,
            0,
            0.1,
            device=device,
            accelerator=accelerator,
        )

        for individual in population:
            individual.algo = None
        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut in ["None", "act"]
            if individual.mut == "act":
                for old_actor, actor in zip(
                    old.actors.values(),
                    individual.actors.values(),
                    strict=False,
                ):
                    assert old_actor.activation != actor.activation
                    assert actor.activation in [
                        "ReLU",
                        "ELU",
                        "GELU",
                    ]
            assert old.index == individual.index

    # The mutation method applies parameter mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "wrapper_cls"),
        [
            ("MADDPG", None),
            ("MATD3", None),
            ("IPPO", None),
            ("IPPO", AsyncAgentsWrapper),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("ma_vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    def test_applies_parameter_mutations_multi_agent(
        self,
        init_pop,
        device,
        accelerator_flag,
        wrapper_cls,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        pre_training_mut = False
        population = init_pop

        if wrapper_cls is not None:
            population = [wrapper_cls(agent) for agent in population]

        mutations = Mutations(
            0,
            0,
            0,
            1,
            0,
            0,
            0.5,
            device=device,
            accelerator=accelerator,
        )

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert individual.mut == "param"
            # Due to randomness, sometimes parameters are not different
            # assert str(old.actors[0].state_dict()) != str(individual.actors[0].state_dict())
            assert old.index == individual.index

            # Compare state dictionaries of the actor (or network)
            policy_name = old.registry.policy()
            old_policy = getattr(old, policy_name)
            new_policy = getattr(individual, policy_name)
            old_sd = old_policy.state_dict()
            new_sd = new_policy.state_dict()
            mutation_found = False
            for key in old_sd:
                if "norm" in key:  # Skip normalization layers
                    continue
                diff_norm = (old_sd[key] - new_sd[key]).norm().item()
                if diff_norm > 1e-6:
                    mutation_found = True
                    break

            assert mutation_found, f"Mutation not applied for agent index {old.index}"

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed"
    )
    @pytest.mark.parametrize(
        ("use_accelerator", "use_deepspeed_optimizer"),
        [
            (True, True),
            (True, False),
            (False, False),
        ],
    )
    @pytest.mark.parametrize("algo", ["GRPO", "DPO"])
    @pytest.mark.parametrize(
        "hp_to_mutate",
        [
            "lr",
            # "max_grad_norm"
        ],
    )
    def test_applies_rl_hp_mutation_llm_algorithm(
        self,
        request,
        vector_space,
        monkeypatch,
        use_accelerator,
        use_deepspeed_optimizer,
        algo,
        hp_to_mutate,
        grpo_hp_config,
        deepspeed_env,
    ):
        if use_accelerator and not torch.cuda.is_available():
            pytest.skip("DeepSpeed accelerator LLM mutation tests require CUDA.")

        if hp_to_mutate == "max_grad_norm":
            grpo_hp_config = HyperparameterConfig(
                max_grad_norm=RLParameter(min=0.1, max=1.0),
            )

        pre_training_mut = False

        if use_accelerator:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            try:
                import deepspeed.comm.comm as ds_comm
                import deepspeed.utils.groups as ds_groups

                for attr in dir(ds_groups):
                    if attr.startswith("_") and attr.endswith("_GROUP"):
                        setattr(ds_groups, attr, None)
                ds_comm.cdb = None
            except ImportError:
                pass
            AcceleratorState._reset_state(True)

            deepspeed_config = {
                "gradient_accumulation_steps": 1,
                "zero_optimization": {
                    "stage": 2,
                },
                "gradient_clipping": 0.3,
            }
            if use_deepspeed_optimizer:
                deepspeed_config["optimizer"] = {
                    "type": "AdamW",
                    "params": {
                        "lr": 1e-4,  # Smaller learning rate
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                    },
                }
            accelerator = Accelerator(
                deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=deepspeed_config),
            )
        else:
            accelerator = None
        init_hp = {
            "PAD_TOKEN_ID": 1000 - 1,
            "PAD_TOKEN": "<pad>",
            "BATCH_SIZE": 2,
            "BETA": 0.001,
            "LR": 0.001,
            "MAX_GRAD_NORM": 0.5,
            "UPDATE_EPOCHS": 1,
            "MAX_OUTPUT_TOKENS": 32,
            "MAX_MODEL_LEN": 100,
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        population = create_population(
            algo=algo,
            observation_space=vector_space,
            action_space=copy.deepcopy(vector_space),
            net_config=None,
            INIT_HP=init_hp,
            hp_config=grpo_hp_config,
            actor_network=create_module(
                input_size=10,
                max_tokens=20,
                vocab_size=1000,
                device=device,
            ),
            algo_kwargs={
                "lora_config": LoraConfig(
                    r=16,
                    lora_alpha=64,
                    target_modules=["linear_1"],
                    task_type="CAUSAL_LM",
                    lora_dropout=0.05,
                ),
                "pad_token_id": 1000 - 1,
                "pad_token": "<pad>",
            },
            accelerator=accelerator,
            device=device,
        )

        mutations = Mutations(
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            device=device,
            accelerator=accelerator,
        )

        print("original lr: ", [agent.lr for agent in population])

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = mutations.mutation(new_population, pre_training_mut)

        print("mutated lr: ", [agent.lr for agent in mutated_population])

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            available_mutations = grpo_hp_config.names()
            assert individual.mut in available_mutations

            new_value = getattr(individual, individual.mut)
            min_value = grpo_hp_config[individual.mut].min
            max_value = grpo_hp_config[individual.mut].max
            assert min_value <= new_value <= max_value
            assert old.index == individual.index
        for agent in mutated_population:
            opt = (
                agent.actor.optimizer
                if (use_deepspeed_optimizer and use_accelerator)
                else agent.optimizer.optimizer
            )
            for param_group in opt.param_groups:
                assert param_group["lr"] == agent.lr
            if use_accelerator:
                assert (
                    agent.accelerator.state.deepspeed_plugin.deepspeed_config[
                        "gradient_clipping"
                    ]
                    == agent.max_grad_norm
                )
        for mut_agent, old_agent in zip(
            mutated_population, new_population, strict=False
        ):
            mut_agent.clean_up()
            old_agent.clean_up()
        if use_accelerator:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            try:
                import deepspeed.comm.comm as ds_comm
                import deepspeed.utils.groups as ds_groups

                for attr in dir(ds_groups):
                    if attr.startswith("_") and attr.endswith("_GROUP"):
                        setattr(ds_groups, attr, None)
                ds_comm.cdb = None
            except ImportError:
                pass
            AcceleratorState._reset_state(True)

    @pytest.mark.skipif(
        not (HAS_VLLM and HAS_DEEPSPEED),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    @pytest.mark.parametrize(
        "mutation_type", ["architecture", "parameters", "activation"]
    )
    @pytest.mark.parametrize("algo", ["GRPO", "DPO"])
    def test_warns_on_llm_algorithm(
        self,
        request,
        grpo_hp_config,
        vector_space,
        mutation_type,
        algo,
    ):
        pre_training_mut = False
        init_hp = {
            "PAD_TOKEN_ID": 1000 - 1,
            "PAD_TOKEN": "<pad>",
            "BATCH_SIZE": 2,
            "BETA": 0.001,
            "LR": 5e-7,
            "MAX_GRAD_NORM": 0.1,
            "UPDATE_EPOCHS": 1,
            "MAX_OUTPUT_TOKENS": 32,
            "MAX_MODEL_LEN": 100,
        }

        population = create_population(
            algo=algo,
            observation_space=vector_space,
            action_space=copy.deepcopy(vector_space),
            net_config=None,
            INIT_HP=init_hp,
            hp_config=grpo_hp_config,
            actor_network=create_module(
                input_size=10,
                max_tokens=20,
                vocab_size=1000,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            algo_kwargs={
                "lora_config": LoraConfig(
                    r=16,
                    lora_alpha=64,
                    target_modules=["linear_1"],
                    task_type="CAUSAL_LM",
                    lora_dropout=0.05,
                ),
                "pad_token_id": 1000 - 1,
                "pad_token": "<pad>",
            },
        )

        mutations = Mutations(
            0,
            1 if mutation_type == "architecture" else 0,
            0.5 if mutation_type == "architecture" else 0,
            1 if mutation_type == "parameters" else 0,
            1 if mutation_type == "activation" else 0,
            0,
            0.1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            accelerator=None,
        )

        new_population = [agent.clone(wrap=False) for agent in population]

        if mutation_type == "architecture":
            with pytest.raises(MutationError):
                mutations.mutation(new_population, pre_training_mut)

            # Since MutationError is expected, create a dummy mutated_population for the assertions
            mutated_population = new_population
            for individual in mutated_population:
                individual.mut = "None"
        else:
            with pytest.warns(UserWarning, match="mutations are not supported"):
                mutated_population = mutations.mutation(
                    new_population, pre_training_mut
                )

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            assert old.mut is None
            assert individual.mut == "None"

        for mut_agent, old_agent in zip(
            mutated_population, new_population, strict=False
        ):
            mut_agent.clean_up()
            old_agent.clean_up()


class TestMutationsArchitectureMutate:
    def test_raises_for_unsupported_individual(self):
        """Mutations.architecture_mutate raises MutationError when individual is not RLAlgorithm or MultiAgentRLAlgorithm."""
        mutations = Mutations(0, 1, 0.5, 0, 0, 0, 0.5, device="cpu")
        with pytest.raises(
            MutationError, match="Architecture mutations are not supported"
        ):
            mutations.architecture_mutate("not_an_algorithm")

    # The mutation method applies architecture mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "action_space", "wrapper_cls"),
        [
            ("DQN", "discrete_space", None),
            ("Rainbow DQN", "discrete_space", None),
            ("DDPG", "vector_space", None),
            ("DDPG", "vector_space", RSNorm),
            ("TD3", "vector_space", None),
            ("PPO", "discrete_space", None),
            ("CQN", "discrete_space", None),
            ("NeuralUCB", "discrete_space", None),
            ("NeuralTS", "discrete_space", None),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [
            ("vector_space", "encoder_mlp_config"),
            ("image_space", "encoder_cnn_config"),
            ("dict_space", "encoder_multi_input_config"),
        ],
    )
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    def test_applies_architecture_mutations(
        self,
        init_pop,
        device,
        accelerator_flag,
        wrapper_cls,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        population: list[EvolvableAlgorithm] = init_pop
        if wrapper_cls is not None:
            population = [wrapper_cls(agent) for agent in population]

        mutations = Mutations(
            0,
            1,
            0.5,
            0,
            0,
            0,
            0.5,
            device=device,
            accelerator=accelerator,
        )

        mut_methods = population[0].actor.mutation_methods

        # Change EvolvableModule random number generator to test mutation methods
        class EvoDummyRNG:
            rng = np.random.default_rng(seed=42)

            def choice(self, a, size=None, replace=True, p=None):
                return 1

            def integers(self, low=0, high=None):
                return self.rng.integers(low, high)

        for individual in population:
            for network in individual.evolvable_attributes(
                networks_only=True,
            ).values():
                network.rng = EvoDummyRNG()

        applied_mutations = set()
        for mut_method in mut_methods:

            class DummyRNG:
                def choice(self, a, size=None, replace=True, p=None, _mut=mut_method):
                    return [_mut]

            mutations.rng = DummyRNG()

            new_population = [agent.clone(wrap=False) for agent in population]

            # Apply architecture mutations to the population
            if isinstance(population[0], RSNorm):
                new_population = [agent.agent for agent in new_population]

            mutated_population = [
                mutations.architecture_mutate(agent) for agent in new_population
            ]
            for individual in mutated_population:
                individual.mutation_hook()

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population, strict=False):
                policy_name = old.registry.policy()
                policy = getattr(individual, policy_name)
                # old_policy = getattr(old, policy_name)
                assert individual.mut == (policy.last_mutation_attr or "None")

                if policy.last_mutation_attr is not None:
                    applied_mutations.add(policy.last_mutation_attr)
                    # assert str(old_policy.state_dict()) != str(policy.state_dict())
                    for group in old.registry.groups:
                        if group.eval_network != policy_name:
                            eval_module = getattr(individual, group.eval_network)
                            # old_eval_module = getattr(old, group.eval_network)
                            assert eval_module.last_mutation_attr is not None
                            assert (
                                eval_module.last_mutation_attr
                                == policy.last_mutation_attr
                            )
                            # assert str(old_eval_module.state_dict()) != str(eval_module.state_dict())

                assert old.index == individual.index

            # assert_equal_state_dict(population, mutated_population)

        assert all(mut in mut_methods for mut in applied_mutations), set(
            mut_methods
        ) - set(
            applied_mutations,
        )

    # The mutation method applies BERT architecture mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.skip(reason="Skipping BERT architecture mutations test.")
    @pytest.mark.parametrize(
        ("algo", "actor_network", "critic_network"),
        [("DDPG", "bert_network", "bert_network")],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["vector_space"])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(
        "mut_method",
        [
            [
                "add_encoder_layer",
                "remove_encoder_layer",
                "add_decoder_layer",
                "remove_decoder_layer",
            ],
            ["add_node", "remove_node"],
        ],
    )
    def test_applies_bert_architecture_mutations_single_agent(
        self,
        algo,
        observation_space,
        action_space,
        device,
        accelerator_flag,
        mut_method,
        actor_network,
        critic_network,
        init_pop,
        request,
    ):
        accelerator = Accelerator(device_placement=True) if accelerator_flag else None
        observation_space = request.getfixturevalue(observation_space)
        action_space = request.getfixturevalue(action_space)

        # Pass the network parameters to init_pop through the test
        actual_actor_network = (
            request.getfixturevalue(actor_network) if actor_network else None
        )
        actual_critic_network = (
            request.getfixturevalue(critic_network) if critic_network else None
        )

        # Create a custom population with the BERT networks
        from agilerl.utils.utils import create_population

        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            hp_config=None,
            net_config=request.getfixturevalue("encoder_mlp_config"),
            INIT_HP=SHARED_INIT_HP,
            population_size=1,
            device=device,
            accelerator=accelerator,
            actor_network=actual_actor_network,
            critic_network=actual_critic_network,
        )

        mutations = Mutations(
            0,
            1,
            0.5,
            0,
            0,
            0,
            0.5,
            device=device,
            accelerator=accelerator,
        )

        class DummyRNG:
            def choice(self, a, size=None, replace=True, p=None):
                return [np.random.choice(mut_method)]

        mutations.rng = DummyRNG()

        new_population = [agent.clone(wrap=False) for agent in population]
        mutated_population = [
            mutations.architecture_mutate(agent) for agent in new_population
        ]

        assert len(mutated_population) == len(population)
        for old, individual in zip(population, mutated_population, strict=False):
            policy = getattr(individual, individual.registry.policy())
            assert individual.mut == policy.last_mutation_attr
            # Due to randomness and constraints on size, sometimes architectures are not different
            # assert str(old.actor.state_dict()) != str(individual.actor.state_dict())
            assert old.index == individual.index

        # assert_equal_state_dict(population, mutated_population)

    # The mutation method applies architecture mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("algo", "wrapper_cls"),
        [
            ("MADDPG", None),
            ("MATD3", None),
            ("IPPO", None),
            ("IPPO", AsyncAgentsWrapper),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [
            ("ma_vector_space", "encoder_mlp_config"),
            ("ma_image_space", "encoder_cnn_config"),
            # ("ma_dict_space_small", "encoder_multi_input_config"), NOTE: Takes too long to run
        ],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("accelerator_flag", [False])
    @pytest.mark.parametrize("torch_compiler", [None])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    def test_applies_architecture_mutations_multi_agent(
        self,
        algo,
        init_pop,
        device,
        accelerator_flag,
        wrapper_cls,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        population: list[EvolvableAlgorithm] = init_pop
        mutations = Mutations(
            0,
            1,
            0.5,
            0,
            0,
            0,
            0.5,
            device=device,
            accelerator=accelerator,
        )

        # Change EvolvableModule random number generator to test mutation methods
        class EvoDummyRNG:
            rng = np.random.default_rng(seed=42)

            def choice(self, a, size=None, replace=True, p=None):
                return 1

            def integers(self, low=0, high=None):
                return self.rng.integers(low, high)

        if wrapper_cls is not None:
            population = [wrapper_cls(agent) for agent in population]

        for individual in population:
            for network in individual.evolvable_attributes(networks_only=True).values():
                network.rng = EvoDummyRNG()

        sample_agent_id = population[0].agent_ids[0]
        test_agent = population[0].get_network_id(sample_agent_id)
        mut_methods = population[0].actors[test_agent].mutation_methods
        applied_mutations = set()
        for mut_method in mut_methods:

            class DummyRNG:
                def choice(self, a, size=None, replace=True, p=None, _mut=mut_method):
                    return [f"{test_agent}.{_mut}"]

            mutations.rng = DummyRNG()

            new_population = [agent.clone(wrap=False) for agent in population]

            if isinstance(new_population[0], AsyncAgentsWrapper):
                new_population = [agent.agent for agent in new_population]

            mutated_population = [
                mutations.architecture_mutate(agent) for agent in new_population
            ]

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population, strict=False):
                policy_name = individual.registry.policy()
                policy = getattr(individual, policy_name)
                # old_policy = getattr(old, policy_name)
                if policy.last_mutation_attr is not None:
                    sampled_mutation = ".".join(
                        policy.last_mutation_attr.split(".")[1:]
                    )
                    applied_mutations.add(sampled_mutation)
                else:
                    sampled_mutation = None

                assert True

                if sampled_mutation is not None:
                    for group in old.registry.groups:
                        if group.eval_network != policy_name:
                            eval_module = getattr(individual, group.eval_network)
                            # old_eval_module = getattr(old, group.eval_network)
                            for module in eval_module.values():
                                bottom_eval_mut = module.last_mutation_attr.split(".")[
                                    -1
                                ]
                                bottom_policy_mut = policy.last_mutation_attr.split(
                                    "."
                                )[-1]
                                assert module.last_mutation_attr is not None
                                assert bottom_eval_mut == bottom_policy_mut

                assert old.index == individual.index

        assert all(mut in applied_mutations for mut in mut_methods), set(
            mut_methods
        ) - set(
            applied_mutations,
        )

    # The mutation method applies BERT architecture mutations to the population and returns the mutated population.
    @pytest.mark.gpu
    @pytest.mark.skip(reason="Skipping BERT architecture mutations test.")
    @pytest.mark.parametrize(
        ("algo", "actor_network", "critic_network"),
        [
            ("MADDPG", "bert_networks_multi_agent", "bert_networks_multi_agent"),
            ("MATD3", "bert_networks_multi_agent", "bert_matd3_critic_networks"),
        ],
    )
    @pytest.mark.parametrize(
        ("observation_space", "net_config"),
        [("ma_vector_space", "encoder_mlp_config")],
    )
    @pytest.mark.parametrize("action_space", ["ma_discrete_space"])
    @pytest.mark.parametrize("INIT_HP", [SHARED_INIT_HP_MA])
    @pytest.mark.parametrize("population_size", [1])
    @pytest.mark.parametrize("hp_config", [None])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("torch_compiler", [None])
    def test_applies_bert_architecture_mutations_multi_agent(
        self,
        algo,
        device,
        accelerator_flag,
        init_pop,
        observation_space,
        action_space,
        request,
        actor_network,
        critic_network,
    ):
        accelerator = Accelerator(device_placement=False) if accelerator_flag else None
        observation_space = request.getfixturevalue(observation_space)
        action_space = request.getfixturevalue(action_space)

        # Pass the network parameters to init_pop through the test
        actual_actor_network = (
            request.getfixturevalue(actor_network) if actor_network else None
        )
        actual_critic_network = (
            request.getfixturevalue(critic_network) if critic_network else None
        )

        # Create a custom population with the BERT networks
        from agilerl.utils.utils import create_population

        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            hp_config=None,
            net_config=request.getfixturevalue("encoder_mlp_config"),
            INIT_HP=SHARED_INIT_HP_MA,
            population_size=1,
            device=device,
            accelerator=accelerator,
            actor_network=actual_actor_network,
            critic_network=actual_critic_network,
        )

        mutations = Mutations(
            0,
            1,
            0.5,
            0,
            0,
            0,
            0.5,
            device=device,
            accelerator=accelerator,
        )

        sample_agent_id = population[0].agent_ids[0]
        test_agent = population[0].get_network_id(sample_agent_id)
        mut_methods = population[0].actors[test_agent].mutation_methods
        for mut_method in mut_methods:

            class DummyRNG:
                def choice(self, a, size=None, replace=True, p=None, _mut=mut_method):
                    return [f"{test_agent}.{_mut}"]

            mutations.rng = DummyRNG()

            new_population = [agent.clone(wrap=False) for agent in population]
            mutated_population = [
                mutations.architecture_mutate(agent) for agent in new_population
            ]

            assert len(mutated_population) == len(population)
            for old, individual in zip(population, mutated_population, strict=False):
                policy_name = individual.registry.policy()
                policy = getattr(individual, policy_name)
                # old_policy = getattr(old, policy_name)
                if policy.last_mutation_attr is not None:
                    sampled_mutation = ".".join(
                        policy.last_mutation_attr.split(".")[1:]
                    )
                else:
                    sampled_mutation = None

                assert True

                if sampled_mutation is not None:
                    for group in old.registry.groups:
                        if group.eval_network != policy_name:
                            eval_module = getattr(individual, group.eval_network)
                            # old_eval_module = getattr(old, group.eval_network)
                            for module in eval_module.values():
                                bottom_eval_mut = module.last_mutation_attr.split(".")[
                                    -1
                                ]
                                bottom_policy_mut = policy.last_mutation_attr.split(
                                    "."
                                )[-1]
                                assert module.last_mutation_attr is not None
                                assert bottom_eval_mut == bottom_policy_mut

                assert old.index == individual.index


class TestMutationsNoMutation:
    def test_sets_mut_none(self):
        class DummyIndividual:
            mut = None

        muts = Mutations(1, 0, 0, 0, 0, 0, 0.1, device="cpu")
        ind = DummyIndividual()
        out = muts.no_mutation(ind)
        assert out.mut == "None"


class TestMutationsActivationMutation:
    @pytest.mark.gpu
    @pytest.mark.parametrize("algo_cls", [PPO, DDPG, TD3])
    def test_warns_for_policy_gradient_algos(
        self, algo_cls, vector_space, encoder_mlp_config, device
    ):
        action_space = (
            generate_random_box_space((2,))
            if algo_cls in (DDPG, TD3)
            else generate_discrete_space(2)
        )
        pop = algo_cls.population(
            size=1,
            observation_space=vector_space,
            action_space=action_space,
            net_config=encoder_mlp_config,
            device=device,
        )
        muts = Mutations(0, 0, 0, 0, 1, 0, 0.1, device=device)
        with pytest.warns(
            UserWarning,
            match=f"Activation mutations are not supported for {algo_cls.__name__}",
        ):
            out = muts.activation_mutation(pop[0].clone(wrap=False))
        assert out.mut == "None"

    @pytest.mark.gpu
    @pytest.mark.parametrize("algo", ["IPPO", "MADDPG", "MATD3"])
    def test_warns_for_multi_agent_policy_gradient_algos(
        self,
        algo,
        ma_discrete_space,
        ma_vector_space,
        encoder_mlp_config,
        device,
    ):
        from agilerl.utils.utils import create_population

        pop = create_population(
            algo=algo,
            observation_space=ma_discrete_space,
            action_space=ma_vector_space,
            net_config=encoder_mlp_config,
            INIT_HP=SHARED_INIT_HP_MA,
            population_size=1,
            device=device,
        )
        muts = Mutations(0, 0, 0, 0, 1, 0, 0.1, device=device)
        with pytest.warns(
            UserWarning,
            match=f"Activation mutations are not supported for {algo}",
        ):
            out = muts.activation_mutation(pop[0].clone(wrap=False))
        assert out.mut == "None"

    @pytest.mark.skipif(
        not (HAS_VLLM and HAS_DEEPSPEED),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    @pytest.mark.parametrize("algo", ["GRPO", "DPO"])
    def test_warns_for_llm_algorithms(self, algo, grpo_hp_config, vector_space, device):
        from agilerl.utils.utils import create_population

        init_hp = {
            "PAD_TOKEN_ID": 1000 - 1,
            "PAD_TOKEN": "<pad>",
            "BATCH_SIZE": 2,
            "BETA": 0.001,
            "LR": 5e-7,
            "MAX_GRAD_NORM": 0.1,
            "UPDATE_EPOCHS": 1,
            "MAX_OUTPUT_TOKENS": 32,
            "MAX_MODEL_LEN": 100,
        }
        pop = create_population(
            algo=algo,
            observation_space=vector_space,
            action_space=copy.deepcopy(vector_space),
            net_config=None,
            INIT_HP=init_hp,
            hp_config=grpo_hp_config,
            actor_network=create_module(
                input_size=10,
                max_tokens=20,
                vocab_size=1000,
                device=device,
            ),
            algo_kwargs={
                "lora_config": LoraConfig(
                    r=16,
                    lora_alpha=64,
                    target_modules=["linear_1"],
                    task_type="CAUSAL_LM",
                    lora_dropout=0.05,
                ),
                "pad_token_id": 1000 - 1,
                "pad_token": "<pad>",
            },
            population_size=1,
            device=device,
        )
        muts = Mutations(0, 0, 0, 0, 1, 0, 0.1, device=device, accelerator=None)
        agent = pop[0].clone(wrap=False)
        try:
            with pytest.warns(
                UserWarning,
                match="Activation mutations are not supported for LLM algorithms",
            ):
                out = muts.activation_mutation(agent)
            assert out.mut == "None"
        finally:
            agent.clean_up()


class TestMutationsRlHyperparamMutation:
    @pytest.mark.gpu
    def test_returns_none_when_hp_config_empty(self, device):
        class DummyIndividual:
            mut = None
            registry = type("R", (), {"hp_config": None})()

        muts = Mutations(0, 0, 0, 0, 0, 1, 0.1, device=device)
        ind = DummyIndividual()
        out = muts.rl_hyperparam_mutation(ind)
        assert out.mut == "None"


class TestMutationsGetMutationsOptions:
    @pytest.mark.parametrize("pretraining", [True, False])
    def test_pretraining_fallback(self, pretraining):
        muts = Mutations(1, 0, 0, 0, 0, 0, 0.1, device="cpu")
        opts, _proba = muts._get_mutations_options(pretraining=pretraining)
        assert len(opts) >= 1
        assert muts.no_mutation in opts

    def test_all_zero_uses_no_mutation(self):
        muts = Mutations(0, 0, 0, 0, 0, 0, 0.1, device="cpu")
        opts, proba = muts._get_mutations_options(pretraining=True)
        assert muts.no_mutation in opts
        assert len(opts) == 1
        assert proba[0] == 1.0


class TestGetExpLayer:
    def test_raises_for_non_evolvable_module(self):
        """get_exp_layer raises TypeError when offspring is not an EvolvableModule."""
        with pytest.raises(
            TypeError, match=r"Bandit algorithm architecture.*not supported"
        ):
            get_exp_layer(torch.nn.Linear(2, 2))

    def test_raises_for_non_linear_output_layer(self):
        """get_exp_layer raises TypeError when the output layer is not nn.Linear."""

        class NonLinearOutputModule(EvolvableModule):
            def __init__(self):
                super().__init__(device="cpu")
                self.out = torch.nn.ReLU()

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

            def get_output_dense(self):
                return self.out

        with pytest.raises(TypeError, match=r"expected a linear output layer"):
            get_exp_layer(NonLinearOutputModule())

    def test_returns_output_layer_for_evolvable_module(
        self, vector_space, discrete_space, encoder_mlp_config
    ):
        pop = NeuralUCB.population(
            size=1,
            observation_space=vector_space,
            action_space=discrete_space,
            net_config=encoder_mlp_config,
            device="cpu",
        )
        offspring = pop[0].actor.clone()
        exp_layer = get_exp_layer(offspring)
        assert exp_layer is not None
        assert hasattr(exp_layer, "parameters")


@pytest.mark.parametrize("seed", [None, 42])
def test_set_global_seed(seed):
    set_global_seed(seed)
    if seed is not None:
        state = np.random.get_state()
        assert state is not None


def test_set_global_seed_seeds_cuda_when_a_device_is_present():
    """CPU CI never takes this branch, so the device check is faked.

    An unseeded CUDA generator makes a GPU run unreproducible even though the
    CPU generators were seeded.
    """
    from unittest.mock import patch

    with (
        patch("agilerl.hpo.mutation.torch.cuda.is_available", return_value=True),
        patch("agilerl.hpo.mutation.torch.cuda.manual_seed") as cuda_seed,
    ):
        set_global_seed(42)

    cuda_seed.assert_called_once_with(42)


def test_set_global_seed_skips_cuda_without_a_device():
    from unittest.mock import patch

    with (
        patch("agilerl.hpo.mutation.torch.cuda.is_available", return_value=False),
        patch("agilerl.hpo.mutation.torch.cuda.manual_seed") as cuda_seed,
    ):
        set_global_seed(42)

    cuda_seed.assert_not_called()


def test_get_eval_modules_returns_policy_and_modules(
    vector_space, discrete_space, encoder_mlp_config
):
    pop = DQN.population(
        size=1,
        observation_space=vector_space,
        action_space=discrete_space,
        net_config=encoder_mlp_config,
        device="cpu",
    )
    policy, offspring_evals = pop[0].get_eval_modules()
    assert isinstance(policy, dict)
    assert isinstance(offspring_evals, dict)
    assert len(policy) >= 1


def test_get_eval_modules_cloning_false_returns_the_live_modules(
    vector_space, discrete_space, encoder_mlp_config
):
    pop = DQN.population(
        size=1,
        observation_space=vector_space,
        action_space=discrete_space,
        net_config=encoder_mlp_config,
        device="cpu",
    )
    agent = pop[0]
    policy_name = agent.registry.policy()

    policy, _offspring_evals = agent.get_eval_modules(cloning=False)

    assert policy[policy_name] is getattr(agent, policy_name)


class _IndexedAgent:
    """Minimal agent stand-in for the indices-path mutation tests."""

    def __init__(self, index):
        self.index = index
        self.mut = None
        self.hook_calls = 0
        self.observation_space = None
        self.action_space = None

    def get_action(self, *args, **kwargs):
        return None

    def learn(self, *args, **kwargs):
        return None

    def mutation_hook(self):
        self.hook_calls += 1


class _StubWrapper(AgentWrapper):
    """Concrete AgentWrapper over _IndexedAgent."""


def _tagging_mutations(mutate_elite=True, seed=0):
    """A mutations class whose only mutation tags the agent."""
    muts = Mutations(1, 0, 0, 0, 0, 0, rand_seed=seed, mutate_elite=mutate_elite)

    def tag(agent):
        agent.mut = "tagged"
        return agent

    muts.mut_options = (tag,)
    muts.mut_proba = np.array([1.0])
    return muts


def _labelled_mutations(seed=0):
    """A mutations class offering two distinguishable, tagging mutations."""
    muts = Mutations(1, 0, 0, 0, 0, 0, rand_seed=seed)

    def tag_a(agent):
        agent.mut = "A"
        return agent

    def tag_b(agent):
        agent.mut = "B"
        return agent

    muts.mut_options = (tag_a, tag_b)
    muts.mut_proba = np.array([0.5, 0.5])
    return muts


def _replacing_mutations(seed=0):
    """This fake returns a fresh object so the branch actually rebinds the wrapped agent."""
    muts = Mutations(1, 0, 0, 0, 0, 0, rand_seed=seed)

    def replace(agent):
        replacement = _IndexedAgent(agent.index)
        replacement.mut = "tagged"
        return replacement

    muts.mut_options = (replace,)
    muts.mut_proba = np.array([1.0])
    return muts


class TestMutationsMutationIndices:
    """The indices path of :meth:`Mutations.mutation`, used by MF-PBT."""

    def test_indices_only_mutates_selected_agents(self):
        muts = _tagging_mutations()
        pop = [_IndexedAgent(i) for i in range(5)]

        out = muts.mutation(pop, indices=[pop[1].index, pop[3].index])

        assert len(out) == 5

        for i in (1, 3):
            assert pop[i].mut == "tagged"
            assert pop[i].hook_calls == 1

        for i in (0, 2, 4):
            assert out[i] is pop[i]
            assert pop[i].mut is None
            assert pop[i].hook_calls == 0

    def test_indices_none_mutates_whole_population(self):
        muts = _tagging_mutations()
        pop = [_IndexedAgent(i) for i in range(4)]

        muts.mutation(pop, indices=None)

        assert all(a.mut == "tagged" for a in pop)
        assert all(a.hook_calls == 1 for a in pop)

    def test_empty_indices_is_a_noop(self):
        muts = _tagging_mutations()
        pop = [_IndexedAgent(i) for i in range(4)]

        out = muts.mutation(pop, indices=[])

        assert out == pop
        assert all(a.mut is None for a in pop)
        assert all(a.hook_calls == 0 for a in pop)

    def test_indices_path_ignores_elite_skip(self):
        # On the whole-population path with mutate_elite=False, agent 0 is spared
        # (no_mutation). On the indices path the caller has already chosen exactly
        # which agents to mutate, so the elite-skip must not apply.
        pop = [_IndexedAgent(i) for i in range(4)]
        _tagging_mutations(mutate_elite=False).mutation(pop, indices=None)
        assert pop[0].mut == "None"

        pop = [_IndexedAgent(i) for i in range(4)]
        _tagging_mutations(mutate_elite=False).mutation(pop, indices=[pop[0].index])
        assert pop[0].mut == "tagged"

    def test_indices_selection_is_reproducible(self):
        def run():
            muts = _labelled_mutations(seed=123)
            pop = [_IndexedAgent(i) for i in range(5)]
            muts.mutation(pop, indices=[pop[1].index, pop[4].index])
            return pop

        first, second = run(), run()

        assert [first[1].mut, first[4].mut] == [second[1].mut, second[4].mut]
        assert all(a.mut in ("A", "B") for a in (first[1], first[4]))
        assert [first[i].mut for i in (0, 2, 3)] == [None, None, None]

    def test_indices_replaces_the_agent_inside_a_wrapper(self):
        # A wrapped individual keeps its wrapper in the population; only the agent it
        # holds is swapped for the mutated one.
        muts = _replacing_mutations()
        pop = [_StubWrapper(_IndexedAgent(i)) for i in range(3)]
        originals = [wrapper.agent for wrapper in pop]

        out = muts.mutation(pop, indices=[1])

        assert out == pop  # the wrapper objects themselves are preserved
        assert pop[1].agent is not originals[1]
        assert pop[1].agent.mut == "tagged"
        assert pop[1].agent.hook_calls == 1

        for i in (0, 2):
            assert pop[i].agent is originals[i]
            assert pop[i].agent.mut is None

    def test_unknown_indices_select_nothing(self):
        muts = _tagging_mutations()
        pop = [_IndexedAgent(i) for i in range(3)]

        out = muts.mutation(pop, indices=[99])

        assert out == pop
        assert all(a.mut is None for a in pop)
        assert all(a.hook_calls == 0 for a in pop)

    def test_unknown_indices_do_not_shift_the_known_ones(self):
        muts = _tagging_mutations()
        pop = [_IndexedAgent(i) for i in range(3)]

        muts.mutation(pop, indices=[99, pop[2].index])

        assert pop[2].mut == "tagged"
        assert [pop[0].mut, pop[1].mut] == [None, None]


class TestMutationsApplyMutation:
    """The wrapper-preserving per-individual apply shared by both entry points."""

    def test_whole_population_replaces_the_agent_inside_a_wrapper(self):
        muts = _replacing_mutations()
        pop = [_StubWrapper(_IndexedAgent(i)) for i in range(3)]
        originals = [wrapper.agent for wrapper in pop]

        out = muts.mutation(pop)

        assert out == pop  # the wrapper objects themselves are preserved
        for wrapper, original in zip(out, originals, strict=True):
            assert wrapper.agent is not original
            assert wrapper.agent.mut == "tagged"
            assert wrapper.agent.hook_calls == 1


def _regrama_mutations(**kwargs) -> Mutations:
    """A parameter-mutation-only operator; every parameter mutation runs ReGraMa."""
    return Mutations(0, 0, 0, 1, 0, 0, rand_seed=0, **kwargs)


def _all_dormant(agent) -> None:
    """Mark every measured neuron of every evaluation network as dormant."""
    agent.grama_scores = grama_scores_for(agent, fill=0.0)


def _dormant_for(agent, network_ids: set[str]):
    """Mark only the sub-agents in network_ids as dormant, the rest healthy."""
    dormant = grama_scores_for(agent, fill=0.0)
    healthy = grama_scores_for(agent, fill=1.0)
    return [
        dormant[idx] if network_id in network_ids else healthy[idx]
        for idx, (network_id, _network) in enumerate(agent.unrolled_eval_networks())
    ]


def _pin_biases(module, value: float = 1.0) -> None:
    """Set every one-dimensional bias of module to a non-zero sentinel.

    The Gaussian pass only ever writes two-dimensional tensors, so a bias sitting
    at zero after a parameter mutation can only have been zeroed by a ReGraMa reset.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("bias") and param.dim() == 1:
                param.fill_(value)


def _zeroed_biases(module) -> int:
    """Count the pinned biases a ReGraMa reset has zeroed."""
    return sum(
        int(bool((value == 0).all()))
        for key, value in module.state_dict().items()
        if key.endswith("bias") and value.dim() == 1
    )


_pinned_weight = 5.0
# Band separators, each mid-gap between what the two bands can reach on a
# pinned weight: ordinary noise is N(0, 0.5) here, while a reset redraws from N(0, 1).
_reset_floor = 3.0
_reset_residual = 2.0


def _pinned_policy() -> EvolvableModule:
    """Return a DQN policy with every mutable weight pinned to one value.

    Pinning turns each Gaussian band into a known displacement, so a band can be
    shown to have fired, or not, without reaching into the operator.
    """
    torch.manual_seed(0)
    network = DQN(
        generate_random_box_space((4,)),
        generate_discrete_space(2),
        device="cpu",
    ).actor
    with torch.no_grad():
        for key, value in network.state_dict().items():
            if value.dim() == 2 and "norm" not in key and "lstm" not in key:
                value.fill_(_pinned_weight)
    return network


def _pinned_gaussian_pass():
    """Run one Gaussian pass over an identically seeded pinned policy."""
    network = _pinned_policy()
    baseline = {key: value.clone() for key, value in network.state_dict().items()}
    Mutations(
        0,
        0,
        0,
        1,
        0,
        0,
        rand_seed=0,
    )._gaussian_parameter_mutation(network)
    return baseline, network.state_dict()


def _largest_step(baseline, after) -> float:
    """Return the largest absolute weight change the pass made."""
    return max((after[key] - baseline[key]).abs().max().item() for key in baseline)


def _smallest_magnitude(baseline, after) -> float:
    """Return the smallest weight left in a tensor that was pinned before the pass."""
    return min(
        after[key].abs().min().item()
        for key, value in baseline.items()
        if bool((value == _pinned_weight).all())
    )


def _ordinary_band_step_exists(baseline, after) -> bool:
    """Return whether any pinned entry moved by an ordinary-band-sized step.

    An ordinary step has standard deviation mutation_sd * pinned_weight = 0.5,
    so a nonzero step under _reset_floor that leaves the weight still far from
    zero can only be the ordinary band: a reset redraws from N(0, 1) and so
    almost never lands within reach of the original pinned value.
    """
    for key, value in baseline.items():
        if not bool((value == _pinned_weight).all()):
            continue
        after_value = after[key]
        delta = (after_value - value).abs()
        small_step = (delta > 0) & (delta < _reset_floor)
        far_from_zero = after_value.abs() > _reset_residual
        if bool((small_step & far_from_zero).any()):
            return True
    return False


class TestMutationsRegramaConstructor:
    """Validate the parameter-mutation dormancy threshold at construction time."""

    def test_defaults(self):
        muts = Mutations(0, 0, 0, 1, 0, 0)

        assert muts.dormant_threshold == 0.01

    def test_zero_dormant_threshold_is_accepted(self):
        muts = Mutations(0, 0, 0, 1, 0, 0, dormant_threshold=0.0)

        assert muts.dormant_threshold == 0.0

    def test_negative_dormant_threshold_is_rejected(self):
        with pytest.raises(AssertionError, match="Dormant threshold"):
            Mutations(0, 0, 0, 1, 0, 0, dormant_threshold=-0.1)


class TestMutationsGaussianParameterMutationFixedSplit:
    """The Gaussian parameter mutation applies a fixed, unconditional 95%/5% split."""

    def test_the_reset_band_fires(self):
        # Only a reset can move a pinned weight this far, and only a reset can
        # leave one this close to zero.
        baseline, after = _pinned_gaussian_pass()

        assert _largest_step(baseline, after) > _reset_floor
        assert _smallest_magnitude(baseline, after) < _reset_residual

    def test_the_ordinary_band_also_fires_in_the_same_pass(self):
        baseline, after = _pinned_gaussian_pass()

        assert _ordinary_band_step_exists(baseline, after)


class TestMutationsRegramaParameterMutation:
    """Reset dormant neurons before the Gaussian pass of a parameter mutation."""

    def make_agent(self):
        return DQN(
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            device="cpu",
        )

    def test_dormant_neurons_are_reset_and_the_mutation_is_still_a_parameter_one(
        self,
    ):
        agent = self.make_agent()
        _all_dormant(agent)
        _pin_biases(agent.actor)

        agent = _regrama_mutations().parameter_mutation(agent)

        assert agent.mut == "param"
        # The Gaussian pass changes the state dict on its own, so the reset has
        # to be witnessed by something only ReGraMa writes.
        assert _zeroed_biases(agent.actor) > 0
        assert all(
            torch.isfinite(value).all() for value in agent.actor.state_dict().values()
        )

    def test_target_network_is_resynced_with_the_reset_policy(self):
        agent = self.make_agent()
        _all_dormant(agent)

        agent = _regrama_mutations().parameter_mutation(agent)

        actor_state = agent.actor.state_dict()
        target_state = agent.actor_target.state_dict()
        assert all(torch.equal(actor_state[k], target_state[k]) for k in actor_state)

    def test_non_policy_networks_are_reset_too(self):
        # ReGraMa measures every evaluation network, not just the policy, so a
        # dormant critic is recovered as well.
        agent = PPO(
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            device="cpu",
        )
        _all_dormant(agent)
        before = {k: v.clone() for k, v in agent.critic.state_dict().items()}

        agent = _regrama_mutations().parameter_mutation(agent)

        # The Gaussian pass only touches the policy, so any change here
        # is the ReGraMa reset.
        after = agent.critic.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_resets_run_before_the_gaussian_pass(self):
        agent = self.make_agent()
        _all_dormant(agent)
        muts = _regrama_mutations()
        original = muts._gaussian_parameter_mutation
        seen = {}

        def recording(network):
            seen["weights"] = {k: v.clone() for k, v in network.state_dict().items()}
            return original(network)

        muts._gaussian_parameter_mutation = recording
        before = {k: v.clone() for k, v in agent.actor.state_dict().items()}

        muts.parameter_mutation(agent)

        assert any(not torch.equal(before[k], seen["weights"][k]) for k in before)

    def test_healthy_agent_is_only_perturbed_by_the_gaussian_pass(self):
        agent = PPO(
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            device="cpu",
        )
        _all_dormant(agent)
        agent.grama_scores = [
            [None if entry is None else torch.ones_like(entry) for entry in network]
            for network in agent.grama_scores
        ]
        before = {k: v.clone() for k, v in agent.critic.state_dict().items()}

        agent = _regrama_mutations().parameter_mutation(agent)

        after = agent.critic.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_the_configured_threshold_decides_what_counts_as_dormant(self):
        # The critic is the clean witness: the Gaussian pass only ever runs on
        # the policy, so any change here is the ReGraMa reset.
        def reset_critic_with(threshold: float):
            agent = PPO(
                generate_random_box_space((4,)),
                generate_discrete_space(2),
                device="cpu",
            )
            agent.grama_scores = grama_scores_for(agent, fill=1.0)
            networks = [
                network for _network_id, network in agent.unrolled_eval_networks()
            ]
            for entry in agent.grama_scores[networks.index(agent.critic)]:
                if entry is not None:
                    entry[0] = 0.5
            before = {k: v.clone() for k, v in agent.critic.state_dict().items()}
            agent = _regrama_mutations(dormant_threshold=threshold).parameter_mutation(
                agent,
            )
            return before, agent.critic.state_dict()

        permissive_before, permissive_after = reset_critic_with(0.6)
        strict_before, strict_after = reset_critic_with(0.1)

        assert any(
            not torch.equal(permissive_before[k], permissive_after[k])
            for k in permissive_before
        )
        assert all(
            torch.equal(strict_before[k], strict_after[k]) for k in strict_before
        )

    def test_missing_snapshot_falls_back_to_gaussian_silently(self):
        muts = _regrama_mutations()
        agent = self.make_agent()
        before = {k: v.clone() for k, v in agent.actor.state_dict().items()}

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent = muts.parameter_mutation(agent)

        assert agent.mut == "param"
        # No snapshot means the reset never runs, so the only source of
        # change is the Gaussian pass.
        after = agent.actor.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_snapshot_that_captured_nothing_behaves_like_a_missing_one(self):
        agent = self.make_agent()
        agent.grama_scores = [
            [None] * len(mutation_utils.target_activations(network))
            for _network_id, network in agent.unrolled_eval_networks()
        ]
        assert agent.grama_scores

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent = _regrama_mutations().parameter_mutation(agent)

        assert agent.mut == "param"

    def test_pre_training_step_with_no_snapshot_does_not_raise(self):
        # No agent has trained yet there, so a missing snapshot is expected.
        muts = _regrama_mutations()
        agent = self.make_agent()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            muts.mutation([agent], pre_training_mut=True)

    def test_reset_is_reproducible_across_equally_seeded_operators(self):
        first, second = self.make_agent(), self.make_agent()
        second.actor.load_state_dict(first.actor.state_dict())
        _all_dormant(first)
        _all_dormant(second)

        torch.manual_seed(0)
        _regrama_mutations().parameter_mutation(first)
        torch.manual_seed(0)
        _regrama_mutations().parameter_mutation(second)

        assert_state_dicts_equal(first.actor.state_dict(), second.actor.state_dict())

    def test_a_reset_failure_propagates(self, monkeypatch):
        agent = PPO(
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            device="cpu",
        )
        _all_dormant(agent)

        def explode(*_args, **_kwargs):
            msg = "surgery blew up"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "agilerl.hpo.mutation.reset_dormant_neurons",
            explode,
        )

        with pytest.raises(RuntimeError, match="surgery blew up"):
            _regrama_mutations().parameter_mutation(agent)

    def test_a_shared_head_lookup_failure_propagates(self, monkeypatch):
        agent = PPO(
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            device="cpu",
        )
        _all_dormant(agent)

        def explode(*_args, **_kwargs):
            msg = "shared head lookup blew up"
            raise RuntimeError(msg)

        monkeypatch.setattr("agilerl.hpo.mutation.shared_encoder_heads", explode)

        with pytest.raises(RuntimeError, match="shared head lookup blew up"):
            _regrama_mutations().parameter_mutation(agent)

    def test_wrapped_agents_are_reset_through_the_wrapper(self):
        agent = RSNorm(self.make_agent())
        _all_dormant(agent.agent)
        _pin_biases(agent.agent.actor)

        result = _regrama_mutations().mutation([agent])

        assert isinstance(result[0], AgentWrapper)
        assert _zeroed_biases(result[0].agent.actor) > 0


def _policy_latent_dormant(agent, index: int = 0) -> None:
    """Mark only the policy's latent unit index dormant; everything else healthy."""
    agent.grama_scores = grama_scores_for(agent, fill=1.0)
    policy = getattr(agent, agent.registry.policy())
    # The latent is the encoder's terminal activation, i.e. the only boundary a
    # shared encoder carries into another network's head.
    terminal = mutation_utils.activation_modules(policy.encoder, include_output=True)[
        -1
    ]
    position = mutation_utils.target_activations(policy).index(terminal)
    policy_entry = next(
        entry
        for (_network_id, network), entry in zip(
            agent.unrolled_eval_networks(),
            agent.grama_scores,
            strict=True,
        )
        if network is policy
    )
    policy_entry[position][index] = 0.0


class TestMutationsRegramaSharedEncoders:
    """Networks borrowing the policy's encoder are faded along with it."""

    def make_agent(self, *, share):
        return PPO(
            generate_random_box_space((4,)),
            generate_discrete_space(2),
            share_encoders=share,
            device="cpu",
        )

    def critic_head(self, agent):
        return mutation_utils.head_entry_layers(agent.critic.head_net)[0]

    def test_shared_critic_head_is_faded_when_the_policy_latent_is_reset(self):
        # The critic borrows the encoder, so it inherits the reset via
        # the mutation hook and its head must be faded to match.
        agent = self.make_agent(share=True)
        _policy_latent_dormant(agent)
        before = self.critic_head(agent).weight.data.clone()

        agent = _regrama_mutations().parameter_mutation(agent)

        after = self.critic_head(agent).weight.data
        assert after[:, 0].norm() < 0.1 * before[:, 0].norm()
        assert torch.equal(after[:, 1:], before[:, 1:])

    def test_unshared_critic_head_is_untouched_by_the_policy_pass(self):
        # A critic that owns its encoder compensates its own resets, so the
        # policy's pass must not reach across into it at all.
        agent = self.make_agent(share=False)
        _policy_latent_dormant(agent)
        before = self.critic_head(agent).weight.data.clone()

        agent = _regrama_mutations().parameter_mutation(agent)

        assert torch.equal(self.critic_head(agent).weight.data, before)


class TestMutationsRegramaMultiAgent:
    """Route each sub-policy's snapshot to its own network."""

    def make_agent(self):
        return IPPO(
            generate_multi_agent_box_spaces(2, (4,)),
            generate_multi_agent_discrete_spaces(2, 2),
            agent_ids=["agent_0", "other_0"],
            device="cpu",
        )

    def policies(self, agent):
        return getattr(agent, agent.registry.policy())

    def test_every_sub_policy_is_reset_from_its_own_entry(self):
        agent = self.make_agent()
        _all_dormant(agent)
        for module in self.policies(agent).values():
            _pin_biases(module)

        agent = _regrama_mutations().parameter_mutation(agent)

        # A changed state dict would prove nothing here: the Gaussian pass runs
        # on every sub-policy regardless of whether ReGraMa reset anything.
        for module in self.policies(agent).values():
            assert _zeroed_biases(module) > 0

    def test_a_sub_policy_left_healthy_is_not_reset(self):
        agent = self.make_agent()
        agent.grama_scores = _dormant_for(agent, {"agent_0"})
        for module in self.policies(agent).values():
            _pin_biases(module)

        agent = _regrama_mutations().parameter_mutation(agent)

        policies = self.policies(agent)
        assert _zeroed_biases(policies["agent_0"]) > 0
        assert _zeroed_biases(policies["other_0"]) == 0
