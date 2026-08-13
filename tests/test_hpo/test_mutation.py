import copy
import gc
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DeepSpeedPlugin
from gymnasium import spaces

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

if HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig
from agilerl.hpo.mutation import (
    MutationError,
    Mutations,
    get_exp_layer,
    get_offspring_eval_modules,
    set_global_seed,
)
from agilerl.modules import EvolvableBERT, EvolvableModule, ModuleDict
from agilerl.utils.utils import create_population
from agilerl.wrappers.agent import AsyncAgentsWrapper, RSNorm
from tests.helper_functions import (
    assert_state_dicts_equal,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_random_box_space,
)

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


@pytest.fixture(scope="function")
def bert_network(device):
    return create_bert_network(device)


@pytest.fixture(scope="function")
def bert_networks_multi_agent(device):
    return create_bert_networks_multi_agent(device)


@pytest.fixture(scope="function")
def bert_matd3_critic_networks(device):
    return [
        create_bert_networks_multi_agent(device),
        create_bert_networks_multi_agent(device),
    ]


@pytest.fixture(scope="function")
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
            mutation_sd,
            activation_selection,
            mutate_elite,
            rand_seed,
            device,
            accelerator,
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

    def test_scrubs_nonfinite_weights(self):
        # A diverged agent can reach parameter mutation carrying non-finite
        # weights while still scoring a finite fitness (a stray NaN on an
        # inactive path need not corrupt the rollout), so it slips past the
        # fitness-level tournament guard. abs(NaN) fails torch.normal's
        # ``std >= 0`` check, which used to abort the whole run. The operator
        # must scrub the weights in place and return finite parameters.
        class WeightModule(EvolvableModule):
            def __init__(self):
                super().__init__(device="cpu")
                self.w = torch.nn.Parameter(torch.full((8, 8), float("nan")))

            def forward(self, x):
                return x

            def recreate_network(self):
                pass

        muts = Mutations(0, 0, 0.5, 1, 0, 0, 0.1, device="cpu", rand_seed=0)
        mod = WeightModule()

        out = muts._gaussian_parameter_mutation(mod)

        assert out is mod
        assert torch.isfinite(mod.w).all()


class TestMutationsArchitectureMutateSingle:
    @pytest.mark.gpu
    def test_no_methods_sets_none(self, monkeypatch, device):
        class DummyPolicy:
            mutation_methods = []

        class DummyIndividual:
            def __init__(self):
                self.mut = None

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        individual = DummyIndividual()
        monkeypatch.setattr(
            "agilerl.hpo.mutation.get_offspring_eval_modules",
            lambda _ind: ({"actor": DummyPolicy()}, {}),
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
            mutation_methods = []

        class DummyIndividual:
            def __init__(self):
                self.mut = None

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        individual = DummyIndividual()
        monkeypatch.setattr(
            "agilerl.hpo.mutation.get_offspring_eval_modules",
            lambda _ind: ({"actors": DummyPolicy()}, {}),
        )
        with pytest.warns(
            UserWarning, match="No mutation methods found for the policy network"
        ):
            out = muts._architecture_mutate_multi(individual)
        assert out.mut == "None"

    @pytest.mark.gpu
    def test_none_applied_mutation_branch(self, monkeypatch, device):
        class DummySubmodule:
            mutation_methods = ["add_node"]

        class DummyPolicyDict(dict):
            mutation_methods = ["agent_0.add_node", "agent_1.add_node"]

            def sample_mutation_method(self, *_args, **_kwargs):
                return "agent_0.add_node"

        class DummyIndividual:
            def mutation_hook(self):
                return None

            def reinit_optimizers(self):
                return None

        policy = DummyPolicyDict(
            {"agent_0": DummySubmodule(), "agent_1": DummySubmodule()}
        )
        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        monkeypatch.setattr(
            "agilerl.hpo.mutation.get_offspring_eval_modules",
            lambda _ind: ({"actors": policy}, {}),
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
        class DummyEval:
            mutation_methods = ["agent_9.other_mut"]
            last_mutation_attr = None

        class DummyPolicy(dict):
            mutation_methods = ["agent_0.add_node"]

            def sample_mutation_method(self, *_args, **_kwargs):
                return "agent_0.add_node"

        class DummyIndividual:
            def mutation_hook(self):
                return None

            def reinit_optimizers(self):
                return None

        policy = DummyPolicy({"agent_0": DummyEval()})
        evals = {"critics": {"agent_0": DummyEval()}}
        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        monkeypatch.setattr(
            "agilerl.hpo.mutation.get_offspring_eval_modules",
            lambda _ind: ({"actors": policy}, evals),
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

        class OldLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.Parameter(torch.ones(2, 2))  # 4
                self.w2 = torch.nn.Parameter(torch.ones(2))  # 2
                self.only_old = torch.nn.Parameter(torch.ones(1))  # 1

        class NewLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.Parameter(torch.ones(1, 2))  # smaller than old w1
                self.w2 = torch.nn.Parameter(torch.ones(4))  # bigger than old w2
                self.only_new = torch.nn.Parameter(torch.ones(2))  # key absent in old

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


class TestMutationsMutation:
    # Checks no mutations if all probabilities set to zero
    @pytest.mark.gpu
    @pytest.mark.parametrize("algo", ["DQN"])
    @pytest.mark.parametrize(
        "observation_space, net_config",
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
        "algo, hp_config, action_space",
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
        "observation_space, net_config",
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
        "algo, action_space",
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
        "observation_space, net_config",
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
        "algo, action_space",
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
        "observation_space, net_config",
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
        "algo, hp_config, action_space",
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
        "observation_space, net_config",
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
        "algo, action_space",
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
        "observation_space, net_config",
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
        "observation_space, net_config",
        [
            ("vector_space", "encoder_mlp_config"),
            ("image_space", "encoder_cnn_config"),
            ("dict_space", "encoder_multi_input_config"),
            ("discrete_space", "encoder_mlp_config"),
        ],
    )
    @pytest.mark.parametrize("algo, action_space", [("DDPG", "vector_space")])
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
        "algo, action_space, wrapper_cls",
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
        "observation_space, net_config",
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
        "observation_space, net_config",
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
        "observation_space, net_config",
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
        "algo, hp_config",
        [
            ("MADDPG", "ac_hp_config"),
            ("MATD3", "ac_hp_config"),
            ("IPPO", "default_hp_config"),
        ],
    )
    @pytest.mark.parametrize(
        "observation_space, net_config",
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
        "observation_space, net_config",
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
        "observation_space, net_config",
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
        "algo, wrapper_cls",
        [
            ("MADDPG", None),
            ("MATD3", None),
            ("IPPO", None),
            ("IPPO", AsyncAgentsWrapper),
        ],
    )
    @pytest.mark.parametrize(
        "observation_space, net_config",
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
        "use_accelerator, use_deepspeed_optimizer",
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
        # Imported here, not at module scope: test_grpo importorskips deepspeed and
        # vllm, so a top-level import would skip this whole module -- including every
        # non-LLM mutation test -- on any machine without them.
        from tests.test_algorithms.test_llms.test_grpo import create_module

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
        not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed"
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
        # See the note in test_applies_rl_hp_mutation_llm_algorithm: this import
        # must stay function-local or it skips the entire module.
        from tests.test_algorithms.test_llms.test_grpo import create_module

        pre_training_mut = False
        init_hp = {
            "PAD_TOKEN_ID": 1000 - 1,
            "PAD_TOKEN": "<pad>",
            "BATCH_SIZE": 2,
            "BETA": 0.001,
            "LR": 5e-7,
            "MAX_GRAD_NORM": 0.1,
            "UPDATE_EPOCHS": 1,
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
            with pytest.warns(UserWarning):
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
        "algo, action_space, wrapper_cls",
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
        "observation_space, net_config",
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
        "algo, actor_network, critic_network",
        [("DDPG", "bert_network", "bert_network")],
    )
    @pytest.mark.parametrize(
        "observation_space, net_config",
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
        "algo, wrapper_cls",
        [
            ("MADDPG", None),
            ("MATD3", None),
            ("IPPO", None),
            ("IPPO", AsyncAgentsWrapper),
        ],
    )
    @pytest.mark.parametrize(
        "observation_space, net_config",
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
        "algo, actor_network, critic_network",
        [
            ("MADDPG", "bert_networks_multi_agent", "bert_networks_multi_agent"),
            ("MATD3", "bert_networks_multi_agent", "bert_matd3_critic_networks"),
        ],
    )
    @pytest.mark.parametrize(
        "observation_space, net_config",
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
    @pytest.mark.parametrize("algo", ["PPO", "DDPG", "TD3"])
    def test_warns_for_policy_gradient_algos(
        self, algo, vector_space, encoder_mlp_config, device
    ):
        from agilerl.utils.utils import create_population

        action_space = (
            generate_random_box_space((2,))
            if algo in ("DDPG", "TD3")
            else generate_discrete_space(2)
        )
        pop = create_population(
            algo=algo,
            observation_space=vector_space,
            action_space=action_space,
            net_config=encoder_mlp_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=1,
            device=device,
        )
        muts = Mutations(0, 0, 0, 0, 1, 0, 0.1, device=device)
        with pytest.warns(UserWarning, match="Activation mutations are not supported"):
            out = muts.activation_mutation(pop[0].clone(wrap=False))
        assert out.mut == "None"


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
        opts, proba = muts._get_mutations_options(pretraining=pretraining)
        assert len(opts) >= 1
        assert muts.no_mutation in opts
        if pretraining:
            assert sum(1 for p in proba if p == 1.0) >= 0

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
            TypeError, match="Bandit algorithm architecture.*not supported"
        ):
            get_exp_layer(torch.nn.Linear(2, 2))

    def test_returns_output_layer_for_evolvable_module(
        self, vector_space, discrete_space, encoder_mlp_config
    ):
        from agilerl.utils.utils import create_population

        pop = create_population(
            algo="NeuralUCB",
            observation_space=vector_space,
            action_space=discrete_space,
            net_config=encoder_mlp_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=1,
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


def test_get_offspring_eval_modules_returns_policy_and_modules(
    vector_space, discrete_space, encoder_mlp_config
):
    from agilerl.utils.utils import create_population

    pop = create_population(
        algo="DQN",
        observation_space=vector_space,
        action_space=discrete_space,
        net_config=encoder_mlp_config,
        INIT_HP=SHARED_INIT_HP,
        population_size=1,
        device="cpu",
    )
    policy, offspring_evals = get_offspring_eval_modules(pop[0])
    assert isinstance(policy, dict)
    assert isinstance(offspring_evals, dict)
    assert len(policy) >= 1


# --------------------------------------------------------------------------- #
# ReBorn parameter mutation (Qin et al.)                                       #
# --------------------------------------------------------------------------- #
def _make_reborn_mutations(seed=42, param_mut_type="reborn", reborn_out_scale=0.25):
    return Mutations(
        no_mutation=0.0,
        architecture=0.0,
        new_layer_prob=0.0,
        parameters=1.0,
        activation=0.0,
        rl_hp=0.0,
        param_mut_type=param_mut_type,
        dormant_tau=0.1,
        overact_beta=3.0,
        reborn_out_scale=reborn_out_scale,
        rand_seed=seed,
        device="cpu",
    )


def _healthy_fill(n):
    """Uniform per-neuron gradient scores -> no dormant / over-active neurons."""
    return torch.ones(n)


def _surgery_fill(n):
    """Inject an over-active neuron (0) and two dormant neurons (1, 2)."""
    t = torch.ones(n)
    if n >= 3:
        t[0] = 10.0
        t[1] = 0.0
        t[2] = 0.0
    return t


def _grama_snapshot(agent, obs, fill=_healthy_fill):
    """Build a synthetic ``_grama_scores`` snapshot for *agent*.

    Discovers each measured activation's neuron count with a throwaway forward pass
    (forward hooks capture output shapes), fills each with *fill*, and stores the
    result as ``agent._grama_scores`` in the exact structure the gradient-based
    dormant diagnostic / ReBorn consume: one list per :func:`_eval_networks`
    network, each aligned to :func:`_target_activations` order.
    """
    from agilerl.utils.dormant_neurons import _eval_networks, _target_activations

    processed = agent.preprocess_observation(obs)
    scores = []
    for _nid, net in _eval_networks(agent):
        targets = _target_activations(net)
        outputs = {}
        handles = [
            m.register_forward_hook(
                lambda mod, inp, out, k=k: outputs.__setitem__(k, out)
            )
            for k, m in enumerate(targets)
        ]
        net.eval()
        with torch.no_grad():
            net(processed)
        for h in handles:
            h.remove()
        net_scores = []
        for k in range(len(targets)):
            o = outputs.get(k)
            if not isinstance(o, torch.Tensor):
                net_scores.append(None)
                continue
            n = o.shape[1] if o.dim() > 1 else o.numel()
            net_scores.append(fill(n))
        scores.append(net_scores)
    agent._grama_scores = scores
    return scores


class TestRebornConstructorValidation:
    def test_rejects_bad_param_mut_type(self):
        with pytest.raises(AssertionError):
            _make_reborn_mutations(param_mut_type="bogus")

    def test_rejects_non_positive_dormant_tau(self):
        with pytest.raises(AssertionError):
            Mutations(
                no_mutation=0.0,
                architecture=0.0,
                new_layer_prob=0.0,
                parameters=1.0,
                activation=0.0,
                rl_hp=0.0,
                param_mut_type="reborn",
                dormant_tau=0.0,
                overact_beta=3.0,
            )

    def test_rejects_overact_beta_below_dormant_tau(self):
        with pytest.raises(AssertionError):
            Mutations(
                no_mutation=0.0,
                architecture=0.0,
                new_layer_prob=0.0,
                parameters=1.0,
                activation=0.0,
                rl_hp=0.0,
                param_mut_type="reborn",
                dormant_tau=0.5,
                overact_beta=0.4,
            )


class TestRebornLayerSurgery:
    """Directly exercise the per-layer surgery on hand-built Linear layers."""

    def _setup(self):
        torch.manual_seed(0)
        producer = torch.nn.Linear(3, 5)
        next_layer = torch.nn.Linear(5, 2)
        # Give distinct, non-zero rows so proportionality checks are meaningful.
        with torch.no_grad():
            producer.weight.copy_(
                torch.arange(1, 16, dtype=torch.float32).reshape(5, 3)
            )
            producer.bias.copy_(torch.arange(1, 6, dtype=torch.float32))
            next_layer.weight.copy_(
                torch.arange(1, 11, dtype=torch.float32).reshape(2, 5)
            )
        # neuron 0 over-active (norm ~4.17), neurons 1-3 dormant (norm 0),
        # neuron 4 normal (norm ~0.83).
        per_neuron = torch.tensor([10.0, 0.0, 0.0, 0.0, 2.0])
        return producer, next_layer, per_neuron

    def test_counts_and_structure(self):
        producer, next_layer, per_neuron = self._setup()
        mut = _make_reborn_mutations(seed=7)
        w_in_x = producer.weight.data[0].clone()
        b_x = producer.bias.data[0].clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        mut._apply_reborn_to_layer(
            producer, [next_layer], None, None, per_neuron, counts
        )

        assert counts["overactive"] == 1
        assert counts["dormant"] == 3
        # Every dormant neuron is either reborn as a partner or Xavier-reset.
        assert counts["reborn"] + counts["xavier"] == 3
        assert counts["reborn"] >= 2  # M in [2, 5], 3 dormant available

        # Over-active row stays a positive scalar multiple of the original row.
        ratio = producer.weight.data[0] / w_in_x
        assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5)
        beta_0 = float(ratio[0])
        assert 0.5 <= beta_0 <= 1.5
        # Bias scaled by the same beta_0.
        assert torch.allclose(producer.bias.data[0], beta_0 * b_x, atol=1e-5)

        # Nothing is NaN/inf after surgery.
        assert torch.isfinite(producer.weight.data).all()
        assert torch.isfinite(next_layer.weight.data).all()

        # The normal neuron (index 4) is untouched.
        assert torch.allclose(
            producer.weight.data[4],
            torch.tensor([13.0, 14.0, 15.0]),
        )

    def test_unclaimed_dormant_outgoing_reseeded_at_scale(self):
        # A Xavier-reset neuron is revived with a small *non-zero* outgoing column
        # -- norm == reborn_out_scale * the median column norm of the neurons this
        # pass leaves alone. A zero column would make the revived neuron score as
        # maximally dormant under gradient detection (and freeze its own incoming
        # weights), which is the pathology reborn_out_scale exists to avoid.
        producer, next_layer, per_neuron = self._setup()
        mut = _make_reborn_mutations(seed=1, reborn_out_scale=0.25)
        # Only neuron 4 is neither dormant nor over-active, so it alone sets the
        # reference scale: column 4 of next_layer is [5, 10].
        expected = 0.25 * float(torch.tensor([5.0, 10.0]).norm())
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}
        mut._apply_reborn_to_layer(
            producer, [next_layer], None, None, per_neuron, counts
        )

        xavier_cols = [
            i
            for i in (1, 2, 3)
            if pytest.approx(expected, rel=1e-5)
            == float(next_layer.weight.data[:, i].norm())
        ]
        assert len(xavier_cols) == counts["xavier"]
        # No dormant neuron is left with a zero outgoing column.
        for i in (1, 2, 3):
            assert torch.count_nonzero(next_layer.weight.data[:, i]) > 0

    def test_zero_out_scale_restores_zeroed_outgoing(self):
        # The ablation path: reborn_out_scale=0.0 must reproduce ReDo's original
        # zero-outgoing revival exactly, so seeded comparisons stay meaningful.
        producer, next_layer, per_neuron = self._setup()
        mut = _make_reborn_mutations(seed=1, reborn_out_scale=0.0)
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}
        mut._apply_reborn_to_layer(
            producer, [next_layer], None, None, per_neuron, counts
        )
        zero_cols = sum(
            int(torch.count_nonzero(next_layer.weight.data[:, i]) == 0)
            for i in (1, 2, 3)
        )
        assert zero_cols == counts["xavier"]

    def test_revived_neuron_receives_incoming_gradient(self):
        # The point of the non-zero outgoing column: with W_out[:, i] == 0 the
        # chain rule gives dL/dW_in[i, :] = dL/dz_i * x = 0, so a revived neuron's
        # *incoming* weights are frozen until its outgoing ones bootstrap off zero.
        # Re-seeding them small unfreezes both, at the cost of exact function
        # preservation.
        for scale, expect_gradient in ((0.0, False), (0.25, True)):
            producer = torch.nn.Linear(3, 5)
            next_layer = torch.nn.Linear(5, 2)
            per_neuron = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])  # 1, 2 dormant
            mut = _make_reborn_mutations(seed=5, reborn_out_scale=scale)
            counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}
            mut._apply_reborn_to_layer(
                producer, [next_layer], None, None, per_neuron, counts
            )
            assert counts["xavier"] == 2  # no over-active neuron -> none claimed

            producer.zero_grad()
            X = torch.rand(8, 3)
            next_layer(torch.relu(producer(X))).square().mean().backward()
            revived_grad = producer.weight.grad[[1, 2]]
            assert bool(revived_grad.abs().sum() > 0) is expect_gradient

    def test_over_active_split_is_function_preserving_relu(self):
        # net2net invariant: splitting an over-active neuron into dormant slots
        # must leave the network's output unchanged for a ReLU network. Pinned to
        # reborn_out_scale=0.0 to isolate the split: the three dormant neurons have
        # exactly-zero activation on the batch, so with their outgoing weights
        # zeroed any that stay unclaimed also contribute zero. At the default scale
        # a revived neuron deliberately perturbs the output (see the sibling test).
        producer = torch.nn.Linear(2, 4)
        next_layer = torch.nn.Linear(4, 1)
        with torch.no_grad():
            producer.weight.copy_(
                torch.tensor([[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]])
            )
            # Large negative bias keeps neurons 1-3 dead for any positive input.
            producer.bias.copy_(torch.tensor([0.5, -5.0, -5.0, -5.0]))
            next_layer.weight.copy_(torch.tensor([[2.0, 3.0, -1.0, 0.5]]))
            next_layer.bias.copy_(torch.tensor([0.7]))

        X = torch.tensor([[0.1, 0.2], [0.7, 0.3], [1.0, 0.0], [0.4, 0.9]])

        def forward(x):
            return next_layer(torch.relu(producer(x)))

        # Neuron 0 fires; neurons 1-3 are dormant (exactly-zero activation).
        hidden = torch.relu(producer(X))
        assert torch.all(hidden[:, 0] > 0)
        assert torch.all(hidden[:, 1:] == 0)

        before = forward(X).clone()

        mut = _make_reborn_mutations(seed=3, reborn_out_scale=0.0)
        # norm = [4, 0, 0, 0]: neuron 0 over-active (>=3), neurons 1-3 dormant.
        per_neuron = torch.tensor([4.0, 0.0, 0.0, 0.0])
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}
        mut._apply_reborn_to_layer(
            producer, [next_layer], None, None, per_neuron, counts
        )
        assert counts["overactive"] == 1
        assert counts["dormant"] == 3
        assert counts["reborn"] + counts["xavier"] == 3

        after = forward(X)
        assert torch.isfinite(after).all()
        assert torch.allclose(after, before, atol=1e-5), (before, after)


class TestRebornBranchedArchitectures:
    """Networks whose layers do not lie in one flat ``nn.Sequential``.

    ``EvolvableMultiInput`` keeps its sub-networks in a ``ModuleDict`` with a bare
    ``final_dense``/``output`` tail, and a duelling Q-network's ``head_net`` holds
    two sibling streams. Both must still be recycled: a resolver that unwraps to a
    single sequential silently skips them, leaving measured layers untouched.
    """

    @staticmethod
    def _dict_obs_dqn(n_image=0, n_vector=2):
        """DQN over a ``Dict`` space -- i.e. an ``EvolvableMultiInput`` encoder.

        *n_image* controls how many subspaces get their own nested CNN sub-encoder;
        vector subspaces are concatenated raw into the fusion layer.
        """
        from gymnasium import spaces

        from agilerl.algorithms import DQN

        obs = generate_dict_or_tuple_space(n_image, n_vector, dict_space=True)
        return DQN(obs, spaces.Discrete(3), device="cpu")

    @staticmethod
    def _dict_obs(agent, batch_size=8):
        """A deterministic observation batch matching *agent*'s ``Dict`` space."""
        batch = {}
        for key, space in agent.observation_space.spaces.items():
            space.seed(0)
            batch[key] = np.stack([space.sample() for _ in range(batch_size)])
        return batch

    @staticmethod
    def _duelling_dqn():
        from gymnasium import spaces

        from agilerl.algorithms import RainbowDQN

        obs = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        return RainbowDQN(obs, spaces.Discrete(4), device="cpu")

    @staticmethod
    def _dormant_per_measured_layer(agent):
        """Dormant neurons ``_surgery_fill`` injects across all measured layers."""
        from agilerl.utils.dormant_neurons import _eval_networks, _target_activations

        return 2 * sum(
            len(_target_activations(net)) for _nid, net in _eval_networks(agent)
        )

    def test_multi_input_encoder_is_recycled(self):
        # Arrange
        agent = self._dict_obs_dqn()
        obs = self._dict_obs(agent)
        scores = _grama_snapshot(agent, obs, fill=_surgery_fill)
        expected = self._dormant_per_measured_layer(agent)

        # Act
        out = _make_reborn_mutations().reborn_parameter_mutation(agent, scores)

        # Assert
        assert out.mut_details["dormant_count"] == expected
        assert all(torch.isfinite(v).all() for v in out.actor.state_dict().values())

    def test_duelling_head_advantage_stream_is_recycled(self):
        # Arrange
        agent = self._duelling_dqn()
        obs = np.random.RandomState(0).uniform(-1, 1, (32, 8)).astype(np.float32)
        scores = _grama_snapshot(agent, obs, fill=_surgery_fill)
        expected = self._dormant_per_measured_layer(agent)

        # Act
        out = _make_reborn_mutations().reborn_parameter_mutation(agent, scores)

        # Assert
        assert out.mut_details["dormant_count"] == expected
        assert all(torch.isfinite(v).all() for v in out.actor.state_dict().values())

    @staticmethod
    def _scores_isolating_nested_sub_encoders(agent, obs):
        """Snapshot where only the nested sub-encoders' layers are unhealthy.

        Every activation outside ``encoder.feature_net`` -- crucially the encoder's
        own latent activation, the one that really does feed the head -- is left
        healthy, so any weight change in the head can only come from a sub-encoder's
        surgery reaching a layer it does not feed.
        """
        from agilerl.utils.dormant_neurons import _eval_networks, _target_activations

        scores = _grama_snapshot(agent, obs, fill=_healthy_fill)
        for net_idx, (_nid, net) in enumerate(_eval_networks(agent)):
            # ``named_modules``, not ``modules``: the latter is overridden on
            # ``EvolvableModule`` to yield mutation-group *names*.
            nested = {id(m) for _n, m in net.encoder.feature_net.named_modules()}
            for k, act in enumerate(_target_activations(net)):
                if id(act) in nested and scores[net_idx][k] is not None:
                    scores[net_idx][k] = _surgery_fill(scores[net_idx][k].numel())
        agent._grama_scores = scores
        return scores

    def test_nested_sub_encoder_never_writes_into_the_head(self):
        """A sub-encoder's neurons are consumed by ``final_dense``, not by the head.

        With several observation subspaces its features are only a *slice* of the
        concatenated fusion input, so no whole-column mapping exists and the layer
        must be skipped -- never rewritten against the head's unrelated columns.
        """
        # Arrange -- one image subspace (its own CNN sub-encoder) plus a vector one
        agent = self._dict_obs_dqn(n_image=1, n_vector=1)
        obs = self._dict_obs(agent)
        scores = self._scores_isolating_nested_sub_encoders(agent, obs)
        head_first = agent.actor.head_net.model[0]
        before = head_first.weight.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act -- surgery only, so the trailing Gaussian pass cannot mask the result
        _make_reborn_mutations()._reborn_network_surgery(agent.actor, scores[0], counts)

        # Assert
        assert torch.equal(before, head_first.weight.detach())

    def test_sole_sub_encoder_recycling_updates_the_fusion_layer(self):
        """With a single subspace the fusion layer's columns *are* its neurons.

        The whole-column mapping is then unambiguous, so the surgery applies -- to
        ``final_dense``, still never to the head.
        """
        # Arrange
        agent = self._dict_obs_dqn(n_image=1, n_vector=0)
        obs = self._dict_obs(agent)
        scores = self._scores_isolating_nested_sub_encoders(agent, obs)
        fusion = agent.actor.encoder.final_dense
        head_first = agent.actor.head_net.model[0]
        fusion_before = fusion.weight.detach().clone()
        head_before = head_first.weight.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act
        _make_reborn_mutations()._reborn_network_surgery(agent.actor, scores[0], counts)

        # Assert
        assert not torch.equal(fusion_before, fusion.weight.detach())
        assert torch.equal(head_before, head_first.weight.detach())

    @staticmethod
    def _scores_isolating_the_conv_to_dense_boundary(agent, obs):
        """Snapshot where only the conv -> flatten -> dense activation is unhealthy.

        Located structurally rather than by name: it is the measured activation
        whose producer is a convolution and whose consumer is a dense layer, i.e.
        the one whose recycling needs the flattened column stride.
        """
        from agilerl.utils.dormant_neurons import _eval_networks, _target_activations

        resolver = _make_reborn_mutations()
        scores = _grama_snapshot(agent, obs, fill=_healthy_fill)
        for net_idx, (_nid, net) in enumerate(_eval_networks(agent)):
            for k, act in enumerate(_target_activations(net)):
                if scores[net_idx][k] is None:
                    continue
                producer, consumers, _is_encoder = resolver._resolve_producer_and_next(
                    act, getattr(net, "encoder", None), getattr(net, "head_net", None)
                )
                if producer is None or not consumers:
                    continue
                if any(
                    resolver._boundary_kind(producer, consumer) == "conv_dense"
                    for consumer in consumers
                ):
                    scores[net_idx][k] = _surgery_fill(scores[net_idx][k].numel())
        agent._grama_scores = scores
        return scores

    def test_nested_cnn_conv_to_dense_boundary_is_recycled(self):
        """``cnn_output_size`` lives on the nested CNN, not on the fusion encoder.

        A conv -> flatten -> dense consumer spends ``H*W`` adjacent columns per
        feature map, and the stride comes from the ``EvolvableCNN`` that owns the
        conv stack. Under ``EvolvableMultiInput`` that is a ``feature_net`` entry
        rather than the encoder itself, so resolving the dims from the encoder
        alone drops the consumer and leaves every nested CNN's last conv layer
        silently unrecycled.
        """
        # Arrange -- only the conv -> dense activation is scored as unhealthy, so
        # the dense layer can only change through that boundary being handled.
        agent = self._dict_obs_dqn(n_image=1, n_vector=0)
        obs = self._dict_obs(agent)
        scores = self._scores_isolating_the_conv_to_dense_boundary(agent, obs)
        sub_encoder = next(iter(agent.actor.encoder.feature_net.values()))
        linear = next(
            module
            for _name, module in sub_encoder.named_modules()
            if isinstance(module, torch.nn.Linear)
        )
        before = linear.weight.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act -- surgery only, so the trailing Gaussian pass cannot mask the result
        _make_reborn_mutations()._reborn_network_surgery(agent.actor, scores[0], counts)

        # Assert
        assert not torch.equal(before, linear.weight.detach())
        assert counts["dormant"] > 0

    def test_plain_cnn_conv_to_dense_boundary_is_recycled(self):
        """The same boundary in a top-level ``EvolvableCNN`` must keep working."""
        # Arrange
        from gymnasium import spaces

        from agilerl.algorithms import DQN

        obs_space = spaces.Box(0, 255, shape=(3, 32, 32), dtype=np.uint8)
        agent = DQN(obs_space, spaces.Discrete(3), device="cpu")
        obs = np.stack([obs_space.sample() for _ in range(8)])
        scores = self._scores_isolating_the_conv_to_dense_boundary(agent, obs)
        linear = agent.actor.encoder.model.encoder_linear_output
        before = linear.weight.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act
        _make_reborn_mutations()._reborn_network_surgery(agent.actor, scores[0], counts)

        # Assert
        assert not torch.equal(before, linear.weight.detach())
        assert counts["dormant"] > 0

    def test_latent_outgoing_weights_updated_in_every_duelling_stream(self):
        """The encoder latent feeds both streams, so both hold outgoing weights.

        Rewriting only the stream the resolver happens to find would leave the
        other one indexing neurons that have been repurposed.
        """
        # Arrange
        agent = self._duelling_dqn()
        obs = np.random.RandomState(0).uniform(-1, 1, (32, 8)).astype(np.float32)
        scores = _grama_snapshot(agent, obs, fill=_surgery_fill)
        advantage_first = agent.actor.head_net.advantage_net[0]
        before = advantage_first.weight_mu.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act -- surgery only, so the trailing Gaussian pass cannot mask the result
        _make_reborn_mutations()._reborn_network_surgery(agent.actor, scores[0], counts)

        # Assert -- the dormant latent columns (indices 1, 2) were rewritten
        after = advantage_first.weight_mu.detach()
        assert not torch.equal(before[:, 1:3], after[:, 1:3])


class TestRebornEndToEnd:
    def _ppo(self):
        from agilerl.algorithms import PPO
        from gymnasium import spaces

        obs = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        act = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        cfg = {
            "encoder_config": {
                "hidden_size": [16],
                "activation": "ReLU",
                "output_activation": "Identity",
            },
            "head_config": {"hidden_size": [16], "activation": "ReLU"},
        }
        return PPO(obs, act, net_config=cfg, device="cpu")

    def _dqn(self):
        from agilerl.algorithms import DQN
        from gymnasium import spaces

        obs = spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8)
        act = spaces.Discrete(6)
        cfg = {
            "encoder_config": {
                "channel_size": [16, 32],
                "kernel_size": [8, 4],
                "stride_size": [4, 2],
                "activation": "ReLU",
            },
            "head_config": {"hidden_size": [32], "activation": "ReLU"},
        }
        return DQN(obs, act, net_config=cfg, device="cpu")

    def test_ppo_mlp_runs(self):
        # Traverses the MLP encoder -> head boundary. Even when the (healthy)
        # network has no dormant / over-active neurons so the ReBorn surgery is a
        # no-op, the trailing reset + ordinary Gaussian pass still runs, so the
        # operator records both the neuron-recycling counts and the weight-noise
        # counts. (The surgery math is covered by TestRebornLayerSurgery.) The
        # amplified ("super") noise band is never applied under ReBorn.
        agent = self._ppo()
        obs = np.random.RandomState(0).uniform(-1, 1, size=(32, 6)).astype(np.float32)
        scores = _grama_snapshot(agent, obs)
        out = _make_reborn_mutations().reborn_parameter_mutation(agent, scores)
        assert out.mut == "param_reborn"
        assert out.mut_details["category"] == "reborn"
        assert set(out.mut_details) >= {
            "neurons_reborn",
            "neurons_xavier_reset",
            "overactive_count",
            "dormant_count",
            "weights_reset",
            "weights_ordinary_noise",
            "weights_amplified_noise",
        }
        # ReBorn never fires the divergence-prone amplified noise band.
        assert out.mut_details["weights_amplified_noise"] == 0
        assert all(torch.isfinite(v).all() for v in out.actor.state_dict().values())

    def test_surgery_triggers_on_injected_scores(self):
        # An injected snapshot with dormant + over-active neurons drives real
        # recycling: neurons are detected and reborn / Xavier-reset, weights finite.
        agent = self._ppo()
        obs = np.random.RandomState(0).uniform(-1, 1, size=(32, 6)).astype(np.float32)
        scores = _grama_snapshot(agent, obs, fill=_surgery_fill)
        out = _make_reborn_mutations().reborn_parameter_mutation(agent, scores)
        assert out.mut_details["dormant_count"] > 0
        assert out.mut_details["overactive_count"] > 0
        assert (
            out.mut_details["neurons_reborn"] + out.mut_details["neurons_xavier_reset"]
            > 0
        )
        assert all(torch.isfinite(v).all() for v in out.actor.state_dict().values())

    def test_dqn_cnn_runs_and_syncs_target(self):
        agent = self._dqn()
        obs = (
            np.random.RandomState(1)
            .randint(0, 255, size=(16, 4, 84, 84))
            .astype(np.uint8)
        )
        scores = _grama_snapshot(agent, obs)
        out = _make_reborn_mutations().reborn_parameter_mutation(agent, scores)
        assert out.mut == "param_reborn"
        assert all(torch.isfinite(v).all() for v in out.actor.state_dict().values())
        # DQN target network is synced from the mutated online network.
        for k, v in out.actor.state_dict().items():
            assert torch.equal(v, out.actor_target.state_dict()[k])

    def test_trailing_noise_changes_policy_weights(self):
        # After the ReBorn surgery, the policy network receives the reset +
        # ordinary Gaussian noise pass, so its weights change even when the surgery
        # itself is a healthy no-op. The target is re-synced from the already-noised
        # online network (noise runs before the sync), so the two stay identical.
        agent = self._dqn()
        before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
        obs = (
            np.random.RandomState(1)
            .randint(0, 255, size=(16, 4, 84, 84))
            .astype(np.uint8)
        )
        scores = _grama_snapshot(agent, obs)
        out = _make_reborn_mutations().reborn_parameter_mutation(agent, scores)
        after = out.actor.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)
        for k, v in after.items():
            assert torch.equal(v, out.actor_target.state_dict()[k])

    def test_gaussian_toggle_excludes_amplified(self):
        # include_amplified=False must never populate the amplified band, while the
        # reset + ordinary bands still run. This is exactly how ReBorn calls the
        # Gaussian operator for its trailing pass.
        mut = _make_reborn_mutations(seed=5)
        agent = self._ppo()
        counts = {"reset": 0, "ordinary": 0, "amplified": 0}
        mut._gaussian_parameter_mutation(
            agent.actor, counts=counts, include_amplified=False
        )
        assert counts["amplified"] == 0
        assert counts["reset"] + counts["ordinary"] > 0

    def test_reproducible_across_equal_agents(self):
        import copy

        a = self._ppo()
        # Deep-copy so all weights match; both agents get an identical injected
        # gradient snapshot (same architecture -> same shapes -> same fill), so the
        # surgery *and* the trailing Gaussian pass must replay identically.
        b = copy.deepcopy(a)
        obs = np.random.RandomState(2).uniform(-1, 1, size=(32, 6)).astype(np.float32)
        scores_a = _grama_snapshot(a, obs, fill=_surgery_fill)
        scores_b = _grama_snapshot(b, obs, fill=_surgery_fill)
        # ReBorn draws from the global numpy RNG (surgery) and the global torch RNG
        # (the trailing Gaussian noise pass), so pin both identically before each
        # call. In a real run seed_everything pins these once and the whole mutation
        # sequence replays deterministically; here we isolate the two calls.
        np.random.seed(0)
        torch.manual_seed(0)
        ra = _make_reborn_mutations(seed=123).reborn_parameter_mutation(a, scores_a)
        np.random.seed(0)
        torch.manual_seed(0)
        rb = _make_reborn_mutations(seed=123).reborn_parameter_mutation(b, scores_b)
        for k in ra.actor.state_dict():
            assert torch.equal(ra.actor.state_dict()[k], rb.actor.state_dict()[k])

    def test_falls_back_to_gaussian_without_snapshot(self):
        # param_mut_type='reborn' but no gradient snapshot -> Gaussian mutation.
        agent = self._ppo()
        mut = _make_reborn_mutations()
        assert mut._grama_side_table is None
        out = mut.parameter_mutation(agent)
        assert out.mut == "param"  # not "param_reborn"

    def test_dispatch_uses_reborn_with_snapshot(self):
        # The full mutation() dispatch looks up each child's parent gradient
        # snapshot (by ``_parent_index``) and routes the parameter mutation
        # through ReBorn.
        pop = [self._ppo(), self._ppo()]
        obs = np.zeros((4, 6), dtype=np.float32)
        side = {}
        for i, ag in enumerate(pop):
            ag._parent_index = i
            side[i] = _grama_snapshot(ag, obs)
        mut = Mutations(
            no_mutation=0.0,
            architecture=0.0,
            new_layer_prob=0.0,
            parameters=1.0,
            activation=0.0,
            rl_hp=0.0,
            param_mut_type="reborn",
            dormant_tau=0.1,
            overact_beta=3.0,
            mutate_elite=True,
            rand_seed=0,
            device="cpu",
        )
        out = mut.mutation(pop, grama_scores=side)
        assert all(agent.mut == "param_reborn" for agent in out)
        # The snapshot table is released after the call.
        assert mut._grama_side_table is None

    def test_dispatch_falls_back_without_snapshot(self):
        pop = [self._ppo(), self._ppo()]
        mut = Mutations(
            no_mutation=0.0,
            architecture=0.0,
            new_layer_prob=0.0,
            parameters=1.0,
            activation=0.0,
            rl_hp=0.0,
            param_mut_type="reborn",
            mutate_elite=True,
            rand_seed=0,
            device="cpu",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = mut.mutation(pop)  # no snapshot -> Gaussian fallback
        assert all(agent.mut == "param" for agent in out)

    def _reborn_dispatch_mut(self):
        return Mutations(
            no_mutation=0.0,
            architecture=0.0,
            new_layer_prob=0.0,
            parameters=1.0,
            activation=0.0,
            rl_hp=0.0,
            param_mut_type="reborn",
            dormant_tau=0.1,
            overact_beta=3.0,
            mutate_elite=True,
            rand_seed=0,
            device="cpu",
        )

    def test_eval_cycle_warns_when_reborn_configured_without_snapshot(self):
        # A reborn regime whose trainer does not thread the gradient snapshot
        # silently falls back to Gaussian, misattributing results; surface it.
        pop = [self._ppo(), self._ppo()]
        mut = self._reborn_dispatch_mut()
        with pytest.warns(UserWarning, match="falling back to the Gaussian"):
            mut.mutation(pop)  # eval-cycle mutation, no snapshot

    def test_no_warning_with_snapshot(self):
        pop = [self._ppo(), self._ppo()]
        obs = np.zeros((4, 6), dtype=np.float32)
        side = {}
        for i, ag in enumerate(pop):
            ag._parent_index = i
            side[i] = _grama_snapshot(ag, obs)
        mut = self._reborn_dispatch_mut()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            mut.mutation(pop, grama_scores=side)
        assert not [
            w for w in record if "falling back to the Gaussian" in str(w.message)
        ]

    def test_pretraining_mutation_does_not_warn_without_snapshot(self):
        # The pre-training mutation step is expected to run env-less; the reborn
        # fallback there is intended and must not warn.
        pop = [self._ppo(), self._ppo()]
        mut = self._reborn_dispatch_mut()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            mut.mutation(pop, pre_training_mut=True)
        assert not [
            w for w in record if "falling back to the Gaussian" in str(w.message)
        ]

    def test_missing_child_snapshot_warns_and_falls_back(self):
        # A snapshot table *was* supplied, but this child's parent is not in it, so
        # only that agent degrades to Gaussian. Silent degradation here is what made
        # a whole ReBorn benchmark record itself as a plain Gaussian run, so the
        # per-agent case must be as loud as the call-level one.
        pop = [self._ppo(), self._ppo()]
        obs = np.zeros((4, 6), dtype=np.float32)
        pop[0]._parent_index = 0
        pop[1]._parent_index = 99  # no snapshot under this key
        side = {0: _grama_snapshot(pop[0], obs)}
        mut = self._reborn_dispatch_mut()
        with pytest.warns(UserWarning, match="no gradient snapshot was found"):
            out = mut.mutation(pop, grama_scores=side)
        assert out[0].mut == "param_reborn"
        assert out[1].mut == "param"


class TestRebornWrappedAgents:
    """ReBorn must survive the :class:`AgentWrapper` indirection.

    The benchmarking harness wraps every PPO agent in :class:`RSNorm` before
    training, so the population handed to selection and mutation is wrappers rather
    than bare algorithms. ``_parent_index`` is the only channel by which a child
    finds its parent's gradient snapshot, and an assignment to a wrapper does not
    reach the agent it wraps -- so without the unwrap in
    :func:`~agilerl.hpo.tournament._record_parent_index` every wrapped agent
    silently falls back to the Gaussian operator.
    """

    def _population(self, size=2):
        from agilerl.algorithms import PPO

        obs = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        act = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        cfg = {
            "encoder_config": {
                "hidden_size": [16],
                "activation": "ReLU",
                "output_activation": "Identity",
            },
            "head_config": {"hidden_size": [16], "activation": "ReLU"},
        }
        pop = []
        for i in range(size):
            agent = PPO(obs, act, net_config=cfg, device="cpu")
            agent.index = i
            agent.fitness = [float(i)]
            pop.append(agent)
        return pop

    @staticmethod
    def _tournament(size):
        from agilerl.hpo.tournament import TournamentSelection

        return TournamentSelection(
            tournament_size=2, elitism=True, population_size=size
        )

    def test_tournament_tags_the_unwrapped_agent(self):
        wrapped = [RSNorm(agent) for agent in self._population()]
        _elite, children = self._tournament(len(wrapped)).select(wrapped)
        for child in children:
            # Must be in the *algorithm's* __dict__, not the wrapper's: the mutation
            # operator reads it after unwrapping.
            assert "_parent_index" in child.agent.__dict__

    def test_reborn_fires_for_wrapped_children(self):
        pop = self._population()
        obs = np.random.RandomState(0).uniform(-1, 1, size=(32, 6)).astype(np.float32)
        # Side table keyed by pre-tournament index, exactly as the trainers build it.
        side = {
            agent.index: _grama_snapshot(agent, obs, fill=_surgery_fill)
            for agent in pop
        }
        wrapped = [RSNorm(agent) for agent in pop]
        _elite, children = self._tournament(len(wrapped)).select(wrapped)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = Mutations(
                no_mutation=0.0,
                architecture=0.0,
                new_layer_prob=0.0,
                parameters=1.0,
                activation=0.0,
                rl_hp=0.0,
                param_mut_type="reborn",
                dormant_tau=0.1,
                overact_beta=3.0,
                mutate_elite=True,
                rand_seed=0,
                device="cpu",
            ).mutation(children, grama_scores=side)
        assert all(child.mut == "param_reborn" for child in out)
        assert not [
            w for w in record if "falling back to the Gaussian" in str(w.message)
        ]


class TestRebornBorrowedEncoderParameters:
    """A network that does not *own* its encoder weights must be left alone.

    :func:`~agilerl.utils.algo_utils.share_encoder_parameters` pins the critic's
    encoder to detached clones of the actor's, which ``mutation_hook`` refreshes
    right after every mutation. Recycling there writes rows that are then thrown
    away, while the matching column rewrite in the critic's *head* survives --
    leaving the head compensating a neuron split that no longer exists. A real run
    never reaches that state (the clones carry no gradient, so those layers are
    captured as ``None``), which is exactly why the guard has to be explicit.
    """

    @staticmethod
    def _shared_encoder_ppo():
        from gymnasium import spaces

        from agilerl.algorithms import PPO

        obs = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        act = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        cfg = {
            "encoder_config": {
                "hidden_size": [16],
                "activation": "ReLU",
                "output_activation": "Identity",
            },
            "head_config": {"hidden_size": [16], "activation": "ReLU"},
        }
        return PPO(obs, act, net_config=cfg, share_encoders=True, device="cpu")

    @staticmethod
    def _scores_isolating_the_encoder(agent, obs):
        """Snapshot where only the critic's borrowed encoder layers are unhealthy.

        Its head is left healthy, so any weight change there can only come from the
        encoder's surgery reaching the head's columns -- the half of the rewrite that
        would survive ``mutation_hook`` re-pinning the encoder.
        """
        from agilerl.utils.dormant_neurons import _eval_networks, _target_activations

        scores = _grama_snapshot(agent, obs, fill=_healthy_fill)
        for net_idx, (_nid, net) in enumerate(_eval_networks(agent)):
            # ``named_modules``, not ``modules``: the latter is overridden on
            # ``EvolvableModule`` to yield mutation-group *names*.
            encoder_modules = {id(m) for _n, m in net.encoder.named_modules()}
            for k, act in enumerate(_target_activations(net)):
                if id(act) in encoder_modules and scores[net_idx][k] is not None:
                    scores[net_idx][k] = _surgery_fill(scores[net_idx][k].numel())
        agent._grama_scores = scores
        return scores

    def test_borrowed_encoder_is_not_recycled(self):
        # Arrange -- a snapshot deliberately scoring the critic's borrowed encoder
        # copy, which a real run leaves as ``None`` for want of a gradient.
        agent = self._shared_encoder_ppo()
        obs = np.random.RandomState(0).uniform(-1, 1, size=(32, 6)).astype(np.float32)
        scores = self._scores_isolating_the_encoder(agent, obs)
        critic_encoder = agent.critic.encoder.model.shared_encoder_linear_layer_1
        critic_head_first = agent.critic.head_net.model.value_linear_layer_1
        encoder_before = critic_encoder.weight.detach().clone()
        head_before = critic_head_first.weight.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act -- surgery only, so the trailing Gaussian pass cannot mask the result
        _make_reborn_mutations()._reborn_network_surgery(
            agent.critic, scores[1], counts
        )

        # Assert -- neither the discarded rows nor the surviving columns are touched
        assert torch.equal(encoder_before, critic_encoder.weight.detach())
        assert torch.equal(head_before, critic_head_first.weight.detach())

    def test_owned_encoder_is_recycled(self):
        """The guard keys on ownership, not on being an encoder.

        Without ``share_encoders`` the critic owns its encoder outright, so the very
        same layers must still be recycled.
        """
        # Arrange
        from gymnasium import spaces

        from agilerl.algorithms import PPO

        obs_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        cfg = {
            "encoder_config": {
                "hidden_size": [16],
                "activation": "ReLU",
                "output_activation": "Identity",
            },
            "head_config": {"hidden_size": [16], "activation": "ReLU"},
        }
        agent = PPO(
            obs_space, act_space, net_config=cfg, share_encoders=False, device="cpu"
        )
        obs = np.random.RandomState(0).uniform(-1, 1, size=(32, 6)).astype(np.float32)
        scores = self._scores_isolating_the_encoder(agent, obs)
        critic_encoder = agent.critic.encoder.model.critic_encoder_linear_layer_1
        before = critic_encoder.weight.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act
        _make_reborn_mutations()._reborn_network_surgery(
            agent.critic, scores[1], counts
        )

        # Assert
        assert not torch.equal(before, critic_encoder.weight.detach())
        assert counts["dormant"] > 0

    def test_owned_head_is_still_recycled(self):
        """Only the *encoder* is borrowed -- the critic's own head must still run."""
        # Arrange
        agent = self._shared_encoder_ppo()
        obs = np.random.RandomState(0).uniform(-1, 1, size=(32, 6)).astype(np.float32)
        scores = _grama_snapshot(agent, obs, fill=_surgery_fill)
        head_output = agent.critic.head_net.model.value_linear_layer_output
        before = head_output.weight.detach().clone()
        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Act
        _make_reborn_mutations()._reborn_network_surgery(
            agent.critic, scores[1], counts
        )

        # Assert -- the head's hidden layer is owned, so its neurons are recycled
        assert not torch.equal(before, head_output.weight.detach())
        assert counts["dormant"] > 0
