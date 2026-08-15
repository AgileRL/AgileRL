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
    generate_discrete_space,
    generate_random_box_space,
)
from tests.test_algorithms.test_llms.test_grpo import create_module

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

    def test_accepts_zero_arch_dormant_tau(self):
        # tau=0 removes only exactly-dead units, i.e. exact preservation.
        muts = Mutations(0, 0, 0.5, 0, 0, 0, 0.1, device="cpu", arch_dormant_tau=0.0)
        assert muts.arch_dormant_tau == 0.0

    def test_raises_for_negative_arch_dormant_tau(self):
        with pytest.raises(AssertionError, match="arch_dormant_tau"):
            Mutations(0, 0, 0.5, 0, 0, 0, 0.1, device="cpu", arch_dormant_tau=-0.1)


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
# Function-preserving architecture mutations (arch_mut_type="func_preserving")
# --------------------------------------------------------------------------- #
import agilerl.hpo.function_preserving as fp  # noqa: E402
from agilerl.utils.dormant_neurons import capture_per_neuron_scores  # noqa: E402


def _fp_dqn_pop(device="cpu", head_hidden=(32,)):
    """A DQN population with a ReLU CNN encoder + MLP head, no norm layers."""
    obs = spaces.Box(0, 255, shape=(4, 32, 32), dtype=np.uint8)
    act = spaces.Discrete(4)
    net_config = {
        "latent_dim": 16,
        "min_latent_dim": 8,
        "max_latent_dim": 64,
        "encoder_config": {
            "channel_size": [8, 8],
            "kernel_size": [3, 3],
            "stride_size": [1, 1],
            "activation": "ReLU",
            "min_channel_size": 4,
            "max_channel_size": 64,
            "layer_norm": False,
        },
        "head_config": {
            "hidden_size": list(head_hidden),
            "activation": "ReLU",
            "min_hidden_layers": 1,
            "max_hidden_layers": 4,
            "min_mlp_nodes": 4,
            "max_mlp_nodes": 256,
            "layer_norm": False,
        },
    }
    return create_population(
        algo="DQN",
        observation_space=obs,
        action_space=act,
        net_config=net_config,
        INIT_HP={
            "POPULATION_SIZE": 1,
            "BATCH_SIZE": 8,
            "LR": 1e-3,
            "GAMMA": 0.99,
            "DOUBLE": True,
            "LEARN_STEP": 1,
            "TAU": 1e-3,
        },
        population_size=1,
        device=device,
    )


def _fp_ppo_pop(activation="ReLU", device="cpu"):
    """A PPO population with a ReLU/Tanh MLP encoder + head, no norm layers."""
    obs = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
    act = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    net_config = {
        "latent_dim": 16,
        "min_latent_dim": 4,
        "max_latent_dim": 64,
        "encoder_config": {
            "hidden_size": [16, 16],
            "activation": activation,
            "output_activation": "Identity",
            "min_mlp_nodes": 4,
            "max_mlp_nodes": 256,
            "layer_norm": False,
        },
        "head_config": {
            "hidden_size": [16, 16],
            "activation": activation,
            "min_hidden_layers": 1,
            "max_hidden_layers": 4,
            "min_mlp_nodes": 4,
            "max_mlp_nodes": 256,
            "layer_norm": False,
        },
    }
    return create_population(
        algo="PPO",
        observation_space=obs,
        action_space=act,
        net_config=net_config,
        INIT_HP={
            "POPULATION_SIZE": 1,
            "BATCH_SIZE": 8,
            "LR": 1e-3,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "ACTION_STD_INIT": 0.0,
            "CLIP_COEF": 0.2,
            "ENT_COEF": 0.0,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "TARGET_KL": None,
            "UPDATE_EPOCHS": 4,
            "LEARN_STEP": 1,
        },
        population_size=1,
        device=device,
    )


def _dqn_q(policy, obs):
    policy.eval()
    with torch.no_grad():
        return policy(obs).clone()


def _fp_muts(arch="func_preserving", seed=0, arch_fp_noise=0.0, arch_dormant_tau=0.1):
    return Mutations(
        0.5,
        0.5,
        0.2,
        0,
        0,
        0,
        0.1,
        arch_mut_type=arch,
        arch_fp_noise=arch_fp_noise,
        arch_dormant_tau=arch_dormant_tau,
        rand_seed=seed,
        device="cpu",
    )


class TestFunctionPreservingMutations:
    """Numeric + behavioural checks for arch_mut_type='func_preserving'."""

    def test_add_node_head_preserves_function(self):
        pop = _fp_dqn_pop()
        ind = pop[0]
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        q0 = _dqn_q(po, pre)
        applied, _md = _fp_muts()._apply_arch_mutation(
            po, "head_net.add_node", {"hidden_layer": 0, "numb_new_nodes": 16}
        )
        assert applied == "head_net.add_node"
        assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-5)

    def test_add_channel_preserves_function_both_boundaries(self):
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)
        # hidden_layer 0 = conv->conv boundary; hidden_layer 1 = conv->linear_output.
        for layer in (0, 1):
            pop = _fp_dqn_pop()
            ind = pop[0]
            pre = ind.preprocess_observation(obs)
            policy, _ = get_offspring_eval_modules(ind)
            _n, po = next(iter(policy.items()))
            q0 = _dqn_q(po, pre)
            applied, _md = _fp_muts()._apply_arch_mutation(
                po,
                "encoder.add_channel",
                {"hidden_layer": layer, "numb_new_channels": 8},
            )
            assert applied == "encoder.add_channel"
            assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-5), f"layer {layer}"

    def test_add_layer_identity_preserves_function_relu(self):
        pop = _fp_dqn_pop(head_hidden=(32,))
        ind = pop[0]
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        q0 = _dqn_q(po, pre)
        applied, _md = _fp_muts()._apply_arch_mutation(po, "head_net.add_layer", {})
        assert applied == "head_net.add_layer"
        assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-5)

    def test_add_node_preserves_under_tanh_activation(self):
        # add_node zeroes the fan-out, so it is function-preserving for ANY
        # activation (only add_layer's identity needs ReLU/Identity).
        pop = _fp_ppo_pop("Tanh")
        ind = pop[0]
        obs = np.random.randn(8, 6).astype(np.float32)
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        po.eval()
        with torch.no_grad():
            y0 = po.head_net.wrapped(po.encoder(pre)).clone()
        with warnings.catch_warnings():
            # add_node under a non-ReLU activation (no norm) is preserving, so it
            # must NOT emit the func-preservation caveat warning.
            warnings.simplefilter("error")
            _fp_muts()._apply_arch_mutation(
                po, "head_net.add_node", {"hidden_layer": 0, "numb_new_nodes": 16}
            )
        with torch.no_grad():
            y1 = po.head_net.wrapped(po.encoder(pre)).clone()
        assert torch.allclose(y0, y1, atol=1e-5)

    def test_add_layer_not_preserving_under_tanh_and_warns(self):
        pop = _fp_ppo_pop("Tanh")
        ind = pop[0]
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        muts = _fp_muts()
        with pytest.warns(UserWarning, match="function preservation cannot be"):
            muts._apply_arch_mutation(po, "head_net.add_layer", {})

    def test_remove_node_drops_exactly_the_dormant_units(self):
        # Make the FIRST 8 head units dead: they are then the only τ-dormant ones,
        # so the removal drops exactly them and the function is preserved. The
        # original positional removal keeps the first units and drops live ones.
        torch.manual_seed(0)
        obs = (
            np.random.default_rng(0)
            .integers(0, 255, size=(8, 4, 32, 32))
            .astype(np.uint8)
        )

        def build():
            pop = _fp_dqn_pop(head_hidden=(32,))
            ind = pop[0]
            pre = ind.preprocess_observation(obs)
            policy, _ = get_offspring_eval_modules(ind)
            _n, po = next(iter(policy.items()))
            head_layers = fp._ordered_weight_layers(po.head_net)
            with torch.no_grad():
                # Pin the layer's activations: units [0:8] are dead, the rest are
                # comfortably above tau. A randomly initialised ReLU layer would
                # otherwise leave a few live-but-weak units below tau as well,
                # making the expected count depend on the seed.
                head_layers[0].weight.data.zero_()
                head_layers[0].bias.data[:8] = 0.0
                head_layers[0].bias.data[8:] = torch.linspace(1.0, 2.0, 24)
            return ind, pre, po

        ind, pre, po = build()
        q0 = _dqn_q(po, pre)
        _applied, mut_dict = _fp_muts()._apply_arch_mutation(
            po, "head_net.remove_node", fp_obs=pre
        )
        # The count is the dormant count, not one of AgileRL's random 16/32/64.
        assert mut_dict["numb_new_nodes"] == 8
        assert fp.hidden_widths(po.head_net)[0] == 24
        assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-4)  # dropped the dead units

        # The original positional removal drops the *live* trailing units.
        _ind2, pre2, po2 = build()
        q0b = _dqn_q(po2, pre2)
        _fp_muts(arch="original")._apply_arch_mutation(
            po2, "head_net.remove_node", {"hidden_layer": 0, "numb_new_nodes": 8}
        )
        assert not torch.allclose(q0b, _dqn_q(po2, pre2), atol=1e-4)

    def test_zero_dormant_layer_removes_nothing(self):
        # Nothing is τ-dormant, so nothing can be removed without changing the
        # function: the removal is a deliberate no-op rather than an arbitrary cut.
        torch.manual_seed(0)
        obs = (
            np.random.default_rng(1)
            .integers(0, 255, size=(8, 4, 32, 32))
            .astype(np.uint8)
        )
        pop = _fp_dqn_pop(head_hidden=(32,))
        ind = pop[0]
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        head_layers = fp._ordered_weight_layers(po.head_net)
        with torch.no_grad():  # every unit comfortably above tau
            head_layers[0].weight.data.zero_()
            head_layers[0].bias.data[:] = torch.linspace(1.0, 2.0, 32)

        q0 = _dqn_q(po, pre)
        _applied, mut_dict = _fp_muts()._apply_arch_mutation(
            po, "head_net.remove_node", fp_obs=pre
        )
        assert mut_dict["numb_new_nodes"] == 0
        assert fp.hidden_widths(po.head_net)[0] == 32
        assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-4)

    def test_dormant_count_is_clamped_to_the_min_width_budget(self):
        # 30 of 32 units dead, but min_mlp_nodes=4 only allows 27 to go: the
        # removal shrinks to the floor instead of vanishing (AgileRL's guard would
        # otherwise silently apply no removal at all).
        torch.manual_seed(0)
        obs = (
            np.random.default_rng(2)
            .integers(0, 255, size=(8, 4, 32, 32))
            .astype(np.uint8)
        )
        pop = _fp_dqn_pop(head_hidden=(32,))
        ind = pop[0]
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        head_layers = fp._ordered_weight_layers(po.head_net)
        with torch.no_grad():
            head_layers[0].weight.data.zero_()
            head_layers[0].bias.data[:30] = 0.0
            head_layers[0].bias.data[30:] = torch.linspace(1.0, 2.0, 2)

        muts = _fp_muts()
        _applied, mut_dict = muts._apply_arch_mutation(
            po, "head_net.remove_node", fp_obs=pre
        )
        assert muts._fp_dormant_count == 30
        assert mut_dict["numb_new_nodes"] == 27  # 32 - min_mlp_nodes(4) - 1
        assert fp.hidden_widths(po.head_net)[0] == 5

    def test_obs_less_removal_falls_back_to_positional(self):
        # No env (pre-training step / accelerator path): degrade to AgileRL's
        # original random-count positional removal rather than silently no-op.
        pop = _fp_dqn_pop(head_hidden=(32,))
        policy, _ = get_offspring_eval_modules(pop[0])
        _n, po = next(iter(policy.items()))

        muts = _fp_muts()
        _applied, mut_dict = muts._apply_arch_mutation(
            po, "head_net.remove_node", fp_obs=None
        )
        assert mut_dict["numb_new_nodes"] in (16, 32, 64)
        assert muts._fp_dormant_count is None

    def test_remove_channel_drops_exactly_the_dormant_channels(self):
        torch.manual_seed(0)
        obs = (
            np.random.default_rng(3)
            .integers(0, 255, size=(8, 4, 32, 32))
            .astype(np.uint8)
        )
        pop = _fp_dqn_pop()
        ind = pop[0]
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        conv_layers = fp._ordered_weight_layers(po.encoder)
        with torch.no_grad():  # pin both conv layers: 4 dead channels, 4 live ones,
            for conv in conv_layers[:2]:  # so the count holds whichever is sampled
                conv.weight.data.zero_()
                conv.bias.data[:4] = 0.0
                conv.bias.data[4:] = torch.linspace(1.0, 2.0, 4)
        q0 = _dqn_q(po, pre)

        _applied, mut_dict = _fp_muts()._apply_arch_mutation(
            po, "encoder.remove_channel", fp_obs=pre
        )
        assert mut_dict["numb_new_channels"] == 4  # == the min_channel_size budget
        assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-4)

    def test_add_latent_node_preserves_function(self):
        # Widening the latent dim adds new encoder outputs whose fan-out lives in
        # the head's first layer; zeroing those new head input columns preserves
        # the function across the encoder->head boundary.
        pop = _fp_dqn_pop(head_hidden=(32,))
        ind = pop[0]
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        old_latent = po.latent_dim
        q0 = _dqn_q(po, pre)
        applied, _md = _fp_muts()._apply_arch_mutation(
            po, "add_latent_node", {"numb_new_nodes": 8}
        )
        assert applied == "add_latent_node"
        assert po.latent_dim == old_latent + 8
        assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-5)
        # The new head input columns (fan-out of the new latent units) are zero.
        head_first = fp.head_first_layer(po)
        assert torch.count_nonzero(head_first.weight.data[:, old_latent:]) == 0

    def test_add_latent_node_noise_breaks_symmetry(self):
        # With arch_fp_noise > 0 the new latent units' head columns are non-zero
        # (recruitable) but small relative to the existing weights.
        pop = _fp_dqn_pop(head_hidden=(32,))
        ind = pop[0]
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        old_latent = po.latent_dim
        _fp_muts(arch_fp_noise=0.1)._apply_arch_mutation(
            po, "add_latent_node", {"numb_new_nodes": 8}
        )
        head_first = fp.head_first_layer(po)
        new_cols = head_first.weight.data[:, old_latent:]
        existing = head_first.weight.data[:, :old_latent]
        assert torch.count_nonzero(new_cols) > 0  # symmetry broken
        assert float(new_cols.std()) < float(existing.std())  # but small

    def test_add_node_noise_reproducible(self):
        # arch_fp_noise draws from the seeded global torch RNG, so two identical
        # seeded runs produce byte-identical noised weights.
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)

        def run():
            torch.manual_seed(0)
            np.random.seed(0)
            pop = _fp_dqn_pop(head_hidden=(32,))
            ind = pop[0]
            policy, _ = get_offspring_eval_modules(ind)
            _n, po = next(iter(policy.items()))
            _fp_muts(seed=0, arch_fp_noise=0.1)._apply_arch_mutation(
                po, "head_net.add_node", {"hidden_layer": 0, "numb_new_nodes": 16}
            )
            return {k: v.clone() for k, v in po.state_dict().items()}

        a, b = run(), run()
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), k

    def test_add_node_noise_recruitable_vs_exact_zero(self):
        # noise=0 leaves the new fan-out exactly zero; noise>0 makes it non-zero.
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)

        def run(noise):
            pop = _fp_dqn_pop(head_hidden=(32,))
            ind = pop[0]
            policy, _ = get_offspring_eval_modules(ind)
            _n, po = next(iter(policy.items()))
            _fp_muts(seed=0, arch_fp_noise=noise)._apply_arch_mutation(
                po, "head_net.add_node", {"hidden_layer": 0, "numb_new_nodes": 16}
            )
            return _ordered_head_weights(po)[:, -16:].clone()

        assert torch.count_nonzero(run(0.0)) == 0
        assert torch.count_nonzero(run(0.1)) > 0

    def test_remove_latent_node_drops_lowest_activation(self):
        # Kill the FIRST 4 latent units: positional removal keeps them and drops
        # live latent units (changing the output), whereas the func-preserving
        # cross-boundary permutation ranks them last and drops them (preserving it).
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)

        def build():
            pop = _fp_dqn_pop(head_hidden=(32,))
            ind = pop[0]
            pre = ind.preprocess_observation(obs)
            policy, _ = get_offspring_eval_modules(ind)
            _n, po = next(iter(policy.items()))
            enc_out = fp.encoder_output_layer(po)
            with torch.no_grad():  # latent units [0:4] dead, the other 12 live
                enc_out.weight.data.zero_()
                enc_out.bias.data[:4] = 0.0
                enc_out.bias.data[4:] = torch.linspace(1.0, 2.0, 12)
            return ind, pre, po

        ind, pre, po = build()
        q0 = _dqn_q(po, pre)
        _applied, mut_dict = _fp_muts()._apply_arch_mutation(
            po, "remove_latent_node", fp_obs=pre
        )
        assert mut_dict["numb_new_nodes"] == 4  # the dormant latent units
        assert torch.allclose(q0, _dqn_q(po, pre), atol=1e-4)  # dropped dead latent

        _ind2, pre2, po2 = build()
        q0b = _dqn_q(po2, pre2)
        _fp_muts(arch="original")._apply_arch_mutation(
            po2, "remove_latent_node", {"numb_new_nodes": 4}
        )
        assert not torch.allclose(q0b, _dqn_q(po2, pre2), atol=1e-4)

    def test_collect_obs_returns_batch_for_latent_remove(self):
        # Latent removals also need an observation batch (for the cross-boundary
        # activation ranking), so the collection gate must accept them.
        pop = _fp_dqn_pop()
        ind = pop[0]

        class _FakeVecEnv:
            def __init__(self):
                self.obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(
                    np.uint8
                )

            def reset(self):
                return self.obs, {}

            def step(self, _action):
                n = len(self.obs)
                z = np.zeros(n, dtype=np.float32)
                return self.obs, z, z.astype(bool), z.astype(bool), {}

        muts = _fp_muts()
        muts._fp_env = _FakeVecEnv()
        assert (
            muts._fp_collect_obs(ind, "remove_latent_node", multi_agent=False)
            is not None
        )

    def test_change_kernel_falls_back_and_warns_once(self):
        pop = _fp_dqn_pop()
        policy, _ = get_offspring_eval_modules(pop[0])
        _n, po = next(iter(policy.items()))
        muts = _fp_muts()
        with pytest.warns(UserWarning, match="change_kernel cannot be made"):
            muts._apply_arch_mutation(po, "encoder.change_kernel", {"hidden_layer": 0})
        assert muts._fp_warned_kernel is True

    def test_layernorm_warns_once(self):
        # A head MLP with LayerNorm cannot guarantee preservation -> one warning.
        obs = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
        act = spaces.Discrete(3)
        net_config = {
            "latent_dim": 16,
            "encoder_config": {
                "hidden_size": [16],
                "activation": "ReLU",
                "layer_norm": True,
                "min_mlp_nodes": 4,
            },
            "head_config": {
                "hidden_size": [16],
                "activation": "ReLU",
                "layer_norm": True,
                "min_mlp_nodes": 4,
                "min_hidden_layers": 1,
                "max_hidden_layers": 3,
            },
        }
        pop = create_population(
            algo="DQN",
            observation_space=obs,
            action_space=act,
            net_config=net_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=1,
            device="cpu",
        )
        policy, _ = get_offspring_eval_modules(pop[0])
        _n, po = next(iter(policy.items()))
        muts = _fp_muts()
        with pytest.warns(UserWarning, match="function preservation cannot be"):
            muts._apply_arch_mutation(
                po, "head_net.add_node", {"hidden_layer": 0, "numb_new_nodes": 8}
            )
        assert muts._fp_warned_layernorm is True
        # Second add does not warn again (guarded).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            muts._apply_arch_mutation(
                po, "head_net.add_node", {"hidden_layer": 0, "numb_new_nodes": 8}
            )

    def test_original_mode_unaffected(self):
        # arch_mut_type="original" must reproduce the stock behaviour exactly.
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)

        def run(arch):
            pop = _fp_dqn_pop()
            ind = pop[0]
            pre = ind.preprocess_observation(obs)
            policy, _ = get_offspring_eval_modules(ind)
            _n, po = next(iter(policy.items()))
            _fp_muts(arch=arch, seed=0)._apply_arch_mutation(
                po, "head_net.add_node", {"hidden_layer": 0, "numb_new_nodes": 16}
            )
            return _ordered_head_weights(po)

        # The func-preserving path zeroes the new fan-out; original does not.
        w_orig = run("original")
        w_fp = run("func_preserving")
        # Original's new outgoing columns are non-zero; func-preserving's are zero.
        assert w_orig.abs().sum() > 0
        assert torch.count_nonzero(w_fp[:, -16:]) == 0

    def test_reproducibility(self):
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)

        def run():
            torch.manual_seed(0)
            np.random.seed(0)
            pop = _fp_dqn_pop()
            ind = pop[0]
            pre = ind.preprocess_observation(obs)
            policy, _ = get_offspring_eval_modules(ind)
            _n, po = next(iter(policy.items()))
            _fp_muts(seed=0)._apply_arch_mutation(
                po, "head_net.remove_node", fp_obs=pre
            )
            return {k: v.clone() for k, v in po.state_dict().items()}

        a, b = run(), run()
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), k

    def test_collect_obs_returns_none_without_env(self):
        muts = _fp_muts()
        muts._fp_env = None
        assert (
            muts._fp_collect_obs(object(), "head_net.remove_node", multi_agent=False)
            is None
        )

    def test_collect_obs_returns_none_for_add(self):
        muts = _fp_muts()
        muts._fp_env = object()  # non-None, but add mutations never collect
        assert (
            muts._fp_collect_obs(object(), "head_net.add_node", multi_agent=False)
            is None
        )

    def test_collect_obs_returns_preprocessed_batch_for_remove(self):
        # The positive glue: a removal + an env yields a preprocessed obs batch the
        # network can consume.
        pop = _fp_dqn_pop()
        ind = pop[0]

        class _FakeVecEnv:
            def __init__(self):
                self.obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(
                    np.uint8
                )

            def reset(self):
                return self.obs, {}

            def step(self, _action):
                n = len(self.obs)
                z = np.zeros(n, dtype=np.float32)
                return self.obs, z, z.astype(bool), z.astype(bool), {}

        muts = _fp_muts()
        muts._fp_env = _FakeVecEnv()
        obs = muts._fp_collect_obs(ind, "encoder.remove_channel", multi_agent=False)
        assert obs is not None
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        po.eval()
        with torch.no_grad():
            po(obs)

    def test_collect_obs_raises_on_collection_failure(self):
        # Fail loud: a collection error must surface, not be swallowed into a
        # silent positional-removal fallback (which would make the func_preserving
        # regime quietly measure the original operator instead).
        pop = _fp_dqn_pop()
        ind = pop[0]

        class _BrokenVecEnv:
            def reset(self):
                raise RuntimeError("env is broken")

        muts = _fp_muts()
        muts._fp_env = _BrokenVecEnv()
        with pytest.raises(RuntimeError, match="func_preserving"):
            muts._fp_collect_obs(ind, "encoder.remove_channel", multi_agent=False)

    def test_ranking_failure_raises_rather_than_falling_back(self, monkeypatch):
        # Fail loud: a ranking error must surface, not be swallowed into a silent
        # positional-removal fallback (which would make the func_preserving regime
        # quietly measure the original operator instead).
        pop = _fp_dqn_pop()
        ind = pop[0]
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))

        def _boom(*_args, **_kwargs):
            raise RuntimeError("ranking is broken")

        monkeypatch.setattr(fp, "permute_submodule_by_activation", _boom)
        with pytest.raises(RuntimeError, match="func_preserving"):
            _fp_muts()._apply_arch_mutation(po, "encoder.remove_channel", fp_obs=pre)


def _ordered_head_weights(policy):
    layers = fp._ordered_weight_layers(policy.head_net)
    # Return the consumer weight of hidden layer 0 (where new fan-out lives).
    return layers[1].weight.data.clone()


def _ma_relu_config():
    return {
        "latent_dim": 16,
        "min_latent_dim": 4,
        "max_latent_dim": 64,
        "encoder_config": {
            "hidden_size": [16],
            "activation": "ReLU",
            "output_activation": "Identity",
            "min_mlp_nodes": 4,
            "max_mlp_nodes": 256,
            "layer_norm": False,
        },
        "head_config": {
            "hidden_size": [16],
            "activation": "ReLU",
            "min_hidden_layers": 1,
            "max_hidden_layers": 4,
            "min_mlp_nodes": 4,
            "max_mlp_nodes": 256,
            "layer_norm": False,
        },
    }


class TestFunctionPreservingMultiAgent:
    def test_ippo_add_node_zeroes_new_fanout_across_moduledict(
        self, ma_vector_space, ma_discrete_space, monkeypatch
    ):
        # Verifies the func-preserving surgery reaches through the multi-agent
        # ``ModuleDict`` dispatch: after a forced add_node on one sub-agent's head,
        # the newly added units' outgoing (consumer) columns are exactly zero.
        pop = create_population(
            algo="IPPO",
            observation_space=ma_vector_space,
            action_space=ma_discrete_space,
            net_config=_ma_relu_config(),
            INIT_HP=SHARED_INIT_HP_MA,
            population_size=1,
            device="cpu",
        )
        ind = pop[0]
        policy_attr = ind.registry.policy()
        policy_md = getattr(ind, policy_attr)

        first_id = next(iter(policy_md.keys()))
        forced = f"{first_id}.head_net.add_node"
        old_width = fp.hidden_widths(policy_md[first_id].head_net)[0]

        # Force the sampled policy mutation to a specific sub-agent add_node.
        monkeypatch.setattr(
            ModuleDict,
            "sample_mutation_method",
            lambda _self, *_a, **_k: forced,
            raising=False,
        )

        muts = _fp_muts()
        muts._architecture_mutate_multi(ind)

        assert ind.mut_details.get("arch_func_preserving") is True
        mutated_head = getattr(ind, policy_attr)[first_id].head_net
        consumer = fp._ordered_weight_layers(mutated_head)[1]
        new_width = fp.hidden_widths(mutated_head)[0]
        assert new_width > old_width
        # The trailing (new) fan-out columns must be exactly zero.
        assert torch.count_nonzero(consumer.weight.data[:, old_width:new_width]) == 0

    def test_ippo_add_latent_node_preserves_across_moduledict(
        self, ma_vector_space, ma_discrete_space, monkeypatch
    ):
        # The cross-boundary latent surgery must reach through the multi-agent
        # ``ModuleDict`` dispatch: after a forced add_latent_node on one sub-agent,
        # the head's new input columns (the new latent units' fan-out) are zero.
        pop = create_population(
            algo="IPPO",
            observation_space=ma_vector_space,
            action_space=ma_discrete_space,
            net_config=_ma_relu_config(),
            INIT_HP=SHARED_INIT_HP_MA,
            population_size=1,
            device="cpu",
        )
        ind = pop[0]
        policy_attr = ind.registry.policy()
        policy_md = getattr(ind, policy_attr)

        first_id = next(iter(policy_md.keys()))
        old_latent = policy_md[first_id].latent_dim
        forced = f"{first_id}.add_latent_node"

        monkeypatch.setattr(
            ModuleDict,
            "sample_mutation_method",
            lambda _self, *_a, **_k: forced,
            raising=False,
        )

        muts = _fp_muts()
        muts._architecture_mutate_multi(ind)

        mutated = getattr(ind, policy_attr)[first_id]
        assert mutated.latent_dim > old_latent
        head_first = fp.head_first_layer(mutated)
        assert torch.count_nonzero(head_first.weight.data[:, old_latent:]) == 0


class TestCapturePerNeuronScores:
    def test_returns_per_layer_scores_in_order(self):
        pop = _fp_dqn_pop(head_hidden=(32,))
        ind = pop[0]
        obs = np.random.randint(0, 255, size=(8, 4, 32, 32)).astype(np.uint8)
        pre = ind.preprocess_observation(obs)
        policy, _ = get_offspring_eval_modules(ind)
        _n, po = next(iter(policy.items()))
        scored = capture_per_neuron_scores(po, pre)
        assert len(scored) >= 1
        for _module, per_neuron in scored:
            assert per_neuron.dim() == 1
            assert torch.isfinite(per_neuron).all()


class TestFunctionPreservingUnsupportedArchitectures:
    """Architectures the dormancy-driven removal cannot cover must degrade loudly.

    The removal needs each hidden layer's units to be the outputs of a measurable
    activation module, which holds for EvolvableMLP/EvolvableCNN only.
    """

    @staticmethod
    def _q_network(**kwargs):
        from agilerl.networks import QNetwork

        return QNetwork(action_space=generate_discrete_space(4), **kwargs)

    def test_simba_removal_warns_and_falls_back(self):
        # EvolvableSimBa hides its trunk in residual blocks and takes no
        # ``hidden_layer`` argument; passing one used to raise TypeError.
        obs_space = generate_random_box_space((6,))
        net = self._q_network(observation_space=obs_space, simba=True)
        obs = torch.randn(8, 6)

        muts = _fp_muts()
        with pytest.warns(UserWarning, match="cannot be sized"):
            _applied, mut_dict = muts._apply_arch_mutation(
                net, "encoder.remove_node", fp_obs=obs
            )
        # Fell back to AgileRL's random count, and recorded no dormancy -- so the
        # mutation-history columns stay blank rather than claiming a τ-dormant cut.
        assert mut_dict["numb_new_nodes"] in (16, 32, 64)
        assert muts._fp_dormant_count is None

    def test_simba_latent_removal_is_still_dormancy_driven(self):
        # The latent removal only needs the encoder->head boundary (one producing
        # weight layer, one consuming one), so it covers one encoder more than the
        # node/channel removal does.
        torch.manual_seed(0)
        obs_space = generate_random_box_space((6,))
        net = self._q_network(observation_space=obs_space, simba=True)
        obs = torch.randn(16, 6)

        net.eval()
        with torch.no_grad():
            before = net(obs).clone()

        muts = _fp_muts()
        fp_net = copy.deepcopy(net)
        _applied, mut_dict = muts._apply_arch_mutation(
            fp_net, "remove_latent_node", fp_obs=obs
        )
        assert muts._fp_dormant_count == mut_dict["numb_new_nodes"]

        # τ-dormant units are near-silent, not necessarily exactly dead, so the
        # guarantee is a far smaller perturbation than the stock removal of the
        # *same* count -- not bit-exactness.
        orig_net = copy.deepcopy(net)
        _fp_muts(arch="original")._apply_arch_mutation(
            orig_net, "remove_latent_node", dict(mut_dict)
        )
        fp_net.eval()
        orig_net.eval()
        with torch.no_grad():
            delta_fp = (fp_net(obs) - before).abs().max()
            delta_orig = (orig_net(obs) - before).abs().max()
        assert delta_fp < delta_orig / 100

    def test_multi_input_latent_removal_warns_and_falls_back(self):
        # EvolvableMultiInput exposes no ``model`` sequential, so no single weight
        # layer produces the latent and the encoder->head boundary cannot be
        # relabelled: the removal must degrade loudly, not report zero dormant.
        obs_space = spaces.Dict(
            {
                "vec": generate_random_box_space((6,)),
                "img": spaces.Box(0, 255, shape=(3, 32, 32), dtype=np.uint8),
            }
        )
        net = self._q_network(observation_space=obs_space)
        obs = {
            "vec": torch.randn(8, 6),
            "img": torch.rand(8, 3, 32, 32),
        }

        muts = _fp_muts()
        with pytest.warns(UserWarning, match="latent boundary could not be resolved"):
            _applied, mut_dict = muts._apply_arch_mutation(
                net, "remove_latent_node", fp_obs=obs
            )
        assert mut_dict["numb_new_nodes"] in (8, 16, 32)  # AgileRL's random count
        assert muts._fp_dormant_count is None

    def test_recurrent_encoder_exposes_no_hidden_layers(self):
        # nn.LSTM fuses its gate non-linearities: nothing to hook, and no single
        # matrix whose rows are one unit's incoming weights.
        net = self._q_network(
            observation_space=generate_random_box_space((6,)), recurrent=True
        )
        assert fp.hidden_widths(net.encoder) == []

    def test_nested_multi_input_removal_does_not_crash(self):
        # 'encoder.feature_net.<key>.remove_channel' parses to an agent_id that is
        # not a ModuleDict key, so resolving it subscripts a plain module.
        obs_space = spaces.Dict(
            {
                "vec": generate_random_box_space((6,)),
                "img": spaces.Box(0, 255, shape=(3, 32, 32), dtype=np.uint8),
            }
        )
        net = self._q_network(observation_space=obs_space)
        nested = [
            m
            for m in net.mutation_methods
            if m.startswith("encoder.feature_net.") and "remove" in m
        ]
        assert nested, "expected a nested sub-encoder removal method"
        _applied, mut_dict = _fp_muts()._apply_arch_mutation(net, nested[0])
        assert mut_dict is not None


class TestDormantRemovalCount:
    """Definition 3.1 arithmetic, isolated from any network."""

    def test_counts_units_at_or_below_tau(self):
        # Mean 1.0 -> normalised scores are the raw values; 2 are <= 0.1.
        scores = torch.tensor([0.0, 0.05, 1.5, 2.45])
        assert fp.dormant_removal_count(scores, 0.1, budget=10) == (2, 2)

    def test_clamps_to_the_budget(self):
        scores = torch.zeros(8)  # everything dormant
        assert fp.dormant_removal_count(scores, 0.1, budget=3) == (8, 3)

    def test_no_scores_removes_nothing(self):
        assert fp.dormant_removal_count(None, 0.1, budget=10) == (0, 0)
        assert fp.dormant_removal_count(torch.tensor([float("nan")]), 0.1, 10) == (0, 0)

    def test_non_finite_units_are_excluded_not_coerced(self):
        # An inf would otherwise drive the layer mean to infinity and read every
        # other unit as dormant.
        scores = torch.tensor([float("inf"), 1.0, 1.0, 0.0])
        assert fp.dormant_removal_count(scores, 0.1, budget=10) == (1, 1)

    def test_removal_budget_respects_the_guard_strictness(self):
        assert (
            fp.removal_budget(32, 4, strict=True) == 27
        )  # EvolvableMLP: width - n > min
        assert (
            fp.removal_budget(8, 4, strict=False) == 4
        )  # EvolvableCNN: width - n >= min
        assert fp.removal_budget(4, 4, strict=True) == 0  # already at the floor
