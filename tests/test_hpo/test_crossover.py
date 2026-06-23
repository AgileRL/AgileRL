import numpy as np
import pytest

from agilerl.hpo.crossover import BUNDLE, Crossover
from agilerl.models.hpo import CrossoverSpec
from agilerl.models.training import TrainingSpec
from agilerl.utils.trainer_utils import build_crossover_from_spec
from agilerl.utils.utils import create_population
from tests.helper_functions import (
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
)

# Shared HP dict reused from the tournament tests.
INIT_HP = {
    "POPULATION_SIZE": 4,
    "DOUBLE": True,
    "BATCH_SIZE": 128,
    "LR": 1e-3,
    "CUDAGRAPHS": False,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "GAMMA": 0.99,
    "LEARN_STEP": 1,
    "TAU": 1e-3,
    "GAE_LAMBDA": 0.95,
    "ACTION_STD_INIT": 0.6,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "UPDATE_EPOCHS": 4,
    "AGENT_IDS": ["agent1", "agent2"],
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}

NET_CONFIG = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}


def _make_population(algo, population_size, fitnesses=None):
    """Build a single-agent population with assigned scalar fitnesses."""
    observation_space = generate_random_box_space((4,))
    action_space = generate_discrete_space(2)
    population = create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=population_size,
        device="cpu",
    )
    if fitnesses is None:
        fitnesses = [[float(i)] for i in range(population_size)]
    for agent, fit in zip(population, fitnesses, strict=False):
        agent.fitness = fit
    return population


def _make_multi_agent_population(algo, population_size):
    observation_space = generate_multi_agent_box_spaces(2, (4,))
    action_space = generate_multi_agent_discrete_spaces(2, 2)
    population = create_population(
        algo=algo,
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=population_size,
        device="cpu",
    )
    for i, agent in enumerate(population):
        agent.fitness = [float(i)]
    return population


class TestCrossoverInit:
    def test_with_given_parameters(self):
        cx = Crossover(num_parents=3, swap_prob=0.7, elitism=True, population_size=8)
        assert cx.num_parents == 3
        assert cx.swap_prob == 0.7
        assert cx.elitism is True
        assert cx.population_size == 8

    @pytest.mark.parametrize(
        "num_parents,swap_prob,elitism,population_size,match",
        [
            (1, 0.7, True, 8, "at least two"),
            (10, 0.7, True, 8, "cannot exceed"),
            (3, 1.5, True, 8, r"\[0, 1\]"),
            (3, 0.7, "no", 8, "boolean"),
            (3, 0.7, True, 0, "greater than zero"),
        ],
    )
    def test_validation(self, num_parents, swap_prob, elitism, population_size, match):
        with pytest.raises(AssertionError, match=match):
            Crossover(
                num_parents=num_parents,
                swap_prob=swap_prob,
                elitism=elitism,
                population_size=population_size,
            )


class TestCrossoverSelect:
    @pytest.mark.parametrize("algo", ["DQN", "PPO"])
    def test_returns_best_agent_and_new_population(self, algo):
        population_size = 6
        population = _make_population(algo, population_size)
        cx = Crossover(
            num_parents=4, swap_prob=0.7, elitism=True, population_size=population_size
        )

        elite, new_population = cx.crossover(population)

        # Elite is the highest-fitness agent, carried over first.
        assert elite.fitness == [float(population_size - 1)]
        assert elite.index == population_size - 1
        assert new_population[0].fitness == [float(population_size - 1)]
        assert new_population[0].index == population_size - 1
        assert new_population[0]._parent_index == population_size - 1

        assert len(new_population) == population_size
        # Indices are unique across the new population.
        assert len({a.index for a in new_population}) == population_size

    @pytest.mark.parametrize("algo", ["DQN", "PPO"])
    def test_without_elitism(self, algo):
        population_size = 6
        population = _make_population(algo, population_size)
        cx = Crossover(
            num_parents=4, swap_prob=0.7, elitism=False, population_size=population_size
        )

        elite, new_population = cx.crossover(population)

        assert elite.fitness == [float(population_size - 1)]
        assert len(new_population) == population_size
        assert len({a.index for a in new_population}) == population_size

    @pytest.mark.parametrize(
        "population_size,elitism",
        [(5, False), (4, True), (7, True)],
    )
    def test_population_size_preserved_odd_slots(self, population_size, elitism):
        # Exercises the single-offspring path on an odd final slot.
        population = _make_population("DQN", population_size)
        cx = Crossover(
            num_parents=min(4, population_size),
            swap_prob=0.5,
            elitism=elitism,
            population_size=population_size,
        )
        _, new_population = cx.crossover(population)
        assert len(new_population) == population_size

    def test_parents_drawn_from_top_pool(self):
        population_size = 6
        num_parents = 3
        population = _make_population("DQN", population_size)
        cx = Crossover(
            num_parents=num_parents,
            swap_prob=0.7,
            elitism=True,
            population_size=population_size,
        )
        _, new_population = cx.crossover(population)

        # Top-num_parents indices by fitness (fitness == index here).
        top_indices = set(range(population_size - num_parents, population_size))
        for agent in new_population:
            assert agent._parent_index in top_indices

    def test_input_population_not_modified(self):
        population_size = 5
        population = _make_population("PPO", population_size)
        before_indices = [a.index for a in population]
        before_lrs = [a.lr for a in population]

        cx = Crossover(
            num_parents=4, swap_prob=0.7, elitism=True, population_size=population_size
        )
        cx.crossover(population)

        assert [a.index for a in population] == before_indices
        assert [a.lr for a in population] == before_lrs

    def test_returned_elite_is_independent_clone(self):
        # The returned elite must be a separate object from the in-population elite
        # clone, so saving it is never affected by a subsequent mutation pass
        # (parity with tournament selection, matters when mutate_elite is True).
        population_size = 5
        population = _make_population("DQN", population_size)
        cx = Crossover(
            num_parents=4, swap_prob=0.7, elitism=True, population_size=population_size
        )
        elite, new_population = cx.crossover(population)
        assert elite is not new_population[0]
        assert elite.index == new_population[0].index == population_size - 1

    def test_multi_agent(self):
        population_size = 5
        population = _make_multi_agent_population("MADDPG", population_size)
        cx = Crossover(
            num_parents=4, swap_prob=0.7, elitism=True, population_size=population_size
        )
        elite, new_population = cx.crossover(population)
        assert elite.fitness == [float(population_size - 1)]
        assert len(new_population) == population_size
        assert len({a.index for a in new_population}) == population_size


class TestCrossoverGenetics:
    @pytest.mark.parametrize("algo", ["DQN", "PPO"])
    def test_hp_genes_inherited_not_blended(self, algo):
        # Two parents with distinct learning rates; every offspring HP value must
        # come whole from one parent or the other (never a blend), and the policy
        # network (the BUNDLE gene) must match one parent's weights exactly.
        population = _make_population(algo, 2)
        parent_a, parent_b = population
        parent_a.lr = 1e-3
        parent_b.lr = 1e-5
        # Keep each parent's optimizer consistent with its (manually set) lr.
        parent_a.reinit_optimizers()
        parent_b.reinit_optimizers()

        cx = Crossover(
            num_parents=2, swap_prob=0.5, elitism=False, population_size=2, rand_seed=0
        )
        _, new_population = cx.crossover(population)

        policy_name = parent_a.registry.policy()
        a_policy_sd = getattr(parent_a, policy_name).state_dict()
        b_policy_sd = getattr(parent_b, policy_name).state_dict()

        for child in new_population:
            # HP gene inherited whole.
            assert child.lr in (1e-3, 1e-5)
            # Optimizer learning rate is kept in sync with the attribute.
            for param_group in child.optimizer.optimizer.param_groups:
                assert param_group["lr"] == child.lr
            # BUNDLE gene (architecture+weights) inherited whole from one parent.
            child_sd = getattr(child, policy_name).state_dict()
            matches_a = all(
                np.array_equal(child_sd[k].cpu().numpy(), a_policy_sd[k].cpu().numpy())
                for k in child_sd
            )
            matches_b = all(
                np.array_equal(child_sd[k].cpu().numpy(), b_policy_sd[k].cpu().numpy())
                for k in child_sd
            )
            assert matches_a or matches_b

    def test_no_swap_keeps_parents(self):
        # swap_prob=0 => no section is ever swapped => offspring are clones of the
        # two parents (offspring 1 from parent_a, offspring 2 from parent_b).
        population = _make_population("PPO", 2)
        parent_a, parent_b = population
        parent_a.lr = 1e-3
        parent_b.lr = 1e-5
        parent_a.reinit_optimizers()
        parent_b.reinit_optimizers()

        cx = Crossover(
            num_parents=2, swap_prob=0.0, elitism=False, population_size=2, rand_seed=0
        )
        _, new_population = cx.crossover(population)

        parent_lrs = {1e-3, 1e-5}
        assert {child.lr for child in new_population} == parent_lrs

    def test_lr_swap_preserves_optimizer_state(self):
        # When an LR gene is swapped onto a bundle parent's network, the optimizer
        # state (Adam moments) must be preserved, not wiped by a rebuild -- crossover
        # never changes the architecture, so only the step size should change.
        import torch

        population = _make_population("PPO", 2)
        parent_a, parent_b = population
        parent_a.lr = 1e-3
        parent_b.lr = 1e-5
        parent_a.reinit_optimizers()
        parent_b.reinit_optimizers()

        # Populate Adam state on the bundle parent (parent_a) with a dummy step.
        opt_a = parent_a.optimizer.optimizer
        for group in opt_a.param_groups:
            for p in group["params"]:
                p.grad = torch.zeros_like(p)
        opt_a.step()
        assert len(opt_a.state) > 0

        cx = Crossover(num_parents=2, swap_prob=0.7, elitism=False, population_size=2)
        # Force a chromosome where the bundle comes from A but the LR comes from B.
        gene_source = {name: parent_a for name in parent_a.registry.hp_config.names()}
        gene_source["lr"] = parent_b
        gene_source[BUNDLE] = parent_a

        child = cx._assemble_offspring(gene_source, new_index=99)

        assert child.lr == parent_b.lr  # LR was swapped in
        child_opt = child.optimizer.optimizer
        for group in child_opt.param_groups:
            assert group["lr"] == parent_b.lr  # step size updated
        assert len(child_opt.state) > 0  # Adam state preserved, not wiped

    def test_determinism(self):
        population_size = 6
        population = _make_population("DQN", population_size)

        def run():
            cx = Crossover(
                num_parents=4,
                swap_prob=0.7,
                elitism=True,
                population_size=population_size,
                rand_seed=123,
            )
            _, new_pop = cx.crossover(population)
            return [(a.index, a._parent_index, a.lr) for a in new_pop]

        assert run() == run()

    def test_bundle_sentinel_distinct_from_hp_names(self):
        # The BUNDLE sentinel must not collide with any real hyperparameter name.
        population = _make_population("DQN", 2)
        assert BUNDLE not in population[0].registry.hp_config.names()


class TestBuildCrossoverFromSpec:
    def test_none_returns_none(self):
        assert build_crossover_from_spec(None, TrainingSpec(pop_size=8)) is None

    def test_builds_crossover(self):
        spec = CrossoverSpec(num_parents=6, swap_prob=0.6, elitism=False, rand_seed=7)
        cx = build_crossover_from_spec(spec, TrainingSpec(pop_size=8))
        assert isinstance(cx, Crossover)
        assert cx.num_parents == 6
        assert cx.swap_prob == 0.6
        assert cx.elitism is False
        assert cx.population_size == 8

    def test_num_parents_exceeds_pop_size_raises(self):
        spec = CrossoverSpec(num_parents=16)
        with pytest.raises(ValueError, match="cannot exceed"):
            build_crossover_from_spec(spec, TrainingSpec(pop_size=8))


class TestCrossoverSpec:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_parents": 1},
            {"swap_prob": 1.5},
            {"swap_prob": -0.1},
            {"rand_seed": -1},
        ],
    )
    def test_validation(self, kwargs):
        with pytest.raises(ValueError):
            CrossoverSpec(**kwargs)

    def test_defaults(self):
        spec = CrossoverSpec()
        assert spec.num_parents == 2
        assert spec.swap_prob == 0.7
        assert spec.elitism is True
        assert spec.rand_seed == 42
