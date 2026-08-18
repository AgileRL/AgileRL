# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import random
import sys
from numbers import Number
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from tensordict import TensorDict
from torch import nn

import agilerl.utils.algo_utils as algo_utils
from agilerl.components.data import Transition
from agilerl.hpo import regrama
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.modules import EvolvableModule
from agilerl.typing import GraMaScores, NumpyObsType, TorchObsType

skip_torch_compile_on_windows_cpu = pytest.mark.skipif(
    sys.platform == "win32" and not torch.cuda.is_available(),
    reason="torch.compile inductor backend on CPU/Windows requires MSVC (cl.exe)",
)


def assert_state_dicts_equal(
    state_dict1: dict[str, torch.Tensor],
    state_dict2: dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Compare two PyTorch state dictionaries using torch.allclose for efficient comparison.

    :param state_dict1: First state dictionary
    :type state_dict1: dict[str, torch.Tensor]
    :param state_dict2: Second state dictionary
    :type state_dict2: dict[str, torch.Tensor]
    :param rtol: Relative tolerance for torch.allclose
    :type rtol: float
    :param atol: Absolute tolerance for torch.allclose
    :type atol: float
    """
    # First check that they have the same keys
    assert set(state_dict1.keys()) == set(
        state_dict2.keys(),
    ), (
        f"State dict keys don't match: {set(state_dict1.keys())} vs {set(state_dict2.keys())}"
    )

    # Then check each tensor
    for key, tensor1 in state_dict1.items():
        tensor2 = state_dict2[key]

        if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
            if tensor1.device != tensor2.device:
                tensor1 = tensor1.cpu()
                tensor2 = tensor2.cpu()

            assert tensor1.shape == tensor2.shape, (
                f"Tensors for key '{key}' have different shapes: {tensor1.shape} != {tensor2.shape}"
            )
            assert torch.allclose(
                tensor1,
                tensor2,
                rtol=rtol,
                atol=atol,
            ), f"Tensors for key '{key}' are not close enough"


def assert_not_equal_state_dict(
    state_dict1: dict[str, torch.Tensor],
    state_dict2: dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Compare two PyTorch state dictionaries using torch.allclose for efficient comparison.

    :param state_dict1: First state dictionary
    :type state_dict1: dict[str, torch.Tensor]
    :param state_dict2: Second state dictionary
    :type state_dict2: dict[str, torch.Tensor]
    :param rtol: Relative tolerance for torch.allclose
    :type rtol: float
    :param atol: Absolute tolerance for torch.allclose
    :type atol: float
    """
    try:
        assert_state_dicts_equal(state_dict1, state_dict2, rtol, atol)
    except AssertionError:
        return

    msg = f"State dicts are equal: {state_dict1} == {state_dict2}"
    raise AssertionError(msg)


def check_equal_params_ind(
    before_ind: nn.Module | EvolvableModule,
    mutated_ind: nn.Module | EvolvableModule,
) -> None:
    before_dict = dict(before_ind.named_parameters())
    after_dict = mutated_ind.named_parameters()
    for key, param in after_dict:
        if key in before_dict:
            old_param = before_dict[key]
            old_size = old_param.data.size()
            new_size = param.data.size()
            if old_size == new_size:
                # If the sizes are the same, just copy the parameter
                param.data = old_param.data
            elif "norm" not in key:
                # Create a slicing index to handle tensors with varying sizes
                slice_index = tuple(
                    slice(0, min(o, n))
                    for o, n in zip(old_size, new_size, strict=False)
                )
                assert torch.all(
                    torch.eq(param.data[slice_index], old_param.data[slice_index]),
                ), (
                    f"Parameter {key} not equal after mutation {mutated_ind.last_mutation_attr}:\n{param.data[slice_index]}\n{old_param.data[slice_index]}"
                )


def unpack_network(model: nn.Sequential) -> list[nn.Module]:
    """Unpacks an nn.Sequential type model."""
    layer_list = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            # If it's an nn.Sequential, recursively unpack its children
            layer_list.extend(unpack_network(layer))
        elif isinstance(layer, nn.Flatten):
            pass
        else:
            layer_list.append(layer)

    return layer_list


def check_models_same(model1: nn.Module, model2: nn.Module) -> bool:
    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def generate_random_box_space(
    shape: tuple[int, ...],
    low: Number | None = None,
    high: Number | None = None,
) -> spaces.Box:
    return spaces.Box(
        low=random.randint(0, 5) if low is None else low,
        high=random.randint(6, 10) if high is None else high,
        shape=shape,
        dtype=np.float32,
    )


def generate_discrete_space(n: int) -> spaces.Discrete:
    return spaces.Discrete(n)


def generate_multidiscrete_space(n: int, m: int) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([n] * m)


def generate_dict_or_tuple_space(
    n_image: int,
    n_vector: int,
    image_shape: tuple[int, ...] = (3, 32, 32),
    vector_shape: tuple[int] = (4,),
    dict_space: bool | None = True,
) -> spaces.Dict | spaces.Tuple:

    if dict_space is None:
        dict_space = random.random() < 0.5

    image_spaces = [
        generate_random_box_space(image_shape, low=0, high=255) for _ in range(n_image)
    ]
    vector_spaces = [
        generate_random_box_space(vector_shape, low=-1, high=1) for _ in range(n_vector)
    ]

    if dict_space:
        image_spaces = {f"image_{i}": space for i, space in enumerate(image_spaces)}
        vector_spaces = {f"vector_{i}": space for i, space in enumerate(vector_spaces)}
        return spaces.Dict(image_spaces | vector_spaces)

    return spaces.Tuple(image_spaces + vector_spaces)


def generate_multi_agent_box_spaces(
    n_agents: int,
    shape: tuple[int, ...],
    low: Number | list[Number] | None = -1,
    high: Number | list[Number] | None = 1,
) -> list[spaces.Box]:
    if isinstance(low, list):
        assert len(low) == n_agents
    if isinstance(high, list):
        assert len(high) == n_agents

    spaces = []
    for i in range(n_agents):
        _low = low[i] if isinstance(low, list) else low
        _high = high[i] if isinstance(high, list) else high

        spaces.append(generate_random_box_space(shape, _low, _high))

    return spaces


def generate_multi_agent_discrete_spaces(
    n_agents: int,
    m: int,
) -> list[spaces.Discrete]:
    return [generate_discrete_space(m) for _ in range(n_agents)]


def generate_multi_agent_multidiscrete_spaces(
    n_agents: int,
    m: int,
) -> list[spaces.MultiDiscrete]:
    return [generate_multidiscrete_space(m, m) for _ in range(n_agents)]


def gen_multi_agent_dict_or_tuple_spaces(
    n_agents: int,
    n_image: int,
    n_vector: int,
    image_shape: tuple[int, ...] = (3, 16, 16),
    vector_shape: tuple[int] = (4,),
    dict_space: bool | None = False,
) -> list[spaces.Dict | spaces.Tuple]:
    return [
        generate_dict_or_tuple_space(
            n_image,
            n_vector,
            image_shape,
            vector_shape,
            dict_space,
        )
        for _ in range(n_agents)
    ]


def get_sample_from_space(
    space: spaces.Space,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> NumpyObsType:
    """Generate a sample from the given space.

    :param space: The space to generate a sample from.
    :type space: spaces.Space
    :param batch_size: The batch size.
    :type batch_size: int
    :return: A sample from the space.
    :rtype: NumpyObsType
    """
    if isinstance(space, spaces.Box):
        if batch_size is None:
            return np.random.uniform(low=space.low, high=space.high, size=space.shape)
        return np.random.uniform(
            low=space.low,
            high=space.high,
            size=(batch_size, *space.shape),
        )
    if isinstance(space, spaces.Discrete):
        if batch_size is None:
            return np.random.randint(space.n, size=(1,))
        return np.random.randint(space.n, size=(batch_size, 1))
    if isinstance(space, spaces.MultiDiscrete):
        if batch_size is None:
            return np.random.randint(space.nvec, size=(len(space.nvec),))
        return np.random.randint(space.nvec, size=(batch_size, len(space.nvec)))
    if isinstance(space, spaces.Dict):
        return {
            key: get_sample_from_space(value, batch_size)
            for key, value in space.items()
        }
    if isinstance(space, spaces.Tuple):
        return tuple(get_sample_from_space(value, batch_size) for value in space)
    msg = f"Unsupported space type: {type(space)}"
    raise ValueError(msg)


def is_processed_observation(observation: TorchObsType, space: spaces.Space) -> bool:
    if isinstance(space, spaces.Box):
        return (
            isinstance(observation, torch.Tensor)
            and observation.shape[1:] == space.shape
        )
    if isinstance(space, spaces.Discrete):
        return isinstance(observation, torch.Tensor) and observation.shape[1:] == (1,)
    if isinstance(space, spaces.MultiDiscrete):
        return isinstance(observation, torch.Tensor) and observation.shape[1:] == (
            len(space.nvec),
        )
    if isinstance(space, spaces.Dict):
        return isinstance(observation, dict) and all(
            is_processed_observation(observation[key], space[key]) for key in space
        )
    if isinstance(space, spaces.Tuple):
        return isinstance(observation, tuple) and all(
            is_processed_observation(value, space[i])
            for i, value in enumerate(observation)
        )
    msg = f"Unsupported space type: {type(space)}"
    raise ValueError(msg)


def get_experiences_batch(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    batch_size: int,
    device: torch.device | None = None,
) -> TensorDict:
    """Generate a batch of experiences from the observation and action spaces.

    :param observation_space: The observation space.
    :type observation_space: spaces.Space
    :param action_space: The action space.
    :type action_space: spaces.Space
    :param batch_size: The batch size.
    :type batch_size: int
    :param device: The device to run the experiences on.
    :type device: torch.device
    :return: A batch of experiences.
    :rtype: TensorDict
    """
    device = device if device is not None else "cpu"
    states = get_sample_from_space(observation_space, batch_size)
    actions = get_sample_from_space(action_space, batch_size)
    rewards = torch.randn((batch_size, 1))
    next_states = get_sample_from_space(observation_space, batch_size)
    dones = torch.randint(0, 2, (batch_size, 1))
    return Transition(
        obs=states,
        action=actions,
        reward=rewards,
        next_obs=next_states,
        done=dones,
        batch_size=[batch_size],
        device=device,
    ).to_tensordict()


def assert_close_dict(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key, value in before.items():
        if isinstance(value, dict):
            assert_close_dict(value, after[key])
        elif isinstance(value, torch.Tensor):
            assert torch.allclose(
                value,
                after[key],
            ), f"Value not close: {value} != {after[key]}"
        else:
            assert value == after[key], f"Value not equal: {value} != {after[key]}"


class TransposeImageObservationSpy:
    """Records calls to ``transpose_image_observation`` while delegating."""

    def __init__(self, original: Any) -> None:
        self._original = original
        self.call_count = 0
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        self.calls.append((args, kwargs))
        return self._original(*args, **kwargs)


def patch_transpose_image_observation(monkeypatch: Any) -> TransposeImageObservationSpy:
    """Patch ``transpose_image_observation`` in algo_utils and return a call spy."""
    spy = TransposeImageObservationSpy(algo_utils.transpose_image_observation)
    monkeypatch.setattr(algo_utils, "transpose_image_observation", spy)
    return spy


def assert_transpose_image_observation_called(
    spy: TransposeImageObservationSpy, min_calls: int = 1
) -> None:
    """Assert ``transpose_image_observation`` was invoked during preprocessing."""
    assert spy.call_count >= min_calls, (
        f"Expected transpose_image_observation to be called at least {min_calls} "
        f"time(s), but it was called {spy.call_count} time(s)"
    )


# Maps actor scalar-output storage pointers to per-arm mu values for the current test.
_broadcast_by_storage: dict[int, torch.Tensor] = {}


def patch_actor_scalar_mu_repeat(
    monkeypatch: pytest.MonkeyPatch,
    bandit: object,
    broadcast_mu: torch.Tensor,
) -> None:
    """Scope mu repeat override to scalar outputs from ``bandit.actor``.

    When the actor returns a single-element tensor, ``repeat(n)`` for ``n > 1``
    returns ``broadcast_mu`` instead of duplicating the scalar. Other tensors
    keep the default ``torch.Tensor.repeat`` behavior.
    """
    actor = bandit.actor
    original_forward = actor.forward
    original_repeat = torch.Tensor.repeat

    def forwarding(obs, *args, **kwargs):
        result = original_forward(obs, *args, **kwargs)
        if result.reshape(-1).numel() == 1:
            _broadcast_by_storage[result.untyped_storage().data_ptr()] = broadcast_mu
        return result

    def selective_repeat(self, *args, **kwargs):
        broadcast = _broadcast_by_storage.get(self.untyped_storage().data_ptr())
        if broadcast is not None and self.numel() == 1:
            repeats = args[0] if args else kwargs.get("repeats", 1)
            if repeats > 1:
                return broadcast
        return original_repeat(self, *args, **kwargs)

    monkeypatch.setattr(actor, "forward", forwarding)
    monkeypatch.setattr(torch.Tensor, "repeat", selective_repeat)
    _broadcast_by_storage.clear()


def rank_population_by_subpopulation(population: list[Any]) -> None:
    """Give every agent a distinct fitness, subpopulation 0 dominating subpopulation 1.

    :param population: Agents tagged with a subpopulation_id.
    :type population: list[Any]
    """
    for position, agent in enumerate(population):
        base = 100.0 if agent.subpopulation_id == 0 else 0.0
        agent.fitness = [base - position]


def weakest_agent_index(population: list[Any], subpop: int) -> int:
    """Return the index of the lowest-fitness member of a subpopulation.

    :param population: Agents tagged with a subpopulation_id.
    :type population: list[Any]
    :param subpop: Subpopulation to search.
    :type subpop: int
    :return: Index of that subpopulation's weakest agent.
    :rtype: int
    """
    members = [agent for agent in population if agent.subpopulation_id == subpop]
    return min(members, key=lambda agent: agent.fitness[-1]).index


class _FakeParam:
    """RLParameter-like stand-in carrying a mutable value."""

    def __init__(self, value=None):
        self.value = value


class FakeHPConfig:
    """Mimics HyperparameterConfig."""

    def __init__(self, names):
        self._params = {name: _FakeParam() for name in names}

    def __iter__(self):
        return iter(self._params)

    def __bool__(self):
        return bool(self._params)

    def names(self):
        return list(self._params)

    def __getitem__(self, key):
        return self._params[key]


class FakeTorchOptimizer:
    """Mimics a torch.optim.Optimizer."""

    def __init__(self, lr):
        self.param_groups = [{"lr": lr}]


class FakeOptimizerWrapper:
    """Mimics a OptimizerWrapper."""

    def __init__(self, lr):
        self.lr = lr
        self.optimizer = FakeTorchOptimizer(lr)


class FakeOptConfig:
    """Mimics OptimizerConfig."""

    def __init__(self, lr="lr", name="optimizer"):
        self.lr = lr
        self.name = name


class FakeRegistry:
    """Mimics an algorithm's NetworkGroup/optimizer registry."""

    def __init__(self, hp_names):
        self.hp_config = FakeHPConfig(hp_names)
        self.optimizers = [FakeOptConfig(lr="lr", name="optimizer")]


class FakeSelectionAgent:
    """Stand-in for an EvolvableAlgorithm, as seen by a selection strategy."""

    def __init__(
        self, index, subpopulation_id, fitness, weights="w", lr=1e-3, batch_size=64
    ):
        self.index = index
        self.subpopulation_id = subpopulation_id
        self.fitness = [fitness]
        self.weights = weights
        self.lr = lr
        self.batch_size = batch_size
        self.registry = FakeRegistry(["lr", "batch_size"])
        self.optimizer = FakeOptimizerWrapper(lr)
        self.reinit_called = False

    def clone(self, index=None, wrap=False):
        # type(self) so subclasses adding their own bookkeeping survive cloning
        new = type(self)(
            self.index if index is None else index,
            self.subpopulation_id,
            self.fitness[-1],
            weights=self.weights,
            lr=self.lr,
            batch_size=self.batch_size,
        )
        for name in self.registry.hp_config.names():
            new.registry.hp_config[name].value = self.registry.hp_config[name].value
        return new

    def mutation_hook(self):
        self.mutation_hook_called = True

    def reinit_optimizers(self, optimizer=None):
        # Mirror the real reinit_optimizers
        self.reinit_called = True
        self.optimizer = FakeOptimizerWrapper(self.lr)

    def save_checkpoint(self, path):
        with open(path, "w") as fh:
            fh.write("checkpoint")


def make_fake_selection_population(
    subpop_fitnesses: dict[int | None, list[float]],
    weights: dict[int | None, list[str]] | None = None,
) -> list[FakeSelectionAgent]:
    """Build a population of :class:`FakeSelectionAgent` with unique indices.

    :param subpop_fitnesses: Fitness of every agent, keyed by subpopulation.
    :type subpop_fitnesses: dict[int | None, list[float]]
    :param weights: Per-agent weight tags in the same layout, defaults to None.
    :type weights: dict[int | None, list[str]] | None, optional
    :return: The agents, laid out subpopulation by subpopulation.
    :rtype: list[FakeSelectionAgent]
    """
    population = []
    idx = 0
    for subpop, fitnesses in subpop_fitnesses.items():
        for j, fit in enumerate(fitnesses):
            w = weights[subpop][j] if weights is not None else f"w{idx}"
            population.append(FakeSelectionAgent(idx, subpop, fit, weights=w))
            idx += 1
    return population


def make_multi_frequency_selection(
    n_subpop=2, population_size=8, ratios=None, w=1, s=1, o=1, ln=1, seed=42
) -> MultiFrequencySelection:
    """Build a :class:`~agilerl.hpo.multi_frequency.MultiFrequencySelection`.

    :param n_subpop: Number of subpopulations, defaults to 2.
    :type n_subpop: int, optional
    :param population_size: Total population size, defaults to 8.
    :type population_size: int, optional
    :param ratios: Evolution frequency ratios, defaults to ``1..n_subpop``.
    :type ratios: list[int] | None, optional
    :param w: Winners per subpopulation, defaults to 1.
    :type w: int, optional
    :param s: Survivors per subpopulation, defaults to 1.
    :type s: int, optional
    :param o: Slots open for migration per subpopulation, defaults to 1.
    :type o: int, optional
    :param ln: Losers per subpopulation, defaults to 1.
    :type ln: int, optional
    :param seed: Seed for the winner-clone RNG, defaults to 42.
    :type seed: int, optional
    :return: The configured operator.
    :rtype: MultiFrequencySelection
    """
    return MultiFrequencySelection(
        population_size=population_size,
        n_subpopulations=n_subpop,
        evolution_frequency_ratios=ratios or list(range(1, n_subpop + 1)),
        n_winners=w,
        n_survivors=s,
        n_open_for_migration=o,
        n_losers=ln,
        seed=seed,
    )


def new_agents(before: list[Any], after: list[Any]) -> list[Any]:
    """Return the agents in ``after`` that are not one of the objects in ``before``.

    :param before: Population before an evolution step.
    :type before: list[Any]
    :param after: Population after an evolution step.
    :type after: list[Any]
    :return: The freshly created agents.
    :rtype: list[Any]
    """
    before_ids = {id(a) for a in before}
    return [a for a in after if id(a) not in before_ids]


def grama_scores_for(agent: Any, fill: float = 0.0) -> GraMaScores:
    """Build a GraMa gradient snapshot for every evaluation network of *agent*.

    Each measured layer gets one score per producing neuron, every one set to
    *fill*; a layer whose producer cannot be resolved is recorded as ``None``,
    which is the shape a real capture leaves behind for a layer that never
    fired. Scores are normalised by their layer mean before being thresholded,
    so a uniform snapshot reads as a score of 1.0 everywhere -- ``fill=0.0``
    marks the whole agent dormant and any positive *fill* marks it healthy.

    :param agent: Agent whose evaluation networks are measured.
    :type agent: EvolvableAlgorithm
    :param fill: Per-neuron gradient magnitude to record for every neuron.
    :type fill: float, optional
    :return: A snapshot laid out exactly as :class:`agilerl.hpo.regrama.GraMaCapture`
        would store it.
    :rtype: GraMaScores
    """
    scores: GraMaScores = []
    for _network_id, network in regrama.eval_networks(agent):
        entries: list[torch.Tensor | None] = []
        for activation in regrama.target_activations(network):
            producer = regrama.resolve_producer_and_next(
                activation,
                getattr(network, "encoder", None),
                getattr(network, "head_net", None),
            ).producer
            entries.append(
                None
                if producer is None
                else torch.full(
                    (regrama.weight_param(producer).shape[0],),
                    fill,
                ),
            )
        scores.append(entries)
    return scores
