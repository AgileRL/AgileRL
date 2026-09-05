# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import (
    Any,
    Generic,
    Literal,
    overload,
)

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from typing_extensions import Self

from agilerl.algorithms.configs import (
    AlgorithmRuntime,
    MultiAgentEnvConfig,
    PopulationIndex,
)
from agilerl.algorithms.core.evolvable_algorithm import EvolvableAlgorithm
from agilerl.algorithms.core.evolvable_helpers import (
    SelfAgentWrapper,
    build_classic_rl_population,
)
from agilerl.metrics import MultiAgentMetrics
from agilerl.modules import ModuleDict
from agilerl.modules.configs import MlpNetConfig, NetConfig
from agilerl.typing import (
    ArrayDict,
    DeviceType,
    ExperiencesT,
    GymSpaceType,
    InfosDict,
    MaybeActionMask,
    ModuleType,
    MultiAgentActionMasks,
    MultiAgentSetup,
    MultiAgentSpacesType,
    NetConfigType,
    ObservationType,
    TorchObsType,
    coerce_action_mask,
)
from agilerl.utils.algo_utils import (
    check_supported_space,
    concatenate_tensors,
    get_output_size_from_space,
    key_in_nested_dict,
    needs_image_transpose,
    preprocess_observation,
    stack_experiences,
    transpose_image_space,
)
from agilerl.utils.evolvable_networks import (
    config_from_dict,
    get_default_encoder_config,
    is_image_space,
    is_vector_space,
)

logger = logging.getLogger(__name__)


class MultiAgentRLAlgorithm(
    EvolvableAlgorithm[ExperiencesT], ABC, Generic[ExperiencesT]
):
    """Base object for all multi-agent algorithms in the AgileRL framework.

    :param observation_spaces: The observation spaces of the agent environments.
    :type observation_spaces: MultiAgentSpacesType
    :param action_spaces: The action spaces of the agent environments.
    :type action_spaces: MultiAgentSpacesType
    :param index: The index of the individual in the population.
    :type index: int.
    :param agent_ids: The agent IDs of the agents in the environment.
    :type agent_ids: list[int] | None, optional
    :param hp_config: Hyperparameter configuration for the algorithm, defaults to None.
    :type hp_config: HyperparameterConfig | None, optional
    :param device: Device to run the algorithm on, defaults to "cpu"
    :type device: str, optional
    :param accelerator: Accelerator object for distributed computing, defaults to None
    :type accelerator: Accelerator | None, optional
    :param torch_compiler: The torch compiler mode to use, defaults to None
    :type torch_compiler: str | None, optional
    :param normalize_images: If True, normalize images, defaults to True
    :type normalize_images: bool, optional
    :param placeholder_value: The value to use as placeholder for missing observations, defaults to -1.
    :type placeholder_value: float | None, optional
    :param name: Name of the algorithm, defaults to the class name
    :type name: str | None, optional
    """

    metrics: MultiAgentMetrics

    possible_observation_spaces: spaces.Dict
    possible_action_spaces: spaces.Dict

    shared_agent_ids: list[str]
    grouped_agents: dict[str, list[str]]
    unique_observation_spaces: dict[str, spaces.Space]
    unique_action_spaces: dict[str, spaces.Space]

    @classmethod
    def population(
        cls,
        size: int,
        observation_space: GymSpaceType,
        action_space: GymSpaceType,
        device: DeviceType = "cpu",
        wrapper_cls: Callable[..., SelfAgentWrapper] | None = None,
        wrapper_kwargs: dict[str, Any] | None = None,
        resume_from_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> list[Self | SelfAgentWrapper]:
        """Create a population of algorithms.

        :param size: The size of the population.
        :type size: int
        :param observation_space: The observation spaces of the agents.
        :type observation_space: GymSpaceType
        :param action_space: The action spaces of the agents.
        :type action_space: GymSpaceType
        :param device: Torch device. Defaults to ``"cpu"``.
        :type device: DeviceType
        :param wrapper_cls: Optional wrapper class to apply to each agent.
        :type wrapper_cls: type | None
        :param wrapper_kwargs: Keyword arguments for the wrapper class.
        :type wrapper_kwargs: dict[str, Any] | None
        :param resume_from_checkpoint: Path to checkpoint to resume from.
        :type resume_from_checkpoint: str | None
        :param kwargs: Additional keyword arguments to pass to the algorithm constructor.
        :type kwargs: Any
        :return: A list of algorithms.
        :rtype: list[MultiAgentRLAlgorithm]
        """
        return build_classic_rl_population(
            cls,
            size,
            observation_space,
            action_space,
            device,
            wrapper_cls,
            wrapper_kwargs,
            resume_from_checkpoint,
            **kwargs,
        )

    def __init__(
        self,
        observation_spaces: MultiAgentSpacesType,
        action_spaces: MultiAgentSpacesType,
        member: PopulationIndex | None = None,
        agents: MultiAgentEnvConfig | None = None,
        runtime: AlgorithmRuntime | None = None,
    ) -> None:
        member = member or PopulationIndex()
        agents = agents or MultiAgentEnvConfig()
        runtime = runtime or AlgorithmRuntime()
        index = member.index
        hp_config = member.hp_config
        agent_ids = agents.agent_ids
        placeholder_value = agents.placeholder_value
        normalize_images = agents.normalize_images
        device = runtime.device
        accelerator = runtime.accelerator
        torch_compiler = runtime.torch_compiler
        name = runtime.name

        super().__init__(index, hp_config, device, accelerator, torch_compiler, name)

        # Reject scalars/strings up front (a non-``isinstance(list)`` check so the
        # per-agent ``Iterable[spaces.Space]`` element type survives narrowing).
        if isinstance(observation_spaces, str) or not hasattr(
            observation_spaces, "__iter__"
        ):
            msg = "Observation spaces must be a list or dictionary."
            raise TypeError(msg)

        assert type(observation_spaces) is type(action_spaces), (
            "Observation spaces and action spaces must be the same type. "
            f"Got {type(observation_spaces)} and {type(action_spaces)}."
        )

        if isinstance(observation_spaces, spaces.Dict):
            assert isinstance(action_spaces, spaces.Dict), (
                "Action spaces must also be passed as a spaces.Dict."
            )
            self.possible_observation_spaces = observation_spaces
            self.possible_action_spaces = action_spaces
        elif isinstance(observation_spaces, Mapping):
            assert isinstance(action_spaces, Mapping), (
                "Action spaces must also be passed as a mapping."
            )
            self.possible_observation_spaces = spaces.Dict(
                dict(observation_spaces),
            )
            self.possible_action_spaces = spaces.Dict(dict(action_spaces))
        else:
            # A sequence of per-agent spaces paired with agent_ids. Excluding the
            # mapping cases above preserves the Iterable[spaces.Space] element
            # type that an isinstance(list, tuple) check would erase.
            assert agent_ids is not None, (
                "Agent IDs must be specified if observation spaces are passed as a list."
            )
            assert not isinstance(action_spaces, Mapping), (
                "Action spaces must also be passed as a list."
            )
            agent_id_list = list(agent_ids)
            obs_space_list = list(observation_spaces)
            action_space_list = list(action_spaces)
            assert len(agent_id_list) == len(obs_space_list), (
                "Number of agent IDs must match number of observation spaces."
            )
            self.possible_observation_spaces = spaces.Dict(
                dict(zip(agent_id_list, obs_space_list, strict=False)),
            )
            self.possible_action_spaces = spaces.Dict(
                dict(zip(agent_id_list, action_space_list, strict=False)),
            )

        for obs_space in self.possible_observation_spaces.values():
            check_supported_space(obs_space)
        for action_space in self.possible_action_spaces.values():
            check_supported_space(action_space)

        self.agent_ids = list(self.possible_observation_spaces.keys())
        self.n_agents = len(self.agent_ids)
        self.placeholder_value = placeholder_value
        self.normalize_images = normalize_images
        self.observation_spaces = list(self.possible_observation_spaces.values())
        self.action_spaces = list(self.possible_action_spaces.values())
        self.action_dims = get_output_size_from_space(self.possible_action_spaces)

        # Check if any observation space is channels-last and transpose if necessary
        self.swap_channels = needs_image_transpose(self.possible_observation_spaces)
        self.env_observation_spaces = self.possible_observation_spaces
        if self.swap_channels:
            logger.warning(
                "Found channels-last observation space. "
                "AgileRL automatically transposes images to be channels-first to support PyTorch convolutions.",
                stacklevel=2,
            )
            # transpose_image_space preserves the space structure, so a Dict
            # space transposes to a Dict space.
            transposed = transpose_image_space(self.possible_observation_spaces)
            assert isinstance(transposed, spaces.Dict)
            self.possible_observation_spaces = transposed

        # Determine groups of agents from their IDs
        self.shared_agent_ids = []
        self.grouped_agents = defaultdict(list)
        self.unique_observation_spaces = OrderedDict()
        self.unique_action_spaces = OrderedDict()
        for agent_id in self.agent_ids:
            obs_space = self.possible_observation_spaces[agent_id]
            action_space = self.possible_action_spaces[agent_id]
            # Split agent names on expected pattern of e.g. speaker_0, speaker_1,
            # listener_0, listener_1, to determine which agents are homogeneous
            group_id = self.get_group_id(agent_id)
            if group_id not in self.grouped_agents:
                self.shared_agent_ids.append(group_id)
                self.unique_observation_spaces[group_id] = obs_space
                self.unique_action_spaces[group_id] = action_space

            assert obs_space == self.unique_observation_spaces[group_id], (
                f"Homogeneous agents, i.e. agents that share the prefix {group_id}, "
                f"must have the same observation space. Found {self.unique_observation_spaces[group_id]} and {obs_space}."
            )
            assert action_space == self.unique_action_spaces[group_id], (
                f"Homogeneous agents, i.e. agents that share the prefix {group_id}, "
                f"must have the same action space. Found {self.unique_action_spaces[group_id]} and {action_space}."
            )

            self.grouped_agents[group_id].append(agent_id)

        self.n_unique_agents = len(self.shared_agent_ids)

        # Dictionary containing groups of agents for each space type
        self.grouped_spaces = defaultdict(list)
        for agent_id in self.agent_ids:
            obs_space = self.possible_observation_spaces[agent_id]
            if is_vector_space(obs_space):
                self.grouped_spaces[ModuleType.MLP].append(agent_id)
            elif is_image_space(obs_space):
                self.grouped_spaces[ModuleType.CNN].append(agent_id)
            elif isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
                self.grouped_spaces[ModuleType.MULTI_INPUT].append(agent_id)
            else:
                msg = f"Unknown observation space type: {type(obs_space)}"
                raise ValueError(msg)

        self.setup = self.get_setup()

        # Build observation space based on setup
        if self.has_grouped_agents():
            self.observation_space = self.unique_observation_spaces
            self.action_space = self.unique_action_spaces
        else:
            self.observation_space = self.possible_observation_spaces
            self.action_space = self.possible_action_spaces

        # Track multi-agent metrics using the effective training IDs. In grouped
        # setups this corresponds to shared group IDs; otherwise raw agent IDs.
        self.metrics = MultiAgentMetrics(list(self.observation_space.keys()))

    def _registry_init(self) -> None:
        super()._registry_init()

        # Additional check to ensure multi-agent networks are initialized with valid keys
        for name, network in self.evolvable_attributes(networks_only=True).items():
            if isinstance(network, ModuleDict):
                for key in network:
                    if key not in set(self.agent_ids + self.shared_agent_ids):
                        msg = (
                            f"Network '{name}' contains key '{key}' which is not present in `self.agent_ids` "
                            f"or `self.shared_agent_ids`. Please initialize multi-agent networks through agilerl.modules.ModuleDict "
                            "objects with the agent or group/shared IDs as keys."
                        )
                        raise ValueError(
                            msg,
                        )

    def has_grouped_agents(self) -> bool:
        """Whether the algorithm contains groups of agents assigned to the same
        policy for centralized execution.

        :rtype: bool
        """
        return len(self.shared_agent_ids) < len(self.agent_ids)

    def add_scores(self, scores: Sequence[float | list[float]]) -> None:
        """Add scores to the metrics, aggregating sub-agents into their groups.

        Multi-agent training loops collect non-summed score rows with one
        entry per environment agent. When agents share policies (grouped
        setups) the metrics track group IDs instead, so each row is reduced
        to the mean score per group before being recorded.

        :param scores: List of scores (or per-agent score rows) to add.
        :type scores: Sequence[float | list[float]]
        """
        is_nested = bool(scores) and isinstance(scores[0], (list, np.ndarray))
        # Grouped setups track metrics under group IDs, so per-env-agent rows
        # must be reduced to a per-group mean before being recorded.
        is_grouped = (
            is_nested
            and self.has_grouped_agents()
            and self.metrics.agent_ids == self.shared_agent_ids
        )
        if is_grouped:
            # ``is_nested`` established that rows are per-agent sequences; pin
            # each one so the per-group reduction can index it. The only shape
            # these loops produce is one entry per raw env agent; anything else
            # would mislabel group columns, so fail loudly rather than misrecord.
            score_rows: list[list[float] | np.ndarray] = []
            for row in scores:
                grouped_error = (
                    "Grouped multi-agent scores expected one entry per agent "
                    f"({len(self.agent_ids)} agents: {self.agent_ids})."
                )
                assert isinstance(row, (list, np.ndarray)), grouped_error
                assert len(row) == len(self.agent_ids), grouped_error
                score_rows.append(row)
            column = {aid: idx for idx, aid in enumerate(self.agent_ids)}
            group_columns = [
                [column[aid] for aid in self.grouped_agents[gid]]
                for gid in self.shared_agent_ids
            ]
            scores = [
                [float(np.mean([row[idx] for idx in cols])) for cols in group_columns]
                for row in score_rows
            ]
        super().add_scores(scores)

    def get_setup(self) -> MultiAgentSetup:
        """Get the type of multi-agent setup, as determined by the observation spaces of the agents.
        By having the 'same' observation space, we mean that the spaces are analogous, i.e. we can use
        the same `EvolvableModule` to process their observations.

        1. HOMOGENEOUS: All agents have the same observation space.
        2. MIXED: Agents can be grouped by their observation spaces.
        3. HETEROGENEOUS: All agents have different observation spaces.

        :return: The type of multi-agent setup.
        :rtype: MultiAgentSetup
        """
        return (
            MultiAgentSetup.HOMOGENEOUS
            if len(self.grouped_spaces) == 1
            else (
                MultiAgentSetup.MIXED
                if len(self.grouped_spaces) < len(self.agent_ids)
                else MultiAgentSetup.HETEROGENEOUS
            )
        )

    def preprocess_observation(
        self,
        observation: Mapping[str, ObservationType],
        group_ids: list[str] | None = None,
    ) -> dict[str, TorchObsType]:
        """Preprocesses observations for forward pass through neural network.

        :param observation: Per-agent observations of the environment.
        :type observation: Mapping[str, ObservationType]
        :param group_ids: Optional list of output IDs. When group IDs are provided
            (e.g., ``["agent", "other_agent"]``), observations are grouped and
            concatenated per group. Otherwise, observations are returned per
            agent ID for backwards compatibility.
        :type group_ids: list[str] | None

        :return: Preprocessed observations
        :rtype: dict[str, TorchObsType]
        """
        obs_dict = observation
        if group_ids is None:
            preprocessed: dict[str, TorchObsType] = {}
            for agent_id, agent_obs in obs_dict.items():
                preprocessed[agent_id] = preprocess_observation(
                    self.possible_observation_spaces.get(agent_id),
                    observation=agent_obs,
                    device=self.device,
                    normalize_images=self.normalize_images,
                    placeholder_value=self.placeholder_value,
                )
            return preprocessed

        buckets: dict[str, list[TorchObsType]] = {
            group_id: [] for group_id in group_ids
        }
        for agent_id, agent_obs in obs_dict.items():
            output_id = self.get_network_id(agent_id)
            if output_id not in buckets:
                buckets[output_id] = []

            buckets[output_id].append(
                preprocess_observation(
                    self.observation_space.get(output_id),
                    observation=agent_obs,
                    device=self.device,
                    normalize_images=self.normalize_images,
                    swap_channels=self.swap_channels,
                    placeholder_value=self.placeholder_value,
                )
            )
        # Populated buckets concatenate to a single tensor; empty buckets (a
        # group with no active agent) become an empty tensor so every supplied
        # group id is present in the output without widening the value type.
        return {
            output_id: (
                concatenate_tensors(obs_list)
                if obs_list
                else torch.empty(0, device=self.device)
            )
            for output_id, obs_list in buckets.items()
        }

    def extract_action_masks(self, infos: InfosDict) -> MultiAgentActionMasks:
        """Extract action masks from info dictionary.

        :param infos: Info dict
        :type infos: InfosDict

        :return: Action masks (``None`` for agents without one). The return is
            a read-only mapping so subclasses may specialise the value type:
            the base yields raw numpy masks; on-policy multi-agent subclasses
            (e.g. IPPO) stack them into per-group tensors.
        :rtype: MultiAgentActionMasks
        """
        # Get dict of form {"agent_id" : [1, 0, 0, 0]...} etc
        action_masks: dict[str, MaybeActionMask] = {}
        for agent, info in infos.items():
            if agent not in self.agent_ids:
                continue
            # Real envs occasionally hand back a non-mapping per-agent info
            # (e.g. a bare string); treat it as carrying no mask.
            mask = info.get("action_mask", None) if isinstance(info, Mapping) else None
            action_masks[agent] = coerce_action_mask(mask)
        return action_masks

    def extract_agent_masks(
        self,
        infos: InfosDict | None = None,
    ) -> tuple[ArrayDict | None, ArrayDict | None]:
        """Extract env_defined_actions from info dictionary and determine agent masks.

        :param infos: Info dict
        :type infos: InfosDict | None

        :return: Env defined actions and agent masks (both ``None`` when the
            info dict defines no actions). Actions are normalized to arrays.
        :rtype: tuple[ArrayDict | None, ArrayDict | None]
        """
        # Deal with case of no env_defined_actions defined in the info dict
        # Deal with empty info dicts for each sub agent
        if (
            infos is None
            or not key_in_nested_dict(infos, "env_defined_actions")
            or all(not info for agent, info in infos.items() if agent in self.agent_ids)
        ):
            return None, None

        raw_actions: dict[str, int | float | np.ndarray | torch.Tensor | None] = {}
        for agent, info in infos.items():
            if agent not in self.agent_ids:
                continue
            raw = info.get("env_defined_actions", None)
            if raw is None or isinstance(raw, (int, float, np.ndarray, torch.Tensor)):
                raw_actions[agent] = raw
            else:
                raw_actions[agent] = None
        env_defined_actions: ArrayDict = {}
        agent_masks: ArrayDict = {}
        for agent_id, action_val in raw_actions.items():
            val = action_val
            # Handle None if environment isn't vectorized
            if val is None:
                if not isinstance(
                    self.possible_action_spaces[agent_id],
                    spaces.Discrete,
                ):
                    nan_arr = np.empty(self.action_dims[agent_id])
                    nan_arr[:] = np.nan
                else:
                    nan_arr = np.array([np.nan])

                val = nan_arr

            # Handle discrete actions + env not vectorized
            if isinstance(val, (int, float)):
                val = np.array([val])
            elif isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()

            env_defined_actions[agent_id] = val
            agent_masks[agent_id] = np.where(
                np.isnan(val),
                0,
                1,
            ).astype(bool)

        return env_defined_actions, agent_masks

    @overload
    def build_net_config(
        self,
        net_config: NetConfigType | None = ...,
        flatten: bool = ...,
        return_encoders: Literal[False] = ...,
    ) -> NetConfigType: ...

    @overload
    def build_net_config(
        self,
        net_config: NetConfigType | None = ...,
        flatten: bool = ...,
        *,
        return_encoders: Literal[True],
    ) -> tuple[NetConfigType, dict[str, NetConfigType]]: ...

    def build_net_config(
        self,
        net_config: NetConfigType | None = None,
        flatten: bool = True,
        return_encoders: bool = False,
    ) -> NetConfigType | tuple[NetConfigType, dict[str, NetConfigType]]:
        """Extract an appropriate net config for each sub-agent from the passed net config dictionary. If
        grouped_agents is True, the net config will be built for the grouped agents i.e. through their
        common prefix in their agent_id, whenever the passed net config is None.

        .. note::
            If return_encoders is True, we return the encoder configs for each sub-agent. The only exception is
            for MLPs, where we only return the deepest architecture found. This is useful for algorithms
            with shared critics that process the observations of all agents, and therefore use an `EvolvableMultiInput`
            module to process the observations of all agents (assigning an encoder to each sub-agent and, optionally, a
            single `EvolvableMLP` to process the concatenated vector observations).

        :param net_config: Net config dictionary
        :type net_config: NetConfigType | None
        :param flatten: Whether to return a net config for each possible sub-agent, even in grouped settings.
        :type flatten: bool, optional
        :param return_encoders: Whether to return the encoder configs for each sub-agent. Defaults to False.
        :type return_encoders: bool, optional
        :return: Net config dictionary for each sub-agent
        :rtype: NetConfigType
        """
        grouped_config = self.has_grouped_agents() and not flatten
        agent_ids = self.shared_agent_ids if grouped_config else self.agent_ids
        observation_spaces = (
            self.unique_observation_spaces
            if grouped_config
            else self.possible_observation_spaces
        )
        encoder_configs = OrderedDict()

        # Helper function to append unique configs to the unique_configs dictionary
        # -> Access to unique configs is relevant for algorithms with networks that process
        # multiple agents' observations (e.g. shared critic in MADDPG)
        def _add_to_encoder_configs(config: NetConfigType, agent_id: str = "") -> None:
            net_config = config_from_dict(config)
            config_key = (
                "mlp_config" if isinstance(net_config, MlpNetConfig) else agent_id
            )

            if config_key not in encoder_configs or (
                isinstance(net_config, MlpNetConfig)
                and len(net_config["hidden_size"])
                > len(
                    encoder_configs["mlp_config"]["hidden_size"],
                )
            ):
                encoder_configs[config_key] = asdict(net_config)

        # Helper function to check if any agent ID exists in the net_config
        def _has_agent_ids(config: NetConfigType) -> bool:
            return any(
                (agent_id in self.agent_ids) or (agent_id in self.shared_agent_ids)
                for agent_id in config
            )

        # Helper function to get or create encoder config for an agent
        def _get_encoder_config(config: NetConfigType, agent_id: str) -> NetConfigType:
            simba = bool(config.get("simba", False))
            if "encoder_config" not in config or config.get("encoder_config") is None:
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id],
                    simba,
                )
                config["encoder_config"] = encoder_config
                return encoder_config
            encoder_config = config["encoder_config"]
            if not isinstance(encoder_config, (dict, NetConfig)):
                msg = (
                    f"encoder_config for agent {agent_id!r} must be a dict or "
                    f"NetConfig, got {type(encoder_config).__name__}"
                )
                raise TypeError(msg)
            return encoder_config

        # 1. net_config is None -> Automatically define an encoder for each sub-agent or group
        if net_config is None:
            net_config = defaultdict(OrderedDict)
            for agent_id in agent_ids:
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id],
                )
                net_config[agent_id]["encoder_config"] = encoder_config
                _add_to_encoder_configs(encoder_config, agent_id)

            if return_encoders:
                return net_config, encoder_configs

            return net_config

        # 2a. (Legacy) -> Passed a single-level config in a multi-agent setting - can only
        # do this in homogeneous settings where all agents have the same observation space as
        # it pertains to the network (i.e. allow as long as the observation spaces result in the
        # same encoder)
        if not _has_agent_ids(net_config):
            assert self.setup == MultiAgentSetup.HOMOGENEOUS, (
                "Single-level net config can only be passed when the multi-agent environment is homogeneous "
                "(i.e. all agents can use the same encoder to process their observations). Please specify "
                "a net config for some combination of agents (or groups of agents) in the multi-agent environment."
            )

            encoder_config = _get_encoder_config(net_config, agent_ids[0])

            full_config = OrderedDict()
            for agent_id in agent_ids:
                # Create a copy of the config for each agent
                full_config[agent_id] = net_config.copy()

                if return_encoders:
                    _add_to_encoder_configs(encoder_config, agent_id)

            if return_encoders:
                return full_config, encoder_configs

            return full_config

        if any(
            agent_id in self.agent_ids and grouped_config for agent_id in net_config
        ):
            msg = (
                "Found key in net_config corresponding to an individual sub-agent in a grouped setting. "
                "Please specify the configuration for groups instead (e.g. {'agent': {...}, ...} rather than {'agent_0': {...}, ...})"
            )
            raise KeyError(
                msg,
            )

        # 2b. Handle nested config with agent/group IDs
        result_config = {}
        config_keys = net_config.keys()
        for agent_id in agent_ids:
            group_id = self.get_group_id(agent_id) if not grouped_config else agent_id

            # 2bi. Check if agent_id is present in net_config
            if agent_id in config_keys:
                agent_config = net_config[agent_id]
                encoder_config = _get_encoder_config(agent_config, agent_id)
                result_config[agent_id] = agent_config

            # 2bii. Check if group_id is present in net_config
            elif group_id in config_keys:
                group_config = net_config[group_id]
                encoder_config = _get_encoder_config(group_config, agent_id)
                result_config[agent_id] = group_config

            # 2biii. agent_id or group_id not in net_config -> Add default encoder config
            else:
                default_config = {}
                encoder_config = get_default_encoder_config(
                    observation_spaces[agent_id],
                )
                default_config["encoder_config"] = encoder_config
                result_config[agent_id] = default_config

            if return_encoders:
                _add_to_encoder_configs(encoder_config, agent_id)

        if return_encoders:
            return result_config, encoder_configs

        return result_config

    ####---------------------------------------####
    #### Grouped Multi-Agent Utility Functions ####
    ####---------------------------------------####

    def get_group_id(self, agent_id: str) -> str:
        """Get the group ID for an agent.

        :param agent_id: The agent ID
        :type agent_id: str
        :return: The group ID
        :rtype: str
        """
        return agent_id.rsplit("_", 1)[0] if isinstance(agent_id, str) else agent_id

    def get_network_id(self, agent_id: str) -> str:
        """Get the actor/critic network ID for an agent.

        :param agent_id: The agent ID
        :type agent_id: str
        :return: The network ID
        :rtype: str
        """
        return self.get_group_id(agent_id) if self.has_grouped_agents() else agent_id

    def assemble_shared_inputs(
        self,
        experience: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Preprocesses inputs by constructing dictionaries by shared agents.

        :param experience: per-agent experience to reshape from environment
        :type experience: Mapping[str, Any]

        :return: Preprocessed inputs, grouped by shared agents
        :rtype: dict[str, dict[str, Any]]
        """
        stacked_experience: dict[str, dict[str, Any]] = {
            group_id: {} for group_id in self.observation_space
        }
        for agent_id, inp in experience.items():
            group_id = (
                self.get_group_id(agent_id) if self.has_grouped_agents() else agent_id
            )
            if isinstance(inp, list):
                stacked_exp = (
                    stack_experiences(inp, to_torch=False)[0] if len(inp) > 0 else None
                )
            else:
                stacked_exp = inp

            stacked_experience[group_id][agent_id] = stacked_exp

        return stacked_experience

    def disassemble_grouped_outputs(
        self,
        group_outputs: ArrayDict,
        vect_dim: int,
        grouped_agents: dict[str, list[str]],
    ) -> ArrayDict:
        """Disassembles batched output by shared policies into their grouped agents' outputs.

        .. note:: This assumes that for any given sub-agent the termination condition is deterministic,
            i.e. any given agent will always terminate at the same timestep in different vectorized environments.

        :param group_outputs: Dictionary to be disassembled, has the form {'agent': [4, 7, 8]}
        :type group_outputs: dict[str, npt.NDArray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :param grouped_agents: Dictionary of grouped agent IDs
        :type grouped_agents: dict[str, list[str]]
        :return: Assembled dictionary, e.g. {'agent_0': 4, 'agent_1': 7, 'agent_2': 8}
        :rtype: dict[str, npt.NDArray]
        """
        output_dict = {}
        for group_id, agent_ids in grouped_agents.items():
            group_outputs[group_id] = np.reshape(
                group_outputs[group_id],
                (len(agent_ids), vect_dim, -1),
            )
            for i, agent_id in enumerate(agent_ids):
                output_dict[agent_id] = group_outputs[group_id][i]

                if (
                    isinstance(self.possible_action_spaces[agent_id], spaces.Discrete)
                    and output_dict[agent_id].shape[-1] == 1
                ):
                    output_dict[agent_id] = output_dict[agent_id].squeeze(-1)

        return output_dict

    def sum_shared_rewards(
        self, rewards: Mapping[str, npt.NDArray | float | int]
    ) -> ArrayDict:
        """Sum the rewards for grouped agents.

        :param rewards: Reward dictionary from environment. Vectorised envs
            provide arrays; a non-vectorised ``ParallelEnv`` provides scalars.
        :type rewards: dict[str, npt.NDArray | float]
        :return: Summed rewards dictionary
        :rtype: dict[str, npt.NDArray]
        """
        reward_shape = next(iter(rewards.values()))
        reward_shape = (
            reward_shape.shape if isinstance(reward_shape, np.ndarray) else (1,)
        )
        summed_rewards = {
            agent_id: np.zeros(reward_shape) for agent_id in self.shared_agent_ids
        }
        for agent_id, reward in rewards.items():
            group_id = self.get_group_id(agent_id)
            summed_rewards[group_id] += reward

        return summed_rewards

    def assemble_grouped_outputs(
        self,
        agent_outputs: ArrayDict,
        vect_dim: int,
    ) -> ArrayDict:
        """Assembles individual agent outputs into batched outputs for shared policies.

        :param agent_outputs: Dictionary with individual agent outputs, e.g. {'agent_0': 4, 'agent_1': 7, 'agent_2': 8}
        :type agent_outputs: dict[str, npt.NDArray]
        :param vect_dim: Vectorization dimension size, i.e. number of vect envs
        :type vect_dim: int
        :return: Assembled dictionary with the form {'agent': [4, 7, 8]}
        :rtype: dict[str, npt.NDArray]
        """
        group_outputs = {}
        for group_id in self.shared_agent_ids:
            # Get all outputs for agents that share this ID
            group_agent_outputs = [
                agent_outputs[group]
                for group in self.grouped_agents[group_id]
                if group in agent_outputs
            ]

            if group_agent_outputs:
                # Stack outputs along first dimension
                stacked_outputs = np.stack(group_agent_outputs, axis=0)
                # Reshape into a form suitable for batch processing
                group_outputs[group_id] = np.reshape(
                    stacked_outputs,
                    (len(group_agent_outputs) * vect_dim, -1),
                )

        return group_outputs
