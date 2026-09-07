# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from agilerl.training.configs import TrainRunConfig
from agilerl.utils.constructor_kwargs import (
    constructor_accepts_var_keyword,
    constructor_kwargs_from_flat,
    constructor_kwargs_from_obj,
    constructor_parameter_names,
    from_hparams,
    own_init_has_var_params,
    with_runtime_wrap,
)


@dataclass
class Runtime:
    device: str = "cpu"
    wrap: bool = True


@dataclass
class Learn:
    lr: float = 1e-4
    batch_size: int = 64


class Toy:
    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learn: Learn | None = None,
        runtime: Runtime | None = None,
    ) -> None:
        learn = learn or Learn()
        runtime = runtime or Runtime()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = learn.lr
        self.batch_size = learn.batch_size
        self.device = runtime.device
        self.wrap = runtime.wrap


class TestConstructorKwargsFromFlat:
    def test_groups_matching_field_names(self):
        kwargs = constructor_kwargs_from_flat(
            Toy,
            {
                "observation_space": 4,
                "action_space": 2,
                "lr": 3e-4,
                "device": "cuda",
                "unknown": True,
            },
        )

        assert kwargs["observation_space"] == 4
        assert kwargs["action_space"] == 2
        assert kwargs["learn"].lr == 3e-4
        assert kwargs["learn"].batch_size == 64
        assert kwargs["runtime"].device == "cuda"
        assert "unknown" not in kwargs

    def test_keeps_an_explicit_dataclass_instance(self):
        learn = Learn(lr=0.5)
        kwargs = constructor_kwargs_from_flat(
            Toy,
            {"observation_space": 1, "action_space": 1, "learn": learn},
        )

        assert kwargs["learn"] is learn

    def test_optional_nested_dataclass_stays_none(self):
        @dataclass
        class Schedule:
            num_epochs: int
            warmup: float

        @dataclass
        class Train:
            lr: float = 1e-4
            schedule: Schedule | None = None

        class Agent:
            def __init__(self, train: Train | None = None) -> None:
                self.train = train or Train()

        kwargs = constructor_kwargs_from_flat(Agent, {"lr": 0.2, "schedule": None})

        assert kwargs["train"].lr == 0.2
        assert kwargs["train"].schedule is None

    def test_existing_dataclass_keeps_values_when_flat_field_is_none(self):
        learn = Learn(lr=0.5, batch_size=8)
        kwargs = constructor_kwargs_from_flat(
            Toy,
            {
                "observation_space": 1,
                "action_space": 1,
                "learn": learn,
                "lr": None,
            },
            strict=True,
        )

        assert kwargs["learn"].lr == 0.5
        assert kwargs["learn"].batch_size == 8
        assert kwargs["learn"] is learn

    def test_existing_dataclass_merges_non_none_flat_fields(self):
        learn = Learn(lr=0.5, batch_size=8)
        kwargs = constructor_kwargs_from_flat(
            Toy,
            {
                "observation_space": 1,
                "action_space": 1,
                "learn": learn,
                "batch_size": 16,
            },
        )

        assert kwargs["learn"].lr == 0.5
        assert kwargs["learn"].batch_size == 16


class TestConstructorKwargsFromObj:
    def test_rebuilds_configs_from_unpacked_attributes(self):
        toy = Toy(3, 1, Learn(lr=0.2), Runtime(device="cpu", wrap=False))

        kwargs = constructor_kwargs_from_obj(toy)

        assert kwargs["observation_space"] == 3
        assert kwargs["learn"].lr == 0.2
        assert kwargs["runtime"].wrap is False

    def test_varargs_subclass_rebuilds_from_grouped_parent_init(self):
        class VarargsToy(Toy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class InheritedVarargs(VarargsToy):
            pass

        toy = InheritedVarargs(3, 1, Learn(lr=0.2), Runtime(device="cpu", wrap=False))

        kwargs = constructor_kwargs_from_obj(toy)

        assert kwargs["observation_space"] == 3
        assert kwargs["learn"].lr == 0.2
        assert kwargs["runtime"].device == "cpu"


class TestOwnInitHasVarParams:
    def test_own_star_kwargs_init(self):
        class VarargsToy(Toy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        assert own_init_has_var_params(VarargsToy) is True

    def test_inherited_star_kwargs_init(self):
        class VarargsToy(Toy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class Child(VarargsToy):
            pass

        assert own_init_has_var_params(Child) is True

    def test_grouped_parent_init(self):
        class Child(Toy):
            pass

        assert own_init_has_var_params(Child) is False
        assert own_init_has_var_params(Toy) is False


class TestFromHparams:
    def test_positional_spaces_and_flat_hparams(self):
        toy = from_hparams(Toy, 8, 4, lr=0.01, wrap=False)

        assert toy.observation_space == 8
        assert toy.action_space == 4
        assert toy.lr == 0.01
        assert toy.wrap is False


class TestWithRuntimeWrap:
    def test_replaces_wrap_on_runtime_dataclass(self):
        kwargs = {"runtime": Runtime(wrap=True)}

        updated = with_runtime_wrap(kwargs, wrap=False)

        assert updated["runtime"].wrap is False
        assert kwargs["runtime"].wrap is True


class TestAssembleLeftoverPositionals:
    def test_fills_flattened_dataclass_fields(self):
        from agilerl.utils.constructor_kwargs import assemble_init_kwargs

        kwargs = assemble_init_kwargs(Toy, (4, 2, 0.3), {})

        assert kwargs["observation_space"] == 4
        assert kwargs["action_space"] == 2
        assert kwargs["learn"].lr == 0.3
        assert kwargs["learn"].batch_size == 64

    def test_dataclass_instances_bind_to_group_params(self):
        from agilerl.utils.constructor_kwargs import assemble_init_kwargs

        kwargs = assemble_init_kwargs(
            Toy,
            (4, 2, Learn(lr=0.3, batch_size=8), Runtime(device="cuda")),
            {},
        )

        assert kwargs["observation_space"] == 4
        assert kwargs["action_space"] == 2
        assert kwargs["learn"].lr == 0.3
        assert kwargs["learn"].batch_size == 8
        assert kwargs["runtime"].device == "cuda"

    def test_agent_ids_list_does_not_bind_to_member_index(self):
        from agilerl.utils.constructor_kwargs import assemble_init_kwargs

        @dataclass
        class Member:
            index: int = 0
            mut: str | None = None

        @dataclass
        class Agents:
            agent_ids: list[str] | None = None
            placeholder_value: float | None = -1

        class MARL:
            def __init__(
                self,
                observation_spaces: list[int],
                action_spaces: list[int],
                member: Member | None = None,
                agents: Agents | None = None,
            ) -> None:
                del observation_spaces, action_spaces, member, agents

        kwargs = assemble_init_kwargs(
            MARL,
            ([0, 1], [0, 1], ["agent_0", "agent_1"]),
            {},
        )

        assert kwargs["member"].index == 0
        assert kwargs["agents"].agent_ids == ["agent_0", "agent_1"]

    def test_net_config_dict_binds_to_network_not_index(self):
        from agilerl.utils.constructor_kwargs import assemble_init_kwargs

        @dataclass
        class Member:
            index: int = 0

        @dataclass
        class Network:
            net_config: dict[str, object] | None = None

        class Bandit:
            def __init__(
                self,
                observation_space: int,
                action_space: int,
                member: Member | None = None,
                network: Network | None = None,
            ) -> None:
                del observation_space, action_space, member, network

        net_config = {"hidden_size": [8]}
        kwargs = assemble_init_kwargs(Bandit, (4, 2, net_config), {})

        assert kwargs["member"].index == 0
        assert kwargs["network"].net_config == net_config


class TestConstructorParameterNames:
    def test_includes_nested_dataclass_fields(self):
        names = constructor_parameter_names(Toy)

        assert names >= {
            "observation_space",
            "action_space",
            "learn",
            "runtime",
            "lr",
            "batch_size",
            "device",
            "wrap",
        }
        assert "unknown" not in names

    def test_var_keyword_is_detected(self):
        class Loose:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

        assert constructor_accepts_var_keyword(Loose)
        assert not constructor_accepts_var_keyword(Toy)

    def test_var_kwargs_subclass_still_lists_nested_fields(self):
        class Child(Toy):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)

        names = constructor_parameter_names(Child)

        assert names >= {"lr", "batch_size", "device", "wrap"}


class TestAcceptFlatKwargs:
    def test_maps_flat_kwargs_onto_a_grouped_function(self):
        from agilerl.utils.constructor_kwargs import accept_flat_kwargs

        @accept_flat_kwargs
        def grouped(env: str, run: TrainRunConfig | None = None) -> int:
            run = run or TrainRunConfig()
            return run.loop.max_steps

        assert grouped("cartpole", max_steps=12) == 12


class TestNestedTrainRunConfig:
    def test_flat_loop_fields_fill_nested_configs(self):
        kwargs = constructor_kwargs_from_flat(
            _train_stub,
            {"env": "cartpole", "max_steps": 12, "wb": True},
        )

        run = kwargs["run"]
        assert kwargs["env"] == "cartpole"
        assert run.loop.max_steps == 12
        assert run.logging.wb is True
        assert run.loop.evo_steps == 10_000


def _train_stub(env: str, run: TrainRunConfig | None = None) -> None:
    del env, run
