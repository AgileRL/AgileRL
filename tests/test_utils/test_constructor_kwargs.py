# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from agilerl.training.configs import TrainRunConfig
from agilerl.utils.constructor_kwargs import (
    constructor_kwargs_from_flat,
    constructor_kwargs_from_obj,
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
