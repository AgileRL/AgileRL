# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Supervised algorithm specification."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import AlgorithmSpec
from agilerl.arena.models.descriptions import LR
from agilerl.arena.models.hpo import RLHyperparameter
from agilerl.arena.models.registry import AgentType, register

SupervisedTaskType = Literal[
    "object_detection",
    "multiclass_classification",
    "binary_classification",
    "regression",
]

TASK_FITNESS_METRIC: dict[str, str] = {
    "object_detection": "mean_iou",
    "multiclass_classification": "accuracy",
    "binary_classification": "accuracy",
    "regression": "mse",
}

TASK_DEFAULT_LOSS: dict[str, str] = {
    "object_detection": "CrossEntropyLoss",
    "multiclass_classification": "CrossEntropyLoss",
    "binary_classification": "BCELoss",
    "regression": "MSELoss",
}

SUPERVISED_HPO_RANGES: dict[str, RLHyperparameter] = {
    "batch_size": RLHyperparameter(min=8, max=512, grow_factor=1.5, shrink_factor=0.75),
    "lr": RLHyperparameter(min=0.00001, max=0.01, grow_factor=1.25, shrink_factor=0.75),
    "weight_decay": RLHyperparameter(
        min=0.0, max=0.1, grow_factor=1.2, shrink_factor=0.8
    ),
    "label_smoothing": RLHyperparameter(
        min=0.0, max=0.2, grow_factor=1.2, shrink_factor=0.8
    ),
    "dropout": RLHyperparameter(min=0.0, max=0.5, grow_factor=1.2, shrink_factor=0.8),
}


class OptimizerSpec(BaseModel):
    """Optimizer selection for supervised training."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="AdamW", min_length=1, description="Optimizer class name."
    )
    kwargs: dict[str, object] = Field(
        default_factory=dict, description="Keyword arguments for the optimizer."
    )


class LossSpec(BaseModel):
    """Loss selection for supervised training."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="CrossEntropyLoss", min_length=1, description="Loss class name."
    )
    kwargs: dict[str, object] = Field(
        default_factory=dict, description="Keyword arguments for the loss."
    )


@register()
class SupervisedSpec(AlgorithmSpec):
    """Supervised learning over a tabular or dataset-backed environment."""

    task_type: SupervisedTaskType = Field(
        default="multiclass_classification",
        description="Task selecting the default loss and fitness metric.",
    )
    lr: float = Field(default=0.001, ge=0.0, description=LR)
    weight_decay: float = Field(
        default=0.0, ge=0.0, description="Optimizer weight decay (L2 penalty)."
    )
    label_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Label smoothing for CrossEntropyLoss.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied to model Dropout layers.",
    )
    entrypoint: str | None = Field(
        default=None, description="Module path of the model to build."
    )
    optimizer_config: OptimizerSpec = Field(
        default_factory=OptimizerSpec, description="Optimizer selection."
    )
    loss_config: LossSpec = Field(
        default_factory=LossSpec, description="Loss selection."
    )
    fitness_metric: str | None = Field(
        default=None,
        description="Metric ranking the population. Unset takes the task default.",
    )
    fitness_metric_kwargs: dict[str, object] = Field(
        default_factory=dict, description="Keyword arguments for the fitness metric."
    )

    agent_type: ClassVar[AgentType] = AgentType.SupervisedAgent
    env_type: ClassVar[str] = "tabular"
    default_evo_steps: ClassVar[int] = 128
    hpo_ranges: ClassVar[dict[str, RLHyperparameter]] = SUPERVISED_HPO_RANGES

    @model_validator(mode="after")
    def _resolve_task_defaults(self) -> Self:
        if (
            "batch_size" not in self.model_fields_set
            and self.task_type == "object_detection"
        ):
            self.batch_size = 32
        if self.fitness_metric is None:
            self.fitness_metric = TASK_FITNESS_METRIC[self.task_type]
        if "loss_config" not in self.model_fields_set:
            self.loss_config = LossSpec(name=TASK_DEFAULT_LOSS[self.task_type])
        if (
            self.task_type == "regression"
            and self.loss_config.name == "CrossEntropyLoss"
        ):
            msg = (
                "task_type=regression is incompatible with loss 'CrossEntropyLoss'. "
                "Set algorithm.loss_config to a regression loss (e.g. MSELoss)."
            )
            raise ValueError(msg)
        return self
