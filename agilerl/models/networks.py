from collections.abc import Callable
from typing import Literal, Self, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T", bound=BaseModel)

MlpActivation = Literal[
    "Tanh",
    "GELU",
    "ReLU",
    "ELU",
    "LeakyReLU",
    "Sigmoid",
    "Softplus",
    "Softmax",
    "GumbelSoftmax",
    "PReLU",
    "Identity",
]


def min_max_validator(min_field: str, max_field: str) -> Callable[[T], T]:
    """Validate that a field is less than or equal to another field.

    :param min_field: The name of the minimum field
    :type min_field: str
    :param max_field: The name of the maximum field
    :type max_field: str
    :return: A validator function
    :rtype: Callable[[T], T]
    """

    def validator(model: T) -> T:
        min_value = getattr(model, min_field)
        max_value = getattr(model, max_field)
        if min_value > max_value:
            msg = f"{min_field} ({min_value}) must be less than or equal to {max_field} ({max_value})."
            raise ValueError(
                msg,
            )
        return model

    return validator


class MlpSpec(BaseModel):
    """Model specification for evolvable MLP networks."""

    hidden_size: list[int] = Field(min_length=1)
    activation: MlpActivation = Field(default="ReLU")
    output_activation: MlpActivation | None = Field(default=None)
    layer_norm: bool = Field(default=True)
    init_layers: bool = Field(default=True)
    output_vanish: bool = Field(default=True)
    min_hidden_layers: int = Field(default=1, gt=0)
    max_hidden_layers: int = Field(default=6, gt=1)
    min_mlp_nodes: int = Field(default=8, gt=0)
    max_mlp_nodes: int = Field(default=256, gt=1)

    arch: Literal["mlp"] = Field(default="mlp", exclude=True)

    @model_validator(mode="after")
    def _check_hidden_layers(self) -> Self:
        min_max_validator("min_hidden_layers", "max_hidden_layers")(self)

        if len(self.hidden_size) < self.min_hidden_layers:
            msg = f"hidden_size must be greater than or equal to min_hidden_layers ({self.min_hidden_layers})."
            raise ValueError(
                msg,
            )

        if len(self.hidden_size) > self.max_hidden_layers:
            msg = f"hidden_size must be less than or equal to max_hidden_layers ({self.max_hidden_layers})."
            raise ValueError(
                msg,
            )

        return self

    @model_validator(mode="after")
    def _check_mlp_nodes(self) -> Self:
        min_max_validator("min_mlp_nodes", "max_mlp_nodes")(self)

        if any(node < self.min_mlp_nodes for node in self.hidden_size):
            msg = f"hidden_size elements must be greater than or equal to min_mlp_nodes ({self.min_mlp_nodes})."
            raise ValueError(
                msg,
            )

        if any(node > self.max_mlp_nodes for node in self.hidden_size):
            msg = f"hidden_size elements must be less than or equal to max_mlp_nodes ({self.max_mlp_nodes})."
            raise ValueError(
                msg,
            )

        return self


class SimbaSpec(BaseModel):
    """Model specification for evolvable SimBA networks.

    Reference: https://arxiv.org/abs/2410.09754
    """

    hidden_size: int = Field(gt=0)
    num_blocks: int = Field(gt=0)
    output_activation: MlpActivation | None = Field(default=None)
    scale_factor: int = Field(default=4, gt=0)
    min_blocks: int = Field(default=1, gt=0)
    max_blocks: int = Field(default=4, gt=1)
    min_mlp_nodes: int = Field(default=8, gt=0)
    max_mlp_nodes: int = Field(default=256, gt=1)

    arch: Literal["simba"] = Field(default="simba", exclude=True)

    @model_validator(mode="after")
    def _check_blocks(self) -> Self:
        min_max_validator("min_blocks", "max_blocks")(self)
        return min_max_validator("min_blocks", "num_blocks")(self)

    @model_validator(mode="after")
    def _check_mlp_nodes(self) -> Self:
        min_max_validator("min_mlp_nodes", "max_mlp_nodes")(self)

        if self.hidden_size < self.min_mlp_nodes:
            msg = f"hidden_size must be greater than or equal to min_mlp_nodes ({self.min_mlp_nodes})."
            raise ValueError(msg)

        if self.hidden_size > self.max_mlp_nodes:
            msg = f"hidden_size must be less than or equal to max_mlp_nodes ({self.max_mlp_nodes})."
            raise ValueError(msg)

        return self


class CnnSpec(BaseModel):
    """Model specification for evolvable convolutional networks."""

    channel_size: list[int] = Field(min_length=1)
    kernel_size: list[int] = Field(min_length=1)
    stride_size: list[int] = Field(min_length=1)
    min_hidden_layers: int = Field(default=1, gt=0)
    max_hidden_layers: int = Field(default=6, gt=1)
    min_channel_size: int = Field(default=8, gt=0)
    max_channel_size: int = Field(default=256, gt=1)
    layer_norm: bool = Field(default=False)
    init_layers: bool = Field(default=True)
    activation: MlpActivation = Field(default="ReLU")

    arch: Literal["cnn"] = Field(default="cnn", exclude=True)

    @model_validator(mode="after")
    def _check_hidden_layers(self) -> Self:
        min_max_validator("min_hidden_layers", "max_hidden_layers")(self)

        if (
            len(self.channel_size) < self.min_hidden_layers
            or len(self.channel_size) > self.max_hidden_layers
        ):
            msg = f"hidden_layers must be between min_hidden_layers ({self.min_hidden_layers}) and max_hidden_layers ({self.max_hidden_layers})."
            raise ValueError(
                msg,
            )

        return self

    @model_validator(mode="after")
    def _check_channel_size(self) -> Self:
        min_max_validator("min_channel_size", "max_channel_size")(self)

        if any(size < self.min_channel_size for size in self.channel_size):
            msg = f"channel_size must be greater than or equal to min_channel_size ({self.min_channel_size})."
            raise ValueError(
                msg,
            )

        if any(size > self.max_channel_size for size in self.channel_size):
            msg = f"channel_size must be less than or equal to max_channel_size ({self.max_channel_size})."
            raise ValueError(
                msg,
            )

        return self

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if len(self.channel_size) != len(self.kernel_size) or len(
            self.channel_size,
        ) != len(self.stride_size):
            msg = f"channel_size, kernel_size, and stride_size must have the same length ({len(self.channel_size)})."
            raise ValueError(
                msg,
            )
        return self


class MultiInputSpec(BaseModel):
    """Model specification for evolvable multi-input networks."""

    latent_dim: int = Field(default=32, gt=0)
    vector_space_mlp: bool = Field(default=False)
    min_latent_dim: int = Field(default=8, gt=0)
    max_latent_dim: int = Field(default=128, gt=1)
    mlp_config: MlpSpec | None = Field(default=None)
    cnn_config: CnnSpec | None = Field(default=None)

    arch: Literal["multiinput"] = Field(default="multiinput", exclude=True)

    @model_validator(mode="after")
    def _check_latent_dim(self) -> Self:
        min_max_validator("min_latent_dim", "max_latent_dim")(self)
        return min_max_validator("min_latent_dim", "latent_dim")(self)


class LstmSpec(BaseModel):
    """Model specification for evolvable LSTM networks."""

    hidden_state_size: int = Field(gt=0)
    num_layers: int = Field(default=1, gt=0)
    min_hidden_state_size: int = Field(default=8, gt=0)
    max_hidden_state_size: int = Field(default=256, gt=1)
    min_layers: int = Field(default=1, ge=0)
    max_layers: int = Field(default=6, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    arch: Literal["lstm"] = Field(default="lstm", exclude=True)

    @model_validator(mode="after")
    def _check_hidden_state_size(self) -> Self:
        min_max_validator("min_hidden_state_size", "max_hidden_state_size")(self)
        return min_max_validator("min_hidden_state_size", "hidden_state_size")(self)

    @model_validator(mode="after")
    def _check_layers(self) -> Self:
        min_max_validator("min_layers", "max_layers")(self)
        return min_max_validator("min_layers", "num_layers")(self)


EncoderType = MlpSpec | CnnSpec | LstmSpec | MultiInputSpec | SimbaSpec


class NetworkSpec(BaseModel):
    """Base model specification for nested AgileRL evolvable networks."""

    latent_dim: int = Field(default=32, gt=0)
    min_latent_dim: int = Field(default=8, gt=0)
    max_latent_dim: int = Field(default=128, gt=1)
    encoder_config: EncoderType = Field(discriminator="arch")
    head_config: MlpSpec
    random_seed: int | None = Field(default=None)
    simba: bool = Field(default=False)

    @model_validator(mode="after")
    def _check_latent_dim(self) -> Self:
        min_max_validator("min_latent_dim", "max_latent_dim")(self)
        return min_max_validator("min_latent_dim", "latent_dim")(self)

    @model_validator(mode="after")
    def _detect_simba(self) -> Self:
        """Auto-detect SimBA from the encoder type so callers don't need to
        set both ``arch: simba`` and ``simba: true`` in the manifest.
        """
        if isinstance(self.encoder_config, SimbaSpec):
            self.simba = True
        return self


class QNetworkSpec(NetworkSpec):
    """Model specification for evolvable Q networks."""


class RainbowQNetworkSpec(NetworkSpec):
    """Model specification for evolvable Rainbow Q networks."""


class ContinuousQNetworkSpec(NetworkSpec):
    """Model specification for evolvable continuous Q networks."""


class DeterministicActorSpec(NetworkSpec):
    """Model specification for evolvable deterministic actor networks."""


class StochasticActorSpec(NetworkSpec):
    """Model specification for evolvable stochastic actor networks."""

    squash_output: bool = Field(default=False)


class ValueNetworkSpec(NetworkSpec):
    """Model specification for evolvable value networks."""
