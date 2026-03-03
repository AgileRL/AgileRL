import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.modules import (
    EvolvableCNN,
    EvolvableLSTM,
    EvolvableMLP,
    EvolvableModule,
    EvolvableMultiInput,
    EvolvableSimBa,
)
from agilerl.networks.base import (
    EvolvableNetwork,
    assert_correct_lstm_net_config,
    assert_correct_simba_net_config,
)
from tests.helper_functions import (
    assert_close_dict,
    assert_not_equal_state_dict,
    assert_state_dicts_equal,
    check_equal_params_ind,
)


class InvalidCustomNetwork(EvolvableNetwork):
    def __init__(self, observation_space: spaces.Space):
        super().__init__(observation_space)

        self.name = "dummy"
        self.build_network_head(net_config={"hidden_size": [16]})

    def build_network_head(self, net_config=None):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=net_config,
        )

        # This should raise an AttributeError since we can't assign have
        # underlying evolvable modules that are not the encoder or head net
        # with mutation methods
        self.invalid_module = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=net_config,
        )

    def recreate_network(self):
        pass

    def forward(self, x):
        pass


class CustomNetwork(EvolvableNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        encoder_config=None,
        head_config=None,
        action_space=None,
        min_latent_dim=8,
        max_latent_dim=128,
        latent_dim=32,
        device="cpu",
    ):
        super().__init__(
            observation_space=observation_space,
            encoder_cls=None,
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            latent_dim=latent_dim,
            simba=False,
            device=device,
        )

        self.name = "dummy"
        self.build_network_head(net_config={"hidden_size": [64, 64]})

    def build_network_head(self, net_config=None):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=net_config,
        )

    def recreate_network(self):
        self.recreate_encoder()
        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=self.head_net.net_config,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head_net(z)


class RecurrentCustomNetwork(EvolvableNetwork):
    def __init__(self, observation_space: spaces.Space, device="cpu"):
        super().__init__(
            observation_space=observation_space, recurrent=True, device=device
        )
        self.name = "recurrent_dummy"
        self.build_network_head(net_config={"hidden_size": [32]})

    def build_network_head(self, net_config=None):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=net_config,
        )

    def recreate_network(self):
        self.recreate_encoder()

    def forward(self, x: torch.Tensor, hidden_state=None):
        z, hidden = self.extract_features(x, hidden_state=hidden_state)
        return self.head_net(z), hidden


class SimBaCustomNetwork(EvolvableNetwork):
    def __init__(self, observation_space: spaces.Space, device="cpu"):
        super().__init__(observation_space=observation_space, simba=True, device=device)
        self.name = "simba_dummy"
        self.build_network_head(net_config={"hidden_size": [32]})

    def build_network_head(self, net_config=None):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=net_config,
        )

    def recreate_network(self):
        self.recreate_encoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_net(self.extract_features(x))


class NoNumOutputsEncoder(EvolvableModule):
    def __init__(
        self, observation_space: spaces.Space, device="cpu", name="enc", **kwargs
    ):
        super().__init__(device=device)
        self.observation_space = observation_space
        self.name = name
        self.linear = torch.nn.Linear(
            spaces.flatdim(observation_space), 8, device=device
        )

    @property
    def net_config(self):
        return {}

    @property
    def activation(self):
        return "ReLU"

    def change_activation(self, activation, output=False):
        return None

    def recreate_network(self):
        return None

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.linear(x.view(x.shape[0], -1))


class DummyEncoder(EvolvableModule):
    def __init__(
        self,
        observation_space: spaces.Space,
        num_outputs: int,
        output_activation: str | None = None,
        device="cpu",
        name="enc",
    ):
        super().__init__(device=device)
        self.observation_space = observation_space
        self.num_outputs = num_outputs
        self.output_activation = output_activation
        self.name = name
        self.linear = torch.nn.Linear(
            spaces.flatdim(observation_space),
            num_outputs,
            device=device,
        )

    @property
    def net_config(self):
        return {}

    @property
    def activation(self):
        return "ReLU"

    def change_activation(self, activation, output=False):
        return None

    def recreate_network(self):
        return None

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return self.linear(x.view(x.shape[0], -1))


class CustomEncoderNetwork(EvolvableNetwork):
    def __init__(self, observation_space: spaces.Space, encoder_cls, device="cpu"):
        super().__init__(
            observation_space=observation_space,
            encoder_cls=encoder_cls,
            encoder_config={},
            device=device,
        )
        self.name = "custom_encoder_dummy"
        self.build_network_head(net_config={"hidden_size": [16]})

    def build_network_head(self, net_config=None):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=net_config,
        )

    def recreate_network(self):
        self.recreate_encoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_net(self.extract_features(x))


def test_network_incorrect_initialization(vector_space):
    with pytest.raises(AttributeError):
        InvalidCustomNetwork(vector_space)


def test_network_encoder_cls_must_be_evolvable_module(vector_space):
    """EvolvableNetwork raises TypeError when encoder_cls is not a subclass of EvolvableModule."""

    class BadEncoderNetwork(EvolvableNetwork):
        def __init__(self, observation_space):
            super().__init__(
                observation_space,
                encoder_cls=torch.nn.Linear,
                encoder_config={"hidden_size": [32]},
            )

        def build_network_head(self, *args, **kwargs):
            pass

        def forward(self, x):
            return x

    with pytest.raises(
        TypeError, match="Encoder class must be a subclass of EvolvableModule"
    ):
        BadEncoderNetwork(vector_space)


def test_network_forward_head_raises_without_head_net(vector_space):
    """EvolvableNetwork.forward_head raises AttributeError when network has no head_net."""

    class NoHeadNet(EvolvableNetwork):
        def __init__(self, obs_space):
            super().__init__(obs_space)

        def build_network_head(self, *a, **k):
            pass

        def forward(self, x):
            return x

    net_no_head = NoHeadNet(vector_space)
    latent = torch.zeros(1, net_no_head.latent_dim)
    with pytest.raises(AttributeError, match="head_net attribute"):
        net_no_head.forward_head(latent)


def test_network_initialize_hidden_state_raises_when_not_recurrent(vector_space):
    """EvolvableNetwork.initialize_hidden_state raises ValueError for non-recurrent networks."""
    network = CustomNetwork(vector_space)
    with pytest.raises(
        ValueError, match="Cannot initialize hidden state for non-recurrent"
    ):
        network.initialize_hidden_state(1)


def test_assert_correct_simba_net_config():
    assert_correct_simba_net_config({"hidden_size": 16, "num_blocks": 2})

    with pytest.raises(AssertionError):
        assert_correct_simba_net_config({"hidden_size": [16], "num_blocks": 2})


def test_assert_correct_lstm_net_config():
    assert_correct_lstm_net_config({"hidden_state_size": 32})

    with pytest.raises(AssertionError):
        assert_correct_lstm_net_config({"hidden_state_size": [32]})


@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        ("dict_space", "multi_input"),
        ("discrete_space", "mlp"),
        ("vector_space", "mlp"),
        ("image_space", "cnn"),
    ],
)
def test_network_initialization(observation_space, encoder_type, request):
    observation_space = request.getfixturevalue(observation_space)

    network = CustomNetwork(observation_space)

    assert network.observation_space == observation_space

    if encoder_type == "multi_input":
        assert isinstance(network.encoder, EvolvableMultiInput)
    elif encoder_type == "mlp":
        assert isinstance(network.encoder, EvolvableMLP)
    elif encoder_type == "cnn":
        assert isinstance(network.encoder, EvolvableCNN)


@pytest.mark.parametrize(
    "observation_space",
    ["dict_space", "discrete_space", "vector_space", "image_space"],
)
def test_network_mutation_methods(observation_space, dummy_rng, request):
    observation_space = request.getfixturevalue(observation_space)

    network = CustomNetwork(observation_space)
    network.rng = dummy_rng

    for method in network.mutation_methods:
        new_network = network.clone()
        getattr(new_network, method)()

        if "." in method:
            net_name = method.split(".")[0]
            mutated_module: EvolvableModule = getattr(new_network, net_name)
            exec_method = new_network.last_mutation_attr.split(".")[-1]

            if isinstance(observation_space, (spaces.Tuple, spaces.Dict)):
                mutated_attr = mutated_module.last_mutation_attr.split(".")[-1]
            else:
                mutated_attr = mutated_module.last_mutation_attr

            assert mutated_attr == exec_method

        if new_network.last_mutation_attr is not None:
            # Check that architecture has changed
            assert_not_equal_state_dict(network.state_dict(), new_network.state_dict())

            # Checks that parameters that are not mutated are the same
            check_equal_params_ind(network, new_network)
        else:
            msg = f"Last mutation attribute is None. Expected {method} to be applied."
            raise ValueError(
                msg,
            )


@pytest.mark.parametrize(
    "observation_space",
    ["dict_space", "discrete_space", "vector_space", "image_space"],
)
def test_network_forward(observation_space: spaces.Space, request):
    observation_space = request.getfixturevalue(observation_space)

    network = CustomNetwork(observation_space)

    x_np = observation_space.sample()

    if isinstance(observation_space, spaces.Discrete):
        x_np = (
            F.one_hot(torch.tensor(x_np), num_classes=observation_space.n)
            .float()
            .numpy()
        )

    out = network(x_np)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 1])

    if isinstance(observation_space, spaces.Dict):
        x = {key: torch.tensor(value) for key, value in x_np.items()}
    elif isinstance(observation_space, spaces.Tuple):
        x = tuple(torch.tensor(value) for value in x_np)
    else:
        x = torch.tensor(x_np)

    out = network(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 1])


@pytest.mark.parametrize(
    "observation_space",
    ["dict_space", "discrete_space", "vector_space", "image_space"],
)
def test_network_clone(observation_space: spaces.Space, request):
    observation_space = request.getfixturevalue(observation_space)

    network = CustomNetwork(observation_space)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert_state_dicts_equal(clone.state_dict(), network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])


def test_head_config_raises_without_head_net(vector_space):
    class NoHeadNet(EvolvableNetwork):
        def __init__(self, obs_space):
            super().__init__(obs_space)

        def build_network_head(self, *a, **k):
            pass

        def forward(self, x):
            return x

    net = NoHeadNet(vector_space)
    with pytest.raises(AttributeError, match="no attribute 'head_config'"):
        _ = net.head_config


def test_activation_forward_head_and_base_not_implemented(vector_space):
    class MinimalNet(EvolvableNetwork):
        def __init__(self, obs_space):
            super().__init__(obs_space)

        def forward(self, x):
            return x

    net = CustomNetwork(vector_space)
    latent = torch.randn(1, net.latent_dim)
    out = net.forward_head(latent)
    assert out.shape == (1, 1)
    assert net.activation == net.encoder.activation

    minimal = MinimalNet(vector_space)
    with pytest.raises(
        NotImplementedError, match="build_network_head must be implemented"
    ):
        minimal.build_network_head()


def test_init_weights_gaussian_and_change_activation(vector_space):
    net = CustomNetwork(vector_space)
    net.init_weights_gaussian()
    net.change_activation("ReLU")
    assert net.encoder.activation == "ReLU"


def test_recurrent_network_hidden_state_cache_and_extract_features(vector_space):
    net = RecurrentCustomNetwork(vector_space)
    assert isinstance(net.encoder, EvolvableLSTM)

    hidden_a = net.initialize_hidden_state(batch_size=2)
    hidden_b = net.initialize_hidden_state(batch_size=2)
    assert hidden_a is not hidden_b

    hidden_c = net.initialize_hidden_state(batch_size=3)
    assert hidden_c[f"{net.encoder_name}_h"].shape[1] == 3

    x = torch.randn(2, spaces.flatdim(vector_space))
    latent, next_hidden = net.extract_features(x, hidden_state=hidden_a)
    assert isinstance(latent, torch.Tensor)
    assert isinstance(next_hidden, dict)


def test_simba_encoder_build(vector_space):
    net = SimBaCustomNetwork(vector_space)
    assert isinstance(net.encoder, EvolvableSimBa)


def test_custom_encoder_warning_and_recreate(vector_space):
    with pytest.warns(UserWarning, match="does not contain `num_outputs`"):
        net_warn = CustomEncoderNetwork(vector_space, encoder_cls=NoNumOutputsEncoder)
    assert isinstance(net_warn.encoder, NoNumOutputsEncoder)

    net = CustomEncoderNetwork(vector_space, encoder_cls=DummyEncoder)
    before = net.encoder
    net.recreate_encoder()
    assert isinstance(net.encoder, DummyEncoder)
    assert net.encoder is not before


def test_extract_features_without_hidden_state(vector_space):
    net = CustomNetwork(vector_space)
    x = torch.randn(1, spaces.flatdim(vector_space))
    features = net.extract_features(x)
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == 1


def test_encoder_alias_resolution(vector_space):
    original_aliases = EvolvableNetwork._encoder_aliases.copy()
    EvolvableNetwork._encoder_aliases["DummyAlias"] = DummyEncoder
    try:
        net = CustomEncoderNetwork(vector_space, encoder_cls="DummyAlias")
        assert isinstance(net.encoder, DummyEncoder)
    finally:
        EvolvableNetwork._encoder_aliases = original_aliases
