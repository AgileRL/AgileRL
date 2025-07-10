import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.modules import (
    EvolvableCNN,
    EvolvableMLP,
    EvolvableModule,
    EvolvableMultiInput,
)
from agilerl.networks.base import EvolvableNetwork
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


def test_network_incorrect_initialization(vector_space):
    with pytest.raises(AttributeError):
        InvalidCustomNetwork(vector_space)


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
    "observation_space", ["dict_space", "discrete_space", "vector_space", "image_space"]
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
            raise ValueError(
                f"Last mutation attribute is None. Expected {method} to be applied."
            )


@pytest.mark.parametrize(
    "observation_space", ["dict_space", "discrete_space", "vector_space", "image_space"]
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
    "observation_space", ["dict_space", "discrete_space", "vector_space", "image_space"]
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
