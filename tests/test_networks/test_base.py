import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.networks.base import EvolvableNetwork
from tests.helper_functions import (
    assert_close_dict,
    check_equal_params_ind,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_random_box_space,
)


class InvalidCustomNetwork(EvolvableNetwork):
    def __init__(self, observation_space: spaces.Space):
        super().__init__(observation_space)

        self.name = "dummy"
        self.net_config = {"hidden_size": [16]}
        self.build_network_head()

    def build_network_head(self):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=self.net_config,
        )

        # This should raise an AttributeError since we can't assign have
        # underlying evolvable modules that are not the encoder or head net
        # with mutation methods
        self.invalid_module = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config=self.net_config,
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
        n_agents=None,
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
            n_agents=n_agents,
            latent_dim=latent_dim,
            simba=False,
            device=device,
        )

        self.name = "dummy"
        # self.net_config = {"hidden_size": [16]}
        self.build_network_head()

    def build_network_head(self):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name=self.name,
            net_config={"hidden_size": [16]},
        )

    def recreate_network(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head_net(z)


def test_network_incorrect_initialization():
    observation_space = generate_random_box_space((8,))
    with pytest.raises(AttributeError):
        InvalidCustomNetwork(observation_space)


@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        (generate_dict_or_tuple_space(2, 3), "multi_input"),
        (generate_discrete_space(4), "mlp"),
        (generate_random_box_space((8,)), "mlp"),
        (generate_random_box_space((3, 32, 32)), "cnn"),
    ],
)
def test_network_initialization(observation_space, encoder_type):
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
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_network_mutation_methods(observation_space):
    network = CustomNetwork(observation_space)

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

        check_equal_params_ind(network, new_network)


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_network_forward(observation_space: spaces.Space):
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
    [
        (generate_dict_or_tuple_space(2, 3)),
        (generate_discrete_space(4)),
        (generate_random_box_space((8,))),
        (generate_random_box_space((3, 32, 32))),
    ],
)
def test_network_clone(observation_space: spaces.Space):
    network = CustomNetwork(observation_space)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert str(clone.state_dict()) == str(network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])
