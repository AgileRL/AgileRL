import pytest
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.cnn import EvolvableCNN
from agilerl.networks.base import EvolvableNetwork
from tests.helper_functions import generate_dict_or_tuple_space, generate_discrete_space, generate_random_box_space

class InvalidCustomNetwork(EvolvableNetwork):
    def __init__(self, observation_space: spaces.Space):
        super().__init__(observation_space)

        self.build_network_head()
    
    def build_network_head(self):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1
        )

        # This should raise an AttributeError since we can't assign have 
        # underlying evolvable modules that are not the encoder or head net 
        # with mutation methods
        self.invalid_module = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1
        )

    def recreate_network(self):
        pass

class CustomNetwork(EvolvableNetwork):
    def __init__(self, observation_space: spaces.Space):
        super().__init__(observation_space)

        self.build_network_head()
    
    def build_network_head(self):
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1
        )

    def recreate_network(self):
        pass

def test_network_invalid_initialization():
    with pytest.raises(AttributeError):
        InvalidCustomNetwork()

@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        (generate_dict_or_tuple_space(2, 3), "multi_input"),
        (generate_discrete_space(4), "mlp"),
        (generate_random_box_space((8,)), "mlp"),
        (generate_random_box_space((3, 32, 32)), "cnn"),
    ]
)
def test_network_initialization(observation_space, encoder_type):
    network = EvolvableNetwork(observation_space)

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
    ]
)
def test_network_mutation_methods(observation_space):
    network = CustomNetwork(observation_space)

    for method in network.mutation_methods:
        getattr(network, method)()

        if "." in method:
            net_name = method.split(".")[0]
            mutated_module: EvolvableModule = getattr(network, net_name)
            exec_method = network.last_mutation_attr.split(".")[-1]
            assert mutated_module.last_mutation_attr == exec_method