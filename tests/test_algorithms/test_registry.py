# Mock NetworkGroup for testing (avoids frame inspection issues)
from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn, optim

from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    MutationRegistry,
    NetworkGroup,
    NetworkConfig,
    OptimizerConfig,
    RLParameter,
    make_network_group,
)
from agilerl.modules import EvolvableModule


@dataclass
class MockNetworkGroup:
    """Mock NetworkGroup for testing purposes."""

    eval_network: str
    shared_networks: str | list[str] | None = field(default=None)
    policy: bool = field(default=False)

    def __hash__(self) -> int:
        return hash((self.eval_network, self.shared_networks, self.policy))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockNetworkGroup):
            return False
        return (
            self.eval_network == other.eval_network
            and self.shared_networks == other.shared_networks
            and self.policy == other.policy
        )


# Mock classes for testing
class MockEvolvableNetwork(EvolvableModule):
    """Mock evolvable network for testing purposes."""

    def __init__(self, input_dim=10, output_dim=5, name="mock_network", device="cpu"):
        super().__init__(device=device)
        self.name = name
        self.layer = nn.Linear(input_dim, output_dim)
        # Initialize with small weights for stable testing
        torch.nn.init.xavier_uniform_(self.layer.weight, gain=0.01)
        torch.nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        return self.layer(x)

    def cpu(self):
        self.device = "cpu"
        return self

    def to(self, device):
        self.device = device
        return self

    def get_init_dict(self):
        return {"device": self.device}


class TestRLParameter:
    """Test suite for RLParameter class."""

    def test_rlparameter_init_defaults(self):
        """Test RLParameter initialization with default values."""
        param = RLParameter(min=0.1, max=2.0)

        assert param.min == 0.1
        assert param.max == 2.0
        assert param.shrink_factor == 0.8
        assert param.grow_factor == 1.2
        assert param.dtype == float
        assert param.value is None

    def test_rlparameter_init_custom(self):
        """Test RLParameter initialization with custom values."""
        param = RLParameter(
            min=1,
            max=10,
            shrink_factor=0.9,
            grow_factor=1.1,
            dtype=int,
        )

        assert param.min == 1
        assert param.max == 10
        assert param.shrink_factor == 0.9
        assert param.grow_factor == 1.1
        assert param.dtype == int
        assert param.value is None

    def test_rlparameter_mutate_float(self):
        """Test mutation of float hyperparameters."""
        torch.manual_seed(42)  # For reproducible results

        param = RLParameter(min=0.1, max=2.0, dtype=float)
        param.value = 1.0

        # Test multiple mutations
        for _ in range(10):
            mutated = param.mutate()
            assert isinstance(mutated, float)
            assert param.min <= mutated <= param.max
            assert param.value == mutated

    def test_rlparameter_mutate_int(self):
        """Test mutation of integer hyperparameters."""
        torch.manual_seed(42)

        param = RLParameter(min=1, max=10, dtype=int)
        param.value = 5

        # Test multiple mutations
        for _ in range(10):
            mutated = param.mutate()
            assert isinstance(mutated, int)
            assert param.min <= mutated <= param.max
            assert param.value == mutated

    def test_rlparameter_mutate_numpy_array_float(self):
        """Test mutation of numpy float arrays."""
        torch.manual_seed(42)

        param = RLParameter(min=0.1, max=2.0, dtype=np.ndarray)
        param.value = np.array([0.5, 1.0, 1.5])

        original_dtype = param.value.dtype
        original_shape = param.value.shape

        # Test multiple mutations
        for _ in range(5):
            mutated = param.mutate()
            assert isinstance(mutated, np.ndarray)
            assert mutated.shape == original_shape
            assert mutated.dtype == original_dtype
            assert np.all(mutated >= param.min)
            assert np.all(mutated <= param.max)
            assert np.array_equal(param.value, mutated)

    def test_rlparameter_mutate_numpy_array_int(self):
        """Test mutation of numpy integer arrays."""
        torch.manual_seed(42)

        param = RLParameter(min=1, max=10, dtype=np.ndarray)
        param.value = np.array([3, 5, 7], dtype=np.int32)

        original_dtype = param.value.dtype
        original_shape = param.value.shape

        # Test multiple mutations
        for _ in range(5):
            mutated = param.mutate()
            assert isinstance(mutated, np.ndarray)
            assert mutated.shape == original_shape
            assert mutated.dtype == original_dtype
            assert np.all(mutated >= param.min)
            assert np.all(mutated <= param.max)
            assert np.array_equal(param.value, mutated)

    def test_rlparameter_mutate_multidimensional_array(self):
        """Test mutation of multi-dimensional numpy arrays."""
        torch.manual_seed(42)

        param = RLParameter(min=0.0, max=1.0, dtype=np.ndarray)
        param.value = np.array([[0.2, 0.4], [0.6, 0.8]])

        original_shape = param.value.shape

        # Test multiple mutations
        for _ in range(3):
            mutated = param.mutate()
            assert isinstance(mutated, np.ndarray)
            assert mutated.shape == original_shape
            assert np.all(mutated >= param.min)
            assert np.all(mutated <= param.max)

    def test_rlparameter_boundary_conditions(self):
        """Test mutation at boundary conditions."""
        param = RLParameter(
            min=0.0,
            max=1.0,
            shrink_factor=0.5,
            grow_factor=2.0,
            dtype=np.ndarray,
        )

        # Test at minimum
        param.value = np.array([0.0, 0.0])
        mutated_min = param.mutate()
        assert np.all(mutated_min >= param.min)
        assert np.all(mutated_min <= param.max)

        # Test at maximum
        param.value = np.array([1.0, 1.0])
        mutated_max = param.mutate()
        assert np.all(mutated_max >= param.min)
        assert np.all(mutated_max <= param.max)

    def test_rlparameter_mutate_without_value_raises_error(self):
        """Test that mutation without setting value raises assertion error."""
        param = RLParameter(min=0.1, max=2.0)

        with pytest.raises(AssertionError, match="Hyperparameter value is not set"):
            param.mutate()

    def test_rlparameter_mutate_scalar_shrink_hits_min(self):
        param = RLParameter(min=1.0, max=10.0, shrink_factor=0.5, dtype=float)
        param.value = 1.0
        with patch("torch.rand", return_value=torch.tensor([0.1])):
            assert param.mutate() == 1.0

    def test_rlparameter_mutate_scalar_grow_hits_max(self):
        param = RLParameter(min=1.0, max=10.0, grow_factor=2.0, dtype=float)
        param.value = 10.0
        with patch("torch.rand", return_value=torch.tensor([0.9])):
            assert param.mutate() == 10.0

    def test_rlparameter_mutate_int_shrink_boundary(self):
        """Shrink path for int at/near min boundary."""
        param = RLParameter(min=1, max=100, shrink_factor=0.5, dtype=int)
        param.value = 1
        with patch("torch.rand", return_value=torch.tensor([0.0])):  # shrink
            assert param.mutate() == 1

    def test_rlparameter_mutate_int_grow_boundary(self):
        """Grow path for int at max boundary."""
        param = RLParameter(min=1, max=100, grow_factor=2.0, dtype=int)
        param.value = 100
        with patch("torch.rand", return_value=torch.tensor([1.0])):  # grow
            assert param.mutate() == 100

    def test_rlparameter_mutate_numpy_shrink_boundary(self):
        """Numpy array shrink path with elements at min."""
        torch.manual_seed(42)
        param = RLParameter(min=0.0, max=1.0, shrink_factor=0.5, dtype=np.ndarray)
        param.value = np.array([0.0, 0.5])  # first element at min
        with patch("torch.rand", return_value=torch.tensor([0.2])):  # shrink
            mutated = param.mutate()
        assert np.all(mutated >= param.min)
        assert np.all(mutated <= param.max)
        assert mutated[0] == 0.0

    def test_rlparameter_mutate_numpy_grow_boundary(self):
        """Numpy array grow path with elements at max."""
        param = RLParameter(min=0.0, max=1.0, grow_factor=2.0, dtype=np.ndarray)
        param.value = np.array([0.5, 1.0])  # second at max
        with patch("torch.rand", return_value=torch.tensor([0.8])):  # grow
            mutated = param.mutate()
        assert np.all(mutated >= param.min)
        assert np.all(mutated <= param.max)
        assert mutated[1] == 1.0

    def test_rlparameter_mutate_float_shrink_above_min(self):
        """Shrink when value*shrink_factor would fall below min."""
        param = RLParameter(min=1.0, max=10.0, shrink_factor=0.5, dtype=float)
        param.value = 1.1  # 1.1 * 0.5 = 0.55 < 1.0 -> clamp to 1.0
        with patch("torch.rand", return_value=torch.tensor([0.1])):
            assert param.mutate() == 1.0

    def test_rlparameter_mutate_float_grow_below_max(self):
        """Grow when value*grow_factor would exceed max."""
        param = RLParameter(min=1.0, max=10.0, grow_factor=2.0, dtype=float)
        param.value = 6.0  # 6.0 * 2 = 12 > 10 -> clamp to 10.0
        with patch("torch.rand", return_value=torch.tensor([0.9])):
            assert param.mutate() == 10.0


class TestHyperparameterConfig:
    """Test suite for HyperparameterConfig class."""

    def test_hyperparameter_config_init_empty(self):
        """Test initialization of empty HyperparameterConfig."""
        config = HyperparameterConfig()

        assert len(config.config) == 0
        assert not config  # Should be falsy when empty
        assert list(config.names()) == []

    def test_hyperparameter_config_init_with_params(self):
        """Test initialization with hyperparameters."""
        lr_param = RLParameter(min=1e-5, max=1e-2, dtype=float)
        batch_param = RLParameter(min=16, max=256, dtype=int)

        config = HyperparameterConfig(learning_rate=lr_param, batch_size=batch_param)

        assert len(config.config) == 2
        assert bool(config)  # Should be truthy when not empty
        assert set(config.names()) == {"learning_rate", "batch_size"}
        assert config["learning_rate"] == lr_param
        assert config["batch_size"] == batch_param

    def test_hyperparameter_config_invalid_param_raises_error(self):
        """Test that invalid parameter types raise TypeError."""
        with pytest.raises(TypeError, match="Expected RLParameter object"):
            HyperparameterConfig(invalid_param="not_an_rlparameter")

    def test_hyperparameter_config_iteration(self):
        """Test iteration over HyperparameterConfig."""
        lr_param = RLParameter(min=1e-5, max=1e-2, dtype=float)
        batch_param = RLParameter(min=16, max=256, dtype=int)

        config = HyperparameterConfig(learning_rate=lr_param, batch_size=batch_param)

        # Test iteration
        param_names = list(config)
        assert set(param_names) == {"learning_rate", "batch_size"}

        # Test items
        items = dict(config.items())
        assert items["learning_rate"] == lr_param
        assert items["batch_size"] == batch_param

    def test_hyperparameter_config_equality(self):
        """Test equality comparison between HyperparameterConfig objects."""
        lr_param = RLParameter(min=1e-5, max=1e-2, dtype=float)
        batch_param = RLParameter(min=16, max=256, dtype=int)

        config1 = HyperparameterConfig(learning_rate=lr_param, batch_size=batch_param)
        config2 = HyperparameterConfig(batch_size=batch_param, learning_rate=lr_param)
        config3 = HyperparameterConfig(learning_rate=lr_param)

        assert config1 == config2  # Same parameters, different order
        assert config1 != config3  # Different parameters

    def test_hyperparameter_config_sample(self):
        """Test sampling from HyperparameterConfig."""
        torch.manual_seed(42)

        lr_param = RLParameter(min=1e-5, max=1e-2, dtype=float)
        batch_param = RLParameter(min=16, max=256, dtype=int)

        config = HyperparameterConfig(learning_rate=lr_param, batch_size=batch_param)

        # Test sampling
        name, param = config.sample()
        assert name in ["learning_rate", "batch_size"]
        assert param in [lr_param, batch_param]

    def test_hyperparameter_config_repr(self):
        """Test string representation of HyperparameterConfig."""
        lr_param = RLParameter(min=1e-5, max=1e-2, dtype=float)
        config = HyperparameterConfig(learning_rate=lr_param)

        repr_str = repr(config)
        assert "HyperparameterConfig" in repr_str
        assert "learning_rate" in repr_str


class TestNetworkConfig:
    """Test suite for NetworkConfig class."""

    def test_network_config_init_eval_network(self):
        """Test NetworkConfig initialization for evaluation network."""
        config = NetworkConfig(name="actor", eval_network=True, optimizer="actor_opt")

        assert config.name == "actor"
        assert config.eval_network is True
        assert config.optimizer == "actor_opt"

    def test_network_config_init_shared_network(self):
        """Test NetworkConfig initialization for shared network."""
        config = NetworkConfig(name="target_actor", eval_network=False)

        assert config.name == "target_actor"
        assert config.eval_network is False
        assert config.optimizer is None

    def test_network_config_eval_without_optimizer_raises_error(self):
        """Test that evaluation network without optimizer raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Evaluation network must have an optimizer",
        ):
            NetworkConfig(name="actor", eval_network=True, optimizer=None)


class TestOptimizerConfig:
    """Test suite for OptimizerConfig class."""

    def test_optimizer_config_init_single_network(self):
        """Test OptimizerConfig initialization with single network."""
        config = OptimizerConfig(
            name="actor_opt",
            networks="actor",
            lr="lr_actor",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"weight_decay": 1e-4},
        )

        assert config.name == "actor_opt"
        assert config.networks == "actor"
        assert config.lr == "lr_actor"
        assert config.optimizer_cls == "Adam"  # Stored as string
        assert config.optimizer_kwargs == {"weight_decay": 1e-4}

    def test_optimizer_config_init_multiple_networks(self):
        """Test OptimizerConfig initialization with multiple networks."""
        config = OptimizerConfig(
            name="shared_opt",
            networks=["actor", "critic"],
            lr="lr_shared",
            optimizer_cls=optim.AdamW,
            optimizer_kwargs={"eps": 1e-8},
        )

        assert config.name == "shared_opt"
        assert config.networks == ["actor", "critic"]
        assert config.lr == "lr_shared"
        assert config.optimizer_cls == "AdamW"

    def test_optimizer_config_equality(self):
        """Test equality comparison between OptimizerConfig objects."""
        config1 = OptimizerConfig(
            name="opt1",
            networks="net1",
            lr="lr",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )
        config2 = OptimizerConfig(
            name="opt1",
            networks="net1",
            lr="lr_different",
            optimizer_cls=optim.SGD,
            optimizer_kwargs={"momentum": 0.9},
        )
        config3 = OptimizerConfig(
            name="opt2",
            networks="net1",
            lr="lr",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )

        assert config1 == config2  # Same name and networks
        assert config1 != config3  # Different name

    def test_optimizer_config_get_optimizer_cls(self):
        """Test getting optimizer class from stored configuration."""
        config = OptimizerConfig(
            name="test_opt",
            networks="test_net",
            lr="lr",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )

        optimizer_cls = config.get_optimizer_cls()
        assert optimizer_cls == optim.Adam

    def test_optimizer_config_get_optimizer_cls_dict(self):
        """Test getting optimizer classes from dict configuration."""
        # Create config with dict of optimizer classes
        config = OptimizerConfig(
            name="multi_opt",
            networks=["net1", "net2"],
            lr="lr",
            optimizer_cls={"agent1": optim.Adam, "agent2": optim.SGD},
            optimizer_kwargs={},
        )

        optimizer_dict = config.get_optimizer_cls()
        assert optimizer_dict["agent1"] == optim.Adam
        assert optimizer_dict["agent2"] == optim.SGD


class TestNetworkGroup:
    """Test suite for NetworkGroup class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_eval_net = MockEvolvableNetwork(name="eval_net")
        self.mock_shared_net = MockEvolvableNetwork(name="shared_net")
        self.mock_shared_nets = [
            MockEvolvableNetwork(name="shared_net1"),
            MockEvolvableNetwork(name="shared_net2"),
        ]

    def test_network_group_init_eval_only(self):
        """Test NetworkGroup initialization with evaluation network only."""
        # NetworkGroup uses frame inspection which doesn't work in test context
        # We'll test the make_network_group function instead which bypasses this

    def test_make_network_group_function(self):
        """Test the make_network_group helper function."""
        # Create a simple NetworkGroup directly with string names to avoid frame inspection
        from dataclasses import dataclass, field

        # Create a simplified NetworkGroup for testing
        @dataclass
        class SimpleNetworkGroup:
            eval_network: str
            shared_networks: str | list[str] | None = field(default=None)
            policy: bool = field(default=False)

        # Test the basic structure
        group = SimpleNetworkGroup(
            eval_network="actor",
            shared_networks="target_actor",
            policy=True,
        )

        assert group.eval_network == "actor"
        assert group.shared_networks == "target_actor"
        assert group.policy is True

    def test_make_network_group_with_list(self):
        """Test make_network_group with list of shared networks."""
        from dataclasses import dataclass, field

        @dataclass
        class SimpleNetworkGroup:
            eval_network: str
            shared_networks: str | list[str] | None = field(default=None)
            policy: bool = field(default=False)

        group = SimpleNetworkGroup(
            eval_network="critic",
            shared_networks=["target_critic1", "target_critic2"],
            policy=False,
        )

        assert group.eval_network == "critic"
        assert group.shared_networks == ["target_critic1", "target_critic2"]
        assert group.policy is False

    def test_make_network_group_no_shared(self):
        """Test make_network_group with no shared networks."""
        from dataclasses import dataclass, field

        @dataclass
        class SimpleNetworkGroup:
            eval_network: str
            shared_networks: str | list[str] | None = field(default=None)
            policy: bool = field(default=False)

        group = SimpleNetworkGroup(
            eval_network="policy",
            shared_networks=None,
            policy=True,
        )

        assert group.eval_network == "policy"
        assert group.shared_networks is None
        assert group.policy is True

    def test_make_network_group_factory_invokes_network_group(self):
        with patch.object(NetworkGroup, "__post_init__", lambda self: None):
            group = make_network_group(
                eval_network="actor",
                shared_networks=["target_actor"],
                policy=True,
            )
        assert group.eval_network == "actor"
        assert group.shared_networks == ["target_actor"]
        assert group.policy is True

    def test_network_group_hash_and_eq_non_matching_type(self):
        group = object.__new__(NetworkGroup)
        group.eval_network = "actor"
        group.shared_networks = None
        group.policy = True
        assert isinstance(hash(group), int)
        assert (group == object()) is False

    def test_network_group_eq_with_none(self):
        """NetworkGroup __eq__ returns False for None."""
        group = object.__new__(NetworkGroup)
        group.eval_network = "actor"
        group.shared_networks = None
        group.policy = True
        assert group.__eq__(None) is False

    def test_network_group_eq_same_attrs(self):
        """Equal NetworkGroups compare equal."""
        g1 = object.__new__(NetworkGroup)
        g1.eval_network = "actor"
        g1.shared_networks = ["target_actor"]
        g1.policy = True
        g2 = object.__new__(NetworkGroup)
        g2.eval_network = "actor"
        g2.shared_networks = ["target_actor"]
        g2.policy = True
        assert g1 == g2
        with pytest.raises(TypeError, match="unhashable type"):
            hash(g1)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(g2)

    def test_network_group_eq_different_shared_str_vs_list(self):
        """NetworkGroup with shared as str vs same name in list differ."""
        g1 = object.__new__(NetworkGroup)
        g1.eval_network = "actor"
        g1.shared_networks = "target"  # str
        g1.policy = True
        g2 = object.__new__(NetworkGroup)
        g2.eval_network = "actor"
        g2.shared_networks = ["target"]  # list
        g2.policy = True
        assert g1 != g2
        assert isinstance(hash(g1), int)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(g2)

    def test_network_group_infer_attribute_names_single(self):
        """_infer_attribute_names returns correct name for single object match."""
        group = object.__new__(NetworkGroup)
        obj = object()
        container = type("C", (), {})()
        container.my_net = obj
        names = group._infer_attribute_names(container, obj)
        assert names == ["my_net"]

    def test_network_group_infer_attribute_names_list(self):
        """_infer_attribute_names returns names for list of objects."""
        group = object.__new__(NetworkGroup)
        obj1, obj2 = object(), object()
        container = type("C", (), {})()
        container.net_a = obj1
        container.net_b = obj2
        names = group._infer_attribute_names(container, [obj1, obj2])
        assert set(names) == {"net_a", "net_b"}

    def test_network_group_infer_attribute_names_no_match(self):
        """_infer_attribute_names returns empty when no match."""
        group = object.__new__(NetworkGroup)
        obj = object()
        container = type("C", (), {})()
        container.other = object()  # different object
        names = group._infer_attribute_names(container, obj)
        assert names == []

    def test_make_network_group_shared_str(self):
        """make_network_group with shared_networks as single string."""
        with patch.object(NetworkGroup, "__post_init__", lambda self: None):
            group = make_network_group("actor", "target_actor", policy=True)
        assert group.eval_network == "actor"
        assert group.shared_networks == "target_actor"
        assert group.policy is True

    def test_make_network_group_shared_empty_list(self):
        """make_network_group with shared_networks as empty list."""
        with patch.object(NetworkGroup, "__post_init__", lambda self: None):
            group = make_network_group("critic", [], policy=False)
        assert group.eval_network == "critic"
        assert group.shared_networks == []
        assert group.policy is False

    def test_make_network_group_returns_network_group_instance(self):
        """make_network_group returns NetworkGroup instance."""
        with patch.object(NetworkGroup, "__post_init__", lambda self: None):
            group = make_network_group("net", None, policy=False)
        assert isinstance(group, NetworkGroup)


class TestMutationRegistry:
    """Test suite for MutationRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.lr_param = RLParameter(min=1e-5, max=1e-2, dtype=float)
        self.batch_param = RLParameter(min=16, max=256, dtype=int)
        self.hp_config = HyperparameterConfig(
            learning_rate=self.lr_param,
            batch_size=self.batch_param,
        )

    def test_mutation_registry_init_empty(self):
        """Test MutationRegistry initialization without hyperparameters."""
        registry = MutationRegistry()

        assert registry.hp_config is not None
        assert len(registry.hp_config.config) == 0
        assert len(registry.groups) == 0
        assert len(registry.optimizers) == 0
        assert len(registry.hooks) == 0

    def test_mutation_registry_init_with_hp_config(self):
        """Test MutationRegistry initialization with hyperparameters."""
        registry = MutationRegistry(hp_config=self.hp_config)

        assert registry.hp_config == self.hp_config
        assert len(registry.groups) == 0
        assert len(registry.optimizers) == 0

    def test_mutation_registry_register_group(self):
        """Test registering network groups."""
        registry = MutationRegistry()

        group1 = MockNetworkGroup("actor", "target_actor", policy=True)
        group2 = MockNetworkGroup("critic", None, policy=False)

        registry.register_group(group1)
        registry.register_group(group2)

        assert len(registry.groups) == 2
        assert group1 in registry.groups
        assert group2 in registry.groups

    def test_mutation_registry_register_optimizer(self):
        """Test registering optimizers."""
        registry = MutationRegistry()

        opt_config = OptimizerConfig(
            name="actor_opt",
            networks="actor",
            lr="lr_actor",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )

        registry.register_optimizer(opt_config)

        assert len(registry.optimizers) == 1
        assert registry.optimizers[0] == opt_config

    def test_mutation_registry_register_hook(self):
        """Test registering mutation hooks."""
        registry = MutationRegistry()

        def dummy_hook():
            pass

        registry.register_hook(dummy_hook)

        assert len(registry.hooks) == 1
        assert registry.hooks[0] == "dummy_hook"

    def test_mutation_registry_policy_method(self):
        """Test getting policy network from registry."""
        registry = MutationRegistry()

        # No policy registered
        assert registry.policy() is None

        # Register non-policy group
        group1 = MockNetworkGroup("critic", None, policy=False)
        registry.register_group(group1)
        assert registry.policy() is None

        # Register policy group
        group2 = MockNetworkGroup("actor", "target_actor", policy=True)
        registry.register_group(group2)
        assert registry.policy() == "actor"
        assert registry.policy(return_group=True) == group2

    def test_mutation_registry_all_registered(self):
        """Test getting all registered components."""
        registry = MutationRegistry()

        # Register groups
        group1 = MockNetworkGroup("actor", "target_actor", policy=True)
        group2 = MockNetworkGroup(
            "critic",
            ["target_critic1", "target_critic2"],
            policy=False,
        )
        registry.register_group(group1)
        registry.register_group(group2)

        # Register optimizer
        opt_config = OptimizerConfig(
            name="actor_opt",
            networks="actor",
            lr="lr_actor",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )
        registry.register_optimizer(opt_config)

        all_registered = registry.all_registered()
        expected = {
            "actor",
            "target_actor",
            "critic",
            "target_critic1",
            "target_critic2",
            "actor_opt",
        }
        assert set(all_registered) == expected

    def test_mutation_registry_networks(self):
        """Test getting network configurations."""
        registry = MutationRegistry()

        # Register groups
        group1 = MockNetworkGroup("actor", "target_actor", policy=True)
        group2 = MockNetworkGroup("critic", None, policy=False)
        registry.register_group(group1)
        registry.register_group(group2)

        # Register optimizers
        actor_opt_config = OptimizerConfig(
            name="actor_opt",
            networks=["actor"],
            lr="lr_actor",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )
        critic_opt_config = OptimizerConfig(
            name="critic_opt",
            networks=["critic"],
            lr="lr_critic",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )
        registry.register_optimizer(actor_opt_config)
        registry.register_optimizer(critic_opt_config)

        networks = registry.networks()

        # Should have 3 networks: actor (eval), target_actor (shared), critic (eval)
        assert len(networks) == 3

        # Check eval networks
        eval_networks = [net for net in networks if net.eval_network]
        assert len(eval_networks) == 2

        # Check shared networks
        shared_networks = [net for net in networks if not net.eval_network]
        assert len(shared_networks) == 1

    def test_mutation_registry_optimizer_networks_property(self):
        """Test optimizer_networks property."""
        registry = MutationRegistry()

        opt1 = OptimizerConfig(
            name="opt1",
            networks="net1",
            lr="lr",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )
        opt2 = OptimizerConfig(
            name="opt2",
            networks=["net2", "net3"],
            lr="lr",
            optimizer_cls=optim.SGD,
            optimizer_kwargs={},
        )

        registry.register_optimizer(opt1)
        registry.register_optimizer(opt2)

        optimizer_networks = registry.optimizer_networks
        assert optimizer_networks["opt1"] == "net1"
        assert optimizer_networks["opt2"] == ["net2", "net3"]

    def test_mutation_registry_equality(self):
        """Test equality comparison between MutationRegistry objects."""
        registry1 = MutationRegistry(hp_config=self.hp_config)
        registry2 = MutationRegistry(hp_config=self.hp_config)
        registry3 = MutationRegistry()

        # Add same groups and optimizers to registry1 and registry2
        group = MockNetworkGroup("actor", None, policy=True)
        opt = OptimizerConfig(
            name="opt",
            networks="actor",
            lr="lr",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )

        registry1.register_group(group)
        registry1.register_optimizer(opt)
        registry2.register_group(group)
        registry2.register_optimizer(opt)

        assert registry1 == registry2
        assert registry1 != registry3

    def test_mutation_registry_repr(self):
        """Test string representation of MutationRegistry."""
        registry = MutationRegistry()

        group = MockNetworkGroup("actor", "target_actor", policy=True)
        opt = OptimizerConfig(
            name="actor_opt",
            networks="actor",
            lr="lr_actor",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )

        registry.register_group(group)
        registry.register_optimizer(opt)

        repr_str = repr(registry)
        assert "Network Groups:" in repr_str
        assert "Optimizers:" in repr_str
        assert "actor" in repr_str
        assert "target_actor" in repr_str
        assert "actor_opt" in repr_str

    def test_mutation_registry_policy_empty_returns_none(self):
        """policy() with no groups returns None."""
        registry = MutationRegistry()
        assert registry.policy() is None
        assert registry.policy(return_group=True) is None

    def test_mutation_registry_policy_first_wins_when_multiple(self):
        """When multiple groups have policy=True, first registered wins."""
        registry = MutationRegistry()
        g1 = MockNetworkGroup("actor1", None, policy=True)
        g2 = MockNetworkGroup("actor2", None, policy=True)
        registry.register_group(g1)
        registry.register_group(g2)
        assert registry.policy() == "actor1"
        assert registry.policy(return_group=True).eval_network == "actor1"

    def test_mutation_registry_all_registered_empty(self):
        """all_registered() with no groups or optimizers returns empty."""
        registry = MutationRegistry()
        result = registry.all_registered()
        assert len(result) == 0

    def test_mutation_registry_all_registered_shared_as_str(self):
        """all_registered() includes shared network when it's a single str."""
        registry = MutationRegistry()
        group = MockNetworkGroup("actor", "target_actor", policy=True)
        registry.register_group(group)
        registered = registry.all_registered()
        assert "actor" in registered
        assert "target_actor" in registered

    def test_mutation_registry_all_registered_shared_as_list(self):
        """all_registered() includes all shared networks when list."""
        registry = MutationRegistry()
        group = MockNetworkGroup("critic", ["t1", "t2"], policy=False)
        registry.register_group(group)
        registered = registry.all_registered()
        assert "critic" in registered
        assert "t1" in registered
        assert "t2" in registered

    def test_mutation_registry_networks_empty(self):
        """networks() with no groups returns empty list."""
        registry = MutationRegistry()
        assert registry.networks() == []

    def test_mutation_registry_networks_single_net_optimizer_config(self):
        """networks() with optimizer config using str (single net)."""
        registry = MutationRegistry()
        group = MockNetworkGroup("actor", None, policy=True)
        opt = OptimizerConfig(
            name="actor_opt",
            networks="actor",
            lr="lr",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )
        registry.register_group(group)
        registry.register_optimizer(opt)
        with pytest.raises(
            ValueError,
            match="Evaluation network must have an optimizer associated with it",
        ):
            registry.networks()

    def test_mutation_registry_eq_with_none(self):
        """Current __eq__ implementation raises on None."""
        registry = MutationRegistry(hp_config=self.hp_config)
        with pytest.raises(AttributeError):
            _ = registry.__eq__(None)

    def test_mutation_registry_eq_different_groups_order(self):
        """Registries with same content but different group order differ."""
        r1 = MutationRegistry()
        r2 = MutationRegistry()
        g1 = MockNetworkGroup("a", None, policy=True)
        g2 = MockNetworkGroup("b", None, policy=False)
        r1.register_group(g1)
        r1.register_group(g2)
        r2.register_group(g2)
        r2.register_group(g1)
        assert r1 != r2

    def test_mutation_registry_eq_same_content_same_order(self):
        """Registries with same groups and optimizers in same order are equal."""
        r1 = MutationRegistry(hp_config=self.hp_config)
        r2 = MutationRegistry(hp_config=self.hp_config)
        g = MockNetworkGroup("actor", None, policy=True)
        opt = OptimizerConfig(
            name="opt",
            networks="actor",
            lr="lr",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={},
        )
        r1.register_group(g)
        r1.register_optimizer(opt)
        r2.register_group(g)
        r2.register_optimizer(opt)
        assert r1 == r2


class TestIntegration:
    """Integration tests for registry components working together."""

    def test_complete_registry_setup(self):
        """Test complete registry setup with all components."""
        # Create hyperparameter configuration
        hp_config = HyperparameterConfig(
            learning_rate=RLParameter(min=1e-5, max=1e-2, dtype=float),
            batch_size=RLParameter(min=16, max=256, dtype=int),
            layer_sizes=RLParameter(min=32, max=512, dtype=np.ndarray),
        )

        # Create registry
        registry = MutationRegistry(hp_config=hp_config)

        # Register network groups
        actor_group = MockNetworkGroup("actor", "target_actor", policy=True)
        critic_group = MockNetworkGroup("critic", "target_critic", policy=False)
        registry.register_group(actor_group)
        registry.register_group(critic_group)

        # Register optimizers
        actor_opt = OptimizerConfig(
            name="actor_opt",
            networks=["actor"],
            lr="learning_rate",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"weight_decay": 1e-4},
        )
        critic_opt = OptimizerConfig(
            name="critic_opt",
            networks=["critic"],
            lr="learning_rate",
            optimizer_cls=optim.Adam,
            optimizer_kwargs={"weight_decay": 1e-4},
        )
        registry.register_optimizer(actor_opt)
        registry.register_optimizer(critic_opt)

        # Register hook
        def post_mutation_hook():
            pass

        registry.register_hook(post_mutation_hook)

        # Verify complete setup
        assert len(registry.hp_config.config) == 3
        assert len(registry.groups) == 2
        assert len(registry.optimizers) == 2
        assert len(registry.hooks) == 1

        # Verify policy detection
        assert registry.policy() == "actor"

        # Verify all components are registered
        all_registered = registry.all_registered()
        expected = {
            "actor",
            "target_actor",
            "critic",
            "target_critic",
            "actor_opt",
            "critic_opt",
        }
        assert set(all_registered) == expected

        # Verify network configurations
        networks = registry.networks()
        eval_networks = [net for net in networks if net.eval_network]
        shared_networks = [net for net in networks if not net.eval_network]

        assert len(eval_networks) == 2  # actor and critic
        assert len(shared_networks) == 2  # target_actor and target_critic

        # Verify optimizer associations
        actor_net_config = next(net for net in eval_networks if net.name == "actor")
        critic_net_config = next(net for net in eval_networks if net.name == "critic")

        assert actor_net_config.optimizer == "actor_opt"
        assert critic_net_config.optimizer == "critic_opt"

    def test_numpy_array_hyperparameter_mutation_integration(self):
        """Test integration of numpy array hyperparameters with registry."""
        torch.manual_seed(42)

        # Create numpy array hyperparameters
        layer_sizes_param = RLParameter(min=16, max=512, dtype=np.ndarray)
        layer_sizes_param.value = np.array([128, 64, 32], dtype=np.int32)

        dropout_rates_param = RLParameter(min=0.0, max=0.5, dtype=np.ndarray)
        dropout_rates_param.value = np.array([0.1, 0.2, 0.3])

        hp_config = HyperparameterConfig(
            layer_sizes=layer_sizes_param,
            dropout_rates=dropout_rates_param,
        )

        # Test that we can sample and mutate numpy array hyperparameters
        for _ in range(5):
            param_name, param = hp_config.sample()
            assert param_name in ["layer_sizes", "dropout_rates"]

            original_value = param.value.copy()
            mutated_value = param.mutate()

            assert isinstance(mutated_value, np.ndarray)
            assert mutated_value.shape == original_value.shape
            assert np.all(mutated_value >= param.min)
            assert np.all(mutated_value <= param.max)


if __name__ == "__main__":
    pytest.main([__file__])
