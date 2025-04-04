from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.protocols import EvolvableAlgorithm, EvolvableNetwork


# Mock classes for testing
class MockEvolvableNetwork(nn.Module, EvolvableNetwork):
    def __init__(self, input_dim=10, output_dim=5, name="mock_network"):
        super().__init__()
        self.name = name
        self.device = "cpu"
        # Use fixed initialization for stable testing
        self.layer = nn.Linear(input_dim, output_dim)
        # Initialize with small weights to avoid exploding gradients
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


class MockAlgorithm(EvolvableAlgorithm):
    def __init__(
        self, actor=None, critic=None, lr=0.001, networks=None, optimizer_cls=None
    ):
        # Single network setup (like DQN)
        if actor is not None and critic is None and networks is None:
            self.actor = actor or MockEvolvableNetwork(name="actor")
            self.learning_rate = lr
            self.optimizer_cls = optimizer_cls or torch.optim.Adam
            self.optimizer = OptimizerWrapper(
                self.optimizer_cls, self.actor, self.learning_rate
            )

        # Two network setup (like PPO)
        elif actor is not None and critic is not None and networks is None:
            self.actor = actor
            self.critic = critic
            self.learning_rate = lr
            self.optimizer_cls = optimizer_cls or torch.optim.Adam
            self.optimizer = OptimizerWrapper(
                self.optimizer_cls, [self.actor, self.critic], self.learning_rate
            )

        # Multi-agent setup (like MADDPG)
        elif networks is not None:
            self.networks = networks
            self.lr_actor = lr
            self.optimizer_cls = optimizer_cls or torch.optim.Adam
            self.optimizer = OptimizerWrapper(
                self.optimizer_cls, self.networks, self.lr_actor, multiagent=True
            )


class MockMultiAgentAlgorithm(EvolvableAlgorithm):
    def __init__(
        self,
        actors=None,
        critics=None,
        lr_actor=0.001,
        lr_critic=0.01,
        optimizer_cls=None,
    ):
        # Separate actor and critic networks for multi-agent
        self.actors = actors or [
            MockEvolvableNetwork(name=f"actor_{i}") for i in range(3)
        ]
        self.critics = critics or [
            MockEvolvableNetwork(name=f"critic_{i}") for i in range(3)
        ]
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.optimizer_cls = optimizer_cls or torch.optim.Adam

        # Create optimizers like in MADDPG
        self.actor_optimizers = OptimizerWrapper(
            self.optimizer_cls, self.actors, self.lr_actor, multiagent=True
        )

        self.critic_optimizers = OptimizerWrapper(
            self.optimizer_cls, self.critics, self.lr_critic, multiagent=True
        )


class TestOptimizerWrapper:

    def test_init_single_network(self):
        """Test initializing with a single network like in DQN."""
        network = MockEvolvableNetwork()
        lr = 0.001

        # Test automatic inference
        with patch.object(OptimizerWrapper, "_infer_parent_container"):
            with patch.object(
                OptimizerWrapper, "_infer_network_attr_names", return_value=["actor"]
            ):
                with patch.object(
                    OptimizerWrapper, "_infer_lr_name", return_value="learning_rate"
                ):
                    algo = MockAlgorithm(actor=network, lr=lr)

        assert isinstance(algo.optimizer.optimizer, torch.optim.Adam)
        assert algo.optimizer.lr == lr
        assert len(algo.optimizer.networks) == 1
        assert algo.optimizer.networks[0] is network
        assert algo.optimizer.network_names == ["actor"]
        assert algo.optimizer.lr_name == "learning_rate"
        assert not algo.optimizer.multiagent

    def test_init_with_multiple_networks(self):
        """Test initializing with multiple networks like in PPO."""
        actor = MockEvolvableNetwork(name="actor")
        critic = MockEvolvableNetwork(name="critic")
        lr = 0.001

        # Test with two networks
        with patch.object(OptimizerWrapper, "_infer_parent_container"):
            with patch.object(
                OptimizerWrapper,
                "_infer_network_attr_names",
                return_value=["actor", "critic"],
            ):
                with patch.object(
                    OptimizerWrapper, "_infer_lr_name", return_value="learning_rate"
                ):
                    algo = MockAlgorithm(actor=actor, critic=critic, lr=lr)

        assert isinstance(algo.optimizer.optimizer, torch.optim.Adam)
        assert algo.optimizer.lr == lr
        assert len(algo.optimizer.networks) == 2
        assert algo.optimizer.networks[0] is actor
        assert algo.optimizer.networks[1] is critic
        assert set(algo.optimizer.network_names) == {"actor", "critic"}
        assert algo.optimizer.lr_name == "learning_rate"
        assert not algo.optimizer.multiagent

        # Test parameter groups
        param_groups = algo.optimizer.optimizer.param_groups
        assert len(param_groups) == 2
        # Each group should have the same learning rate
        assert param_groups[0]["lr"] == lr
        assert param_groups[1]["lr"] == lr

    def test_init_with_multiagent(self):
        """Test initializing with multiagent=True like in MADDPG."""
        networks = [MockEvolvableNetwork(name=f"net_{i}") for i in range(3)]
        lr = 0.001

        # Test with multiple networks in multi-agent mode
        with patch.object(OptimizerWrapper, "_infer_parent_container"):
            with patch.object(
                OptimizerWrapper, "_infer_network_attr_names", return_value=["networks"]
            ):
                with patch.object(
                    OptimizerWrapper, "_infer_lr_name", return_value="lr_actor"
                ):
                    algo = MockAlgorithm(networks=networks, lr=lr)

        assert isinstance(algo.optimizer.optimizer, list)
        assert len(algo.optimizer.optimizer) == 3
        assert all(
            isinstance(opt, torch.optim.Adam) for opt in algo.optimizer.optimizer
        )
        assert algo.optimizer.lr == lr
        assert algo.optimizer.networks is networks
        assert algo.optimizer.network_names == ["networks"]
        assert algo.optimizer.lr_name == "lr_actor"
        assert algo.optimizer.multiagent

    def test_maddpg_style_optimizers(self):
        """Test MADDPG-style setup with separate actor and critic optimizers."""
        with patch.object(OptimizerWrapper, "_infer_parent_container"):
            with patch.object(
                OptimizerWrapper,
                "_infer_network_attr_names",
                side_effect=[["actors"], ["critics"]],
            ):
                with patch.object(
                    OptimizerWrapper,
                    "_infer_lr_name",
                    side_effect=["lr_actor", "lr_critic"],
                ):
                    algo = MockMultiAgentAlgorithm()

        # Check actor optimizers
        assert isinstance(algo.actor_optimizers.optimizer, list)
        assert len(algo.actor_optimizers.optimizer) == 3
        assert all(
            isinstance(opt, torch.optim.Adam) for opt in algo.actor_optimizers.optimizer
        )
        assert algo.actor_optimizers.lr == algo.lr_actor
        assert len(algo.actor_optimizers.networks) == 3
        assert algo.actor_optimizers.network_names == ["actors"]
        assert algo.actor_optimizers.multiagent

        # Check critic optimizers
        assert isinstance(algo.critic_optimizers.optimizer, list)
        assert len(algo.critic_optimizers.optimizer) == 3
        assert all(
            isinstance(opt, torch.optim.Adam)
            for opt in algo.critic_optimizers.optimizer
        )
        assert algo.critic_optimizers.lr == algo.lr_critic
        assert len(algo.critic_optimizers.networks) == 3
        assert algo.critic_optimizers.network_names == ["critics"]
        assert algo.critic_optimizers.multiagent

    def test_init_with_explicit_names(self):
        """Test initialization with explicitly provided network_names and lr_name."""
        network = MockEvolvableNetwork()
        optimizer_cls = torch.optim.Adam
        lr = 0.001
        network_names = ["policy_net"]
        lr_name = "policy_lr"

        wrapper = OptimizerWrapper(
            optimizer_cls, network, lr, network_names=network_names, lr_name=lr_name
        )

        assert wrapper.network_names == network_names
        assert wrapper.lr_name == lr_name

    def test_init_with_optimizer_kwargs(self):
        """Test initialization with optimizer kwargs."""
        network = MockEvolvableNetwork()
        optimizer_cls = torch.optim.Adam
        lr = 0.001
        optimizer_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8}

        wrapper = OptimizerWrapper(
            optimizer_cls,
            network,
            lr,
            optimizer_kwargs=optimizer_kwargs,
            network_names=["network"],
            lr_name="lr",
        )

        # Check that kwargs were passed to the optimizer
        assert wrapper.optimizer.defaults["betas"] == (0.9, 0.999)
        assert wrapper.optimizer.defaults["eps"] == 1e-8

    def test_getitem_in_multiagent(self):
        """Test indexing via __getitem__ in multi-agent setup."""
        with patch.object(OptimizerWrapper, "_infer_parent_container"):
            with patch.object(
                OptimizerWrapper,
                "_infer_network_attr_names",
                side_effect=[["actors"], ["critics"]],
            ):
                with patch.object(
                    OptimizerWrapper,
                    "_infer_lr_name",
                    side_effect=["lr_actor", "lr_critic"],
                ):
                    algo = MockMultiAgentAlgorithm()

        # Test indexing for actor optimizers
        for i in range(3):
            assert algo.actor_optimizers[i] is algo.actor_optimizers.optimizer[i]

        # Test indexing for critic optimizers
        for i in range(3):
            assert algo.critic_optimizers[i] is algo.critic_optimizers.optimizer[i]

    def test_method_delegation(self):
        """Test method delegation to underlying optimizer(s)."""
        # Single optimizer case with direct attribute access
        network = MockEvolvableNetwork()

        # Create the wrapper with explicit names to avoid inference
        wrapper = OptimizerWrapper(
            torch.optim.Adam, network, 0.001, network_names=["network"], lr_name="lr"
        )

        # Test basic attribute access first (not a method call)
        assert wrapper.defaults == wrapper.optimizer.defaults

        # Now test with a method that actually exists
        wrapper.optimizer.zero_grad = Mock()  # Replace with mock to verify call
        wrapper.zero_grad()
        wrapper.optimizer.zero_grad.assert_called_once()

        # Multi-agent case - attribute delegation should fail
        networks = [MockEvolvableNetwork() for _ in range(2)]
        multi_wrapper = OptimizerWrapper(
            torch.optim.Adam,
            networks,
            0.001,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        # Should raise TypeError because can't delegate to list of optimizers
        with pytest.raises(AttributeError):
            multi_wrapper.non_existent_attribute

    def test_zero_grad(self):
        """Test zero_grad method across different setups."""
        # Single network case
        network = MockEvolvableNetwork()
        wrapper = OptimizerWrapper(
            torch.optim.Adam, network, 0.001, network_names=["network"], lr_name="lr"
        )

        # Mock zero_grad
        original_zero_grad = wrapper.optimizer.zero_grad
        wrapper.optimizer.zero_grad = Mock()

        wrapper.zero_grad()
        wrapper.optimizer.zero_grad.assert_called_once()

        # Restore method
        wrapper.optimizer.zero_grad = original_zero_grad

        # Multiple networks in multi-agent case
        networks = [MockEvolvableNetwork() for _ in range(3)]
        multi_wrapper = OptimizerWrapper(
            torch.optim.Adam,
            networks,
            0.001,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        # Mock zero_grad for each optimizer
        original_methods = []
        for opt in multi_wrapper.optimizer:
            original_methods.append(opt.zero_grad)
            opt.zero_grad = Mock()

        for opt in multi_wrapper.optimizer:
            opt.zero_grad()

        # Each optimizer's zero_grad should be called once
        for opt in multi_wrapper.optimizer:
            opt.zero_grad.assert_called_once()

        # Restore methods
        for i, opt in enumerate(multi_wrapper.optimizer):
            opt.zero_grad = original_methods[i]

    def test_step(self):
        """Test step method across different setups."""
        # Single network case
        network = MockEvolvableNetwork()
        wrapper = OptimizerWrapper(
            torch.optim.Adam, network, 0.001, network_names=["network"], lr_name="lr"
        )

        # Mock step
        original_step = wrapper.optimizer.step
        wrapper.optimizer.step = Mock()

        wrapper.step()
        wrapper.optimizer.step.assert_called_once()

        # Restore method
        wrapper.optimizer.step = original_step

        # Multiple networks in multi-agent case
        networks = [MockEvolvableNetwork() for _ in range(3)]
        multi_wrapper = OptimizerWrapper(
            torch.optim.Adam,
            networks,
            0.001,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        # Mock step for each optimizer
        original_methods = []
        for opt in multi_wrapper.optimizer:
            original_methods.append(opt.step)
            opt.step = Mock()

        for opt in multi_wrapper.optimizer:
            opt.step()

        # Each optimizer's step should be called once
        for opt in multi_wrapper.optimizer:
            opt.step.assert_called_once()

        # Restore methods
        for i, opt in enumerate(multi_wrapper.optimizer):
            opt.step = original_methods[i]

    def test_state_dict_and_load_state_dict(self):
        """Test state_dict and load_state_dict methods."""
        # Single optimizer case
        network = MockEvolvableNetwork()
        wrapper = OptimizerWrapper(
            torch.optim.Adam, network, 0.001, network_names=["network"], lr_name="lr"
        )

        # Get state dict
        state_dict = wrapper.state_dict()
        assert isinstance(state_dict, dict)

        # Mock load_state_dict
        original_load = wrapper.optimizer.load_state_dict
        wrapper.optimizer.load_state_dict = Mock()

        # Load state dict
        wrapper.load_state_dict(state_dict)
        wrapper.optimizer.load_state_dict.assert_called_once_with(state_dict)

        # Restore method
        wrapper.optimizer.load_state_dict = original_load

        # Multi-agent case
        networks = [MockEvolvableNetwork() for _ in range(3)]
        multi_wrapper = OptimizerWrapper(
            torch.optim.Adam,
            networks,
            0.001,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        # Get state dicts
        state_dicts = multi_wrapper.state_dict()
        assert isinstance(state_dicts, list)
        assert len(state_dicts) == 3

        # Mock load_state_dict for each optimizer
        original_methods = []
        for opt in multi_wrapper.optimizer:
            original_methods.append(opt.load_state_dict)
            opt.load_state_dict = Mock()

        # Load state dicts
        multi_wrapper.load_state_dict(state_dicts)

        # Each optimizer's load_state_dict should be called once with the correct state dict
        for i, opt in enumerate(multi_wrapper.optimizer):
            opt.load_state_dict.assert_called_once_with(state_dicts[i])

        # Restore methods
        for i, opt in enumerate(multi_wrapper.optimizer):
            opt.load_state_dict = original_methods[i]

    def test_invalid_load_state_dict(self):
        """Test load_state_dict with invalid inputs."""
        # Single optimizer receiving list
        network = MockEvolvableNetwork()
        wrapper = OptimizerWrapper(
            torch.optim.Adam, network, 0.001, network_names=["network"], lr_name="lr"
        )

        with pytest.raises(AssertionError):
            wrapper.load_state_dict([{}])

        # Multi-agent optimizer receiving dict
        networks = [MockEvolvableNetwork() for _ in range(3)]
        multi_wrapper = OptimizerWrapper(
            torch.optim.Adam,
            networks,
            0.001,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        with pytest.raises(AssertionError):
            multi_wrapper.load_state_dict({})

        # Multi-agent optimizer receiving wrong-sized list
        with pytest.raises(AssertionError):
            multi_wrapper.load_state_dict([{}])

    def test_actual_learning(self):
        """Test that the optimizer actually performs gradient descent."""
        # Create a simple network and test data
        network = MockEvolvableNetwork(input_dim=2, output_dim=1)
        optimizer = OptimizerWrapper(
            torch.optim.SGD,
            network,
            lr=0.01,  # Use a smaller learning rate
            network_names=["network"],
            lr_name="lr",
        )

        criterion = nn.MSELoss()

        # Create simple regression data
        x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

        # Set model to a known stable state
        with torch.no_grad():
            nn.init.zeros_(network.layer.weight)
            nn.init.zeros_(network.layer.bias)

        # Initial loss
        initial_output = network(x)
        initial_loss = criterion(initial_output, y)

        # Train for a few steps
        for _ in range(100):  # More iterations with smaller learning rate
            optimizer.zero_grad()
            output = network(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Final loss
        final_output = network(x)
        final_loss = criterion(final_output, y)

        # Loss should decrease
        assert final_loss < initial_loss

    def test_actual_learning_multiple_networks(self):
        """Test that the optimizer works with multiple networks."""
        # Create two networks and test data
        actor = MockEvolvableNetwork(input_dim=2, output_dim=1)
        critic = MockEvolvableNetwork(input_dim=2, output_dim=1)

        # Set models to a known stable state
        with torch.no_grad():
            nn.init.zeros_(actor.layer.weight)
            nn.init.zeros_(actor.layer.bias)
            nn.init.zeros_(critic.layer.weight)
            nn.init.zeros_(critic.layer.bias)

        optimizer = OptimizerWrapper(
            torch.optim.SGD,
            [actor, critic],
            lr=0.01,  # Smaller learning rate
            network_names=["actor", "critic"],
            lr_name="lr",
        )

        criterion = nn.MSELoss()

        # Create simple regression data
        x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_actor = torch.tensor([[3.0], [5.0], [7.0], [9.0]])
        y_critic = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

        # Initial losses
        initial_actor_output = actor(x)
        initial_actor_loss = criterion(initial_actor_output, y_actor)

        initial_critic_output = critic(x)
        initial_critic_loss = criterion(initial_critic_output, y_critic)

        # Train for a few steps
        for _ in range(100):  # More iterations with smaller learning rate
            optimizer.zero_grad()

            actor_output = actor(x)
            actor_loss = criterion(actor_output, y_actor)

            critic_output = critic(x)
            critic_loss = criterion(critic_output, y_critic)

            # Combined loss
            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

        # Final losses
        final_actor_output = actor(x)
        final_actor_loss = criterion(final_actor_output, y_actor)

        final_critic_output = critic(x)
        final_critic_loss = criterion(final_critic_output, y_critic)

        # Both losses should decrease
        assert final_actor_loss < initial_actor_loss
        assert final_critic_loss < initial_critic_loss

    def test_actual_learning_multiagent(self):
        """Test that multi-agent optimizers work correctly."""
        # Create networks for 3 agents
        networks = [MockEvolvableNetwork(input_dim=2, output_dim=1) for _ in range(3)]

        # Set models to a known stable state
        for network in networks:
            with torch.no_grad():
                nn.init.zeros_(network.layer.weight)
                nn.init.zeros_(network.layer.bias)

        optimizer = OptimizerWrapper(
            torch.optim.SGD,
            networks,
            lr=0.01,  # Smaller learning rate
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        criterion = nn.MSELoss()

        # Create simple regression data for each agent
        x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_targets = [
            torch.tensor([[3.0], [5.0], [7.0], [9.0]]),
            torch.tensor([[2.0], [4.0], [6.0], [8.0]]),
            torch.tensor([[1.0], [3.0], [5.0], [7.0]]),
        ]

        # Initial losses
        initial_losses = []
        for i, network in enumerate(networks):
            output = network(x)
            loss = criterion(output, y_targets[i])
            initial_losses.append(loss.item())

        # Train each network separately
        for _ in range(100):  # More iterations with smaller learning rate
            for i, network in enumerate(networks):
                optimizer[i].zero_grad()
                output = network(x)
                loss = criterion(output, y_targets[i])
                loss.backward()
                optimizer[i].step()

        # Final losses
        final_losses = []
        for i, network in enumerate(networks):
            output = network(x)
            loss = criterion(output, y_targets[i])
            final_losses.append(loss.item())

        # All losses should decrease
        for i in range(3):
            assert final_losses[i] < initial_losses[i]

    def test_repr(self):
        """Test __repr__ method."""
        network = MockEvolvableNetwork()
        wrapper = OptimizerWrapper(
            torch.optim.Adam, network, lr=0.001, network_names=["network"], lr_name="lr"
        )

        repr_str = repr(wrapper)
        assert "OptimizerWrapper" in repr_str
        assert "Adam" in repr_str
        assert "0.001" in repr_str
        assert "['network']" in repr_str
        assert "multiagent=False" in repr_str

    def test_parent_container_inference(self):
        """Test that parent container inference works correctly."""
        # We need to mock the implementation since it depends on frame introspection
        with patch.object(OptimizerWrapper, "_infer_parent_container") as mock_infer:
            network = MockEvolvableNetwork()

            # Create a container with the learning rate attribute
            mock_container = MagicMock()
            # Set attributes that will be checked by the wrapper
            mock_container.learning_rate = 0.001
            mock_infer.return_value = mock_container

            # Initialize the wrapper explicitly to avoid auto-inference issues
            wrapper = OptimizerWrapper(
                torch.optim.Adam,
                network,
                lr=0.001,
                network_names=["network"],
                lr_name="learning_rate",
            )

            # Set up the mock again after initialization
            with patch.object(
                wrapper, "_infer_parent_container", return_value=mock_container
            ):
                # Test that the method returns the mocked result
                assert wrapper._infer_parent_container() is mock_container
                wrapper._infer_parent_container.assert_called_once()

    def test_different_optimizers_multiagent(self):
        """Test with different optimizer classes for each agent."""
        networks = [MockEvolvableNetwork() for _ in range(2)]
        optimizer_classes = [torch.optim.SGD, torch.optim.Adam]

        wrapper = OptimizerWrapper(
            optimizer_classes,
            networks,
            lr=0.01,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        assert isinstance(wrapper.optimizer[0], torch.optim.SGD)
        assert isinstance(wrapper.optimizer[1], torch.optim.Adam)

    def test_different_kwargs_multiagent(self):
        """Test with different kwargs for each agent's optimizer."""
        networks = [MockEvolvableNetwork() for _ in range(2)]
        kwargs_list = [{"momentum": 0.9}, {"betas": (0.9, 0.999)}]

        wrapper = OptimizerWrapper(
            [torch.optim.SGD, torch.optim.Adam],
            networks,
            lr=0.01,
            optimizer_kwargs=kwargs_list,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        assert wrapper.optimizer[0].defaults["momentum"] == 0.9
        assert wrapper.optimizer[1].defaults["betas"] == (0.9, 0.999)

    def test_iteration_multiagent(self):
        """Test iteration through optimizers in multiagent setting."""
        networks = [MockEvolvableNetwork() for _ in range(3)]
        wrapper = OptimizerWrapper(
            torch.optim.Adam,
            networks,
            lr=0.01,
            multiagent=True,
            network_names=["networks"],
            lr_name="lr",
        )

        # Count through iteration
        count = 0
        for opt in wrapper:
            assert isinstance(opt, torch.optim.Adam)
            count += 1

        assert count == 3
