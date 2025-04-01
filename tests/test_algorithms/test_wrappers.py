from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
import torch.optim as optim

from agilerl.algorithms.core.wrappers import OptimizerWrapper


class MockNetwork(nn.Module):
    """Simple mock network for testing optimizers"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

    def clone(self):
        return MockNetwork()

    def init_dict(self):
        return {}


class DummyAlgorithm:
    """Simulates an algorithm class that uses OptimizerWrapper"""

    def __init__(self):
        self.actor = MockNetwork()
        self.critic = MockNetwork()
        self.learning_rate = 0.01

        # OptimizerWrapper created in __init__
        self.optimizer = OptimizerWrapper(
            optimizer_cls=optim.Adam, networks=self.actor, lr=self.learning_rate
        )

    def create_optimizer_outside_init(self):
        """Create an optimizer outside of __init__"""
        # Without explicit attribute names, the inference will fail
        # We need to patch the _infer_parent_container method to make
        # sure it fails as expected in our test
        return OptimizerWrapper(
            optimizer_cls=optim.Adam, networks=self.critic, lr=self.learning_rate
        )

    def create_optimizer_outside_init_with_names(self):
        """Create an optimizer outside of __init__ with explicit attribute names"""
        return OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=self.critic,
            lr=self.learning_rate,
            network_names=["critic"],
            lr_name="learning_rate",
        )


class TestOptimizerWrapper:

    @pytest.fixture
    def mock_parent(self):
        """Mock parent container with network and optimizer attributes"""
        parent = MagicMock()
        parent.network1 = MockNetwork()
        parent.network2 = MockNetwork()
        parent.networks = [parent.network1, parent.network2]
        parent.lr = 0.01
        return parent

    def test_single_optimizer_single_network(self, mock_parent):
        """Test initialization of a single optimizer for a single network"""
        with patch(
            "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_parent_container",
            return_value=mock_parent,
        ):
            with patch(
                "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_network_attr_names",
                return_value=["network1"],
            ):
                with patch(
                    "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_lr_name",
                    return_value="lr",
                ):
                    wrapper = OptimizerWrapper(
                        optimizer_cls=optim.Adam,
                        networks=mock_parent.network1,
                        lr=mock_parent.lr,
                    )

                    # Check optimizer was initialized correctly
                    assert wrapper.optimizer_cls == optim.Adam
                    assert wrapper.lr == mock_parent.lr
                    assert wrapper.networks == [mock_parent.network1]
                    assert wrapper.network_names == ["network1"]
                    assert wrapper.lr_name == "lr"
                    assert wrapper.multiagent is False
                    assert isinstance(wrapper.optimizer, optim.Adam)

    def test_single_optimizer_multiple_networks(self, mock_parent):
        """Test initialization of a single optimizer for multiple networks"""
        with patch(
            "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_parent_container",
            return_value=mock_parent,
        ):
            with patch(
                "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_network_attr_names",
                return_value=["network1", "network2"],
            ):
                with patch(
                    "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_lr_name",
                    return_value="lr",
                ):
                    wrapper = OptimizerWrapper(
                        optimizer_cls=optim.Adam,
                        networks=[mock_parent.network1, mock_parent.network2],
                        lr=mock_parent.lr,
                    )

                    # Check optimizer was initialized correctly
                    assert wrapper.optimizer_cls == optim.Adam
                    assert wrapper.lr == mock_parent.lr
                    assert wrapper.networks == [
                        mock_parent.network1,
                        mock_parent.network2,
                    ]
                    assert wrapper.network_names == ["network1", "network2"]
                    assert wrapper.lr_name == "lr"
                    assert wrapper.multiagent is False
                    assert isinstance(wrapper.optimizer, optim.Adam)

    def test_multi_agent_optimizers(self, mock_parent):
        """Test initialization of multiple optimizers for multi-agent setup"""
        with patch(
            "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_parent_container",
            return_value=mock_parent,
        ):
            with patch(
                "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_network_attr_names",
                return_value=["networks"],
            ):
                with patch(
                    "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_lr_name",
                    return_value="lr",
                ):
                    wrapper = OptimizerWrapper(
                        optimizer_cls=optim.Adam,
                        networks=[mock_parent.network1, mock_parent.network2],
                        lr=mock_parent.lr,
                        multiagent=True,
                    )

                    # Check optimizers were initialized correctly
                    assert wrapper.optimizer_cls == optim.Adam
                    assert wrapper.lr == mock_parent.lr
                    assert wrapper.networks == [
                        mock_parent.network1,
                        mock_parent.network2,
                    ]
                    assert wrapper.network_names == ["networks"]
                    assert wrapper.lr_name == "lr"
                    assert wrapper.multiagent is True
                    assert isinstance(wrapper.optimizer, list)
                    assert len(wrapper.optimizer) == 2
                    assert all(isinstance(opt, optim.Adam) for opt in wrapper.optimizer)

    def test_explicit_network_names(self):
        """Test explicitly providing network_names and lr_name"""
        network = MockNetwork()
        wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=network,
            lr=0.01,
            network_names=["actor"],
            lr_name="actor_lr",
        )

        assert wrapper.network_names == ["actor"]
        assert wrapper.lr_name == "actor_lr"

    def test_missing_lr_name(self):
        """Test error when network_names is provided but lr_name is not"""
        network = MockNetwork()
        with pytest.raises(
            AssertionError, match="Learning rate attribute name must be passed"
        ):
            OptimizerWrapper(
                optimizer_cls=optim.Adam,
                networks=network,
                lr=0.01,
                network_names=["actor"],
                lr_name=None,
            )

    def test_state_dict_single_optimizer(self):
        """Test state_dict() and load_state_dict() for single optimizer"""
        network1 = MockNetwork()
        network2 = MockNetwork()

        # Create two wrappers with the same structure
        wrapper1 = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=network1,
            lr=0.01,
            network_names=["network"],
            lr_name="lr",
        )

        wrapper2 = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=network2,
            lr=0.02,  # Different learning rate
            network_names=["network"],
            lr_name="lr",
        )

        # Get state from wrapper1 and load into wrapper2
        state = wrapper1.state_dict()
        wrapper2.load_state_dict(state)

        # Both should now have the same state
        assert (
            wrapper1.optimizer.state_dict()["param_groups"][0]["lr"]
            == wrapper2.optimizer.state_dict()["param_groups"][0]["lr"]
        )

    def test_state_dict_multi_agent(self):
        """Test state_dict() and load_state_dict() for multi-agent optimizers"""
        network1a = MockNetwork()
        network1b = MockNetwork()
        network2a = MockNetwork()
        network2b = MockNetwork()

        # Create two wrappers with multi-agent setup
        wrapper1 = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=[network1a, network1b],
            lr=0.01,
            network_names=["networks"],
            lr_name="lr",
            multiagent=True,
        )

        wrapper2 = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=[network2a, network2b],
            lr=0.02,  # Different learning rate
            network_names=["networks"],
            lr_name="lr",
            multiagent=True,
        )

        # Get state from wrapper1 and load into wrapper2
        state = wrapper1.state_dict()
        wrapper2.load_state_dict(state)

        # Both should now have the same state
        for i in range(2):
            assert (
                wrapper1.optimizer[i].state_dict()["param_groups"][0]["lr"]
                == wrapper2.optimizer[i].state_dict()["param_groups"][0]["lr"]
            )

    def test_operation_methods_single_optimizer(self):
        """Test zero_grad() and step() methods for single optimizer"""
        network = MockNetwork()
        wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=network,
            lr=0.01,
            network_names=["network"],
            lr_name="lr",
        )

        # Mock the optimizer methods
        wrapper.optimizer.zero_grad = MagicMock()
        wrapper.optimizer.step = MagicMock()

        # Call wrapper methods
        wrapper.zero_grad()
        wrapper.step()

        # Assert optimizer methods were called
        wrapper.optimizer.zero_grad.assert_called_once()
        wrapper.optimizer.step.assert_called_once()

    def test_operation_methods_multi_agent(self):
        """Test that zero_grad() and step() raise ValueError for multi-agent optimizers"""
        network1 = MockNetwork()
        network2 = MockNetwork()

        wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=[network1, network2],
            lr=0.01,
            network_names=["networks"],
            lr_name="lr",
            multiagent=True,
        )

        # Methods should raise ValueError
        with pytest.raises(ValueError, match="Please use the zero_grad()"):
            wrapper.zero_grad()

        with pytest.raises(ValueError, match="Please use the step()"):
            wrapper.step()

    def test_wrong_state_dict_format(self):
        """Test that wrong state_dict format raises errors"""
        network = MockNetwork()
        multiagent_networks = [MockNetwork(), MockNetwork()]

        # Single optimizer
        single_wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=network,
            lr=0.01,
            network_names=["network"],
            lr_name="lr",
        )

        # Multi-agent optimizers
        multi_wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=multiagent_networks,
            lr=0.01,
            network_names=["networks"],
            lr_name="lr",
            multiagent=True,
        )

        # Test loading list state_dict into single optimizer
        with pytest.raises(
            AssertionError, match="Expected a single optimizer state dictionary"
        ):
            single_wrapper.load_state_dict([{}])

        # Test loading dict state_dict into multi-agent optimizers
        with pytest.raises(
            AssertionError, match="Expected a list of optimizer state dictionaries"
        ):
            multi_wrapper.load_state_dict({})

    def test_getitem_access(self):
        """Test __getitem__ access for multi-agent optimizers"""
        networks = [MockNetwork(), MockNetwork()]

        wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=networks,
            lr=0.01,
            network_names=["networks"],
            lr_name="lr",
            multiagent=True,
        )

        # Should be able to access individual optimizers
        assert isinstance(wrapper[0], optim.Adam)
        assert isinstance(wrapper[1], optim.Adam)

        # Single optimizer should not support indexing
        single_wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=networks[0],
            lr=0.01,
            network_names=["network"],
            lr_name="lr",
        )

        with pytest.raises(TypeError, match="Can't access item of a single"):
            _ = single_wrapper[0]

    def test_optimizer_iteration(self):
        """Test iteration over optimizers in multi-agent case"""
        networks = [MockNetwork(), MockNetwork()]

        wrapper = OptimizerWrapper(
            optimizer_cls=optim.Adam,
            networks=networks,
            lr=0.01,
            network_names=["networks"],
            lr_name="lr",
            multiagent=True,
        )

        # Should be able to iterate over optimizers
        count = 0
        for opt in wrapper:
            assert isinstance(opt, optim.Adam)
            count += 1

        assert count == 2

    def test_optimizer_wrapper_in_algorithm_init(self):
        """Test that OptimizerWrapper works correctly when created in __init__ method"""
        algorithm = DummyAlgorithm()

        # Optimizer created in __init__ should have inferred attribute names
        assert algorithm.optimizer.network_names == ["actor"]
        assert algorithm.optimizer.lr_name == "learning_rate"
        assert isinstance(algorithm.optimizer.optimizer, optim.Adam)

    def test_optimizer_wrapper_outside_init_fails(self):
        """Test that OptimizerWrapper fails when created outside __init__ without explicit names"""
        algorithm = DummyAlgorithm()

        # When created outside __init__, the inference methods should fail
        # We need to mock the _infer_parent_container method to simulate the failure
        with patch(
            "agilerl.algorithms.core.wrappers.OptimizerWrapper._infer_parent_container"
        ) as mock_infer:
            # Make the _infer_parent_container method raise an AttributeError
            mock_infer.side_effect = AttributeError(
                "Cannot infer parent container outside __init__"
            )

            # Now the call should raise AttributeError
            with pytest.raises(AttributeError):
                algorithm.create_optimizer_outside_init()

    def test_optimizer_wrapper_outside_init_with_names(self):
        """Test that OptimizerWrapper works outside __init__ when given explicit attribute names"""
        algorithm = DummyAlgorithm()

        # When created outside __init__ with explicit names, it should work
        optimizer = algorithm.create_optimizer_outside_init_with_names()

        assert optimizer.network_names == ["critic"]
        assert optimizer.lr_name == "learning_rate"
        assert isinstance(optimizer.optimizer, optim.Adam)
