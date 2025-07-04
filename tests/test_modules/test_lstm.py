import copy

import pytest
import torch

from agilerl.modules.lstm import EvolvableLSTM
from tests.helper_functions import assert_state_dicts_equal

######### Define fixtures #########
# Device fixture moved to conftest.py


######### Test instantiation #########
@pytest.mark.parametrize(
    "input_size, hidden_size, num_outputs, num_layers",
    [(10, 64, 5, 1), (2, 32, 1, 2), (100, 128, 3, 3)],
)
def test_instantiation(input_size, hidden_size, num_outputs, num_layers, device):
    evolvable_lstm = EvolvableLSTM(
        input_size=input_size,
        hidden_state_size=hidden_size,
        num_outputs=num_outputs,
        num_layers=num_layers,
        device=device,
    )
    assert isinstance(evolvable_lstm, EvolvableLSTM)


@pytest.mark.parametrize(
    "input_size, hidden_size, num_outputs, num_layers",
    [(0, 64, 5, 1), (10, 0, 5, 1), (10, 64, 0, 1), (10, 64, 5, 0)],
)
def test_incorrect_instantiation(
    input_size, hidden_size, num_outputs, num_layers, device
):
    with pytest.raises(AssertionError):
        EvolvableLSTM(
            input_size=input_size,
            hidden_state_size=hidden_size,
            num_outputs=num_outputs,
            num_layers=num_layers,
            device=device,
        )


@pytest.mark.parametrize(
    "output_activation",
    ["Softmax", "Tanh", "Sigmoid", "ReLU", None],
)
def test_instantiation_with_different_activations(output_activation, device):
    evolvable_lstm = EvolvableLSTM(
        input_size=10,
        hidden_state_size=64,
        num_outputs=4,
        output_activation=output_activation,
        device=device,
    )
    assert isinstance(evolvable_lstm, EvolvableLSTM)


######### Test forward #########
def test_forward_with_states(device):
    input_size = 10
    hidden_size = 64
    num_outputs = 5
    num_layers = 2
    batch_size = 1
    seq_len = 5

    evolvable_lstm = EvolvableLSTM(
        input_size=input_size,
        hidden_state_size=hidden_size,
        num_outputs=num_outputs,
        num_layers=num_layers,
        device=device,
    )

    input_tensor = torch.randn(batch_size, seq_len, input_size).to(device)
    h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

    with torch.no_grad():
        # Pass hidden states to forward manually
        lstm_output, (hn, cn) = evolvable_lstm.model[f"{evolvable_lstm.name}_lstm"](
            input_tensor, (h0, c0)
        )
        output = evolvable_lstm.model[f"{evolvable_lstm.name}_lstm_output"](
            lstm_output[:, -1, :]
        )
        output = evolvable_lstm.model[f"{evolvable_lstm.name}_output_activation"](
            output
        )

        # Regular forward pass
        output_tensor, (hn, cn) = evolvable_lstm.forward(
            input_tensor,
            hidden_state={
                f"{evolvable_lstm.name}_h": h0,
                f"{evolvable_lstm.name}_c": c0,
            },
        )

    assert output_tensor.shape == (batch_size, num_outputs)
    torch.testing.assert_close(output, output_tensor)


######### Test add_lstm_layer #########
@pytest.mark.parametrize(
    "input_size, hidden_size, num_outputs, num_layers",
    [(10, 64, 5, 1), (2, 32, 1, 2), (100, 128, 3, 1)],
)
def test_add_layer(input_size, hidden_size, num_outputs, num_layers, device):
    evolvable_lstm = EvolvableLSTM(
        input_size=input_size,
        hidden_state_size=hidden_size,
        num_outputs=num_outputs,
        num_layers=num_layers,
        max_layers=3,
        device=device,
    )

    initial_layers = evolvable_lstm.num_layers
    initial_net = evolvable_lstm.model
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_lstm.add_layer()
    new_net = evolvable_lstm.model

    if initial_layers < 3:  # max_layers = 3
        assert evolvable_lstm.num_layers == initial_layers + 1
        # Check only non-LSTM parameters remain the same (output layer)
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys() and "lstm" not in key:
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert evolvable_lstm.num_layers == initial_layers


######### Test remove_lstm_layer #########
@pytest.mark.parametrize(
    "input_size, hidden_size, num_outputs, num_layers",
    [(10, 64, 5, 2), (2, 32, 1, 3), (100, 128, 3, 2)],
)
def test_remove_layer(input_size, hidden_size, num_outputs, num_layers, device):
    evolvable_lstm = EvolvableLSTM(
        input_size=input_size,
        hidden_state_size=hidden_size,
        num_outputs=num_outputs,
        num_layers=num_layers,
        min_layers=1,
        device=device,
    )

    initial_layers = evolvable_lstm.num_layers
    initial_net = evolvable_lstm.model
    initial_net_dict = dict(initial_net.named_parameters())
    evolvable_lstm.remove_layer()
    new_net = evolvable_lstm.model

    if initial_layers > 1:  # min_layers = 1
        assert evolvable_lstm.num_layers == initial_layers - 1
        # Check only non-LSTM parameters remain the same (output layer)
        for key, param in new_net.named_parameters():
            if key in initial_net_dict.keys() and "lstm" not in key:
                torch.testing.assert_close(param, initial_net_dict[key])
    else:
        assert evolvable_lstm.num_layers == initial_layers


######### Test add_lstm_node #########
@pytest.mark.parametrize(
    "input_size, hidden_size, num_outputs, num_layers, numb_new_nodes",
    [
        (10, 64, 5, 1, 32),
        (2, 32, 1, 2, None),
        (100, 128, 3, 3, 64),
    ],
)
def test_add_nodes(
    input_size, hidden_size, num_outputs, num_layers, numb_new_nodes, device
):
    lstm = EvolvableLSTM(
        input_size=input_size,
        hidden_state_size=hidden_size,
        num_outputs=num_outputs,
        num_layers=num_layers,
        device=device,
    )
    original_hidden_size = copy.deepcopy(lstm.hidden_state_size)
    result = lstm.add_node(numb_new_nodes=numb_new_nodes)
    if numb_new_nodes is None:
        numb_new_nodes = result["numb_new_nodes"]
    assert lstm.hidden_state_size == original_hidden_size + numb_new_nodes


######### Test remove_lstm_node #########
@pytest.mark.parametrize(
    "input_size, hidden_size, num_outputs, num_layers, numb_new_nodes",
    [
        (10, 256, 5, 1, 32),
        (2, 128, 1, 2, None),
        (100, 256, 3, 3, 64),
    ],
)
def test_remove_nodes(
    input_size, hidden_size, num_outputs, num_layers, numb_new_nodes, device
):
    lstm = EvolvableLSTM(
        input_size=input_size,
        hidden_state_size=hidden_size,
        num_outputs=num_outputs,
        num_layers=num_layers,
        min_hidden_state_size=32,
        device=device,
    )
    original_hidden_size = copy.deepcopy(lstm.hidden_state_size)
    result = lstm.remove_node(numb_new_nodes=numb_new_nodes)
    if numb_new_nodes is None:
        numb_new_nodes = result["numb_new_nodes"]
    assert lstm.hidden_state_size == original_hidden_size - numb_new_nodes


######### Test activation change #########
def test_change_activation(device):
    lstm = EvolvableLSTM(
        input_size=10,
        hidden_state_size=64,
        num_outputs=5,
        num_layers=1,
        output_activation=None,
        device=device,
    )

    # Test changing output activation
    activations = ["Sigmoid", "Tanh", "ReLU"]
    for activation in activations:
        lstm.change_activation(activation, output=True)
        assert lstm.output_activation == activation

        # Verify the activation works through forward pass
        input_tensor = torch.randn(1, 5, 10).to(device)
        h0 = torch.zeros(1, 1, 64).to(device)
        c0 = torch.zeros(1, 1, 64).to(device)
        output, (hn, cn) = lstm.forward(
            input_tensor,
            hidden_state={
                f"{lstm.name}_h": h0,
                f"{lstm.name}_c": c0,
            },
        )
        assert output.shape == (1, 5)


######### Test clone #########
@pytest.mark.parametrize(
    "input_size, hidden_size, num_outputs, num_layers",
    [(10, 64, 5, 1), (2, 32, 1, 2), (100, 128, 3, 3)],
)
def test_clone_instance(input_size, hidden_size, num_outputs, num_layers, device):
    evolvable_lstm = EvolvableLSTM(
        input_size=input_size,
        hidden_state_size=hidden_size,
        num_outputs=num_outputs,
        num_layers=num_layers,
        device=device,
    )
    original_net_dict = dict(evolvable_lstm.model.named_parameters())
    clone = evolvable_lstm.clone()
    clone_net = clone.model

    assert isinstance(clone, EvolvableLSTM)
    assert clone.init_dict == evolvable_lstm.init_dict
    assert_state_dicts_equal(clone.state_dict(), evolvable_lstm.state_dict())

    for key, param in clone_net.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key]), evolvable_lstm
