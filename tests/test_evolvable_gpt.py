import numpy as np
import torch
import torch.nn as nn

from agilerl.networks.evolvable_gpt import (
    MLP,
    CausalSelfAttention,
    EvolvableGPT,
    PositionalEncoding,
    TokenEmbedding,
    new_gelu,
)

#### TESTING EvolvableGPT CLASS ####


# The model can be initialized with default parameters.
def test_default_parameters_initialization():
    model = EvolvableGPT()
    assert model.n_layer == 12
    assert model.vocab_size == 50257
    assert model.n_embd == 768
    assert model.n_head == 12
    assert model.dim_feedfwd == 3072
    assert model.block_size == 1024
    assert model.dropout == 0.0
    assert model.activation == "GELU"
    assert model.layer_norm_eps == 1e-05
    assert model.min_layers == 8
    assert model.max_layers == 16
    assert model.bias is True
    assert model.device == "cpu"


# The model can be initialized with custom parameters.
def test_custom_parameters_initialization():
    model = EvolvableGPT(
        n_layer=6,
        vocab_size=10000,
        n_embd=512,
        n_head=8,
        dim_feedfwd=2048,
        block_size=512,
        dropout=0.2,
        activation="ReLU",
        layer_norm_eps=1e-06,
        min_layers=4,
        max_layers=10,
        bias=False,
        device="cpu",
    )
    assert model.n_layer == 6
    assert model.vocab_size == 10000
    assert model.n_embd == 512
    assert model.n_head == 8
    assert model.dim_feedfwd == 2048
    assert model.block_size == 512
    assert model.dropout == 0.2
    assert model.activation == "ReLU"
    assert model.layer_norm_eps == 1e-06
    assert model.min_layers == 4
    assert model.max_layers == 10
    assert model.bias is False
    assert model.device == "cpu"


# The model can be loaded from a pretrained GPT model.
def test_pretrained_model_loading():
    model = EvolvableGPT.from_pretrained("gpt2", override_args={"dropout": 0.1})
    assert model.n_layer == 12
    assert model.vocab_size == 50257
    assert model.n_embd == 768
    assert model.n_head == 12
    assert model.dim_feedfwd == 3072
    assert model.block_size == 1024
    assert model.dropout == 0.1
    assert model.activation == "GELU"
    assert model.layer_norm_eps == 1e-05
    assert model.min_layers == 8
    assert model.max_layers == 16
    assert model.bias is True
    assert model.device == "cpu"


# Returns nn.ReLU for activation name "ReLU".
def test_returns_activation():
    activation_functions = {
        "Tanh": nn.Tanh,
        "Identity": nn.Identity,
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "Softsign": nn.Softsign,
        "Sigmoid": nn.Sigmoid,
        "Softplus": nn.Softplus,
        "Softmax": nn.Softmax,
        "LeakyReLU": nn.LeakyReLU,
        "PReLU": nn.PReLU,
        "GELU": new_gelu,
    }

    model = EvolvableGPT()
    for activation in activation_functions.keys():
        activation_function = model.get_activation(activation)
        assert isinstance(activation_function, activation_functions[activation])


# Configures optimizers for EvoGPT
def test_configure_optimizers():
    weight_decay = 0.0
    learning_rate = 1e-6
    betas = (0.9, 0.999)

    model = EvolvableGPT()
    optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, "cpu")

    assert isinstance(optimizer, torch.optim.AdamW)


# The model can handle a sequence of tokens with a target for training.
def test_sequence_with_target_handling():
    model = EvolvableGPT()
    input_sequence = torch.randint(0, model.vocab_size, (1, model.block_size))
    target_sequence = torch.randint(0, model.vocab_size, (1, model.block_size))
    logits, all_hidden_states, presents, loss = model(
        input_sequence, targets=target_sequence
    )
    assert logits.shape[1] == model.block_size
    assert loss is not None

    tok_emb = model.transformer.wte(input_sequence)
    t = tok_emb.size(-2)
    # past_key_values = tuple([None] * model.n_layer)
    past_length = 0
    pos = torch.arange(
        past_length, t + past_length, dtype=torch.long, device=tok_emb.device
    ).unsqueeze(0)
    logits, all_hidden_states, presents, loss = model(tok_emb=tok_emb, pos=pos)
    assert logits.shape[1] == model.block_size
    assert loss is None


# The model can handle a sequence of tokens without a target for generation.
def test_sequence_without_target_handling():
    model = EvolvableGPT()
    input_sequence = torch.randint(0, model.vocab_size, (1, model.block_size))
    generated_sequence = model.generate(input_sequence, max_new_tokens=10, top_k=3)
    assert generated_sequence.shape[1] == model.block_size + 10


# Decrease block size successfully
def test_decrease_block_size_successfully():
    block_size = 512
    model = EvolvableGPT()
    assert model.block_size == 1024

    # Manually override flash attention
    for block in model.transformer.h:
        block.attn.flash = False
        block.attn.register_buffer(
            "attention_bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    model.crop_block_size(block_size)
    assert model.block_size == block_size
    assert model.transformer.wpe.weight.shape[0] == block_size
    for block in model.transformer.h:
        if hasattr(block.attn, "attention_bias"):
            assert block.attn.attention_bias.shape[2] == block_size
            assert block.attn.attention_bias.shape[3] == block_size


# The model can estimate the MFU (Million Floating Point Operations per Second).
def test_estimate_mfu():
    model = EvolvableGPT()
    fwdbwd_per_iter = 10
    dt = 0.1
    mfu = model.estimate_mfu(fwdbwd_per_iter, dt)
    assert isinstance(mfu, float)
    assert mfu >= 0.0


# The model can generate new tokens based on an input sequence.
def test_generate_new_tokens():
    model = EvolvableGPT()
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    max_new_tokens = 5
    temperature = 1.0
    top_k = None

    generated_tokens = model.generate(idx, max_new_tokens, temperature, top_k)

    assert generated_tokens.size() == (1, 10)


# The model can forward pass a sequence of tokens and return the logits.
def test_forward_pass():
    model = EvolvableGPT()
    input_tokens = torch.tensor([[1, 2, 3, 4, 5]])
    B = 1  # Batch size
    C = 768  # Embedding dim

    k = torch.randn((B, 12, 0, C // 12))
    v = torch.randn((B, 12, 0, C // 12))
    past_kv = [(k, v) for _ in range(12)]
    logits, _, _, _ = model(input_tokens, past_key_values=past_kv)
    assert logits.shape == (1, 5, model.vocab_size)


# The model can count the number of parameters.
def test_count_parameters():
    model = EvolvableGPT()
    num_params = model.get_num_params()
    assert isinstance(num_params, int)
    assert num_params >= 0


# Adds a layer to transformer
def test_add_layer():
    model = EvolvableGPT()
    initial_n_layer = model.n_layer
    model.add_layer()
    assert model.n_layer == initial_n_layer + 1
    assert len(model.transformer.h) == initial_n_layer + 1


# Removes a layer to transformer
def test_remove_layer():
    model = EvolvableGPT()
    initial_n_layer = model.n_layer
    model.remove_layer()
    assert model.n_layer == initial_n_layer - 1
    assert len(model.transformer.h) == initial_n_layer - 1


# Adds nodes to transformer
def test_add_nodes():
    model = EvolvableGPT()
    initial_dim_feedfwd = model.dim_feedfwd
    model.add_node()
    assert model.dim_feedfwd > initial_dim_feedfwd
    for block in model.transformer.h:
        assert block.mlp.hidden_size[0] > initial_dim_feedfwd


# Removes nodes to transformer
def test_remove_nodes():
    model = EvolvableGPT()
    initial_dim_feedfwd = model.dim_feedfwd
    model.remove_node()
    assert model.dim_feedfwd < initial_dim_feedfwd
    for block in model.transformer.h:
        assert block.mlp.hidden_size[0] < initial_dim_feedfwd


# The model can clone itself.
def test_model_clone():
    model = EvolvableGPT()
    clone = model.clone()
    assert isinstance(clone, EvolvableGPT)
    assert clone.n_layer == model.n_layer
    assert clone.vocab_size == model.vocab_size
    assert clone.n_embd == model.n_embd
    assert clone.n_head == model.n_head
    assert clone.dim_feedfwd == model.dim_feedfwd
    assert clone.block_size == model.block_size
    assert clone.dropout == model.dropout
    assert clone.activation == model.activation
    assert clone.layer_norm_eps == model.layer_norm_eps
    assert clone.min_layers == model.min_layers
    assert clone.max_layers == model.max_layers
    assert clone.bias == model.bias
    assert clone.device == model.device


#### TESTING CAUSAL SELF ATTENTION CLASS ####
def test_causal_self_attention_forward():
    block_size = 1024
    attn = CausalSelfAttention(768, 12, True, 0.1, block_size)
    attn.flash = False
    attn.register_buffer(
        "attention_bias",
        torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        ),
    )
    B = 4  # Batch size
    T = 128  # Sequence length
    C = 768  # Embedding dim
    x = torch.randint(0, 32, (B, T, C)).float()

    k = torch.randn((B, 12, 0, C // 12))
    v = torch.randn((B, 12, 0, C // 12))
    layer_past = (k, v)

    y, present = attn(x, layer_past=layer_past)

    assert isinstance(y, torch.Tensor)
    assert isinstance(present[0], torch.Tensor)
    assert isinstance(present[1], torch.Tensor)


#### TESTING MLP CLASS ####
# Returns nn.ReLU for activation name "ReLU".
def test_returns_activation_mlp():
    activation_functions = {
        "Tanh": nn.Tanh,
        "Identity": nn.Identity,
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "Softsign": nn.Softsign,
        "Sigmoid": nn.Sigmoid,
        "Softplus": nn.Softplus,
        "Softmax": nn.Softmax,
        "LeakyReLU": nn.LeakyReLU,
        "PReLU": nn.PReLU,
        "GELU": new_gelu,
    }

    model = MLP(32, 0.1, 64)
    for activation in activation_functions.keys():
        activation_function = model.get_activation(activation)
        assert isinstance(activation_function, activation_functions[activation])


def test_forward_mlp():
    input_array = np.random.rand(1, 32)
    model = MLP(32, 0.1, 64)

    output = model(input_array)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 32)


#### TESTING POSITIONAL ENCODING CLASS ####
def test_pos_encoding():
    input_tensor = torch.LongTensor([[1, 2, 4, 5]])
    pos = PositionalEncoding(10, 3)
    enc = pos(input_tensor)

    assert enc.shape == (1, 4, 3)


#### TESTING TOKEN EMBEDDING CLASS ####
def test_tok_embedding():
    input_tensor = torch.LongTensor([[1, 2, 4, 5]])
    tok = TokenEmbedding(10, 3)
    emb = tok(input_tensor)

    assert emb.shape == (1, 4, 3)
