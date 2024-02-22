import copy
import math
import warnings
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn


class EvolvableBERT(nn.Module):
    """The Evolvable BERT class.

    :param encoder_layers: Encoder layer(s) hidden size
    :type encoder_layers: list[int]
    :param decoder_layers: Decoder layer(s) hidden size
    :type decoder_layers: list[int]
    :param end2end: End to end transformer, using positional and token embeddings, defaults to True
    :type end2end: bool, optional
    :param src_vocab_size: Source vocabulary size, defaults to 10837
    :type src_vocab_size: int, optional
    :param tgt_vocab_size: Target vocabulary size, defaults to 10837
    :type tgt_vocab_size: int, optional
    :param encoder_norm: Encoder output normalization, defaults to True
    :type encoder_norm: bool, optional
    :param decoder_norm: Decoder output normalization, defaults to True
    :type decoder_norm: bool, optional
    :param d_model: Number of expected features in the encoder/decoder inputs, defaults to 512
    :type d_model: int, optional
    :param n_head: Number of heads in the multiheadattention models, defaults to 8
    :type n_head: int, optional
    :param dropout: Dropout value, defaults to 0.1
    :type dropout: float, optional
    :param activation: Activation function of encoder/decoder intermediate layer, defaults to 'ReLU'
    :type activation: str, optional
    :param layer_norm_eps: Epsilon value in layer normalization components, defaults to 1e-5
    :type layer_norm_eps: float, optional
    :param batch_first: Input/output tensor order. True:(batch, seq, feat.) False:(seq, batch, feat.). Defaults to False
    :type batch_first: bool, optional
    :param norm_first: Perform LayerNorm before other attention and feedforward operations, defaults to False
    :type norm_first: bool, optional
    :param max_encoder_layers: Maximum number of encoder layers, defaults to 12
    :type max_encoder_layers: int, optional
    :param max_decoder_layers: Maximum number of decoder layers, defaults to 12
    :type max_decoder_layers: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
        self,
        encoder_layers: List[int],
        decoder_layers: List[int],
        end2end: bool = True,
        src_vocab_size: int = 10837,
        tgt_vocab_size: int = 10837,
        encoder_norm: bool = True,
        decoder_norm: bool = True,
        d_model: int = 512,
        n_head: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        max_encoder_layers: int = 12,
        max_decoder_layers: int = 12,
        device="cpu",
    ):
        super().__init__()

        self.arch = "bert"
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.end2end = end2end
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder_norm = encoder_norm
        self.decoder_norm = decoder_norm
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.max_encoder_layers = max_encoder_layers
        self.max_decoder_layers = max_decoder_layers

        self.device = device

        if self.end2end:
            self.generator = nn.Linear(self.d_model, tgt_vocab_size)
            self.src_tok_emb = TokenEmbedding(src_vocab_size, self.d_model)
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, self.d_model)
            self.positional_encoding = PositionalEncoder(self.d_model, self.dropout)
        else:
            self.wte = TokenEmbedding(src_vocab_size, self.d_model)
            if len(self.encoder_layers) > 0:
                self.wpe = PositionalEncoding(self.d_model, self.encoder_layers[0])
            else:
                self.wpe = PositionalEncoding(self.d_model, self.decoder_layers[0])

        self.encoder, self.decoder = self.create_nets()
        self.encoder_keys = list(self.encoder.keys())
        self.decoder_keys = list(self.decoder.keys())

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
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
            "GELU": nn.GELU,
        }

        return activation_functions[activation_names]()

    def create_nets(self):
        """Creates and returns transformer neural network."""
        encoder_dict = OrderedDict()
        decoder_dict = OrderedDict()

        # Create the encoder
        for n, dim_feedfwd in enumerate(self.encoder_layers):
            encoder_dict[f"encoder_layer_{str(n)}"] = (
                nn.modules.TransformerEncoderLayer(
                    self.d_model,
                    self.n_head,
                    dim_feedfwd,
                    self.dropout,
                    self.activation,
                    self.layer_norm_eps,
                    self.batch_first,
                    self.norm_first,
                    device=self.device,
                )
            )
        if self.encoder_norm:
            encoder_dict["encoder_norm_0"] = nn.modules.normalization.LayerNorm(
                self.d_model, eps=self.layer_norm_eps, device=self.device
            )

        # Create the decoder
        for n, dim_feedfwd in enumerate(self.decoder_layers):
            decoder_dict[f"decoder_layer_{str(n)}"] = (
                nn.modules.TransformerDecoderLayer(
                    self.d_model,
                    self.n_head,
                    dim_feedfwd,
                    self.dropout,
                    self.activation,
                    self.layer_norm_eps,
                    self.batch_first,
                    self.norm_first,
                    device=self.device,
                )
            )
        if self.decoder_norm:
            decoder_dict["decoder_norm_0"] = nn.modules.normalization.LayerNorm(
                self.d_model, eps=self.layer_norm_eps, device=self.device
            )

        self._reset_parameters()

        return nn.ModuleDict(encoder_dict), nn.ModuleDict(decoder_dict)

    def generate_square_subsequent_mask(self, sz):
        """Returns a square mask for the sequence that prevents the model from looking into the future words when
        making predictions.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).

        :param sz: Size of mask to generate
        :type sz: int
        """
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt, pad_idx):
        """Returns masks to hide source and target padding tokens.

        :param src: Source
        :type src: torch.Tensor
        :param tgt: Target
        :type tgt: torch.Tensor
        :param pad_idx: Index of padding symbol <pad> in special symbols list
        :type pad_idx: int
        """
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )

        src_padding_mask = (src == pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
        return (
            src_mask.to(self.device),
            tgt_mask.to(self.device),
            src_padding_mask.to(self.device),
            tgt_padding_mask.to(self.device),
        )

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        is_causal=False,
    ):
        """Returns output of neural network.

        :param src: Encoder input sequence
        :type src: torch.Tensor
        :param tgt: Decoder input sequence
        :type tgt: torch.Tensor
        :param src_mask: Additive mask for the src sequence, defaults to None
        :type src_mask: torch.Tensor, optional
        :param tgt_mask: Additive mask for the tgt sequence, defaults to None
        :type tgt_mask: torch.Tensor, optional
        :param memory_mask: Additive mask for the encoder output, defaults to None
        :type memory_mask: torch.Tensor, optional
        :param src_key_padding_mask: Tensor mask for src keys per batch, defaults to None
        :type src_key_padding_mask: torch.Tensor, optional
        :param tgt_key_padding_mask: Tensor mask for tgt keys per batch, defaults to None
        :type tgt_key_padding_mask: torch.Tensor, optional
        :param memory_key_padding_mask: Tensor mask for memory keys per batch, defaults to None
        :type memory_key_padding_mask: torch.Tensor, optional
        :param is_causal: Applies a causal mask as mask and ignores attn_mask for computing scaled dot product attention, defaults to False
        :type is_causal: bool, optional
        """
        encoder_output, encoder_hidden_states = self.encode(
            src, src_mask, src_key_padding_mask, is_causal
        )
        memory = encoder_output
        decoder_output, decoder_hidden_states = self.decode(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )

        if self.end2end:
            decoder_output = self.generator(decoder_output)

        return decoder_output

    def encode(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """Returns encoded transformer input.

        :param src: Encoder input sequence
        :type src: torch.Tensor
        :param src_mask: Additive mask for the src sequence, defaults to None
        :type src_mask: torch.Tensor, optional
        :param src_key_padding_mask: Tensor mask for src keys per batch, defaults to None
        :type src_key_padding_mask: torch.Tensor, optional
        :param is_causal: Applies a causal mask as mask and ignores attn_mask for computing scaled dot product attention, defaults to False
        :type is_causal: bool, optional
        """
        if self.end2end:
            src = self.positional_encoding(self.src_tok_emb(src))

        # Encoder forward pass preparation
        src_key_padding_mask = _canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=_none_or_dtype(src_mask),
            other_name="mask",
            target_type=src.dtype,
        )
        encoder_output = src
        first_layer = self.encoder[self.encoder_keys[0]]
        str_first_layer = "self.net[0]"
        src_key_padding_mask_for_layers = src_key_padding_mask
        (
            encoder_output,
            convert_to_nested,
            src_key_padding_mask_for_layers,
        ) = self.check_encoder_sparsity_fast_path(
            src,
            encoder_output,
            first_layer,
            str_first_layer,
            src_mask,
            src_key_padding_mask,
            src_key_padding_mask_for_layers,
        )

        # Prevent type refinement
        make_causal = is_causal is True

        if is_causal is None:
            if src_mask is not None:
                sz = src_mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=src_mask.device) * float("-inf"),
                    diagonal=1,
                ).to(src_mask.dtype)
                if torch.equal(src_mask, causal_comparison):
                    make_causal = True
        is_causal = make_causal

        all_hidden_states = ()

        # Encoder forward pass
        for key in self.encoder_keys:
            if "norm" not in key:
                all_hidden_states = all_hidden_states + (encoder_output,)
                encoder_output = self.encoder[key](
                    encoder_output,
                    src_mask=src_mask,
                    is_causal=is_causal,
                    src_key_padding_mask=src_key_padding_mask_for_layers,
                )
        all_hidden_states = all_hidden_states + (encoder_output,)
        if convert_to_nested:
            encoder_output = encoder_output.to_padded_tensor(0.0)
            all_hidden_states = all_hidden_states + (encoder_output,)
        if "encoder_norm_0" in self.encoder_keys:
            encoder_output = self.encoder["encoder_norm_0"](encoder_output)
            all_hidden_states = all_hidden_states + (encoder_output,)
        return encoder_output, all_hidden_states

    def decode(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """Returns decoded transformer input.

        :param tgt: Decoder input sequence
        :type tgt: torch.Tensor
        :param memory: Encoder output sequence
        :type memory: torch.Tensory
        :param tgt_mask: Additive mask for the tgt sequence, defaults to None
        :type tgt_mask: torch.Tensor, optional
        :param memory_mask: Additive mask for the encoder output, defaults to None
        :type memory_mask: torch.Tensor, optional
        :param tgt_key_padding_mask: Tensor mask for tgt keys per batch, defaults to None
        :type tgt_key_padding_mask: torch.Tensor, optional
        :param memory_key_padding_mask: Tensor mask for memory keys per batch, defaults to None
        :type memory_key_padding_mask: torch.Tensor, optional
        """
        if self.end2end:
            tgt = self.positional_encoding(self.src_tok_emb(tgt))

        all_hidden_states = ()

        # Decoder forward pass
        decoder_output = tgt
        for key in self.decoder_keys:
            if "norm" not in key:
                all_hidden_states = all_hidden_states + (decoder_output,)
                decoder_output = self.decoder[key](
                    decoder_output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
        all_hidden_states = all_hidden_states + (decoder_output,)
        if "decoder_norm_0" in self.decoder_keys:
            decoder_output = self.decoder["decoder_norm_0"](decoder_output)
            all_hidden_states = all_hidden_states + (decoder_output,)
        return decoder_output, all_hidden_states

    def check_encoder_sparsity_fast_path(
        self,
        src,
        output,
        first_layer,
        str_first_layer,
        mask,
        src_key_padding_mask,
        src_key_padding_mask_for_layers,
    ):
        """Returns encoder output, conversion to nested and padding mask depending on if sparsity fast path possible.
        :param src: Encoder input sequence
        :type src: torch.Tensor
        :param output: Encoder output sequence
        :type output: torch.Tensor
        :param first_layer: First layer of encoder
        :type first_layer: torch.Module()
        :param str_first_layer: Name of first layer of encoder
        :type str_first_layer: str
        :param mask: Mask for the src sequence
        :type mask: torch.Tensor
        :param src_key_padding_mask: Tensor mask for src keys per batch
        :type src_key_padding_mask: torch.Tensor
        :param src_key_padding_mask_for_layers: Tensor mask for src keys per batch for layers
        :type src_key_padding_mask_for_layers: torch.Tensor
        """
        convert_to_nested = False
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            if first_layer.norm_first:
                if first_layer.training:
                    if first_layer.self_attn.batch_first:
                        if first_layer.self_attn._qkv_same_embed_dim:
                            if first_layer.norm1.eps == first_layer.norm2.eps:
                                if src.dim() == 3:
                                    if src_key_padding_mask is not None:
                                        if torch._nested_tensor_from_mask_left_aligned(
                                            src, src_key_padding_mask.logical_not()
                                        ):
                                            if not output.is_nested:
                                                if mask is None:
                                                    if (
                                                        first_layer.self_attn.num_heads
                                                        % 2
                                                        != 1
                                                    ):
                                                        if (
                                                            not torch.is_autocast_enabled()
                                                        ):
                                                            tensor_args = (
                                                                src,
                                                                first_layer.self_attn.in_proj_weight,
                                                                first_layer.self_attn.in_proj_bias,
                                                                first_layer.self_attn.out_proj.weight,
                                                                first_layer.self_attn.out_proj.bias,
                                                                first_layer.norm1.weight,
                                                                first_layer.norm1.bias,
                                                                first_layer.norm2.weight,
                                                                first_layer.norm2.bias,
                                                                first_layer.linear1.weight,
                                                                first_layer.linear1.bias,
                                                                first_layer.linear2.weight,
                                                                first_layer.linear2.bias,
                                                            )

                                                            if (
                                                                src.is_cuda
                                                                or "cpu"
                                                                in str(src.device)
                                                            ):
                                                                if torch.is_grad_enabled() and any(
                                                                    x.requires_grad
                                                                    for x in tensor_args
                                                                ):
                                                                    if (
                                                                        src_key_padding_mask
                                                                        is not None
                                                                    ):
                                                                        convert_to_nested = (
                                                                            True
                                                                        )
                                                                        output = torch._nested_tensor_from_mask(
                                                                            output,
                                                                            src_key_padding_mask.logical_not(),
                                                                            mask_check=False,
                                                                        )
                                                                        src_key_padding_mask_for_layers = (
                                                                            None
                                                                        )

        return output, convert_to_nested, src_key_padding_mask_for_layers

    def count_parameters(self, without_layer_norm=False):
        """Returns number of parameters in neural network.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                count += param.data.cpu().numpy().flatten().shape[0]
        return count

    @property
    def init_dict(self):
        """Returns model information in dictionary."""
        init_dict = {
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "end2end": self.end2end,
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "encoder_norm": self.encoder_norm,
            "decoder_norm": self.decoder_norm,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "dropout": self.dropout,
            "activation": self.activation,
            "layer_norm_eps": self.layer_norm_eps,
            "batch_first": self.batch_first,
            "norm_first": self.norm_first,
            "max_encoder_layers": self.max_encoder_layers,
            "max_decoder_layers": self.max_decoder_layers,
            "device": self.device,
        }
        return init_dict

    def add_encoder_layer(self):
        """Adds an encoder layer to transformer."""
        if len(self.encoder_layers) < self.max_encoder_layers:
            self.encoder_layers += [self.encoder_layers[-1]]
            self.recreate_nets()
        # else:
        #     self.add_node()

    def add_decoder_layer(self):
        """Adds a decoder layer to transformer."""
        if len(self.decoder_layers) < self.max_decoder_layers:
            self.decoder_layers += [self.decoder_layers[-1]]
            self.recreate_nets()
        # else:
        #     self.add_node()

    def remove_encoder_layer(self):
        """Removes an encoder layer from transformer."""
        if len(self.encoder_layers) > 1:
            self.encoder_layers = self.encoder_layers[:-1]
            self.recreate_shrunk_nets()
        # else:
        #     self.add_node()

    def remove_decoder_layer(self):
        """Removes a decoder layer from transformer."""
        if len(self.decoder_layers) > 1:
            self.decoder_layers = self.decoder_layers[:-1]
            self.recreate_shrunk_nets()
        # else:
        #     self.add_node()

    def add_node(self, network=None, hidden_layer=None, numb_new_nodes=None):
        """Adds nodes to hidden layer of encoder/decoder.

        :param network: Network to add node to, 'encoder' or 'decoder', defaults to None
        :type network: str, optional
        :param hidden_layer: Depth of hidden layer to add nodes to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to add to hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if network is None:
            network = np.random.choice(["encoder", "decoder"], 1)[0]
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]
        if network == "encoder":
            if hidden_layer is None:
                hidden_layer = np.random.randint(0, len(self.encoder_layers), 1)[0]
            else:
                hidden_layer = min(hidden_layer, len(self.encoder_layers) - 1)

            self.encoder_layers[hidden_layer] += numb_new_nodes
        else:
            if hidden_layer is None:
                hidden_layer = np.random.randint(0, len(self.decoder_layers), 1)[0]
            else:
                hidden_layer = min(hidden_layer, len(self.decoder_layers) - 1)

            self.decoder_layers[hidden_layer] += numb_new_nodes
        self.recreate_nets()
        return {
            "hidden_layer": hidden_layer,
            "numb_new_nodes": numb_new_nodes,
            "network": network,
        }

    def remove_node(self, network=None, hidden_layer=None, numb_new_nodes=None):
        """Removes nodes from hidden layer of encoder/decoder.

        :param network: Network to remove node from, 'encoder' or 'decoder', defaults to None
        :type network: str, optional
        :param hidden_layer: Depth of hidden layer to remove nodes from, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to remove from hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if network is None:
            network = np.random.choice(["encoder", "decoder"], 1)[0]
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]
        if network == "encoder":
            if hidden_layer is None:
                hidden_layer = np.random.randint(0, len(self.encoder_layers), 1)[0]
            else:
                hidden_layer = min(hidden_layer, len(self.encoder_layers) - 1)
            if self.encoder_layers[hidden_layer] - numb_new_nodes > 64:  # HARD LIMIT
                self.encoder_layers[hidden_layer] -= numb_new_nodes
        else:
            if hidden_layer is None:
                hidden_layer = np.random.randint(0, len(self.decoder_layers), 1)[0]
            else:
                hidden_layer = min(hidden_layer, len(self.decoder_layers) - 1)
            if self.decoder_layers[hidden_layer] - numb_new_nodes > 64:  # HARD LIMIT
                self.decoder_layers[hidden_layer] -= numb_new_nodes
        self.recreate_shrunk_nets()
        return {
            "hidden_layer": hidden_layer,
            "numb_new_nodes": numb_new_nodes,
            "network": network,
        }

    def recreate_nets(self):
        """Recreates neural networks."""
        new_encoder, new_decoder = self.create_nets()
        new_encoder = self.preserve_parameters(
            old_net=self.encoder, new_net=new_encoder
        )
        new_decoder = self.preserve_parameters(
            old_net=self.decoder, new_net=new_decoder
        )
        self.encoder, self.decoder = new_encoder, new_decoder

    def recreate_shrunk_nets(self):
        """Recreates shrunk neural networks."""
        new_encoder, new_decoder = self.create_nets()
        new_encoder = self.shrink_preserve_parameters(
            old_net=self.encoder, new_net=new_encoder
        )
        new_decoder = self.shrink_preserve_parameters(
            old_net=self.decoder, new_net=new_decoder
        )
        self.encoder, self.decoder = new_encoder, new_decoder

    def clone(self):
        """Returns clone of neural net with identical parameters."""
        clone = EvolvableBERT(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(self.state_dict())
        return clone

    def preserve_parameters(self, old_net, new_net):
        """Returns new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        if len(param.data.size()) == 1:
                            param.data[: min(old_size[0], new_size[0])] = old_net_dict[
                                key
                            ].data[: min(old_size[0], new_size[0])]
                        else:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ]

        return new_net

    def shrink_preserve_parameters(self, old_net, new_net):
        """Returns shrunk new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        min_0 = min(old_size[0], new_size[0])
                        if len(param.data.size()) == 1:
                            param.data[:min_0] = old_net_dict[key].data[:min_0]
                        else:
                            min_1 = min(old_size[1], new_size[1])
                            param.data[:min_0, :min_1] = old_net_dict[key].data[
                                :min_0, :min_1
                            ]
        return new_net


def _canonical_mask(
    mask, mask_name, other_type, other_name, target_type, check_other=True
):
    """Returns canconical mask. Adapted from torch.nn.functional"""
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(
                mask, float("-inf")
            )
    return mask


class PositionalEncoder(nn.Module):
    """The Positional Encoder class.
    Adds positional encoding to the token embedding to introduce a notion of word order.

    :param emb_size: Number of expected features
    :type emb_size: int
    :param dropout: Dropout value, defaults to 0.1
    :type dropout: float, optional
    :param maxlen: Maximum length of sequence, defaults to 5000
    :type maxlen: int, optional
    """

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x: torch.Tensor):
        """Forward pass through positional encoder.
        :param x: Input to positional encoder, shape [seq_len, batch_size, embedding_dim]
        :type x: torch.Tensor
        """
        return self.dropout(x + self.pos_embedding[: x.size(0), :])


class PositionalEncoding(nn.Module):
    """The positional embedding class.
    Converts tensor of input indices into corresponding tensor of position embeddings.
    """

    def __init__(self, max_positions: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(max_positions, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        """Forward pass through position embedding module.
        :param tokens: Tokens to embed
        :type tokens: torch.Tensor
        """
        return self.embedding(tokens)


class TokenEmbedding(nn.Module):
    """The token embedding class. Converts tensor of input indices into corresponding tensor of token embeddings."""

    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        """Forward pass through token embedding module.
        :param tokens: Tokens to embed
        :type tokens: torch.Tensor
        """
        # return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        return self.embedding(tokens)


def _none_or_dtype(input):
    """Returns None or dtype of input. Adapted from torch.nn.functional.
    :param input: Input to return dtype of
    :type input: Any
    """
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")
