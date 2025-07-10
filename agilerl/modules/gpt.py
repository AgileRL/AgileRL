import inspect
import math
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.modules.mlp import EvolvableMLP
from agilerl.typing import DeviceType


class EvolvableGPT(EvolvableModule):
    """The Evolvable GPT class.

    :param n_layer: Number of transformer block layers, defaults to 12
    :type encoder_layers: int, optional
    :param vocab_size: Vocabulary size, defaults to 50257
    :type vocab_size: int, optional
    :param n_embd: Transformer embedding dimension size, defaults to 768
    :type n_embd: int, optional
    :param n_head: Number of heads in the multiheadattention models, defaults to 12
    :type n_head: int, optional
    :param dim_feedfwd: Size of transformer block hidden layer, defaults to 3072 (4*768)
    :type dim_feedfwd: int, optional
    :param block_size: Transformer block context size, defaults to 1024
    :type block_size: int, optional
    :param dropout: Dropout value, defaults to 0.0
    :type dropout: float, optional
    :param activation: Activation function of transformer intermediate layer, defaults to 'GELU'
    :type activation: str, optional
    :param layer_norm_eps: Epsilon value in layer normalization components, defaults to 1e-5
    :type layer_norm_eps: float, optional
    :param min_layers: Minimum number of transformer block layers, defaults to 8
    :type min_layers: int, optional
    :param max_layers: Maximum number of transformer block layers, defaults to 16
    :type max_layers: int, optional
    :param bias: Use bias in Linears and LayerNorms, defaults to True
    :type bias: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """

    arch: str = "gpt"

    def __init__(
        self,
        n_layer: int = 12,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_head: int = 12,
        dim_feedfwd: int = 3072,
        block_size: int = 1024,
        dropout: float = 0.0,
        activation: str = "GELU",
        layer_norm_eps: float = 1e-5,
        min_layers: int = 8,
        max_layers: int = 16,
        bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__(device)

        assert isinstance(n_layer, int), "Number of layers must be an integer."
        assert n_layer >= 1, "Number of layers must be greater than or equal to one."
        assert isinstance(vocab_size, int), "Vocabulary size must be an integer."
        assert vocab_size >= 1, "Vocabulary size must be greater than or equal to one."
        assert isinstance(n_embd, int), "Embedding dimension must be an integer."
        assert n_embd >= 1, "Embedding dimension must be greater than or equal to one."
        assert isinstance(n_head, int), "Number of attention heads must be an integer."
        assert (
            n_head >= 1
        ), "Number of attention heads must be greater than or equal to one."
        assert isinstance(
            dim_feedfwd, int
        ), "Feed forward dimension must be an integer."
        assert (
            dim_feedfwd >= 1
        ), "Feed forward dimension must be greater than or equal to one."
        assert isinstance(block_size, int), "Block size must be an integer."
        assert block_size >= 1, "Block size must be greater than or equal to one."
        assert isinstance(dropout, (float, int)), "Dropout must be a float."
        assert 0 <= dropout <= 1, "Dropout must be between zero and one (inclusive)."
        assert isinstance(layer_norm_eps, float), "Layer norm epsilon must be a float."
        assert layer_norm_eps > 0, "Layer norm epsilon must be greater than zero."
        assert isinstance(
            min_layers, int
        ), "Minimum number of layers must be an integer."
        assert (
            min_layers >= 1
        ), "Minimum number of layers must be greater than or equal to one."
        assert isinstance(
            max_layers, int
        ), "Maximum number of layers must be an integer."
        assert (
            max_layers >= 1
        ), "Maximum number of layers must be greater than or equal to one."
        assert (
            max_layers >= min_layers
        ), "Maximum number of layers must be greater than or equal to minimum number of layers."
        assert isinstance(bias, bool), "Bias flag must be boolean value True or False."

        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.dim_feedfwd = dim_feedfwd
        self.block_size = block_size
        self.dropout = dropout
        self._activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.bias = bias

        self.transformer = self.build_networks()
        self.transformer_keys = list(self.transformer.keys())

        self.lm_head = nn.Linear(
            self.n_embd, self.vocab_size, bias=False, device=self.device
        )

        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
                )

        # report number of parameters
        # print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    @property
    def activation(self) -> str:
        """Return activation function."""
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        """Set activation function."""
        self._activation = activation

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        :param non_embedding: If True, subtracts the position embeddings from the count, defaults to True
        :type non_embedding: bool, optional
        :return: Number of parameters in the model
        :rtype: int
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model.

        :param module: The module to initialize
        :type module: nn.Module
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def build_networks(self) -> nn.ModuleDict:
        """Creates and returns transformer neural network.

        :return: Transformer neural network as a ModuleDict
        :rtype: nn.ModuleDict
        """
        net_dict = OrderedDict()
        net_dict["wte"] = nn.Embedding(self.vocab_size, self.n_embd, device=self.device)
        net_dict["wpe"] = nn.Embedding(self.block_size, self.n_embd, device=self.device)
        net_dict["drop"] = nn.Dropout(self.dropout, inplace=True)
        net_dict["h"] = nn.ModuleList(
            [
                Block(
                    self.n_embd,
                    self.n_head,
                    self.bias,
                    self.dropout,
                    self.block_size,
                    self.dim_feedfwd,
                    self.activation,
                    self.layer_norm_eps,
                    device=self.device,
                )
                for _ in range(self.n_layer)
            ]
        )
        net_dict["ln_f"] = LayerNorm(self.n_embd, bias=self.bias, device=self.device)
        return nn.ModuleDict(net_dict)

    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        tok_emb: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        pos: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Forward pass through evolvable GPT model.

        :param idx: Input ids
        :type idx: torch.Tensor, optional
        :param tok_emb: Token embeddings
        :type tok_emb: torch.Tensor, optional
        :param targets: Target ids
        :type targets: torch.Tensor, optional
        :param attn_mask: Attention mask
        :type attn_mask: torch.Tensor, optional
        :param past_key_values: Past key values for caching
        :type past_key_values: Tuple[torch.Tensor], optional
        :param pos: Position ids
        :type pos: torch.Tensor, optional
        :param is_causal: Whether to apply causal mask
        :type is_causal: bool, optional
        :return: Tuple containing logits, all hidden states, presents, and loss
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor], Optional[torch.Tensor]]
        """
        if idx is not None:
            device = idx.device
            t = idx.size(1)
        else:
            device = tok_emb.device
            t = tok_emb.size(-2)

        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        presents = ()
        all_hidden_states = ()
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.n_layer)
        else:
            past_length = past_key_values[0][0].size(-2)

        if pos is not None:
            pos = pos.view(-1, t)
        else:
            pos = torch.arange(
                past_length, t + past_length, dtype=torch.long, device=device
            ).unsqueeze(
                0
            )  # shape (1, t)

        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        if tok_emb is None:
            tok_emb = self.transformer.wte(idx)

        # position embeddings of shape (1, t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        all_hidden_states = all_hidden_states + (x,)
        for block, layer_past in zip(self.transformer.h, past_key_values):
            # torch.cuda.set_device(x.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(x.device) for past_state in layer_past)
            x, pres = block(x, attn_mask, layer_past, is_causal)
            all_hidden_states = all_hidden_states + (x,)
            presents = presents + (pres,)
        x = self.transformer.ln_f(x)
        all_hidden_states = all_hidden_states + (x,)

        logits = self.lm_head(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1).type(torch.LongTensor).to(self.device),
                ignore_index=-1,
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last
            # position
            # note: using list [-1] to preserve the time dim
            # logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, all_hidden_states, presents, loss

    def crop_block_size(self, block_size: int) -> None:
        """
        Adjust the block size of the model.

        This method performs model surgery to decrease the block size if necessary.
        For example, we may load the GPT2 pretrained model checkpoint (block size 1024)
        but want to use a smaller block size for some smaller, simpler model.

        :param block_size: The new block size to set. Must be less than or equal to the current block size.
        :type block_size: int
        :raises AssertionError: If the new block size is greater than the current block size.
        """
        assert block_size <= self.block_size
        self.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "attention_bias"):
                block.attn.attention_bias = block.attn.attention_bias[
                    :, :, :block_size, :block_size
                ]

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        override_args: Optional[dict] = None,
        custom_sd: Optional[str] = None,
    ) -> "EvolvableGPT":
        """
        Load a pretrained GPT model with the option to override certain configuration parameters or use a custom state dictionary.

        :param model_type: The type of GPT model to load. Must be one of {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}.
        :type model_type: str
        :param override_args: A dictionary of arguments to override the default configuration. Defaults to None.
        :type override_args: Optional[dict]
        :param custom_sd: Path to a custom state dictionary to load. If None, the default pretrained weights are used. Defaults to None.
        :type custom_sd: Optional[str]
        :return: An instance of the EvolvableGPT model with the specified configuration and weights.
        :rtype: EvolvableGPT
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        from transformers import GPT2Config, GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        if custom_sd is not None:
            if "vocab_size" in override_args:
                print(f"overriding vocab_size to {override_args['vocab_size']}")
                config_args["vocab_size"] = override_args["vocab_size"]
            model = EvolvableGPT(**config_args)
            sd_hf = torch.load(custom_sd, weights_only=False)
            config = GPT2Config(**config_args)
            model_hf = GPT2LMHeadModel(config)
            sd_hf = {k.split("model.")[-1]: v for k, v in sd_hf.items()}
            model_hf.load_state_dict(sd_hf)
        else:
            # create a from-scratch initialized Evolvable GPT model
            model = EvolvableGPT(**config_args)
            # init a huggingface/transformers model
            model_hf = GPT2LMHeadModel.from_pretrained(model_type)
            sd_hf = model_hf.state_dict()

        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.attention_bias")]

        # copy while ensuring all of the parameters are aligned and match in names
        # and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for khf, k in zip(sd_keys_hf, sd_keys):
            if any(khf.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[khf].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[khf].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[khf].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[khf])
        return model

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str
    ) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model by separating parameters into those that will and won't experience weight decay.

        This function separates all parameters of the model into two buckets: those that will experience weight decay for
        regularization and those that won't (biases, and layernorm/embedding weights). It then returns the PyTorch optimizer object.

        :param weight_decay: The weight decay factor for regularization.
        :type weight_decay: float
        :param learning_rate: The learning rate for the optimizer.
        :type learning_rate: float
        :param betas: Coefficients used for computing running averages of gradient and its square.
        :type betas: tuple
        :param device_type: The type of device being used ('cuda' or 'cpu').
        :type device_type: str
        :return: Configured PyTorch optimizer.
        :rtype: torch.optim.Optimizer
        :raises AssertionError: If any parameter is in both decay and no_decay sets or if any parameter is not considered.
        """
        # separate out all parameters to those that will and won't experience
        # regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurrence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not
        # decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters {} made it into both decay/no_decay sets!".format(
            str(inter_params)
        )
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters {} were not separated into either decay/no_decay set!".format(
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        lr = torch.tensor(learning_rate, device=self.device)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        :param fwdbwd_per_iter: Number of forward-backward passes per iteration.
        :type fwdbwd_per_iter: int
        :param dt: Time taken per iteration in seconds.
        :type dt: float
        :return: Model flops utilization as a ratio of A100 bfloat16 peak FLOPS.
        :rtype: float
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = (
            self.n_layer,
            self.n_head,
            self.n_embd // self.n_head,
            self.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens.

        This method takes a conditioning sequence of indices `idx` (LongTensor of shape (b, t))
        and completes the sequence `max_new_tokens` times, feeding the predictions back into the
        model each time. Most likely you'll want to make sure to be in `model.eval()` mode of operation
        for this.

        :param idx: Conditioning sequence of indices.
        :type idx: torch.Tensor
        :param max_new_tokens: Number of new tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. Higher values mean more random samples, defaults to 1.0.
        :type temperature: float, optional
        :param top_k: If specified, only consider the top k tokens for sampling, defaults to None.
        :type top_k: Optional[int], optional
        :return: Generated sequence of indices.
        :rtype: torch.Tensor
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @mutation(MutationType.LAYER)
    def add_layer(self):
        """Adds a block layer to transformer."""
        if self.n_layer < self.max_layers:
            self.n_layer += 1
        # else:
        #     self.add_node()

    @mutation(MutationType.LAYER)
    def remove_layer(self):
        """Removes a block layer from transformer."""
        if self.n_layer > self.min_layers:
            self.n_layer -= 1
        # else:
        #     self.add_node()

    @mutation(MutationType.NODE)
    def add_node(self, numb_new_nodes=None):
        """Adds nodes to hidden layers of transformer.

        :param numb_new_nodes: Number of nodes to add to hidden layers, defaults to None
        :type numb_new_nodes: int, optional
        """
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([32, 64, 128], 1)[0]
        self.dim_feedfwd += numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_node(self, numb_new_nodes=None):
        """Removes nodes from hidden layers of transformer.

        :param numb_new_nodes: Number of nodes to remove from hidden layers, defaults to None
        :type numb_new_nodes: int, optional
        """
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([32, 64, 128], 1)[0]

        self.dim_feedfwd -= numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    def recreate_network(self) -> None:
        """Recreates neural network."""
        new_transformer = self.build_networks()

        self.transformer = EvolvableModule.preserve_parameters(
            old_net=self.transformer, new_net=new_transformer
        )


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False.

    :param int ndim: The number of dimensions in the input tensor.
    :param bool bias: If True, adds a learnable bias to the normalization.
    :param float layer_norm_eps: A value added to the denominator for numerical stability (default: 1e-5).

    :ivar torch.nn.Parameter weight: The learnable weights for normalization.
    :ivar torch.nn.Parameter bias: The learnable bias for normalization, if bias is True.
    :ivar float layer_norm_eps: The epsilon value for numerical stability.

    :method forward: Applies layer normalization to the input tensor.
    :param torch.Tensor input: The input tensor to normalize.
    :return: The normalized tensor.
    :rtype: torch.Tensor
    """

    def __init__(
        self,
        ndim: int,
        bias: bool,
        layer_norm_eps: float = 1e-5,
        device: DeviceType = "cpu",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim, device=device))
        self.bias = nn.Parameter(torch.zeros(ndim, device=device)) if bias else None
        self.layer_norm_eps = layer_norm_eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.weight.shape, self.weight, self.bias, self.layer_norm_eps
        )


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention module for transformer models.

    This module implements a causal self-attention mechanism, ensuring that each position in the sequence
    can only attend to previous positions.

    :param n_embd: The embedding dimensionality.
    :type n_embd: int
    :param n_head: The number of attention heads.
    :type n_head: int
    :param bias: Whether to use bias in the linear projections.
    :type bias: bool
    :param dropout: Dropout probability for attention and residual connections.
    :type dropout: float
    :param block_size: The maximum block size for the causal mask.
    :type block_size: int
    """

    attention_bias: torch.Tensor

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        device: DeviceType = "cpu",
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias, device=device)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias, device=device)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.device = device
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the
            # input sequence
            self.register_buffer(
                "attention_bias",
                torch.tril(torch.ones(block_size, block_size, device=device)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass through the CausalSelfAttention module.

        :param x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
        :type x: torch.Tensor
        :param attn_mask: Optional attention mask tensor.
        :type attn_mask: Optional[torch.Tensor]
        :param layer_past: Optional tuple of past key and value tensors for caching.
        :type layer_past: Optional[Tuple[torch.Tensor]]
        :param is_causal: Whether to apply causal mask.
        :type is_causal: bool
        :return: Tuple containing the output tensor and the present key and value tensors.
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor]]
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) ->
        # (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal,
            )
        else:
            # manual implementation of attention
            att: torch.Tensor = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(k.size(-1))
            )
            att = att.masked_fill(self.attention_bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, present


class Block(nn.Module):
    """
    Transformer block consisting of layer normalization, causal self-attention, and MLP.

    :param n_embd: The embedding dimensionality.
    :type n_embd: int
    :param n_head: The number of attention heads.
    :type n_head: int
    :param bias: Whether to use bias in the linear projections.
    :type bias: bool
    :param dropout: Dropout probability for attention and residual connections.
    :type dropout: float
    :param block_size: The maximum block size for the causal mask.
    :type block_size: int
    :param hidden_size: The size of the hidden layer in the MLP.
    :type hidden_size: int
    :param activation: The activation function to use in the MLP, defaults to "GELU".
    :type activation: str, optional
    :param layer_norm_eps: A value added to the denominator for numerical stability in layer normalization, defaults to 1e-5.
    :type layer_norm_eps: float, optional
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        hidden_size: int,
        activation: str = "GELU",
        layer_norm_eps: float = 1e-5,
        device: DeviceType = "cpu",
    ):
        super().__init__()
        self.device = device
        self.ln_1 = LayerNorm(
            n_embd, bias=bias, layer_norm_eps=layer_norm_eps, device=device
        )
        self.attn = CausalSelfAttention(
            n_embd, n_head, bias, dropout, block_size, device=device
        )
        self.ln_2 = LayerNorm(
            n_embd, bias=bias, layer_norm_eps=layer_norm_eps, device=device
        )
        self.mlp = MLP(n_embd, dropout, hidden_size, activation, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass through the transformer block.

        :param x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
        :type x: torch.Tensor
        :param attn_mask: Optional attention mask tensor.
        :type attn_mask: Optional[torch.Tensor]
        :param layer_past: Optional tuple of past key and value tensors for caching.
        :type layer_past: Optional[Tuple[torch.Tensor]]
        :param is_causal: Whether to apply causal mask.
        :type is_causal: bool
        :return: Tuple containing the output tensor and the present key and value tensors.
        :rtype: Tuple[torch.Tensor, Tuple[torch.Tensor]]
        """
        attn_output, present = self.attn(
            self.ln_1(x),
            attn_mask=attn_mask,
            layer_past=layer_past,
            is_causal=is_causal,
        )
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present


class MLP(EvolvableMLP):
    def __init__(self, n_embd, dropout, hidden_size, activation="GELU", **kwargs):
        super().__init__(
            num_inputs=n_embd,
            num_outputs=n_embd,
            hidden_size=[hidden_size],
            layer_norm=False,
            output_activation=activation,
            new_gelu=True,
            **kwargs,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        """
        # convert input to tensor if it is not already
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(np.array(x), device=self.device)

        # forward pass through the network
        for value in self.model:
            x = value(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """The positional embedding class.
    Converts tensor of input indices into corresponding tensor of position embeddings.
    """

    def __init__(self, max_positions: int, emb_size: int, device: DeviceType = "cpu"):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(max_positions, emb_size, device=device)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        """Forward pass through position embedding module.
        :param tokens: Tokens to embed
        :type tokens: torch.Tensor
        """
        return self.embedding(tokens)


class TokenEmbedding(nn.Module):
    """The token embedding class. Converts tensor of input indices into corresponding tensor of token embeddings."""

    def __init__(self, vocab_size: int, emb_size: int, device: DeviceType = "cpu"):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, emb_size, device=device)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        """Forward pass through token embedding module.
        :param tokens: Tokens to embed
        :type tokens: torch.Tensor
        """
        # return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        return self.embedding(tokens)
