import copy
import math
from collections import defaultdict
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn import functional as F
from tqdm import tqdm

from agilerl.data.rl_data import DataPoint
from agilerl.networks.evolvable_gpt import EvolvableGPT
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.utils.sampling_utils import (
    always_terminate,
    map_all_kvs,
    pad_sequence,
    process_logits,
    update_kvs,
)


class ILQL(nn.Module):
    """The Implicit Language Q Learning algorithm class. ILQL paper: https://arxiv.org/pdf/2206.11871.pdf

    :param dataset: Language dataset to perform ILQL on
    :type dataset: torch.utils.data.Dataset
    :param net_config: Network configuration, defaults to GPT2 configuration
    :type net_config: dict, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-5
    :type lr: float, optional
    :param alpha: For soft update of target network parameters, defaults to 0.005
    :type alpha: float, optional
    :param beta: For AWR policy extraction, defaults to 0.0
    :type beta: float, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param tau: For value network loss, defaults to 0.6
    :type tau: float, optional
    :param mutation: Most recent mutation to agent, defaults to None
    :type mutation: str, optional
    :param transition_weight: Value to use temporarily for weights in transition, defaults to 0.0
    :type transition_weight: float, optional
    :param clip_weight: Maximum value to clip weights at, defaults to None
    :type clip_weight: float, optional
    :param value_max: Maximum Q value for clipping, defaults to None
    :type value_max: float, optional
    :param value_min: Minimum Q value for clipping, defaults to None
    :type value_min: float, optional
    :param detach_v: Detach V network, defaults to False
    :type detach_v: bool, optional
    :param detach_q: Detach Q network, defaults to False
    :type detach_q: bool, optional
    :param detach_pi: Detach Policy network, defaults to False
    :type detach_pi: bool, optional
    :param double_q: Use double Q learning, defaults to True
    :type double_q: bool, optional
    :param per_token: Do per_token ILQL, defaults to True
    :type per_token: bool, optional
    :param exp_weights: Exponential advantage weights, defaults to True
    :type exp_weights: bool, optional
    :param dm_margin: Margin for DM loss, defaults to 0.0
    :type dm_margin: float, optional
    :param cql_temp: Temperature parameter for CQL loss, defaults to 1.0
    :type cql_temp: float, optional
    :param weight_decay: weight decay for optimizer, defaults to 0.0
    :type weight_decay: float, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
        self,
        dataset,
        net_config={
            "arch": "gpt",
            "vocab_size": 50257,
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12,
            "dim_feedfwd": 3072,
            "block_size": 1024,
            "activation": "GELU",
            "dropout": 0.1,
            "layer_norm_eps": 1e-5,
            "min_layers": 8,
            "max_layers": 16,
            "bias": True,
        },
        index=0,
        batch_size=64,
        lr=1e-5,
        alpha=0.005,
        beta=0.0,
        gamma=0.99,
        tau=0.6,
        mutation=None,
        transition_weight=0.0,
        clip_weight=None,
        value_max=None,
        value_min=None,
        detach_v=False,
        detach_q=False,
        detach_pi=False,
        double_q=True,
        per_token=True,
        exp_weights=True,
        dm_margin=0.0,
        cql_temp=1.0,
        weight_decay=0.0,
        device="cpu",
    ):
        super().__init__()

        self.algo = "ILQL"
        self.dataset = dataset
        self.net_config = net_config
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.transition_weight = transition_weight
        self.clip_weight = clip_weight
        self.value_max = value_max
        self.value_min = value_min
        self.detach_v = detach_v
        self.detach_pi = detach_pi
        self.detach_q = detach_q
        self.double_q = double_q
        self.tau = tau
        self.exp_weights = exp_weights
        self.dm_margin = dm_margin
        self.cql_temp = cql_temp
        self.per_token = per_token
        self.double_q = double_q
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.mut = mutation
        self.device = device
        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # model
        self.model = EvolvableGPT(
            n_layer=net_config["n_layer"],
            vocab_size=net_config["vocab_size"],
            n_embd=net_config["n_embd"],
            n_head=net_config["n_head"],
            dim_feedfwd=net_config["dim_feedfwd"],
            block_size=net_config["block_size"],
            dropout=net_config["dropout"],
            activation=net_config["activation"],
            layer_norm_eps=net_config["layer_norm_eps"],
            min_layers=net_config["min_layers"],
            max_layers=net_config["max_layers"],
            bias=net_config["bias"],
            device=self.device,
        ).to(self.device)
        # lm policy
        self.actor = EvolvableGPT(
            n_layer=net_config["n_layer"],
            vocab_size=net_config["vocab_size"],
            n_embd=net_config["n_embd"],
            n_head=net_config["n_head"],
            dim_feedfwd=net_config["dim_feedfwd"],
            block_size=net_config["block_size"],
            dropout=net_config["dropout"],
            activation=net_config["activation"],
            layer_norm_eps=net_config["layer_norm_eps"],
            min_layers=net_config["min_layers"],
            max_layers=net_config["max_layers"],
            bias=net_config["bias"],
            device=self.device,
        ).to(self.device)
        # lm target
        self.actor_target = EvolvableGPT(
            n_layer=net_config["n_layer"],
            vocab_size=net_config["vocab_size"],
            n_embd=net_config["n_embd"],
            n_head=net_config["n_head"],
            dim_feedfwd=net_config["dim_feedfwd"],
            block_size=net_config["block_size"],
            dropout=net_config["dropout"],
            activation=net_config["activation"],
            layer_norm_eps=net_config["layer_norm_eps"],
            min_layers=net_config["min_layers"],
            max_layers=net_config["max_layers"],
            bias=net_config["bias"],
            device=self.device,
        ).to(self.device)

        self.copy_model_to_actor_target()

        # v and q networks
        self.v = EvolvableMLP(
            num_inputs=net_config["n_embd"],
            num_outputs=1,
            hidden_size=[net_config["n_embd"] * 2, net_config["n_embd"] * 2],
            device=self.device,
        ).to(self.device)
        self.q = EvolvableMLP(
            num_inputs=net_config["n_embd"],
            num_outputs=self.dataset.tokenizer.num_tokens(),
            hidden_size=[net_config["n_embd"] * 2, net_config["n_embd"] * 2],
            device=self.device,
        ).to(self.device)
        self.target_q = EvolvableMLP(
            num_inputs=net_config["n_embd"],
            num_outputs=self.dataset.tokenizer.num_tokens(),
            hidden_size=[net_config["n_embd"] * 2, net_config["n_embd"] * 2],
            device=self.device,
        ).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        if self.double_q:
            self.q2 = EvolvableMLP(
                num_inputs=net_config["n_embd"],
                num_outputs=self.dataset.tokenizer.num_tokens(),
                hidden_size=[net_config["n_embd"] * 2, net_config["n_embd"] * 2],
                device=self.device,
            ).to(self.device)
            self.target_q2 = EvolvableMLP(
                num_inputs=net_config["n_embd"],
                num_outputs=self.dataset.tokenizer.num_tokens(),
                hidden_size=[net_config["n_embd"] * 2, net_config["n_embd"] * 2],
                device=self.device,
            ).to(self.device)
            self.target_q2.load_state_dict(self.q2.state_dict())

        self.pi = EvolvableMLP(
            num_inputs=net_config["n_embd"],
            num_outputs=self.dataset.tokenizer.num_tokens(),
            hidden_size=[net_config["n_embd"] * 2, net_config["n_embd"] * 2],
            device=self.device,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def copy_model_to_actor_target(self):
        self.actor.load_state_dict(self.model.state_dict())
        self.actor_target.load_state_dict(self.model.state_dict())

    def forward(
        self,
        tokens: torch.Tensor,
        state_idxs: torch.Tensor,
        action_idxs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prefix_embs: Optional[torch.Tensor] = None,
        prefix_attn_mask: Optional[torch.Tensor] = None,
        remove_prefix_position_embs: bool = False,
        qv_kwargs=None,
        policy_kwargs=None,
        target_kwargs=None,
        skip_policy_on_train: bool = False,
        detach_full_policy: bool = False,
    ):
        """Forward pass through transformers.

        :param tokens: Tokens to input to model
        :type tokens: torch.Tensor
        :param state_idxs: State indexes
        :type state_idxs: torch.Tensor
        :param action_idxs: Action indexes
        :type action_idxs: torch.Tensor
        :param attn_mask: Attention mask for transformers, defaults to None
        :type attn_mask: torch.Tensor, optional
        :param prefix_embs: Prefix embeddings, defaults to None
        :type prefix_embs: torch.Tensor, optional
        :param skip_policy_on_train: Skip policy language model when training, defaults to False
        :type skip_policy_on_train: bool, optional
        :param detach_full_policy: Use policy language model without gradients, defaults to False
        :type detach_full_policy: bool, optional
        """
        if qv_kwargs is None:
            qv_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
        if policy_kwargs is None:
            policy_kwargs = {}
        if prefix_embs is None:
            prefix_embs = torch.empty(
                (tokens.shape[0], 0, self.net_config["n_embd"])
            ).to(self.device)
            prefix_t = prefix_embs.shape[1]
        else:
            prefix_t = 0
        set_pos_ids = prefix_attn_mask is not None
        if prefix_attn_mask is not None and attn_mask is not None:
            input_attn_mask = torch.cat((prefix_attn_mask, attn_mask), dim=1)
        else:
            input_attn_mask = None
        position_ids = torch.cumsum(input_attn_mask, dim=1) - 1 if set_pos_ids else None

        target_prefix_embs = prefix_embs.clone()
        policy_prefix_embs = prefix_embs.clone()

        if remove_prefix_position_embs:
            prefix_embs -= self.model.transformer.wpe(
                position_ids[:, : prefix_embs.shape[1]]
            )
            target_prefix_embs -= self.actor_target.transformer.wpe(
                position_ids[:, : prefix_embs.shape[1]]
            )

        input_embeddings = torch.cat(
            (prefix_embs, self.model.transformer.wte(tokens)), dim=1
        )
        target_input_embeddings = torch.cat(
            (target_prefix_embs, self.actor_target.transformer.wte(tokens)), dim=1
        )

        # Model forward passes
        (
            model_outputs,
            model_hidden_states,
            model_past_key_values,
            model_loss,
        ) = self.model(
            tok_emb=input_embeddings,
            attn_mask=input_attn_mask,
            pos=position_ids,
            **qv_kwargs
        )
        hidden_states = model_hidden_states[-1][:, prefix_t:, :]

        with torch.no_grad():
            (
                target_outputs,
                target_hidden_states,
                target_past_key_values,
                target_loss,
            ) = self.actor_target(
                tok_emb=target_input_embeddings,
                attn_mask=input_attn_mask,
                pos=position_ids,
                **target_kwargs
            )
        target_hidden_states = target_hidden_states[-1][:, prefix_t:, :]

        # Prepare policy inputs
        if skip_policy_on_train and self.training:
            policy_outputs = model_outputs
            policy_hidden_states = hidden_states
            policy_past_key_values = model_past_key_values
        else:
            if remove_prefix_position_embs:
                policy_prefix_embs -= self.actor.transformer.wpe(
                    position_ids[:, : prefix_embs.shape[1]]
                )
            policy_input_embeddings = torch.cat(
                (policy_prefix_embs, self.actor.transformer.wte(tokens)), dim=1
            )
            if detach_full_policy:
                with torch.no_grad():
                    (
                        policy_outputs,
                        policy_hidden_states,
                        policy_past_key_values,
                        policy_loss,
                    ) = self.actor(
                        tok_emb=policy_input_embeddings,
                        attn_mask=input_attn_mask,
                        pos=position_ids,
                        **policy_kwargs
                    )
            else:
                (
                    policy_outputs,
                    policy_hidden_states,
                    policy_past_key_values,
                    policy_loss,
                ) = self.actor(
                    tok_emb=policy_input_embeddings,
                    attn_mask=input_attn_mask,
                    pos=position_ids,
                    **policy_kwargs
                )
            policy_hidden_states = policy_hidden_states[-1][:, prefix_t:, :]

        all_model_outputs = {
            "qv_model_outputs": {"past_key_values": model_past_key_values},
            "policy_model_outputs": {"past_key_values": policy_past_key_values},
            "target_model_outputs": {"past_key_values": target_past_key_values},
        }
        all_hidden_states = {
            "qv_hidden_states": model_hidden_states,
            "policy_hidden_states": target_hidden_states,
            "target_hidden_states": policy_hidden_states,
        }

        state_hidden_states = torch.gather(
            input=hidden_states,
            dim=1,
            index=state_idxs.unsqueeze(2).repeat(1, 1, self.net_config["n_embd"]),
        )
        action_hidden_states = torch.gather(
            input=hidden_states,
            dim=1,
            index=action_idxs.unsqueeze(2).repeat(1, 1, self.net_config["n_embd"]),
        )
        action_target_hidden_states = torch.gather(
            input=target_hidden_states,
            dim=1,
            index=action_idxs.unsqueeze(2).repeat(1, 1, self.net_config["n_embd"]),
        )
        vs = self.v(
            state_hidden_states.detach() if self.detach_v else state_hidden_states
        ).squeeze(2)
        qs = self.q(
            action_hidden_states.detach() if self.detach_q else action_hidden_states
        )
        if self.double_q:
            qs2 = self.q2(
                action_hidden_states.detach() if self.detach_q else action_hidden_states
            )
        with torch.no_grad():
            target_qs = self.target_q(action_target_hidden_states)
            if self.double_q:
                target_qs2 = self.target_q2(action_target_hidden_states)
        if skip_policy_on_train and self.training and self.actor is not None:
            logits = torch.zeros(
                (
                    policy_hidden_states.shape[0],
                    policy_hidden_states.shape[1],
                    self.dataset.tokenizer.num_tokens(),
                )
            ).to(self.device)
        else:
            if detach_full_policy:
                with torch.no_grad():
                    logits = self.pi(
                        policy_hidden_states.detach()
                        if self.detach_pi
                        else policy_hidden_states
                    )
            else:
                logits = self.pi(
                    policy_hidden_states.detach()
                    if self.detach_pi
                    else policy_hidden_states
                )
        return {
            "model_outputs": all_model_outputs,
            "hidden_states": all_hidden_states,
            "vs": vs,
            "target_vs": vs,
            "qs": (
                (
                    qs,
                    qs2,
                )
                if self.double_q
                else qs
            ),
            "target_qs": self.clip_values(
                torch.minimum(target_qs, target_qs2) if self.double_q else target_qs
            ),
            "logits": logits,
        }

    def clip_values(self, values):
        if self.value_min is not None or self.value_max is not None:
            return torch.clip(values, self.value_min, self.value_max)
        return values

    def get_downstream_rs(self, rs, gamma):
        gamma_row = torch.cumprod(torch.full(rs.shape, gamma).to(self.device), dim=1)
        gamma_tensor = torch.triu(gamma_row.unsqueeze(1) / gamma_row.unsqueeze(2))
        return (gamma_tensor * rs.unsqueeze(1)).sum(dim=2)

    def get_weights(
        self,
        tokens: torch.Tensor,
        vs: torch.Tensor,
        qs: Optional[torch.Tensor],
        state_idxs: torch.Tensor,
        action_idxs: torch.Tensor,
        terminals: torch.Tensor,
    ):
        weights = torch.full(tokens.shape, self.transition_weight).to(self.device)
        if self.exp_weights:
            w_values = torch.exp(self.beta * (qs - vs))
        else:
            # w_values = ((qs - vs) > 0.0).float()
            adv_sign = ((qs - vs) > 0.0).float()
            w_values = self.beta * adv_sign + (1 - self.beta) * (1 - adv_sign)
        if action_idxs.shape[1] == 0:
            n = torch.zeros((tokens.shape[0],)).long().to(self.device)
        else:
            n = torch.argmax(action_idxs, dim=1) + 1
        for i in range(tokens.shape[0]):
            weights[i] = torch.scatter(
                weights[i], dim=0, index=action_idxs[i, : n[i]], src=w_values[i, : n[i]]
            )
        if self.clip_weight is not None:
            weights = torch.clip(weights, max=self.clip_weight)
        return weights

    def awac_loss(self, tokens, attn_mask, logits, w):
        w = w.detach()
        losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
            reduction="none",
        )
        losses = losses.reshape(tokens.shape[0], tokens.shape[1] - 1)
        return (losses * w[:, :-1] * attn_mask[:, 1:]).sum() / attn_mask[:, 1:].sum()

    def get_v_loss(self, vs, target_qs, terminals):
        target_qs = target_qs.detach()
        return (
            (
                (target_qs >= vs).int() * self.tau * (target_qs - vs) ** 2
                + (target_qs < vs).int() * (1 - self.tau) * (target_qs - vs) ** 2
            )
            * (1 - terminals[:, :-1])
        ).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)

    def get_q_loss(self, vns, qs, rs, gamma, terminals):
        vns = vns.detach()
        if self.double_q:
            q1, q2 = qs
            l1 = (
                (((1 - terminals[:, 1:]) * vns * gamma + rs - q1) ** 2)
                * (1 - terminals[:, :-1])
            ).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            l2 = (
                (((1 - terminals[:, 1:]) * vns * gamma + rs - q2) ** 2)
                * (1 - terminals[:, :-1])
            ).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            return l1 + l2
        return (
            (((1 - terminals[:, 1:]) * vns * gamma + rs - qs) ** 2)
            * (1 - terminals[:, :-1])
        ).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)

    def get_cql_loss(self, qs, action_tokens, terminals):
        n = (1 - terminals[:, :-1]).sum()
        if self.double_q:
            q1, q2 = qs
            b, t, d = q1.shape
            t1 = F.cross_entropy(
                q1.reshape(-1, d) / self.cql_temp,
                action_tokens.reshape(-1),
                reduction="none",
            ).reshape(b, t) * (1 - terminals[:, :-1])

            t2 = F.cross_entropy(
                q2.reshape(-1, d) / self.cql_temp,
                action_tokens.reshape(-1),
                reduction="none",
            ).reshape(b, t) * (1 - terminals[:, :-1])
            return ((t1) + (t2)).sum() / max(n.item(), 1.0)
        b, t, d = qs.shape
        return (
            F.cross_entropy(
                qs.reshape(-1, d) / self.cql_temp,
                action_tokens.reshape(-1),
                reduction="none",
            ).reshape(b, t)
            * (1 - terminals[:, :-1])
        ).sum() / max(n.item(), 1.0)

    def get_dm_loss(self, qs, data_qs, terminals, margin):
        n = (1 - terminals[:, :-1]).sum()
        if self.double_q:
            q1, q2 = qs
            data_q1, data_q2 = data_qs
            return (
                (
                    (
                        torch.max(
                            q1 - data_q1.unsqueeze(-1) + margin,
                            torch.tensor(0.0).to(self.device),
                        )
                        ** 2
                    ).sum(dim=-1)
                    * (1 - terminals[:, :-1])
                )
                + (
                    (
                        torch.max(
                            q2 - data_q2.unsqueeze(-1) + margin,
                            torch.tensor(0.0).to(self.device),
                        )
                        ** 2
                    ).sum(dim=-1)
                    * (1 - terminals[:, :-1])
                )
            ).sum() / max(n.item(), 1.0)
        return (
            (
                torch.max(
                    qs - data_qs.unsqueeze(-1) + margin,
                    torch.tensor(0.0).to(self.device),
                )
                ** 2
            ).sum(dim=-1)
            * (1 - terminals[:, :-1])
        ).sum() / max(n.item(), 1.0)

    def prepare_inputs(self, items):
        if isinstance(items, dict):
            return items
        return to(self.dataset.collate(items, self.device), self.device)

    def get_qvs(
        self,
        items,
        prefix_embs: Optional[torch.Tensor] = None,
        prefix_attn_mask: Optional[torch.Tensor] = None,
        remove_prefix_position_embs: bool = False,
        qv_kwargs=None,
        policy_kwargs=None,
        target_kwargs=None,
        **kwargs
    ):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs["tokens"], prepared_inputs["attn_mask"]
        s_idx, a_idx = prepared_inputs["state_idxs"], prepared_inputs["action_idxs"]
        rs, terminals = prepared_inputs["rewards"], prepared_inputs["terminals"]
        self_outputs = self(tokens, s_idx, a_idx, attn_mask, prefix_embs, **kwargs)
        model_outputs, vs, qs = (
            self_outputs["model_outputs"],
            self_outputs["vs"],
            self_outputs["qs"],
        )
        target_qs, logits = self_outputs["target_qs"], self_outputs["logits"]
        vt = vs[:, :-1]
        vtp1 = vs[:, 1:]
        select_tokens = torch.gather(tokens[:, 1:], dim=1, index=a_idx)
        cql_term = self.get_cql_loss(qs, select_tokens, terminals)
        full_qs = qs
        if self.double_q:
            q1, q2 = qs
            q1 = torch.gather(q1, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
            q2 = torch.gather(q2, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
            # tok_seq = [self.dataset.tokenizer.id_to_token(
            #   token) for token in select_tokens[0].detach().cpu().tolist()][:(1-terminals[0, :-1]).sum()]
            # max_q_seq = torch.max(q1, q2)[0, :(1-terminals[0, :-1]).sum()].detach().cpu().tolist()
            # print(self.dataset.tokenizer.decode(tokens[0, :][:attn_mask[0, :].sum().long()].tolist(),
            #   clean_up_tokenization_spaces=False))
            # print(list(zip(tok_seq, max_q_seq)))
            # print(rs)
            qs = (
                q1,
                q2,
            )
        else:
            qs = torch.gather(qs, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
        dm_term = self.get_dm_loss(full_qs, qs, terminals, self.dm_margin)
        target_qs = torch.gather(
            target_qs, dim=2, index=select_tokens.unsqueeze(2)
        ).squeeze(2)
        with torch.no_grad():
            weights = self.get_weights(tokens, vt, target_qs, s_idx, a_idx, terminals)
        return {
            "tokens": tokens,
            "attn_mask": attn_mask,
            "model_outputs": model_outputs,
            "vs": vt,
            "qs": qs,
            "vns": vtp1,
            "target_vs": vt,
            "target_qs": target_qs,
            "target_vns": vtp1,
            "rs": rs,
            "terminals": terminals,
            "logits": logits,
            "weights": weights,
            "cql_term": cql_term,
            "dm_term": dm_term,
        }

    def get_loss(
        self,
        items,
        awac_weight=0.0,
        v_loss_weight=0.0,
        q_loss_weight=0.0,
        cql_loss_weight=0.0,
        dm_loss_weight=0.0,
        mc_returns=False,
    ):
        prepared_inputs = self.prepare_inputs(items)
        a_idx = prepared_inputs["action_idxs"]
        get_qvs_outputs = self.get_qvs(
            items,
            qv_kwargs={"output_attentions": True},
            policy_kwargs={"output_attentions": True},
            target_kwargs={"output_attentions": True},
            skip_policy_on_train=(awac_weight == 0.0),
        )
        tokens, attn_mask, _ = (
            get_qvs_outputs["tokens"],
            get_qvs_outputs["attn_mask"],
            get_qvs_outputs["model_outputs"],
        )
        vs, qs = get_qvs_outputs["vs"], get_qvs_outputs["qs"]
        vns, target_qs, rs = (
            get_qvs_outputs["vns"],
            get_qvs_outputs["target_qs"],
            get_qvs_outputs["rs"],
        )
        terminals, logits, weights = (
            get_qvs_outputs["terminals"],
            get_qvs_outputs["logits"],
            get_qvs_outputs["weights"],
        )

        logs = {}
        transformer_logs = {}
        # transformer_logs['qv_transformer_logs'] = get_transformer_logs(
        #     model_outputs['qv_model_outputs'].attentions, self.model, attn_mask)
        # if self.actor is not None and (not (self.training and awac_weight == 0.0)):
        #     transformer_logs['policy_transformer_logs'] = get_transformer_logs(
        #         model_outputs['policy_model_outputs'].attentions, self.actor, attn_mask)
        # if self.actor_target is not None:
        #     transformer_logs['target_transformer_logs'] = get_transformer_logs(
        #         model_outputs['target_model_outputs'].attentions, self.actor_target, attn_mask)
        n = (1 - terminals[:, :-1]).sum().item()
        rs_downstream = self.get_downstream_rs(rs, self.gamma)
        if mc_returns:
            v_loss = self.get_v_loss(vs, rs_downstream, terminals)
        else:
            v_loss = self.get_v_loss(vs, target_qs, terminals)
        q_loss = self.get_q_loss(vns, qs, rs, self.gamma, terminals)
        cql_loss = get_qvs_outputs["cql_term"]
        dm_loss = get_qvs_outputs["dm_term"]
        token_loss = self.awac_loss(tokens, attn_mask, logits, weights)
        logs["token_loss"] = (token_loss.item(), n)
        loss = (
            awac_weight * token_loss
            + v_loss_weight * v_loss
            + q_loss_weight * q_loss
            + cql_loss_weight * cql_loss
            + dm_loss_weight * dm_loss
        )
        logs["v_loss"] = (v_loss.item(), n)
        logs["q_loss"] = (q_loss.item(), n)
        logs["cql_loss"] = (cql_loss.item(), n)
        logs["dm_loss"] = (dm_loss.item(), n)
        advantages = sum(
            [
                ((target_qs[i] - vs[i])[: (1 - terminals[i, :-1]).sum().long().item()])
                .detach()
                .cpu()
                .tolist()
                for i in range(tokens.shape[0])
            ],
            [],
        )
        if self.double_q:
            q1, q2 = qs
            logs["q1_avg"] = (
                (q1 * (1 - terminals[:, :-1])).sum().item() / max(n, 1),
                n,
            )
            logs["q1_var"] = (
                (
                    (((q1 - logs["q1_avg"][0]) ** 2) * (1 - terminals[:, :-1])).sum()
                    / max(n, 1)
                ).item(),
                1,
            )
            logs["q2_avg"] = (
                (q2 * (1 - terminals[:, :-1])).sum().item() / max(n, 1),
                n,
            )
            logs["q2_var"] = (
                (
                    (((q2 - logs["q2_avg"][0]) ** 2) * (1 - terminals[:, :-1])).sum()
                    / max(n, 1)
                ).item(),
                1,
            )
        else:
            logs["q_avg"] = ((qs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
            logs["q_var"] = (
                (
                    (((qs - logs["q_avg"][0]) ** 2) * (1 - terminals[:, :-1])).sum()
                    / max(n, 1)
                ).item(),
                1,
            )
        logs["v_avg"] = ((vs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
        logs["v_var"] = (
            (
                (((vs - logs["v_avg"][0]) ** 2) * (1 - terminals[:, :-1])).sum()
                / max(n, 1)
            ).item(),
            1,
        )
        act_weights = torch.gather(weights, dim=1, index=a_idx)
        logs["act_weight_avg"] = (
            ((act_weights * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(),
            n,
        )
        logs["transformer"] = transformer_logs

        def postproc_f(x):
            return x.update(
                {
                    "loss": awac_weight * x["token_loss"]
                    + q_loss_weight * x["q_loss"]
                    + v_loss_weight * x["v_loss"]
                    + cql_loss_weight * x["cql_loss"]
                    + dm_loss_weight * x["dm_loss"]
                }
            )

        def hist_f(x):
            return x.update({"advantage_hist": wandb.Histogram(advantages)})

        return loss, logs, [postproc_f, hist_f]

    def score(
        self,
        tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        state_idxs: Optional[torch.Tensor],
        action_idxs: Optional[torch.Tensor],
        prefix_embs: Optional[torch.Tensor] = None,
        prefix_attn_mask: Optional[torch.Tensor] = None,
        remove_prefix_position_embs: bool = False,
        qv_kwargs=None,
        policy_kwargs=None,
        target_kwargs=None,
        beta: float = 1.0,
        exp_weights: bool = False,
        clip_weight: Optional[float] = None,
        logit_temp: float = 1.0,
        logit_top_k: Optional[int] = None,
        logit_top_p: Optional[float] = None,
        include_logits: bool = False,
        include_advantage: bool = True,
        action_mask: Optional[torch.Tensor] = None,
    ):
        trivial_value_query = False
        if state_idxs is None or action_idxs is None:
            state_idxs = (
                torch.full(
                    (
                        tokens.shape[0],
                        1,
                    ),
                    tokens.shape[1] - 1,
                )
                .long()
                .to(self.device)
            )
            action_idxs = (
                torch.full(
                    (
                        tokens.shape[0],
                        1,
                    ),
                    tokens.shape[1] - 1,
                )
                .long()
                .to(self.device)
            )
            trivial_value_query = True
        self_outputs = self(
            tokens,
            state_idxs,
            action_idxs,
            attn_mask,
            prefix_embs,
            prefix_attn_mask,
            remove_prefix_position_embs,
            qv_kwargs,
            policy_kwargs,
            target_kwargs,
        )
        model_outputs = self_outputs["model_outputs"]
        weights = torch.zeros(self_outputs["logits"].shape).to(self.device)
        if include_advantage:
            if action_mask is None:
                action_mask = torch.ones((tokens.shape[0],)).to(self.device)
            vs, qs = self_outputs["target_vs"], self_outputs["target_qs"]
            if not trivial_value_query:
                vs = vs[:, :-1]
            if exp_weights:
                w_values = beta * (qs - vs.unsqueeze(2))
            else:
                adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
                w_values = beta * adv_sign + (1 - beta) * (1 - adv_sign)
                w_values = torch.log(w_values)
            if clip_weight is not None:
                w_values = torch.clip(w_values, max=clip_weight)
            n = torch.argmax(action_idxs, dim=1) + 1
            for i in range(tokens.shape[0]):
                weights[i] += (
                    torch.scatter(
                        weights[i],
                        dim=0,
                        index=action_idxs[i, : n[i]]
                        .unsqueeze(1)
                        .repeat(1, weights.shape[2]),
                        src=w_values[i, : n[i], :],
                    )
                    * action_mask[i]
                )
        if include_logits:
            logits = process_logits(
                self_outputs["logits"],
                temp=logit_temp,
                top_k=logit_top_k,
                top_p=logit_top_p,
            )
            weights += torch.log(F.softmax(logits, dim=-1))
        return weights, model_outputs

    def get_scores(
        self,
        items,
        beta: float = 1.0,
        exp_weights: bool = False,
        clip_weight: Optional[float] = None,
        logit_temp: float = 1.0,
        logit_top_k: Optional[int] = None,
        logit_top_p: Optional[float] = None,
        include_logits: bool = False,
        include_advantage: bool = True,
    ) -> torch.Tensor:
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs["tokens"], prepared_inputs["attn_mask"]
        s_idx, a_idx = prepared_inputs["state_idxs"], prepared_inputs["action_idxs"]
        return self.score(
            tokens,
            attn_mask,
            s_idx,
            a_idx,
            beta=beta,
            exp_weights=exp_weights,
            clip_weight=clip_weight,
            logit_temp=logit_temp,
            logit_top_k=logit_top_k,
            logit_top_p=logit_top_p,
            include_logits=include_logits,
            include_advantage=include_advantage,
            action_mask=None,
        )[0]

    def initial_score(
        self,
        items,
        beta: float = 1.0,
        exp_weights: bool = False,
        clip_weight: Optional[float] = None,
        logit_temp: float = 1.0,
        logit_top_k: Optional[int] = None,
        logit_top_p: Optional[float] = None,
        include_logits: bool = False,
        include_advantage: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        prepared_inputs = self.prepare_inputs(items)
        tokens = prepared_inputs["tokens"]
        is_state = (
            (tokens == self.dataset.tokenizer.bos_token_id).float()
            + (tokens == self.dataset.tokenizer.eoa_token_id).float()
        ) > 0
        is_action = (
            (tokens == self.dataset.tokenizer.boa_token_id).float()
            + (tokens == self.dataset.tokenizer.eos_token_id).float()
        ) > 0
        state_points = torch.where(
            is_state,
            torch.arange(tokens.shape[1])
            .unsqueeze(0)
            .repeat(tokens.shape[0], 1)
            .to(self.device),
            -1,
        )
        action_points = torch.where(
            is_action,
            torch.arange(tokens.shape[1])
            .unsqueeze(0)
            .repeat(tokens.shape[0], 1)
            .to(self.device),
            -1,
        )
        action_mask = (
            action_points.argmax(dim=1) >= state_points.argmax(dim=1)
        ).float()
        scores, model_outputs = self.score(
            tokens,
            None,
            None,
            None,
            beta=beta,
            exp_weights=exp_weights,
            clip_weight=clip_weight,
            logit_temp=logit_temp,
            logit_top_k=logit_top_k,
            logit_top_p=logit_top_p,
            include_logits=include_logits,
            include_advantage=include_advantage,
            action_mask=action_mask,
        )
        return scores[:, -1, :], (
            model_outputs["qv_model_outputs"]["past_key_values"],
            model_outputs["policy_model_outputs"]["past_key_values"],
            model_outputs["target_model_outputs"]["past_key_values"],
            action_mask,
        )

    def soft_update(self):
        """Soft updates target networks."""
        for target_param, local_param in zip(
            self.target_q.parameters(), self.q.parameters()
        ):
            target_param.data.copy_(
                self.alpha * local_param.data + (1.0 - self.alpha) * target_param.data
            )
        if self.double_q:
            for target_param, local_param in zip(
                self.target_q2.parameters(), self.q2.parameters()
            ):
                target_param.data.copy_(
                    self.alpha * local_param.data
                    + (1.0 - self.alpha) * target_param.data
                )
        if self.actor_target is not None:
            for target_param, local_param in zip(
                self.actor_target.parameters(), self.model.parameters()
            ):
                target_param.data.copy_(
                    self.alpha * local_param.data
                    + (1.0 - self.alpha) * target_param.data
                )

    def hardUpdate(self):
        """Hard updates target networks."""
        for target_param, local_param in zip(
            self.target_q.parameters(), self.q.parameters()
        ):
            target_param.data.copy_(local_param.data)
        if self.double_q:
            for target_param, local_param in zip(
                self.target_q2.parameters(), self.q2.parameters()
            ):
                target_param.data.copy_(local_param.data)
        if self.actor_target is not None:
            del self.actor_target
            self.actor_target = None
            self.actor_target = copy.deepcopy(self.model)

    def clone(self, index=None):
        """Returns cloned agent identical to self.

        :param index: Index to keep track of agent for tournament selection and mutation, defaults to None
        :type index: int, optional
        """
        if index is None:
            index = self.index

        clone = type(self)(
            dataset=self.dataset,
            net_config=self.net_config,
            index=index,
            batch_size=self.batch_size,
            lr=self.lr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            tau=self.tau,
            mutation=self.mut,
            transition_weight=self.transition_weight,
            clip_weight=self.clip_weight,
            value_max=self.value_max,
            value_min=self.value_min,
            detach_v=self.detach_v,
            detach_q=self.detach_q,
            detach_pi=self.detach_pi,
            double_q=self.double_q,
            per_token=self.per_token,
            exp_weights=self.exp_weights,
            dm_margin=self.dm_margin,
            cql_temp=self.cql_temp,
            weight_decay=self.weight_decay,
            device=self.device,
        )

        clone.v = self.v.clone().to(self.device)
        clone.pi = self.pi.clone().to(self.device)
        clone.q = self.q.clone().to(self.device)
        clone.target_q = self.target_q.clone().to(self.device)
        if self.double_q:
            clone.q2 = self.q2.clone().to(self.device)
            clone.target_q2 = self.target_q2.clone().to(self.device)
        clone.model = self.model.clone().to(self.device)
        clone.actor = self.actor.clone().to(self.device)
        clone.actor_target = self.actor_target.clone().to(self.device)
        clone.optimizer = optim.Adam(
            clone.actor.parameters(), lr=clone.lr, weight_decay=clone.weight_decay
        )
        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def save_checkpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        torch.save(
            {
                "model_init_dict": self.model.init_dict,
                "model_state_dict": self.model.state_dict(),
                "actor_init_dict": self.actor.init_dict,
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_init_dict": self.actor_target.init_dict,
                "actor_target_state_dict": self.actor_target.state_dict(),
                "v_init_dict": self.v.init_dict,
                "v_state_dict": self.v.state_dict(),
                "pi_init_dict": self.pi.init_dict,
                "pi_state_dict": self.pi.state_dict(),
                "q_init_dict": self.q.init_dict,
                "q_state_dict": self.q.state_dict(),
                "target_q_init_dict": self.target_q.init_dict,
                "target_q_state_dict": self.target_q.state_dict(),
                "q2_init_dict": self.q2.init_dict if self.double_q else None,
                "q2_state_dict": self.q.state_dict() if self.double_q else None,
                "target_q2_init_dict": (
                    self.target_q2.init_dict if self.double_q else None
                ),
                "target_q2_state_dict": (
                    self.target_q2.state_dict() if self.double_q else None
                ),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "dataset": self.dataset,
                "net_config": self.net_config,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "tau": self.tau,
                "mutation": self.mut,
                "transition_weight": self.transition_weight,
                "clip_weight": self.clip_weight,
                "value_max": self.value_max,
                "value_min": self.value_min,
                "detach_v": self.detach_v,
                "detach_q": self.detach_q,
                "detach_pi": self.detach_pi,
                "double_q": self.double_q,
                "per_token": self.per_token,
                "exp_weights": self.exp_weights,
                "dm_margin": self.dm_margin,
                "cql_temp": self.cql_temp,
                "weight_decay": self.weight_decay,
                "index": self.index,
                "scores": self.scores,
                "fitness": self.fitness,
                "steps": self.steps,
            },
            path,
        )

    def load_checkpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.double_q = checkpoint["double_q"]
        self.net_config = checkpoint["net_config"]
        self.model = EvolvableGPT(**checkpoint["model_init_dict"])
        self.actor = EvolvableGPT(**checkpoint["actor_init_dict"])
        self.actor_target = EvolvableGPT(**checkpoint["actor_target_init_dict"])
        self.v = EvolvableMLP(**checkpoint["v_init_dict"])
        self.pi = EvolvableMLP(**checkpoint["pi_init_dict"])
        self.q = EvolvableMLP(**checkpoint["q_init_dict"])
        self.target_q = EvolvableMLP(**checkpoint["target_q_init_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.v.load_state_dict(checkpoint["v_state_dict"])
        self.pi.load_state_dict(checkpoint["pi_state_dict"])
        self.q.load_state_dict(checkpoint["q_state_dict"])
        self.target_q.load_state_dict(checkpoint["target_q_state_dict"])
        if self.double_q:
            self.q2 = EvolvableMLP(**checkpoint["q2_init_dict"])
            self.target_q2 = EvolvableMLP(**checkpoint["target_q2_init_dict"])
            self.q2.load_state_dict(checkpoint["q2_state_dict"])
            self.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.batch_size = checkpoint["batch_size"]
        self.lr = checkpoint["lr"]
        self.alpha = checkpoint["alpha"]
        self.beta = checkpoint["beta"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.mut = checkpoint["mutation"]
        self.transition_weight = checkpoint["transition_weight"]
        self.clip_weight = checkpoint["clip_weight"]
        self.value_max = checkpoint["value_max"]
        self.value_min = checkpoint["value_min"]
        self.detach_v = checkpoint["detach_v"]
        self.detach_q = checkpoint["detach_q"]
        self.detach_pi = checkpoint["detach_pi"]
        self.per_token = checkpoint["per_token"]
        self.exp_weights = checkpoint["exp_weights"]
        self.dm_margin = checkpoint["dm_margin"]
        self.cql_temp = checkpoint["cql_temp"]
        self.weight_decay = checkpoint["weight_decay"]
        self.index = checkpoint["index"]
        self.scores = checkpoint["scores"]
        self.fitness = checkpoint["fitness"]
        self.steps = checkpoint["steps"]


class ILQL_Policy:
    def __init__(self, iql_model: ILQL, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.iql_model = iql_model
        assert kind in {"beam", "sample"}
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.kls_all = []
        self.logprobs_all = []

    def sample_raw(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        state_idxs: torch.Tensor,
        action_idxs: torch.Tensor,
        termination_condition: Callable[[np.ndarray], bool],
        num_generations=1,
        max_generation_len=None,
        temp=1.0,
        top_k=None,
        top_p=None,
        exp_adv=False,
        adv_weight=0.0,
        adv_clip=None,
        include_logits=True,
        include_adv=True,
        rerank_log_prob_weight: float = 0.0,
        rerank_advantage_weight: float = 0.0,
        prefix_embs: Optional[torch.Tensor] = None,
        prefix_attn_mask: Optional[torch.Tensor] = None,
        remove_prefix_position_embs: bool = False,
    ):
        assert include_logits or include_adv

        tokenizer = self.iql_model.dataset.tokenizer
        max_length = self.iql_model.dataset.max_len
        if max_length is None:
            max_length = self.iql_model.model.block_size
        max_length = min(max_length, self.iql_model.model.block_size)
        device = self.iql_model.device
        bsize = tokens.shape[0]
        n = bsize * num_generations
        if max_generation_len is None:
            max_generation_len = max_length + 1
        input_strs = [
            tokenizer.decode(
                tokens[i, :][: attn_mask[i, :].sum().long()].tolist(),
                clean_up_tokenization_spaces=False,
            )
            for i in range(len(tokens))
        ]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        model_outputs = self.iql_model(
            tokens,
            state_idxs,
            action_idxs,
            attn_mask,
            prefix_embs=prefix_embs,
            prefix_attn_mask=prefix_attn_mask,
            remove_prefix_position_embs=remove_prefix_position_embs,
            qv_kwargs={"is_causal": False},
            target_kwargs={"is_causal": False},
            policy_kwargs={"is_causal": False},
        )["model_outputs"]
        kvs = {"qv": model_outputs["qv_model_outputs"]["past_key_values"]}
        if self.iql_model.actor_target is not None:
            kvs["target"] = model_outputs["target_model_outputs"]["past_key_values"]
        if self.iql_model.actor is not None:
            kvs["policy"] = model_outputs["policy_model_outputs"]["past_key_values"]
        dialogue_lens = attn_mask.sum(dim=1)
        tokens = pad_sequence(
            torch.repeat_interleave(tokens, num_generations, dim=0),
            max_length,
            tokenizer.pad_token_id,
            device,
            1,
        )
        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)
        kvs["qv"] = map_all_kvs(
            lambda x: pad_sequence(
                torch.repeat_interleave(x, num_generations, dim=0),
                max_length,
                0.0,
                device,
                2,
            ),
            kvs["qv"],
        )
        if "target" in kvs:
            kvs["target"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, num_generations, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["target"],
            )
        if "policy" in kvs:
            kvs["policy"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, num_generations, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["policy"],
            )
        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        kls = torch.full(
            (dialogue_lens.shape[0],),
            math.log(num_generations) - ((num_generations - 1) / num_generations),
        ).to(device)
        advantages = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(device)
        state_idxs_temp, action_idxs_temp = torch.zeros(
            (
                dialogue_lens.shape[0],
                1,
            )
        ).long().to(device), torch.zeros(
            (
                dialogue_lens.shape[0],
                1,
            )
        ).long().to(
            device
        )
        t = torch.min(dialogue_lens).int()
        base_logits = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        while termination_mask.sum() > 0 and (t + prefix_t) < max_length:
            curr_token = tokens[:, t - 1].unsqueeze(1)
            curr_kvs = map_all_kvs(
                lambda x: x[:, :, : (t + prefix_t) - 1, :], kvs["qv"]
            )
            curr_target_kvs, curr_policy_kvs = curr_kvs, curr_kvs
            if "target" in kvs:
                map_all_kvs(lambda x: x[:, :, : (t + prefix_t) - 1, :], kvs["target"])
            if "policy" in kvs:
                map_all_kvs(lambda x: x[:, :, : (t + prefix_t) - 1, :], kvs["policy"])
            iql_outputs = self.iql_model(
                curr_token,
                state_idxs_temp,
                action_idxs_temp,
                None,
                qv_kwargs={"past_key_values": curr_kvs},
                policy_kwargs={"past_key_values": curr_policy_kvs},
                target_kwargs={"past_key_values": curr_target_kvs},
                is_causal=False,
            )
            model_outputs, logits = iql_outputs["model_outputs"], iql_outputs["logits"]

            logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1, float("-inf"), 1e7
            )
            logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens, 1e7
            )
            edited_logits = process_logits(
                logits.clone(), temp=temp, top_k=top_k, top_p=top_p
            )

            vs, qs = iql_outputs["target_vs"], iql_outputs["target_qs"]
            if exp_adv:
                adv_logits = adv_weight * (qs - vs.unsqueeze(2))
            else:
                adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
                adv_logits = adv_weight * adv_sign + (1 - adv_weight) * (1 - adv_sign)
                adv_logits = torch.log(adv_logits)
            if adv_clip is not None:
                adv_logits = torch.clip(adv_logits, max=adv_clip)
            adv_logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1, float("-inf"), 1e7
            )
            adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens, 1e7
            )

            full_logits = (
                (edited_logits if include_logits else 0.0)
                + (adv_logits if include_adv else 0.0)
                + base_logits.unsqueeze(1).unsqueeze(2)
            )

            cat_dist = torch.distributions.categorical.Categorical(
                logits=full_logits[:, 0]
            )
            original_cat_dist = torch.distributions.categorical.Categorical(
                logits=logits[:, 0]
            )

            new_tokens = cat_dist.sample()
            log_probs += cat_dist.log_prob(new_tokens)
            kls += cat_dist.log_prob(new_tokens) - original_cat_dist.log_prob(
                new_tokens
            )
            qs_chosen = torch.gather(
                qs.squeeze(1), dim=1, index=new_tokens.unsqueeze(1)
            ).squeeze(1)
            advantages += qs_chosen - vs.squeeze(1)
            tokens[:, t] = new_tokens
            kvs["qv"] = update_kvs(
                kvs["qv"],
                model_outputs["qv_model_outputs"]["past_key_values"],
                torch.arange(0, n).to(device),
                (t + prefix_t) - 1,
            )
            if "target" in kvs:
                kvs["target"] = update_kvs(
                    kvs["target"],
                    model_outputs["target_model_outputs"]["past_key_values"],
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            if "policy" in kvs:
                kvs["policy"] = update_kvs(
                    kvs["policy"],
                    model_outputs["policy_model_outputs"]["past_key_values"],
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= 1 - int(
                        termination_condition(
                            tokenizer.decode(
                                tokens[idx, :].tolist(),
                                clean_up_tokenization_spaces=False,
                            )
                        )
                    )
            t += 1
            termination_mask *= ((t - dialogue_lens) < max_generation_len).int()

        scores = (
            (advantages * rerank_advantage_weight)
            + (log_probs * rerank_log_prob_weight)
        ).reshape(-1, num_generations)
        order = torch.argsort(-scores, dim=1)
        output_strs = [
            tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False)
            for i in range(len(tokens))
        ]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(num_generations):
                processed_str = output_strs[i * num_generations + order[i, x]][
                    len(input_strs[i]) :
                ].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.pad_token_id)
                        )
                    ].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.eoa_token_id)
                        )
                    ].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        scores = torch.gather(scores, dim=1, index=order)
        log_probs = torch.gather(
            log_probs.reshape(-1, num_generations), dim=1, index=order
        )
        kls = torch.gather(kls.reshape(-1, num_generations), dim=1, index=order)
        return (
            list(zip(input_strs, processed_outputs)),
            log_probs.reshape(-1, num_generations),
            kls,
        )

    def beam_raw(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        state_idxs: torch.Tensor,
        action_idxs: torch.Tensor,
        termination_condition: Callable[[np.ndarray], bool],
        max_generation_len: Optional[int] = None,
        beam_width=1,
        temp=1.0,
        top_k=None,
        top_p=None,
        exp_adv=False,
        adv_weight=0.0,
        adv_clip=None,
        include_logits=True,
        include_adv=True,
        prefix_embs: Optional[torch.Tensor] = None,
        prefix_attn_mask: Optional[torch.Tensor] = None,
        remove_prefix_position_embs: bool = False,
    ):
        tokenizer = self.iql_model.dataset.tokenizer
        max_length = self.iql_model.dataset.max_len
        if max_length is None:
            max_length = self.iql_model.model.block_size
        max_length = min(max_length, self.iql_model.model.block_size)
        device = self.iql_model.device
        bsize, vocab_size = tokens.shape[0], tokenizer.num_tokens()
        n = bsize * beam_width
        if max_generation_len is None:
            max_generation_len = max_length + 1
        input_strs = [
            tokenizer.decode(
                tokens[i, :][: attn_mask[i, :].sum().long()].tolist(),
                clean_up_tokenization_spaces=False,
            )
            for i in range(len(tokens))
        ]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        model_outputs = self.iql_model(
            tokens,
            state_idxs,
            action_idxs,
            attn_mask,
            prefix_embs=prefix_embs,
            prefix_attn_mask=prefix_attn_mask,
            remove_prefix_position_embs=remove_prefix_position_embs,
            qv_kwargs={"is_causal": False},
            target_kwargs={"is_causal": False},
            policy_kwargs={"is_causal": False},
        )["model_outputs"]
        kvs = {"qv": model_outputs["qv_model_outputs"]["past_key_values"]}
        if self.iql_model.actor_target is not None:
            kvs["target"] = model_outputs["target_model_outputs"]["past_key_values"]
        if self.iql_model.actor is not None:
            kvs["policy"] = model_outputs["policy_model_outputs"]["past_key_values"]
        original_dialogue_lens = attn_mask.sum(dim=1)
        batch_indicator = torch.stack(
            beam_width * [torch.arange(0, bsize).to(device)], dim=1
        )

        tokens = pad_sequence(
            torch.repeat_interleave(tokens, beam_width, dim=0),
            max_length,
            tokenizer.pad_token_id,
            device,
            1,
        )
        dialogue_lens = torch.repeat_interleave(
            original_dialogue_lens, beam_width, dim=0
        )
        kvs["qv"] = map_all_kvs(
            lambda x: pad_sequence(
                torch.repeat_interleave(x, beam_width, dim=0),
                max_length,
                0.0,
                device,
                2,
            ),
            kvs["qv"],
        )
        if "target" in kvs:
            kvs["target"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, beam_width, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["target"],
            )
        if "policy" in kvs:
            kvs["policy"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, beam_width, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["policy"],
            )
        curr_scores = torch.zeros(bsize, beam_width).to(device)  # (batch, k)
        logit_scores = torch.zeros(bsize, beam_width).to(device)  # (batch, k)
        termination_mask = torch.full((n,), 1).to(device)
        state_idxs_temp, action_idxs_temp = torch.zeros(
            (
                dialogue_lens.shape[0],
                1,
            )
        ).long().to(device), torch.zeros(
            (
                dialogue_lens.shape[0],
                1,
            )
        ).long().to(
            device
        )
        t = torch.min(dialogue_lens).int()
        base_logits = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        while termination_mask.sum() > 0 and (t + prefix_t) < max_length:
            curr_token = tokens[:, t - 1].unsqueeze(1)
            curr_kvs = map_all_kvs(
                lambda x: x[:, :, : (t + prefix_t) - 1, :], kvs["qv"]
            )
            curr_target_kvs, curr_policy_kvs = curr_kvs, curr_kvs
            if "target" in kvs:
                map_all_kvs(lambda x: x[:, :, : (t + prefix_t) - 1, :], kvs["target"])
            if "policy" in kvs:
                map_all_kvs(lambda x: x[:, :, : (t + prefix_t) - 1, :], kvs["policy"])
            iql_outputs = self.iql_model(
                curr_token,
                state_idxs_temp,
                action_idxs_temp,
                None,
                qv_kwargs={"past_key_values": curr_kvs},
                policy_kwargs={"past_key_values": curr_policy_kvs},
                target_kwargs={"past_key_values": curr_target_kvs},
                is_causal=False,
            )
            model_outputs, logits = iql_outputs["model_outputs"], iql_outputs["logits"]

            logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1, float("-inf"), 1e7
            )
            logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens, 1e7
            )
            edited_logits = process_logits(
                logits.clone(), temp=temp, top_k=top_k, top_p=top_p
            )

            vs, qs = iql_outputs["target_vs"], iql_outputs["target_qs"]
            if exp_adv:
                adv_logits = adv_weight * (qs - vs.unsqueeze(2))
            else:
                adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
                adv_logits = adv_weight * adv_sign + (1 - adv_weight) * (1 - adv_sign)
                adv_logits = torch.log(adv_logits)
            if adv_clip is not None:
                adv_logits = torch.clip(adv_logits, max=adv_clip)
            adv_logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1, float("-inf"), 1e7
            )
            adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens, 1e7
            )

            full_logits = (
                (edited_logits if include_logits else 0.0)
                + (adv_logits if include_adv else 0.0)
                + base_logits.unsqueeze(1).unsqueeze(2)
            )

            scores = (
                (
                    torch.log(F.softmax(full_logits, dim=-1))
                    .reshape(1, bsize, beam_width, -1)
                    .permute(3, 0, 1, 2)
                    + curr_scores
                )
                .permute(1, 2, 3, 0)
                .reshape(1, bsize, -1)
            )  # (time, batch, k*vocab)
            scores[0, :, vocab_size:] = scores[0, :, vocab_size:].masked_fill_(
                (t == original_dialogue_lens)
                .unsqueeze(1)
                .repeat(1, scores.shape[2] - vocab_size),
                float("-inf"),
            )
            curr_scores, top_k_ = torch.topk(
                scores[0, :, :], k=beam_width, dim=1
            )  # (batch, k), (batch, k)
            tokens = tokens[
                (batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1), :
            ]
            logits = logits[
                (batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1),
                :,
                :,
            ]
            logit_scores += (
                torch.gather(
                    torch.log(F.softmax(logits, dim=-1)).squeeze(1),
                    dim=1,
                    index=(top_k_.reshape(-1) % vocab_size).unsqueeze(1),
                )
                .squeeze(1)
                .reshape(-1, beam_width)
            )
            tokens[:, t] = top_k_.reshape(-1) % vocab_size  # (batch*k,)
            fixed_kvs = map_all_kvs(
                lambda x: x[
                    (
                        batch_indicator * beam_width
                        + torch.div(top_k_, vocab_size, rounding_mode="trunc")
                    ).reshape(-1),
                    :,
                    :,
                    :,
                ],
                model_outputs["qv_model_outputs"]["past_key_values"],
            )
            kvs["qv"] = map_all_kvs(
                lambda x: x[
                    (
                        batch_indicator * beam_width
                        + torch.div(top_k_, vocab_size, rounding_mode="trunc")
                    ).reshape(-1),
                    :,
                    :,
                    :,
                ],
                kvs["qv"],
            )
            kvs["qv"] = update_kvs(
                kvs["qv"], fixed_kvs, torch.arange(0, n).to(device), (t + prefix_t) - 1
            )
            if "target" in kvs:
                fixed_target_kvs = map_all_kvs(
                    lambda x: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(top_k_, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    model_outputs["target_model_outputs"]["past_key_values"],
                )
                kvs["target"] = map_all_kvs(
                    lambda x: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(top_k_, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    kvs["target"],
                )
                kvs["target"] = update_kvs(
                    kvs["target"],
                    fixed_target_kvs,
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            if "policy" in kvs:
                fixed_policy_kvs = map_all_kvs(
                    lambda x: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(top_k_, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    model_outputs["policy_model_outputs"]["past_key_values"],
                )
                kvs["policy"] = map_all_kvs(
                    lambda x: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(top_k_, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    kvs["policy"],
                )
                kvs["policy"] = update_kvs(
                    kvs["policy"],
                    fixed_policy_kvs,
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            termination_mask = termination_mask[
                (batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1)
            ]
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= 1 - int(
                        termination_condition(
                            tokenizer.decode(
                                tokens[idx, :].tolist(),
                                clean_up_tokenization_spaces=False,
                            )
                        )
                    )
            t += 1
            termination_mask *= ((t - dialogue_lens) < max_generation_len).int()

        output_strs = [
            tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False)
            for i in range(n)
        ]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(beam_width):
                processed_str = output_strs[i * beam_width + x][
                    len(input_strs[i]) :
                ].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.pad_token_id)
                        )
                    ].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.eoa_token_id)
                        )
                    ].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return list(zip(input_strs, processed_outputs)), curr_scores, -logit_scores

    def generate(
        self, items, termination_condition: Callable[[np.ndarray], bool], **kwargs
    ):
        prepared_inputs = self.iql_model.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs["tokens"], prepared_inputs["attn_mask"]
        state_idxs, action_idxs = (
            prepared_inputs["state_idxs"],
            prepared_inputs["action_idxs"],
        )
        if self.kind == "beam":
            method = self.beam_raw
        elif self.kind == "sample":
            method = self.sample_raw
        else:
            raise NotImplementedError
        generations, info, kls = method(
            tokens, attn_mask, state_idxs, action_idxs, termination_condition, **kwargs
        )
        return generations, info, kls

    def act(self, obs):
        item = DataPoint.from_obs(
            obs, self.iql_model.dataset.tokenizer, self.iql_model.dataset.token_reward
        )
        generations, logprobs, kls = self.generate(
            [item], always_terminate, **self.generation_kwargs
        )
        self.kls_all.append(kls[0, 0].item())
        self.logprobs_all.append(logprobs[0, 0].item())
        return generations[0][1][0]

    def train(self):
        self.iql_model.train()

    def eval(self):
        self.iql_model.eval()


class ILQL_Evaluator:
    def __init__(self, env, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.all_results = []
        self.all_entropy = []

    def evaluate(self, model: ILQL, items):
        policy = ILQL_Policy(model, self.kind, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)["tokens"]
        total_token_reward = 0
        total_env_reward = 0
        for i in range(tokens.shape[0]):
            result, sequence = interact_environment(self.env, policy, None)
            self.all_results.append(
                (
                    result,
                    sequence,
                )
            )
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(
                DataPoint.get_token_reward(
                    result, model.dataset.tokenizer, model.dataset.token_reward
                )
            )
            total_env_reward += env_reward
            total_token_reward += token_reward
            if self.verbose:
                print(result)
                print("=" * 25)
                print("token reward:", token_reward)
                print("env reward:", env_reward)
                print("avg token reward:", total_token_reward / (i + 1))
                print("avg env reward:", total_env_reward / (i + 1))
                print("=" * 25)
        kl_total = sum(policy.kls_all)
        entropy_total = -sum(policy.logprobs_all)
        self.all_entropy.extend(policy.logprobs_all)
        return {
            "token_reward": (total_token_reward / tokens.shape[0], tokens.shape[0]),
            "env_reward": (total_env_reward / tokens.shape[0], tokens.shape[0]),
            "kl": (kl_total / len(policy.kls_all), len(policy.kls_all)),
            "entropy": (
                entropy_total / len(policy.logprobs_all),
                len(policy.logprobs_all),
            ),
        }

    def dump(self):
        return {"results": self.all_results, "entropies": self.all_entropy}


class TopAdvantageNGrams:
    def __init__(self, data, print_every, print_k, n_gram):
        self.data = data
        self.print_every = print_every
        self.print_k = print_k
        self.n_gram = n_gram

    def evaluate(self, model, items):
        top_actions = defaultdict(float)
        total_actions = defaultdict(int)
        for i in tqdm(range(self.data.size())):
            item = self.data.get_item(i)
            prepared_inputs = model.prepare_inputs([item])
            tokens, a_idx = prepared_inputs["tokens"], prepared_inputs["action_idxs"]
            model_outputs = model.get_qvs([item])
            select_tokens = torch.gather(tokens[0, 1:], dim=0, index=a_idx[0, :])
            advantages = (
                model_outputs["target_qs"][0, :] - model_outputs["target_vs"][0, :]
            )
            curr_idx = 0
            for x, token in enumerate(select_tokens):
                start_idx = curr_idx
                if self.n_gram is not None:
                    if (
                        x - self.n_gram >= 0
                        and a_idx[0, x - self.n_gram].item()
                        > a_idx[0, start_idx].item()
                    ):
                        start_idx = x - self.n_gram
                    if x - start_idx < self.n_gram:
                        continue
                elif select_tokens[x].item() != self.data.tokenizer.eoa_token_id:
                    continue
                total_advantage = advantages[start_idx:x].sum().item()
                utterance = self.data.tokenizer.decode(
                    tokens[
                        0, (a_idx[0, start_idx].item() + 1) : (a_idx[0, x].item() + 1)
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
                top_actions[utterance] += total_advantage
                total_actions[utterance] += 1
                if select_tokens[x].item() == self.data.tokenizer.eoa_token_id:
                    curr_idx = x + 1
            if i % self.print_every == 0:
                ranked_actions = sorted(
                    {
                        k: top_actions[k] / total_actions[k]
                        for k in total_actions.keys()
                    }.items(),
                    key=lambda x: x[1],
                )
                print(ranked_actions[-self.print_k :])
                print(ranked_actions[: self.print_k])


def interact_environment(env, policy, obs):
    obs_sequence = []
    if obs is None:
        obs = env.reset()
    while not env.is_terminal():
        action = policy.act(obs)
        new_obs, r, t = env.step(action)
        obs_sequence.append((obs, action, r, t))
        obs = new_obs
    obs_sequence.append((obs, None, 0, True))
    return obs, obs_sequence


def map_pytree(f: Callable[[Union[np.ndarray, torch.Tensor]], Any], item: Any):
    if isinstance(item, dict):
        return {k: map_pytree(f, v) for k, v in item.items()}
    elif isinstance(item, list) or isinstance(item, set) or isinstance(item, tuple):
        return [map_pytree(f, v) for v in item]
    elif isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
        return f(item)
    else:
        return item


def to(item: Any, device: torch.device):
    return map_pytree(lambda x: torch.tensor(x).to(device), item)


def to_decorator(f, device):
    def new_f(*args, **kwargs):
        return to(f(*args, **kwargs), device)

    return new_f


def parameter_norm(model: nn.Module):
    norm = 0.0
    for param in model.parameters():
        norm += (param.norm() ** 2).item()
    return math.sqrt(norm)
