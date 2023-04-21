import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from typing import Any, Callable, List, Tuple, Union, Optional
import wandb
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.networks.evolvable_gpt import EvolvableGPT


class ILQL(nn.Module):
    """The Implicit Language Q Learning algorithm class. ILQL paper: https://arxiv.org/pdf/2206.11871.pdf

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Network configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 5
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param tau: For soft update of target network parameters, defaults to 1e-3
    :type tau: float, optional
    :param mutation: Most recent mutation to agent, defaults to None
    :type mutation: str, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
            self,
            dataset,
            net_config={
                'arch': 'gpt',
                'vocab_size': 50257,
                'n_embd': 768,
                'n_head': 12,
                'dim_feedfwd': 3072,
                'block_size': 1024,
                'activation': 'gelu',
                'dropout': 0.1,
                'layer_norm_eps': 1e-5,
                'min_layers': 8,
                'max_layers': 16,
                'bias': True,
            },
            index=0,
            batch_size=64,
            lr=1e-4,
            learn_step=5,
            alpha=0.005,
            beta=0.,
            gamma=0.99,
            tau=0.6,
            mutation=None,
            transition_weight=0.,
            clip_weight=None,
            value_max=None,
            value_min=None,
            detach_v=False,
            detach_q=False,
            detach_pi=False,
            double_q=True,
            per_token=True,
            exp_weights=True,
            dm_margin=0.,
            cql_temp=1.,
            weight_decay=0.,
            device='cpu'):
        super(ILQL, self).__init__()

        self.algo = 'ILQL'
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
        self.learn_step = learn_step
        self.mut = mutation
        self.device = device
        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # model
        self.model = EvolvableGPT(
            n_layer=net_config['n_layer'],
            vocab_size=net_config['vocab_size'],
            n_embd=net_config['n_embd'],
            n_head=net_config['n_head'],
            dim_feedfwd=net_config['dim_feedfwd'],
            block_size=net_config['block_size'],
            dropout=net_config['dropout'],
            activation=net_config['activation'],
            layer_norm_eps=net_config['layer_norm_eps'],
            min_layers=net_config['min_layers'],
            max_layers=net_config['max_layers'],
            bias=net_config['bias'],
            device=self.device).to(
            self.device)
        # lm policy
        self.actor = EvolvableGPT(
            n_layer=net_config['n_layer'],
            vocab_size=net_config['vocab_size'],
            n_embd=net_config['n_embd'],
            n_head=net_config['n_head'],
            dim_feedfwd=net_config['dim_feedfwd'],
            block_size=net_config['block_size'],
            dropout=net_config['dropout'],
            activation=net_config['activation'],
            layer_norm_eps=net_config['layer_norm_eps'],
            min_layers=net_config['min_layers'],
            max_layers=net_config['max_layers'],
            bias=net_config['bias'],
            device=self.device).to(
            self.device)
        # lm target
        self.actor_target = EvolvableGPT(
            n_layer=net_config['n_layer'],
            vocab_size=net_config['vocab_size'],
            n_embd=net_config['n_embd'],
            n_head=net_config['n_head'],
            dim_feedfwd=net_config['dim_feedfwd'],
            block_size=net_config['block_size'],
            dropout=net_config['dropout'],
            activation=net_config['activation'],
            layer_norm_eps=net_config['layer_norm_eps'],
            min_layers=net_config['min_layers'],
            max_layers=net_config['max_layers'],
            bias=net_config['bias'],
            device=self.device).to(
            self.device)

        self.actor.load_state_dict(self.model.state_dict())
        self.actor_target.load_state_dict(self.model.state_dict())

        # v and q networks
        self.v = EvolvableMLP(
            num_inputs=net_config['d_model'],
            num_outputs=1,
            hidden_size=[
                net_config['d_model'] * 2,
                net_config['d_model'] * 2],
            device=self.device).to(
            self.device)
        self.q = EvolvableMLP(
            num_inputs=net_config['d_model'],
            num_outputs=1,
            hidden_size=[
                net_config['d_model'] * 2,
                net_config['d_model'] * 2],
            device=self.device).to(
            self.device)
        self.target_q = EvolvableMLP(
            num_inputs=net_config['d_model'],
            num_outputs=1,
            hidden_size=[
                net_config['d_model'] * 2,
                net_config['d_model'] * 2],
            device=self.device).to(
            self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        if self.double_q:
            self.q2 = EvolvableMLP(
                num_inputs=net_config['d_model'],
                num_outputs=1,
                hidden_size=[
                    net_config['d_model'] * 2,
                    net_config['d_model'] * 2],
                device=self.device).to(
                self.device)
            self.target_q2 = EvolvableMLP(
                num_inputs=net_config['d_model'],
                num_outputs=1,
                hidden_size=[
                    net_config['d_model'] * 2,
                    net_config['d_model'] * 2],
                device=self.device).to(
                self.device)
            self.target_q2.load_state_dict(self.q2.state_dict())

        self.pi = EvolvableMLP(
            num_inputs=net_config['d_model'],
            num_outputs=1,
            hidden_size=[
                net_config['d_model'] * 2,
                net_config['d_model'] * 2],
            device=self.device).to(
            self.device)

        self.optimizer = optim.AdamW(
            self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self,
                tokens: torch.Tensor,
                state_idxs: torch.Tensor,
                action_idxs: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                prefix_embs: Optional[torch.Tensor] = None,
                skip_policy_on_train: bool = False,
                detach_full_policy: bool = False):
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
        # Model forward passes
        model_outputs, model_hidden_states = self.model.transformer(
            tokens, attn_mask)
        hidden_states = model_hidden_states[-1]

        with torch.no_grad():
            target_outputs, target_hidden_states = self.actor_target.transformer(
                tokens, attn_mask)
        target_hidden_states = target_hidden_states[-1]

        # Prepare policy inputs
        if skip_policy_on_train and self.training:
            policy_outputs = model_outputs
            policy_hidden_states = hidden_states
        else:
            if detach_full_policy:
                with torch.no_grad():
                    policy_outputs, policy_hidden_states = self.actor.transformer(
                        tokens, attn_mask)
            else:
                policy_outputs, policy_hidden_states = self.actor.transformer(
                    tokens, attn_mask)
            policy_hidden_states = policy_hidden_states[-1]

        all_model_outputs = {'qv_model_outputs': model_outputs,
                             'policy_model_outputs': target_outputs,
                             'target_model_outputs': policy_outputs}
        all_hidden_states = {'qv_hidden_states': model_hidden_states,
                             'policy_hidden_states': target_hidden_states,
                             'target_hidden_states': policy_hidden_states}

        state_hidden_states = torch.gather(
            input=hidden_states, dim=1, index=state_idxs.unsqueeze(2).repeat(
                1, 1, self.net_config['d_model']))
        action_hidden_states = torch.gather(
            input=hidden_states, dim=1, index=action_idxs.unsqueeze(2).repeat(
                1, 1, self.net_config['d_model']))
        action_target_hidden_states = torch.gather(
            input=target_hidden_states, dim=1, index=action_idxs.unsqueeze(2).repeat(
                1, 1, self.net_config['d_model']))
        vs = self.v(state_hidden_states.detach()
                    if self.detach_v else state_hidden_states).squeeze(2)
        qs = self.q(action_hidden_states.detach()
                    if self.detach_q else action_hidden_states)
        if self.double_q:
            qs2 = self.q2(action_hidden_states.detach()
                          if self.detach_q else action_hidden_states)
        with torch.no_grad():
            target_qs = self.target_q(action_target_hidden_states)
            if self.double_q:
                target_qs2 = self.target_q2(action_target_hidden_states)
        if skip_policy_on_train and self.training and self.lm_policy is not None:
            logits = torch.zeros(
                (policy_hidden_states.shape[0],
                 policy_hidden_states.shape[1],
                 self.dataset.tokenizer.num_tokens(),
                 )).to(
                self.device)
        else:
            if detach_full_policy:
                with torch.no_grad():
                    logits = self.pi(policy_hidden_states.detach(
                    ) if self.detach_pi else policy_hidden_states)
            else:
                logits = self.pi(policy_hidden_states.detach()
                                 if self.detach_pi else policy_hidden_states)
        return {
            'model_outputs': all_model_outputs,
            'hidden_states': all_hidden_states,
            'vs': vs,
            'target_vs': vs,
            'qs': (
                qs,
                qs2,
            ) if self.double_q else qs,
            'target_qs': self.clip_values(
                torch.minimum(
                    target_qs,
                    target_qs2) if self.double_q else target_qs),
            'logits': logits,
        }

    def clip_values(self, values):
        if self.value_min is not None or self.value_max is not None:
            return torch.clip(values, self.value_min, self.value_max)
        return values

    def get_downstream_rs(self, rs, gamma):
        gamma_row = torch.cumprod(torch.full(
            rs.shape, gamma).to(self.device), dim=1)
        gamma_tensor = torch.triu(
            gamma_row.unsqueeze(1) / gamma_row.unsqueeze(2))
        return (gamma_tensor * rs.unsqueeze(1)).sum(dim=2)

    def get_weights(self,
                    tokens: torch.Tensor,
                    vs: torch.Tensor,
                    qs: Optional[torch.Tensor],
                    state_idxs: torch.Tensor,
                    action_idxs: torch.Tensor,
                    terminals: torch.Tensor):
        weights = torch.full(
            tokens.shape, self.transition_weight).to(self.device)
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
                weights[i], dim=0, index=action_idxs[i, :n[i]], src=w_values[i, :n[i]])
        if self.clip_weight is not None:
            weights = torch.clip(weights, max=self.clip_weight)
        # print(list(map(lambda x: list(map(lambda y: (y[0], self.dataset.tokenizer.id_to_token(y[1].item()),), zip(*x))), zip(weights.detach().cpu().tolist(), tokens))))
        return weights

    def awac_loss(self, tokens, attn_mask, logits, w):
        w = w.detach()
        losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction='none')
        losses = losses.reshape(tokens.shape[0], tokens.shape[1] - 1)
        return (losses * w[:, :-1] * attn_mask[:, 1:]).sum() / attn_mask[:, 1:].sum()

    def get_v_loss(self, vs, target_qs, terminals):
        target_qs = target_qs.detach()
        return (((target_qs >= vs).int() * self.tau * (target_qs - vs)**2 + (target_qs < vs).int() * (1 - self.tau)
                * (target_qs - vs)**2) * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)

    def get_q_loss(self, vns, qs, rs, gamma, terminals):
        vns = vns.detach()
        if self.double_q:
            q1, q2 = qs
            l1 = ((((1 - terminals[:, 1:]) * vns * gamma + rs - q1) ** 2) * (1 - \
                  terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            l2 = ((((1 - terminals[:, 1:]) * vns * gamma + rs - q2) ** 2) * (1 - \
                  terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            return l1 + l2
        return ((((1 - terminals[:, 1:]) * vns * gamma + rs - qs) ** 2) * (1 - \
                terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)

    def get_cql_loss(self, qs, action_tokens, terminals):
        n = (1 - terminals[:, :-1]).sum()
        if self.double_q:
            q1, q2 = qs
            b, t, d = q1.shape
            return ((F.cross_entropy(q1.reshape(-1,
                                                d) / self.cql_temp,
                                     action_tokens.reshape(-1),
                                     reduction='none').reshape(b,
                                                               t) * (1 - terminals[:,
                                                                                   :-1])) + (F.cross_entropy(q2.reshape(-1,
                                                                                                                        d) / self.cql_temp,
                                                                                                             action_tokens.reshape(-1),
                                                                                                             reduction='none').reshape(b,
                                                                                                                                       t) * (1 - terminals[:,
                                                                                                                                                           :-1]))).sum() / max(n.item(),
                                                                                                                                                                               1.0)
        b, t, d = qs.shape
        return (F.cross_entropy(qs.reshape(-1, d) / self.cql_temp, action_tokens.reshape(-1),
                reduction='none').reshape(b, t) * (1 - terminals[:, :-1])).sum() / max(n.item(), 1.0)

    def get_dm_loss(self, qs, data_qs, terminals, margin):
        n = (1 - terminals[:, :-1]).sum()
        if self.double_q:
            q1, q2 = qs
            data_q1, data_q2 = data_qs
            return (((torch.max(q1 - data_q1.unsqueeze(-1) + margin,
                                torch.tensor(0.0).to(self.device)) ** 2).sum(dim=-1) * (1 - terminals[:,
                                                                                                      :-1])) + ((torch.max(q2 - data_q2.unsqueeze(-1) + margin,
                                                                                                                           torch.tensor(0.0).to(self.device)) ** 2).sum(dim=-1) * (1 - terminals[:,
                                                                                                                                                                                                 :-1]))).sum() / max(n.item(),
                                                                                                                                                                                                                     1.0)
        return ((torch.max(qs - data_qs.unsqueeze(-1) + margin, torch.tensor(0.0).to(self.device))
                ** 2).sum(dim=-1) * (1 - terminals[:, :-1])).sum() / max(n.item(), 1.0)

    def prepare_inputs(self, items):
        if isinstance(items, dict):
            return items
        return to(self.dataset.collate(items, self.device), self.device)

    def get_qvs(self, items,
                prefix_embs: Optional[torch.Tensor] = None,
                prefix_attn_mask: Optional[torch.Tensor] = None,
                remove_prefix_position_embs: bool = False,
                qv_kwargs=None, policy_kwargs=None, target_kwargs=None,
                **kwargs):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        s_idx, a_idx = prepared_inputs['state_idxs'], prepared_inputs['action_idxs']
        rs, terminals = prepared_inputs['rewards'], prepared_inputs['terminals']
        self_outputs = self(tokens, attn_mask, s_idx, a_idx,
                            prefix_embs, prefix_attn_mask,
                            remove_prefix_position_embs,
                            qv_kwargs, policy_kwargs, target_kwargs,
                            **kwargs)
        model_outputs, vs, qs = self_outputs['model_outputs'], self_outputs['vs'], self_outputs['qs']
        target_qs, logits = self_outputs['target_qs'], self_outputs['logits']
        vt = vs[:, :-1]
        vtp1 = vs[:, 1:]
        select_tokens = torch.gather(tokens[:, 1:], dim=1, index=a_idx)
        cql_term = self.get_cql_loss(qs, select_tokens, terminals)
        full_qs = qs
        if self.double_q:
            q1, q2 = qs
            q1 = torch.gather(
                q1, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
            q2 = torch.gather(
                q2, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
            # tok_seq = [self.dataset.tokenizer.id_to_token(token) for token in select_tokens[0].detach().cpu().tolist()][:(1-terminals[0, :-1]).sum()]
            # max_q_seq = torch.max(q1, q2)[0, :(1-terminals[0, :-1]).sum()].detach().cpu().tolist()
            # print(self.dataset.tokenizer.decode(tokens[0, :][:attn_mask[0, :].sum().long()].tolist(), clean_up_tokenization_spaces=False))
            # print(list(zip(tok_seq, max_q_seq)))
            # print(rs)
            qs = (q1, q2,)
        else:
            qs = torch.gather(
                qs, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
        dm_term = self.get_dm_loss(full_qs, qs, terminals, self.dm_margin)
        target_qs = torch.gather(
            target_qs, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            weights = self.get_weights(
                tokens, vt, target_qs, s_idx, a_idx, terminals)
        return {
            'tokens': tokens,
            'attn_mask': attn_mask,
            'model_outputs': model_outputs,
            'vs': vt,
            'qs': qs,
            'vns': vtp1,
            'target_vs': vt,
            'target_qs': target_qs,
            'target_vns': vtp1,
            'rs': rs,
            'terminals': terminals,
            'logits': logits,
            'weights': weights,
            'cql_term': cql_term,
            'dm_term': dm_term,
        }

    def get_loss(self,
                 items,
                 awac_weight=0.0,
                 v_loss_weight=0.0,
                 q_loss_weight=0.0,
                 cql_loss_weight=0.0,
                 dm_loss_weight=0.0,
                 mc_returns=False):
        prepared_inputs = self.prepare_inputs(items)
        a_idx = prepared_inputs['action_idxs']
        get_qvs_outputs = self.get_qvs(items,
                                       qv_kwargs={'output_attentions': True},
                                       policy_kwargs={
                                           'output_attentions': True},
                                       target_kwargs={
                                           'output_attentions': True},
                                       skip_policy_on_train=(
                                           awac_weight == 0.0),
                                       )
        tokens, attn_mask, model_outputs = get_qvs_outputs[
            'tokens'], get_qvs_outputs['attn_mask'], get_qvs_outputs['model_outputs']
        vs, qs = get_qvs_outputs['vs'], get_qvs_outputs['qs']
        vns, target_qs, rs = get_qvs_outputs['vns'], get_qvs_outputs['target_qs'], get_qvs_outputs['rs']
        terminals, logits, weights = get_qvs_outputs[
            'terminals'], get_qvs_outputs['logits'], get_qvs_outputs['weights']

        logs = {}
        transformer_logs = {}
        transformer_logs['qv_transformer_logs'] = get_transformer_logs(
            model_outputs['qv_model_outputs'].attentions, self.model, attn_mask)
        if self.lm_policy is not None and (not (self.training and awac_weight == 0.0)):
            transformer_logs['policy_transformer_logs'] = get_transformer_logs(
                model_outputs['policy_model_outputs'].attentions, self.lm_policy, attn_mask)
        if self.lm_target is not None:
            transformer_logs['target_transformer_logs'] = get_transformer_logs(
                model_outputs['target_model_outputs'].attentions, self.lm_target, attn_mask)
        n = (1 - terminals[:, :-1]).sum().item()
        rs_downstream = self.get_downstream_rs(rs, self.gamma)
        if mc_returns:
            v_loss = self.get_v_loss(vs, rs_downstream, terminals)
        else:
            v_loss = self.get_v_loss(vs, target_qs, terminals)
        q_loss = self.get_q_loss(vns, qs, rs, self.gamma, terminals)
        cql_loss = get_qvs_outputs['cql_term']
        dm_loss = get_qvs_outputs['dm_term']
        token_loss = self.awac_loss(tokens, attn_mask, logits, weights)
        logs['token_loss'] = (token_loss.item(), n)
        loss = awac_weight * token_loss + v_loss_weight * v_loss + q_loss_weight * \
            q_loss + cql_loss_weight * cql_loss + dm_loss_weight * dm_loss
        logs['v_loss'] = (v_loss.item(), n)
        logs['q_loss'] = (q_loss.item(), n)
        logs['cql_loss'] = (cql_loss.item(), n)
        logs['dm_loss'] = (dm_loss.item(), n)
        advantages = sum([((target_qs[i] - vs[i])[:(1 - terminals[i, :-1]).sum().long().item()]
                           ).detach().cpu().tolist() for i in range(tokens.shape[0])], [])
        if self.double_q:
            q1, q2 = qs
            logs['q1_avg'] = ((q1 * (1 - terminals[:, :-1])
                               ).sum().item() / max(n, 1), n)
            logs['q1_var'] = (((((q1 - logs['q1_avg'][0]) ** 2) *
                              (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
            logs['q2_avg'] = ((q2 * (1 - terminals[:, :-1])
                               ).sum().item() / max(n, 1), n)
            logs['q2_var'] = (((((q2 - logs['q2_avg'][0]) ** 2) *
                              (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        else:
            logs['q_avg'] = ((qs * (1 - terminals[:, :-1])
                              ).sum().item() / max(n, 1), n)
            logs['q_var'] = (((((qs - logs['q_avg'][0]) ** 2) *
                             (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        logs['v_avg'] = ((vs * (1 - terminals[:, :-1])
                          ).sum().item() / max(n, 1), n)
        logs['v_var'] = (((((vs - logs['v_avg'][0]) ** 2) *
                         (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        act_weights = torch.gather(weights, dim=1, index=a_idx)
        logs['act_weight_avg'] = (
            ((act_weights * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), n)
        logs['transformer'] = transformer_logs

        def postproc_f(x):
            return x.update({'loss': awac_weight * x['token_loss'] + q_loss_weight * x['q_loss'] + v_loss_weight * x['v_loss'] + cql_loss_weight * x['cql_loss'] + dm_loss_weight * x['dm_loss']})
        def hist_f(x):
            return x.update({'advantage_hist': wandb.Histogram(advantages)})
        
        return loss, logs, [postproc_f, hist_f]

    def score(self,
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
              action_mask: Optional[torch.Tensor] = None):
        trivial_value_query = False
        if state_idxs is None or action_idxs is None:
            state_idxs = torch.full(
                (tokens.shape[0], 1,), tokens.shape[1] - 1).long().to(self.device)
            action_idxs = torch.full(
                (tokens.shape[0], 1,), tokens.shape[1] - 1).long().to(self.device)
            trivial_value_query = True
        self_outputs = self(tokens, attn_mask,
                            state_idxs, action_idxs,
                            prefix_embs, prefix_attn_mask,
                            remove_prefix_position_embs,
                            qv_kwargs, policy_kwargs, target_kwargs)
        model_outputs = self_outputs['model_outputs']
        weights = torch.zeros(self_outputs['logits'].shape).to(self.device)
        if include_advantage:
            if action_mask is None:
                action_mask = torch.ones((tokens.shape[0],)).to(self.device)
            vs, qs = self_outputs['target_vs'], self_outputs['target_qs']
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
                weights[i] += torch.scatter(weights[i], dim=0,
                                            index=action_idxs[i, :n[i]].unsqueeze(
                                                1).repeat(1, weights.shape[2]),
                                            src=w_values[i, :n[i], :]) * action_mask[i]
        if include_logits:
            logits = process_logits(
                self_outputs['logits'],
                temp=logit_temp,
                top_k=logit_top_k,
                top_p=logit_top_p)
            weights += torch.log(F.softmax(logits, dim=-1))
        return weights, model_outputs

    def get_scores(self,
                   items,
                   beta: float = 1.0,
                   exp_weights: bool = False,
                   clip_weight: Optional[float] = None,
                   logit_temp: float = 1.0,
                   logit_top_k: Optional[int] = None,
                   logit_top_p: Optional[float] = None,
                   include_logits: bool = False,
                   include_advantage: bool = True) -> torch.Tensor:
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        s_idx, a_idx = prepared_inputs['state_idxs'], prepared_inputs['action_idxs']
        return self.score(tokens, attn_mask, s_idx, a_idx,
                          beta=beta, exp_weights=exp_weights, clip_weight=clip_weight,
                          logit_temp=logit_temp, logit_top_k=logit_top_k,
                          logit_top_p=logit_top_p, include_logits=include_logits,
                          include_advantage=include_advantage, action_mask=None)[0]

    def initial_score(self,
                      items,
                      beta: float = 1.0,
                      exp_weights: bool = False,
                      clip_weight: Optional[float] = None,
                      logit_temp: float = 1.0,
                      logit_top_k: Optional[int] = None,
                      logit_top_p: Optional[float] = None,
                      include_logits: bool = False,
                      include_advantage: bool = True) -> Tuple[torch.Tensor, Any]:
        prepared_inputs = self.prepare_inputs(items)
        tokens = prepared_inputs['tokens']
        is_state = ((tokens == self.dataset.tokenizer.bos_token_id).float(
        ) + (tokens == self.dataset.tokenizer.eoa_token_id).float()) > 0
        is_action = ((tokens == self.dataset.tokenizer.boa_token_id).float(
        ) + (tokens == self.dataset.tokenizer.eos_token_id).float()) > 0
        state_points = torch.where(is_state, torch.arange(tokens.shape[1]).unsqueeze(
            0).repeat(tokens.shape[0], 1).to(self.device), -1)
        action_points = torch.where(is_action, torch.arange(tokens.shape[1]).unsqueeze(
            0).repeat(tokens.shape[0], 1).to(self.device), -1)
        action_mask = (action_points.argmax(dim=1) >=
                       state_points.argmax(dim=1)).float()
        scores, model_outputs = self.score(tokens, None, None, None,
                                           qv_kwargs={'use_cache': True},
                                           policy_kwargs={'use_cache': True},
                                           target_kwargs={'use_cache': True},
                                           beta=beta, exp_weights=exp_weights,
                                           clip_weight=clip_weight,
                                           logit_temp=logit_temp, logit_top_k=logit_top_k,
                                           logit_top_p=logit_top_p, include_logits=include_logits,
                                           include_advantage=include_advantage, action_mask=action_mask)
        return scores[:, -1, :], (
            model_outputs['qv_model_outputs'].past_key_values,
            model_outputs['policy_model_outputs'].past_key_values,
            model_outputs['target_model_outputs'].past_key_values,
            action_mask,
        )

    def next_score(self,
                   tokens: torch.Tensor,
                   state: Any,
                   beta: float = 1.0,
                   exp_weights: bool = False,
                   clip_weight: Optional[float] = None,
                   logit_temp: float = 1.0,
                   logit_top_k: Optional[int] = None,
                   logit_top_p: Optional[float] = None,
                   include_logits: bool = False,
                   include_advantage: bool = True) -> Tuple[torch.Tensor, Any]:
        qv_kvs, policy_kvs, target_kvs, action_mask = state
        action_mask *= (tokens != self.dataset.tokenizer.eoa_token_id).float()
        action_mask += (tokens == self.dataset.tokenizer.eos_token_id).float()
        action_mask = (action_mask > 0.0).float()
        scores, model_outputs = self.score(tokens.unsqueeze(1), None, None, None,
                                           qv_kwargs={'use_cache': True,
                                                      'past_key_values': qv_kvs},
                                           policy_kwargs={'use_cache': True,
                                                          'past_key_values': policy_kvs},
                                           target_kwargs={'use_cache': True,
                                                          'past_key_values': target_kvs},
                                           beta=beta, exp_weights=exp_weights, clip_weight=clip_weight,
                                           logit_temp=logit_temp, logit_top_k=logit_top_k,
                                           logit_top_p=logit_top_p, include_logits=include_logits,
                                           include_advantage=include_advantage, action_mask=action_mask)
        return scores.squeeze(1), (
            model_outputs['qv_model_outputs'].past_key_values,
            model_outputs['policy_model_outputs'].past_key_values,
            model_outputs['target_model_outputs'].past_key_values,
            action_mask,
        )

    def softUpdate(self):
        """Soft updates target networks.
        """
        for target_param, local_param in zip(
                self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(
                self.alpha * local_param.data + (1.0 - self.alpha) * target_param.data)
        if self.double_q:
            for target_param, local_param in zip(
                    self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(
                    self.alpha * local_param.data + (1.0 - self.alpha) * target_param.data)
        if self.lm_target is not None:
            for target_param, local_param in zip(
                    self.actor_target.parameters(), self.model.parameters()):
                target_param.data.copy_(
                    self.alpha * local_param.data + (1.0 - self.alpha) * target_param.data)

    def hardUpdate(self):
        """Hard updates target networks.
        """
        for target_param, local_param in zip(
                self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(local_param.data)
        if self.double_q:
            for target_param, local_param in zip(
                    self.target_q2.parameters(), self.q2.parameters()):
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

        clone = type(self)(dataset=self.dataset,
                           net_config=self.net_config,
                           index=index,
                           batch_size=self.batch_size,
                           lr=self.lr,
                           learn_step=self.learn_step,
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

        clone.model = self.model.clone().to(self.device)
        clone.actor = self.actor.clone().to(self.device)
        clone.actor_target = self.actor_target.clone().to(self.device)
        clone.optimizer = optim.Adam(clone.actor.parameters(
        ), lr=clone.lr, weight_decay=clone.weight_decay)
        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def saveCheckpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        torch.save({
            'model_init_dict': self.model.init_dict,
            'model_state_dict': self.model.state_dict(),
            'actor_init_dict': self.actor.init_dict,
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_init_dict': self.actor_target.init_dict,
            'actor_target_state_dict': self.actor_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dataset': self.dataset,
            'net_config': self.net_config,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'learn_step': self.learn_step,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'tau': self.tau,
            'mutation': self.mut,
            'transition_weight': self.transition_weight,
            'clip_weight': self.clip_weight,
            'value_max': self.value_max,
            'value_min': self.value_min,
            'detach_v': self.detach_v,
            'detach_q': self.detach_q,
            'detach_pi': self.detach_pi,
            'double_q': self.double_q,
            'per_token': self.per_token,
            'exp_weights': self.exp_weights,
            'dm_margin': self.dm_margin,
            'cql_temp': self.cql_temp,
            'weight_decay': self.weight_decay,
            'index': self.index,
            'scores': self.scores,
            'fitness': self.fitness,
            'steps': self.steps,
        }, path)

    def loadCheckpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path)
        self.net_config = checkpoint['net_config']
        self.model = EvolvableGPT(**checkpoint['model_init_dict'])
        self.actor = EvolvableGPT(**checkpoint['actor_init_dict'])
        self.actor_target = EvolvableGPT(
            **checkpoint['actor_target_init_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(
            checkpoint['actor_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.lr = checkpoint['lr']
        self.learn_step = checkpoint['learn_step']
        self.alpha = checkpoint['alpha']
        self.beta = checkpoint['beta']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.mut = checkpoint['mutation']
        self.transition_weight = checkpoint['transition_weight']
        self.clip_weight = checkpoint['clip_weight']
        self.value_max = checkpoint['value_max']
        self.value_min = checkpoint['value_min']
        self.detach_v = checkpoint['detach_v']
        self.detach_q = checkpoint['detach_q']
        self.detach_pi = checkpoint['detach_pi']
        self.double_q = checkpoint['double_q']
        self.per_token = checkpoint['per_token']
        self.exp_weights = checkpoint['exp_weights']
        self.dm_margin = checkpoint['dm_margin']
        self.cql_temp = checkpoint['cql_temp']
        self.weight_decay = checkpoint['weight_decay']
        self.index = checkpoint['index']
        self.scores = checkpoint['scores']
        self.fitness = checkpoint['fitness']
        self.steps = checkpoint['steps']


def map_pytree(f: Callable[[Union[np.ndarray, torch.Tensor]], Any],
               item: Any):
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


def get_transformer_logs(
        attentions: List[torch.Tensor], model: nn.Module, attn_mask: torch.Tensor):
    logs = {}
    n = attn_mask.sum()
    model_attention_entropy = -sum(map(lambda x: ((x * torch.log(x + 1e-7)).sum(
        dim=-1) * attn_mask.unsqueeze(1)).sum().item(), attentions)) / (len(attentions) * n)
    model_parameter_norm = parameter_norm(model)
    logs['attention_entropy'] = (model_attention_entropy, n * len(attentions))
    logs['parameter_norm'] = (model_parameter_norm, 1)
    return logs


def top_k_logits(logits, k):
    # logits = (batch, time, dim)
    _, bottom_k_idx = torch.topk(-logits, logits.shape[2] - k, dim=2)
    return torch.scatter(logits, dim=2, index=bottom_k_idx, value=float('-inf'))


def top_p_logits(logits, p):
    # logits = (batch, time, dim)
    sorted_logits, _ = torch.sort(logits, dim=2, descending=True)
    num_to_take = torch.sum(torch.cumsum(
        F.softmax(sorted_logits, dim=2), dim=2) <= p, dim=2).unsqueeze(2)
    mask = logits < torch.gather(sorted_logits, dim=2, index=torch.clamp(
        num_to_take, max=logits.shape[2] - 1))
    return logits.masked_fill(mask, float('-inf'))


def process_logits(logits, temp=1.0, top_k=None, top_p=None):
    logits /= temp
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    return logits
