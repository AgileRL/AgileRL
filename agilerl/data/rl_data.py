from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from agilerl.data.language_environment import Language_Observation
from agilerl.data.tokenizer import Tokenizer


class TokenReward(ABC):
    @abstractmethod
    def get_token_reward(self, tokens: List[int]) -> List[float]:
        pass


class ConstantTokenReward(TokenReward):
    def __init__(self, c: float = 0.0):
        self.c = c

    def get_token_reward(self, tokens: List[int]) -> List[float]:
        return [self.c] * (len(tokens) - 1)


class SpecifiedTokenReward(TokenReward):
    def __init__(
        self, token_data: Dict[int, float], scale: float = 1.0, shift: float = 0.0
    ):
        self.token_data = token_data
        self.scale = scale
        self.shift = shift

    def get_token_reward(self, tokens: List[int]) -> List[float]:
        return [
            (
                (self.token_data[tok] * self.scale + self.shift)
                if tok in self.token_data
                else (0.0 * self.scale + self.shift)
            )
            for tok in tokens[1:]
        ]


@dataclass
class DataPoint:
    raw_str: str
    tokens: list[int]
    state_idxs: list[int]
    action_idxs: list[int]
    rewards: list[float]
    terminals: list[int]
    utterance_state_idxs: list[int]
    utterance_action_idxs: list[int]
    utterance_rewards: list[float]
    utterance_terminals: list[int]
    meta: Optional[Dict[str, Any]] = None

    def to_tensors(self, device, max_length: Optional[int]):
        tok = torch.tensor(self.tokens).to(device)
        s = torch.tensor(self.state_idxs).long().to(device)
        a = torch.tensor(self.action_idxs).long().to(device)
        r = torch.tensor(self.rewards).to(device)
        term = torch.tensor(self.terminals).to(device)
        u_s = torch.tensor(self.utterance_state_idxs).long().to(device)
        u_a = torch.tensor(self.utterance_action_idxs).long().to(device)
        u_r = torch.tensor(self.utterance_rewards).long().to(device)
        u_term = torch.tensor(self.utterance_terminals).long().to(device)
        if max_length is not None:
            tok = tok[:max_length]
            s = s[: (s < max_length).sum()]
            a = a[: max(min((a < (max_length - 1)).sum().item(), s.shape[0] - 1), 0)]
            r = r[: a.shape[0]]
            term = term[: s.shape[0]]
            u_s = u_s[: (s < max_length).sum()]
            u_a = u_a[
                : max(min((u_a < (max_length - 1)).sum().item(), u_s.shape[0] - 1), 0)
            ]
            u_r = u_r[: u_a.shape[0]]
            u_term = u_term[: u_s.shape[0]]
        return tok, s, a, r, term, u_s, u_a, u_r, u_term

    @classmethod
    def from_obs(
        cls,
        obs: Language_Observation,
        tokenizer: Tokenizer,
        token_reward: TokenReward,
        meta: Optional[Dict[str, Any]] = None,
    ):
        sequence, terminal = obs.to_sequence()
        obs_meta = obs.metadata()
        if meta is not None and obs_meta is not None:
            meta = {**obs_meta, **meta}
        elif obs_meta is not None:
            meta = obs_meta
        if len(sequence) == 0 or sequence[0][1] is not None:
            raw_str = tokenizer.id_to_token(tokenizer.boa_token_id)
        else:
            raw_str = tokenizer.id_to_token(tokenizer.bos_token_id)
        action_rewards = []
        for s, r in sequence:
            raw_str += s
            if r is None:
                raw_str += tokenizer.id_to_token(tokenizer.eos_token_id)
            else:
                raw_str += tokenizer.id_to_token(tokenizer.eoa_token_id)
                action_rewards.append(r)
        if terminal:
            raw_str += tokenizer.id_to_token(tokenizer.eod_token_id)
        tokens = tokenizer.encode(raw_str)[0]
        token_rewards = token_reward.get_token_reward(tokens)
        state_idxs = []
        action_idxs = []
        reward = []
        utterance_state_idxs = []
        utterance_action_idxs = []
        utterance_rewards = []
        curr_idx = 0
        curr_action_idx = 0
        for i, t in enumerate(tokens):
            if t == tokenizer.eos_token_id:
                curr_idx = i
            elif t == tokenizer.eoa_token_id:
                action_idxs.extend(list(range(curr_idx, i)))
                state_idxs.extend(list(range(curr_idx, i)))
                reward.extend([token_rewards[x] for x in range(curr_idx, i)])
                reward[-1] += action_rewards[curr_action_idx]
                utterance_action_idxs.append(i)
                utterance_state_idxs.append(curr_idx)
                utterance_rewards.append(
                    action_rewards[curr_action_idx]
                    + sum([token_rewards[x] for x in range(curr_idx, i)])
                )
                curr_idx = i
                curr_action_idx += 1
        state_idxs.append(len(tokens) - 1)
        utterance_state_idxs.append(len(tokens) - 1)
        terminals = ([0] * (len(state_idxs) - 1)) + [int(terminal)]
        utterance_terminals = ([0] * (len(utterance_state_idxs) - 1)) + [int(terminal)]
        return cls(
            raw_str,
            tokens,
            state_idxs,
            action_idxs,
            reward,
            terminals,
            utterance_state_idxs,
            utterance_action_idxs,
            utterance_rewards,
            utterance_terminals,
            meta=meta,
        )

    @staticmethod
    def get_token_reward(
        obs: Language_Observation, tokenizer: Tokenizer, token_reward: TokenReward
    ):
        return DataPoint.from_obs(obs, tokenizer, token_reward).rewards


class RL_Dataset(ABC):
    def __init__(
        self, tokenizer: Tokenizer, token_reward: TokenReward, max_len: Optional[int]
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.token_reward = token_reward
        self.max_len = max_len

    def collate(self, items: List[DataPoint], device):
        (
            tokens,
            state_idxs,
            action_idxs,
            rewards,
            terminals,
            u_state_idxs,
            u_action_idxs,
            u_rewards,
            u_terminals,
        ) = zip(*map(lambda x: x.to_tensors(device, self.max_len), items))
        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attn_mask = (tokens != self.tokenizer.pad_token_id).float()
        state_idxs = torch.nn.utils.rnn.pad_sequence(
            state_idxs, batch_first=True, padding_value=0
        )
        action_idxs = torch.nn.utils.rnn.pad_sequence(
            action_idxs, batch_first=True, padding_value=0
        )
        terminals = torch.nn.utils.rnn.pad_sequence(
            terminals, batch_first=True, padding_value=1
        )
        rewards = torch.nn.utils.rnn.pad_sequence(
            rewards, batch_first=True, padding_value=0.0
        )
        u_state_idxs = torch.nn.utils.rnn.pad_sequence(
            u_state_idxs, batch_first=True, padding_value=0
        )
        u_action_idxs = torch.nn.utils.rnn.pad_sequence(
            u_action_idxs, batch_first=True, padding_value=0
        )
        u_terminals = torch.nn.utils.rnn.pad_sequence(
            u_terminals, batch_first=True, padding_value=1
        )
        u_rewards = torch.nn.utils.rnn.pad_sequence(
            u_rewards, batch_first=True, padding_value=0.0
        )
        return {
            "tokens": tokens,
            "attn_mask": attn_mask,
            "state_idxs": state_idxs,
            "action_idxs": action_idxs,
            "rewards": rewards,
            "terminals": terminals,
            "u_state_idxs": u_state_idxs,
            "u_action_idxs": u_action_idxs,
            "u_rewards": u_rewards,
            "u_terminals": u_terminals,
        }


class List_RL_Dataset(RL_Dataset):
    @abstractmethod
    def get_item(self, idx: int) -> DataPoint:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


class Iterable_RL_Dataset(RL_Dataset):
    @abstractmethod
    def sample_item(self) -> DataPoint:
        pass
