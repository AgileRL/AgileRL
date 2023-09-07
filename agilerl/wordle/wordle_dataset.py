import json
import pickle as pkl
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from agilerl.data.language_environment import Policy, interact_environment
from agilerl.data.rl_data import (
    ConstantTokenReward,
    DataPoint,
    Iterable_RL_Dataset,
    List_RL_Dataset,
    TokenReward,
)
from agilerl.utils.ilql_utils import convert_path
from agilerl.wordle.wordle_env import WordleEnvironment, WordleObservation
from agilerl.wordle.wordle_game import Vocabulary, WordleGame, WordleState
from agilerl.wordle.wordle_tokenizer import WordleTokenizer


class WordleListDataset(List_RL_Dataset):
    def __init__(
        self,
        items: List[Tuple[WordleObservation, Optional[Dict[str, Any]]]],
        max_len: Optional[int],
        token_reward: TokenReward,
    ) -> None:
        tokenizer = WordleTokenizer()
        super().__init__(tokenizer, token_reward, max_len)
        self.items = items

    def get_item(self, idx: int):
        return DataPoint.from_obs(
            self.items[idx][0], self.tokenizer, self.token_reward, self.items[idx][1]
        )

    def size(self):
        return len(self.items)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        max_len: Optional[int],
        vocab: Optional[Vocabulary],
        token_reward: TokenReward,
    ):
        with open(file_path, "rb") as f:
            d = pkl.load(f)
        if vocab is None:
            vocab = Vocabulary.from_file(convert_path(d["vocab_path"]))
            if d["vocab_cache_path"] is not None:
                vocab.cache.load(convert_path(d["vocab_cache_path"]))
        wordle_items = [
            WordleObservation(
                WordleGame(
                    item["state"], vocab.update_vocab(item["state"]), item["actions"]
                )
            )
            for item in tqdm(d["state_actions"])
        ]
        meta = [
            {**item["meta"], "self": wordle_items[i]}
            if "meta" in item
            else {"self": wordle_items[i]}
            for i, item in enumerate(d["state_actions"])
        ]
        return WordleListDataset(list(zip(wordle_items, meta)), max_len, token_reward)


class WordleIterableDataset(Iterable_RL_Dataset):
    def __init__(
        self,
        policy: Policy,
        vocab: Vocabulary,
        max_len: Optional[int],
        token_reward: TokenReward,
    ) -> None:
        tokenizer = WordleTokenizer()
        super().__init__(tokenizer, token_reward, max_len)
        self.policy = policy
        self.env = WordleEnvironment(vocab)

    def sample_item(self):
        return DataPoint.from_obs(
            interact_environment(self.env, self.policy, None)[0],
            self.tokenizer,
            self.token_reward,
            None,
        )


class WordleHumanDataset(Iterable_RL_Dataset):
    def __init__(
        self,
        games: List[Tuple[str, List[str]]],
        transitions: Dict[str, Dict[str, List[str]]],
        use_true_word: bool,
        max_len: Optional[int],
        token_reward: TokenReward,
        game_indexes: Optional[List[int]],
        top_p: Optional[float],
    ) -> None:
        tokenizer = WordleTokenizer()
        super().__init__(tokenizer, token_reward, max_len)
        self.games = games
        if game_indexes is not None:
            self.games = [self.games[idx] for idx in game_indexes]
        if top_p is not None:
            lens = [len(game) for _, game in self.games]
            self.games = [
                self.games[idx] for idx in np.argsort(lens)[: int(len(lens) * top_p)]
            ]
        self.transitions = transitions
        self.use_true_word = use_true_word

    def sample_item(self):
        true_word, game = random.choice(self.games)
        if self.use_true_word:
            while True:
                actions = []
                for transition in game:
                    if (
                        transition not in self.transitions[true_word]
                        or len(self.transitions[true_word][transition]) == 0
                    ):
                        break
                    actions.append(
                        random.choice(self.transitions[true_word][transition])
                    )
                if len(actions) == len(game):
                    break
                else:
                    true_word, game = random.choice(self.games)
        else:
            word_choices = list(self.transitions.keys())
            while True:
                true_word = random.choice(word_choices)
                actions = []
                for transition in game:
                    if (
                        transition not in self.transitions[true_word]
                        or len(self.transitions[true_word][transition]) == 0
                    ):
                        break
                    actions.append(
                        random.choice(self.transitions[true_word][transition])
                    )
                if len(actions) == len(game):
                    break
                else:
                    true_word, game = random.choice(self.games)
        state = WordleState.initial_state()
        for action in actions:
            state = state.transition_state(action, true_word)
        vocab = Vocabulary([true_word], state, cache=None, fill_cache=False)
        obs = WordleObservation(WordleGame(state, vocab, actions))
        return DataPoint.from_obs(obs, self.tokenizer, self.token_reward, {"obs": obs})

    @classmethod
    def from_file(
        cls,
        file_path: str,
        use_true_word: bool = False,
        max_len: Optional[int] = None,
        token_reward: Optional[TokenReward] = None,
        game_indexes: Optional[List[int]] = None,
        top_p: Optional[float] = None,
    ):
        if token_reward is None:
            token_reward = ConstantTokenReward(0.0)
        with open(file_path) as f:
            d = json.load(f)
        return WordleHumanDataset(
            d["games"],
            d["transitions"],
            use_true_word,
            max_len,
            token_reward,
            game_indexes,
            top_p,
        )
