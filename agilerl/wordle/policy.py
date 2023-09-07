import math
import random
from collections import defaultdict
from typing import List, Optional, Union

from tqdm.auto import tqdm

from agilerl.data.language_environment import Policy
from agilerl.utils.cache import Cache
from agilerl.wordle.wordle_env import WordleObservation
from agilerl.wordle.wordle_game import Vocabulary


class UserPolicy(Policy):
    def __init__(
        self, hint_policy: Optional[Policy], vocab: Optional[Union[str, Vocabulary]]
    ):
        super().__init__()
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
        self.hint_policy = hint_policy

    def act(self, obs: WordleObservation) -> str:
        print(obs.game)
        while True:
            if self.hint_policy is not None:
                want_a_hint = input("hint? ")
                if want_a_hint.lower() == "y" or want_a_hint.lower() == "yes":
                    result = self.hint_policy.act(obs)
                    print()
                    return result
            result = input("Enter a word: ")
            if len(result) != 5:
                print("Please enter a 5 letter word.")
            elif (self.vocab is not None) and (result not in self.vocab.all_vocab):
                print("Not a word. Try again.")
            else:
                break
        print()
        return result


class StartWordPolicy(Policy):
    def __init__(self, start_words: Optional[List[str]] = None):
        super().__init__()
        self.start_words = start_words
        if self.start_words is None:
            # "tales" is the optimal word under 10k_words.txt
            # "raise" is the optimal word under wordle_official.txt
            self.start_words = [
                "opera",
                "tears",
                "soare",
                "roate",
                "raise",
                "arose",
                "earls",
                "laser",
                "reals",
                "aloes",
                "reais",
                "slate",
                "sauce",
                "slice",
                "shale",
                "saute",
                "share",
                "sooty",
                "shine",
                "suite",
                "crane",
                "adieu",
                "audio",
                "stare",
                "roast",
                "ratio",
                "arise",
                "tales",
            ]

    def act(self, obs: WordleObservation) -> str:
        filtered_start_words = list(
            filter(lambda x: x in obs.game.vocab.filtered_vocab, self.start_words)
        )
        if len(filtered_start_words) == 0:
            filtered_start_words = obs.game.vocab.filtered_vocab
        return random.choice(filtered_start_words)


class OptimalPolicy(Policy):
    def __init__(
        self, start_word_policy: Optional[Policy] = None, progress_bar: bool = False
    ):
        super().__init__()
        self.start_word_policy = start_word_policy
        self.progress_bar = progress_bar
        self.cache = Cache()

    def act(self, obs: WordleObservation) -> str:
        if obs.game.state in self.cache:
            return random.choice(self.cache[obs.game.state])
        if len(obs.game.action_history) == 0 and self.start_word_policy is not None:
            return self.start_word_policy.act(obs)
        best_words = []
        best_info = float("-inf")
        for word in (
            tqdm(obs.game.vocab.filtered_vocab)
            if self.progress_bar
            else obs.game.vocab.filtered_vocab
        ):
            total_entropy = 0.0
            total = 0
            for next_state, state_count in obs.game.all_next(word):
                total_entropy += (
                    math.log(next_state.vocab.filtered_vocab_size()) * state_count
                )
                total += state_count
            info_gain = math.log(obs.game.vocab.filtered_vocab_size()) - (
                total_entropy / total
            )
            if info_gain > best_info:
                best_words, best_info = [word], info_gain
            elif info_gain == best_info:
                best_words.append(word)
        self.cache[obs.game.state] = best_words
        return random.choice(best_words)


class RepeatPolicy(Policy):
    def __init__(self, start_word_policy: Optional[Policy], first_n: Optional[int]):
        super().__init__()
        self.first_n = first_n
        self.start_word_policy = start_word_policy

    def act(self, obs: WordleObservation) -> str:
        if len(obs.game.action_history) == 0:
            if self.start_word_policy is not None:
                return self.start_word_policy.act(obs)
            return obs.game.vocab.get_random_word_all()
        if self.first_n is None:
            return random.choice(obs.game.action_history)
        return random.choice(obs.game.action_history[: self.first_n])


class RandomMixturePolicy(Policy):
    def __init__(self, prob_smart: float, vocab: Optional[Union[str, Vocabulary]]):
        super().__init__()
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
        self.prob_smart = prob_smart

    def act(self, obs: WordleObservation) -> str:
        if self.vocab is None:
            v = obs.game.vocab
        else:
            v = self.vocab
        if random.random() < self.prob_smart:
            if self.vocab is not None:
                v = v.update_vocab(obs.game.state)
            return v.get_random_word_filtered()
        return v.get_random_word_all()


class WrongPolicy(Policy):
    def __init__(self, vocab: Union[str, Vocabulary]):
        super().__init__()
        self.vocab = vocab
        if isinstance(self.vocab, str):
            self.vocab = Vocabulary.from_file(self.vocab)
        self.choices = set(self.vocab.all_vocab)

    def act(self, obs: WordleObservation) -> str:
        bad_options = self.choices.difference(obs.game.vocab.filtered_vocab)
        if len(bad_options) == 0:
            return self.vocab.get_random_word_all()
        return random.sample(bad_options, 1)[0]


class MixturePolicy(Policy):
    def __init__(self, prob1: float, policy1: Policy, policy2: Policy):
        super().__init__()
        self.prob1 = prob1
        self.policy1 = policy1
        self.policy2 = policy2

    def act(self, obs: WordleObservation) -> str:
        if random.random() < self.prob1:
            return self.policy1.act(obs)
        return self.policy2.act(obs)


class MonteCarloPolicy(Policy):
    def __init__(self, n_samples: int, sample_policy: Policy):
        super().__init__()
        self.n_samples = n_samples
        self.sample_policy = sample_policy

    def act(self, obs: WordleObservation) -> str:
        action_scores = defaultdict(list)
        for _ in range(self.n_samples):
            curr_obs = obs
            total_reward = 0
            while not curr_obs.game.is_terminal():
                word_choice = self.sample_policy.act(curr_obs)
                curr_obs, r, _ = curr_obs.game.next(word_choice)
                total_reward += r
            action_scores[curr_obs.action_history[len(obs.game.action_history)]].append(
                total_reward
            )
        return max(action_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))[0]
