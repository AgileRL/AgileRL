from typing import List, Optional, Tuple

from agilerl.data.language_environment import Language_Environment, Language_Observation
from agilerl.wordle.wordle_game import Vocabulary, WordleGame


class WordleObservation(Language_Observation):
    def __init__(self, game: WordleGame):
        self.game = game

    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        return self.game.transition_sequence()

    def __str__(self) -> str:
        return str(self.game)


class WordleEnvironment(Language_Environment):
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.state = self.reset()

    def step(self, action: str) -> Tuple[WordleObservation, float, bool]:
        wordle_game, r, t = self.state.game.next(action)
        self.state = WordleObservation(wordle_game)
        return self.state, r, t

    def reset(self) -> WordleObservation:
        self.state = WordleObservation(WordleGame.initialize(self.vocab))
        return self.state

    def is_terminal(self) -> bool:
        return self.state.game.is_terminal()
