import re

from agilerl.data.language_environment import (
    Language_Environment,
    Language_Observation,
    Policy,
    interact_environment,
)
from agilerl.data.rl_data import (
    ConstantTokenReward,
    DataPoint,
    Iterable_RL_Dataset,
    List_RL_Dataset,
    SpecifiedTokenReward,
    TokenReward,
)
from agilerl.data.tokenizer import Tokenizer
from agilerl.data.torch_datasets import GeneralDataset, GeneralIterDataset
from agilerl.utils.cache import Cache


class WordleTokenizer(Tokenizer):
    def __init__(self):
        self.special_vocab = [
            "<g>",
            "<b>",
            "<y>",
            "<|pad|>",
            "</a>",
            "</s>",
            "<s>",
            "<a>",
            "</eod>",
        ]
        self.vocab = list("abcdefghijklmnopqrstuvwxyz") + self.special_vocab
        self.t2i = {w: i for i, w in enumerate(self.vocab)}
        super().__init__(
            self.token_to_id("<|pad|>"),
            self.token_to_id("</s>"),
            self.token_to_id("</a>"),
            self.token_to_id("<s>"),
            self.token_to_id("<a>"),
            self.token_to_id("</eod>"),
        )

    def encode(self, str_, **kwargs):
        if isinstance(str_, str):
            special_idxs = []
            for special_char in self.special_vocab:
                special_idxs += list(
                    map(
                        lambda x: (x.start(), x.end(), self.token_to_id(special_char)),
                        re.finditer(re.escape(special_char), str_),
                    )
                )
            special_idxs.sort(key=lambda x: x[0])
            tokens = []
            curr = 0
            for s, e, tok in special_idxs:
                tokens.extend([self.token_to_id(c) for c in str_[curr:s]])
                tokens.append(tok)
                curr = e
            tokens.extend([self.token_to_id(c) for c in str_[curr:]])
            return tokens, [int(t != self.pad_token_id) for t in tokens]
        elif isinstance(str_, list):
            tokens, pads = zip(*[self.encode(item) for item in str_])
            max_len = max(map(len, tokens))
            return [
                list(item) + ([self.pad_token_id] * (max_len - len(item)))
                for item in tokens
            ], [list(item) + ([0] * (max_len - len(item))) for item in pads]
        else:
            raise ValueError("str_ must be a string or a list of strings")

    def decode(self, tokens, **kwargs):
        if len(tokens) == 0:
            return ""
        if not isinstance(tokens[0], list):
            return "".join([self.id_to_token(item) for item in tokens])
        elif isinstance(tokens[0], list):
            return [self.decode(item) for item in tokens]
        else:
            raise ValueError("tokens must be a list of ints or a list of lists of ints")

    def num_tokens(self):
        return len(self.vocab)

    def id_to_token(self, id_):
        return self.vocab[id_]

    def token_to_id(self, token):
        return self.t2i[token]

    def get_vocab(self):
        return self.vocab


def test_language_observation_base_class():
    Language_Observation.__abstractmethods__ = set()
    lang_obs = Language_Observation()
    lang_obs.to_sequence()
    lang_obs.__str__()
    metadata = lang_obs.metadata()

    assert metadata is None


def test_language_environment_base_class():
    Language_Environment.__abstractmethods__ = set()

    lang_env = Language_Environment()
    lang_env.step(None)
    lang_env.reset()
    lang_env.is_terminal()

    assert True


def test_policy_base_class():
    Policy.__abstractmethods__ = set()

    policy = Policy()
    policy.act(None)
    policy.train()
    policy.eval()

    assert isinstance(policy.cache, Cache)


def test_interact_environment():
    class LangEnv(Language_Environment):
        def __init__(self):
            super().__init__()
            self.term = False

        def step(self, action):
            self.term = True
            return 1, 1, 1

        def reset(self):
            self.term = False
            return 0

        def is_terminal(self):
            return self.term

    class Pol(Policy):
        def act(self, obs):
            return 1

        def train(self):
            return 0

        def eval(self):
            return False

    env = LangEnv()
    policy = Pol()
    obs = None

    obs, obs_sequence = interact_environment(env, policy, obs)

    assert isinstance(obs, int)
    assert isinstance(obs_sequence, list)
    assert len(obs_sequence) == 2


def test_tokenreward_base_class():
    TokenReward.__abstractmethods__ = set()

    tok_rew = TokenReward()
    tok_rew.get_token_reward(None)

    assert True


def test_constanttokenreward_base_class():
    tokens = [1, 2, 3]
    tok_rew = ConstantTokenReward(1)
    reward = tok_rew.get_token_reward(tokens)

    assert tok_rew.c == 1
    assert reward == [1, 1]


def test_specifiedtokenreward_base_class():
    token_data = {0: 10.0}
    scale = 1.0
    shift = 2.0
    tokens = [0, 0, 1, 2]
    tok_rew = SpecifiedTokenReward(token_data, scale, shift)
    reward = tok_rew.get_token_reward(tokens)

    assert tok_rew.token_data == token_data
    assert tok_rew.scale == scale
    assert tok_rew.shift == shift
    assert reward == [12.0, 2.0, 2.0]


def test_list_rl_dataset_base_class():
    List_RL_Dataset.__abstractmethods__ = set()
    GeneralDataset.__abstractmethods__ = set()

    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("test", 1), ("test", None)], True

    lang_obs = LangObs()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    meta = None

    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    rl_ds.get_item(0)
    rl_ds.size()

    dp = DataPoint.from_obs(lang_obs, tokenizer, token_reward, meta)
    _ = DataPoint.from_obs(lang_obs, tokenizer, token_reward, 1)

    reward = dp.get_token_reward(lang_obs, tokenizer, token_reward)

    ds = GeneralDataset(rl_ds, None)
    len_ds = ds.__len__()
    item_ds = ds.__getitem__(0)
    collate_ds = ds.collate([dp])
    collate_simple_ds = ds.collate_simple(None)

    assert isinstance(reward, list)
    assert len_ds is None
    assert item_ds is None
    assert isinstance(collate_ds, dict)
    assert collate_simple_ds is None


def test_dp_from_obs():
    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("test", None)], True

        def metadata(self):
            return {"test": 1}

    lang_obs = LangObs()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    meta = {"test": 1}

    dp = DataPoint.from_obs(lang_obs, tokenizer, token_reward, meta)

    assert isinstance(dp, DataPoint)

    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("test", None)], True

        def metadata(self):
            return {"test": 1}

    lang_obs = LangObs()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    meta = None

    dp = DataPoint.from_obs(lang_obs, tokenizer, token_reward, meta)

    assert isinstance(dp, DataPoint)


def test_iterable_rl_dataset_base_class():
    Iterable_RL_Dataset.__abstractmethods__ = set()
    GeneralIterDataset.__abstractmethods__ = set()

    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("test", 1), ("test", 2), ("test", None)], True

    lang_obs = LangObs()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    meta = None

    iter_rl_ds = Iterable_RL_Dataset(tokenizer, token_reward, 10)
    iter_rl_ds.sample_item()

    dp = DataPoint.from_obs(lang_obs, tokenizer, token_reward, meta)
    _ = DataPoint.from_obs(lang_obs, tokenizer, token_reward, 1)

    reward = dp.get_token_reward(lang_obs, tokenizer, token_reward)

    ds = GeneralIterDataset(iter_rl_ds, None)
    iter_ds = ds.__iter__()
    next_ds = ds.__next__()
    collate_ds = ds.collate([dp])
    collate_simple_ds = ds.collate_simple(None)

    assert isinstance(reward, list)
    assert iter_ds == ds
    assert next_ds is None
    assert isinstance(collate_ds, dict)
    assert collate_simple_ds is None


def test_tokenizer_base_class():
    Tokenizer.__abstractmethods__ = set()

    tok = Tokenizer(0, 1, 2, 3, 4, 5)
    tok.encode("test")
    tok.decode([0, 1, 2])
    tok.num_tokens()
    tok.id_to_token(0)
    tok.token_to_id("a")
    tok.get_vocab()

    assert tok.pad_token_id == 0
    assert tok.eos_token_id == 1
    assert tok.eoa_token_id == 2
    assert tok.bos_token_id == 3
    assert tok.boa_token_id == 4
    assert tok.eod_token_id == 5
