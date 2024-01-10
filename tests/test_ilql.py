from agilerl.algorithms.ilql import ILQL
from agilerl.data.language_environment import Language_Observation
from agilerl.data.rl_data import ConstantTokenReward, List_RL_Dataset
from agilerl.data.torch_datasets import GeneralDataset
from tests.test_data import WordleTokenizer


def test_ilql_init():
    List_RL_Dataset.__abstractmethods__ = set()
    GeneralDataset.__abstractmethods__ = set()

    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("test", 1), ("test", None)], True

    # lang_obs = LangObs()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)

    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)

    algo = ILQL(rl_ds)

    assert algo.algo == "ILQL"
    assert algo.index == 0
    assert algo.batch_size == 64
    assert algo.lr == 1e-5
    assert algo.alpha == 0.005
    assert algo.beta == 0.0
    assert algo.gamma == 0.99
    assert algo.tau == 0.6
    assert algo.mutation is None
    assert algo.transition_weight == 0.0
    assert algo.clip_weight is None
    assert algo.value_max is None
    assert algo.value_min is None
    assert algo.detach_v is False
    assert algo.detach_q is False
    assert algo.detach_pi is False
    assert algo.double_q is True
    assert algo.per_token is True
    assert algo.exp_weights is True
    assert algo.dm_margin == 0.0
    assert algo.cql_temp == 1.0
    assert algo.weight_decay == 0.0
    assert algo.device == "cpu"
