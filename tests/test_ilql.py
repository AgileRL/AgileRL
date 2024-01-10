import torch

from agilerl.algorithms.ilql import ILQL
from agilerl.data.language_environment import Language_Observation
from agilerl.data.rl_data import ConstantTokenReward, DataPoint, List_RL_Dataset
from agilerl.data.torch_datasets import GeneralDataset
from tests.test_data import WordleTokenizer


def test_ilql_init():
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    net_config = {
        "arch": "gpt",
        "vocab_size": 12,
        "n_layer": 2,
        "n_embd": 12,
        "n_head": 2,
        "dim_feedfwd": 8,
        "block_size": 8,
        "activation": "GELU",
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "min_layers": 8,
        "max_layers": 16,
        "bias": True,
    }

    algo = ILQL(rl_ds, net_config=net_config)

    assert algo.algo == "ILQL"
    assert algo.index == 0
    assert algo.batch_size == 64
    assert algo.lr == 1e-5
    assert algo.alpha == 0.005
    assert algo.beta == 0.0
    assert algo.gamma == 0.99
    assert algo.tau == 0.6
    assert algo.mut is None
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


def test_forward():
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    net_config = {
        "arch": "gpt",
        "vocab_size": 12,
        "n_layer": 2,
        "n_embd": 12,
        "n_head": 2,
        "dim_feedfwd": 8,
        "block_size": 8,
        "activation": "GELU",
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "min_layers": 8,
        "max_layers": 16,
        "bias": True,
    }
    double_q = True

    algo = ILQL(
        rl_ds, net_config=net_config, double_q=double_q, value_min=0, value_max=1
    )

    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    state_idxs = torch.tensor([[0, 1, 2, 3, 4]])
    action_idxs = torch.tensor([[0, 1, 2, 3, 4]])

    outputs = algo(
        tokens=tokens,
        state_idxs=state_idxs,
        action_idxs=action_idxs,
    )

    assert outputs["vs"].shape == (1, 5)
    assert outputs["target_vs"].shape == (1, 5)
    assert outputs["logits"].shape == (1, 5, 35)

    prefix_embs = torch.empty((tokens.shape[0], 0, algo.net_config["n_embd"])).to(
        algo.device
    )

    prefix_attn_mask = torch.tensor([[1, 1, 1]]).bool()
    attn_mask = torch.tensor([[1, 1]]).bool()
    qv_kwargs = {"is_causal": False}
    target_kwargs = {"is_causal": False}
    policy_kwargs = {"is_causal": False}
    detach_full_policy = True
    remove_prefix_position_embs = True

    algo = ILQL(rl_ds, net_config=net_config, double_q=double_q)

    for skip_policy_on_train in [True, False]:
        outputs = algo(
            tokens=tokens,
            state_idxs=state_idxs,
            action_idxs=action_idxs,
            prefix_embs=prefix_embs,
            prefix_attn_mask=prefix_attn_mask,
            attn_mask=attn_mask,
            remove_prefix_position_embs=remove_prefix_position_embs,
            qv_kwargs=qv_kwargs,
            target_kwargs=target_kwargs,
            policy_kwargs=policy_kwargs,
            skip_policy_on_train=skip_policy_on_train,
            detach_full_policy=detach_full_policy,
        )

        assert outputs["vs"].shape == (1, 5)
        assert outputs["target_vs"].shape == (1, 5)
        assert outputs["logits"].shape == (1, 5, 35)


def test_get_loss():
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    net_config = {
        "arch": "gpt",
        "vocab_size": 12,
        "n_layer": 2,
        "n_embd": 12,
        "n_head": 2,
        "dim_feedfwd": 8,
        "block_size": 8,
        "activation": "GELU",
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "min_layers": 8,
        "max_layers": 16,
        "bias": True,
    }

    algo = ILQL(rl_ds, net_config=net_config, double_q=True)

    inputs = {
        "tokens": torch.tensor([[0, 1, 2, 3, 1]]),
        "state_idxs": torch.tensor([[0, 1, 2, 3, 1, 0]]),
        "action_idxs": torch.tensor([[0, 1, 2, 3, 1]]),
        "attn_mask": torch.tensor([[1, 1, 1, 1, 1]]).bool(),
        "rewards": torch.tensor([[0, 0, 0, 0, 0]]),
        "terminals": torch.tensor([[0, 0, 0, 0, 0, 1]]),
    }

    loss = algo.get_loss(inputs)
    assert isinstance(loss[1], dict)
    assert isinstance(loss[0].item(), float)

    algo = ILQL(
        rl_ds, net_config=net_config, double_q=False, exp_weights=False, clip_weight=0.1
    )

    inputs = {
        "tokens": torch.tensor([[0, 1, 2, 3, 1]]),
        "state_idxs": torch.tensor([[0, 1, 2, 3, 1, 0]]),
        "action_idxs": torch.tensor([[0, 1, 2, 3, 1]]),
        "attn_mask": torch.tensor([[1, 1, 1, 1, 1]]).bool(),
        "rewards": torch.tensor([[0, 0, 0, 0, 0]]),
        "terminals": torch.tensor([[0, 0, 0, 0, 0, 1]]),
    }

    loss = algo.get_loss(inputs, mc_returns=True)

    assert isinstance(loss[0].item(), float)
    assert isinstance(loss[1], dict)

    print(loss[1])

    loss_dict = {}
    loss_dict["token_loss"] = loss[1]["token_loss"][0]
    loss_dict["q_loss"] = loss[1]["q_loss"][0]
    loss_dict["v_loss"] = loss[1]["v_loss"][0]
    loss_dict["cql_loss"] = loss[1]["cql_loss"][0]
    loss_dict["dm_loss"] = loss[1]["dm_loss"][0]

    _ = loss[2][0](loss_dict)
    _ = loss[2][1](loss_dict)

    assert "loss" in loss_dict.keys()
    assert "advantage_hist" in loss_dict.keys()


def test_prepare_inputs():
    List_RL_Dataset.__abstractmethods__ = set()
    GeneralDataset.__abstractmethods__ = set()

    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    net_config = {
        "arch": "gpt",
        "vocab_size": 12,
        "n_layer": 2,
        "n_embd": 12,
        "n_head": 2,
        "dim_feedfwd": 8,
        "block_size": 8,
        "activation": "GELU",
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "min_layers": 8,
        "max_layers": 16,
        "bias": True,
    }

    algo = ILQL(rl_ds, net_config=net_config, double_q=True)

    inputs = {
        "tokens": torch.tensor([[0, 1, 2, 3, 1]]),
        "state_idxs": torch.tensor([[0, 1, 2, 3, 1, 0]]),
        "action_idxs": torch.tensor([[0, 1, 2, 3, 1]]),
        "attn_mask": torch.tensor([[1, 1, 1, 1, 1]]).bool(),
        "rewards": torch.tensor([[0, 0, 0, 0, 0]]),
        "terminals": torch.tensor([[0, 0, 0, 0, 0, 1]]),
    }

    items = algo.prepare_inputs(inputs)
    assert items == inputs
    assert isinstance(items, dict)

    Language_Observation.__abstractmethods__ = set()

    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("test", None)], True

        def metadata(self):
            return {"test": 1}

    lang_obs = LangObs()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    meta = {"test": 1}
    dps = [
        DataPoint.from_obs(lang_obs, tokenizer, token_reward, meta) for _ in range(5)
    ]

    items = algo.prepare_inputs(dps)
    assert isinstance(items, dict)
    assert "tokens" in items.keys()
    assert "state_idxs" in items.keys()
    assert "action_idxs" in items.keys()


def test_get_scores():
    List_RL_Dataset.__abstractmethods__ = set()
    GeneralDataset.__abstractmethods__ = set()

    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    net_config = {
        "arch": "gpt",
        "vocab_size": 12,
        "n_layer": 2,
        "n_embd": 12,
        "n_head": 2,
        "dim_feedfwd": 8,
        "block_size": 8,
        "activation": "GELU",
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "min_layers": 8,
        "max_layers": 16,
        "bias": True,
    }

    algo = ILQL(rl_ds, net_config=net_config, double_q=True)

    inputs = {
        "tokens": torch.tensor([[0, 1, 2, 3, 1]]),
        "state_idxs": torch.tensor([[0, 1, 2, 3, 1, 0]]),
        "action_idxs": torch.tensor([[0, 1, 2, 3, 1]]),
        "attn_mask": torch.tensor([[1, 1, 1, 1, 1]]).bool(),
        "rewards": torch.tensor([[0, 0, 0, 0, 0]]),
        "terminals": torch.tensor([[0, 0, 0, 0, 0, 1]]),
    }

    items = algo.prepare_inputs(inputs)
    score = algo.get_scores(items)
    assert isinstance(score, torch.Tensor)

    inputs = {
        "tokens": torch.tensor([[0, 1, 2, 3, 1]]),
        "state_idxs": None,
        "action_idxs": torch.tensor([[0, 1, 2, 3, 1]]),
        "attn_mask": torch.tensor([[1, 1, 1, 1, 1]]).bool(),
        "rewards": torch.tensor([[0, 0, 0, 0, 0]]),
        "terminals": torch.tensor([[0, 0, 0, 0, 0, 1]]),
    }

    items = algo.prepare_inputs(inputs)
    score = algo.get_scores(
        items, exp_weights=True, clip_weight=0.1, include_logits=True
    )
    assert isinstance(score, torch.Tensor)
