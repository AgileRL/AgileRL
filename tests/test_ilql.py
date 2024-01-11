from pathlib import Path

import torch

from agilerl.algorithms.ilql import ILQL, ILQL_Policy
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

    initial_score = algo.initial_score(items)

    assert initial_score[0].shape == (1, 35)


def test_soft_update():
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

    algo.softUpdate()

    eval_params = list(algo.q.parameters())
    target_params = list(algo.target_q.parameters())
    expected_params = [
        algo.alpha * eval_param + (1.0 - algo.alpha) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )

    eval_params = list(algo.q2.parameters())
    target_params = list(algo.target_q2.parameters())
    expected_params = [
        algo.alpha * eval_param + (1.0 - algo.alpha) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )

    eval_params = list(algo.actor_target.parameters())
    target_params = list(algo.model.parameters())
    expected_params = [
        algo.alpha * eval_param + (1.0 - algo.alpha) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )


def test_hard_update():
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

    algo.hardUpdate()

    assert str(algo.q.state_dict()) == str(algo.target_q.state_dict())
    assert str(algo.q2.state_dict()) == str(algo.target_q2.state_dict())
    assert str(algo.actor_target.state_dict()) == str(algo.model.state_dict())


def test_clone():
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

    clone = algo.clone()

    assert algo.algo == clone.algo
    assert algo.index == clone.index
    assert algo.batch_size == clone.batch_size
    assert algo.lr == clone.lr
    assert algo.alpha == clone.alpha
    assert algo.beta == clone.beta
    assert algo.gamma == clone.gamma
    assert algo.tau == clone.tau
    assert algo.mut is clone.mut
    assert algo.transition_weight == clone.transition_weight
    assert algo.clip_weight is clone.clip_weight
    assert algo.value_max is clone.value_max
    assert algo.value_min is clone.value_min
    assert algo.detach_v is clone.detach_v
    assert algo.detach_q is clone.detach_q
    assert algo.detach_pi is clone.detach_pi
    assert algo.double_q is clone.double_q
    assert algo.per_token is clone.per_token
    assert algo.exp_weights is clone.exp_weights
    assert algo.dm_margin == clone.dm_margin
    assert algo.cql_temp == clone.cql_temp
    assert algo.weight_decay == clone.weight_decay
    assert algo.device == clone.device
    assert str(algo.v.state_dict()) == str(clone.v.state_dict())
    assert str(algo.pi.state_dict()) == str(clone.pi.state_dict())
    assert str(algo.q.state_dict()) == str(clone.q.state_dict())
    assert str(algo.target_q.state_dict()) == str(clone.target_q.state_dict())
    assert str(algo.q2.state_dict()) == str(clone.q2.state_dict())
    assert str(algo.target_q2.state_dict()) == str(clone.target_q2.state_dict())
    assert str(algo.actor.state_dict()) == str(clone.actor.state_dict())
    assert str(algo.actor_target.state_dict()) == str(clone.actor_target.state_dict())
    assert str(algo.model.state_dict()) == str(clone.model.state_dict())


def test_save_load_checkpoint(tmpdir):
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

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    algo.saveCheckpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path)

    assert "model_init_dict" in checkpoint
    assert "model_state_dict" in checkpoint
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "v_init_dict" in checkpoint
    assert "v_state_dict" in checkpoint
    assert "pi_init_dict" in checkpoint
    assert "pi_state_dict" in checkpoint
    assert "q_init_dict" in checkpoint
    assert "q_state_dict" in checkpoint
    assert "target_q_init_dict" in checkpoint
    assert "target_q_state_dict" in checkpoint
    assert "q2_init_dict" in checkpoint
    assert "q2_state_dict" in checkpoint
    assert "target_q2_init_dict" in checkpoint
    assert "target_q2_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "dataset" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "alpha" in checkpoint
    assert "beta" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mutation" in checkpoint
    assert "transition_weight" in checkpoint
    assert "clip_weight" in checkpoint
    assert "value_max" in checkpoint
    assert "value_min" in checkpoint
    assert "detach_v" in checkpoint
    assert "detach_q" in checkpoint
    assert "detach_pi" in checkpoint
    assert "double_q" in checkpoint
    assert "per_token" in checkpoint
    assert "exp_weights" in checkpoint
    assert "dm_margin" in checkpoint
    assert "cql_temp" in checkpoint
    assert "weight_decay" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    algo = ILQL(rl_ds, net_config=net_config, double_q=True)
    algo.loadCheckpoint(checkpoint_path)

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


def test_init_policy():
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

    policy = ILQL_Policy(algo, "beam")

    assert policy.iql_model == algo
    assert policy.kind == "beam"
    assert policy.generation_kwargs == {}
    assert policy.kls_all == []
    assert policy.logprobs_all == []


def test_act():
    List_RL_Dataset.__abstractmethods__ = set()
    GeneralDataset.__abstractmethods__ = set()

    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("asdfg", None), ("zxcvb", 1)], True

    lang_obs = LangObs()

    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 8)
    net_config = {
        "arch": "gpt",
        "vocab_size": 50,
        "n_layer": 2,
        "n_embd": 12,
        "n_head": 2,
        "dim_feedfwd": 8,
        "block_size": 12,
        "activation": "GELU",
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "min_layers": 8,
        "max_layers": 16,
        "bias": True,
    }

    algo = ILQL(rl_ds, net_config=net_config, double_q=True)
    algo.max_length = None

    policy = ILQL_Policy(algo, "beam", beam_width=5)

    action = policy.act(lang_obs)

    assert isinstance(action, str)

    policy = ILQL_Policy(algo, "sample")

    action = policy.act(lang_obs)

    assert isinstance(action, str)
