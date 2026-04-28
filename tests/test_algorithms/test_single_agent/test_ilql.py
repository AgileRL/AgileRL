from pathlib import Path

import torch
import pytest
from agilerl.algorithms.ilql import (
    ILQL,
    ILQL_Evaluator,
    ILQL_Policy,
    TopAdvantageNGrams,
    interact_environment,
    map_pytree,
    parameter_norm,
    to,
    to_decorator,
)
from agilerl.data.language_environment import Language_Observation
from agilerl.data.rl_data import ConstantTokenReward, DataPoint, List_RL_Dataset
from agilerl.data.torch_datasets import GeneralDataset
from tests.helper_functions import assert_state_dicts_equal
from tests.test_data import WordleTokenizer

torch.serialization.add_safe_globals(
    [
        WordleTokenizer,
        List_RL_Dataset,
        ConstantTokenReward,
        DataPoint,
        GeneralDataset,
        Language_Observation,
    ],
)


def test_ilql_init():
    """Use a tiny EvolvableGPT config (same spirit as TINY_LLM_FIXTURE_PATH for HF tests).

    net_config=None builds GPT-2–scale networks (three EvolvableGPT stacks); that is too
    slow for default CI and is not needed to assert hyperparameter wiring.
    """
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    algo = ILQL(rl_ds, net_config=_net_config())

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
    algo.clean_up()


def test_forward():
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    net_config = _net_config()
    double_q = True

    algo = ILQL(
        rl_ds,
        net_config=net_config,
        double_q=double_q,
        value_min=0,
        value_max=1,
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
        algo.device,
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
    algo.clean_up()


def test_get_weights_sample_raw():
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    algo = ILQL(rl_ds, net_config=_net_config())

    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    vs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    qs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    state_idxs = torch.tensor([[0, 1, 2, 3, 4]])
    action_idxs = torch.empty((1, 0)).long()
    terminals = torch.tensor([[0, 0, 0, 0, 1]])
    weights = algo.get_weights(tokens, vs, qs, state_idxs, action_idxs, terminals)
    assert weights.shape == (1, 5)


def test_get_loss():
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    net_config = _net_config()
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
        rl_ds,
        net_config=net_config,
        double_q=False,
        exp_weights=False,
        clip_weight=0.1,
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

    loss_dict = {}
    loss_dict["token_loss"] = loss[1]["token_loss"][0]
    loss_dict["q_loss"] = loss[1]["q_loss"][0]
    loss_dict["v_loss"] = loss[1]["v_loss"][0]
    loss_dict["cql_loss"] = loss[1]["cql_loss"][0]
    loss_dict["dm_loss"] = loss[1]["dm_loss"][0]

    _ = loss[2][0](loss_dict)
    _ = loss[2][1](loss_dict)

    assert "loss" in loss_dict
    assert "advantage_hist" in loss_dict
    algo.clean_up()


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

        def __str__(self):
            return "LangObs"

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
    assert "tokens" in items
    assert "state_idxs" in items
    assert "action_idxs" in items
    algo.clean_up()


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
        items,
        exp_weights=True,
        clip_weight=0.1,
        include_logits=True,
    )
    assert isinstance(score, torch.Tensor)

    initial_score = algo.initial_score(items)

    assert initial_score[0].shape == (1, 35)
    algo.clean_up()


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

    algo.soft_update()

    eval_params = list(algo.q.parameters())
    target_params = list(algo.target_q.parameters())
    expected_params = [
        algo.alpha * eval_param + (1.0 - algo.alpha) * target_param
        for eval_param, target_param in zip(eval_params, target_params, strict=False)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(
            expected_params,
            target_params,
            strict=False,
        )
    )

    eval_params = list(algo.q2.parameters())
    target_params = list(algo.target_q2.parameters())
    expected_params = [
        algo.alpha * eval_param + (1.0 - algo.alpha) * target_param
        for eval_param, target_param in zip(eval_params, target_params, strict=False)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(
            expected_params,
            target_params,
            strict=False,
        )
    )

    eval_params = list(algo.actor_target.parameters())
    target_params = list(algo.model.parameters())
    expected_params = [
        algo.alpha * eval_param + (1.0 - algo.alpha) * target_param
        for eval_param, target_param in zip(eval_params, target_params, strict=False)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(
            expected_params,
            target_params,
            strict=False,
        )
    )
    algo.clean_up()


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

    algo.hard_update()

    assert_state_dicts_equal(algo.q.state_dict(), algo.target_q.state_dict())
    assert_state_dicts_equal(algo.q2.state_dict(), algo.target_q2.state_dict())
    assert_state_dicts_equal(algo.actor_target.state_dict(), algo.model.state_dict())
    algo.clean_up()


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
    assert_state_dicts_equal(algo.v.state_dict(), clone.v.state_dict())
    assert_state_dicts_equal(algo.pi.state_dict(), clone.pi.state_dict())
    assert_state_dicts_equal(algo.q.state_dict(), clone.q.state_dict())
    assert_state_dicts_equal(algo.target_q.state_dict(), clone.target_q.state_dict())
    assert_state_dicts_equal(algo.q2.state_dict(), clone.q2.state_dict())
    assert_state_dicts_equal(algo.target_q2.state_dict(), clone.target_q2.state_dict())
    assert_state_dicts_equal(algo.actor.state_dict(), clone.actor.state_dict())
    assert_state_dicts_equal(
        algo.actor_target.state_dict(),
        clone.actor_target.state_dict(),
    )
    assert_state_dicts_equal(algo.model.state_dict(), clone.model.state_dict())
    algo.clean_up()
    clone.clean_up()


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
    algo.save_checkpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)

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
    algo.load_checkpoint(checkpoint_path)

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
    algo.clean_up()


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
    algo.clean_up()


def test_act():
    List_RL_Dataset.__abstractmethods__ = set()
    GeneralDataset.__abstractmethods__ = set()

    class LangObs(Language_Observation):
        def to_sequence(self, **kwargs):
            return [("asdfg", None), ("zxcvb", 1)], True

        def __str__(self):
            return "LangObs"

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
    algo.clean_up()


def _net_config(vocab_size=12, block_size=8):
    return {
        "arch": "gpt",
        "vocab_size": vocab_size,
        "n_layer": 2,
        "n_embd": 12,
        "n_head": 2,
        "dim_feedfwd": 8,
        "block_size": block_size,
        "activation": "GELU",
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "min_layers": 8,
        "max_layers": 16,
        "bias": True,
    }


def _make_algo(double_q=True, **kwargs):
    List_RL_Dataset.__abstractmethods__ = set()
    tokenizer = WordleTokenizer()
    token_reward = ConstantTokenReward(1)
    rl_ds = List_RL_Dataset(tokenizer, token_reward, 10)
    return ILQL(rl_ds, net_config=_net_config(), double_q=double_q, **kwargs)


def _simple_inputs():
    return {
        "tokens": torch.tensor([[0, 1, 2, 3, 1]]),
        "state_idxs": torch.tensor([[0, 1, 2, 3, 1]]),
        "action_idxs": torch.tensor([[0, 1, 2, 3, 1]]),
        "attn_mask": torch.tensor([[1, 1, 1, 1, 1]]).bool(),
        "rewards": torch.tensor([[0, 0, 0, 0, 0]]),
        "terminals": torch.tensor([[0, 0, 0, 0, 0, 1]]),
    }


def test_forward_skip_policy_branch_train_vs_eval():
    algo = _make_algo(double_q=False)
    data = _simple_inputs()

    algo.train()
    out_train = algo(
        data["tokens"],
        data["state_idxs"],
        data["action_idxs"],
        data["attn_mask"],
        skip_policy_on_train=True,
    )
    assert torch.all(out_train["logits"] == 0)

    algo.eval()
    out_eval = algo(
        data["tokens"],
        data["state_idxs"],
        data["action_idxs"],
        data["attn_mask"],
        skip_policy_on_train=True,
        detach_full_policy=False,
    )
    assert not torch.all(out_eval["logits"] == 0)
    algo.clean_up()


def test_score_no_logits_no_advantage_is_zero():
    algo = _make_algo(double_q=False)
    data = _simple_inputs()

    weights, _ = algo.score(
        data["tokens"],
        data["attn_mask"],
        data["state_idxs"],
        data["action_idxs"],
        include_logits=False,
        include_advantage=False,
    )

    assert torch.allclose(weights, torch.zeros_like(weights))
    algo.clean_up()


def test_score_trivial_value_query_with_action_mask():
    algo = _make_algo(double_q=False)
    tokens = torch.tensor([[0, 1, 2, 3, 1]])

    weights, _ = algo.score(
        tokens=tokens,
        attn_mask=None,
        state_idxs=None,
        action_idxs=None,
        exp_weights=True,
        include_logits=False,
        include_advantage=True,
        action_mask=torch.tensor([0.0]),
    )
    assert torch.allclose(weights, torch.zeros_like(weights))
    algo.clean_up()


def test_single_q_update_checkpoint_and_cleanup(tmpdir):
    algo = _make_algo(double_q=False)
    assert not hasattr(algo, "q2")

    algo.soft_update()
    algo.hard_update()

    checkpoint_path = Path(tmpdir) / "checkpoint_single_q.pth"
    algo.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["q2_init_dict"] is None
    assert checkpoint["target_q2_init_dict"] is None

    loaded = _make_algo(double_q=False)
    loaded.load_checkpoint(checkpoint_path)
    assert loaded.double_q is False
    assert not hasattr(loaded, "q2")

    algo.clean_up()
    loaded.clean_up()


def test_clip_values_returns_unchanged_when_bounds_none():
    algo = _make_algo(double_q=False)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    result = algo.clip_values(values)
    assert torch.equal(result, values)
    algo.clean_up()


@pytest.mark.parametrize("double_q", [True, False])
def test_loss_helpers_return_scalar(double_q):
    algo = _make_algo(double_q=double_q)
    terminals = torch.tensor([[0, 0, 0, 1]])
    action_tokens = torch.tensor([[0, 1, 2]])

    if double_q:
        q1 = torch.randn(1, 3, algo.dataset.tokenizer.num_tokens())
        q2 = torch.randn(1, 3, algo.dataset.tokenizer.num_tokens())
        qs = (q1, q2)
        chosen_qs = (
            torch.gather(q1, 2, action_tokens.unsqueeze(2)).squeeze(2),
            torch.gather(q2, 2, action_tokens.unsqueeze(2)).squeeze(2),
        )
    else:
        qs = torch.randn(1, 3, algo.dataset.tokenizer.num_tokens())
        chosen_qs = torch.gather(qs, 2, action_tokens.unsqueeze(2)).squeeze(2)

    cql_loss = algo.get_cql_loss(qs, action_tokens, terminals)
    dm_loss = algo.get_dm_loss(qs, chosen_qs, terminals, margin=0.1)
    q_loss = algo.get_q_loss(
        vns=torch.randn(1, 3),
        qs=chosen_qs,
        rs=torch.zeros(1, 3),
        gamma=0.99,
        terminals=terminals,
    )

    assert cql_loss.ndim == 0
    assert dm_loss.ndim == 0
    assert q_loss.ndim == 0
    algo.clean_up()


def test_policy_generate_dispatch_and_sample_assert():
    algo = _make_algo(double_q=False)
    data = _simple_inputs()

    policy_beam = ILQL_Policy(algo, "beam")
    policy_beam.beam_raw = lambda *args, **kwargs: (
        [("x", ["y"])],
        torch.zeros(1, 1),
        torch.zeros(1, 1),
    )
    generations, _, _ = policy_beam.generate(data, lambda _x: True)
    assert generations[0][1][0] == "y"

    policy_sample = ILQL_Policy(algo, "sample")
    policy_sample.sample_raw = lambda *args, **kwargs: (
        [("a", ["b"])],
        torch.zeros(1, 1),
        torch.zeros(1, 1),
    )
    generations, _, _ = policy_sample.generate(data, lambda _x: True)
    assert generations[0][1][0] == "b"

    with pytest.raises(AssertionError):
        ILQL_Policy(algo, "sample").sample_raw(
            data["tokens"],
            data["attn_mask"],
            data["state_idxs"],
            data["action_idxs"],
            termination_condition=lambda _x: True,
            include_logits=False,
            include_adv=False,
        )
    algo.clean_up()


def test_ilql_evaluator_evaluate_and_dump(monkeypatch):
    algo = _make_algo(double_q=False)
    data = _simple_inputs()
    data["tokens"] = torch.tensor([[0, 1, 2, 3, 1], [0, 1, 2, 3, 1]])
    data["attn_mask"] = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]).bool()

    def fake_interact(_env, policy, _obs):
        policy.kls_all.append(0.5)
        policy.logprobs_all.append(-0.25)
        return "obs", [("obs", "act", 1.0, False), ("obs2", None, 0.0, True)]

    monkeypatch.setattr(
        "agilerl.algorithms.ilql.interact_environment",
        fake_interact,
    )
    monkeypatch.setattr(
        DataPoint,
        "get_token_reward",
        staticmethod(lambda *_args, **_kwargs: [1.0, 2.0]),
    )

    evaluator = ILQL_Evaluator(env=object(), verbose=False, kind="beam")
    logs = evaluator.evaluate(algo, data)
    dump = evaluator.dump()

    assert set(logs) == {"token_reward", "env_reward", "kl", "entropy"}
    assert "results" in dump
    assert "entropies" in dump
    assert len(dump["results"]) == 2
    algo.clean_up()


def test_top_advantage_ngrams_evaluate_and_utils():
    class FakeData:
        def __init__(self):
            self.tokenizer = WordleTokenizer()

        def size(self):
            return 1

        def get_item(self, _idx):
            return object()

    class FakeModel:
        def prepare_inputs(self, _items):
            return {
                "tokens": torch.tensor([[0, 1, 2, 8, 3]]),
                "action_idxs": torch.tensor([[0, 1, 2]]),
            }

        def get_qvs(self, _items):
            return {
                "target_qs": torch.tensor([[1.0, 1.0, 1.0]]),
                "target_vs": torch.tensor([[0.5, 0.5, 0.5]]),
            }

    evaluator = TopAdvantageNGrams(FakeData(), print_every=1, print_k=1, n_gram=None)
    evaluator.evaluate(FakeModel(), None)

    nested = {"a": [torch.tensor([1, 2]), {"b": torch.tensor([3])}]}
    mapped = map_pytree(lambda x: x + 1, nested)
    moved = to({"x": [1, 2]}, "cpu")
    inc = to_decorator(lambda x: x + 1, "cpu")

    assert mapped["a"][0].tolist() == [2, 3]
    assert moved["x"] == [1, 2]
    assert inc(1) == 2
    assert parameter_norm(torch.nn.Linear(3, 2)) > 0


def test_sample_and_beam_raw_cover_generation_loops(monkeypatch):
    # ``sample_raw`` mixes ``edited_logits``, ``log(adv_logits)`` (which can
    # be -inf), and ``base_logits``; depending on the random GPT initialisation
    # those can sum to NaN, failing ``Categorical``'s validation. Seeding
    # makes the coverage check deterministic across xdist worker orderings.
    torch.manual_seed(0)
    algo = _make_algo(double_q=False)
    tokens = torch.tensor([[0, 1, 2]])
    attn_mask = torch.tensor([[1, 1, 1]]).bool()
    state_idxs = torch.tensor([[0, 1, 2]])
    action_idxs = torch.tensor([[0, 1, 2]])
    tokenizer = algo.dataset.tokenizer

    monkeypatch.setattr(
        torch.distributions.categorical.Categorical,
        "sample",
        lambda self: torch.full(
            (self.logits.shape[0],),
            tokenizer.eoa_token_id,
            dtype=torch.long,
            device=self.logits.device,
        ),
    )

    policy = ILQL_Policy(algo, "sample")
    sample_out = policy.sample_raw(
        tokens=tokens,
        attn_mask=attn_mask,
        state_idxs=state_idxs,
        action_idxs=action_idxs,
        termination_condition=lambda _x: True,
        max_generation_len=1,
        num_generations=1,
    )
    assert len(sample_out[0]) == 1

    original_forward = algo.forward

    def patched_forward(*args, **kwargs):
        kwargs.pop("is_causal", None)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(algo, "forward", patched_forward)
    policy = ILQL_Policy(algo, "beam")
    beam_out = policy.beam_raw(
        tokens=tokens,
        attn_mask=attn_mask,
        state_idxs=state_idxs,
        action_idxs=action_idxs,
        termination_condition=lambda _x: True,
        max_generation_len=1,
        beam_width=1,
    )
    assert len(beam_out[0]) == 1
    algo.clean_up()


def test_policy_train_eval_and_invalid_kind_path():
    algo = _make_algo(double_q=False)
    policy = ILQL_Policy(algo, "beam")
    policy.train()
    assert algo.training is True
    policy.eval()
    assert algo.training is False

    policy.kind = "invalid"
    with pytest.raises(NotImplementedError):
        policy.generate(_simple_inputs(), lambda _x: True)
    algo.clean_up()


def test_ilql_interact_environment_function():
    class Env:
        def __init__(self):
            self._terminal = False

        def reset(self):
            self._terminal = False
            return "start"

        def is_terminal(self):
            return self._terminal

        def step(self, _action):
            self._terminal = True
            return "next", 1.0, True

    class Policy:
        def act(self, _obs):
            return "action"

    obs, seq = interact_environment(Env(), Policy(), None)
    assert obs == "next"
    assert len(seq) == 2


def test_ilql_evaluator_verbose_and_top_advantage_ngram_branch(monkeypatch):
    algo = _make_algo(double_q=False)
    data = _simple_inputs()
    data["tokens"] = torch.tensor([[0, 1, 2, 3, 1]])
    data["attn_mask"] = torch.tensor([[1, 1, 1, 1, 1]]).bool()

    def fake_interact(_env, policy, _obs):
        policy.kls_all.append(0.25)
        policy.logprobs_all.append(-0.5)
        return "obs", [("obs", "act", 1.0, False), ("obs2", None, 0.0, True)]

    monkeypatch.setattr(
        "agilerl.algorithms.ilql.interact_environment",
        fake_interact,
    )
    monkeypatch.setattr(
        DataPoint,
        "get_token_reward",
        staticmethod(lambda *_args, **_kwargs: [0.0, 1.0]),
    )

    evaluator = ILQL_Evaluator(env=object(), verbose=True, kind="sample")
    logs = evaluator.evaluate(algo, data)
    assert "entropy" in logs

    class FakeData:
        def __init__(self):
            self.tokenizer = WordleTokenizer()

        def size(self):
            return 1

        def get_item(self, _idx):
            return object()

    class FakeModel:
        def prepare_inputs(self, _items):
            return {
                "tokens": torch.tensor([[0, 1, 2, 8, 3]]),
                "action_idxs": torch.tensor([[0, 1, 2]]),
            }

        def get_qvs(self, _items):
            return {
                "target_qs": torch.tensor([[1.0, 1.0, 1.0]]),
                "target_vs": torch.tensor([[0.5, 0.5, 0.5]]),
            }

    TopAdvantageNGrams(FakeData(), print_every=1, print_k=1, n_gram=1).evaluate(
        FakeModel(),
        None,
    )
    algo.clean_up()
