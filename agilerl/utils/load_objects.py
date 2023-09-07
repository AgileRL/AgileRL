import json

import torch

from agilerl.algorithms.bc_lm import BC_LM, BC_Evaluator, BC_Policy
from agilerl.algorithms.ilql import (
    ILQL,
    ILQL_Evaluator,
    ILQL_Policy,
    TopAdvantageNGrams,
)
from agilerl.data.rl_data import ConstantTokenReward, SepcifiedTokenReward
from agilerl.networks.evolvable_gpt import EvolvableGPT
from agilerl.utils.ilql_utils import convert_path
from agilerl.wordle.policy import (
    MixturePolicy,
    MonteCarloPolicy,
    OptimalPolicy,
    RandomMixturePolicy,
    RepeatPolicy,
    StartWordPolicy,
    UserPolicy,
    WrongPolicy,
)
from agilerl.wordle.wordle_dataset import (
    WordleHumanDataset,
    WordleIterableDataset,
    WordleListDataset,
)
from agilerl.wordle.wordle_env import WordleEnvironment
from agilerl.wordle.wordle_evaluators import (
    Action_Ranking_Evaluator,
    Action_Ranking_Evaluator_Adversarial,
)
from agilerl.wordle.wordle_game import Vocabulary

registry = {}
cache = {}


def register(name):
    def add_f(f):
        registry[name] = f
        return f

    return add_f


def load_item(config, *args, verbose=True):
    config = config.copy()
    name = config.pop("name")
    if name not in registry:
        raise NotImplementedError
    if "cache_id" in config:
        if (name, config["cache_id"]) in cache:
            if verbose:
                print(f'loading from cache ({name}, {config["cache_id"]})')
            return cache[(name, config["cache_id"])]
    if verbose:
        print(f"loading {name}: {config}")
    item = registry[name](config, *args, verbose=verbose)
    if "cache_id" in config:
        print(f'saving to cache ({name}, {config["cache_id"]})')
        cache[(name, config["cache_id"])] = item
    return item


def load_model(config, model, device, verbose=True):
    model = model.to(device)
    if config["checkpoint_path"] is not None:
        if verbose:
            print(
                "loading %s state dict from: %s"
                % (config["name"], convert_path(config["checkpoint_path"]))
            )
        chkpt_state_dict = torch.load(
            convert_path(config["checkpoint_path"]), map_location="cpu"
        )
        model.load_state_dict(chkpt_state_dict, strict=config["strict_load"])
        if verbose:
            print("loaded.")
    elif config["gpt_pretrained"]:
        if verbose:
            print(
                "loading %s state dict from: %s"
                % (config["name"], convert_path(config["gpt_checkpoint_path"]))
            )
        pretrained_state_dict = convert_path(config["gpt_checkpoint_path"])
        model.model = EvolvableGPT.from_pretrained(
            model_type=config["gpt_model_type"],
            override_args=model.net_config,
            custom_sd=pretrained_state_dict,
        )
        model.copy_model_to_actor_target()
    return model


@register("constant_token_reward")
def load_constant_token_reward(config, device, verbose=True):
    return ConstantTokenReward(config["c"])


@register("specified_token_reward")
def load_specified_token_reward(config, device, verbose=True):
    with open(convert_path(config["token_file"])) as f:
        token_data = {int(k): v for k, v in json.load(f).items()}
    return SepcifiedTokenReward(token_data, config["scale"], config["shift"])


@register("gpt2")
def load_gpt2(config, verbose=True):
    if config["from_pretrained"]:
        return EvolvableGPT.from_pretrained(config["gpt2_type"])
    return EvolvableGPT()


@register("bc_lm")
def load_bc_lm(config, device, verbose=True):
    dataset = load_item(config["dataset"], device, verbose=verbose)
    config.pop("dataset")
    load_config = config.pop("load")
    model = BC_LM(**config, dataset=dataset, device=device)
    return load_model(load_config, model, device, verbose=verbose)


@register("bc_policy")
def load_bc_policy(config, device, verbose=True):
    bc_lm = load_item(config["bc_lm"], device, verbose=verbose)
    return BC_Policy(bc_lm, config["kind"], **config["generation_kwargs"])


@register("bc_evaluator")
def load_bc_evaluator(config, device, verbose=True):
    env = load_item(config["env"], device, verbose=verbose)
    return BC_Evaluator(
        env, config["env"], config["kind"], **config["generation_kwargs"]
    )


@register("per_token_iql")
def load_per_token_iql(config, device, verbose=True):
    dataset = load_item(config["dataset"], device, verbose=verbose)
    config.pop("dataset")
    load_config = config.pop("load")
    model = ILQL(**config, dataset=dataset, device=device)
    return load_model(load_config, model, device, verbose=verbose)


@register("iql_policy")
def load_iql_policy(config, device, verbose=True):
    iql_model = load_item(config["iql_model"], device, verbose=verbose)
    return ILQL_Policy(iql_model, config["kind"], **config["generation_kwargs"])


@register("iql_evaluator")
def load_iql_evaluator(config, device, verbose=True):
    env = load_item(config["env"], device, verbose=verbose)
    return ILQL_Evaluator(
        env, config["verbose"], config["kind"], **config["generation_kwargs"]
    )


@register("top_advantage_n_grams")
def load_top_advantage_n_grams(config, device, verbose=True):
    data = load_item(config["data"], device, verbose=verbose)
    return TopAdvantageNGrams(
        data, config["print_every"], config["print_k"], config["n_gram"]
    )


@register("vocab")
def load_vocab(config, verbose=True):
    vocab = Vocabulary.from_file(
        convert_path(config["vocab_path"]), config["fill_cache"]
    )
    if config["cache_path"] is not None:
        if verbose:
            print("loading vocab cache from: %s" % convert_path(config["cache_path"]))
        vocab.cache.load(convert_path(config["cache_path"]))
        if verbose:
            print("loaded.")
    return vocab


@register("wordle_env")
def load_wordle_environment(config, device, verbose=True):
    vocab = load_item(config["vocab"], verbose=verbose)
    return WordleEnvironment(vocab)


@register("user_policy")
def load_user_policy(config, device, verbose=True):
    vocab, hint_policy = None, None
    if config["hint_policy"] is not None:
        hint_policy = load_item(config["hint_policy"], device, verbose=verbose)
    if config["vocab"] is not None:
        vocab = load_item(config["vocab"], verbose=verbose)
    return UserPolicy(hint_policy=hint_policy, vocab=vocab)


@register("start_word_policy")
def load_start_word_policy(config, device, verbose=True):
    return StartWordPolicy(config["start_words"])


@register("optimal_policy")
def load_optimal_policy(config, device, verbose=True):
    start_word_policy = None
    if config["start_word_policy"] is not None:
        start_word_policy = load_item(
            config["start_word_policy"], device, verbose=verbose
        )
    policy = OptimalPolicy(
        start_word_policy=start_word_policy, progress_bar=config["progress_bar"]
    )
    if config["cache_path"] is not None:
        if verbose:
            print(
                "loading optimal policy cache from: %s"
                % convert_path(config["cache_path"])
            )
        policy.cache.load(convert_path(config["cache_path"]))
        if verbose:
            print("loaded.")
    return policy


@register("wrong_policy")
def load_wrong_policy(config, device, verbose=True):
    vocab = load_item(config["vocab"], verbose=verbose)
    return WrongPolicy(vocab=vocab)


@register("repeat_policy")
def load_repeat_policy(config, device, verbose=True):
    start_word_policy = None
    if config["start_word_policy"] is not None:
        start_word_policy = load_item(
            config["start_word_policy"], device, verbose=verbose
        )
    return RepeatPolicy(start_word_policy=start_word_policy, first_n=config["first_n"])


@register("mixture_policy")
def load_mixture_policy(config, device, verbose=True):
    policy1 = load_item(config["policy1"], device, verbose=verbose)
    policy2 = load_item(config["policy2"], device, verbose=verbose)
    return MixturePolicy(config["prob1"], policy1, policy2)


@register("random_mixture_policy")
def load_random_mixture_policy(config, device, verbose=True):
    vocab = None
    if config["vocab"] is not None:
        vocab = load_item(config["vocab"], verbose=verbose)
    return RandomMixturePolicy(prob_smart=config["prob_smart"], vocab=vocab)


@register("monte_carlo_policy")
def load_monte_carlo_policy(config, device, verbose=True):
    sample_policy = load_item(config["sample_policy"], device, verbose=verbose)
    return MonteCarloPolicy(n_samples=config["n_samples"], sample_policy=sample_policy)


@register("wordle_iterable_dataset")
def load_wordle_iterable_dataset(config, device, verbose=True):
    policy = load_item(config["policy"], device, verbose=verbose)
    vocab = load_item(config["vocab"], verbose=verbose)
    token_reward = load_item(config["token_reward"], device, verbose=verbose)
    return WordleIterableDataset(
        policy, vocab, max_len=config["max_len"], token_reward=token_reward
    )


@register("wordle_dataset")
def load_wordle_dataset(config, device, verbose=True):
    if config["vocab"] is not None:
        vocab = load_item(config["vocab"], verbose=verbose)
    else:
        vocab = None
    token_reward = load_item(config["token_reward"], device, verbose=verbose)
    return WordleListDataset.from_file(
        convert_path(config["file_path"]), config["max_len"], vocab, token_reward
    )


@register("wordle_human_dataset")
def load_human_dataset(config, device, verbose=True):
    token_reward = load_item(config["token_reward"], device, verbose=verbose)
    game_indexes = None
    if config["index_file"] is not None:
        with open(convert_path(config["index_file"])) as f:
            game_indexes = json.load(f)
    return WordleHumanDataset.from_file(
        convert_path(config["file_path"]),
        config["use_true_word"],
        config["max_len"],
        token_reward,
        game_indexes,
        config["top_p"],
    )


@register("action_ranking_evaluator")
def load_action_ranking_evaluator(config, device, verbose=True):
    branching_data = load_item(config["branching_data"], device, verbose=verbose)
    return Action_Ranking_Evaluator(branching_data)


@register("action_ranking_evaluator_adversarial")
def load_action_ranking_evaluator_adversarial(config, device, verbose=True):
    adversarial_data = load_item(config["adversarial_data"], device, verbose=verbose)
    return Action_Ranking_Evaluator_Adversarial(adversarial_data)
