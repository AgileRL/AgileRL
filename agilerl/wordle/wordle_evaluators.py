import random
from collections import defaultdict
from typing import Any, Dict, Optional

from agilerl.algorithms.ilql import ILQL
from agilerl.data.rl_data import DataPoint
from agilerl.wordle.wordle_dataset import WordleListDataset
from agilerl.wordle.wordle_env import WordleObservation
from agilerl.wordle.wordle_game import N_CHARS, WordleGame


class Action_Ranking_Evaluator:
    def __init__(self, branching_data: WordleListDataset) -> None:
        self.branching_data = branching_data
        self.expert_actions = defaultdict(list)
        self.non_expert_actions = defaultdict(list)
        for i in range(self.branching_data.size()):
            item = self.branching_data.get_item(i)
            assert item.meta is not None and "kind" in item.meta
            if item.meta["kind"] == "expert":
                for prefix in item.meta["prefixes"]:
                    self.expert_actions[self.hashable_state(prefix)].append(
                        item.meta["self_actions"][len(prefix[1])]
                    )
            elif item.meta["kind"] == "branch_suboptimal":
                start = item.meta["start"]
                self.non_expert_actions[self.hashable_state(start)].append(
                    item.meta["self_actions"][len(start[1])]
                )
            else:
                raise NotImplementedError
        self.states = list(
            set(self.expert_actions.keys()).intersection(
                set(self.non_expert_actions.keys())
            )
        )
        self.vocab = self.branching_data.get_item(0).meta["self"].game.vocab

    def hashable_state(self, state):
        return (
            state[0],
            tuple(state[1]),
        )

    def evaluate(self, model: ILQL, items) -> Optional[Dict[str, Any]]:
        assert not model.double_q
        tokens = model.prepare_inputs(items)["tokens"]
        total_correct = [0 for _ in range(N_CHARS + 1)]
        total_correct_target = [0 for _ in range(N_CHARS + 1)]
        total = 0
        for _ in range(tokens.shape[0]):
            s, a = random.choice(self.states)
            obs = WordleObservation(WordleGame(s, self.vocab.update_vocab(s), list(a)))
            expert_state, _, _ = obs.game.next(
                random.choice(
                    self.expert_actions[
                        self.hashable_state(
                            (
                                s,
                                a,
                            )
                        )
                    ]
                )
            )
            non_expert_state, _, _ = obs.game.next(
                random.choice(
                    self.non_expert_actions[
                        self.hashable_state(
                            (
                                s,
                                a,
                            )
                        )
                    ]
                )
            )
            expert_datapoint = DataPoint.from_obs(
                WordleObservation(expert_state),
                self.branching_data.token_reward,
                self.branching_data.tokenizer,
            )
            non_expert_datapoint = DataPoint.from_obs(
                WordleObservation(non_expert_state),
                self.branching_data.token_reward,
                self.branching_data.tokenizer,
            )
            iql_outputs = model.get_qvs([expert_datapoint, non_expert_datapoint])
            qs, target_qs, terminals = (
                iql_outputs["qs"],
                iql_outputs["target_qs"],
                iql_outputs["terminals"],
            )
            for i in range(N_CHARS + 1):
                total_correct[i] += int(
                    qs[0, (1 - terminals[0, :-1]).sum() - 1 - i]
                    > qs[1, (1 - terminals[1, :-1]).sum() - 1 - i]
                )
                total_correct_target[i] += int(
                    target_qs[0, (1 - terminals[0, :-1]).sum() - 1 - i]
                    > target_qs[1, (1 - terminals[1, :-1]).sum() - 1 - i]
                )
            total += 1
        return {
            **{
                ("q_rank_acc_-%d" % (i + 1)): (total_correct[i] / total, total)
                for i in range(N_CHARS + 1)
            },
            **{
                ("q_target_rank_acc_-%d" % (i + 1)): (
                    total_correct_target[i] / total,
                    total,
                )
                for i in range(N_CHARS + 1)
            },
        }


class Action_Ranking_Evaluator_Adversarial:
    def __init__(self, adversarial_data: WordleListDataset) -> None:
        self.adversarial_data = adversarial_data
        self.expert_actions = defaultdict(list)
        self.adversarial_actions = defaultdict(list)
        self.suboptimal_actions = defaultdict(list)
        for i in range(self.adversarial_data.size()):
            item = self.adversarial_data.get_item(i)
            assert item.meta is not None and "kind" in item.meta
            if item.meta["kind"] == "expert":
                self.expert_actions[self.hashable_state(item.meta["s_0"])].append(
                    item.meta["a_0"]
                )
                if "s_2" in item.meta:
                    self.expert_actions[self.hashable_state(item.meta["s_2"])].append(
                        item.meta["a_2"]
                    )
            elif item.meta["kind"] == "adversarial":
                self.adversarial_actions[self.hashable_state(item.meta["s_2"])].append(
                    item.meta["a_2"]
                )
            elif item.meta["kind"] == "suboptimal":
                self.suboptimal_actions[self.hashable_state(item.meta["s_0"])].append(
                    item.meta["a_0"]
                )
            else:
                raise NotImplementedError
        self.initial_states = list(
            set(self.expert_actions.keys()).intersection(
                set(self.suboptimal_actions.keys())
            )
        )
        self.branch_states = list(
            set(self.expert_actions.keys()).intersection(
                set(self.adversarial_actions.keys())
            )
        )
        self.vocab = self.adversarial_data.get_item(0).meta["self"].game.vocab

    def hashable_state(self, state):
        return (
            state[0],
            tuple(state[1]),
        )

    def evaluate(self, model: ILQL, items) -> Optional[Dict[str, Any]]:
        # evaluate Q-values for suboptimal verses expert at the first action
        # evaluate Q-values for expert versus adversarial at the third action
        assert not model.double_q
        tokens = model.prepare_inputs(items)["tokens"]
        initial_total_correct = [0 for _ in range(N_CHARS + 1)]
        initial_total_correct_target = [0 for _ in range(N_CHARS + 1)]
        branch_total_correct = [0 for _ in range(N_CHARS + 1)]
        branch_total_correct_target = [0 for _ in range(N_CHARS + 1)]
        total = 0
        for _ in range(tokens.shape[0]):
            initial_s, initial_a = random.choice(self.initial_states)
            initial_obs = WordleObservation(
                WordleGame(
                    initial_s, self.vocab.update_vocab(initial_s), list(initial_a)
                )
            )
            initial_expert_a = random.choice(
                self.expert_actions[
                    self.hashable_state(
                        (
                            initial_s,
                            initial_a,
                        )
                    )
                ]
            )
            initial_suboptimal_a = random.choice(
                self.suboptimal_actions[
                    self.hashable_state(
                        (
                            initial_s,
                            initial_a,
                        )
                    )
                ]
            )
            initial_expert_state, _, _ = initial_obs.game.next(initial_expert_a)
            initial_suboptimal_state, _, _ = initial_obs.game.next(initial_suboptimal_a)
            initial_expert_datapoint = DataPoint.from_obs(
                WordleObservation(initial_expert_state),
                self.adversarial_data.tokenizer,
                self.adversarial_data.token_reward,
            )
            initial_suboptimal_datapoint = DataPoint.from_obs(
                WordleObservation(initial_suboptimal_state),
                self.adversarial_data.tokenizer,
                self.adversarial_data.token_reward,
            )
            iql_outputs = model.get_qvs(
                [initial_expert_datapoint, initial_suboptimal_datapoint]
            )
            qs, target_qs, terminals = (
                iql_outputs["qs"],
                iql_outputs["target_qs"],
                iql_outputs["terminals"],
            )
            for i in range(N_CHARS + 1):
                initial_total_correct[i] += int(
                    qs[0, (1 - terminals[0, :-1]).sum() - 1 - i]
                    > qs[1, (1 - terminals[1, :-1]).sum() - 1 - i]
                )
                initial_total_correct_target[i] += int(
                    target_qs[0, (1 - terminals[0, :-1]).sum() - 1 - i]
                    > target_qs[1, (1 - terminals[1, :-1]).sum() - 1 - i]
                )

            branch_s, branch_a = random.choice(self.branch_states)
            branch_obs = WordleObservation(
                WordleGame(branch_s, self.vocab.update_vocab(branch_s), list(branch_a))
            )
            branch_expert_a = random.choice(
                self.expert_actions[
                    self.hashable_state(
                        (
                            branch_s,
                            branch_a,
                        )
                    )
                ]
            )
            branch_adversarial_a = random.choice(
                self.adversarial_actions[
                    self.hashable_state(
                        (
                            branch_s,
                            branch_a,
                        )
                    )
                ]
            )
            branch_expert_state, _, _ = branch_obs.game.next(branch_expert_a)
            branch_adversarial_state, _, _ = branch_obs.game.next(branch_adversarial_a)
            branch_expert_datapoint = DataPoint.from_obs(
                WordleObservation(branch_expert_state),
                self.adversarial_data.tokenizer,
                self.adversarial_data.token_reward,
            )
            branch_adversarial_datapoint = DataPoint.from_obs(
                WordleObservation(branch_adversarial_state),
                self.adversarial_data.tokenizer,
                self.adversarial_data.token_reward,
            )
            iql_outputs = model.get_qvs(
                [branch_expert_datapoint, branch_adversarial_datapoint]
            )
            qs, target_qs, terminals = (
                iql_outputs["qs"],
                iql_outputs["target_qs"],
                iql_outputs["terminals"],
            )
            for i in range(N_CHARS + 1):
                branch_total_correct[i] += int(
                    qs[0, (1 - terminals[0, :-1]).sum() - 1 - i]
                    > qs[1, (1 - terminals[1, :-1]).sum() - 1 - i]
                )
                branch_total_correct_target[i] += int(
                    target_qs[0, (1 - terminals[0, :-1]).sum() - 1 - i]
                    > target_qs[1, (1 - terminals[1, :-1]).sum() - 1 - i]
                )
            total += 1
        return {
            **{
                ("initial_q_rank_acc_-%d" % (i + 1)): (
                    initial_total_correct[i] / total,
                    total,
                )
                for i in range(N_CHARS + 1)
            },
            **{
                ("initial_q_target_rank_acc_-%d" % (i + 1)): (
                    initial_total_correct_target[i] / total,
                    total,
                )
                for i in range(N_CHARS + 1)
            },
            **{
                ("branch_q_rank_acc_-%d" % (i + 1)): (
                    branch_total_correct[i] / total,
                    total,
                )
                for i in range(N_CHARS + 1)
            },
            **{
                ("branch_q_target_rank_acc_-%d" % (i + 1)): (
                    branch_total_correct_target[i] / total,
                    total,
                )
                for i in range(N_CHARS + 1)
            },
        }
