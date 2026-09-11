# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""This tutorial shows how to train a DQN agent on the connect four environment, using curriculum learning and self play.

Author: Nick (https://github.com/nicku-a)
"""

import os
import random
from collections import deque
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
import torch
import wandb
import yaml
from pettingzoo import ParallelEnv
from pettingzoo.classic import connect_four_v3
from tensordict import TensorDict
from tqdm import trange

from agilerl.algorithms import DQN
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection


class CurriculumEnv:
    """Wrapper around environment to modify reward for curriculum learning.

    :param env: Environment to learn in
    :type env: PettingZoo-style environment
    :param lesson: Lesson settings for curriculum learning
    :type lesson: dict
    """

    def __init__(self, env: ParallelEnv, lesson: dict):
        self.env = env
        self.lesson = lesson

    def check_winnable(self, lst: list[int], piece: int) -> bool:
        """Checks if four pieces in a row represent a winnable opportunity, e.g. [1, 1, 1, 0] or [2, 0, 2, 2].

        :param lst: List of pieces in row
        :type lst: List
        :param piece: Player piece we are checking (1 or 2)
        :type piece: int
        """
        return lst.count(piece) == 3 and lst.count(0) == 1

    def check_vertical_win(self, player: int) -> bool:
        """Checks if a win is vertical.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        board = np.array(self.env.env.board).reshape(6, 7)
        piece = player + 1

        column_count = 7
        row_count = 6

        # Check vertical locations for win
        for c in range(column_count):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c] == piece
                    and board[r + 2][c] == piece
                    and board[r + 3][c] == piece
                ):
                    return True
        return False

    def check_three_in_row(self, player: int) -> int:
        """Checks if there are three pieces in a row and a blank space next, or two pieces - blank - piece.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        board = np.array(self.env.env.board).reshape(6, 7)
        piece = player + 1

        # Check horizontal locations
        column_count = 7
        row_count = 6
        three_in_row_count = 0

        # Check vertical locations
        for c in range(column_count):
            for r in range(row_count - 3):
                if self.check_winnable(board[r : r + 4, c].tolist(), piece):
                    three_in_row_count += 1

        # Check horizontal locations
        for r in range(row_count):
            for c in range(column_count - 3):
                if self.check_winnable(board[r, c : c + 4].tolist(), piece):
                    three_in_row_count += 1

        # Check positively sloped diagonals
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if self.check_winnable(
                    [
                        board[r, c],
                        board[r + 1, c + 1],
                        board[r + 2, c + 2],
                        board[r + 3, c + 3],
                    ],
                    piece,
                ):
                    three_in_row_count += 1

        # Check negatively sloped diagonals
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if self.check_winnable(
                    [
                        board[r, c],
                        board[r - 1, c + 1],
                        board[r - 2, c + 2],
                        board[r - 3, c + 3],
                    ],
                    piece,
                ):
                    three_in_row_count += 1

        return three_in_row_count

    def reward(self, done: bool, player: int) -> float:
        """Processes and returns reward from environment according to lesson criteria.

        :param done: Environment has terminated
        :type done: bool
        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        if done:
            reward = (
                self.lesson["rewards"]["vertical_win"]
                if self.check_vertical_win(player)
                else self.lesson["rewards"]["win"]
            )
        else:
            agent_three_count = self.check_three_in_row(1 - player)
            opp_three_count = self.check_three_in_row(player)
            if (agent_three_count + opp_three_count) == 0:
                reward = self.lesson["rewards"]["play_continues"]
            else:
                reward = (
                    self.lesson["rewards"]["three_in_row"] * agent_three_count
                    + self.lesson["rewards"]["opp_three_in_row"] * opp_three_count
                )
        return reward

    def last(self) -> tuple[dict, float, bool, bool, dict]:
        """Wrapper around PettingZoo env last method."""
        return self.env.last()

    def step(self, action: int) -> None:
        """Wrapper around PettingZoo env step method."""
        self.env.step(action)

    def reset(self) -> None:
        """Wrapper around PettingZoo env reset method."""
        self.env.reset()


class Opponent:
    """Connect 4 opponent to train and/or evaluate against.

    :param env: Environment to learn in
    :type env: PettingZoo-style environment
    :param difficulty: Difficulty level of opponent, 'random', 'weak' or 'strong'
    :type difficulty: str
    """

    def __init__(self, env: ParallelEnv, difficulty: str):
        self.env = env.env
        self.difficulty = difficulty
        if self.difficulty == "random":
            self.get_action = self.random_opponent
        elif self.difficulty == "weak":
            self.get_action = self.weak_rule_based_opponent
        else:
            self.get_action = self.strong_rule_based_opponent
        self.num_cols = 7
        self.num_rows = 6
        self.length = 4
        self.top = [0] * self.num_cols

    def update_top(self) -> None:
        """Updates self.top, a list which tracks the row on top of the highest piece in each column."""
        board = np.array(self.env.env.board).reshape(self.num_rows, self.num_cols)
        non_zeros = np.where(board != 0)
        rows, cols = non_zeros
        top = np.zeros(board.shape[1], dtype=int)
        for col in range(board.shape[1]):
            column_pieces = rows[cols == col]
            if len(column_pieces) > 0:
                top[col] = np.min(column_pieces) - 1
            else:
                top[col] = 5
        full_columns = np.all(board != 0, axis=0)
        top[full_columns] = 6
        self.top = top

    def random_opponent(
        self,
        action_mask: list[int],
        last_opp_move: int | None = None,
        block_vert_coef: float = 1,
    ) -> int:
        """Takes move for random opponent. If the lesson aims to randomly block vertical
        wins with a higher probability, this is done here too.

        :param action_mask: Mask of legal actions: 1=legal, 0=illegal
        :type action_mask: List
        :param last_opp_move: Most recent action taken by agent against this opponent
        :type last_opp_move: int
        :param block_vert_coef: How many times more likely to block vertically
        :type block_vert_coef: float
        """
        if last_opp_move is not None:
            action_mask[last_opp_move] *= block_vert_coef
        return random.choices(list(range(self.num_cols)), action_mask)[0]

    def weak_rule_based_opponent(self, player: int) -> int:
        """Takes move for weak rule-based opponent.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        self.update_top()
        max_length = -1
        best_actions = []
        for action in range(self.num_cols):
            possible, reward, ended, lengths = self.outcome(
                action,
                player,
                return_length=True,
            )
            if possible and lengths.sum() > max_length:
                best_actions = []
                max_length = lengths.sum()
            if possible and lengths.sum() == max_length:
                best_actions.append(action)
        return random.choice(best_actions)

    def strong_rule_based_opponent(self, player: int) -> int:
        """Takes move for strong rule-based opponent.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        self.update_top()

        winning_actions = []
        for action in range(self.num_cols):
            possible, reward, ended = self.outcome(action, player)
            if possible and ended:
                winning_actions.append(action)
        if len(winning_actions) > 0:
            return random.choice(winning_actions)

        opp = 1 if player == 0 else 0
        loss_avoiding_actions = []
        for action in range(self.num_cols):
            possible, reward, ended = self.outcome(action, opp)
            if possible and ended:
                loss_avoiding_actions.append(action)
        if len(loss_avoiding_actions) > 0:
            return random.choice(loss_avoiding_actions)

        return self.weak_rule_based_opponent(player)  # take best possible move

    def outcome(
        self,
        action: int,
        player: int,
        return_length: bool = False,
    ) -> tuple[bool, float | None, bool, np.ndarray | None]:
        """Takes move for weak rule-based opponent.

        :param action: Action to take in environment
        :type action: int
        :param player: Player who we are checking, 0 or 1
        :type player: int
        :param return_length: Return length of outcomes, defaults to False
        :type return_length: bool, optional
        """
        if not (self.top[action] < self.num_rows):  # action column is full
            return (False, None, None) + ((None,) if return_length else ())

        row, col = self.top[action], action
        piece = player + 1

        # down, up, left, right, down-left, up-right, down-right, up-left,
        directions = np.array(
            [
                [[-1, 0], [1, 0]],
                [[0, -1], [0, 1]],
                [[-1, -1], [1, 1]],
                [[-1, 1], [1, -1]],
            ],
        )  # |4x2x2|

        positions = np.array([row, col]).reshape(1, 1, 1, -1) + np.expand_dims(
            directions,
            -2,
        ) * np.arange(1, self.length).reshape(
            1,
            1,
            -1,
            1,
        )  # |4x2x3x2|
        valid_positions = np.logical_and(
            np.logical_and(
                positions[:, :, :, 0] >= 0,
                positions[:, :, :, 0] < self.num_rows,
            ),
            np.logical_and(
                positions[:, :, :, 1] >= 0,
                positions[:, :, :, 1] < self.num_cols,
            ),
        )  # |4x2x3|
        d0 = np.where(valid_positions, positions[:, :, :, 0], 0)
        d1 = np.where(valid_positions, positions[:, :, :, 1], 0)
        board = np.array(self.env.env.board).reshape(self.num_rows, self.num_cols)
        board_values = np.where(valid_positions, board[d0, d1], 0)
        a = (board_values == piece).astype(int)
        b = np.concatenate(
            (a, np.zeros_like(a[:, :, :1])),
            axis=-1,
        )  # padding with zeros to compute length
        lengths = np.argmin(b, -1)

        ended = False
        # check if winnable in any direction
        for both_dir in board_values:
            # |2x3|
            line = np.concatenate((both_dir[0][::-1], [piece], both_dir[1]))
            if "".join(map(str, [piece] * self.length)) in "".join(map(str, line)):
                ended = True
                break

        # ended = np.any(np.greater_equal(np.sum(lengths, 1), self.length - 1))
        draw = True
        for c, v in enumerate(self.top):
            draw &= (v == self.num_rows) if c != col else (v == (self.num_rows - 1))
        ended |= draw
        reward = (-1) ** (player) if ended and not draw else 0

        return (True, reward, ended) + ((lengths,) if return_length else ())


def agent_state(observation: dict) -> np.ndarray:
    """Player-perspective CHW float state from a Connect-Four observation.

    PettingZoo returns the observation from the *current* player's point of view,
    so a plain channel-move is enough for whoever is to move (no plane swap).

    :param observation: Raw PettingZoo observation dict.
    :type observation: dict
    :return: (channels, height, width) float32 array.
    :rtype: numpy.ndarray
    """
    return np.moveaxis(observation["observation"], -1, -3).astype(np.float32)


class ConnectFourVecEnv:
    """Vectorized self-play Connect Four as a single-agent MDP.

    ``num_envs`` games are stepped in lockstep with the opponent embedded in
    :meth:`step`, so one batched ``get_action`` drives them all. Buffers are
    preallocated and written in place, and terminated games auto-reset. A fixed
    ``num_envs`` keeps the agent's batch shapes static for CUDA graph capture.

    :param num_envs: Number of parallel games.
    :type num_envs: int
    :param lesson: Curriculum lesson settings (opponent, rewards, ...).
    :type lesson: dict
    :param opponent_policy: Frozen agent used as the opponent for self-play
        (``lesson['opponent'] == 'self'``); ``None`` for rule-based opponents.
    :type opponent_policy: DQN | None
    """

    def __init__(self, num_envs: int, lesson: dict, opponent_policy=None):
        self.num_envs = num_envs
        self.lesson = lesson
        self.opponent_policy = opponent_policy
        raw = connect_four_v3.env().observation_space("player_0")["observation"]
        self.single_observation_space = gym.spaces.Box(
            low=raw.low.transpose(2, 0, 1),
            high=raw.high.transpose(2, 0, 1),
            dtype=np.float32,
        )
        self.single_action_space = connect_four_v3.env().action_space("player_0")
        self.observations = np.zeros(
            (num_envs, *self.single_observation_space.shape), dtype=np.float32
        )
        self.masks = np.ones((num_envs, 7), dtype=np.int8)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.terminals = np.zeros(num_envs, dtype=bool)
        self.games = [
            CurriculumEnv(connect_four_v3.env(), lesson) for _ in range(num_envs)
        ]
        self.rule_opponents = (
            [Opponent(g, difficulty=lesson["opponent"]) for g in self.games]
            if lesson["opponent"] != "self"
            else None
        )

    def transition(
        self, prev_observations: np.ndarray, actions: np.ndarray
    ) -> TensorDict:
        """Build one batched replay transition from the pre-step obs and current buffers."""
        n = self.num_envs
        return TensorDict(
            {
                "obs": torch.from_numpy(prev_observations.copy()),
                "action": torch.from_numpy(np.asarray(actions, dtype=np.int64)).reshape(
                    n, 1
                ),
                "reward": torch.from_numpy(self.rewards.copy()).reshape(n, 1),
                "next_obs": torch.from_numpy(self.observations.copy()),
                "done": torch.from_numpy(self.terminals.astype(np.float32)).reshape(
                    n, 1
                ),
            },
            batch_size=[n],
        )

    def _write_obs(self, i: int) -> None:
        obs, _, _, _, _ = self.games[i].last()
        self.observations[i] = agent_state(obs)
        self.masks[i] = obs["action_mask"]

    def _reset_game(self, i: int) -> None:
        self.games[i].reset()
        self._write_obs(i)

    def reset(self, seed: int | None = None) -> None:
        """Reset every game and fill the observation/mask buffers in place."""
        for i in range(self.num_envs):
            self._reset_game(i)
        self.terminals[:] = False

    def _opponent_actions(self, pending: list[int]) -> list[int]:
        """Opponent (player_1) moves for the games in ``pending``."""
        if not pending:
            return []
        if self.opponent_policy is not None:
            # self-play: one batched greedy forward through the frozen opponent net
            obs = np.stack([agent_state(self.games[i].last()[0]) for i in pending])
            masks = np.stack([self.games[i].last()[0]["action_mask"] for i in pending])
            return list(
                self.opponent_policy.get_action(obs, epsilon=0.0, action_mask=masks)
            )
        # 'random' takes the action mask; 'weak'/'strong' introspect the board.
        if self.lesson["opponent"] == "random":
            block = self.lesson.get("block_vert_coef", 1)
            return [
                self.rule_opponents[i].get_action(
                    self.games[i].last()[0]["action_mask"], None, block
                )
                for i in pending
            ]
        return [self.rule_opponents[i].get_action(player=1) for i in pending]

    def step(self, actions: np.ndarray) -> None:
        """Apply agent actions, play the opponent's reply, write buffers, auto-reset."""
        pending = []
        for i, g in enumerate(self.games):
            g.step(int(actions[i]))
            _, _, done, trunc, _ = g.last()
            if done or trunc:
                self.rewards[i] = g.reward(done=True, player=0)
                self.terminals[i] = True
                self._reset_game(i)
            else:
                pending.append(i)
        opp_actions = self._opponent_actions(pending)
        for j, i in enumerate(pending):
            g = self.games[i]
            g.step(int(opp_actions[j]))
            _, _, done, trunc, _ = g.last()
            if done or trunc:
                self.rewards[i] = self.lesson["rewards"]["lose"]
                self.terminals[i] = True
                self._reset_game(i)
            else:
                self.rewards[i] = g.reward(done=False, player=0)
                self.terminals[i] = False
                self._write_obs(i)


@torch.no_grad()
def evaluate(agent, lesson, num_envs: int, n_games: int = 192) -> float:
    """Greedy win-rate vs the eval opponent, played on ``num_envs`` parallel games.

    Uses the same ``num_envs`` as training so the CUDA-graph-captured
    ``get_action`` always sees the same static batch shape.
    """
    eval_lesson = dict(lesson, opponent=lesson["eval_opponent"])
    venv = ConnectFourVecEnv(num_envs, eval_lesson)
    venv.reset()
    wins = done = 0
    while done < n_games:
        actions = agent.get_action(
            venv.observations, epsilon=0.0, action_mask=venv.masks
        )
        venv.step(np.asarray(actions))
        for i in range(num_envs):
            if venv.terminals[i]:
                wins += venv.rewards[i] >= lesson["rewards"]["win"] * 0.9
                done += 1
    return wins / max(done, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Curriculum Learning Demo (vectorized + CUDA graphs) =====")

    # Fixed so the agent's batch shapes stay static for CUDA graph capture.
    NUM_ENVS = 32
    # Replays a step's kernel launches as one call, removing the CPU dispatch
    # overhead that dominates small networks. Needs static shapes.
    USE_CUDAGRAPHS = device.type == "cuda"

    for lesson_number in range(1, 5):
        with open(f"./curriculums/connect_four/lesson{lesson_number}.yaml") as file:
            LESSON = yaml.safe_load(file)

        net_config = {
            "encoder_config": {
                "channel_size": [128],
                "kernel_size": [4],
                "stride_size": [1],
            },
            "head_config": {"hidden_size": [64, 64]},
        }
        init_hp = {
            "double": True,
            "batch_size": 256,  # fixed (not mutated) so the CUDA graph stays valid
            "lr": 1e-4,
            "gamma": 0.99,
            "learn_step": 1,
            "tau": 0.01,
        }
        population_size = 6
        memory_size = 10000

        probe = ConnectFourVecEnv(NUM_ENVS, LESSON)
        observation_space = probe.single_observation_space
        action_space = probe.single_action_space

        # batch_size is not mutated: it would invalidate the captured CUDA graph.
        hp_config = HyperparameterConfig(
            lr=RLParameter(min=1e-4, max=1e-2),
            learn_step=RLParameter(
                min=1, max=120, dtype=int, grow_factor=1.5, shrink_factor=0.75
            ),
        )

        pop = DQN.population(
            size=population_size,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            hp_config=hp_config,
            device=device,
            cudagraphs=USE_CUDAGRAPHS,
            **init_hp,
        )

        memory = ReplayBuffer(max_size=memory_size, device=device)
        tournament = TournamentSelection(
            tournament_size=2, elitism=True, population_size=population_size
        )
        mutations = Mutations(
            no_mutation=0.2,
            architecture=0,
            new_layer_prob=0.2,
            parameters=0.2,
            activation=0,
            rl_hp=0.2,
            mutation_sd=0.1,
            rand_seed=1,
            device=device,
        )

        # Training-loop parameters
        max_episodes = LESSON["max_train_episodes"]
        evo_epochs = 5  # evolve every N vectorized rollout blocks
        block_steps = 100  # batched steps per agent per block
        epsilon, eps_end, eps_decay = 1.0, 0.1, 0.9995

        if LESSON["pretrained_path"] is not None:
            for agent in pop:
                agent.load_checkpoint(LESSON["pretrained_path"])

        opponent_pool = None
        if LESSON["opponent"] == "self":
            opponent_pool = deque(maxlen=LESSON["opponent_pool_size"])
            for _ in range(LESSON["opponent_pool_size"]):
                opponent_pool.append(pop[0].clone())

        # Buffer + agent warm-up (vectorized random rollout)
        if LESSON["buffer_warm_up"]:
            warm_lesson = dict(LESSON, opponent=LESSON["warm_up_opponent"])
            warm_env = ConnectFourVecEnv(NUM_ENVS, warm_lesson)
            warm_env.reset()
            print("Filling replay buffer ...")
            while len(memory) < memory.max_size:
                acts = np.array(
                    [
                        random.choices(range(7), warm_env.masks[i])[0]
                        for i in range(NUM_ENVS)
                    ]
                )
                prev = warm_env.observations.copy()
                warm_env.step(acts)
                memory.add(warm_env.transition(prev, acts))
            if LESSON["agent_warm_up"] > 0:
                print("Warming up agents ...")
                for agent in pop:
                    for _ in range(LESSON["agent_warm_up"]):
                        agent.learn(memory.sample(agent.batch_size))

        if max_episodes > 0:
            wandb.init(
                project="AgileRL",
                name="{}-EvoHPO-{}-{}Opposition-CNN-{}".format(
                    "connect_four_v3",
                    "DQN",
                    LESSON["opponent"],
                    datetime.now(tz=timezone.utc).strftime("%m%d%Y%H%M%S"),
                ),
                config={
                    "algo": "Evo HPO DQN",
                    "env": "connect_four_v3",
                    "init_hp": init_hp,
                    "lesson": LESSON,
                },
            )

        elite = pop[0]
        total_steps = 0
        # Each step collects NUM_ENVS transitions.
        n_blocks = 0 if max_episodes == 0 else 25

        pbar = trange(n_blocks)
        for block in pbar:
            for agent in pop:
                opp = (
                    random.choice(opponent_pool) if opponent_pool is not None else None
                )
                venv = ConnectFourVecEnv(NUM_ENVS, LESSON, opponent_policy=opp)
                venv.reset()
                for agent_step in range(block_steps):
                    prev = venv.observations.copy()
                    actions = agent.get_action(venv.observations, epsilon, venv.masks)
                    venv.step(np.asarray(actions))
                    memory.add(venv.transition(prev, np.asarray(actions)))
                    if (
                        len(memory) >= agent.batch_size
                        and agent_step % agent.learn_step == 0
                    ):
                        agent.learn(memory.sample(agent.batch_size))
                    total_steps += NUM_ENVS
                epsilon = max(eps_end, epsilon * eps_decay)

            # Self-play: refresh the opponent pool with the current elite
            if opponent_pool is not None and (block + 1) % evo_epochs == 0:
                elite_opp, _, _ = tournament._elitism(pop)
                opponent_pool.append(elite_opp.clone())

            if (block + 1) % evo_epochs == 0:
                fitnesses = [evaluate(agent, LESSON, NUM_ENVS) for agent in pop]
                for agent, fit in zip(pop, fitnesses):
                    agent.metrics.add_fitness(fit)
                elite, pop = tournament.select(pop)
                pop = mutations.mutation(pop)
                pbar.set_postfix_str(
                    f"Lesson {lesson_number}  Eval win-rate (best): {max(fitnesses):.2f}  "
                    f"Total steps: {total_steps}"
                )

        if max_episodes > 0:
            wandb.finish()

        save_path = LESSON["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        elite.save_checkpoint(save_path)
        print(f"Elite agent saved to '{save_path}'.")
