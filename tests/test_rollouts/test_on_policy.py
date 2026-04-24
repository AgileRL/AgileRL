import pytest
import torch
import numpy as np


from agilerl.rollouts.on_policy import collect_rollouts_llm
from agilerl.llm_envs import (
    SyncMultiTurnVecEnv,
    TokenObservationWrapper,
)
from agilerl.algorithms.ppo import PPO
from agilerl.rollouts.on_policy import (
    _collect_rollouts,
    collect_rollouts,
    collect_rollouts_recurrent,
)


from tests.test_algorithms.test_llms.test_ppo_llm import _cpu_llmppo
from tests.test_algorithms.test_llms.test_reinforce_llm import _cpu_llmreinforce


pytestmark = pytest.mark.llm


class _TinyTokenizer:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        tokens = [((ord(ch) % 50) + 1) for ch in text][:16]
        return tokens or [1]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for token in token_ids:
            token_id = int(token)
            if skip_special_tokens and token_id == self.pad_token_id:
                continue
            chars.append(chr(((token_id - 1) % 26) + 97))
        return "".join(chars)

    def __call__(
        self,
        texts,
        return_tensors: str = "pt",
        padding: bool = True,
        padding_side: str = "left",
        return_attention_mask: bool = True,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, return_attention_mask
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text) for text in texts]
        max_len = max(len(item) for item in encoded) if padding else None
        padded_ids = []
        padded_masks = []
        for item in encoded:
            if max_len is None:
                padded_ids.append(item)
                padded_masks.append([1] * len(item))
                continue
            pad = max_len - len(item)
            if padding_side == "left":
                ids = [self.pad_token_id] * pad + item
                mask = [0] * pad + [1] * len(item)
            else:
                ids = item + [self.pad_token_id] * pad
                mask = [1] * len(item) + [0] * pad
            padded_ids.append(ids)
            padded_masks.append(mask)
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }


class _SingleTurnTextEnv:
    def __init__(self):
        self._done = False

    def reset(self, seed=None):
        del seed
        self._done = False
        return "solve 1 + 1", {}

    def step(self, action):
        del action
        self._done = True
        return "correct", 1.0, True, False, {}

    def close(self):
        return None


@pytest.mark.parametrize("hf_generate_chunk_size", [1, 2, 4])
@pytest.mark.parametrize("algo_name", ["ppo", "reinforce"])
def test_collect_rollouts_llm_hf_chunk_sizes_in_process(
    hf_generate_chunk_size: int, algo_name: str
):
    tokenizer = _TinyTokenizer()

    def env_fn():
        return TokenObservationWrapper(
            _SingleTurnTextEnv(),
            tokenizer=tokenizer,
            max_turns=1,
            pad_id=tokenizer.pad_token_id,
            apply_chat_template=False,
            max_model_len=128,
            max_output_tokens=8,
        )

    env = SyncMultiTurnVecEnv(env_factory=env_fn, batch_size=2, group_size=1)
    if algo_name == "ppo":
        agent = _cpu_llmppo(
            use_vllm=False,
            hf_generate_chunk_size=hf_generate_chunk_size,
            max_model_len=128,
            max_output_tokens=8,
        )
    else:
        agent = _cpu_llmreinforce(
            use_vllm=False,
            hf_generate_chunk_size=hf_generate_chunk_size,
            max_model_len=128,
            max_output_tokens=8,
        )

    experiences, masks, turns, rewards, steps, next_group_seed = collect_rollouts_llm(
        agent=agent,
        env=env,
        n_steps=1,
        batch_size=2,
        group_seed=0,
    )
    assert len(experiences) == 2
    assert len(masks) == 2
    assert len(rewards) == 2
    assert len(turns) == 2
    assert steps > 0
    assert next_group_seed == 2

    agent.clean_up()
    env.close()


def test_collect_rollouts_llm_preserves_batch_group_ordering_batch_size_4() -> None:
    """Preserve (batch_idx, group_idx) ordering across LLM rollout collection."""
    prompt_tokens_by_env_index = [42, 7, 99, 13, 55, 21, 88, 3]
    creation_idx = {"value": 0}

    class _OrderingEnv:
        """Minimal env that records which completion token it receives."""

        def __init__(self, prompt_token: int) -> None:
            self.prompt_token = prompt_token
            self._seen_token: int | None = None
            self.turn_boundaries: list[tuple[int, int, int]] = []

        def reset(
            self,
            seed: int | None = None,
        ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
            """Return a one-token prompt that uniquely marks this trajectory."""
            del seed
            return {
                "input_ids": torch.tensor([[self.prompt_token]], dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            }, {}

        def step(
            self,
            full_completion: torch.Tensor,
        ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, object]]:
            """Record completion identity and terminate after one turn."""
            self._seen_token = int(full_completion[0, 0].item())
            self.turn_boundaries = [(0, 1, 0)]
            return {}, 1.0, True, False, {}

        def get_episode_data(
            self,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Expose marker token via episode ids and rewards for ordering checks."""
            token = self._seen_token if self._seen_token is not None else -1
            ep_ids = torch.tensor([[token, token + 1]], dtype=torch.long)
            action_mask = torch.tensor([[True]], dtype=torch.bool)
            turn_ids = torch.tensor([[0]], dtype=torch.long)
            rewards = torch.tensor([float(token)], dtype=torch.float32)
            return ep_ids, action_mask, turn_ids, rewards

        def close(self) -> None:
            """Provide a close method compatible with vector env cleanup."""
            return None

    class _EchoAgent:
        """Echo prompt marker tokens into completions in row order."""

        def get_action(
            self, prompts: dict[str, torch.Tensor], training: bool = True
        ) -> tuple[list[torch.Tensor], None]:
            """Return one completion per prompt row while preserving input order."""
            del training
            row_tokens = prompts["input_ids"][:, 0].tolist()
            completions = [
                torch.tensor([[int(tok), int(tok) + 1]], dtype=torch.long)
                for tok in row_tokens
            ]
            return completions, None

    def env_fn() -> _OrderingEnv:
        """Create deterministic env instances in construction order."""
        idx = creation_idx["value"]
        creation_idx["value"] += 1
        return _OrderingEnv(prompt_tokens_by_env_index[idx])

    env = SyncMultiTurnVecEnv(env_factory=env_fn, batch_size=4, group_size=2)
    agent = _EchoAgent()

    completion_ids_list, _masks, _turns, rewards, steps, next_group_seed = (
        collect_rollouts_llm(
            agent=agent,
            env=env,
            n_steps=1,
            batch_size=4,
            group_size=2,
            group_seed=123,
        )
    )

    returned_first_tokens = [int(ids[0, 0].item()) for ids in completion_ids_list]
    assert returned_first_tokens == prompt_tokens_by_env_index

    returned_reward_markers = [int(r[0].item()) for r in rewards]
    assert returned_reward_markers == prompt_tokens_by_env_index

    assert steps == 8
    assert next_group_seed == 127

    env.close()


class DummyEnv:
    def __init__(self, state_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.vect = vect
        self.num_envs = num_envs
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.n_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size), {}

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            {},
        )


def test_collect_rollouts_use_rollout_buffer_false_raises(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=False,
        learn_step=5,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    with pytest.raises(RuntimeError, match="use_rollout_buffer=True"):
        collect_rollouts(ppo, env, n_steps=5)
    ppo.clean_up()


def test_collect_rollouts_returns_scores(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    result = collect_rollouts(ppo, env, n_steps=4)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert isinstance(result[0], list)
    ppo.clean_up()


def test_collect_rollouts_recurrent_returns_scores(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
        recurrent=True,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    result = collect_rollouts_recurrent(ppo, env, n_steps=4)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert isinstance(result[0], list)
    ppo.clean_up()


def test_collect_rollouts_warm_start_with_last_obs(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    obs, info = env.reset()
    result = _collect_rollouts(
        ppo,
        env,
        n_steps=2,
        last_obs=obs,
        last_done=np.zeros(1),
        last_scores=np.zeros(1),
        last_info=info,
        recurrent=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 5
    ppo.clean_up()


def test_collect_rollouts_n_steps_default(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=8,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    result = collect_rollouts(ppo, env)
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    ppo.clean_up()


@pytest.mark.parametrize("recurrent", [False, True])
def test_collect_rollouts_recurrent_warm_start(vector_space, discrete_space, recurrent):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
        recurrent=recurrent,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    obs, info = env.reset()
    if recurrent:
        ppo.hidden_state = ppo.get_initial_hidden_state(1)
    result = _collect_rollouts(
        ppo,
        env,
        n_steps=2,
        last_obs=obs,
        last_done=np.zeros(1),
        last_scores=np.zeros(1),
        last_info=info,
        recurrent=recurrent,
    )
    assert isinstance(result, tuple)
    assert len(result) == 5
    ppo.clean_up()
