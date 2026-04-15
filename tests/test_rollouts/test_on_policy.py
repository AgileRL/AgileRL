import pytest
import torch

from agilerl.rollouts.on_policy import collect_rollouts_llm
from agilerl.wrappers.multiturn_wrappers import SyncMultiTurnVecEnv, TokenObservationWrapper
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
