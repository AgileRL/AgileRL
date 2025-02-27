import pytest
import torch

from agilerl.utils.llm_utils import HuggingFaceGym


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 1000

    def batch_decode(self, *args, **kwargs):
        return ["This is a test completion."]

    def apply_chat_template(self, *args, **kwargs):
        return "This is a test completion."

    def __call__(self, *args, **kwargs):
        return torch.tensor([1, 2, 3, 4, 5])


@pytest.fixture
def hugging_face_gym(dataset_name):
    tokenizer = DummyTokenizer()
    return HuggingFaceGym(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        reward_fn=lambda *args, **kwargs: 1,
    )


def test_hugging_face_gym_init():
    dataset_name = "gsm8k"
    tokenizer = DummyTokenizer()

    def reward_fn(*args, **kwargs):
        return 1

    env = HuggingFaceGym(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
    )
    assert env.name == dataset_name
    assert env.reward_fn == reward_fn
    assert env.tokenizer == tokenizer
