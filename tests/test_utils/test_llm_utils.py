from contextlib import contextmanager
import sys

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("datasets", reason="LLM dependencies not installed")

from datasets import Dataset as Datasets
from torch import nn
from transformers import AutoTokenizer
from accelerate.state import AcceleratorState
from accelerate import Accelerator
from torch.utils.data import DataLoader
from agilerl.utils.algo_utils import DummyOptimizer
from agilerl.utils.llm_utils import (
    PreferenceGym,
    ReasoningGym,
    _auto_zero_stage,
    align_deepspeed_lr,
    create_llm_accelerator,
    get_model_name_or_path,
    gather_if_zero3,
    get_state_dict,
    masked_mean,
    masked_var,
    move_params_to_cpu,
    move_params_to_gpu,
    normalize_reasoning_prompt_batch,
    pool_by_turns,
    stitch_completion_after_windowed_hf_generate,
    stitch_completion_after_windowed_vllm_generate,
    compare_responses,
    gather_if_zero3,
    get_state_dict,
    sample_eval_prompts,
)

pytestmark = pytest.mark.llm

DUMMY_CONVERSATION_TEMPLATE = [
    {
        "role": "system",
        "content": "question: {question}\nanswer: {answer}",
    },
]


def test_stitch_completion_after_windowed_hf_generate_no_stitch_passthrough():
    completion_id = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    out, full_prompt_len = stitch_completion_after_windowed_hf_generate(
        completion_id=completion_id,
        stitch=None,
        initial_len=2,
    )
    assert torch.equal(out, completion_id)
    assert full_prompt_len == 2


def test_stitch_completion_after_windowed_hf_generate_basic_stitch_insertion():
    completion_id = torch.tensor([[1, 2, 7, 8]], dtype=torch.long)
    stitch = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    out, full_prompt_len = stitch_completion_after_windowed_hf_generate(
        completion_id=completion_id,
        stitch=stitch,
        initial_len=2,
    )
    expected = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    assert torch.equal(out, expected)
    assert full_prompt_len == 6


def test_stitch_completion_after_windowed_hf_generate_output_stays_on_completion_device():
    completion_id = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    stitch = torch.tensor([[9, 10]], dtype=torch.long)
    out, _ = stitch_completion_after_windowed_hf_generate(
        completion_id=completion_id,
        stitch=stitch,
        initial_len=2,
    )
    assert out.device == completion_id.device


def test_stitch_completion_after_windowed_hf_generate_empty_stitch_tensor_keeps_sequence():
    completion_id = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    stitch = torch.empty((1, 0), dtype=torch.long)
    out, full_prompt_len = stitch_completion_after_windowed_hf_generate(
        completion_id=completion_id,
        stitch=stitch,
        initial_len=2,
    )
    assert torch.equal(out, completion_id)
    assert full_prompt_len == 2


def test_stitch_completion_after_windowed_vllm_generate_rejects_group_size_not_one():
    with pytest.raises(ValueError, match="only implemented for group_size=1"):
        stitch_completion_after_windowed_vllm_generate(
            completion_ids=[torch.tensor([[1, 2, 3]], dtype=torch.long)],
            stitch_prefixes=[torch.tensor([[9]], dtype=torch.long)],
            group_prompts=[{"initial_prompt_len": 1}],
            group_size=2,
            prompts=[{"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}],
        )


def test_stitch_completion_after_windowed_vllm_generate_selective_stitching_per_prompt():
    completion_ids = [
        torch.tensor([[1, 2, 7]], dtype=torch.long),
        torch.tensor([[4, 5, 6]], dtype=torch.long),
    ]
    stitch_prefixes = [
        torch.tensor([[9, 10]], dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
    ]
    group_prompts = [{"initial_prompt_len": 2}, {"initial_prompt_len": 1}]
    prompts = [{}, {}]
    out = stitch_completion_after_windowed_vllm_generate(
        completion_ids=completion_ids,
        stitch_prefixes=stitch_prefixes,
        group_prompts=group_prompts,
        group_size=1,
        prompts=prompts,
    )
    assert torch.equal(out[0], torch.tensor([[1, 2, 9, 10, 7]], dtype=torch.long))
    assert torch.equal(out[1], completion_ids[1])


def test_stitch_completion_after_windowed_vllm_generate_inserts_at_initial_prompt_len():
    completion_ids = [torch.tensor([[10, 11, 12, 13]], dtype=torch.long)]
    stitch_prefixes = [torch.tensor([[99, 98]], dtype=torch.long)]
    group_prompts = [{"initial_prompt_len": 1}]
    out = stitch_completion_after_windowed_vllm_generate(
        completion_ids=completion_ids,
        stitch_prefixes=stitch_prefixes,
        group_prompts=group_prompts,
        group_size=1,
        prompts=[{}],
    )
    assert torch.equal(out[0], torch.tensor([[10, 99, 98, 11, 12, 13]]))


def test_stitch_completion_after_windowed_vllm_generate_broadcasts_single_stitch_row_across_group_rows():
    completion_ids = [torch.tensor([[1, 2, 7], [3, 4, 8]], dtype=torch.long)]
    stitch_prefixes = [torch.tensor([[9, 10]], dtype=torch.long)]
    group_prompts = [{"initial_prompt_len": 2}]
    out = stitch_completion_after_windowed_vllm_generate(
        completion_ids=completion_ids,
        stitch_prefixes=stitch_prefixes,
        group_prompts=group_prompts,
        group_size=1,
        prompts=[{}],
    )
    expected = torch.tensor([[1, 2, 9, 10, 7], [3, 4, 9, 10, 8]], dtype=torch.long)
    assert torch.equal(out[0], expected)


def test_stitch_completion_after_windowed_vllm_generate_raises_when_initial_prompt_len_missing_with_non_empty_stitch():
    with pytest.raises(
        ValueError,
        match="initial_prompt_len required when stitch_prefix_ids is non-empty",
    ):
        stitch_completion_after_windowed_vllm_generate(
            completion_ids=[torch.tensor([[1, 2, 3]], dtype=torch.long)],
            stitch_prefixes=[torch.tensor([[9]], dtype=torch.long)],
            group_prompts=[{}],
            group_size=1,
            prompts=[{}],
        )


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 1000

    def batch_decode(self, *args, **kwargs):
        return ["This is a test completion."]

    def apply_chat_template(self, *args, **kwargs):
        return "This is a test completion."

    def __call__(self, *args, **kwargs):
        return torch.tensor([1, 2, 3, 4, 5])


class Info:
    def __init__(self, name):
        self.dataset_name = name


class DummyReasoningDataset:
    def __init__(self, num_samples):
        # Create dummy questions and answers
        self.questions = [f"This is question {i}?" for i in range(num_samples)]
        self.answers = [f"This is answer {i}." for i in range(num_samples)]
        self.features = {"question": self.questions, "answer": self.answers}
        self.info = Info("dummy_dataset")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return {"question": self.questions[index], "answer": self.answers[index]}

    def filter(self, fn):
        keep_indices = [
            i
            for i in range(len(self))
            if fn({"question": self.questions[i], "answer": self.answers[i]})
        ]
        filtered = DummyReasoningDataset(0)
        filtered.questions = [self.questions[i] for i in keep_indices]
        filtered.answers = [self.answers[i] for i in keep_indices]
        filtered.features = {"question": filtered.questions, "answer": filtered.answers}
        filtered.info = self.info
        return filtered


class DummyPreferenceDataset:
    def __init__(self, num_samples):
        self.prompt = [f"This is prompt {i}." for i in range(num_samples)]
        self.chosen = [f"This is chosen {i}." for i in range(num_samples)]
        self.rejected = [f"This is rejected {i}." for i in range(num_samples)]
        self.features = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        self.info = Info("dummy_dataset")

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, index):
        return {
            "prompt": self.prompt[index],
            "chosen": self.chosen[index],
            "rejected": self.rejected[index],
        }

    def filter(self, fn):
        keep_indices = [
            i
            for i in range(len(self))
            if fn(
                {
                    "prompt": self.prompt[i],
                    "chosen": self.chosen[i],
                    "rejected": self.rejected[i],
                }
            )
        ]
        filtered = DummyPreferenceDataset(0)
        filtered.prompt = [self.prompt[i] for i in keep_indices]
        filtered.chosen = [self.chosen[i] for i in keep_indices]
        filtered.rejected = [self.rejected[i] for i in keep_indices]
        filtered.features = {
            "prompt": filtered.prompt,
            "chosen": filtered.chosen,
            "rejected": filtered.rejected,
        }
        filtered.info = self.info
        return filtered


def dummy_reward_fn(*args, **kwargs):
    return 1.0


def dummy_chat_template_fn_custom(q, a, tokenizer):
    """Chat template function for test_reasoning_gym_reset_dataloaders, gives unique input_ids for each question so
    we can test equality.
    """
    index = int(q.split(" ")[-1][0])
    return {
        "input_ids": torch.tensor([index]),
        "attention_mask": torch.ones(1),
    }


def dummy_chat_template_fn(q, a, tokenizer):
    return {
        "input_ids": torch.randint(0, 1000, (1, 356)),
        "attention_mask": torch.ones(1, 356),
    }


@pytest.fixture(scope="function")
def accelerator_factory():
    def generate_accelerator(use_accelerator):
        AcceleratorState._reset_state(True)
        return Accelerator() if use_accelerator else None

    return generate_accelerator


@pytest.fixture
def reasoning_dataset(num_samples):
    train_dataset = DummyReasoningDataset(int(num_samples * 0.8))
    test_dataset = DummyReasoningDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


@pytest.fixture
def preference_dataset(num_samples):
    train_dataset = DummyPreferenceDataset(int(num_samples * 0.8))
    test_dataset = DummyPreferenceDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("use_accelerator", [True, False])
def test_reasoning_gym_init(
    reasoning_dataset,
    accelerator_factory,
    num_samples,
    use_accelerator,
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=accelerator_factory(use_accelerator),
    )
    assert env.name == "dummy_dataset"
    assert callable(env.reward_fn)
    assert hasattr(env, "tokenizer")
    assert env.tokenizer is not None
    assert isinstance(env.train_dataloader, DataLoader)
    assert isinstance(env.test_dataloader, DataLoader)
    assert list(next(env.train_dataloader_iter).keys()) == [
        "question",
        "answer",
        "tokenized_prompts",
    ]
    assert env.dataloader == env.train_dataloader_iter
    assert not env.reset_called
    assert not env.evaluation_mode
    assert env.data_batch_size_per_gpu == data_batch_size


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("eval_mode", [True, False])
def test_reasoning_gym_step(
    reasoning_dataset,
    num_samples,
    eval_mode,
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    env.evaluation_mode = eval_mode
    env.reset()
    completions = [torch.randint(0, 1000, (10, 356)) for _ in range(data_batch_size)]
    tokenized_prompts, rewards = env.step(completions)
    assert isinstance(tokenized_prompts, list)
    assert isinstance(rewards, torch.Tensor)
    assert len(tokenized_prompts) > 0
    assert isinstance(tokenized_prompts[0]["input_ids"], torch.Tensor)
    assert isinstance(tokenized_prompts[0]["attention_mask"], torch.Tensor)


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("reset_dataloaders", [True, False])
def test_reasoning_gym_reset(
    reasoning_dataset,
    num_samples,
    reset_dataloaders,
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    tokenized_prompts = env.reset(reset_dataloaders)
    assert isinstance(tokenized_prompts, list)
    assert len(tokenized_prompts) > 0
    assert isinstance(tokenized_prompts[0]["input_ids"], torch.Tensor)
    assert isinstance(tokenized_prompts[0]["attention_mask"], torch.Tensor)


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("reset_dataloaders", [True, False])
def test_reasoning_gym_reset_dataloaders(
    reasoning_dataset,
    num_samples,
    reset_dataloaders,
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    first_data_point = next(
        env.test_dataloader_iter,
    )  # use test_dataloader_iter as it is not shuffled
    env._reset_dataloaders()
    first_data_point_reset = next(env.test_dataloader_iter)
    assert first_data_point["question"] == first_data_point_reset["question"]
    assert first_data_point["answer"] == first_data_point_reset["answer"]
    for prompt_a, prompt_b in zip(
        first_data_point["tokenized_prompts"],
        first_data_point_reset["tokenized_prompts"],
        strict=False,
    ):
        assert torch.equal(prompt_a["input_ids"], prompt_b["input_ids"])
        assert torch.equal(prompt_a["attention_mask"], prompt_b["attention_mask"])


@pytest.mark.parametrize("num_samples", [200])
def test_reset_warning(reasoning_dataset, num_samples):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    with pytest.warns():
        env.reset()
        env.reset()


@pytest.mark.parametrize("num_samples", [200])
def test_reasoning_gym_len(reasoning_dataset, num_samples):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    env.reset()
    assert len(env) == 200 * 0.8  # Length returns the training length
    with env.eval_mode():
        assert len(env) == 200 * 0.2


@pytest.mark.parametrize("num_samples", [20])
def test_create_chat_collate_fn(reasoning_dataset, num_samples):
    """Test the create_chat_collate_fn method."""
    # Create a mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8

    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )

    # Create the collate function
    collate_fn = env.create_collate_fn(tokenizer)

    # Create a sample batch
    batch = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 3+3?", "answer": "6"},
    ]

    # Apply the collate function
    result = collate_fn(batch)

    # Verify the result structure
    assert isinstance(result, dict)
    assert "question" in result
    assert "answer" in result
    assert "tokenized_prompts" in result

    # Verify the content
    assert result["question"] == ["What is 2+2?", "What is 3+3?"]
    assert result["answer"] == ["4", "6"]
    assert len(result["tokenized_prompts"]) == 2
    assert isinstance(result["tokenized_prompts"][0]["input_ids"], torch.Tensor)
    assert isinstance(result["tokenized_prompts"][0]["attention_mask"], torch.Tensor)


@pytest.mark.parametrize("num_samples", [20])
@pytest.mark.parametrize("data_batch_size", [8, 10])
def test_reset_dataloaders_when_train_dataloader_exhausted(
    reasoning_dataset,
    num_samples,
    data_batch_size,
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    total_sampled = 0
    for _ in range(3):
        env._get_next_batch()
        total_sampled += data_batch_size

    assert env.num_epochs == 1


@pytest.mark.parametrize("num_samples", [20])
@pytest.mark.parametrize("data_batch_size", [8, 10])
def test_not_reset_dataloaders_when_test_dataloader_exhausted(
    reasoning_dataset,
    num_samples,
    data_batch_size,
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    total_sampled = 0
    env.reset()
    for _ in range(10):
        with env.eval_mode():
            env._get_next_batch()
            total_sampled += data_batch_size

    assert env.num_epochs == 0


def test_dummy_optimizer_init():
    """Test DummyOptimizer initialization."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr=lr)
    assert optimizer is not None


def test_dummy_optimizer_step():
    """Test DummyOptimizer step method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr=lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.step()

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


def test_dummy_optimizer_zero_grad():
    """Test DummyOptimizer zero_grad method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr=lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.zero_grad()

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


def test_dummy_optimizer_state_dict():
    """Test DummyOptimizer state_dict method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr=lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.state_dict()

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


def test_dummy_optimizer_load_state_dict():
    """Test DummyOptimizer load_state_dict method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr=lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.load_state_dict({})

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


@pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
def test_gather_if_zero3(zero_stage):
    """Test gather_if_zero3 context manager."""
    params = [torch.tensor([1.0, 2.0, 3.0])]

    @contextmanager
    def dummy_gather_parameters(*args, **kwargs):
        yield

    with (
        patch(
            "deepspeed.zero.GatheredParameters",
            side_effect=dummy_gather_parameters,
        ) as mock_gathered_parameters,
        gather_if_zero3(zero_stage, params),
    ):
        assert mock_gathered_parameters.call_count == (zero_stage == 3)


def test_get_state_dict():
    model = nn.Linear(10, 10)
    state_dict = get_state_dict(model)
    assert isinstance(state_dict, dict)
    for key, value in state_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)


def _make_tokenizer(vocab_size: int = 100, prompt_len: int = 3) -> MagicMock:
    """Return a mock tokenizer compatible with compare_responses."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    # tokenizer(text, return_tensors="pt") → encoding with .to(device)
    encoding = MagicMock()
    encoding.__getitem__ = lambda self, key: (
        torch.zeros(1, prompt_len, dtype=torch.long)
        if key == "input_ids"
        else torch.ones(1, prompt_len, dtype=torch.long)
    )
    encoding.to.return_value = {
        "input_ids": torch.zeros(1, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(1, prompt_len, dtype=torch.long),
    }
    tokenizer.return_value = encoding
    tokenizer.decode.return_value = "decoded response"
    return tokenizer


def _make_agent(has_adapter: bool, device: str = "cpu") -> MagicMock:
    """Return a mock agent with actor and device attributes."""
    agent = MagicMock()
    agent.device = device
    model = MagicMock()
    model.generate.return_value = torch.zeros(1, 5, dtype=torch.long)
    if has_adapter:
        # disable_adapter() must work as a context manager
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=None)
        cm.__exit__ = MagicMock(return_value=False)
        model.disable_adapter = MagicMock(return_value=cm)
    else:
        del model.disable_adapter  # hasattr() returns False
    agent.actor = model
    return agent


def test_compare_responses_no_adapter_with_reference(capsys):
    """Without an adapter only the model response section is printed; reference is shown."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [("What is 2+2?", "It is 4.", "It is 5.")]

    compare_responses(agent, tokenizer, samples)

    captured = capsys.readouterr().out
    assert "PROMPT" in captured
    assert "DATASET RESPONSE (CHOSEN)" in captured
    assert "DATASET RESPONSE (REJECTED)" in captured
    assert "MODEL RESPONSE" in captured
    assert "BASE MODEL" not in captured
    assert "FINE-TUNED MODEL" not in captured
    # generate called exactly once (no base model pass)
    assert agent.actor.generate.call_count == 1


def test_compare_responses_no_adapter_no_reference(capsys):
    """When reference is None the DATASET RESPONSE section is skipped."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [("What is 2+2?", None, None)]

    compare_responses(agent, tokenizer, samples)

    captured = capsys.readouterr().out
    assert "PROMPT" in captured
    assert "DATASET RESPONSE" not in captured
    assert "MODEL RESPONSE" in captured


def test_compare_responses_with_adapter_shows_base_and_finetuned(capsys):
    """With an adapter both BASE MODEL and FINE-TUNED MODEL sections are printed."""
    agent = _make_agent(has_adapter=True)
    tokenizer = _make_tokenizer()
    samples = [
        (
            "Tell me a joke.",
            "Why did the chicken cross the road?",
            "To get to the bar.",
        ),
    ]

    compare_responses(agent, tokenizer, samples)

    captured = capsys.readouterr().out
    assert "DATASET RESPONSE (REJECTED)" in captured
    assert "BASE MODEL" in captured
    assert "FINE-TUNED MODEL" in captured
    assert "MODEL RESPONSE" not in captured
    # generate called twice: once inside disable_adapter, once without
    assert agent.actor.generate.call_count == 2
    agent.actor.disable_adapter.assert_called_once()


def test_compare_responses_multiple_samples_enter_continues(capsys):
    """Pressing Enter (empty string) advances to the next sample."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [
        ("Q1", "A1", None),
        ("Q2", "A2", "rejected for Q2"),
        ("Q3", "A3", None),
    ]

    with patch("builtins.input", return_value=""):
        compare_responses(agent, tokenizer, samples)

    captured = capsys.readouterr().out
    assert "DATASET RESPONSE (REJECTED)" in captured
    # Navigation prompt appears between samples (not after the last one)
    assert captured.count("[Enter] next sample") == len(samples) - 1
    # All three samples were generated
    assert agent.actor.generate.call_count == len(samples)


def test_compare_responses_quit_early(capsys):
    """Pressing 'q' stops processing remaining samples."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [("Q1", "A1", None), ("Q2", "A2", None), ("Q3", "A3", None)]

    with patch("builtins.input", return_value="q"):
        compare_responses(agent, tokenizer, samples)

    # Only the first sample's generation runs; loop exits before Q2 and Q3
    assert agent.actor.generate.call_count == 1


def test_compare_responses_eof_breaks_loop(capsys):
    """An EOFError from input() (non-interactive environment) stops the loop gracefully."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [("Q1", "A1", None), ("Q2", "A2", None)]

    with patch("builtins.input", side_effect=EOFError):
        compare_responses(agent, tokenizer, samples)

    # Only the first sample is generated; EOFError prevents further iteration
    assert agent.actor.generate.call_count == 1


def test_compare_responses_single_sample_no_input_prompt(capsys):
    """With a single sample the navigation prompt and input() are never shown/called."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [("Only prompt", "Only response", None)]

    with patch("builtins.input") as mock_input:
        compare_responses(agent, tokenizer, samples)

    mock_input.assert_not_called()


@pytest.mark.parametrize("do_sample,temperature", [(False, 1.0), (True, 0.7)])
def test_compare_responses_generation_kwargs_forwarded(do_sample, temperature):
    """do_sample and temperature are forwarded to model.generate."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [("prompt", None, None)]

    compare_responses(
        agent,
        tokenizer,
        samples,
        max_new_tokens=50,
        temperature=temperature,
        do_sample=do_sample,
    )

    _, call_kwargs = agent.actor.generate.call_args
    assert call_kwargs["max_new_tokens"] == 50
    assert call_kwargs["temperature"] == temperature
    assert call_kwargs["do_sample"] == do_sample


def test_compare_responses_skip_special_tokens_forwarded():
    """skip_special_tokens is forwarded to tokenizer.decode."""
    agent = _make_agent(has_adapter=False)
    tokenizer = _make_tokenizer()
    samples = [("prompt", None, None)]

    compare_responses(agent, tokenizer, samples, skip_special_tokens=False)

    _, decode_kwargs = tokenizer.decode.call_args
    assert decode_kwargs["skip_special_tokens"] is False


def test_sample_eval_prompts_sft_style_response_column():
    """Covers SFTGym-style envs that expose ``response_column``."""
    from types import SimpleNamespace

    ds = Datasets.from_dict(
        {"prompt": ["p0", "p1"], "response": ["r0", "r1"]},
    )
    env = SimpleNamespace(
        response_column="response",
        test_dataloader=SimpleNamespace(dataset=ds),
    )
    rows = sample_eval_prompts(env, n=2, seed=0)
    assert len(rows) == 2
    assert {rows[0][0], rows[1][0]} == {"p0", "p1"}
    assert all(r[2] is None for r in rows)


def test_sample_eval_prompts_preference_style_chosen_rejected():
    """Covers PreferenceGym-style datasets with ``chosen`` / ``rejected`` columns."""
    from types import SimpleNamespace

    ds = Datasets.from_dict(
        {
            "prompt": ["p0", "p1"],
            "chosen": ["c0", "c1"],
            "rejected": ["x0", "x1"],
        },
    )
    env = SimpleNamespace(test_dataloader=SimpleNamespace(dataset=ds))
    rows = sample_eval_prompts(env, n=2, seed=0)
    assert len(rows) == 2
    prompts = {r[0] for r in rows}
    assert prompts == {"p0", "p1"}
    for p, c, r in rows:
        if p == "p0":
            assert (c, r) == ("c0", "x0")
        else:
            assert (c, r) == ("c1", "x1")


def test_preference_gym_max_context_length_warning():
    train_dataset = Datasets.from_dict(
        {
            "prompt": [
                "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                "This is a prompt that is shorter.",
            ],
            "chosen": ["This is an answer.", "This is an answer."],
            "rejected": ["This is an answer.", "This is an answer."],
        },
    )
    test_dataset = Datasets.from_dict(
        {
            "prompt": ["This is a normal length prompt"],
            "chosen": ["This is an answer."],
            "rejected": ["This is an answer."],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    with pytest.warns(
        UserWarning,
        match=r"1 samples were filtered out of the train dataset due to the max context length constraint.",
    ):
        env = PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            max_context_length=10,
            min_completion_length=1,
        )
    assert len(env.train_dataloader) == 1
    assert len(env.test_dataloader) == 1


def test_reasoning_gym_max_context_length_warning():
    train_dataset = Datasets.from_dict(
        {
            "question": [
                "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                "This is a prompt that is shorter.",
            ],
            "answer": ["This is an answer.", "This is an answer."],
        },
    )
    test_dataset = Datasets.from_dict(
        {
            "question": ["This is a normal length prompt"],
            "answer": ["This is an answer."],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    with pytest.warns(
        UserWarning,
        match=r"1 samples were filtered out of the train dataset due to the max context length constraint.",
    ):
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
            max_context_length=10,
        )
    assert len(env.train_dataloader) == 1
    assert len(env.test_dataloader) == 1


def test_llm_utils_fallback_types_when_no_llm_dependencies():
    """Test that llm_utils sets type aliases to Any when HAS_LLM_DEPENDENCIES is False."""
    # Remove the module from cache to force reimport
    original_module = sys.modules.pop("agilerl.utils.llm_utils", None)

    try:
        # Patch HAS_LLM_DEPENDENCIES before reimporting
        with patch("agilerl.HAS_LLM_DEPENDENCIES", False):
            # Reimport the module - it will see HAS_LLM_DEPENDENCIES as False
            import agilerl.utils.llm_utils as llm_utils_reloaded

            # Verify the fallback type aliases are set to Any
            assert llm_utils_reloaded.AutoTokenizer is Any
            assert llm_utils_reloaded.PreTrainedModel is Any
            assert llm_utils_reloaded.Dataset is Any
            assert llm_utils_reloaded.AutoModelForCausalLM is Any
    finally:
        # Restore original module to avoid affecting other tests
        sys.modules["agilerl.utils.llm_utils"] = original_module


# ---------------------------------------------------------------------------
# Tests for create_llm_accelerator / _auto_zero_stage
# ---------------------------------------------------------------------------


def test_auto_zero_stage_none_model_size_returns_1():
    assert _auto_zero_stage(4, None) == 1


def test_auto_zero_stage_small_model_returns_1():
    """Model using < 60% of per-GPU VRAM -> ZeRO-1."""
    with patch("torch.cuda.get_device_properties") as mock_props:
        mock_props.return_value.total_mem = 24 * (1024**3)  # 24 GB
        assert _auto_zero_stage(2, 10.0) == 1  # 10/24 ~= 0.42


def test_auto_zero_stage_medium_model_returns_2():
    """Model using 60-90% of per-GPU VRAM -> ZeRO-2."""
    with patch("torch.cuda.get_device_properties") as mock_props:
        mock_props.return_value.total_mem = 24 * (1024**3)  # 24 GB
        assert _auto_zero_stage(2, 17.0) == 2  # 17/24 ~= 0.71


def test_auto_zero_stage_large_model_returns_3():
    """Model exceeding 90% of per-GPU VRAM -> ZeRO-3."""
    with patch("torch.cuda.get_device_properties") as mock_props:
        mock_props.return_value.total_mem = 24 * (1024**3)  # 24 GB
        assert _auto_zero_stage(2, 23.0) == 3  # 23/24 ~= 0.96


def test_auto_zero_stage_device_properties_exception_returns_1():
    """Fallback to ZeRO-1 when get_device_properties raises."""
    with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("no CUDA")):
        assert _auto_zero_stage(2, 14.0) == 1


def test_create_llm_accelerator_no_gpus_returns_none():
    with patch("torch.cuda.device_count", return_value=0):
        result = create_llm_accelerator()
    assert result is None


def test_create_llm_accelerator_uses_explicit_plugin_when_provided():
    AcceleratorState._reset_state(True)
    explicit_plugin = MagicMock(name="explicit_plugin")
    expected_accelerator = MagicMock(spec=Accelerator)
    mock_ctor = MagicMock(return_value=expected_accelerator)
    with (
        patch("torch.cuda.device_count", return_value=1),
        patch.dict(create_llm_accelerator.__globals__, {"Accelerator": mock_ctor}),
    ):
        result = create_llm_accelerator(deepspeed_plugin=explicit_plugin)
    assert result is expected_accelerator
    mock_ctor.assert_called_once_with(deepspeed_plugin=explicit_plugin)


def test_create_llm_accelerator_uses_launch_configured_plugin_when_available():
    AcceleratorState._reset_state(True)
    launch_plugin = object()
    launch_accelerator = MagicMock(spec=Accelerator)
    launch_accelerator.state = MagicMock()
    launch_accelerator.state.deepspeed_plugin = launch_plugin
    mock_ctor = MagicMock(return_value=launch_accelerator)
    with (
        patch("torch.cuda.device_count", return_value=1),
        patch.dict(create_llm_accelerator.__globals__, {"Accelerator": mock_ctor}),
    ):
        result = create_llm_accelerator()
    assert result is launch_accelerator
    mock_ctor.assert_called_once_with()


def test_create_llm_accelerator_raises_without_explicit_or_launch_plugin():
    AcceleratorState._reset_state(True)
    launch_accelerator = MagicMock(spec=Accelerator)
    launch_accelerator.state = MagicMock()
    launch_accelerator.state.deepspeed_plugin = None
    mock_ctor = MagicMock(return_value=launch_accelerator)
    with (
        patch("torch.cuda.device_count", return_value=1),
        patch.dict(create_llm_accelerator.__globals__, {"Accelerator": mock_ctor}),
        pytest.raises(RuntimeError, match="DeepSpeed is required"),
    ):
        create_llm_accelerator()


def test_normalize_reasoning_prompt_batch_stacked_dict_to_per_sample_list():
    prompts = {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "question": ["q0", "q1"],
        "meta": {"constant": True},
    }
    out = normalize_reasoning_prompt_batch(prompts)
    assert isinstance(out, list)
    assert len(out) == 2
    assert torch.equal(out[0]["input_ids"], torch.tensor([[1, 2]], dtype=torch.long))
    assert out[1]["question"] == "q1"
    assert out[0]["meta"] == {"constant": True}


def test_masked_stats_and_pool_by_turns_helpers():
    values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert masked_mean(values, mask) == pytest.approx(2.0)
    assert masked_var(values, mask, unbiased=False) == pytest.approx(1.0)

    token_values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    turn_ids = torch.tensor([[0, 0, 1, -1]])
    pooled = pool_by_turns(token_values, turn_ids, num_turns=2, reduction="mean")
    assert pooled.shape == (1, 2)
    assert pooled[0, 0].item() == pytest.approx(2.0)
    assert pooled[0, 1].item() == pytest.approx(5.0)


def test_move_params_helpers_call_model_move_and_cuda_sync():
    model_gpu = MagicMock()
    gpu_param = MagicMock()
    gpu_param.device = torch.device("cpu")
    model_gpu.parameters.return_value = iter([gpu_param])
    with patch("torch.cuda.synchronize") as sync:
        move_params_to_gpu(model_gpu, torch.device("cuda:0"))
    model_gpu.to.assert_called_once_with(torch.device("cuda:0"), non_blocking=True)
    sync.assert_called_once()

    model_cpu = MagicMock()
    cpu_param = MagicMock()
    cpu_param.device = torch.device("cuda:0")
    model_cpu.parameters.return_value = iter([cpu_param])
    with (
        patch("torch.cuda.synchronize") as sync,
        patch("torch.cuda.empty_cache") as empty_cache,
    ):
        move_params_to_cpu(model_cpu)
    model_cpu.to.assert_called_once_with("cpu", non_blocking=True)
    sync.assert_called_once()
    empty_cache.assert_called_once()


def test_get_model_name_or_path_and_align_deepspeed_lr_helpers():
    class _DirectModel:
        name_or_path = "direct_name"

    model = _DirectModel()
    assert get_model_name_or_path(model) == "direct_name"

    class _Inner:
        name_or_path = "inner_name"

    class _Nested:
        pretrained_model = _Inner()

    nested = _Nested()
    assert get_model_name_or_path(nested) == "inner_name"

    class _Missing:
        pass

    missing = _Missing()
    with pytest.raises(ValueError, match="Model name or path not found"):
        get_model_name_or_path(missing)

    accelerator = MagicMock()
    accelerator.state.deepspeed_plugin.deepspeed_config = {
        "optimizer": {"params": {"lr": 1e-3}}
    }
    with pytest.warns(UserWarning, match="DeepSpeed learning rate is set to"):
        out = align_deepspeed_lr(2e-3, accelerator)
    assert out == pytest.approx(2e-3)
    assert accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"][
        "lr"
    ] == pytest.approx(2e-3)


def test_gather_if_zero3_stage_not_three_noop():
    """ZeRO stages other than 3 should be a no-op context manager."""
    with gather_if_zero3(1, []):
        assert True


def test_liger_dpo_with_alpha_backward_returns_sixteen_outputs_with_trailing_nones() -> (
    None
):
    """``_LigerDPOWithAlpha.backward`` forwards to the base, keeps four grads, pads twelve ``None``."""
    from agilerl import HAS_LIGER_KERNEL

    if not HAS_LIGER_KERNEL:
        pytest.skip("liger-kernel not installed")

    import agilerl.utils.llm_utils as llm_utils_mod

    def fake_parent_backward(ctx, grad_output):
        return tuple(range(16))

    with patch.object(
        llm_utils_mod.LigerFusedLinearPreferenceBase,
        "backward",
        staticmethod(fake_parent_backward),
    ):
        out = llm_utils_mod._LigerDPOWithAlpha.backward(MagicMock(), torch.tensor(1.0))

    assert len(out) == 16
    assert out[:4] == (0, 1, 2, 3)
    assert out[4:] == (None,) * 12
