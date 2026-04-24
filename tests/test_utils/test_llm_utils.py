from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("datasets", reason="LLM dependencies not installed")

from datasets import Dataset as Datasets
from torch import nn
from transformers import AutoTokenizer

from agilerl.utils.algo_utils import DummyOptimizer
from agilerl.utils.llm_utils import (
    compare_responses,
    gather_if_zero3,
    get_state_dict,
    sample_eval_prompts,
)

pytestmark = pytest.mark.llm


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 1000

    def batch_decode(self, *args, **kwargs):
        return ["This is a test completion."]

    def apply_chat_template(self, *args, **kwargs):
        return "This is a test completion."

    def __call__(self, *args, **kwargs):
        return torch.tensor([1, 2, 3, 4, 5])


def test_dummy_optimizer_init():
    """Test DummyOptimizer initialization."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr)
    assert optimizer is not None


def test_dummy_optimizer_step():
    """Test DummyOptimizer step method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr)

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
    optimizer = DummyOptimizer(params, lr)

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
    optimizer = DummyOptimizer(params, lr)

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
    optimizer = DummyOptimizer(params, lr)

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


def test_gather_if_zero3_stage_not_three_noop():
    """ZeRO stages other than 3 should be a no-op context manager."""
    with gather_if_zero3(1, []):
        assert True
