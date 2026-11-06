# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared LLM algorithm test helpers that must not require vLLM.

SFT/DPO tests import these without pulling in ``test_grpo``'s vLLM
``importorskip``, which is unavailable on macOS/Windows.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    PackedSequence,
    position_ids_from_mask,
)
from agilerl.distributed.context_parallel.sequence import pad_right_for_cp
from agilerl.utils.llm_packing import (
    PackedBatch,
    pack_next_token_frame,
    pack_padded_batch,
    unpack_logprobs,
)
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

LAYOUTS = ("padded", "packed", "cp")
# Multiple the one-rank context-parallel layout pads its action frame to.
CP_ROUND_TO = 8


def use_packed(agent: Any, varlen: bool) -> None:
    """Pack *agent*'s actor forwards, with FlashAttention varlen offsets when *varlen*."""
    agent.sequence_packing = True
    agent.sequence_layout = PackedSequence(
        varlen=varlen,
    )


def use_layout(agent: Any, kind: str) -> None:
    """Run *agent*'s actor forwards padded, packed, or context-parallel on one gloo rank."""
    if kind == "packed":
        use_packed(agent, False)
    elif kind == "cp":
        agent.sequence_layout = ContextParallelSequence(
            pad_token_id=agent.pad_token_id,
            cp=1,
            group=dist.group.WORLD,
            rank=0,
            round_to=CP_ROUND_TO,
        )


def real_columns(mask: torch.Tensor) -> int:
    """Columns up to and including the last real token of any row."""
    return int(mask.any(dim=0).nonzero().max()) + 1


def hand_laid_inputs(
    agent: Any, kind: str, ids: torch.Tensor, mask: torch.Tensor, copies: int = 1
) -> tuple[dict[str, Any], PackedBatch | None, torch.Tensor, torch.Tensor]:
    """Actor kwargs of *kind* built from the packing and CP primitives.

    :return: ``(kwargs, packed, ids, mask)``; ``ids`` and ``mask`` are cut to
        the real width and CP-aligned under ``"cp"``, and ``packed`` is ``None``
        unless packed.
    """
    if kind == "packed":
        packed = pack_padded_batch(ids, mask)
        kwargs = {
            "input_ids": packed.input_ids.repeat(copies, 1),
            "position_ids": packed.position_ids.repeat(copies, 1),
            "use_cache": False,
        }
        return kwargs, packed, ids, mask
    if kind == "cp":
        real_width = real_columns(mask)
        ids, mask = pad_right_for_cp(
            ids[:, :real_width], CP_ROUND_TO, agent.pad_token_id, mask[:, :real_width]
        )
        ids, mask = ids.repeat(copies, 1), mask.repeat(copies, 1)
        kwargs = {
            "input_ids": ids[:, :-1],
            "position_ids": position_ids_from_mask(mask)[:, :-1],
            "attention_mask": mask[:, :-1],
            "use_cache": False,
        }
        return kwargs, None, ids, mask
    ids, mask = ids.repeat(copies, 1), mask.repeat(copies, 1)
    kwargs = {"input_ids": ids, "attention_mask": mask, "use_cache": False}
    if agent.calc_position_embeddings:
        kwargs["position_ids"] = position_ids_from_mask(mask)
    return kwargs, None, ids, mask


def hand_laid_logprobs(
    agent: Any,
    kind: str,
    ids: torch.Tensor,
    mask: torch.Tensor,
    score_mask: torch.Tensor | None = None,
    support_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B, T - 1)`` log-probs of one unsplit forward of *kind*, scored on its own frame."""
    kwargs, _, _, _ = hand_laid_inputs(agent, kind, ids, mask)
    hidden, _ = agent._forward_hidden_and_value(kwargs)
    return score_hand_laid_hidden(
        agent, kind, hidden, ids, mask, score_mask, support_ids
    )


def score_hand_laid_hidden(
    agent: Any,
    kind: str,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    mask: torch.Tensor,
    score_mask: torch.Tensor | None = None,
    support_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B, T - 1)`` log-probs of a hand-laid *kind* forward's hidden states of ``(B, T)`` ``ids``."""
    width = ids.shape[1]
    fused_fn, head_w, head_b = agent._fused_logprob_fn_and_head()
    if kind == "packed":
        packed = pack_padded_batch(ids, mask, score_mask)
        log_probs = agent._score_hidden(
            fused_fn,
            hidden[:, :-1],
            head_w,
            head_b,
            packed.input_ids[:, 1:],
            packed.score_mask,
            None
            if support_ids is None
            else pack_next_token_frame(support_ids, packed, -1),
            packed.n_scored,
        )
        return unpack_logprobs(log_probs, packed)
    if kind == "cp":
        real_width = real_columns(mask)
        hidden = torch.cat([hidden, hidden.new_zeros(hidden[:, :1].shape)], dim=1)
        ids, score_mask, support_ids = pad_right_for_cp(
            ids[:, :real_width],
            CP_ROUND_TO,
            agent.pad_token_id,
            None if score_mask is None else score_mask[:, : real_width - 1],
            None if support_ids is None else support_ids[:, : real_width - 1],
        )
    log_probs = agent._score_hidden(
        fused_fn, hidden[:, :-1], head_w, head_b, ids[:, 1:], score_mask, support_ids
    )
    log_probs = log_probs[:, : width - 1]
    return F.pad(log_probs, (0, width - 1 - log_probs.shape[1]))


def layout_batch(
    seq_len: int = 14,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows of 12, 9, 6 and 11 tokens with two all-pad trailing columns.

    :return: ``(ids, mask, score_mask, support_ids)``; the score mask skips a
        two-token prompt and each 3-token support holds its target.
    """
    ids, mask = right_padded_batch([12, 9, 6, 11], seq_len=seq_len)
    score_mask = mask[:, 1:] & (torch.arange(seq_len - 1) >= 2)
    generator = torch.Generator().manual_seed(5)
    support_ids = torch.randint(0, 63, (4, seq_len - 1, 3), generator=generator)
    support_ids[..., 0] = ids[:, 1:]
    return ids, mask.long(), score_mask, support_ids


def take_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone and clear every parameter gradient."""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
            param.grad = None
    return grads


class DummyConfig(PretrainedConfig):
    def __init__(
        self,
        input_size=16,
        max_tokens=8,
        vocab_size=100,
        intermediate_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size


# A dataclass, as HF ``ModelOutput`` is, so FSDP2 finds ``logits`` and hooks
# its pre-backward all-gather.
@dataclass
class DummyForwardOutput:
    logits: torch.Tensor


class DummyMLPPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = DummyConfig
    base_model_prefix = "dummy_mlp"
    supports_gradient_checkpointing = True

    def __init__(self, config: DummyConfig, device="cpu"):
        super().__init__(config)
        self.input_size = config.input_size
        self.max_tokens = config.max_tokens
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing_enabled = False
        self.datatype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        hidden_size = 32
        # Standard causal-LM shape (embed -> body -> lm_head) so the
        # (now unconditional) fused-linear-logprob path can identity-patch
        # ``lm_head`` and read the hidden state. ``linear_1`` stays the LoRA
        # target the fixtures expect.
        self.embed = nn.Embedding(
            self.vocab_size, hidden_size, device=device, dtype=self.datatype
        )
        self.linear_1 = nn.Linear(
            hidden_size,
            hidden_size,
            device=device,
            dtype=self.datatype,
        )
        self.lm_head = nn.Linear(
            hidden_size,
            self.vocab_size,
            bias=False,
            device=device,
            dtype=self.datatype,
        )

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @property
    def model(self):
        return self

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> DummyForwardOutput:
        # ``lm_head`` may be identity-patched by the fused-linear-logprob path,
        # in which case this returns the hidden state instead of logits.
        hidden = self.linear_1(self.embed(input_ids.long()))
        return DummyForwardOutput(logits=self.lm_head(hidden))

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            msg = "`input_ids` must be provided for generation."
            raise ValueError(msg)
        input_shape = input_ids.shape
        group_size = input_shape[0]
        prompt_size = input_shape[1]
        # Simple generation: just return random tokens based on vocab size and desired length
        return torch.randint(
            0,
            self.vocab_size,
            (group_size, prompt_size + self.config.max_tokens),
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return


def create_module(input_size, max_tokens, vocab_size, device):
    return DummyMLPPreTrainedModel(
        config=DummyConfig(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
        ),
        device=device,
    )


class DummyHiddenStatesModel(DummyMLPPreTrainedModel):
    """Dummy causal LM that returns hidden states for a value head."""

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden = self.linear_1(self.embed(input_ids.long()))
        return CausalLMOutputWithPast(
            logits=self.lm_head(hidden), hidden_states=(hidden,)
        )


def create_value_head_module(
    input_size: int, max_tokens: int, vocab_size: int, device: str
) -> AutoModelForCausalLMWithValueHead:
    """Dummy causal LM wrapped with a value head, as PPO trains it."""
    config = DummyConfig(
        input_size=input_size,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
        hidden_size=32,
    )
    return AutoModelForCausalLMWithValueHead(
        DummyHiddenStatesModel(config=config, device=device)
    )


def create_tiny_qwen3() -> Qwen3ForCausalLM:
    """Seeded two-layer fp32 Qwen3 with real attention and a 64-token vocab."""
    torch.manual_seed(0)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=48,
            max_position_embeddings=64,
        )
    )


def right_padded_batch(
    lengths: list[int], seq_len: int = 12, pad_token_id: int = 63
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded random token rows of the given real lengths, right-padded with the pad id."""
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, pad_token_id, (len(lengths), seq_len), generator=generator)
    mask = torch.arange(seq_len) < torch.tensor(lengths).unsqueeze(1)
    return ids.masked_fill(~mask, pad_token_id), mask


def scale_losses(
    loss_fn: Callable[..., Any], scales: Iterator[float]
) -> Callable[..., Any]:
    """``loss_fn`` whose loss, the first output of a tuple, carries the next scale."""

    def scaled(*args: Any, **kwargs: Any) -> Any:
        out = loss_fn(*args, **kwargs)
        if isinstance(out, tuple):
            return (out[0] * next(scales), *out[1:])
        return out * next(scales)

    return scaled


def trainable_weights(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def optimizer_state(optimizer: OptimizerWrapper) -> list[torch.Tensor]:
    """Copy of every tensor of the optimizer's per-parameter state, in a fixed order."""
    state = optimizer.state_dict()["state"]
    return [
        value.detach().clone()
        for _, param_state in sorted(state.items())
        for _, value in sorted(param_state.items())
        if isinstance(value, torch.Tensor)
    ]


def record_outputs(fn: Callable[..., Any]) -> tuple[Callable[..., Any], list[Any]]:
    """``fn`` that also appends each output to the returned list."""
    outputs: list[Any] = []

    def recorded(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        outputs.append(out)
        return out

    return recorded, outputs


def record_calls(
    fn: Callable[..., Any],
) -> tuple[Callable[..., Any], list[tuple[tuple[Any, ...], dict[str, Any], Any]]]:
    """``fn`` that also appends each call's ``(args, kwargs, output)`` to the returned list."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []

    def recorded(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        calls.append((args, kwargs, out))
        return out

    return recorded, calls


def score_every_token(agent: Any, monkeypatch: Any) -> None:
    """Make ``agent._get_logprobs`` run the lm_head on every position, ignoring ``score_mask``."""
    get_logprobs = agent._get_logprobs

    def full_head(*args: Any, score_mask: Any = None, **kwargs: Any) -> torch.Tensor:
        return get_logprobs(*args, **kwargs)

    monkeypatch.setattr(agent, "_get_logprobs", full_head)
