# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, NoReturn, Literal

import numpy as np
import numpy.typing as npt
import torch

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import PreTrainedModelProtocol
from agilerl.typing import (
    MultiAgentObservationType,
    ObservationType,
    SFTPrompts,
)
from agilerl.utils.distributed import FSDPConfig, barrier, resolve_device
from agilerl.utils.llm_utils import aggregate_metrics_dict

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    from agilerl.llm_envs import SFTGym

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )


class SFT(LLMAlgorithm[SFTPrompts]):
    """Supervised Fine-Tuning (SFT) algorithm.

    Trains an LLM via token-level cross-entropy loss computed exclusively on the
    response tokens of each ``(prompt, response)`` pair.  The dataset should
    simply contain a prompt and a target response — no rejected/negative
    responses are needed or used.

    This is typically the *first* stage of a two-step alignment pipeline:

    1. **SFT** (this class) — warm-up the model to follow instructions by
       minimising cross-entropy on ``(prompt, good_response)`` pairs.
    2. **DPO** — further align the SFT-initialised model using
       ``(prompt, chosen_response, rejected_response)`` triples.

    :param pad_token_id: Pad token id
    :type pad_token_id: int
    :param pad_token: Pad token string
    :type pad_token: str
    :param model_name: HuggingFace model name or path, used when no
        ``actor_network`` is supplied
    :type model_name: str, optional
    :param actor_network: Pre-built HuggingFace causal LM
    :type actor_network: PreTrainedModelProtocol, optional
    :param model_config: Extra kwargs forwarded to the model constructor
    :type model_config: dict, optional
    :param hp_config: Hyperparameter mutation config for AgileRL HPO, defaults
        to None (mutations disabled)
    :type hp_config: HyperparameterConfig, optional
    :param index: Population index, defaults to 0
    :type index: int, optional
    :param batch_size: Total training batch size (across all GPUs), defaults to 16
    :type batch_size: int, optional
    :param lr: Learning rate, defaults to 5e-5
    :type lr: float, optional
    :param max_grad_norm: Gradient clipping norm, defaults to 0.1
    :type max_grad_norm: float, optional
    :param update_epochs: Number of passes over each data batch, defaults to 1
    :type update_epochs: int, optional
    :param calc_position_embeddings: Whether to recompute position ids from the
        attention mask (recommended for packed/padded inputs), defaults to True
    :type calc_position_embeddings: bool, optional
    :param micro_batch_size_per_gpu: Micro-batch size for gradient accumulation.
        When None the full batch is used in a single forward pass.
    :type micro_batch_size_per_gpu: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param lora_config: LoRA config; when supplied the base model is wrapped with
        PEFT adapters, defaults to None
    :type lora_config: LoraConfig, optional
    :param gradient_accumulation_steps: Micro-batches to accumulate per optimizer step, defaults to 1
    :type gradient_accumulation_steps: int, optional
    :param fsdp_config: FSDP2 sharding settings for distributed runs, defaults to None
    :type fsdp_config: FSDPConfig | None, optional
    :param ep: Expert Parallel degree for packed-expert MoE (1 disables).
    :type ep: int, optional
    :param cp: Context-parallel degree (``1`` disables CP), defaults to 1
    :type cp: int, optional
    :param cp_style: CP attention style (``ulysses`` or ``ring``), defaults to ``ulysses``
    :type cp_style: Literal["ulysses", "ring"], optional
    :param wrap: Wrap models for distributed training on construction, defaults to
        True
    :type wrap: bool, optional
    :param clone: Flag that suppresses adapter initialisation when cloning an
        existing agent, defaults to False
    :type clone: bool, optional
    :param seed: Random seed, defaults to 42
    :type seed: int, optional
    :param gradient_checkpointing: Use gradient checkpointing to trade compute for
        memory, defaults to True
    :type gradient_checkpointing: bool, optional
    :param use_liger_loss: Use the Liger fused-linear cross-entropy kernel,
        defaults to ``False`` (requires ``liger-kernel``; warns and falls back
        otherwise). Both this and the standard path are memory-bounded — the
        full ``(B, L, V)`` logits are never materialized — so this is mainly a
        speed/kernel choice. The Liger kernel auto-sizes its own chunk; the
        standard path's chunk is set by ``chunk_rows``.
    :type use_liger_loss: bool, optional
    :param chunk_rows: Primary chunk-size knob for fused logit tiles. On SFT's
        standard path this controls the fused-logprob chunk rows directly.
    :type chunk_rows: int | None, optional
    :param reduce_memory_peak: Deprecated and ignored; previously hinted
        peak-memory batching. Configure ``micro_batch_size_per_gpu`` instead.
    :type reduce_memory_peak: bool, optional
    :param use_separate_reference_adapter: Also create a ``reference`` LoRA adapter
        alongside ``actor``. SFT does not itself use a reference policy, so this
        defaults to ``False``; enable it when you plan to save an SFT checkpoint
        that will be consumed by a downstream algorithm (e.g. DPO/GRPO) which
        expects a reference adapter. Defaults to False.
    :type use_separate_reference_adapter: bool, optional
    :param quantization_config: Optional ``transformers.BitsAndBytesConfig`` for
        loading the base model in 4-/8-bit (QLoRA). ``lm_head`` is kept
        unquantized so the fused-linear-logprob path stays numerically exact.
    :type quantization_config: BitsAndBytesConfig | None, optional
    :param activation_offload: When ``True``, run the training forward inside
        ``torch.autograd.graph.save_on_cpu`` so tensors saved for backward live
        in pinned host RAM instead of GPU memory. Trades PCIe bandwidth for GPU
        memory (the win grows with sequence length); a no-op during rollout /
        reference forwards.
    :type activation_offload: bool, optional
    :param lora_target_scope: Optional PEFT LoRA path scope for multimodal models
        (e.g. ``"language_model"``). Passed to
        :func:`adapt_lora_config_for_model`.
    :type lora_target_scope: str | None, optional
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_token: str,
        model_name: str | None = None,
        actor_network: PreTrainedModelProtocol | None = None,
        model_config: dict[str, Any] | None = None,
        hp_config: HyperparameterConfig | None = None,
        index: int = 0,
        batch_size: int = 16,
        lr: float = 5e-5,
        max_grad_norm: float = 0.1,
        update_epochs: int = 1,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        device: str | torch.device | None = None,
        lora_config: LoraConfig | None = None,
        gradient_accumulation_steps: int = 1,
        fsdp_config: FSDPConfig | None = None,
        ep: int = 1,
        cp: int = 1,
        cp_style: Literal["ulysses", "ring"] = "ulysses",
        wrap: bool = True,
        clone: bool = False,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        use_liger_loss: bool = False,
        chunk_rows: int | None = None,
        reduce_memory_peak: bool = False,
        use_separate_reference_adapter: bool = False,
        quantization_config: BitsAndBytesConfig | None = None,
        activation_offload: bool = False,
        lora_target_scope: str | None = None,
    ) -> None:
        resolved_device = resolve_device(device)
        super().__init__(
            index=index,
            batch_size=batch_size,
            lr=lr,
            max_grad_norm=max_grad_norm,
            clone=clone,
            calc_position_embeddings=calc_position_embeddings,
            seed=seed,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            use_liger_loss=use_liger_loss,
            chunk_rows=chunk_rows,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=None,
            hp_config=hp_config,
            wrap=wrap,
            device=resolved_device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fsdp_config=fsdp_config,
            ep=ep,
            cp=cp,
            cp_style=cp_style,
            name="SFT",
            gradient_checkpointing=gradient_checkpointing,
            reduce_memory_peak=reduce_memory_peak,
            quantization_config=quantization_config,
            activation_offload=activation_offload,
            lora_target_scope=lora_target_scope,
        )
        self.temperature = 0
        self.use_vllm = False
        self.update_epochs = update_epochs

        self._initialize_actors(actor_network, not clone)
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

        self.metrics.register("loss")
        self.metrics.register("perplexity")

    def get_action(
        self,
        obs: ObservationType | MultiAgentObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        """Not implemented — SFT is an offline supervised algorithm.

        :raises NotImplementedError: Always.
        """
        msg = "SFT is an offline supervised algorithm and does not generate actions."
        raise NotImplementedError(msg)

    def learn(
        self,
        experiences: SFTPrompts,
        training: bool = True,
    ) -> dict[str, float]:
        """Update model parameters using cross-entropy loss on response tokens.

        The loss is computed only on response tokens; prompt tokens and padding
        are masked out via ``ignore_index=-100``.

        :param experiences: Dict with keys ``input_ids`` (prompt + response token
            IDs), ``attention_mask``, and ``prompt_lengths`` (number of prompt
            tokens per sample) as produced by :class:`~agilerl.llm_envs.SFTGym`.
        :type experiences: SFTPrompts
        :param training: When ``False`` the backward pass is skipped (eval mode).
        :type training: bool
        :return: ``(loss, perplexity)`` averaged over all samples in
            the batch.
        :rtype: tuple[float, float]
        """
        gc.collect()
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        input_ids = experiences["input_ids"]
        attention_mask = experiences["attention_mask"]
        # Check first that all tensors have the same max length before calculating the masks
        assert input_ids.shape[1] == attention_mask.shape[1], (
            "All tensors must have the same max length"
        )
        max_length = input_ids.shape[1]
        prompt_lengths = experiences["prompt_lengths"]
        # Build the response mask on CPU (same device as dataloader tensors).
        prompt_masks = LLMAlgorithm._create_prompt_masks(
            prompt_lengths, max_length=max_length
        )  # CPU tensor
        # Mask has to be shifted by 1 as output log probs dims are 1 shorter than input ids as first token is used to predict the first log prob
        response_mask = (prompt_masks * attention_mask.cpu())[:, 1:]  # [B, L-1], CPU
        # Create labels for CE loss
        labels = torch.where(
            response_mask.bool(), input_ids[:, 1:].cpu(), -100
        )  # [B, L-1]

        num_samples = input_ids.shape[0]
        micro_bs = min(
            num_samples,
            getattr(self, "micro_batch_size_per_gpu", self.batch_size_per_process),
        )
        batch_idxs = np.arange(num_samples)
        num_updates = 0

        learn_metrics = {
            "loss": 0.0,
            "perplexity": 0.0,
        }

        for _ in range(self.update_epochs):
            for start in range(0, num_samples, micro_bs):
                end = min(start + micro_bs, num_samples)
                idxs = batch_idxs[start:end]
                loss = self._sft_loss(
                    input_ids[idxs].to(self.device),
                    attention_mask[idxs].to(self.device),
                    labels[idxs].to(self.device),
                    training=training,
                )
                if training:
                    self._backward_pass(loss)
                loss_val = loss.item()
                learn_metrics["loss"] += loss_val
                learn_metrics["perplexity"] += float(np.exp(min(loss_val, 100)))
                num_updates += 1

        # ``aggregate_metrics_dict`` takes an invariant dict over the full raw
        # metric-value union, so annotate the averaged dict to that exact type.
        averaged_metrics: dict[str, torch.Tensor | npt.NDArray | float] = {
            key: value / max(num_updates, 1) for key, value in learn_metrics.items()
        }

        learn_metrics = aggregate_metrics_dict(averaged_metrics)

        if training:
            self.metrics.log("loss", learn_metrics["loss"])
            self.metrics.log("perplexity", learn_metrics["perplexity"])

        return learn_metrics

    def _sft_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for a single micro-batch.

        :param input_ids: Token IDs ``[B, L]``
        :type input_ids: torch.Tensor
        :param attention_mask: Attention mask ``[B, L]``
        :type attention_mask: torch.Tensor
        :param labels: Shifted labels ``[B, L-1]`` with ``-100`` at ignored positions
        :type labels: torch.Tensor
        :param training: Whether gradients are needed
        :type training: bool
        :return: Scalar cross-entropy loss
        :rtype: torch.Tensor
        """
        self.actor.train(mode=training)

        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.calc_position_embeddings:
            model_kwargs["position_ids"] = self._position_ids_from_mask(attention_mask)

        # Run the transformer with the lm_head patched to identity so
        # ``.logits`` is the final hidden state, then compute the loss from the
        # hidden states + lm_head weight without ever materializing the full logits tensor.
        with self._patch_lm_head_to_identity():
            # FSDP2 all-gather hooks run on ``Module.__call__``.
            hidden = self.actor(**model_kwargs).logits  # [B, L, H]
        shift_hidden = hidden[:, :-1, :].contiguous()  # [B, L-1, H]

        if self.use_liger_loss:
            # Liger fused-linear CE: loss computed in bounded ``(chunk, V)`` tiles.
            flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
            lm_head = self._get_lm_head()
            with self._liger_head_gather():
                loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)(
                    lm_head.weight, flat_hidden, labels.view(-1), lm_head.bias
                )

        else:
            # Standard path, also token-chunked: per-token target logprobs via the
            # fused-linear-logprob kernel (bounded to ``(chunk_rows, V)``)
            fused_fn, lm_head_weight, lm_head_bias = self._fused_logprob_fn_and_head()
            ignore = labels == -100
            logps = fused_fn(
                shift_hidden,
                lm_head_weight,
                lm_head_bias,
                labels.masked_fill(ignore, 0),  # safe gather index; masked out below
                temperature=1.0,
                cast_to_fp32=self.cast_logprobs_to_fp32,
                chunk_rows=self.chunk_rows,
            )  # [B, L-1]
            token_mask = (~ignore).to(logps.dtype)
            loss = -(logps * token_mask).sum() / token_mask.sum().clamp_min(1.0)
        return loss

    def test(
        self,
        env: SFTGym,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Return the negative mean loss as a fitness score (higher is better).

        :param env: SFT environment providing evaluation batches
        :type env: SFTGym
        :param loop: Number of evaluation batches, defaults to 1
        :type loop: int, optional
        :return: Mean negative loss (scalar numpy array)
        :rtype: npt.NDArray
        """
        with env.eval_mode(), torch.no_grad():
            prompts = env.reset()
            losses = []
            for _ in range(loop):
                metrics = self.learn(prompts, training=False)
                losses.append(metrics["loss"])
                prompts = env.step()
            mean_fit = -float(np.mean(losses))
        self.metrics.add_fitness(mean_fit)
        if self.distributed:
            barrier()
        return np.array(mean_fit)
