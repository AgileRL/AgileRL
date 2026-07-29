# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from agilerl import HAS_LIGER_KERNEL

if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    from agilerl.llm_envs import PreferenceGym

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import PreTrainedModelProtocol
from agilerl.typing import (
    MultiAgentObservationType,
    ObservationType,
    PreferencePrompts,
)
from agilerl.utils.algo_utils import get_experiences_samples
from agilerl.utils.llm_utils import aggregate_metrics_dict, resolve_llm_device

if HAS_LIGER_KERNEL:
    from agilerl.algorithms.core.llm_ops.fused_loss import LigerDPOWithAlpha


class DPO(LLMAlgorithm[PreferencePrompts]):
    """Direct Preference Optimization (DPO).

    Paper: https://arxiv.org/pdf/2305.18290

    :param pad_token_id: Pad token id
    :type pad_token_id: int
    :param pad_token: Pad token
    :type pad_token: str
    :param model_name: Model name
    :type model_name: str, optional
    :param actor_network: HuggingFace LLM
    :type actor_network: PreTrainedModelProtocol
    :param model_config: Model configuration, to be used when creating the model from a name or path.
    :type model_config: dict[str, Any] | None
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param batch_size: Batch size for training, defaults to 16
    :type batch_size: int, optional
    :param lr: Learning rate, defaults to 0.000005
    :type lr: float, optional
    :param beta: DPO beta parameter, defaults to 0.1
    :type beta: float, optional
    :param nll_alpha: Weight for the NLL loss on chosen responses (DPO + NLL), defaults to 1.0.
        Set to 0 to disable the NLL term entirely.
    :type nll_alpha: float, optional
    :param max_grad_norm: Maximum gradient norm, defaults to 0.1
    :type max_grad_norm: float, optional
    :param update_epochs: Number of update epochs, defaults to 1
    :type update_epochs: int, optional
    :param calc_position_embeddings: Flag to indicate if position embeddings should be calculated, defaults to True
    :type calc_position_embeddings: bool, optional
    :param micro_batch_size_per_gpu: Micro batch size per GPU, defaults to None
    :type micro_batch_size_per_gpu: int, optional
    :param device: Device to train on. Ignored when an accelerator is given (each rank
        owns its own GPU); ``None`` auto-detects CUDA/MPS/CPU.
    :type device: str, optional
    :param lora_config: Config for LoRA, defaults to None
    :type lora_config: LoraConfig, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    :param clone: Flag to indicate if the instantiation is a cloning, defaults to False
    :type clone: bool, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    :param gradient_checkpointing: Flag to indicate if gradient checkpointing should be used, defaults to True
    :type gradient_checkpointing: bool, optional
    :param torch_compiler: Torch compile mode (e.g. ``'default'``), defaults to None
    :type torch_compiler: str | None, optional
    :param use_liger_loss: Use Liger kernel for memory-efficient loss
        computation. Defaults to ``False``. Pass ``True`` to opt in
        (requires ``liger-kernel`` to be installed; warns and falls back
        to ``False`` otherwise). When ``training=False`` the standard
        path is always used regardless of this flag.
    :type use_liger_loss: bool, optional
    :param chunk_rows: Primary chunk-size knob for fused logit tiles used by
        both standard and Liger paths.
    :type chunk_rows: int | None, optional
    :param reduce_memory_peak: Deprecated and ignored; previously hinted
        peak-memory batching. Configure ``micro_batch_size_per_gpu`` instead.
    :type reduce_memory_peak: bool, optional
    :param cast_logprobs_to_fp32: When ``True`` (default), run the per-token
        log-prob reduction (``gather`` / ``logsumexp``) in fp32 before casting
        back to the input dtype, for numerically stable log-probs. ``False`` runs
        it in the input dtype, saving a little memory at the cost of a per-token
        bf16 quantisation error that can bias importance-sampling ratios.
    :type cast_logprobs_to_fp32: bool, optional
    :param use_separate_reference_adapter: Keep a dedicated ``reference`` LoRA
        adapter whose weights are frozen snapshots of the actor used for the
        DPO log-probability baseline. When ``False`` the reference log-probs
        are obtained by disabling the actor adapter at inference time.
        Defaults to True.
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
        lr: float = 0.000005,
        beta: float = 0.1,
        nll_alpha: float = 1.0,
        max_grad_norm: float = 0.1,
        update_epochs: int = 1,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        device: str | torch.device | None = None,
        lora_config: LoraConfig | None = None,
        accelerator: Accelerator | None = None,
        wrap: bool = True,
        clone: bool = False,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
        use_liger_loss: bool = False,
        chunk_rows: int | None = None,
        reduce_memory_peak: bool = False,
        cast_logprobs_to_fp32: bool = True,
        use_separate_reference_adapter: bool = True,
        quantization_config: BitsAndBytesConfig | None = None,
        activation_offload: bool = False,
        lora_target_scope: str | None = None,
    ) -> None:
        resolved_device = resolve_llm_device(accelerator, device)
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
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=None,
            hp_config=hp_config,
            wrap=wrap,
            device=resolved_device,
            accelerator=accelerator,
            name="DPO",
            gradient_checkpointing=gradient_checkpointing,
            torch_compiler=torch_compiler,
            reduce_memory_peak=reduce_memory_peak,
            cast_logprobs_to_fp32=cast_logprobs_to_fp32,
            use_separate_reference_adapter=use_separate_reference_adapter,
            quantization_config=quantization_config,
            activation_offload=activation_offload,
            lora_target_scope=lora_target_scope,
        )
        self.beta = beta
        self.nll_alpha = nll_alpha
        self.temperature = (
            1  # Temperature for logits calculation, DPO does not use temperature
        )
        self.use_vllm = False  # DPO does not use VLLM
        self.update_epochs = update_epochs

        self._initialize_actors(actor_network, not clone)
        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

        # Register metrics to keep track of during training
        self.metrics.register("loss")
        self.metrics.register("chosen_reward")
        self.metrics.register("rejected_reward")
        self.metrics.register("reward_margin")

    def get_action(
        self,
        obs: ObservationType | MultiAgentObservationType,
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        """Not implemented — DPO is an offline preference algorithm.

        :param obs: The observation of the agent
        :type obs: ObservationType | MultiAgentObservationType
        :param args: Additional arguments (unused; for base contract compatibility)
        :type args: Any
        :param kwargs: Additional keyword arguments (e.g. training; unused)
        :type kwargs: Any
        :raises NotImplementedError: Always.
        """
        msg = "DPO is an offline algorithm and therefore does not require completions to be generated."
        raise NotImplementedError(
            msg,
        )

    def learn(
        self,
        experiences: PreferencePrompts,
        training: bool = True,
    ) -> dict[str, float]:
        """Update agent network parameters to learn from preference data.

        :param experiences: Batched chosen/rejected input ids and attention masks
            with prompt lengths, as produced by :class:`~agilerl.llm_envs.PreferenceGym`.
        :type experiences: PreferencePrompts
        :param training: Whether the agent is training or not
        :type training: bool
        :return: Dict with keys ``loss``, ``chosen_reward``, ``rejected_reward``.
        :rtype: dict[str, float]
        """
        gc.collect()
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # The following tensors are size [batch_size, max_length]
        chosen_input_ids = experiences["chosen_input_ids"].to(self.device)
        rejected_input_ids = experiences["rejected_input_ids"].to(self.device)
        chosen_attention_mask = experiences["chosen_attention_mask"].to(self.device)
        rejected_attention_mask = experiences["rejected_attention_mask"].to(self.device)
        # Check first that all tensors have the same max length before calculating the masks
        assert (
            chosen_input_ids.shape[1]
            == rejected_input_ids.shape[1]
            == chosen_attention_mask.shape[1]
            == rejected_attention_mask.shape[1]
        ), "All tensors must have the same max length"

        max_length = chosen_input_ids.shape[1]
        prompt_lengths = experiences["prompt_lengths"]

        # Build the response mask on CPU (same device as dataloader tensors).
        prompt_masks = LLMAlgorithm._create_prompt_masks(
            prompt_lengths,
            max_length=max_length,
        ).to(self.device)

        # Mask has to be shifted by 1 as output log probs dims are 1 shorter than input ids as first token is used to predict the first log prob
        chosen_mask = (prompt_masks * chosen_attention_mask)[:, 1:]
        rejected_mask = (prompt_masks * rejected_attention_mask)[:, 1:]
        num_samples = chosen_input_ids.shape[0]
        batch_size = min(
            num_samples,
            getattr(self, "micro_batch_size_per_gpu", self.batch_size_per_process),
        )
        batch_idxs = np.arange(num_samples)
        learn_metrics = {
            "loss": 0.0,
            "chosen_reward": 0.0,
            "rejected_reward": 0.0,
        }
        ref_rejected_log_probs, ref_chosen_log_probs = None, None
        if not self.use_liger_loss:
            with torch.no_grad():
                ref_rejected_log_probs = self._get_logprobs(
                    rejected_input_ids,
                    batch_size,
                    use_reference=True,
                    eval_mode=True,
                    attention_mask=rejected_attention_mask,
                )
                ref_chosen_log_probs = self._get_logprobs(
                    chosen_input_ids,
                    batch_size,
                    use_reference=True,
                    eval_mode=True,
                    attention_mask=chosen_attention_mask,
                )

        for _ in range(self.update_epochs):
            for start in range(0, num_samples, batch_size):
                minibatch_idxs = batch_idxs[
                    start : min((start + batch_size), num_samples)
                ]
                loss, chosen_reward, rejected_reward = self._dpo_loss(
                    batch_size,
                    minibatch_idxs,
                    chosen_input_ids,
                    chosen_attention_mask,
                    rejected_input_ids,
                    rejected_attention_mask,
                    chosen_mask,
                    rejected_mask,
                    ref_rejected_log_probs,
                    ref_chosen_log_probs,
                    training,
                )
                if training:
                    self._backward_pass(loss)

                learn_metrics["loss"] += loss.item()
                learn_metrics["chosen_reward"] += chosen_reward.mean().item()
                learn_metrics["rejected_reward"] += rejected_reward.mean().item()

        learn_metrics = {
            key: value / num_samples for key, value in learn_metrics.items()
        }

        # Aggregate metrics across GPUs for both train/test paths. (Fresh dict
        # display so ty checks the values against the parameter's wider,
        # invariant dict value union.)
        agg = aggregate_metrics_dict(self.accelerator, {**learn_metrics})

        if training:
            self.metrics.log("loss", agg["loss"])
            self.metrics.log("chosen_reward", agg["chosen_reward"])
            self.metrics.log("rejected_reward", agg["rejected_reward"])
            self.metrics.log(
                "reward_margin", agg["chosen_reward"] - agg["rejected_reward"]
            )

        return learn_metrics

    def _dpo_loss(
        self,
        batch_size: int,
        minibatch_idxs: npt.NDArray,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor | None,
        ref_chosen_log_probs: torch.Tensor | None,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the DPO loss.

        :param batch_size: Batch size
        :type batch_size: int
        :param minibatch_idxs: Indices selecting rows for this minibatch.
        :type minibatch_idxs: numpy.ndarray
        :param chosen_input_ids: Chosen input IDs
        :type chosen_input_ids: torch.Tensor
        :param chosen_attention_mask: Chosen attention mask
        :type chosen_attention_mask: torch.Tensor
        :param rejected_input_ids: Rejected input IDs
        :type rejected_input_ids: torch.Tensor
        :param rejected_attention_mask: Rejected attention mask
        :type rejected_attention_mask: torch.Tensor
        :param chosen_mask: Chosen mask
        :type chosen_mask: torch.Tensor
        :param rejected_mask: Rejected mask
        :type rejected_mask: torch.Tensor
        :param ref_rejected_log_probs: Rejected log probabilities using the reference model
        :type ref_rejected_log_probs: torch.Tensor | None
        :param ref_chosen_log_probs: Chosen log probabilities using the reference model
        :type ref_chosen_log_probs: torch.Tensor | None
        :param training: Whether the agent is training or not
        :type training: bool
        :return: Loss, chosen rewards, rejected rewards
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        (
            batch_chosen_input_ids,
            batch_chosen_attention_mask,
            batch_rejected_input_ids,
            batch_rejected_attention_mask,
            batch_chosen_mask,
            batch_rejected_mask,
            batch_ref_rejected_log_probs,
            batch_ref_chosen_log_probs,
        ) = get_experiences_samples(
            minibatch_idxs,
            chosen_input_ids,
            chosen_attention_mask,
            rejected_input_ids,
            rejected_attention_mask,
            chosen_mask,
            rejected_mask,
            ref_rejected_log_probs,
            ref_chosen_log_probs,
        )
        if self.use_liger_loss:
            return self._dpo_loss_liger(
                batch_chosen_input_ids,
                batch_rejected_input_ids,
                batch_chosen_attention_mask,
                batch_rejected_attention_mask,
                batch_chosen_mask,
                batch_rejected_mask,
            )
        # Standard (non-Liger) path: ``learn`` precomputed both reference
        # log-prob tensors, so they are never None here.
        assert batch_ref_chosen_log_probs is not None
        assert batch_ref_rejected_log_probs is not None
        batch_rejected_log_probs = self._get_logprobs(
            batch_rejected_input_ids,
            batch_size,
            use_reference=False,
            eval_mode=(not training),
            attention_mask=batch_rejected_attention_mask,
        )
        batch_chosen_log_probs = self._get_logprobs(
            batch_chosen_input_ids,
            batch_size,
            use_reference=False,
            eval_mode=(not training),
            attention_mask=batch_chosen_attention_mask,
        )
        return self._dpo_loss_standard(
            batch_chosen_log_probs,
            batch_rejected_log_probs,
            batch_ref_chosen_log_probs,
            batch_ref_rejected_log_probs,
            batch_chosen_mask,
            batch_rejected_mask,
        )

    def _dpo_loss_standard(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor,
        ref_chosen_log_probs: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the DPO loss (standard PyTorch path, summed log-probs per sequence).

        :param chosen_log_probs: Policy log-probs for chosen completions ``[B, seq_len-1]``.
        :type chosen_log_probs: torch.Tensor
        :param rejected_log_probs: Policy log-probs for rejected completions ``[B, seq_len-1]``.
        :type rejected_log_probs: torch.Tensor
        :param ref_chosen_log_probs: Reference log-probs for chosen ``[B, seq_len-1]``.
        :type ref_chosen_log_probs: torch.Tensor
        :param ref_rejected_log_probs: Reference log-probs for rejected ``[B, seq_len-1]``.
        :type ref_rejected_log_probs: torch.Tensor
        :param chosen_mask: Mask over completion tokens (shifted) for chosen.
        :type chosen_mask: torch.Tensor
        :param rejected_mask: Mask over completion tokens (shifted) for rejected.
        :type rejected_mask: torch.Tensor
        :return: Tuple of ``(loss, implicit_chosen_reward, implicit_rejected_reward)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # Mask and sum the logprobs
        assert chosen_log_probs.shape == chosen_mask.shape, (
            f"Chosen log probabilities and mask must have the same shape, got {chosen_log_probs.shape} and {chosen_mask.shape}"
        )
        chosen_log_probs = (chosen_log_probs * chosen_mask).sum(dim=-1)
        rejected_log_probs = (rejected_log_probs * rejected_mask).sum(dim=-1)
        ref_chosen_log_probs = (ref_chosen_log_probs * chosen_mask).sum(dim=-1)
        ref_rejected_log_probs = (ref_rejected_log_probs * rejected_mask).sum(dim=-1)
        rejected_ratio = rejected_log_probs - ref_rejected_log_probs
        chosen_ratio = chosen_log_probs - ref_chosen_log_probs
        with torch.no_grad():
            implicit_chosen_reward = self._compute_implicit_reward(
                chosen_log_probs,
                ref_chosen_log_probs,
            )
            implicit_rejected_reward = self._compute_implicit_reward(
                rejected_log_probs,
                ref_rejected_log_probs,
            )
        loss = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean()
        if self.nll_alpha > 0:
            loss = loss - self.nll_alpha * chosen_log_probs.sum() / chosen_mask.sum()

        return (
            loss,
            implicit_chosen_reward,
            implicit_rejected_reward,
        )

    def _dpo_loss_liger(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_attn: torch.Tensor,
        rejected_attn: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the DPO loss using the Liger fused linear kernel.

        :param chosen_ids: Input IDs for chosen completions (B, seq_len).
        :type chosen_ids: torch.Tensor
        :param rejected_ids: Input IDs for rejected completions (B, seq_len).
        :type rejected_ids: torch.Tensor
        :param chosen_attn: Attention mask for chosen completions (B, seq_len).
        :type chosen_attn: torch.Tensor
        :param rejected_attn: Attention mask for rejected completions (B, seq_len).
        :type rejected_attn: torch.Tensor
        :param chosen_mask: Completion token mask for chosen, shifted (B, seq_len-1).
        :type chosen_mask: torch.Tensor
        :param rejected_mask: Completion token mask for rejected, shifted (B, seq_len-1).
        :type rejected_mask: torch.Tensor
        :return: Loss, chosen rewards, rejected rewards.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        if not HAS_LIGER_KERNEL:
            msg = (
                "Liger DPO loss was requested but `liger-kernel` is not available. "
                "Set use_liger_loss=False."
            )
            raise ImportError(msg)

        lm_head = self._get_lm_head()
        lm_head_weight = lm_head.weight  # (vocab_size, hidden_size)
        lm_head_bias = lm_head.bias

        def _get_hidden(ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            """Run a forward pass and return hidden states fed into the language-model head.

            :param ids: Token IDs ``[batch, seq_len]``.
            :type ids: torch.Tensor
            :param attn_mask: Attention mask ``[batch, seq_len]``.
            :type attn_mask: torch.Tensor
            :return: Hidden states before the LM head ``[batch, seq_len, hidden]``.
            :rtype: torch.Tensor
            """
            captured = []
            hook = lm_head.register_forward_pre_hook(
                lambda m, inputs: captured.append(inputs[0])
            )
            try:
                self.actor(input_ids=ids, attention_mask=attn_mask, use_cache=False)
            finally:
                hook.remove()
            return captured[0]  # (B, seq_len, hidden_size)

        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        chosen_attn = chosen_attn.to(self.device)
        rejected_attn = rejected_attn.to(self.device)

        # Reference hidden states — no gradient, two separate forward passes (B each)
        with torch.no_grad():
            with self.select_adapter("reference"):
                self.actor.eval()
                ref_chosen_hidden = _get_hidden(
                    chosen_ids, chosen_attn
                )  # (B, seq_len, H)
                ref_rejected_hidden = _get_hidden(
                    rejected_ids, rejected_attn
                )  # (B, seq_len, H)
        ref_hidden = torch.cat(
            [ref_chosen_hidden, ref_rejected_hidden], dim=0
        )  # (2B, seq_len, H)

        # Policy hidden states — with gradient, two separate forward passes (B each)
        with self.select_adapter("actor"):
            self.actor.train()
            policy_chosen_hidden = _get_hidden(
                chosen_ids, chosen_attn
            )  # (B, seq_len, H)
            policy_rejected_hidden = _get_hidden(
                rejected_ids, rejected_attn
            )  # (B, seq_len, H)
        policy_hidden = torch.cat(
            [policy_chosen_hidden, policy_rejected_hidden], dim=0
        )  # (2B, seq_len, H)

        # Build shifted targets; mask prompt/padding tokens with -100
        def _make_target(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            t = ids[:, 1:].clone()  # (B, seq_len-1)
            t[~mask.bool()] = -100
            return t

        chosen_target = _make_target(chosen_ids, chosen_mask.to(self.device))
        rejected_target = _make_target(rejected_ids, rejected_mask.to(self.device))
        stacked_target = torch.cat(
            [chosen_target, rejected_target], dim=0
        )  # (2B, seq_len-1)

        # Trim hidden states to seq_len-1 to align with shifted targets
        policy_hidden = policy_hidden[:, :-1, :].contiguous()
        ref_hidden = ref_hidden[:, :-1, :].contiguous()

        loss, aux = LigerDPOWithAlpha.apply(
            policy_hidden,
            lm_head_weight,
            stacked_target,
            lm_head_bias,  # bias (None for most LLMs)
            ref_hidden,  # ref_input
            lm_head_weight,  # ref_weight (lm_head is never LoRA-adapted, so is the same as the policy weight)
            lm_head_bias,  # ref_bias (same weight → same bias)
            -100,  # ignore_index
            self.beta,
            self.nll_alpha,  # alpha — scales NLL in the fused kernel
            self.nll_alpha > 0,  # compute_nll_loss
            True,  # compiled
            True,  # use_ref_model
            False,  # average_log_prob (sum, matching _dpo_loss)
            self.chunk_rows or 1,  # chunk_size (sequences per chunk)
            "sigmoid",  # loss_type
        )
        # aux = (chosen_logps, rejected_logps, chosen_logits_mean, rejected_logits_mean,
        #        nll_loss, chosen_rewards, rejected_rewards)
        chosen_reward = aux[5]  # beta * (chosen_logps  - ref_chosen_logps)
        rejected_reward = aux[6]  # beta * (rejected_logps - ref_rejected_logps)

        return loss, chosen_reward, rejected_reward

    def _compute_implicit_reward(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the preference reward for the chosen and rejected completions.

        :param log_probs: Summed log-probabilities under the policy (1-D per batch row).
        :type log_probs: torch.Tensor
        :param ref_log_probs: Summed log-probabilities under the reference policy.
        :type ref_log_probs: torch.Tensor
        :return: Implicit reward (beta * (log_probs - ref_log_probs))
        :rtype: torch.Tensor
        """
        implicit_reward = log_probs - ref_log_probs
        return self.beta * implicit_reward

    def test(
        self,
        env: PreferenceGym,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Return the fitness (test) score of the agent.

        :param env: The environment to be tested in
        :type env: PreferenceGym environment
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 1
        :type loop: int, optional
        :return: Mean test score (numpy array)
        :rtype: npt.NDArray
        """
        with env.eval_mode(), torch.no_grad():
            prompts = env.reset()
            rewards = []
            for _ in range(loop):
                learn_result = self.learn(prompts, training=False)
                chosen_reward = learn_result["chosen_reward"]
                rejected_reward = learn_result["rejected_reward"]
                reward_margin = chosen_reward - rejected_reward
                rewards.append(np.asarray(reward_margin).item())
                prompts = env.step()
            mean_fit = float(np.mean(rewards))
        self.metrics.add_fitness(mean_fit)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        return np.array(mean_fit)
