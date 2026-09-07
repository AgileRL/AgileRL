# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import numpy.typing as npt
import torch

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms.configs import PopulationIndex, SFTObjective, SFTSetup
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.llm_init import named_llm_setup
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.typing import (
    MultiAgentObservationType,
    ObservationType,
    SFTPrompts,
)
from agilerl.utils.llm_utils import (
    aggregate_metrics_dict,
    is_sft_prompts,
)

if TYPE_CHECKING:
    from agilerl.llm_envs import DatasetEnv

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

    :param llm: Base model, tokenizer, LoRA, and training setup
    :type llm: SFTSetup
    :param objective: Algorithm-specific objective hyperparameters
    :type objective: SFTObjective | None
    :param member: Population index, mutation config, and last mutation
    :type member: PopulationIndex | None

    """

    def __init__(
        self,
        llm: SFTSetup,
        objective: SFTObjective | None = None,
        member: PopulationIndex | None = None,
    ) -> None:
        objective = objective or SFTObjective()
        member = member or PopulationIndex()
        super().__init__(named_llm_setup(llm, "SFT"), member)
        self._bind_sft(llm, objective)

    def _bind_sft(self, llm: SFTSetup, objective: SFTObjective) -> None:
        """Bind SFT epoch count and actor networks."""
        self.temperature = 0
        self.use_vllm = False
        self.update_epochs = objective.update_epochs
        self._initialize_actors(llm.model.actor_network, not llm.train.clone)
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
            tokens per sample) as produced by a ``objective="sft"``
            :class:`~agilerl.llm_envs.DatasetEnv`.
        :type experiences: ExperiencesType
        :param training: When ``False`` the backward pass is skipped (eval mode).
        :type training: bool
        :return: ``(loss, perplexity)`` averaged over all samples in
            the batch.
        :rtype: dict[str, float]
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
                    self._raise_if_loss_not_finite_on_any_rank(loss)
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

        learn_metrics = aggregate_metrics_dict(self.accelerator, averaged_metrics)

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
            hidden = self.actor.forward(**model_kwargs).logits  # [B, L, H]
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
        env: DatasetEnv,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """Return the negative mean loss as a fitness score (higher is better).

        :param env: SFT environment providing evaluation batches (``objective="sft"``)
        :type env: DatasetEnv
        :param loop: Number of evaluation batches, defaults to 1
        :type loop: int, optional
        :return: Mean negative loss (scalar numpy array)
        :rtype: npt.NDArray
        """
        with env.eval_mode(), torch.no_grad():
            losses = []
            for _ in range(loop):
                prompts = env.reset()
                if not is_sft_prompts(prompts):
                    msg = (
                        f"SFT.test needs an objective='sft' DatasetEnv; got a "
                        f"batch with keys {sorted(prompts)}."
                    )
                    raise TypeError(msg)
                metrics = self.learn(prompts, training=False)
                losses.append(metrics["loss"])
            mean_fit = -float(np.mean(losses))
        self.metrics.add_fitness(mean_fit)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        return np.array(mean_fit)
