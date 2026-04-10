from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import PreTrainedModelProtocol
from agilerl.typing import ExperiencesType, LLMObsType

if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import LoraConfig

    from agilerl.utils.llm_utils import SFTGym

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss


class SFT(LLMAlgorithm):
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
    :param reduce_memory_peak: Enable extra memory-reduction heuristics, defaults
        to False
    :type reduce_memory_peak: bool, optional
    :param device: Compute device, defaults to ``"cpu"``
    :type device: str, optional
    :param lora_config: LoRA config; when supplied the base model is wrapped with
        PEFT adapters, defaults to None
    :type lora_config: LoraConfig, optional
    :param accelerator: Accelerate distributed-training handle, defaults to None
    :type accelerator: accelerate.Accelerator, optional
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
    :param use_liger_loss: Use Liger kernel for memory-efficient cross-entropy loss
        computation, defaults to False. Requires ``liger_kernel`` to be installed;
        pass ``False`` to fall back to the standard PyTorch path.
    :type use_liger_loss: bool, optional
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
        reduce_memory_peak: bool = False,
        device: str = "cpu",
        lora_config: LoraConfig | None = None,
        accelerator: Accelerator | None = None,
        wrap: bool = True,
        clone: bool = False,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        use_liger_loss: bool = False,
    ) -> None:
        resolved_device = (
            f"cuda:{accelerator.process_index}"
            if accelerator is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        super().__init__(
            index=index,
            batch_size=batch_size,
            lr=lr,
            max_grad_norm=max_grad_norm,
            clone=clone,
            reduce_memory_peak=reduce_memory_peak,
            calc_position_embeddings=calc_position_embeddings,
            seed=seed,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            use_liger_loss=use_liger_loss,
            lora_config=lora_config,
            use_separate_reference_adapter=False,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=None,
            hp_config=hp_config,
            wrap=wrap,
            device=resolved_device,
            accelerator=accelerator,
            name="SFT",
            gradient_checkpointing=gradient_checkpointing,
        )
        self.temperature = 0
        self.use_vllm = False
        self.update_epochs = update_epochs

        self.loss_fn = (
            LigerCrossEntropyLoss(ignore_index=-100)
            if self.use_liger_loss
            else F.cross_entropy
        )

        self._initialize_actors(actor_network, not clone)
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

    def get_action(
        self,
        obs: LLMObsType,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Not implemented — SFT is an offline supervised algorithm.

        :raises NotImplementedError: Always.
        """
        msg = "SFT is an offline supervised algorithm and does not generate actions."
        raise NotImplementedError(msg)

    def learn(
        self,
        experiences: ExperiencesType,
        training: bool = True,
    ) -> tuple[float, float]:
        """Update model parameters using cross-entropy loss on response tokens.

        The loss is computed only on response tokens; prompt tokens and padding
        are masked out via ``ignore_index=-100``.

        :param experiences: Dict with keys ``input_ids`` (prompt + response token
            IDs), ``attention_mask``, and ``prompt_lengths`` (number of prompt
            tokens per sample) as produced by :class:`~agilerl.utils.llm_utils.SFTGym`.
        :type experiences: ExperiencesType
        :param training: When ``False`` the backward pass is skipped (eval mode).
        :type training: bool
        :return: ``(mean_loss, mean_perplexity)`` averaged over all samples in
            the batch.
        :rtype: tuple[float, float]
        """
        gc.collect()
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        input_ids: torch.Tensor = experiences["input_ids"]
        attention_mask: torch.Tensor = experiences["attention_mask"]
        # Check first that all tensors have the same max length before calculating the masks
        assert input_ids.shape[1] == attention_mask.shape[1], (
            "All tensors must have the same max length"
        )
        max_length = input_ids.shape[1]
        prompt_lengths: list[int] = experiences["prompt_lengths"]
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
        mean_loss = 0.0
        mean_perplexity = 0.0
        num_updates = 0

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
                mean_loss += loss_val
                mean_perplexity += float(np.exp(min(loss_val, 100)))
                num_updates += 1

        mean_loss /= num_updates
        mean_perplexity /= num_updates
        return mean_loss, mean_perplexity

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
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            model_kwargs["position_ids"] = position_ids

        logits = self.actor.forward(**model_kwargs).logits  # [B, L, V]
        # Shift: predict token i+1 from hidden state i
        shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, V]
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = labels.view(-1)
        return self.loss_fn(flat_logits, flat_labels, ignore_index=-100)

    def test(
        self,
        env: SFTGym,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the negative mean loss as a fitness score (higher is better).

        :param env: SFT environment providing evaluation batches
        :type env: SFTGym
        :param loop: Number of evaluation batches, defaults to 1
        :type loop: int, optional
        :return: Mean negative loss (scalar numpy array)
        :rtype: np.ndarray
        """
        with env.eval_mode(), torch.inference_mode():
            prompts = env.reset()
            losses = []
            for _ in range(loop):
                loss, _ = self.learn(prompts, training=False)
                losses.append(loss)
                prompts = env.step()
            mean_fit = -float(np.mean(losses))
        self.fitness.append(mean_fit)
        return np.array(mean_fit)
