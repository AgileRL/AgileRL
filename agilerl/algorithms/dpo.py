from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from agilerl import HAS_LIGER_KERNEL

if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import LoraConfig

    from agilerl.utils.llm_utils import PreferenceGym

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import PreTrainedModelProtocol
from agilerl.typing import ExperiencesType, LLMObsType
from agilerl.utils.algo_utils import get_experiences_samples

if HAS_LIGER_KERNEL or TYPE_CHECKING:
    from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
    from liger_kernel.chunked_loss.fused_linear_preference import (
        LigerFusedLinearPreferenceBase,
    )

    class _LigerDPOWithAlpha(LigerFusedLinearPreferenceBase):
        """Thin wrapper that exposes ``alpha`` for NLL scaling.

        ``LigerFusedLinearDPOFunction`` passes ``compute_nll_loss`` as a bool
        but never forwards ``alpha`` to the base class (which defaults to 1.0).
        This subclass reuses the DPO preference loss and adds ``alpha`` so the
        fused kernel correctly scales the NLL component.
        """

        preference_loss_fn = staticmethod(
            LigerFusedLinearDPOFunction.preference_loss_fn
        )

        @classmethod
        def forward(
            cls,
            ctx,
            _input,
            weight,
            target,
            bias=None,
            ref_input=None,
            ref_weight=None,
            ref_bias=None,
            ignore_index=-100,
            beta=0.1,
            alpha=1.0,
            compute_nll_loss=True,
            compiled=True,
            use_ref_model=True,
            average_log_prob=False,
            chunk_size=1,
            loss_type="sigmoid",
        ):
            return LigerFusedLinearPreferenceBase.forward(
                cls=cls,
                ctx=ctx,
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ignore_index=ignore_index,
                alpha=alpha,
                beta=beta,
                compute_nll_loss=compute_nll_loss,
                compiled=compiled,
                use_ref_model=use_ref_model,
                ref_input=ref_input,
                ref_weight=ref_weight,
                ref_bias=ref_bias,
                average_log_prob=average_log_prob,
                chunk_size=chunk_size,
                loss_type=loss_type,
            )

        @staticmethod
        def backward(ctx, *grad_output):
            grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
            return (*grads, *(None,) * 12)


class DPO(LLMAlgorithm):
    """The DPO algorithm class. DPO paper: https://arxiv.org/pdf/2305.18290.

    :param pad_token_id: Pad token id
    :type pad_token_id: int
    :param pad_token: Pad token
    :type pad_token: str
    :param model_name: Model name
    :type model_name: str, optional
    :param actor_network: HuggingFace LLM
    :type actor_network: PreTrainedModelProtocol
    :param model_config: Model configuration, to be used when creating the model from a name or path
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
    :param reduce_memory_peak: Flag to indicate if memory peak should be reduced, defaults to False
    :type reduce_memory_peak: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param lora_config: Config for LoRA, defaults to None
    :type lora_config: LoraConfig, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    :param clone: Flag to indicate if the instantiation is a cloning, defaults to False
    :type clone: bool, optional
    :param use_separate_reference_adapter: Flag to indicate if the reference policy should have a separate adapter, defaults to False
    :type use_separate_reference_adapter: bool, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    :param gradient_checkpointing: Flag to indicate if gradient checkpointing should be used, defaults to True
    :type gradient_checkpointing: bool, optional
    :param use_liger_loss: Use Liger kernel for memory-efficient loss computation, defaults to False.
        Requires ``liger_kernel`` to be installed; pass ``False`` to fall back to the standard PyTorch path.
        When ``training=False`` the standard path is always used regardless of this flag.
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
        lr: float = 0.000005,
        beta: float = 0.1,
        nll_alpha: float = 1.0,
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
        use_separate_reference_adapter: bool = False,
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
            use_separate_reference_adapter=use_separate_reference_adapter,
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

    def get_action(
        self,
        obs: LLMObsType,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return the action of the agent.

        :param obs: The observation of the agent
        :type obs: LLMObsType
        :param args: Additional arguments (unused; for base contract compatibility)
        :param kwargs: Additional keyword arguments (e.g. training; unused)
        :return: The action of the agent
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
        msg = "DPO is an offline algorithm and therefore does not require completions to be generated."
        raise NotImplementedError(
            msg,
        )

    def learn(
        self,
        experiences: ExperiencesType,
        training: bool = True,
    ) -> tuple[float, float, float]:
        """Update agent network parameters to learn from preference data.

        :param experiences: Batched chosen_input_ids, rejected_input_ids, chosen_attention_mask, rejected_attention_mask and rewards
        :type experiences: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        :param training: Whether the agent is training or not
        :type training: bool
        :return: mean loss, mean chosen reward, mean rejected reward
        :rtype: tuple[float, float, float]
        """
        gc.collect()
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # The following tensors are size [batch_size, max_length]
        chosen_input_ids: torch.Tensor = experiences["chosen_input_ids"].to(self.device)
        rejected_input_ids: torch.Tensor = experiences["rejected_input_ids"].to(
            self.device
        )
        chosen_attention_mask: torch.Tensor = experiences["chosen_attention_mask"].to(
            self.device
        )
        rejected_attention_mask: torch.Tensor = experiences[
            "rejected_attention_mask"
        ].to(self.device)
        # Check first that all tensors have the same max length before calculating the masks
        assert (
            chosen_input_ids.shape[1]
            == rejected_input_ids.shape[1]
            == chosen_attention_mask.shape[1]
            == rejected_attention_mask.shape[1]
        ), "All tensors must have the same max length"
        max_length = chosen_input_ids.shape[1]
        prompt_lengths: list[int] = experiences["prompt_lengths"]
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
        mean_loss, mean_chosen_reward, mean_rejected_reward = 0.0, 0.0, 0.0
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
                mean_loss += loss.item()
                mean_chosen_reward += chosen_reward.mean().item()
                mean_rejected_reward += rejected_reward.mean().item()
        mean_loss /= num_samples
        mean_chosen_reward /= num_samples
        mean_rejected_reward /= num_samples
        return mean_loss, mean_chosen_reward, mean_rejected_reward

    def _dpo_loss(
        self,
        batch_size: int,
        minibatch_idxs: np.ndarray,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor | None,
        ref_chosen_log_probs: torch.Tensor | None,
        training: bool,
    ):
        """Calculates the DPO loss.

        :param batch_size: Batch size
        :type batch_size: int
        :param minibatch_idxs: Minibatch indices
        :type minibatch_idxs: torch.Tensor
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
        """Calculate the DPO loss.xw.

        :param chosen_mask: Mask for the prompt and padding tokens of the chosen completions
        :type chosen_mask: torch.Tensor
        :param rejected_mask: Mask for the prompt and padding tokens of the rejected completions
        :type rejected_mask: torch.Tensor
        :param chosen_log_probs: Log probabilities of the chosen completions
        :type chosen_log_probs: torch.Tensor
        :param rejected_log_probs: Log probabilities of the rejected completions
        :type rejected_log_probs: torch.Tensor
        :param ref_chosen_log_probs: Log probabilities of the chosen completions using the reference model

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

        def _get_hidden(ids, attn_mask):
            """Run forward pass; capture the input to lm_head via a pre-hook."""
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
            with self.select_policy(use_reference=True):
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
        with self.select_policy(use_reference=False):
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
        def _make_target(ids, mask):
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

        loss, aux = _LigerDPOWithAlpha.apply(
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
            1,  # chunk_size
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

        :param log_probs: Log probabilities of the completions
        :type log_probs: torch.Tensor
        :param ref_log_probs: Log probabilities of the completions using the reference model
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
    ) -> np.ndarray:
        """Return the fitness (test) score of the agent.

        :param env: The environment to be tested in
        :type env: PreferenceGym environment
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 1
        :type loop: int, optional
        :return: Mean test score (numpy array)
        :rtype: np.ndarray
        """
        with env.eval_mode(), torch.no_grad():
            prompts = env.reset()
            rewards = []
            for _ in range(loop):
                _, chosen_reward, rejected_reward = self.learn(prompts, training=False)
                reward_margin = chosen_reward - rejected_reward
                rewards.append(np.asarray(reward_margin).item())
                prompts = env.step()
            mean_fit = float(np.mean(rewards))
        self.fitness.append(mean_fit)
        return np.array(mean_fit)
