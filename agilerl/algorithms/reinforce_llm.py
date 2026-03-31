from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import (
    LoraConfigProtocol,
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import ExperiencesType, LLMObsType
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    VLLMConfig,
    get_experiences_samples,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_utils import (
    ReasoningGym,
    masked_mean,
    move_params_to_cpu,
    move_params_to_gpu,
    pool_by_turns,
)

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig


class REINFORCE(LLMAlgorithm):
    """Turn-level REINFORCE with Return Batch Normalization (ReBN) for LLM
    finetuning.

    ReBN normalizes per-turn Monte Carlo returns across the entire batch of
    transitions. This gives per-turn credit assignment with arbitrary discount
    factors.

    Optionally uses PPO-style clipped surrogate objectives for safe multi-epoch
    updates (controlled by ``clip_coef`` and ``update_epochs``).

    :param pad_token_id: Pad token id.
    :type pad_token_id: int
    :param pad_token: Pad token string.
    :type pad_token: str
    :param model_name: Model name or path.
    :type model_name: str | None
    :param actor_network: Pre-instantiated HuggingFace model.
    :type actor_network: PreTrainedModelProtocol | None
    :param model_config: Model configuration dict.
    :type model_config: dict[str, Any] | None
    :param hp_config: RL hyperparameter mutation configuration.
    :type hp_config: HyperparameterConfig | None
    :param index: Instance index for tournament selection.
    :type index: int
    :param batch_size: Mini-batch size for learning.
    :type batch_size: int
    :param beta: KL penalty coefficient against the reference policy.
    :type beta: float
    :param clip_coef: PPO-style surrogate clipping coefficient.
    :type clip_coef: float
    :param gamma: Discount factor for multi-turn returns.
    :type gamma: float
    :param lr: Learning rate for the actor optimizer.
    :type lr: float
    :param max_grad_norm: Maximum gradient norm for clipping.
    :type max_grad_norm: float
    :param update_epochs: Number of policy update epochs per batch.
    :type update_epochs: int
    :param temperature: Sampling temperature for generation.
    :type temperature: float
    :param repetition_penalty: Repetition penalty for generation.
    :type repetition_penalty: float
    :param top_p: Top-p (nucleus) sampling parameter.
    :type top_p: float
    :param top_k: Top-k sampling parameter.
    :type top_k: int
    :param min_p: Min-p sampling parameter.
    :type min_p: float
    :param use_separate_reference_adapter: Use a dedicated LoRA adapter for
        the frozen reference policy.
    :type use_separate_reference_adapter: bool
    :param calc_position_embeddings: Calculate position embeddings explicitly.
    :type calc_position_embeddings: bool
    :param micro_batch_size_per_gpu: Micro-batch size for gradient accumulation.
    :type micro_batch_size_per_gpu: int | None
    :param reduce_memory_peak: Reduce memory peak in log-prob computation.
    :type reduce_memory_peak: bool
    :param max_output_tokens: Maximum new tokens per generation.
    :type max_output_tokens: int | None
    :param min_output_tokens: Minimum new tokens per generation.
    :type min_output_tokens: int | None
    :param max_model_len: Maximum context window length.
    :type max_model_len: int | None
    :param lora_config: LoRA adapter configuration.
    :type lora_config: LoraConfigProtocol | None
    :param cosine_lr_schedule_config: Cosine LR schedule configuration.
    :type cosine_lr_schedule_config: CosineLRScheduleConfig | None
    :param accelerator: HuggingFace Accelerator for distributed training.
    :type accelerator: Accelerator | None
    :param device: Device string.
    :type device: str
    :param wrap: Wrap models for distributed training upon creation.
    :type wrap: bool
    :param clone: Whether this is a clone instantiation.
    :type clone: bool
    :param use_vllm: Use vLLM for generation.
    :type use_vllm: bool
    :param vllm_config: vLLM configuration.
    :type vllm_config: VLLMConfig | None
    :param seed: Random seed.
    :type seed: int
    :param gradient_checkpointing: Enable gradient checkpointing.
    :type gradient_checkpointing: bool
    :param torch_compiler: Torch compiler mode.
    :type torch_compiler: str | None
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_token: str,
        model_name: str | None = None,
        actor_network: Any | None = None,
        model_config: dict[str, Any] | None = None,
        hp_config: HyperparameterConfig | None = None,
        index: int = 0,
        batch_size: int = 16,
        beta: float = 0.01,
        clip_coef: float = 0.2,
        gamma: float = 1.0,
        lr: float = 5e-7,
        max_grad_norm: float = 1.0,
        update_epochs: int = 1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        use_separate_reference_adapter: bool = True,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        reduce_memory_peak: bool = False,
        max_output_tokens: int | None = 1024,
        min_output_tokens: int | None = None,
        max_model_len: int | None = None,
        lora_config: LoraConfigProtocol | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        accelerator: Accelerator | None = None,
        device: str = "cpu",
        wrap: bool = True,
        clone: bool = False,
        use_vllm: bool = False,
        vllm_config: VLLMConfig | None = None,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
    ) -> None:

        device = (
            f"cuda:{accelerator.process_index}"
            if accelerator is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
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
            use_value_head=False,
            use_liger_loss=False,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            hp_config=hp_config,
            wrap=wrap,
            device=device,
            accelerator=accelerator,
            name="LLMReinforce",
            gradient_checkpointing=gradient_checkpointing,
            torch_compiler=torch_compiler,
        )
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(clip_coef, (float, int)), (
            "Clipping coefficient must be a float."
        )
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(update_epochs, int), (
            "Policy update epochs must be an integer."
        )
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        if clone and actor_network is not None:
            assert isinstance(
                actor_network,
                (PeftModelProtocol, PreTrainedModelProtocol),
            ), "Actor network must be a PeftModelProtocol or PreTrainedModelProtocol"
        if max_output_tokens is None and max_model_len is None:
            msg = "Either max_output_tokens or max_model_len must be specified"
            raise ValueError(msg)

        self.beta = beta
        self.clip_coef = clip_coef
        self.gamma = gamma
        self.update_epochs = update_epochs
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_output_tokens = max_output_tokens
        self.min_output_tokens = min_output_tokens
        self.max_model_len = (
            max_model_len if max_model_len is not None else max_output_tokens + 512
        )
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_length=self.max_model_len,
            max_new_tokens=max_output_tokens,
            min_new_tokens=min_output_tokens,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        self.use_vllm = use_vllm
        self.vllm_config = vllm_config
        if self.use_vllm:
            self._configure_vllm()
        self._initialize_actors(actor_network, not clone)

        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

        unwrapped_model = (
            self.accelerator.unwrap_model(self.actor)
            if self.accelerator is not None
            else self.actor
        )
        if self.use_vllm:
            move_params_to_cpu(unwrapped_model)
            self.llm.wake_up()
            self._move_model_to_vllm()

    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return generated completion ids and corresponding action masks."""
        prompt_batch = [obs] if isinstance(obs, dict) else list(obs)

        with self.select_adapter("actor"):
            self.actor.eval()
            if not self.use_vllm:
                actor_module = self._get_unwrapped_actor()
                try:
                    actor_device = next(actor_module.parameters()).device
                except StopIteration:
                    actor_device = torch.device(self.device)
                with torch.inference_mode(), self._amp_ctx():
                    completion_ids = []
                    action_masks = []
                    for raw in prompt_batch:
                        if not isinstance(raw, dict):
                            msg = (
                                "Each HuggingFace observation must be a dict with "
                                "input_ids and attention_mask; got "
                                f"{type(raw).__name__}. For string-observation GEM "
                                "envs, wrap the env with TokenObservationWrapper."
                            )
                            raise TypeError(
                                msg,
                            )
                        prompt = dict(raw)
                        prompt.pop("text", None)
                        prompt.pop("model_text", None)
                        stitch = prompt.pop("stitch_prefix_ids", None)
                        if stitch is not None and stitch.shape[1] > 0:
                            msg = (
                                "LLMReinforce HuggingFace generation does not support "
                                "sliding-window stitch_prefix_ids; use LLMPPO, vLLM, "
                                "or disable sliding-window fields on observations."
                            )
                            raise ValueError(
                                msg,
                            )
                        model_input_ids = prompt.pop("model_input_ids", None)
                        model_attention_mask = prompt.pop(
                            "model_attention_mask",
                            None,
                        )
                        prompt.pop("model_window_initial_len", None)
                        if model_input_ids is not None:
                            prompt["input_ids"] = model_input_ids.to(actor_device)
                            if model_attention_mask is not None:
                                prompt["attention_mask"] = model_attention_mask.to(
                                    actor_device,
                                )
                        else:
                            prompt["input_ids"] = prompt["input_ids"].to(actor_device)
                            prompt["attention_mask"] = prompt["attention_mask"].to(
                                actor_device,
                            )
                        prompt_len = prompt["input_ids"].shape[1]
                        completion_id = self.actor.generate(
                            **prompt,
                            generation_config=self.generation_config,
                        )
                        completion_ids.append(completion_id)
                        action_mask = torch.zeros_like(
                            completion_id,
                            dtype=torch.bool,
                            device=completion_id.device,
                        )
                        action_mask[:, prompt_len:] = True
                        action_mask[completion_id == self.pad_token_id] = False
                        action_mask = action_mask[:, 1:]
                        action_masks.append(action_mask)
            else:
                completion_ids, action_masks = self._generate_with_vllm_colocate(
                    prompt_batch,
                    1,
                )

        return completion_ids, action_masks

    def learn(
        self,
        experiences: ExperiencesType,
        tokenizer,
        turn_ids: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float, float]:
        """Update actor using REINFORCE with Return Batch Normalization.

        :param experiences: Tuple of (completion_ids, action_masks, rewards).
            For single-turn, rewards is a flat list/tensor of scalars.
            For multi-turn, rewards should be [batch, max_turns] per-turn rewards.
        :param tokenizer: Tokenizer (unused, kept for API compatibility with LLMPPO).
        :param turn_ids: Optional [batch, seq_len] tensor mapping each token
            to its turn index (0-indexed). -1 for non-action tokens.
            When None, defaults to all action tokens belonging to turn 0.
        :return: (loss, kl, pg_loss, 0.0, entropy) — critic_loss is always 0.
        """
        with self.memory_efficient_params():
            completion_ids, action_masks, rewards = stack_and_pad_experiences(
                *experiences,
                padding_values=[self.pad_token_id, False, None],
            )
            completion_ids = completion_ids.to(self.device)
            action_masks = action_masks.to(self.device)
            action_mask_bool = action_masks.bool()
            num_samples = completion_ids.shape[0]

            if turn_ids is None:
                turn_ids = torch.where(
                    action_mask_bool,
                    torch.zeros_like(action_masks, dtype=torch.long),
                    torch.full_like(action_masks, -1, dtype=torch.long),
                )
                rewards_2d = rewards.flatten().to(self.device).float().unsqueeze(-1)
            else:
                turn_ids = turn_ids.to(self.device)
                rewards_2d = rewards.to(self.device).float()
                if rewards_2d.dim() == 1:
                    rewards_2d = rewards_2d.unsqueeze(-1)

            del rewards

            batch_idxs = np.arange(num_samples)
            batch_size = (
                min(num_samples, self.micro_batch_size_per_gpu)
                if hasattr(self, "micro_batch_size_per_gpu")
                else num_samples
            )
            mean_pg_loss, mean_loss, mean_kl, mean_entropy, updates = (
                0.0,
                0.0,
                0.0,
                0.0,
                0,
            )

            with torch.inference_mode():
                reference_log_probs = self._get_logprobs(
                    completion_ids,
                    batch_size=batch_size,
                    use_reference=True,
                    eval_mode=True,
                )
                old_log_probs = self._get_logprobs(
                    completion_ids,
                    batch_size=batch_size,
                    use_reference=False,
                    eval_mode=True,
                )

                token_rewards = self._compute_token_rewards(
                    action_masks, rewards_2d, turn_ids
                )

                old_log_probs = torch.masked_fill(old_log_probs, ~action_mask_bool, 1.0)
                reference_log_probs = torch.masked_fill(
                    reference_log_probs, ~action_mask_bool, 1.0
                )
                token_penalised_rewards = token_rewards - self.beta * (
                    old_log_probs - reference_log_probs
                )

                advantages = self._compute_rebn_advantages(
                    token_penalised_rewards, action_masks, turn_ids
                )
                del token_rewards, token_penalised_rewards

            self.actor.train()
            for _epoch_idx in range(self.update_epochs):
                self.rng.shuffle(batch_idxs)
                for start in range(0, num_samples, batch_size):
                    minibatch_idxs = batch_idxs[
                        start : min((start + batch_size), num_samples)
                    ]
                    (
                        batch_ids,
                        batch_action_mask,
                        batch_old_log_probs,
                        batch_reference_log_probs,
                        batch_advantages,
                    ) = get_experiences_samples(
                        minibatch_idxs,
                        completion_ids,
                        action_masks,
                        old_log_probs,
                        reference_log_probs,
                        advantages,
                    )

                    batch_mask_bool = batch_action_mask.bool()

                    with self.select_adapter("actor"):
                        batch_log_probs = self._get_logprobs(
                            batch_ids,
                            batch_size=batch_size,
                            use_reference=False,
                            eval_mode=False,
                        )
                    batch_log_probs = torch.masked_fill(
                        batch_log_probs, ~batch_mask_bool, 1.0
                    )

                    kl = batch_log_probs - batch_reference_log_probs
                    masked_entropy = masked_mean(
                        -batch_log_probs.detach(), batch_action_mask
                    )

                    policy_ratio = torch.exp(
                        batch_log_probs - batch_old_log_probs,
                    )
                    clipped_ratio = torch.clamp(
                        policy_ratio,
                        1 - self.clip_coef,
                        1 + self.clip_coef,
                    )
                    pg_loss_unclipped = -batch_advantages * policy_ratio
                    pg_loss_clipped = -batch_advantages * clipped_ratio
                    pg_loss = masked_mean(
                        torch.max(pg_loss_unclipped, pg_loss_clipped), batch_action_mask
                    )

                    self._backward_pass(pg_loss)

                    mean_kl += masked_mean(kl, batch_action_mask).item()
                    mean_entropy += masked_entropy.mean().item()
                    mean_pg_loss += pg_loss.mean().item()
                    mean_loss += pg_loss.item()
                    updates += 1

        return (
            mean_loss / max(updates, 1),
            mean_kl / max(updates, 1),
            mean_pg_loss / max(updates, 1),
            0.0,
            mean_entropy / max(updates, 1),
        )

    def test(
        self,
        env: ReasoningGym,
        loop: int = 1,
    ) -> torch.Tensor:
        """Return fitness (test) score tensor of llm on test sub-set."""
        with env.eval_mode(), torch.inference_mode():
            prompts = env.reset()
            rewards = []
            for _ in range(loop):
                completion_ids, _ = self.get_action(prompts, training=False)
                next_prompts, reward = env.step(completion_ids)
                prompts = next_prompts
                rewards.append(reward)
        reward_tensor = torch.cat(rewards)
        mean_fit = torch.mean(reward_tensor).item()
        self.fitness.append(mean_fit)
        return reward_tensor

    def _compute_rebn_advantages(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute Return Batch Normalization (ReBN) advantages.

        For each turn, computes the discounted Monte Carlo return G_t, then
        z-scores all per-turn returns across the batch to produce advantages.
        Advantages are broadcast back to token level for the policy gradient.

        :param rewards: [batch, seq_len] per-token rewards (already assigned
            from per-turn rewards by ``_compute_token_rewards``).
        :param action_mask: [batch, seq_len] float mask of action positions.
        :param turn_ids: [batch, seq_len] turn index per token, -1 for non-action.
        :param eps: Epsilon for numerical stability.
        :return: [batch, seq_len] token-level advantages.
        """
        batch_size = rewards.shape[0]
        num_turns = turn_ids.max().item() + 1

        turn_rewards = pool_by_turns(rewards, turn_ids, num_turns)

        per_sample_num_turns = turn_ids.max(dim=1).values + 1

        # Compute Monte Carlo returns: G_t = r_t + gamma * G_{t+1}
        turn_returns = torch.zeros_like(turn_rewards)
        for t in reversed(range(num_turns)):
            is_last_turn = t >= (per_sample_num_turns - 1)
            if t == num_turns - 1:
                next_return = torch.zeros(batch_size, device=rewards.device)
            else:
                next_return = turn_returns[:, t + 1]
            next_return = torch.where(
                is_last_turn, torch.zeros_like(next_return), next_return
            )
            turn_returns[:, t] = turn_rewards[:, t] + self.gamma * next_return

        # ReBN: z-score returns across all valid (sample, turn) pairs
        valid_mask = torch.zeros_like(turn_returns, dtype=torch.bool)
        for t in range(num_turns):
            valid_mask[:, t] = per_sample_num_turns > t

        valid_returns = turn_returns[valid_mask]
        if valid_returns.numel() > 1:
            mean_g = valid_returns.mean()
            std_g = valid_returns.std() + eps
            normalized_returns = (turn_returns - mean_g) / std_g
        else:
            normalized_returns = torch.zeros_like(turn_returns)

        # Broadcast turn-level advantages to token level
        token_advantages = torch.zeros_like(rewards)
        for t in range(num_turns):
            mask_t = (turn_ids == t).float()
            token_advantages += mask_t * normalized_returns[:, t : t + 1]

        return token_advantages * action_mask

    def _compute_token_rewards(
        self,
        action_mask: torch.Tensor,
        rewards: torch.Tensor,
        turn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Assign per-turn rewards to each action token based on turn_ids.

        :param action_mask: [batch, seq_len] bool mask of action positions.
        :param rewards: [batch, max_turns] per-turn reward scalars.
        :param turn_ids: [batch, seq_len] turn index per token (-1 for non-action).
        :return: [batch, seq_len] per-token rewards.
        """
        num_turns = rewards.shape[1]
        token_rewards = torch.zeros_like(action_mask, dtype=torch.float)
        for t in range(num_turns):
            mask_t = (turn_ids == t).float()
            token_rewards += mask_t * rewards[:, t : t + 1]
        return token_rewards

    def _calculate_kl_divergence(
        self,
        log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-token reverse-KL-style penalty term."""
        return (
            torch.exp(reference_log_probs - log_probs)
            - (reference_log_probs - log_probs)
            - 1
        )

    def _get_unwrapped_actor(self) -> Any:
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.actor)
        return self.actor

    @contextmanager
    def memory_efficient_params(self) -> None:
        """Memory efficient params context manager for vLLM colocate mode."""
        vllm_cfg = getattr(self, "vllm_config", None)
        use_vllm = getattr(self, "use_vllm", False)
        sleep_mode = use_vllm and vllm_cfg is not None and vllm_cfg.sleep_mode
        if sleep_mode:
            self.llm.sleep(level=2)
            unwrapped_model = (
                self.accelerator.unwrap_model(self.actor)
                if self.accelerator is not None
                else self.actor
            )
            move_params_to_gpu(unwrapped_model, self.device)
        yield
        if sleep_mode:
            move_params_to_cpu(unwrapped_model)
            self.llm.wake_up()
            self._move_model_to_vllm()
