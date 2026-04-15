import warnings
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOFunction

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import (
    LoraConfigProtocol,
    MultiTurnEpisodeEnv,
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
    move_params_to_cpu,
    normalize_reasoning_prompt_batch,
    prepare_prompt_hf_generate,
    stitch_completion_after_windowed_hf_generate,
)

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig


class GRPO(LLMAlgorithm):
    """The GRPO algorithm class. GRPO paper: https://arxiv.org/pdf/2402.03300.

    :param pad_token_id: Pad token id
    :type pad_token_id: int
    :param pad_token: Pad token
    :type pad_token: str
    :param model_name: Model name
    :type model_name: str, optional
    :param actor_network: HuggingFace LLM
    :type actor_network: PreTrainedModelProtocol
    :param model_config: Model configuration, to be used when creating the model from a name or path
    :type model_config: dict[str, Any], optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param batch_size: Mini-batch size for learning, defaults to 64
    :type batch_size: int, optional
    :param beta: Beta coefficient, controls the strength of the KL divergence penalty, defaults to 0.001
    :type beta: float, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param clip_coef: Surrogate clipping coefficient, defaults to 0.2
    :type clip_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.1
    :type max_grad_norm: float, optional
    :param update_epochs: Number of policy update epochs, defaults to 4
    :type update_epochs: int, optional
    :param group_size: Group size, defaults to 8
    :type group_size: int, optional
    :param temperature: Temperature, controls randomness of text generation
    :type temperature: float, optional
    :param calc_position_embeddings: Flag indicating whether to calculate position embeddings, defaults to True
    :type calc_position_embeddings: bool, optional
    :param micro_batch_size_per_gpu: If specified, gradient_accumulation_steps will be
        calculated to achieve the target batch_size. If None, uses existing
        gradient_accumulation_steps from DeepSpeed config, defaults to None
    :type micro_batch_size_per_gpu: int, optional
    :param max_output_tokens: Max number of answer tokens, defaults to None
    :type max_output_tokens: int, optional
    :param min_output_tokens: Minimum output tokens, defaults to 0
    :type min_output_tokens: int, optional
    :param max_model_len: Maximum context window length, defaults to 1024
    :type max_model_len: int, optional
    :param lora_config: Config for LoRA, defaults to None
    :type lora_config: LoraConfigProtocol, optional
    :param cosine_lr_schedule_config: Config for cosine lr scheduling, defaults to None
    :type cosine_lr_schedule_config: CosineLRScheduleConfig, optional
    :param use_memory_efficient_params: Use memory efficient params.
    :type use_memory_efficient_params: bool
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    :param clone: Flag to indicate if the instantiation is a cloning, defaults to False
    :type clone: bool, optional
    :param use_separate_reference_adapter: Flag to indicate if the reference policy should have a separate adapter, defaults to False
    :type use_separate_reference_adapter: bool, optional
    :param use_vllm: Flag to indicate if the model should use vllm for generation, defaults to False
    :type use_vllm: bool, optional
    :param vllm_config: Config for VLLM generation, defaults to None
    :type vllm_config: VLLMConfig, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    :param gradient_checkpointing: Flag to indicate if gradient checkpointing should be used, defaults to True
    :type gradient_checkpointing: bool, optional
    :param torch_compiler: Torch compile mode (e.g. ``'default'``), defaults to None
    :type torch_compiler: str | None, optional
    :param use_liger_loss: Use Liger kernel for memory-efficient loss computation, defaults to False.
        Requires ``liger_kernel`` to be installed; pass ``False`` to fall back to the standard PyTorch path.
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
        beta: float = 0.001,
        lr: float = 5e-7,
        clip_coef: float = 0.2,
        max_grad_norm: float = 0.1,
        update_epochs: int = 1,
        group_size: int = 8,
        temperature: float = 0.9,
        repetition_penalty: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        use_memory_efficient_params: bool = True,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        max_output_tokens: int | None = None,
        min_output_tokens: int | None = None,
        max_model_len: int | None = 1024,
        hf_generate_chunk_size: int | None = None,
        lora_config: LoraConfigProtocol | None = None,
        cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
        accelerator: Accelerator | None = None,
        device: str = "cpu",
        wrap: bool = True,
        clone: bool = False,
        use_separate_reference_adapter: bool = False,
        use_vllm: bool = False,
        vllm_config: VLLMConfig | None = None,
        seed: int = 42,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
        use_liger_loss: bool = False,
        use_kl_advantage_shaping: bool = False,
        adv_norm: str = "mean_std",
        importance_sampling_level: str = "token",
        whiten_advantages: bool = False,
        adv_clip_range: float | None = None,
        filter_zero_advantages: bool = False,
        advantage_filter_eps: float = 0.0,
    ) -> None:

        device = (
            f"cuda:{accelerator.process_index}" if accelerator is not None else device
        )
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
            use_memory_efficient_params=use_memory_efficient_params,
            use_liger_loss=use_liger_loss,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            use_vllm=use_vllm,
            vllm_config=vllm_config,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            wrap=wrap,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            name="GRPO",
            gradient_checkpointing=gradient_checkpointing,
            torch_compiler=torch_compiler,
        )
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(
            clip_coef,
            (float, int),
        ), "Clipping coefficient must be a float."
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(
            update_epochs,
            int,
        ), "Policy update epochs must be an integer."
        assert update_epochs >= 1, (
            "Policy update epochs must be greater than or equal to one."
        )
        if actor_network is not None:
            assert isinstance(
                actor_network,
                (PeftModelProtocol, PreTrainedModelProtocol),
            ), "Actor network must be a PeftModelProtocol or PreTrainedModelProtocol"
        self.use_liger_loss = use_liger_loss
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.group_size = group_size
        self.beta = beta
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        if adv_norm not in {"mean_std", "mean_only"}:
            msg = (
                f"Invalid adv_norm '{adv_norm}'. Expected one of "
                "['mean_std', 'mean_only']."
            )
            raise ValueError(msg)
        self.adv_norm = adv_norm
        if importance_sampling_level not in {"token", "sequence", "average"}:
            msg = (
                f"Invalid importance_sampling_level '{importance_sampling_level}'. "
                "Expected one of ['token', 'sequence', 'average']."
            )
            raise ValueError(msg)
        if adv_clip_range is not None and adv_clip_range <= 0:
            msg = "adv_clip_range must be > 0 when provided."
            raise ValueError(msg)
        if advantage_filter_eps < 0:
            msg = "advantage_filter_eps must be >= 0."
            raise ValueError(msg)
        self.importance_sampling_level = importance_sampling_level
        self.whiten_advantages = whiten_advantages
        self.adv_clip_range = adv_clip_range
        self.filter_zero_advantages = filter_zero_advantages
        self.advantage_filter_eps = advantage_filter_eps
        if use_liger_loss and use_kl_advantage_shaping:
            warnings.warn(
                "use_kl_advantage_shaping is not supported with use_liger_loss=True; "
                "disabling KL advantage shaping.",
                stacklevel=2,
            )
            use_kl_advantage_shaping = False
        if use_liger_loss and self.importance_sampling_level != "token":
            warnings.warn(
                "importance_sampling_level!='token' is not supported with "
                "use_liger_loss=True; falling back to 'token'.",
                stacklevel=2,
            )
            self.importance_sampling_level = "token"
        self.use_kl_advantage_shaping = use_kl_advantage_shaping
        if max_output_tokens is None and max_model_len is None:
            msg = "Either max_output_tokens or max_model_len must be specified"
            raise ValueError(
                msg,
            )
        self.max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else max_model_len
        )
        self.min_output_tokens = min_output_tokens
        self.max_model_len = (
            max_model_len if max_model_len is not None else max_output_tokens
        )
        self.hf_generate_chunk_size = int(
            1 if hf_generate_chunk_size is None else max(1, hf_generate_chunk_size)
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

        if self.use_vllm:
            self._configure_vllm()
        self._initialize_actors(actor_network, not clone)
        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
        repeat_prompts: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return generated completions for each prompt (GRPO groups when training).

        :param obs: List of HF-style prompt dicts (this implementation mutates them).
        :type obs: LLMObsType
        :param training: If ``True``, generate with training sampling settings.
        :type training: bool
        :param repeat_prompts: If ``True`` and ``training=True``, duplicate each
            prompt ``self.group_size`` times (legacy GRPO grouped mode). If
            ``False``, treat the batch as already expanded trajectories.
        :type repeat_prompts: bool
        :return: Completion token IDs and per-sequence action masks.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
        prompt_batch = normalize_reasoning_prompt_batch(obs)
        group_size = self.group_size if training and repeat_prompts else 1
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
                    completion_masks = []

                    for start in range(
                        0,
                        len(prompt_batch),
                        self.hf_generate_chunk_size,
                    ):
                        chunk = prompt_batch[
                            start : start + self.hf_generate_chunk_size
                        ]
                        for prompt_dict in chunk:
                            prompt = prepare_prompt_hf_generate(
                                prompt_dict, actor_device
                            )
                            if training and group_size > 1:
                                prompt["input_ids"] = prompt["input_ids"].repeat(
                                    group_size,
                                    1,
                                )
                                prompt["attention_mask"] = prompt[
                                    "attention_mask"
                                ].repeat(
                                    group_size,
                                    1,
                                )
                            stitch_ids = prompt.pop("stitch_prefix_ids", None)
                            if (
                                stitch_ids is not None
                                and training
                                and group_size > 1
                                and stitch_ids.shape[0] == 1
                            ):
                                stitch_ids = stitch_ids.repeat(group_size, 1)
                            initial_prompt_len = prompt.pop("initial_prompt_len", None)
                            completion_id = self.actor.generate(
                                **prompt,
                                generation_config=self.generation_config,
                            )
                            completion_id, full_prompt_len = (
                                stitch_completion_after_windowed_hf_generate(
                                    completion_id,
                                    stitch_ids,
                                    initial_prompt_len,
                                )
                            )
                            completion_ids.append(completion_id)
                            completion_mask = torch.zeros_like(
                                completion_id,
                                dtype=torch.bool,
                                device=completion_id.device,
                            )
                            completion_mask[:, full_prompt_len:] = True
                            completion_mask[completion_id == self.pad_token_id] = False
                            completion_mask = completion_mask[:, 1:]
                            completion_masks.append(completion_mask)
            else:
                self._prepare_vllm_for_generation()
                completion_ids, completion_masks = self._generate_with_vllm_colocate(
                    prompt_batch,
                    group_size,
                    temperature=self.temperature
                    if training
                    else 0.01,  # Almost deterministic for evaluation
                )

        return completion_ids, completion_masks

    def learn(
        self,
        experiences: ExperiencesType,
    ) -> dict[str, float]:
        """Update agent network parameters to learn from experiences.

        :param experiences: ``(completion_ids, action_masks, rewards)`` stacked batch.
        :type experiences: ExperiencesType
        :return: Dict with keys ``mean_loss`` and ``mean_kl``, averaged over the update.
        :rtype: dict[str, float]
        """
        self._prepare_vllm_for_training()

        with self.memory_efficient_params_context():
            completion_ids, action_masks, rewards = stack_and_pad_experiences(
                *experiences,
                padding_values=[self.pad_token_id, False, None],
            )
            action_masks = action_masks.to(self.device)
            rewards = rewards.to(self.device).float()
            completion_ids = completion_ids.to(self.device)
            # GRPO expects one scalar reward per trajectory. If callers pass
            # per-turn rewards [batch, max_turns], collapse to episode returns.
            if rewards.dim() > 1 and rewards.shape[0] == completion_ids.shape[0]:
                rewards = rewards.sum(dim=1)
            rewards = rewards.flatten()
            if rewards.shape[0] != completion_ids.shape[0]:
                msg = (
                    "Rewards must provide one scalar per trajectory after collapse: "
                    f"got rewards={tuple(rewards.shape)} and "
                    f"completion_ids={tuple(completion_ids.shape)}."
                )
                raise ValueError(msg)

            num_samples = rewards.shape[0]
            if num_samples % self.group_size != 0:
                msg = (
                    f"Batch size ({num_samples}) must be divisible by "
                    f"group_size ({self.group_size}) for GRPO."
                )
                raise ValueError(msg)

            advantages = self._calculate_advantage(rewards).to(self.device)
            active_adv_mask = None
            if self.filter_zero_advantages:
                active_adv_mask = (
                    advantages.squeeze(-1).abs() > self.advantage_filter_eps
                )
            if self.whiten_advantages:
                advantages = advantages.squeeze(-1)
                if active_adv_mask is not None and active_adv_mask.any():
                    active_advantages = advantages[active_adv_mask]
                    whitened_active = (active_advantages - active_advantages.mean()) / (
                        active_advantages.std(unbiased=False) + 1e-8
                    )
                    advantages[active_adv_mask] = whitened_active
                else:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(unbiased=False) + 1e-8
                    )
                advantages = advantages.unsqueeze(-1)
            if self.adv_clip_range is not None:
                advantages = advantages.clamp(-self.adv_clip_range, self.adv_clip_range)

            if active_adv_mask is not None:
                batch_idxs = np.where(active_adv_mask.detach().cpu().numpy())[0]
                if batch_idxs.size == 0:
                    warnings.warn(
                        "All samples were filtered by advantage threshold; skipping GRPO update.",
                        stacklevel=2,
                    )
                    return {"mean_loss": 0.0, "mean_kl": 0.0}
            else:
                batch_idxs = np.arange(num_samples)
            learn_metrics = {
                "mean_loss": 0.0,
                "mean_kl": 0.0,
            }
            updates = 0
            batch_size = (
                min(num_samples, self.micro_batch_size_per_gpu)
                if hasattr(self, "micro_batch_size_per_gpu")
                else num_samples
            )

            with torch.no_grad():
                reference_log_probs, old_log_probs, _ = self._fused_forward_no_grad(
                    completion_ids,
                    batch_size,
                )

            effective_num_samples = len(batch_idxs)
            if effective_num_samples == 0:
                warnings.warn(
                    "No active samples after filtering; skipping GRPO update.",
                    stacklevel=2,
                )
                return {"mean_loss": 0.0, "mean_kl": 0.0}

            # Ensure batch_size is not larger than the number of active samples
            batch_size = min(batch_size, effective_num_samples)

            for _ in range(self.update_epochs):
                self.rng.shuffle(batch_idxs)
                for start in range(0, effective_num_samples, batch_size):
                    minibatch_idxs = batch_idxs[
                        start : min((start + batch_size), effective_num_samples)
                    ]
                    if len(minibatch_idxs) == 0:
                        continue
                    loss, kl = self._grpo_loss(
                        batch_size,
                        minibatch_idxs,
                        completion_ids,
                        action_masks,
                        advantages,
                        old_log_probs,
                        reference_log_probs,
                    )
                    if not loss.isfinite():
                        msg = f"Loss is not finite: {loss}"
                        raise ValueError(msg)
                    self._backward_pass(loss)
                    learn_metrics["mean_loss"] += loss.item()
                    learn_metrics["mean_kl"] += kl.item()
                    updates += 1
        return {
            metric: value / max(updates, 1) for metric, value in learn_metrics.items()
        }

    def test(
        self,
        env: ReasoningGym | MultiTurnEpisodeEnv,
        loop: int = 1,
    ) -> torch.Tensor:
        """Return fitness (test) score tensor of llm on test sub-set.

        :param env: Dataset-style ``ReasoningGym`` environment or tokenized
            multi-turn episode environment.
        :type env: ReasoningGym | MultiTurnEpisodeEnv
        :param loop: Number of outer test iterations over ``reset`` / ``step``.
        :type loop: int
        :return: Concatenated reward tensor from the test loop.
        :rtype: torch.Tensor
        """
        eval_context = getattr(env, "eval_mode", nullcontext)
        with eval_context(), torch.no_grad():
            if isinstance(env, ReasoningGym):
                prompts = env.reset()
                rewards = []
                for _ in range(loop):
                    completion_ids, _ = self.get_action(prompts, training=False)
                    next_prompts, reward = env.step(completion_ids)
                    prompts = next_prompts
                    rewards.append(reward)
                reward_tensor = torch.cat(rewards)
            elif isinstance(env, MultiTurnEpisodeEnv):
                all_rewards: list[torch.Tensor] = []
                for _ in range(loop):
                    prompt_dict, _info = env.reset()
                    terminated, truncated = False, False
                    while not terminated and not truncated:
                        completion_ids, _ = self.get_action(
                            [prompt_dict],
                            training=False,
                        )
                        full = completion_ids[0]
                        prompt_dict, reward, terminated, truncated, _step_info = env.step(
                            full,
                        )
                        all_rewards.append(
                            torch.tensor(
                                [float(reward)],
                                dtype=torch.float32,
                                device=full.device,
                            )
                        )
                reward_tensor = torch.cat(all_rewards)
            else:
                msg = (
                    "env must be a ReasoningGym (or subclass) or "
                    f"MultiTurnEpisodeEnv; got {type(env).__name__}"
                )
                raise TypeError(msg)
        mean_fit = torch.mean(reward_tensor).item()
        self.fitness.append(mean_fit)
        return reward_tensor

    def _calculate_advantage(
        self,
        rewards: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Calculate the group relative advantage for each groups reward.

        :param rewards: Tensor of rewards.
        :type rewards: torch.Tensor
        :param eps: Epsilon to prevent zero division error, defaults to 1e-8
        :type eps: float, optional
        :return: Tensor of group relative advantages.
        :rtype: torch.Tensor
        """
        rewards = rewards.view(-1, self.group_size)
        if self.adv_norm == "mean_only":
            advantage = rewards - rewards.mean(dim=1, keepdim=True)
        else:
            advantage = (rewards - rewards.mean(dim=1, keepdim=True)) / (
                rewards.std(dim=1, keepdim=True) + eps
            )
        return advantage.flatten().unsqueeze(1)

    def _calculate_kl_divergence(
        self,
        log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL divergence between the current and reference log probabilities.

        :param log_probs: Current policy log probabilities.
        :type log_probs: torch.Tensor
        :param reference_log_probs: Reference policy log probabilities.
        :type reference_log_probs: torch.Tensor
        :return: Kl divergence between the current and reference log probabilities.
        :rtype: torch.Tensor
        """
        return (
            torch.exp(reference_log_probs - log_probs)
            - (reference_log_probs - log_probs)
            - 1
        )

    def _grpo_loss(
        self,
        batch_size: int,
        minibatch_idxs: np.ndarray,
        completion_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a minibatch and compute the GRPO loss, dispatching to the
        Liger or standard implementation based on ``self.use_liger_loss``.

        :param batch_size: Micro-batch size used for log-prob computation.
        :type batch_size: int
        :param minibatch_idxs: Indices selecting the current minibatch.
        :type minibatch_idxs: np.ndarray
        :param completion_ids: Full completion token IDs.
        :type completion_ids: torch.Tensor
        :param action_mask: Full action mask.
        :type action_mask: torch.Tensor
        :param advantages: Full advantages tensor.
        :type advantages: torch.Tensor
        :param old_log_probs: Full old policy log probabilities.
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: Full reference policy log probabilities.
        :type reference_log_probs: torch.Tensor
        :return: Mean loss and mean KL divergence.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        (
            batch_ids,
            batch_action_mask,
            batch_advantages,
            batch_old_log_probs,
            batch_reference_log_probs,
        ) = get_experiences_samples(
            minibatch_idxs,
            completion_ids,
            action_mask,
            advantages,
            old_log_probs,
            reference_log_probs,
        )
        if self.use_liger_loss:
            return self._grpo_loss_liger(
                batch_ids,
                batch_action_mask,
                batch_advantages,
                batch_old_log_probs,
                batch_reference_log_probs,
            )
        batch_log_probs = self._get_logprobs(
            batch_ids,
            batch_size=batch_size,
            use_reference=False,
            eval_mode=False,
        )
        return self._grpo_loss_standard(
            batch_action_mask,
            batch_log_probs,
            batch_old_log_probs,
            batch_reference_log_probs,
            batch_advantages,
        )

    def _grpo_loss_standard(
        self,
        mask: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the GRPO loss using the standard PyTorch path.

        :param mask: Attention mask.
        :type mask: torch.Tensor
        :param log_probs: Log probabilities of the current policy.
        :type log_probs: torch.Tensor
        :param old_log_probs: Log probabilities of the old policy.
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: Log probabilities of the reference policy.
        :type reference_log_probs: torch.Tensor
        :param advantages: Advantages.
        :type advantages: torch.Tensor
        :return: Mean loss and mean KL divergence.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        kl = self._calculate_kl_divergence(log_probs, reference_log_probs)
        if self.use_kl_advantage_shaping:
            mask_f = mask.float()
            masked_kl = kl * mask_f
            avg_kl = masked_kl.sum(dim=-1, keepdim=True) / mask_f.sum(
                dim=-1, keepdim=True
            ).clamp(min=1.0)
            # ART-style zero-mean KL shaping of token advantages:
            # lower-then-average KL gets boosted, higher KL gets damped.
            advantages = advantages + self.beta * (avg_kl - masked_kl)
        token_log_ratio = log_probs - old_log_probs
        log_probs_ratio = torch.exp(token_log_ratio)
        if self.importance_sampling_level != "token":
            mask_f = mask.float()
            seq_log_ratio = (token_log_ratio * mask_f).sum(
                dim=-1, keepdim=True
            ) / mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            seq_log_probs_ratio = torch.exp(seq_log_ratio)
            if self.importance_sampling_level == "sequence":
                log_probs_ratio = seq_log_probs_ratio
            else:  # average
                log_probs_ratio = 0.5 * (log_probs_ratio + seq_log_probs_ratio)
        clipped_log_probs_ratio = log_probs_ratio.clamp(
            1 - self.clip_coef,
            1 + self.clip_coef,
        )
        surrogate = log_probs_ratio * advantages
        clipped_surrogate = clipped_log_probs_ratio * advantages
        loss = -torch.min(surrogate, clipped_surrogate)
        if not self.use_kl_advantage_shaping:
            loss = loss + self.beta * kl
        denominator = mask.sum(dim=-1)
        denominator = torch.where(
            denominator > 0,
            denominator,
            torch.ones_like(denominator),
        )
        loss = (loss * mask).sum(dim=-1) / denominator
        log_probs_ratio, clipped_log_probs_ratio, surrogate, clipped_surrogate = (
            None,
            None,
            None,
            None,
        )
        return loss.mean(), kl.mean()

    def _grpo_loss_liger(
        self,
        batch_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the GRPO loss using the Liger Triton-fused kernel.

        :param batch_ids: Input token IDs.
        :type batch_ids: torch.Tensor
        :param action_mask: Boolean action mask (B, seq_len-1).
        :type action_mask: torch.Tensor
        :param advantages: Per-sample advantages (B,) or (B, 1).
        :type advantages: torch.Tensor
        :param old_log_probs: Log probs from the frozen old policy (B, seq_len-1).
        :type old_log_probs: torch.Tensor
        :param reference_log_probs: Log probs from the reference policy (B, seq_len-1).
        :type reference_log_probs: torch.Tensor
        :return: Mean loss and mean KL divergence.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        batch_ids = batch_ids.to(self.device)
        mask = action_mask.to(self.device).contiguous()  # (B, seq_len-1)
        adv = advantages.squeeze(-1).to(self.device).contiguous()  # (B,)
        old_log_probs = old_log_probs.to(self.device).contiguous()
        reference_log_probs = (
            reference_log_probs.to(self.device).contiguous()
            if self.beta != 0.0
            else None
        )
        lm_head = self._get_lm_head()
        lm_head_weight = lm_head.weight
        lm_head_bias = lm_head.bias

        def _get_hidden(input_ids, attention_mask, use_cache=False, position_ids=None):
            """Run a forward pass and return hidden states fed into the language-model head.

            :param input_ids: Token IDs ``[batch, seq_len]``.
            :type input_ids: torch.Tensor
            :param attention_mask: Attention mask ``[batch, seq_len]``.
            :type attention_mask: torch.Tensor
            :param use_cache: Passed to the underlying model ``forward``.
            :type use_cache: bool
            :param position_ids: Optional explicit position IDs.
            :type position_ids: torch.Tensor | None
            :return: Hidden states immediately before the LM head ``[batch, seq_len, hidden]``.
            :rtype: torch.Tensor
            """
            captured = []
            hook = lm_head.register_forward_pre_hook(
                lambda m, inputs: captured.append(inputs[0])
            )
            try:
                self.actor(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    position_ids=position_ids,
                )
            finally:
                hook.remove()
            return captured[0]

        attention_mask = (batch_ids != self.pad_token_id).long()
        model_kwargs = {
            "input_ids": batch_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.calc_position_embeddings:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            model_kwargs["position_ids"] = position_ids
        with self.select_adapter("actor"):
            self.actor.train()
            policy_hidden = _get_hidden(**model_kwargs)  # (B, seq_len, H)
        target_ids = batch_ids[:, 1:].contiguous()  # (B, seq_len-1)
        loss, aux = LigerFusedLinearGRPOFunction.apply(
            policy_hidden,
            lm_head_weight,
            target_ids,
            mask,
            adv,
            lm_head_bias,
            reference_log_probs,
            old_log_probs,
            None,
            None,
            None,
            self.beta,
            self.clip_coef,
            self.clip_coef,
            "grpo",
            self.max_output_tokens,
            "token",  # Sequence for gspo when we implement it
            None,
            None,
            self.temperature,
            None,
            True,
            1,  # Chunk size
            None,
        )

        kl = aux[0]
        return loss.mean(), kl
