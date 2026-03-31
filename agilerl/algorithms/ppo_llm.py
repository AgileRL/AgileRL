from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from contextlib import nullcontext
from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm

from agilerl.algorithms.core.fused_lora import clear_fused_adapter_routing
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
    masked_whiten,
    move_params_to_cpu,
    move_params_to_gpu,
    pool_by_turns,
)

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig


class PPO(LLMAlgorithm):
    """Turn-level PPO for LLM finetuning with actor/reference adapters.

    Each generation sequence (turn) is treated as a single RL action.
    GAE discounts between turns, not between tokens within a turn.
    Single-turn is the special case where all action tokens share turn 0.
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
        vf_coef: float = 0.5,
        clip_coef: float = 0.2,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        lr_actor: float = 5e-7,
        lr_critic: float | None = 5e-5,
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
            lr=lr_actor,
            max_grad_norm=max_grad_norm,
            clone=clone,
            reduce_memory_peak=reduce_memory_peak,
            calc_position_embeddings=calc_position_embeddings,
            seed=seed,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            # Keep the standard PyTorch PPO path for now because value-head
            # training requires explicit hidden-state/value computation.
            use_value_head=True,
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
            name="LLMPPO",
            gradient_checkpointing=gradient_checkpointing,
            torch_compiler=torch_compiler,
        )
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr_actor, float), "Actor learning rate must be a float."
        assert lr_actor > 0, "Actor learning rate must be greater than zero."
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
            raise ValueError(
                msg,
            )

        self.beta = beta
        self.vf_coef = vf_coef
        self.clip_coef = clip_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
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

        self.lr_critic = lr_critic if lr_critic is not None else lr_actor

        self.use_vllm = use_vllm
        self.vllm_config = vllm_config
        if self.use_vllm:
            self._configure_vllm()
        self._initialize_actors(actor_network, not clone)
        self._apply_critic_lr()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

        # We call get_action before learn so we need to put the model to CPU and wake it up
        unwrapped_model = (
            self.accelerator.unwrap_model(self.actor)
            if self.accelerator is not None
            else self.actor
        )
        # Wake up LLM
        if self.use_vllm:
            move_params_to_cpu(unwrapped_model)
            self.llm.wake_up()
            self._move_model_to_vllm()

    @property
    def lr_actor(self) -> float:
        """Actor learning rate, aliased to the base-class ``self.lr``.

        Kept in sync automatically: mutations and lr-scheduler updates
        that modify ``self.lr`` are reflected here, and vice-versa.
        """
        return self.lr

    @lr_actor.setter
    def lr_actor(self, value: float) -> None:
        self.lr = value

    @staticmethod
    def _stitch_hf_completion_after_windowed_generate(
        completion_id: torch.Tensor,
        stitch: torch.Tensor,
        initial_len: int,
    ) -> torch.Tensor:
        """Reinsert dropped middle tokens after HF ``generate`` on a windowed prompt.

        ``completion_id`` is ``concat(model_input_ids, new_tokens)``. The full
        chronological sequence is
        ``concat(completion_id[:, :initial_len], stitch, completion_id[:, initial_len:])``.

        :param completion_id: Output of ``generate`` on the truncated prompt,
            shape ``(1, seq_len)``.
        :type completion_id: torch.Tensor
        :param stitch: Middle segment removed for context windowing, shape ``(1, K)``.
        :type stitch: torch.Tensor
        :param initial_len: ``model_window_initial_len`` (length of the initial
            user segment within ``model_input_ids``).
        :type initial_len: int
        :return: Full prompt plus generation with stitch restored.
        :rtype: torch.Tensor
        """
        stitch = stitch.to(completion_id.device, non_blocking=True)
        return torch.cat(
            [
                completion_id[:, :initial_len],
                stitch,
                completion_id[:, initial_len:],
            ],
            dim=1,
        )

    def _apply_critic_lr(self) -> None:
        """Set the critic param-group lr to ``self.lr_critic``.

        Called after optimizer creation and after any optimizer reinit
        so the critic group always uses its own learning rate.
        """
        opt = self.optimizer.optimizer
        if hasattr(opt, "param_groups") and len(opt.param_groups) > 1:
            opt.param_groups[1]["lr"] = self.lr_critic

    def _reinit_opt_from_config(self, config) -> None:
        """Reinit optimizer, then re-apply the critic learning rate.

        The base-class reinit sets all param groups to ``self.lr``
        (the actor lr).  We override to restore the critic group's
        separate learning rate afterwards.
        """
        super()._reinit_opt_from_config(config)
        self._apply_critic_lr()

    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return generated completion ids and corresponding action masks.

        Each observation dict normally includes ``input_ids`` and ``attention_mask``.
        Keys ``text`` and ``model_text`` are removed before the forward pass.

        For sliding-window multi-turn prompts (HuggingFace path only), optional
        keys are consumed: ``model_input_ids``, ``model_attention_mask``,
        ``stitch_prefix_ids``, and ``model_window_initial_len`` (required when
        ``stitch_prefix_ids`` is non-empty). vLLM colocate uses the same optional
        keys via the base implementation.

        :param obs: Batch of prompt dicts.
        :type obs: LLMObsType
        :param training: Unused; kept for API compatibility with other agents.
        :type training: bool
        :return: Lists of completion tensors and per-token action masks.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
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
                    for prompt in obs:
                        prompt.pop("text", None)
                        prompt.pop("model_text", None)
                        stitch = prompt.pop("stitch_prefix_ids", None)
                        model_window_initial_len = prompt.pop(
                            "model_window_initial_len",
                            None,
                        )
                        model_input_ids = prompt.pop("model_input_ids", None)
                        model_attention_mask = prompt.pop(
                            "model_attention_mask",
                            None,
                        )
                        if model_input_ids is not None:
                            prompt["input_ids"] = model_input_ids.to(actor_device)
                            if model_attention_mask is not None:
                                prompt["attention_mask"] = model_attention_mask.to(
                                    actor_device,
                                )
                        else:
                            prompt["input_ids"] = prompt["input_ids"].to(actor_device)
                            prompt["attention_mask"] = prompt[
                                "attention_mask"
                            ].to(
                                actor_device,
                            )
                        model_in_len = prompt["input_ids"].shape[1]
                        completion_id = self.actor.generate(
                            **prompt,
                            generation_config=self.generation_config,
                        )
                        stitch_mid_len = 0
                        if stitch is not None and stitch.shape[1] > 0:
                            if model_window_initial_len is None:
                                msg = (
                                    "model_window_initial_len is required when "
                                    "stitch_prefix_ids is non-empty"
                                )
                                raise ValueError(
                                    msg,
                                )
                            stitch_mid_len = stitch.shape[1]
                            completion_id = (
                                self._stitch_hf_completion_after_windowed_generate(
                                    completion_id,
                                    stitch,
                                    int(model_window_initial_len),
                                )
                            )
                        full_prompt_len = model_in_len + stitch_mid_len
                        completion_ids.append(completion_id)
                        action_mask = torch.zeros_like(
                            completion_id,
                            dtype=torch.bool,
                            device=completion_id.device,
                        )
                        action_mask[:, full_prompt_len:] = True
                        action_mask[completion_id == self.pad_token_id] = False
                        action_mask = action_mask[:, 1:]
                        action_masks.append(action_mask)
            else:
                completion_ids, action_masks = self._generate_with_vllm_colocate(
                    obs,
                    len(obs),
                    temperature=self.temperature if training else 0.01 # Almost deterministic for evaluation
                )

        return completion_ids, action_masks

    def learn(
        self,
        experiences: ExperiencesType,
        tokenizer,
        turn_ids: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float, float]:
        """Update actor and critic adapters using turn-level PPO objectives.

        :param experiences: Tuple of (completion_ids, action_masks, rewards).
            For single-turn, rewards is a flat list/tensor of scalars.
            For multi-turn, rewards should be [batch, max_turns] per-turn rewards.
        :param turn_ids: Optional [batch, seq_len] tensor mapping each token
            to its turn index (0-indexed). -1 for non-action tokens.
            When None, defaults to all action tokens belonging to turn 0.
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
            mean_pg_loss, mean_vf_loss, mean_loss, mean_kl, mean_entropy, updates = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
            )

            with torch.inference_mode():
                reference_log_probs, old_log_probs, old_values = (
                    self._fused_forward_no_grad(
                        completion_ids,
                        batch_size=batch_size,
                    )
                )
                old_values = torch.masked_fill(old_values, ~action_mask_bool, 0.0)

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

                returns, advantages = self._compute_gae_returns(
                    token_penalised_rewards, old_values, action_masks, turn_ids
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
                        batch_returns,
                        batch_advantages,
                        batch_old_values,
                        batch_turn_ids,
                    ) = get_experiences_samples(
                        minibatch_idxs,
                        completion_ids,
                        action_masks,
                        old_log_probs,
                        reference_log_probs,
                        returns,
                        advantages,
                        old_values,
                        turn_ids,
                    )

                    batch_mask_bool = batch_action_mask.bool()

                    # Fused forward: actor logprobs + critic values in one pass.
                    batch_log_probs, batch_values = self._fused_forward(
                        batch_ids,
                        batch_size=batch_size,
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

                    # Turn-level value loss
                    # Pool per-token critic values into one scalar per turn,
                    # then compute clipped MSE against turn-level returns.
                    batch_values = torch.masked_fill(
                        batch_values, ~batch_mask_bool, 0.0
                    )
                    mb_num_turns = batch_turn_ids.max().item() + 1
                    turn_pred = pool_by_turns(
                        batch_values, batch_turn_ids, mb_num_turns
                    )
                    turn_old = pool_by_turns(
                        batch_old_values, batch_turn_ids, mb_num_turns
                    )
                    turn_ret = pool_by_turns(
                        batch_returns, batch_turn_ids, mb_num_turns
                    )

                    # Mask: which (sample, turn) pairs actually exist in this batch.
                    turn_mask = torch.zeros_like(turn_pred)
                    for t in range(mb_num_turns):
                        turn_mask[:, t] = (batch_turn_ids == t).any(dim=1).float()

                    vf_loss = (turn_ret - turn_pred).pow(2)
                    clipped_turn_values = turn_old + torch.clamp(
                        turn_pred - turn_old, -self.clip_coef, self.clip_coef
                    )
                    clipped_vf_loss = (turn_ret - clipped_turn_values).pow(2)
                    vf_loss = (
                        0.5
                        * (torch.max(vf_loss, clipped_vf_loss) * turn_mask).sum()
                        / turn_mask.sum().clamp(min=1)
                        * self.vf_coef
                    )

                    total_loss = pg_loss + vf_loss

                    self._backward_pass(total_loss)
                    clear_fused_adapter_routing(self._get_unwrapped_actor())

                    mean_kl += masked_mean(kl, batch_action_mask).item()
                    mean_entropy += masked_entropy.mean().item()
                    mean_pg_loss += pg_loss.mean().item()
                    mean_vf_loss += vf_loss.mean().item()
                    mean_loss += total_loss.item()
                    updates += 1

        return (
            mean_loss / max(updates, 1),
            mean_kl / max(updates, 1),
            mean_pg_loss / max(updates, 1),
            mean_vf_loss / max(updates, 1),
            mean_entropy / max(updates, 1),
        )

    def test(
        self,
        env: ReasoningGym,
        loop: int = 1,
    ) -> torch.Tensor:
        """Return fitness (test) score tensor of llm on test sub-set."""
        eval_context = getattr(env, "eval_mode", nullcontext)
        with eval_context(), torch.inference_mode():
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

    def _compute_gae_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
        turn_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute turn-level GAE and broadcast advantages to all action tokens.

        Each generation turn is treated as a single RL action.  Per-turn values
        are the mean of critic values over the turn's action tokens, and gamma
        discounts between turns (not between tokens within a turn).
        """
        batch_size = values.shape[0]
        num_turns = turn_ids.max().item() + 1

        turn_values = pool_by_turns(values, turn_ids, num_turns)
        turn_rewards = pool_by_turns(rewards, turn_ids, num_turns)

        turn_advantages = torch.zeros(batch_size, num_turns, device=values.device)
        last_gae = torch.zeros(batch_size, device=values.device)
        per_sample_num_turns = turn_ids.max(dim=1).values + 1

        for t in reversed(range(num_turns)):
            is_last_turn = t >= (per_sample_num_turns - 1)
            if t == num_turns - 1:
                next_turn_value = torch.zeros_like(turn_values[:, 0])
            else:
                next_turn_value = turn_values[:, t + 1]
            next_turn_value = torch.where(
                is_last_turn, torch.zeros_like(next_turn_value), next_turn_value
            )

            delta = (
                turn_rewards[:, t] + self.gamma * next_turn_value - turn_values[:, t]
            )
            has_turn = (per_sample_num_turns > t).float()
            last_gae = (delta + self.gamma * self.gae_lambda * last_gae) * has_turn
            turn_advantages[:, t] = last_gae

        del turn_rewards

        token_advantages = torch.zeros_like(values)
        token_returns = torch.zeros_like(values)
        for t in range(num_turns):
            mask_t = (turn_ids == t).float()
            token_advantages += mask_t * turn_advantages[:, t : t + 1]
            token_returns += mask_t * (
                turn_advantages[:, t : t + 1] + turn_values[:, t : t + 1]
            )

        del turn_values, turn_advantages

        token_advantages = masked_whiten(token_advantages, action_mask)
        return token_returns, token_advantages * action_mask

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
        """
        num_turns = rewards.shape[1]
        token_rewards = torch.zeros_like(action_mask, dtype=torch.float)
        for t in range(num_turns):
            mask_t = (turn_ids == t).float()
            token_rewards += mask_t * rewards[:, t : t + 1]
        return token_rewards

    def _get_values(
        self,
        ids: torch.Tensor,
        batch_size: int,
        eval_mode: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        with self.select_adapter("critic"):
            # With DeepSpeed we cannot split backward into separate actor/critic
            # passes (engine.backward triggers allreduce per call).  Instead we
            # disable gradient checkpointing for the critic forward so that
            # activations are stored rather than recomputed during the single
            # backward pass — avoiding recomputation with the wrong adapter.
            unwrapped = self._get_unwrapped_actor()
            disable_gc = self.gradient_checkpointing and (
                self.accelerator is not None
                and self.accelerator.state.deepspeed_plugin is not None
            )
            if disable_gc:
                unwrapped.gradient_checkpointing_disable()

            self.actor.train(mode=not eval_mode)
            num_samples = ids.shape[0]
            if attention_mask is None:
                attention_mask = ids != self.pad_token_id
            if self.calc_position_embeddings:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
            values = []
            for batch in range(0, num_samples, batch_size):
                end_idx = min((batch + batch_size), num_samples)
                batch_ids = ids[batch:end_idx, :]
                batch_attention_mask = attention_mask[batch:end_idx, :]
                batch_model_kwargs = {
                    "input_ids": batch_ids,
                    "attention_mask": batch_attention_mask,
                    "use_cache": False,
                }
                if self.calc_position_embeddings:
                    batch_position_ids = position_ids[batch:end_idx, :]
                    batch_model_kwargs |= {"position_ids": batch_position_ids}
                with self._amp_ctx():
                    *_, value = self.actor.forward(**batch_model_kwargs)

                values.append(value[:, :-1])

            if disable_gc:
                unwrapped.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )

            return torch.cat(values, dim=0)

    def _calculate_kl_divergence(
        self,
        log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-token reverse-KL-style penalty term.

        This corresponds to the common Schulman-style approximation used with a
        fixed reference policy, matching the same sign convention as GRPO.
        """
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
        """Memory efficient params context manager.

        :param agent: Distributed agent
        :type agent: DistributedLLMAgent
        :return: None
        :rtype: None
        """
        # FIXME add in zero 3 compatibilitys
        vllm_cfg = getattr(self, "vllm_config", None)
        use_vllm = getattr(self, "use_vllm", False)
        # Only offload params for vLLM colocate mode. If use_vllm is False,
        # moving params to CPU between updates causes generate() device mismatches.
        sleep_mode = use_vllm and vllm_cfg is not None and vllm_cfg.sleep_mode
        if sleep_mode:
            # Put LLM to sleep
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
            # Wake up LLM
            self.llm.wake_up()
            self._move_model_to_vllm()
