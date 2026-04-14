from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator

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
    normalize_reasoning_prompt_batch,
    pool_by_turns,
    prepare_prompt_hf_generate,
    stitch_completion_after_windowed_hf_generate,
)
from agilerl.protocols import MultiTurnEnv

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
        use_vllm: bool = False,
        use_memory_efficient_params: bool = True,
        vllm_config: VLLMConfig | None = None,
        seed: int = 42,
        turn_level_clip: bool = True,
        gradient_checkpointing: bool = True,
        torch_compiler: str | None = None,
    ) -> None:

        device = (
            f"cuda:{accelerator.process_index}" if accelerator is not None else device
        )
        super().__init__(
            index=index,
            batch_size=batch_size,
            lr=lr_actor,
            lr_critic=lr_critic,
            max_grad_norm=max_grad_norm,
            clone=clone,
            calc_position_embeddings=calc_position_embeddings,
            seed=seed,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            use_value_head=True,
            use_vllm=use_vllm,
            vllm_config=vllm_config,
            use_liger_loss=False,
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
            model_name=model_name,
            actor_network=actor_network,
            model_config=model_config,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            hp_config=hp_config,
            use_memory_efficient_params=use_memory_efficient_params,
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
        self.turn_level_clip = turn_level_clip
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

        self.lr_critic = lr_critic if lr_critic is not None else lr_actor
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
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generate completion tokens for each prompt in the batch.

        :param obs: A single prompt dict or a list of HF-style prompt dicts.
        :type obs: LLMObsType
        :param training: If ``False``, use near-deterministic decoding where applicable.
        :type training: bool
        :return: Per-prompt completion token IDs and masks over generated positions.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
        prompt_batch = normalize_reasoning_prompt_batch(obs)

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
                            stitch_ids = prompt.pop("stitch_prefix_ids", None)
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
                    1,
                    temperature=self.temperature
                    if training
                    else 0.01,  # Almost deterministic for evaluation
                )

        return completion_ids, completion_masks

    def learn(
        self,
        experiences: ExperiencesType,
        turn_ids: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float, float]:
        """Update actor and critic adapters using turn-level PPO objectives.

        :param experiences: ``(completion_ids, action_masks, rewards)``. For
            single-turn, ``rewards`` is a flat tensor of scalars; for multi-turn,
            shape ``[batch, max_turns]`` per-turn rewards.
        :type experiences: ExperiencesType
        :param turn_ids: Optional ``[batch, seq_len - 1]`` tensor of turn indices;
            ``-1`` for non-action tokens. If ``None``, all action tokens are turn ``0``.
        :type turn_ids: torch.Tensor | None
        :return: ``(mean_loss, mean_kl, mean_pg_loss, mean_vf_loss, mean_entropy)``.
        :rtype: tuple[float, float, float, float, float]
        """
        self._prepare_vllm_for_training()

        with self.memory_efficient_params_context():
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
            updates = 0
            learn_metrics = {
                "mean_loss": 0.0,
                "mean_pg_loss": 0.0,
                "mean_vf_loss": 0.0,
                "mean_loss": 0.0,
                "mean_kl": 0.0,
                "mean_entropy": 0.0,
            }
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

                    # Compute turn-level quantities shared by both policy and
                    # value loss: num_turns, pooled values, and turn mask.
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

                    token_log_ratio = batch_log_probs - batch_old_log_probs
                    if self.turn_level_clip:
                        # Turn-PPO: sum token log-ratios per turn so the
                        # ratio is the product of token-level ratios.
                        log_ratio = pool_by_turns(
                            token_log_ratio,
                            batch_turn_ids,
                            mb_num_turns,
                            reduction="sum",
                        )
                        adv = pool_by_turns(
                            batch_advantages, batch_turn_ids, mb_num_turns
                        )
                        pg_mask = turn_mask
                    else:
                        # Standard PPO: use token-level log-ratios.
                        log_ratio = token_log_ratio
                        adv = batch_advantages
                        pg_mask = batch_action_mask

                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = masked_mean(
                        torch.max(-adv * ratio, -adv * clipped_ratio), pg_mask
                    )

                    vf_loss = (turn_ret - turn_pred).pow(2)
                    clipped_turn_values = turn_old + torch.clamp(
                        turn_pred - turn_old, - self.clip_coef, self.clip_coef
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

                    learn_metrics["mean_kl"] += masked_mean(
                        kl, batch_action_mask
                    ).item()
                    learn_metrics["mean_entropy"] += masked_entropy.mean().item()
                    learn_metrics["mean_pg_loss"] += pg_loss.mean().item()
                    learn_metrics["mean_vf_loss"] += vf_loss.mean().item()
                    learn_metrics["mean_loss"] += total_loss.item()
                    updates += 1

        return {
            metric: value / max(updates, 1) for metric, value in learn_metrics.items()
        }

    def test(
        self,
        env: ReasoningGym | MultiTurnEnv,
        loop: int = 1,
    ) -> torch.Tensor:
        """Return fitness (test) score tensor of llm on test sub-set.

        ``ReasoningGym`` (and compatible dataset envs): ``reset`` returns a batch
        of prompt dicts; each ``step`` accepts completion id tensors and returns
        the next batch plus rewards. ``loop`` iterations advance the test
        dataloader that many times.

        :param env: A :class:`~agilerl.utils.llm_utils.ReasoningGym` or
            :class:`~agilerl.wrappers.multiturn_wrappers.TokenObservationWrapper`.
        :type env: ReasoningGym | MultiTurnEnv
        :param loop: Number of outer test iterations (dataloader passes or episodes).
        :type loop: int
        :return: Concatenated per-step rewards from the test loop.
        :rtype: torch.Tensor
        """
        eval_context = getattr(env, "eval_mode", nullcontext)
        with eval_context(), torch.inference_mode():
            if isinstance(env, ReasoningGym):
                prompts = env.reset()
                rewards = []
                for _ in range(loop):
                    completion_ids, _ = self.get_action(prompts, training=False)
                    next_prompts, reward = env.step(completion_ids)
                    prompts = next_prompts
                    rewards.append(reward)
                reward_tensor = torch.cat(rewards)
            elif isinstance(env, MultiTurnEnv):
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
                        prompt_dict, reward, terminated, truncated, _step_info = (
                            env.step(
                                full,
                            )
                        )
                        all_rewards.append(
                            torch.tensor(
                                [float(reward)],
                                dtype=torch.float32,
                                device=full.device,
                            ),
                        )
                reward_tensor = torch.cat(all_rewards)
            else:
                msg = (
                    "env must be a ReasoningGym (or subclass) or "
                    f"MultiTurnEnv; got {type(env).__name__}"
                )
                raise TypeError(msg)
        mean_fit = torch.mean(reward_tensor.float()).item()
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

        :param rewards: Per-token (penalised) rewards ``[batch, seq_len]``.
        :type rewards: torch.Tensor
        :param values: Per-token critic values ``[batch, seq_len]``.
        :type values: torch.Tensor
        :param action_mask: Bool mask of valid action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for padding.
        :type turn_ids: torch.Tensor
        :return: Tuple of ``(token_returns, token_advantages)``, each ``[batch, seq_len]``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
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

        :param action_mask: Bool mask of action positions ``[batch, seq_len]``.
        :type action_mask: torch.Tensor
        :param rewards: Per-turn scalars ``[batch, max_turns]``.
        :type rewards: torch.Tensor
        :param turn_ids: Turn index per token ``[batch, seq_len]``; ``-1`` for non-action.
        :type turn_ids: torch.Tensor
        :return: Per-token rewards ``[batch, seq_len]``.
        :rtype: torch.Tensor
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
        """Compute critic values for each prompt token (excluding the last logits position).

        :param ids: Token IDs ``[batch, seq_len]``.
        :type ids: torch.Tensor
        :param batch_size: Micro-batch size for forward passes.
        :type batch_size: int
        :param eval_mode: If ``True``, run the critic in eval mode (no dropout).
        :type eval_mode: bool
        :param attention_mask: Optional mask matching ``ids``; defaults to non-pad tokens.
        :type attention_mask: torch.Tensor | None
        :return: Values aligned to next-token prediction, shape ``[batch, seq_len - 1]``.
        :rtype: torch.Tensor
        """
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
