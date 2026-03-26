import gc
from pathlib import Path
import token
from typing import Any
import warnings
import time
import numpy as np
import torch
from accelerate import Accelerator
from torchviz import make_dot

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core import LLMAlgorithm, OptimizerWrapper
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.protocols import (
    LoraConfigProtocol,
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.typing import ExperiencesType, LLMObsType
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    DummyOptimizer,
    VLLMConfig,
    create_warmup_cosine_scheduler,
    get_experiences_samples,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_utils import ReasoningGym, masked_whiten, masked_mean

if HAS_LLM_DEPENDENCIES:
    from transformers import GenerationConfig

def print_zero2_bucket_layout(ds_engine):
    optimizer = ds_engine.optimizer  # ZeRO optimizer
    reduce_bucket_size = optimizer.reduce_bucket_size  # bytes threshold  

    print("Optimizer: ", optimizer) 
    base_optimizer = optimizer.optimizer

    # Build name lookup
    param_to_name = {id(p): n for n, p in ds_engine.named_parameters()}

    for i, group in enumerate(base_optimizer.param_groups):
        print(f"\nGroup {i} (lr={group['lr']}):")
        for p in group['params']:
            num_params_in_opt = p.numel()
            name = param_to_name.get(id(p), "unknown")
            print(f"  {name} | shape={p.shape} | requires_grad={p.requires_grad}")

    # get num params in loras
    lora_params = [p for n, p in ds_engine.named_parameters() if "lora" in n]
    num_params_in_loras = sum([p.numel() for p in lora_params])
    print("Num params in loras: ", num_params_in_loras, "num params in opt: ", num_params_in_opt)

    

                                                                                           
                                            
    param_to_name = {p: n for n, p in ds_engine.module.named_parameters()}                                                                                                                        
                                                    
    # Collect all params in optimizer order, then reverse (ZeRO processes back-to-front)                                                                                                          
    all_params = []                                                                                                                                                                               
    for group in optimizer.param_groups:                                                                                                                                                          
        all_params.extend(p for p in group["params"])                                                                                                                          
    all_params = list(reversed(all_params))   

    print("All params: ", all_params[0].shape)       
                                                                                                                                                                                                
    bucket_idx, bucket_names, bucket_numel = 0, [], 0
    for p in all_params:                                                                                                                                                                          
        name = param_to_name.get(p, "?")                                                                                                                                                          
        bucket_names.append(name)                                                                                                                                                                 
        bucket_numel += p.numel() * p.element_size()                                                                                                                                              
        if bucket_numel >= reduce_bucket_size:                                                                                                                                                    
            print(f"Bucket {bucket_idx} ({bucket_numel/1e6:.1f} MB): {bucket_names}")                                                                                                             
            bucket_idx += 1                 
            bucket_names, bucket_numel = [], 0                                                                                                                                                    
    if bucket_names:                                 
        print(f"Bucket {bucket_idx} ({bucket_numel/1e6:.1f} MB): {bucket_names}") 



class PPO(LLMAlgorithm):
    """Token-level PPO for LLM finetuning with actor/reference adapters."""

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
        )
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(clip_coef, (float, int)), "Clipping coefficient must be a float."
        assert clip_coef >= 0, (
            "Clipping coefficient must be greater than or equal to zero."
        )
        assert isinstance(update_epochs, int), "Policy update epochs must be an integer."
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

        self.use_vllm = use_vllm
        self.vllm_config = vllm_config
        if self.use_vllm:
            self._configure_vllm()
        self._initialize_actors(actor_network, not clone)
        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

        print("Actor: ", self.actor)
        # print_zero2_bucket_layout(self.actor)
        # assert False

    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return generated completion ids and corresponding action masks."""
        self.actor.eval()
        if not self.use_vllm:
            actor_module = self._get_unwrapped_actor()
            try:
                actor_device = next(actor_module.parameters()).device
            except StopIteration:
                actor_device = torch.device(self.device)
            with torch.no_grad():
                completion_ids = []
                action_masks = []
                for prompt in obs:
                    prompt.pop("text", None)
                    prompt["input_ids"] = prompt["input_ids"].to(actor_device)
                    prompt["attention_mask"] = prompt["attention_mask"].to(actor_device)
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
                    action_mask[:, prompt["input_ids"].shape[1] :] = True
                    action_mask[completion_id == self.pad_token_id] = False

                    action_mask = action_mask[:, 1:]
                    
                    action_masks.append(action_mask)
        else:
            if self.vllm_config.sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()
            self._move_model_to_vllm()
            completion_ids, action_masks = self._generate_with_vllm_colocate(
                obs,
                1,
            )
            if self.vllm_config.sleep_mode:
                self.llm.sleep(level=2)

        return completion_ids, action_masks


    def learn(
        self,
        experiences: ExperiencesType,
        tokenizer
    ) -> tuple[float, float, float, float, float]:
        """Update actor and critic adapters using token-level PPO objectives."""
        gc.collect()
        torch.cuda.empty_cache()


        completion_ids, action_masks, rewards = stack_and_pad_experiences(
            *experiences,
            padding_values=[self.pad_token_id, False, None],
        )
        completion_ids = completion_ids.to(self.device)
        sequence_rewards = rewards.flatten().to(self.device).float()
        action_masks = action_masks.to(self.device)
        num_samples = completion_ids.shape[0]

        if sequence_rewards.shape[0] != num_samples:
            msg = (
                "Expected one scalar reward per sampled completion. "
                f"Got {sequence_rewards.shape[0]} rewards for {num_samples} samples."
            )
            raise ValueError(
                msg,
            )
        batch_idxs = np.arange(num_samples)
        batch_size = min(num_samples, self.micro_batch_size_per_gpu) if hasattr(self, "micro_batch_size_per_gpu") else num_samples
        mean_pg_loss, mean_vf_loss, mean_loss, mean_kl, mean_entropy, updates = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )


        with torch.no_grad():
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
            with self.select_adapter("critic"):
                old_values = self._get_values(
                    completion_ids,
                    batch_size=batch_size,
                    eval_mode=True,
                )
                old_values = torch.masked_fill(old_values, ~action_masks.bool(), 0.0)
            # old_log_probs, old_values = self._get_logprobs_and_values(
            #     completion_ids,
            #     batch_size=batch_size,
            #     use_reference=False,
            #     eval_mode=True,
            # )
            

            token_rewards = self._compute_token_rewards(action_masks, sequence_rewards)

            # FIXME the below is for policy testing
            # target_token_id = 13
            # token_rewards = (completion_ids[:, 1:] == target_token_id) * action_masks.float() # NOTE for policy net testing

            old_log_probs = torch.masked_fill(old_log_probs, ~action_masks.bool(), 1.0)
            reference_log_probs = torch.masked_fill(reference_log_probs, ~action_masks.bool(), 1.0)
            token_kl = old_log_probs - reference_log_probs

            token_penalised_rewards = token_rewards - self.beta * token_kl


            torch.set_printoptions(threshold=torch.inf)
            # print(token_penalised_rewards)
            # print(old_log_probs == reference_log_probs)
            # assert False
            returns, advantages = self._compute_gae_returns(token_penalised_rewards, old_values, action_masks)            

            torch.set_printoptions(threshold=torch.inf)

            # print(old_log_probs)

        params = {}

        for name, param in self.actor.named_parameters():
            if "lora" in name and ("actor" in name or "critic" in name):
                params[name] = param.clone().detach()


        aligned_action_masks = action_masks#:, 1:]
        aligned_advantages = advantages#[:, :-1]
        algined_returns = returns#[:, :-1]
        aligned_old_values = old_values#[:, :-1]

        for _ in range(self.update_epochs):
            self.rng.shuffle(batch_idxs)
            for start in range(0, num_samples, batch_size):
                minibatch_idxs = batch_idxs[start : min((start + batch_size), num_samples)]
                (
                    batch_ids,
                    batch_action_mask,
                    batch_old_log_probs,
                    batch_reference_log_probs,
                    batch_returns,
                    batch_advantages,
                    batch_old_values,
                ) = get_experiences_samples(
                    minibatch_idxs,
                    completion_ids,
                    aligned_action_masks,
                    old_log_probs,
                    reference_log_probs,
                    algined_returns,
                    aligned_advantages,
                    aligned_old_values,
                )
                # batch_log_probs, batch_values = self._get_logprobs_and_values(
                #     batch_ids,
                #     batch_size=batch_size,
                #     use_reference=False,
                #     eval_mode=False,
                # )

                # Pass 1: actor forward — builds pg_loss computation graph with actor LoRA.
                # Restore critic trainability first: previous iteration's _use_policy()
                # called set_adapter("actor") which PEFT side-effects to critic requires_grad=False.
                # ZeRO-2 bucket hooks registered at prepare() time must keep firing for all adapters.
                # self._restore_lora_trainability(["critic"])
                batch_log_probs = self._get_logprobs(
                    batch_ids,
                    batch_size=batch_size,
                    use_reference=False,
                    eval_mode=False,
                )
                # _get_logprobs calls _use_policy() -> set_adapter("actor") -> disables critic again.
                # self._restore_lora_trainability(["critic"])

                batch_log_probs = torch.masked_fill(batch_log_probs, ~batch_action_mask.bool(), 1.0)
                kl = batch_log_probs - batch_reference_log_probs

                # Proxy entropy (-log pi(a_t|s_t)) avoids full-vocab entropy tensors.
                masked_entropy = masked_mean(-batch_log_probs.detach(), batch_action_mask)

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
                pg_loss = masked_mean(torch.max(pg_loss_unclipped, pg_loss_clipped), batch_action_mask)


                self._backward_pass(pg_loss)

                # Pass 2: critic forward — builds vf_loss computation graph with critic LoRA.
                # Switch adapter then immediately restore actor trainability (PEFT disabled it).
                self.actor.set_adapter("critic")
                # self._restore_lora_trainability(["actor"])
                batch_values = self._get_values(
                    batch_ids,
                    batch_size=batch_size,
                    eval_mode=False,
                )
                batch_values = torch.masked_fill(batch_values, ~batch_action_mask.bool(), 0.0)

                vf_loss = (batch_returns - batch_values).pow(2)
                clipped_batch_values = batch_old_values + torch.clamp(
                    batch_values - batch_old_values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                clipped_vf_loss = (batch_returns - clipped_batch_values).pow(2)
                vf_loss = 0.5 * masked_mean(torch.max(vf_loss, clipped_vf_loss), batch_action_mask)

                # Combined backward — ZeRO-2 compatible single step per minibatch.
                # Both actor LoRA (in pg_loss graph) and critic LoRA (in vf_loss graph)
                # have requires_grad=True so all ZeRO-2 gradient bucket hooks fire.
                # Gradient isolation is via computation graph structure, not requires_grad.
                # total_loss = pg_loss + self.vf_coef * vf_loss # 1774514395.905044

                # if not total_loss.isfinite():
                #     raise ValueError(f"Loss is not finite: {total_loss}")

                # self._restore_lora_trainability(["actor", "critic"])
                self._backward_pass(vf_loss)

                # Restore actor adapter for next iteration; keep critic trainable.
                self._use_policy()
                # self._restore_lora_trainability(["critic"])

                mean_kl += masked_mean(kl, batch_action_mask).item()
                mean_entropy += masked_entropy.mean().item()
                del masked_entropy
                mean_pg_loss += pg_loss.mean().item()
                mean_vf_loss += vf_loss.mean().item()
                # mean_loss += total_loss.item()
                updates += 1

        for name, param in self.actor.named_parameters():
            # Check loras have updated
            if "lora" in name and ("actor" in name or "critic" in name):
                if torch.equal(param.data, params[name].data):
                    print(f"Lora {name} has not updated {time.time()}")
                    


        return (
            mean_loss / max(updates, 1),
            mean_kl / max(updates, 1),
            mean_pg_loss / max(updates, 1),
            mean_vf_loss / max(updates, 1),
            mean_entropy / max(updates, 1),
            # reporting_reward / max(updates, 1),
        )

    def test(
        self,
        env: ReasoningGym,
        loop: int = 1,
    ) -> torch.Tensor:
        """Return fitness (test) score tensor of llm on test sub-set."""
        with env.eval_mode(), torch.no_grad():
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns R_t = V_t + A_t^GAE for each token position.

        Bootstrap from the next token only if it is a valid action position
        (action_mask[:, t+1] is True). At the boundary between the last
        generated token and padding, next_valid is 0, which zeros next_values
        and resets the GAE carry — correctly treating that position as terminal.
        Prompt and padding positions are masked out in the loss so their
        advantages do not affect training.
        """
        batch_size, sequence_length = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=rewards.device)

        # Sequence length 10
        
        for t in reversed(range(sequence_length)):
            mask_t = action_mask[:, t]

            if t == sequence_length - 1:
                next_values = torch.zeros_like(values[:, 0])
            else:
                next_values = values[:, t + 1] * action_mask[:, t + 1]
            
            # if t + 1 < sequence_length:
            #     next_values = values[:, t + 1] * action_mask[:, t + 1]
            # else:
            #     next_values = torch.zeros_like(values[:, 0])
            
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae = (delta + self.gamma * self.gae_lambda * last_gae) * mask_t
            advantages[:, t] = last_gae


        returns = advantages + values

        advantages = masked_whiten(advantages, action_mask)

        return returns, advantages * action_mask
            

    def _compute_token_rewards(
        self, 
        action_mask: torch.Tensor,
        sequence_rewards: torch.Tensor,
    ) -> torch.Tensor:
        token_rewards = torch.zeros_like(action_mask, dtype=torch.float32)
        valid = action_mask.any(dim=-1)

        # FIXME this is a hack to test value function!
        # Option B: more robust — last valid action position per row
        # last_action = action_mask.long().cumsum(dim=1).max(dim=1)[1]
        # token_rewards.scatter_(dim=1, index=last_action.unsqueeze(1), src=torch.ones_like(last_action, dtype=torch.float32).unsqueeze(1))
        # return token_rewards

        # sequence_rewards = torch.ones_like(sequence_rewards) # FIXME for testing

        # reward 1.0 if response contains token id 1234, else 0.0

        # print(action_mask.sum(dim=-1))
        if valid.any():
            reward_idx = action_mask[valid].long().cumsum(dim=-1).argmax(dim=-1)
            row_ids = torch.arange(
                token_rewards.shape[0],
                device=token_rewards.device,
            )[valid]
            token_rewards[row_ids, reward_idx] = sequence_rewards[valid]

        return token_rewards

    def _get_values(
        self,
        ids: torch.Tensor,
        batch_size: int,
        eval_mode: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        # with self.select_adapter("critic"):
        self.actor.train(mode=not eval_mode)
        num_samples = ids.shape[0]
        if attention_mask is None:
            attention_mask = ids != self.pad_token_id
        if self.calc_position_embeddings:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
        # Split the sample into batches
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
            *_, value = self.actor.forward(**batch_model_kwargs)
            
            values.append(value[:, 1:])
        return torch.cat(values, dim=0)
                
        

    # def _get_logprobs_and_values(
    #     self,
    #     ids: torch.Tensor,
    #     batch_size: int,
    #     use_reference: bool = False,
    #     eval_mode: bool = False,
    #     attention_mask: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     """Compute token-level log-probs and value estimates.

    #     When training (eval_mode=False, use_reference=False), two separate forward
    #     passes are used so that critic loss cannot backprop into actor LoRA weights:
    #       Pass 1 (actor LoRA unfrozen): logits -> log_probs
    #       Pass 2 (actor LoRA frozen):   hidden_states -> v_head -> values
    #     For eval/reference passes a single forward pass is used since no gradients
    #     are computed.
    #     """
    #     # Only split passes when gradients matter and we're using the actor adapter.

    #     with self.select_adapter("reference" if use_reference else "actor"):
    #         self.actor.train(mode=not eval_mode)
    #         num_samples = ids.shape[0]
    #         if attention_mask is None:
    #             attention_mask = ids != self.pad_token_id
    #         if self.calc_position_embeddings:
    #             position_ids = attention_mask.long().cumsum(dim=-1) - 1
    #             position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

    #         log_probs = []
    #         collected_values = []
    #         for batch in range(0, num_samples, batch_size):
    #             end_idx = min((batch + batch_size), num_samples)
    #             batch_ids = ids[batch:end_idx, :]
    #             batch_attention_mask = attention_mask[batch:end_idx, :]
    #             batch_model_kwargs = {
    #                 "input_ids": batch_ids,
    #                 "attention_mask": batch_attention_mask,
    #                 "use_cache": False,
    #             }
    #             if self.calc_position_embeddings:
    #                 batch_position_ids = position_ids[batch:end_idx, :]
    #                 batch_model_kwargs |= {"position_ids": batch_position_ids}

    #             # Pass 1: actor LoRA active and trainable — gradient flows into actor LoRA.
    #             # The v_head hook captures hidden states for the critic pass below.
    #             outputs = self.actor.forward(**batch_model_kwargs)
    #             # TRL value-head outputs are token values over input_ids
    #             # with shape (B, seq_len). Align with logprobs/action masks
    #             # which operate on next-token predictions (B, seq_len-1).
    #             logits, *_, values = outputs
    #             del outputs
    #             log_prob = LLMAlgorithm._memory_efficient_logits(
    #                 logits[:, :-1],
    #                 batch_ids[:, 1:],
    #             )
    #             del logits
    #             log_probs.append(log_prob)
    #             collected_values.append(values[:, :-1])

    #     return torch.cat(log_probs, dim=0), torch.cat(collected_values, dim=0), None

    # def _get_logprobs_and_values(
    #     self,
    #     ids: torch.Tensor,
    #     batch_size: int,
    #     use_reference: bool = False,
    #     eval_mode: bool = False,
    #     attention_mask: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     """Compute token-level log-probs and value estimates.

    #     When training (eval_mode=False, use_reference=False), two separate forward
    #     passes are used so that critic loss cannot backprop into actor LoRA weights:
    #       Pass 1 (actor LoRA unfrozen): logits -> log_probs
    #       Pass 2 (actor LoRA frozen):   hidden_states -> v_head -> values
    #     For eval/reference passes a single forward pass is used since no gradients
    #     are computed.
    #     """
    #     # Only split passes when gradients matter and we're using the actor adapter.
    #     two_pass = not eval_mode and not use_reference

    #     with self.select_adapter("reference" if use_reference else "actor"):
    #         self.actor.train(mode=not eval_mode)
    #         num_samples = ids.shape[0]
    #         if attention_mask is None:
    #             attention_mask = ids != self.pad_token_id
    #         if self.calc_position_embeddings:
    #             position_ids = attention_mask.long().cumsum(dim=-1) - 1
    #             position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

    #         log_probs = []
    #         collected_values = []

    #         # For the two-pass path, resolve the v_head module once and register a
    #         # forward hook to capture the hidden states that are fed into it during
    #         # pass 1.  We re-run only v_head (not the full transformer) on detached
    #         # hidden states so that critic loss gradients reach v_head params but
    #         # never propagate into the actor LoRA weights.
    #         captured_hidden_states: list[torch.Tensor] = []
    #         v_head_hook = None
    #         unwrapped = (
    #             self.accelerator.unwrap_model(self.actor)
    #             if self.accelerator is not None
    #             else self.actor
    #         )
    #         if two_pass:
    #             def _capture_v_head_input(module, inp, out):  # noqa: E306
    #                 captured_hidden_states.append(inp[0])
    #             v_head_hook = unwrapped.v_head.register_forward_hook(
    #                 _capture_v_head_input
    #             )

    #         for batch in range(0, num_samples, batch_size):
    #             end_idx = min((batch + batch_size), num_samples)
    #             batch_ids = ids[batch:end_idx, :]
    #             batch_attention_mask = attention_mask[batch:end_idx, :]
    #             batch_model_kwargs = {
    #                 "input_ids": batch_ids,
    #                 "attention_mask": batch_attention_mask,
    #                 "use_cache": False,
    #             }
    #             if self.calc_position_embeddings:
    #                 batch_position_ids = position_ids[batch:end_idx, :]
    #                 batch_model_kwargs |= {"position_ids": batch_position_ids}

    #             # Pass 1: actor LoRA active and trainable — gradient flows into actor LoRA.
    #             # The v_head hook captures hidden states for the critic pass below.
    #             outputs = self.actor.forward(**batch_model_kwargs)
    #             # TRL value-head outputs are token values over input_ids
    #             # with shape (B, seq_len). Align with logprobs/action masks
    #             # which operate on next-token predictions (B, seq_len-1).
    #             logits, *_, values = outputs
    #             del outputs
    #             log_prob = LLMAlgorithm._memory_efficient_logits(
    #                 logits[:, :-1],
    #                 batch_ids[:, 1:],
    #             )
    #             del logits
    #             log_probs.append(log_prob)

    #             if two_pass:
    #                 # Pass 2: run only the v_head on hidden states captured from
    #                 # pass 1 (detached from the backbone graph) so that critic loss
    #                 # gradients reach v_head params but not actor LoRA.
    #                 # NOTE: requires_grad_(False) between forward and backward corrupts
    #                 # gradient checkpointing recomputation; we avoid that here by
    #                 # never mutating requires_grad between passes.
    #                 del values
    #                 backbone_hidden = captured_hidden_states[0]
    #                 captured_hidden_states.clear()
    #                 values = unwrapped.v_head(backbone_hidden.detach())
    #                 # The direct v_head call bypasses any squeeze applied by
    #                 # AutoModelForCausalLMWithValueHead.forward(), so normalise
    #                 # to (B, seq_len) here if the ValueHead returned (B, seq_len, 1).
    #                 if values.dim() == 3:
    #                     values = values.squeeze(-1)

    #             collected_values.append(values[:, :-1])

    #         if v_head_hook is not None:
    #             v_head_hook.remove()

    #     return torch.cat(log_probs, dim=0), torch.cat(collected_values, dim=0)

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

    