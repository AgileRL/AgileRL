import gc
from typing import Any, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from peft import LoraConfig, PeftModel
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.core import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.typing import ExperiencesType, LLMObsType
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    VLLMConfig,
    get_experiences_samples,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_utils import (
    ReasoningGym,
)

DeepSpeedOptimizerType = Union[
    DeepSpeedZeroOptimizer,  # ZeRO Stage 1 & 2 optimizer
    DeepSpeedZeroOptimizer_Stage3,  # ZeRO Stage 3 optimizer
]


class GRPO(LLMAlgorithm):
    """The GRPO algorithm class. GRPO paper: https://arxiv.org/pdf/2402.03300

    :param pad_token_id: Pad token id
    :type pad_token_id: int
    :param pad_token: Pad token
    :type pad_token: str
    :param model_name: Model name
    :type model_name: str, optional
    :param actor_network: HuggingFace LLM
    :type actor_network: PreTrainedModel
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
    :param reduce_memory_peak: Flag to reduce memory peak in the _get_logprobs method, defaults to False
    :type reduce_memory_peak: bool, optional
    :param micro_batch_size_per_gpu: If specified, gradient_accumulation_steps will be
        calculated to achieve the target batch_size. If None, uses existing
        gradient_accumulation_steps from DeepSpeed config, defaults to None
    :type micro_batch_size_per_gpu: int, optional
    :param max_output_tokens: Max number of answer tokens, defaults to 512
    :type max_output_tokens: int, optional
    :param min_output_tokens: Minimum output tokens, defaults to 0
    :type min_output_tokens: int, optional
    :param max_model_len: Maximum context window length, defaults to None
    :type max_model_len: int, optional
    :param lora_config: Config for LoRA, defaults to None
    :type lora_config: LoraConfig, optional
    :param cosine_lr_schedule_config: Config for cosine lr scheduling, defaults to None
    :type cosine_lr_schedule_config: CosineLRScheduleConfig, optional
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
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_token: str,
        model_name: str | None = None,
        actor_network: PreTrainedModel | None = None,
        model_config: dict[str, Any] | None = None,
        hp_config: Optional[HyperparameterConfig] = None,
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
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        reduce_memory_peak: bool = False,
        max_output_tokens: int | None = 1024,
        min_output_tokens: Optional[int] = None,
        max_model_len: Optional[int] = None,
        lora_config: Optional[LoraConfig] = None,
        cosine_lr_schedule_config: Optional[CosineLRScheduleConfig] = None,
        accelerator: Optional[Accelerator] = None,
        device: str = "cpu",
        wrap: bool = True,
        clone: bool = False,
        use_separate_reference_adapter: bool = False,
        use_vllm: bool = False,
        vllm_config: Optional[VLLMConfig] = None,
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
            lora_config=lora_config,
            use_separate_reference_adapter=use_separate_reference_adapter,
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
        )
        assert isinstance(batch_size, int), "Batch size must be an integer."
        assert batch_size >= 1, "Batch size must be greater than or equal to one."
        assert isinstance(lr, float), "Learning rate must be a float."
        assert lr > 0, "Learning rate must be greater than zero."
        assert isinstance(
            clip_coef, (float, int)
        ), "Clipping coefficient must be a float."
        assert (
            clip_coef >= 0
        ), "Clipping coefficient must be greater than or equal to zero."
        assert isinstance(
            update_epochs, int
        ), "Policy update epochs must be an integer."
        assert (
            update_epochs >= 1
        ), "Policy update epochs must be greater than or equal to one."
        if actor_network is not None:
            assert isinstance(
                actor_network, (PeftModel, PreTrainedModel)
            ), "Actor network must be a PeftModel or PreTrainedModel"

        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.group_size = group_size
        self.beta = beta
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        if max_output_tokens is None and max_model_len is None:
            raise ValueError(
                "Either max_output_tokens or max_model_len must be specified"
            )
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

    def get_action(
        self, obs: LLMObsType, training: bool = True
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns the next action to take in the environment.

        :param obs: Environment observation, or multiple observations in a batch
        :type obs: numpy.ndarray[float]
        :param training: Flag to indicate training mode, defaults to True
        :type training: bool, optional
        :return: Completion IDs and action masks
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
        group_size = self.group_size if training else 1
        self.actor.eval()
        if not self.use_vllm:
            with torch.no_grad():
                completion_ids = []
                action_masks = []
                for prompt in obs:
                    prompt.pop("text", None)
                    prompt["input_ids"] = (
                        prompt["input_ids"].repeat(group_size, 1).to(self.actor.device)
                    )
                    prompt["attention_mask"] = (
                        prompt["attention_mask"]
                        .repeat(group_size, 1)
                        .to(self.actor.device)
                    )
                    completion_id = self.actor.generate(
                        **prompt,
                        generation_config=self.generation_config,
                    )
                    completion_ids.append(completion_id)
                    action_mask = torch.zeros_like(
                        completion_id, dtype=torch.bool, device=self.device
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
                obs, group_size
            )
            if self.vllm_config.sleep_mode:
                self.llm.sleep(level=2)

        return completion_ids, action_masks

    def learn(self, experiences: ExperiencesType) -> tuple[float, float]:
        """Updates agent network parameters to learn from experiences.

        :param experiences: Batched completion_ids, action_masks and rewards
        :type experiences: ExperiencesType
        """
        gc.collect()
        torch.cuda.empty_cache()

        completion_ids, action_masks, rewards = stack_and_pad_experiences(
            *experiences, padding_values=[self.pad_token_id, False, None]
        )
        advantages = self._calculate_advantage(rewards).to(self.device)

        num_samples = advantages.shape[0]
        batch_idxs = np.arange(num_samples)
        mean_loss, mean_kl = 0, 0
        batch_size = min(num_samples, self.micro_batch_size_per_gpu)

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

        experiences = (
            completion_ids,
            action_masks,
            advantages,
            old_log_probs,
            reference_log_probs,
        )

        for _ in range(self.update_epochs):
            self.rng.shuffle(batch_idxs)
            for start in range(0, num_samples, batch_size):
                minibatch_idxs = batch_idxs[
                    start : min((start + batch_size), num_samples)
                ]
                (
                    batch_ids,
                    batch_action_mask,
                    batch_advantages,
                    batch_old_log_probs,
                    batch_reference_log_probs,
                ) = get_experiences_samples(minibatch_idxs, *experiences)
                batch_log_probs = self._get_logprobs(
                    batch_ids,
                    batch_size=batch_size,
                    use_reference=False,
                    eval_mode=False,
                )
                loss, kl = self._grpo_loss(
                    batch_action_mask,
                    batch_log_probs,
                    batch_old_log_probs,
                    batch_reference_log_probs,
                    batch_advantages,
                )
                if not loss.isfinite():
                    raise ValueError(f"Loss is not finite: {loss}")
                self._backward_pass(loss)
                mean_loss += loss.item()
                mean_kl += kl.item()
        mean_loss /= len(completion_ids)
        mean_kl /= len(completion_ids)
        return mean_loss, mean_kl

    def test(
        self,
        env: ReasoningGym,
        loop: int = 1,
    ) -> torch.Tensor:
        """Returns fitness (test) score tensor of llm on test sub-set.

        :param env: The environment to be tested in
        :type env: ReasoningGym environment
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :return: Mean test score of the agent
        :rtype: float
        """
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

    def _calculate_advantage(
        self, rewards: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Calculate the group relative advantage for each groups reward.

        :param rewards: Tensor of rewards.
        :type rewards: torch.Tensor
        :param eps: Epsilon to prevent zero division error, defaults to 1e-8
        :type eps: float, optional
        :return: Tensor of group relative advantages.
        :rtype: torch.Tensor
        """
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(0)

        advantage = (rewards - rewards.mean(dim=1).unsqueeze(1)) / (
            rewards.std(dim=1).unsqueeze(1) + eps
        )
        return advantage.flatten().unsqueeze(1)

    def _calculate_kl_divergence(
        self, log_probs: torch.Tensor, reference_log_probs: torch.Tensor
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
        mask: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the GRPO loss.

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
        log_probs_ratio = torch.exp(log_probs - old_log_probs)
        clipped_log_probs_ratio = log_probs_ratio.clamp(
            1 - self.clip_coef, 1 + self.clip_coef
        )
        surrogate = log_probs_ratio * advantages
        clipped_surrogate = clipped_log_probs_ratio * advantages
        loss = -torch.min(surrogate, clipped_surrogate) + self.beta * kl
        denominator = mask.sum(dim=-1)
        denominator = torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
        loss = (loss * mask).sum(dim=-1) / denominator
        log_probs_ratio, clipped_log_probs_ratio, surrogate, clipped_surrogate = (
            None,
            None,
            None,
            None,
        )
        return loss.mean(), kl.mean()
