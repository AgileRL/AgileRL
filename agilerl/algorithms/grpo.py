import gc
import os
import re
import warnings
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from gymnasium import spaces
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams

from agilerl.algorithms.core import LLMAlgorithm, OptimizerWrapper
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.modules.dummy import DummyEvolvable
from agilerl.typing import ExperiencesType, LLMObsType
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    VLLMConfig,
    create_warmup_cosine_scheduler,
    get_experiences_samples,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_utils import (
    DummyOptimizer,
    HuggingFaceGym,
)

DeepSpeedOptimizerType = Union[
    DeepSpeedZeroOptimizer,  # ZeRO Stage 1 & 2 optimizer
    DeepSpeedZeroOptimizer_Stage3,  # ZeRO Stage 3 optimizer
]


class GRPO(LLMAlgorithm):
    """The GRPO algorithm class. GRPO paper: https://arxiv.org/pdf/2402.03300

    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
    :param actor_network: HuggingFace LLM
    :type actor_network: PreTrainedModel
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
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        actor_network: PreTrainedModel,
        pad_token_id: int,
        pad_token: str,
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
        max_output_tokens: int = 1024,
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
    ) -> None:

        device = (
            f"cuda:{os.getenv('LOCAL_RANK', '0')}"
            if accelerator is not None and torch.cuda.is_available()
            else device
        )
        super().__init__(
            observation_space,
            action_space,
            index=index,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
            name="GRPO",
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
        assert isinstance(
            actor_network, (PeftModel, PreTrainedModel)
        ), "Actor network must be a PeftModel or PreTrainedModel"
        if (
            accelerator is not None
            and cosine_lr_schedule_config is not None
            and accelerator.is_main_process
        ):
            warnings.warn(
                "Cannot specify the optimizer in the deepspeed config and use AgileRL's LR scheduler. If you want to use LR scheduling, \
            please specify in the deepspeed config. Setting LR scheduler to None."
            )
            cosine_lr_schedule_config = None

        if not clone and reduce_memory_peak and micro_batch_size_per_gpu is not None:
            raise ValueError(
                "Cannot specify micro_batch_size_per_gpu when reduce_memory_peak is True."
            )

        self._configure_batch_size(
            batch_size, clone, reduce_memory_peak, micro_batch_size_per_gpu
        )

        if self.accelerator is not None:
            if (
                self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                    "optimizer", None
                )
                is not None
            ):
                optim_lr = self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "optimizer"
                ]["params"]["lr"]
                if optim_lr is not None and optim_lr != lr:
                    warnings.warn(
                        "Argument 'lr' will be overwritten by the 'lr' value set in the deepspeed config."
                    )
                    lr = optim_lr

        self.lr = lr
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.group_size = group_size
        self.beta = beta
        self.calc_position_embeddings = calc_position_embeddings
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_output_tokens = max_output_tokens
        self.min_output_tokens = min_output_tokens
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
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
        if lora_config is None and not isinstance(actor_network, PeftModel):
            warnings.warn(
                "No LoRA config provided. Using default LoRA configuration for RL finetuning."
            )
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
        self.lora_config = lora_config
        self.cosine_lr_schedule_config = cosine_lr_schedule_config
        self.wrap = wrap
        self.use_separate_reference_adapter = use_separate_reference_adapter
        if max_grad_norm and (accelerator is not None) and accelerator.is_main_process:
            warnings.warn(
                "Argument 'max_grad_norm' will be overwritten by the 'gradient_clipping' value set in the deepspeed config."
            )
            self.max_grad_norm = None
        else:
            self.max_grad_norm = max_grad_norm
        self.reduce_memory_peak = reduce_memory_peak
        self.local_rank = (
            "0" if self.accelerator is None else self.accelerator.local_process_index
        )

        self.pretrained_model_name_or_path = actor_network.name_or_path
        self._initialize_actors(actor_network, not clone)
        self.use_vllm = use_vllm
        self.vllm_config = vllm_config

        if self.accelerator is not None:
            set_seed(seed, device_specific=True)

        if self.use_vllm:
            if self.vllm_config is None:
                warnings.warn(
                    "No VLLM config provided. Using default VLLM configuration for generation."
                )
                self.vllm_config = VLLMConfig()
            if self.use_vllm and self.accelerator is not None:
                if (
                    self.accelerator.num_processes
                    % self.vllm_config.tensor_parallel_size
                    != 0
                ):
                    raise ValueError(
                        f"Tensor parallel size {self.vllm_config.tensor_parallel_size} must be a multiple of the number of processes {self.accelerator.num_processes}."
                    )

                if self.vllm_config.tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_config.tensor_parallel_size,
                                    (i + 1) * self.vllm_config.tensor_parallel_size,
                                )
                            )
                            for i in range(
                                self.accelerator.num_processes
                                // self.vllm_config.tensor_parallel_size
                            )
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

                self.llm = LLM(
                    model=self.pretrained_model_name_or_path,
                    tensor_parallel_size=self.vllm_config.tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
                    max_num_seqs=self.vllm_config.max_num_seqs,
                    max_model_len=self.max_model_len,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index
                    // self.vllm_config.tensor_parallel_size,
                    max_num_batched_tokens=self.vllm_config.max_num_seqs
                    * self.max_model_len,
                    model_impl="vllm",
                )

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

    def get_action(
        self, obs: LLMObsType, training: bool = True
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns the next action to take in the environment.

        :param states: Environment observation, or multiple observations in a batch
        :type states: numpy.ndarray[float]
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
            # Move model to vllm
            self._move_model_to_vllm()
            completion_ids, action_masks = self._generate_with_vllm_colocate(
                obs, group_size
            )

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

    def set_reference_policy(self, reference_update_tracker: int) -> None:
        """Update the reference policy when the reference policy update tracker is greater than the current reference policy update tracker.

        :param reference_update_tracker: The reference policy update tracker
        :type reference_update_tracker: int
        """
        assert (
            reference_update_tracker >= self.reference_update_tracker
        ), "Reference policy update tracker should be greater than or equal to the current reference policy update tracker."
        if reference_update_tracker > self.reference_update_tracker:

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            # Merge adapter into base model
            # Update the reference update tracker
            if self.use_separate_reference_adapter:
                # Activate both adapters
                # Iterate over the params
                ref_param = None
                actor_param = None
                for name, param in self.actor.named_parameters():
                    if "lora" in name:
                        if "reference" in name:
                            ref_param = param
                        elif "actor" in name:
                            actor_param = param
                        else:
                            raise ValueError(
                                f"Only adapter names 'actor' and 'reference' are allowed, nether was found in {name}"
                            )
                    if ref_param is not None and actor_param is not None:
                        ref_param.data.copy_(actor_param.data)
                        ref_param = None
                        actor_param = None
            else:
                if self.accelerator is not None:
                    merged_base_model = self.accelerator.unwrap_model(
                        self.actor
                    ).merge_and_unload()
                else:
                    merged_base_model = self.actor.merge_and_unload()
                self.actor = None  # De-reference the old actor base model
                self.actor = get_peft_model(
                    merged_base_model, self.lora_config, adapter_name="actor"
                )
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                self.actor.set_adapter("actor")

                # Reinit optimizer
                optim_class = self._select_optim_class()
                self.optimizer = OptimizerWrapper(
                    optim_class, networks=[self.actor], lr=self.lr
                )
                self.wrap_models()
            self.reference_update_tracker += 1

    def test(
        self,
        env: HuggingFaceGym,
        loop: int = 1,
    ) -> torch.Tensor:
        """Returns test score tensor of llm on test sub-set.

        :param env: The environment to be tested in
        :type env: HuggingFaceGym environment
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :return: Mean test score of the agent
        :rtype: float
        """
        with env.eval_mode():
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

    def _initialize_actors(
        self, base_model: PreTrainedModel, add_adapters: bool = True
    ):
        """Initialize the actor network.

        :param base_model: Base model
        :type base_model: PreTrainedModel
        :param add_adapters: Flag to indicate if adapters should be added to the model, defaults to True
        :type add_adapters: bool, optional
        """

        if isinstance(base_model, PeftModel) and add_adapters:
            # Handles backwards compatibility with user providing a peft model as the actor network
            adapter_name = list(base_model.peft_config.keys())
            if len(adapter_name) > 1:
                warnings.warn(
                    "AgileRL RL finetuning is only compatible with one adapter."
                )
            self.lora_config = base_model.peft_config[adapter_name[0]]
            for adapter in adapter_name:
                base_model.delete_adapter(adapter)
            base_model = base_model.model

        self.actor: PeftModel = (
            get_peft_model(base_model, self.lora_config, adapter_name="actor")
            if add_adapters
            else base_model
        )

        if self.use_separate_reference_adapter and add_adapters:
            self.actor.add_adapter(
                adapter_name="reference", peft_config=self.lora_config  # type: ignore
            )

        self.actor.set_adapter("actor")

        if self.accelerator is None:
            self.actor = DummyEvolvable(module=self.actor, device=self.device)

        optim_class = self._select_optim_class()
        self.optimizer = OptimizerWrapper(
            optim_class, networks=[self.actor], lr=self.lr
        )
        self.lr_scheduler = (
            create_warmup_cosine_scheduler(
                (
                    self.optimizer.optimizer
                    if self.optimizer.optimizer_cls != DummyOptimizer
                    else self.actor.optimizer
                ),
                self.cosine_lr_schedule_config,
                1e-8,
                self.lr,
            )
            if self.cosine_lr_schedule_config is not None
            else None
        )

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

    def _get_logprobs(
        self,
        ids: torch.Tensor,
        batch_size: int,
        use_reference: bool = False,
        eval_mode: bool = False,
    ) -> torch.Tensor:
        """Find the log probabilities for a set of previously generated ids.

        :param ids: Completion IDs.
        :type ids: torch.Tensor
        :param batch_size: Batch size.
        :type batch_size: int
        :param use_reference: Flag to indicate to use reference policy, defaults to False
        :type use_reference: bool, optional
        :param eval_mode: Flag to indicate setting policy network to evaluation mode, defaults to False
        :type eval_mode: bool, optional
        :return: Log probabilities of the completion IDs.
        :rtype: torch.Tensor
        """

        with self.select_policy(use_reference):
            self.actor.train(mode=not eval_mode)
            num_samples = ids.shape[0]
            attention_mask = ids != self.pad_token_id
            if self.calc_position_embeddings:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

            # Split the sample into batches
            log_probs = []
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
                logits = self.actor.forward(**batch_model_kwargs).logits
                logits = logits / self.temperature
                log_prob = GRPO._memory_efficient_logits(
                    logits[:, :-1], batch_ids[:, 1:]
                )
                batch_model_kwargs = None
                logits = None
                log_probs.append(log_prob)
        return torch.cat(log_probs, dim=0)

    def _backward_pass(self, loss: float) -> None:
        """Perform a backward pass

        :param loss: Loss
        :type loss: float
        """

        if self.accelerator is not None:
            self.accelerator.backward(loss)
            if (
                self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                    "optimizer", None
                )
                is None
            ):
                # Accelerate handles optimizer step and zero grad if optimizer is defined in deepspeed config
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr = self.lr_scheduler.get_last_lr()[0]

    @contextmanager
    def select_policy(self, use_reference: bool = False) -> None:
        """Select the policy."""
        if use_reference:
            self._use_reference_policy()
        else:
            self._use_policy()
        yield
        self._use_policy()

    def _use_reference_policy(self) -> None:
        """Use the reference policy."""
        if self.use_separate_reference_adapter:
            self.actor.set_adapter("reference")
            for name, param in self.actor.named_parameters():
                if param is not None and "reference" in name:
                    param.requires_grad = False
        else:
            self.actor.base_model.disable_adapter_layers()

    def _use_policy(self) -> None:
        """Use the policy."""
        if self.use_separate_reference_adapter:
            self.actor.set_adapter("actor")
        else:
            self.actor.base_model.enable_adapter_layers()

    def _move_model_to_vllm(self) -> None:
        """Move the deepspeed model to vllm."""

        # TODO: Add support for ZeRO Stage 3
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        gather_if_zero3 = nullcontext
        model_ref = self.accelerator.unwrap_model(self.actor)
        model_ref.set_adapter("actor")
        with gather_if_zero3(list(model_ref.parameters())):
            model_ref.merge_adapter()
            for name, param in model_ref.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if model_ref.prefix in name:
                    continue

                if "original_module" in name:
                    continue

                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights([(name, param.data)])
            model_ref.unmerge_adapter()

        self.llm.reset_prefix_cache()

    def _generate_with_vllm_colocate(
        self, prompts: list[tuple[str, int]], group_size: int
    ) -> list[torch.Tensor]:

        # I need to make the following happen
        # prompts = [prompt1, prompt1, ..., prompt1 (group_size times), prompt2, prompt2, ..., prompt2 (group_size times), ...]

        # The below line returns a list: [prompt1 * group_size, ..., promptN * group_size],
        # where N is the data batch size per gpu, list length is group_size * N
        group_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        prompts_ids = [prompt["input_ids"] for prompt in group_prompts]
        prompts_text = [prompt["text"] for prompt in group_prompts]
        prompts_text = [
            re.sub(rf"^({re.escape(str(self.pad_token))})+", "", text)
            for text in prompts_text
        ]

        generation_kwargs = {
            "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1 if self.top_k is None else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "max_tokens": self.max_output_tokens,
            "min_tokens": (
                0 if self.min_output_tokens is None else self.min_output_tokens
            ),
        }
        sampling_params = SamplingParams(**generation_kwargs)

        if self.vllm_config.tensor_parallel_size > 1:

            orig_size = len(prompts_text)

            gathered_prompts_ids = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]
            gathered_prompts_text = [
                None for _ in range(self.vllm_config.tensor_parallel_size)
            ]

            torch.distributed.all_gather_object(
                gathered_prompts_ids, prompts_ids, group=self.tp_group
            )
            torch.distributed.all_gather_object(
                gathered_prompts_text, prompts_text, group=self.tp_group
            )

            all_prompts_ids = [
                prompt_id for sublist in gathered_prompts_ids for prompt_id in sublist
            ]
            all_prompts_text = [
                prompt_text
                for sublist in gathered_prompts_text
                for prompt_text in sublist
            ]
        else:
            all_prompts_text = prompts_text
            all_prompts_ids = prompts_ids

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        all_outputs = self.llm.generate(
            all_prompts_text,
            sampling_params=sampling_params,
            use_tqdm=True,
        )  # Change this to False

        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        if self.vllm_config.tensor_parallel_size > 1:
            # Slice completions for this rank within its TP group.
            # Each rank generates all outputs — we keep only our share.
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(
                local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size
            )
            completion_ids = completion_ids[tp_slice]
            prompts_ids = all_prompts_ids[tp_slice]

        completion_ids = [
            torch.cat(
                [
                    torch.cat(
                        prompts_ids[group_size * i : group_size * (i + 1)], dim=0
                    ),
                    stack_and_pad_experiences(
                        completion_ids[group_size * i : group_size * (i + 1)],
                        padding_values=[self.pad_token_id],
                        device=self.device,
                    )[0],
                ],
                dim=1,
            )
            for i, _ in enumerate(prompts)
        ]

        num_input_tokens = [prompt_ids.shape[1] for prompt_ids in prompts_ids][
            ::group_size
        ]
        action_masks = []

        for i, completion_id in enumerate(completion_ids):
            action_mask = torch.zeros_like(completion_id, device=self.device)
            action_mask[:, num_input_tokens[i] :] = True
            action_mask[completion_id == self.pad_token_id] = False
            action_mask = action_mask[:, 1:]
            action_masks.append(action_mask)

        return completion_ids, action_masks

    @staticmethod
    def _memory_efficient_logits(
        logits: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the log probabilities for a set of previously generated ids, looping to reduce peak memory consumption.

        :param logits: Logits.
        :type logits: torch.Tensor
        :param index: Index.
        :type index: torch.Tensor
        :return: Log probabilities of the completion IDs.
        :rtype: torch.Tensor
        """
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def _configure_batch_size(
        self,
        batch_size: int,
        clone: bool,
        reduce_memory_peak: bool,
        micro_batch_size_per_gpu: int | None,
    ) -> None:
        if self.accelerator is None or clone:
            self.batch_size_per_process = batch_size
            return

        if batch_size % self.accelerator.num_processes != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by the number of processes ({self.accelerator.num_processes})."
            )

        ds_config = self.accelerator.state.deepspeed_plugin.deepspeed_config

        if reduce_memory_peak:
            self.batch_size_per_process = 1
            self.micro_batch_size_per_gpu = 1
            ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
            gradient_accumulation_steps = batch_size / self.accelerator.num_processes
            ds_config["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
            return

        self.batch_size_per_process = int(batch_size / self.accelerator.num_processes)

        if micro_batch_size_per_gpu is None:
            if (
                self.batch_size_per_process
                % ds_config.get("gradient_accumulation_steps", 1)
                != 0
            ):
                raise ValueError(
                    f"Batch size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and gradient accumulation steps ({self.accelerator.state.deepspeed_plugin.deepspeed_config.get('gradient_accumulation_steps', 1)})."
                    "Gradient accumulation steps can be updated in the deepspeed config by changing the 'gradient_accumulation_steps' parameter."
                )
            self.micro_batch_size_per_gpu = (
                self.batch_size_per_process
                // ds_config.get("gradient_accumulation_steps", 1)
            )
            if self.micro_batch_size_per_gpu == 0:
                raise ValueError("Calculated micro_batch_size_per_gpu is 0...")

            if ds_config.get("train_micro_batch_size_per_gpu", "auto") == "auto":
                ds_config["train_micro_batch_size_per_gpu"] = (
                    self.micro_batch_size_per_gpu
                )
            return

        self.micro_batch_size_per_gpu = int(micro_batch_size_per_gpu)
        if (
            batch_size
            % (self.micro_batch_size_per_gpu * self.accelerator.num_processes)
            != 0
        ):
            raise ValueError(
                f"When specifying micro_batch_size_per_gpu, batch_size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and micro_batch_size_per_gpu ({self.micro_batch_size_per_gpu})."
            )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
        gradient_accumulation_steps = (
            batch_size / self.accelerator.num_processes / self.micro_batch_size_per_gpu
        )
        warnings.warn(
            f"Overwriting deepspeed config gradient accumulation steps from {self.accelerator.state.deepspeed_plugin.deepspeed_config.get('gradient_accumulation_steps', 'auto')} to {gradient_accumulation_steps}"
        )
        ds_config["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
        return
