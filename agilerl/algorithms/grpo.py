import copy
import gc
import glob
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from gymnasium import spaces
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.core import EvolvableAlgorithm, RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.typing import DeviceType, ExperiencesType
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    create_warmup_cosine_scheduler,
    get_experiences_samples,
    remove_nested_files,
    stack_and_pad_experiences,
)
from agilerl.utils.llm_utils import (
    HuggingFaceGym,
)

DeepSpeedOptimizerType = Union[
    DeepSpeedZeroOptimizer,  # ZeRO Stage 1 & 2 optimizer
    DeepSpeedZeroOptimizer_Stage3,  # ZeRO Stage 3 optimizer
]


class GRPO(RLAlgorithm):
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
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param clip_coef: Surrogate clipping coefficient, defaults to 0.2
    :type clip_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.5
    :type max_grad_norm: float, optional
    :param update_epochs: Number of policy update epochs, defaults to 4
    :type update_epochs: int, optional
    :param group_size: Group size, defaults to 8
    :type group_size: int, optional
    :param temperature: Temperature, controls randomness of text generation
    :type temperature: float, optional
    :param calc_position_embeddings: Flag indicating whether to calculate position embeddings, defaults to True
    :type calc_position_embeddings: bool, optional
    :param reduce_memory_peak: Flag to reduce memory peak in the _get_log_probs method, defaults to False
    :type reduce_memory_peak: bool, optional
    :param max_output_tokens: Max number of answer tokens, defaults to 512
    :type max_output_tokens: int, optional
    :param min_output_tokens: Minimum output tokens, defaults to 0
    :type min_output_tokens: int, optional
    :param cosine_lr_schedule_config: Config for cosine lr scheduling, defaults to None
    :type cosine_lr_schedule_config: CosineLRScheduleConfig, optional
    :param gradient_checkpointing: Flag to enable gradient checkpointing, defaults to False
    :type gradient_checkpointing: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param clone: Flag to indicate if the instantiation is a cloning, defaults to False
    :type clone: bool, optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        actor_network: PreTrainedModel,
        pad_token_id: int,
        hp_config: Optional[HyperparameterConfig] = None,
        index: int = 0,
        batch_size: int = 8,
        beta: float = 0.001,
        lr: float = 5e-7,
        clip_coef: float = 0.2,
        max_grad_norm: float = 0.1,
        update_epochs: int = 1,
        group_size: int = 8,
        temperature: float = 0.9,
        calc_position_embeddings: bool = True,
        reduce_memory_peak: bool = False,
        max_output_tokens: int = 1024,
        min_output_tokens: Optional[int] = None,
        cosine_lr_schedule_config: Optional[CosineLRScheduleConfig] = None,
        gradient_checkpointing: bool = True,
        accelerator: Optional[Accelerator] = None,
        device: str = "cpu",
        clone: bool = False,
        wrap: bool = False,
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
            accelerator=None,
            normalize_images=False,
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
        self.batch_size = batch_size
        self.lr = lr
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.group_size = group_size
        self.beta = beta
        self.pad_token_id = pad_token_id
        self.calc_position_embeddings = calc_position_embeddings
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_output_tokens,
            min_new_tokens=min_output_tokens,
            pad_token_id=pad_token_id,
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.cosine_lr_schedule_config = cosine_lr_schedule_config
        self.wrap = wrap
        if max_grad_norm and accelerator is not None and device == "cuda:0":
            warnings.warn(
                "Argument 'max_grad_norm' will be overwritten by the 'gradient_clipping' value set in the deepspeed config."
            )
            self.max_grad_norm = None
        else:
            self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.reduce_memory_peak = reduce_memory_peak
        self.local_rank = device.split(":")[-1]
        self.accelerator = accelerator
        self._initialize_actors(actor_network, not clone)
        del actor_network

    def get_action(
        self, states: List[Dict[str, torch.Tensor]], training: bool = True
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Returns the next action to take in the environment.

        :param states: Environment observation, or multiple observations in a batch
        :type states: numpy.ndarray[float]
        :param training: Flag to indicate training mode, defaults to True
        :type training: bool, optional
        """
        group_size = self.group_size if training else 1
        self.actor.eval()
        with torch.no_grad():
            action_masks = []
            completion_ids = []
            for state in states:
                state["input_ids"] = (
                    state["input_ids"].repeat(group_size, 1).to(self.actor.device)
                )
                state["attention_mask"] = (
                    state["attention_mask"].repeat(group_size, 1).to(self.actor.device)
                )
                completion_id = self.actor.generate(
                    **state,
                    generation_config=self.generation_config,
                )
                completion_ids.append(completion_id)
                action_mask = torch.zeros_like(
                    completion_id, dtype=torch.bool, device=self.device
                )
                action_mask[:, state["input_ids"].shape[1] :] = True
                action_mask[completion_id == self.pad_token_id] = False
                action_mask = action_mask[:, 1:]
                action_masks.append(action_mask)
        return completion_ids, action_masks

    def learn(self, experiences: ExperiencesType) -> Tuple[float, float]:
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
        with torch.no_grad():
            reference_log_probs = self._get_logprobs(
                completion_ids, use_reference=True, eval_mode=True
            )
            old_log_probs = self._get_logprobs(
                completion_ids, use_reference=False, eval_mode=True
            )
        experiences = (
            completion_ids,
            action_masks,
            advantages,
            old_log_probs,
            reference_log_probs,
        )
        num_samples = advantages.shape[0]
        batch_idxs = np.arange(num_samples)
        mean_loss, mean_kl = 0, 0
        for _ in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[
                    start : min((start + self.batch_size), num_samples)
                ]
                (
                    batch_ids,
                    batch_action_mask,
                    batch_advantages,
                    batch_old_log_probs,
                    batch_reference_log_probs,
                ) = get_experiences_samples(minibatch_idxs, *experiences)
                batch_log_probs = self._get_logprobs(
                    batch_ids, use_reference=False, eval_mode=False
                )
                loss, kl = self._grpo_loss(
                    batch_action_mask,
                    batch_log_probs,
                    batch_old_log_probs,
                    batch_reference_log_probs,
                    batch_advantages,
                )
                if not loss.isfinite():
                    continue
                self._backward_pass(loss)
                mean_loss += loss.item()
                mean_kl += kl.item()
        mean_loss /= len(completion_ids)
        mean_kl /= len(completion_ids)
        return mean_loss, mean_kl

    def test(
        self,
        env: HuggingFaceGym,
        loop: int = 1,
    ) -> torch.Tensor:
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: HuggingFaceGym environment
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :return: Mean test score of the agent
        :rtype: float
        """
        with env.eval():
            prompts = env.reset(reset_dataloaders=False)
            rewards = []
            for _ in range(loop):
                completion_ids, _ = self.get_action(prompts, training=False)
                next_prompts, reward = env.step(completion_ids)
                prompts = next_prompts
                rewards.append(reward)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        reward_tensor = torch.cat(rewards)
        return reward_tensor

    def _initialize_actors(
        self, actor_network: PreTrainedModel, create_reference_net: bool
    ):
        """Initialize the actor network and reference network.

        :param actor_network: Actor network
        :type actor_network: PreTrainedModel
        :param create_reference_net: Flag to indicate to create a reference network
        :type create_reference_net: bool
        """
        self._create_policy_network(actor_network)
        if create_reference_net:
            self._create_reference_policy_network(actor_network)
        if self.accelerator is not None:
            self.wrap_models(create_reference_net)
        else:
            self.actor = self.actor.to(self.device)
            if self.gradient_checkpointing:
                self.actor.gradient_checkpointing_enable()
            if create_reference_net:
                self.reference_actor = self.reference_actor.to(self.device)

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval=self.actor, policy=True))
        if create_reference_net:
            self.register_network_group(
                NetworkGroup(eval=self.reference_actor, policy=True)
            )
            self.reference_actor.eval()

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        kl = self._calculate_kl_divergence(log_probs, reference_log_probs)
        log_probs_ratio = torch.exp(log_probs - old_log_probs)
        clipped_log_probs_ratio = log_probs_ratio.clamp(
            1 - self.clip_coef, 1 + self.clip_coef
        )
        surrogate = log_probs_ratio * advantages
        clipped_surrogate = clipped_log_probs_ratio * advantages
        loss = -torch.min(surrogate, clipped_surrogate) + self.beta * kl
        loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)
        return loss.mean(), kl.mean()

    def _get_logprobs(
        self, ids: torch.Tensor, use_reference: bool = False, eval_mode: bool = False
    ) -> torch.Tensor:
        """Find the log probabilities for a set of previously generated ids.

        :param ids: Completion IDs.
        :type ids: torch.Tensor
        :param use_reference: Flag to indicate to use reference policy, defaults to False
        :type use_reference: bool, optional
        :param eval_mode: Flag to indicate setting policy network to evaluation mode, defaults to False
        :type eval_mode: bool, optional
        :return: Log probabilities of the completion IDs.
        :rtype: torch.Tensor
        """
        policy = self.reference_actor if use_reference else self.actor
        policy.train(mode=not eval_mode)
        attention_mask = ids != self.pad_token_id
        model_kwargs = {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.calc_position_embeddings:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
            model_kwargs |= {"position_ids": position_ids}
        if self.reduce_memory_peak:
            log_probs_list = []
            for input_id, mask in zip(ids, attention_mask):
                input_id = input_id.reshape(1, -1)
                mask = mask.reshape(1, -1)
                kwargs = {
                    "input_ids": input_id,
                    "attention_mask": mask,
                    "use_cache": False,
                }
                logit = policy.forward(**kwargs).logits
                log_prob = (
                    F.log_softmax(logit[:, :-1], dim=-1)
                    .gather(dim=-1, index=input_id[:, 1:].unsqueeze(-1))
                    .squeeze(-1)
                )
                log_probs_list.append(log_prob)
                del logit
            log_probs = torch.cat(log_probs_list)
            del log_probs_list
        else:
            logits = policy.forward(**model_kwargs).logits
            log_probs = (
                F.log_softmax(logits[:, :-1], dim=-1)
                .gather(dim=-1, index=ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )

        return log_probs

    def _create_policy_network(
        self, network: PreTrainedModel
    ) -> Tuple[
        Union[nn.Module, DeepSpeedEngine], Union[Optimizer, DeepSpeedOptimizerType]
    ]:
        """Create policy network.

        :param network: Pre-trained LLM
        :type network: PreTrainedModel
        :param ds_config: Deepspeed config
        :type ds_config: Union[Dict[str, Any], None]
        :return: Policy network and reference network
        :rtype: Tuple[Union[nn.Module, DeepSpeedEngine], Union[Optimizer, DeepSpeedOptimizerType]]
        """
        if self.accelerator is not None:
            if (
                self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "train_micro_batch_size_per_gpu"
                ]
                == "auto"
            ):
                self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "train_micro_batch_size_per_gpu"
                ] = 2

            if self.gradient_checkpointing:
                self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "activation_checkpointing"
                ] = {
                    "partition_activations": True,
                    "cpu_checkpointing": True,
                    "synchronize_checkpoint_boundary": True,
                    "number_checkpoints": 2,
                }
        self.actor = network
        self.optimizer = OptimizerWrapper(
            optim.AdamW, networks=[self.actor], lr=self.lr
        )
        self.lr_scheduler = (
            create_warmup_cosine_scheduler(
                self.optimizer.optimizer, self.cosine_lr_schedule_config, 1e-8, self.lr
            )
            if self.cosine_lr_schedule_config is not None
            else None
        )

    def _create_reference_policy_network(
        self, network: PreTrainedModel
    ) -> Union[nn.Module, DeepSpeedEngine]:
        """Create reference policy network.

        :param network: Pre-trained LLM
        :type network: PreTrainedModel
        :param ds_config: Deepspeed config
        :type ds_config: Union[Dict[str, Any], None]Í
        :return: Policy network and reference network
        :rtype: Union[nn.Module, DeepSpeedEngine]
        """
        self.reference_actor = copy.deepcopy(network)
        self.reference_actor.eval()
        for param in self.reference_actor.parameters():
            param.requires_grad = False

    def _backward_pass(self, loss: float) -> None:
        """Perform a backward pass

        :param loss: Loss
        :type loss: float
        """
        if self.accelerator is not None:
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr = self.lr_scheduler.get_last_lr()[0]

    def save_checkpoint(self, path: str) -> None:
        """
        Override the save_checkpoint method to provide guidance on the correct method to use.
        :param path: Location to save checkpoint at
        :type path: string
        """
        raise NotImplementedError(
            "The save_checkpoint method is not supported for this algorithm class. "
            "Please use agent.actor.save_pretrained(checkpoint_path) instead."
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Location to load checkpoint from
        :type path: string
        """
        raise NotImplementedError(
            "The load_checkpoint method is not supported for this algorithm class."
            """
            To load a saved LLM, please load the model as follows, and then re-instantiate the GRPO
            class.

            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
            model = PeftModel.from_pretrained(base_model, "/path/to/adapter/folder")
            """
        )

    def _save_distributed_actor(self, path: str) -> None:
        """
        Override the save_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to save the checkpoint at
        :type path: str
        """
        if self.accelerator is not None:
            os.makedirs(path, exist_ok=True)
            self.actor.save_checkpoint(path, tag="checkpoint")
        else:
            warnings.warn(
                "Distributed actor save not supported for non-distributed training."
            )

    def _load_distributed_actor(self, path: str) -> None:
        """
        Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to load the checkpoint from
        :type path: str
        """
        if self.accelerator is not None:
            deepspeed_dirs = sorted(glob.glob(f"{path}/checkpoint"))
            assert len(deepspeed_dirs) > 0
            self.actor.load_checkpoint(
                path,
                tag="checkpoint",
                load_module_strict=True,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            self.accelerator.deepspeed_engine_wrapped.engine = self.actor
        else:
            warnings.warn(
                "Distributed actor load not supported for non-distributed training."
            )

    @classmethod
    def load(
        cls,
        path: str,
        device: DeviceType = "cpu",
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        raise NotImplementedError(
            "The load class method is not supported for this algorithm class."
            """
            To load a saved LLM, please load the model as follows, and then re-instantiate the GRPO
            class, using the pre-trained model.

            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
            model = PeftModel.from_pretrained(base_model, "/path/to/adapter/folder")
            """
        )

    def wrap_models(self, create_reference_net):
        """Wrap the models in the accelerator

        :param create_reference_net: Flag to indicate if the reference network should be wrapped
        :type create_reference_net: bool
        """
        if self.accelerator is not None:
            self.actor, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.actor, self.optimizer.optimizer, self.lr_scheduler
            )
            if create_reference_net:
                deepspeed_plugin = self.accelerator.state.deepspeed_plugin
                config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)
                config_kwargs["zero_optimization"]["stage"] = 0
                self.reference_actor, *_ = deepspeed.initialize(
                    model=self.reference_actor, config=config_kwargs
                )

    def clone(self, index: Optional[int] = None, wrap: bool = True):
        """Creates a clone of the algorithm.

        :param index: The index of the clone, defaults to None
        :type index: Optional[int], optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: A clone of the algorithm
        :rtype: EvolvableAlgorithm
        """
        if self.accelerator is not None:
            self.accelerator.free_memory()
            self.accelerator.wait_for_everyone()
            self._save_distributed_actor(f"GRPO_test/agent_{self.index}")
            self.accelerator.wait_for_everyone()
            input_args = EvolvableAlgorithm.inspect_attributes(
                self, input_args_only=True
            )
            input_args["clone"] = True
            input_args["actor_network"] = self.accelerator.unwrap_model(self.actor)
            clone = type(self)(**input_args)
            clone.reference_actor = self.reference_actor
            clone.reference_actor.eval()
            accelerator = clone.accelerator
            self.lr_scheduler = self.accelerator.unwrap_model(self.lr_scheduler)
            lr_scheduler = self.lr_scheduler
            self.lr_scheduler = None
            clone = EvolvableAlgorithm.copy_attributes(self, clone)
            clone.accelerator = accelerator
            clone.lr_scheduler = lr_scheduler
            if index is not None:
                clone.index = index
            clone.accelerator.wait_for_everyone()
            clone._load_distributed_actor(f"GRPO_test/agent_{self.index}")
            clone.accelerator.wait_for_everyone()
            saved_state_files = glob.glob(f"GRPO_test/agent_{self.index}/*")
            if clone.accelerator.is_main_process:
                remove_nested_files(saved_state_files)
        else:
            actor_state_dict = self.actor.state_dict()
            optimizer_state_dict = self.optimizer.optimizer.state_dict()
            lr_scheduler_state_dict = (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            )
            input_args = EvolvableAlgorithm.inspect_attributes(
                self, input_args_only=True
            )
            input_args["clone"] = True
            input_args["actor_network"] = self.actor
            clone = type(self)(**input_args)
            clone.reference_actor = self.reference_actor
            clone.reference_actor.eval()
            lr_scheduler = self.lr_scheduler
            self.lr_scheduler = None
            clone = EvolvableAlgorithm.copy_attributes(self, clone)
            clone.lr_scheduler = lr_scheduler
            if index is not None:
                clone.index = index
            clone.actor.load_state_dict(actor_state_dict)
            clone.optimizer.optimizer.load_state_dict(optimizer_state_dict)
            if lr_scheduler_state_dict is not None:
                clone.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        return clone
