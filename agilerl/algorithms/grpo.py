import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.nn.utils import clip_grad_norm_
from transformers import GenerationConfig

from agilerl.algorithms.core import RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.typing import ExperiencesType
from agilerl.utils.algo_utils import get_experiences_samples, stack_and_pad_experiences
from agilerl.utils.llm_utils import (
    DEEPSPEED_INFERENCE_CONFIG,
    DEEPSPEED_TRAINING_CONFIG,
    HuggingFaceGym,
)


class GRPO(RLAlgorithm):
    """The PPO algorithm class. PPO paper: https://arxiv.org/abs/1707.06347v2

    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
    :param actor_network: HuggingFace LLM
    :type actor_network: nn.Module
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
    :param deepspeed_training_config: Deepspeed training config, defaults to DEEPSPEED_TRAINING_CONFIG
    :type deepspeed_training_config: Dict[str, Any], optional
    :param deepspeed_inference_config: Deepspeed training config, defaults to DEEPSPEED_INFERENCE_CONFIG
    :type deepspeed_inference_config: Dict[str, Any], optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        actor_network: nn.Module,
        pad_token_id: int,
        hp_config: Optional[HyperparameterConfig] = None,
        index: int = 0,
        batch_size: int = 8,
        beta: float = 0.04,
        lr: float = 5e-6,
        clip_coef: float = 0.2,
        max_grad_norm: float = 0.1,
        update_epochs: int = 1,
        group_size: int = 8,
        temperature: float = 0.9,
        calc_position_embeddings: bool = True,
        reduce_memory_peak: bool = False,
        deepspeed_training_config: Optional[Dict[str, Any]] = DEEPSPEED_TRAINING_CONFIG,
        deepspeed_inference_config: Optional[
            Dict[str, Any]
        ] = DEEPSPEED_INFERENCE_CONFIG,
        device: str = "cpu",
    ) -> None:
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
        self.group_size = (
            group_size  # What if we assume that group size == num_envs ???
        )
        self.beta = beta
        self.pad_token_id = pad_token_id
        self.calc_position_embeddings = calc_position_embeddings
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_new_tokens=action_space.shape[0],
            pad_token_id=pad_token_id,
        )
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.reduce_memory_peak = reduce_memory_peak
        self.local_rank = device.split(":")[-1]
        if actor_network is not None:
            actor_network = actor_network.to(device)
            self.actor, self.optimizer, *_ = deepspeed.initialize(
                config=deepspeed_training_config,
                model=actor_network,
                model_parameters=actor_network.parameters(),
            )
            self.reference_actor = deepspeed.init_inference(
                model=actor_network, **deepspeed_inference_config
            )
        else:
            raise ValueError(
                "Actor network must be provided to GRPO in the form of a pre-trained huggingface model wrapped with DummyEvolvable"
            )
        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval=self.actor, policy=True))
        self.register_network_group(
            NetworkGroup(eval=self.reference_actor, policy=True)
        )

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
                completion_id = completion_id.to(self.device)
                completion_ids.append(completion_id)
                action_mask = torch.zeros_like(
                    completion_id, dtype=torch.bool, device=self.device
                )
                action_mask[:, state["input_ids"].shape[1] :] = True
                action_mask[completion_id == self.pad_token_id] = False
                action_mask = action_mask[:, 1:]
                action_masks.append(action_mask)
        return completion_ids, action_masks

    def learn(self, experiences: ExperiencesType) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experiences: Batched completion_ids, action_masks and rewards
        :type experiences: ExperiencesType
        """
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
        mean_loss, mean_kl, mean_grad_norm = 0, 0, 0
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
                self.actor.backward(loss)
                grad_norm = clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.step()
                mean_loss += loss.item()
                mean_kl += kl.item()
                mean_grad_norm += grad_norm.item()
        mean_loss /= len(completion_ids)
        mean_kl /= len(completion_ids)
        mean_grad_norm /= len(completion_ids)
        return mean_loss, mean_kl, mean_grad_norm

    def test(
        self,
        env: HuggingFaceGym,
        loop: int = 1,
    ) -> float:
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
        return mean_fit

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
            rewards = rewards.unsqueeze(-1)
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
            logit_list = []
            for input_id, mask in zip(ids, attention_mask):
                input_id = input_id.reshape(1, -1)
                mask = mask.reshape(1, -1)
                kwargs = {
                    "input_ids": input_id,
                    "attention_mask": mask,
                    "use_cache": False,
                }
                logit = policy.forward(**kwargs).logits
                logit = logit.to(self.device)
                logit_list.append(logit)
            logits = torch.cat(logit_list)
        else:
            logits = policy.forward(**model_kwargs).logits
        log_probs = (
            F.log_softmax(logits[:, :-1], dim=-1)
            .gather(dim=-1, index=ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        return log_probs

    def _set_reference_policy(self) -> None:
        """Set the reference policy to the current policy."""
        self.reference_actor = copy.deepcopy(self.actor)
        self.reference_actor.eval()
