from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from gymnasium import spaces
from torch.nn.utils import clip_grad_norm_

from agilerl.algorithms.core import RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.base import EvolvableModule
from agilerl.typing import ExperiencesType, GymEnvType
from agilerl.utils.algo_utils import (
    flatten_experiences,
    get_experiences_samples,
    is_vectorized_experiences,
    obs_channels_to_first,
    stack_experiences,
)


class GRPO(RLAlgorithm):
    """The PPO algorithm class. PPO paper: https://arxiv.org/abs/1707.06347v2

    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
    :type hp_config: HyperparameterConfig, optional
    :param net_config: Network configuration, defaults to None
    :type net_config: dict, optional
    :param head_config: Head network configuration, defaults to None
    :type head_config: dict, optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 2048
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param mut: Most recent mutation to agent, defaults to None
    :type mut: str, optional
    :param clip_coef: Surrogate clipping coefficient, defaults to 0.2
    :type clip_coef: float, optional
    :param max_grad_norm: Maximum norm for gradient clipping, defaults to 0.5
    :type max_grad_norm: float, optional
    :param normalize_images: Flag to normalize images, defaults to True
    :type normalize_images: bool, optional
    :param update_epochs: Number of policy update epochs, defaults to 4
    :type update_epochs: int, optional
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        actor_network: EvolvableModule,
        reward_fns: List[Callable[..., float]],
        hp_config: Optional[HyperparameterConfig] = None,
        index: int = 0,
        batch_size: int = 64,
        beta: float = 0.04,
        lr: float = 2e-5,
        mut: Optional[str] = None,
        clip_coef: float = 0.2,
        update_epochs: int = 4,
        group_size: int = 10,
        device: str = "cpu",
        accelerator: Optional[Any] = None,
        wrap: bool = True,
    ) -> None:

        super().__init__(
            observation_space,
            action_space,
            index=index,
            hp_config=hp_config,
            device=device,
            accelerator=accelerator,
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
        assert isinstance(
            wrap, bool
        ), "Wrap models flag must be boolean value True or False."

        self.batch_size = batch_size
        self.lr = lr
        self.mut = mut
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.group_size = (
            group_size  # What if we assume that group size == num_envs ???
        )

        if actor_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"Passed actor network is of type {type(actor_network)}, but must be of type EvolvableModule."
                )
            self.actor = actor_network

        else:
            raise ValueError(
                "Actor network must be provided to GRPO in the form of a pre-trained huggingface model wrapped with DummyEvolvable"
            )

        # Use optim.W for LLM fine-tuning
        self.optimizer = OptimizerWrapper(optim.W, networks=[self.actor], lr=self.lr)

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval=self.actor, policy=True))

    def get_action(
        self,
        state: np.ndarray,
        action: Optional[torch.Tensor] = None,
        grad: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[
        Union[np.ndarray, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Returns the next action to take in the environment.

        :param state: Environment observation, or multiple observations in a batch
        :type state: numpy.ndarray[float]
        :param action: Action in environment to evaluate, defaults to None
        :type action: torch.Tensor(), optional
        :param grad: Calculate gradients on actions, defaults to False
        :type grad: bool, optional
        :param action_mask: Mask of legal actions 1=legal 0=illegal, defaults to None
        :type action_mask: numpy.ndarray, optional
        :param preprocess_obs: Flag to preprocess observations, defaults to True
        :type preprocess_obs: bool, optional
        """
        state = self.preprocess_observation(state)

        if not grad:
            self.actor.eval()
            with torch.no_grad():
                action_dist = self.actor(state, action_mask=action_mask)
        else:
            self.actor.train()
            action_dist = self.actor(state, action_mask=action_mask)

        if not isinstance(action_dist, torch.distributions.Distribution):
            raise ValueError(
                f"Expected action_dist to be a torch.distributions.Distribution, got {type(action_dist)}."
            )

        return_tensors = True
        if action is None:
            # Collect a group of samples from the policy
            action = action_dist.sample(sample_shape=torch.Size([self.group_size]))
            return_tensors = False
        else:
            action = action.to(self.device)

        # Determines action log prob of all samples in the group
        action_logprob = action_dist.log_prob(action)

        # Commented this out as will affect grpo, why was this included initially in ppo??
        # if len(action_logprob.shape) > 1:
        #     action_logprob = action_logprob.sum(dim=1)

        if return_tensors:
            return (
                (
                    self.scale_to_action_space(action, convert_to_torch=True)
                    if not self.discrete_actions
                    else action
                ),
                action_logprob,
            )
        else:
            return (
                (
                    self.scale_to_action_space(action.cpu().data.numpy())
                    if not self.discrete_actions
                    else action.cpu().data.numpy()
                ),
                action_logprob.cpu().data.numpy(),
            )

    def learn(self, experiences: ExperiencesType) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, log_probs, rewards, dones, values, next_state in that order.
        :type experience: Tuple[Union[numpy.ndarray, Dict[str, numpy.ndarray]], ...]
        """
        (states, actions, log_probs, rewards, dones, values, next_state) = (
            stack_experiences(*experiences)
        )

        # Bootstrapping returns using GAE advantage estimation
        dones = dones.long()

        # rewards of length self.group_size
        advantages = (rewards - np.mean(rewards)) / np.std(rewards)

        # Flatten experiences from (batch_size, num_envs, ...) to (batch_size*num_envs, ...)
        # after checking if experiences are vectorized
        experiences = (states, actions, log_probs, advantages, values)
        if is_vectorized_experiences(*experiences):
            experiences = flatten_experiences(*experiences)

        # Move experiences to algo device
        experiences = self.to_device(*experiences)

        (
            states,
            actions,
            log_probs,
            advantages,
            returns,
            values,
        ) = experiences

        num_samples = returns.size(0)
        batch_idxs = np.arange(num_samples)
        clipfracs = []
        mean_loss = 0
        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_idxs)
            for start in range(0, num_samples, self.batch_size):
                minibatch_idxs = batch_idxs[start : start + self.batch_size]
                (
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_advantages,
                    batch_returns,
                    batch_values,
                ) = get_experiences_samples(minibatch_idxs, *experiences)

                batch_actions = batch_actions.squeeze()
                batch_returns = batch_returns.squeeze()
                batch_log_probs = batch_log_probs.squeeze()
                batch_advantages = batch_advantages.squeeze()
                batch_values = batch_values.squeeze()

                if len(minibatch_idxs) > 1:
                    _, log_prob, entropy, value = self.get_action(
                        state=batch_states, action=batch_actions, grad=True
                    )

                    logratio = log_prob - batch_log_probs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    minibatch_advs = batch_advantages
                    minibatch_advs = (minibatch_advs - minibatch_advs.mean()) / (
                        minibatch_advs.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -minibatch_advs * ratio
                    pg_loss2 = -minibatch_advs * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.view(-1)
                    v_loss_unclipped = (value - batch_returns) ** 2
                    v_clipped = batch_values + torch.clamp(
                        value - batch_values, -self.clip_coef, self.clip_coef
                    )

                    v_loss_clipped = (v_clipped - batch_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    # actor + critic loss backprop
                    self.optimizer.zero_grad()
                    if self.accelerator is not None:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()

                    clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    mean_loss += loss.item()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        mean_loss /= num_samples * self.update_epochs
        return mean_loss

    def test(
        self,
        env: GymEnvType,
        swap_channels: bool = False,
        max_steps: Optional[int] = None,
        loop: int = 3,
    ) -> float:
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
        :param max_steps: Maximum number of testing steps, defaults to None
        :type max_steps: int, optional
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional

        :return: Mean test score of agent in environment
        :rtype: float
        """
        with torch.no_grad():
            rewards = []
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            for i in range(loop):
                state, info = env.reset()
                scores = np.zeros(num_envs)
                completed_episode_scores = np.zeros(num_envs)
                finished = np.zeros(num_envs)
                step = 0
                while not np.all(finished):
                    if swap_channels:
                        state = obs_channels_to_first(state)

                    action_mask = info.get("action_mask", None)
                    action, _, _, _ = self.get_action(state, action_mask=action_mask)
                    state, reward, done, trunc, info = env.step(action)
                    step += 1
                    scores += np.array(reward)
                    for idx, (d, t) in enumerate(zip(done, trunc)):
                        if (
                            d or t or (max_steps is not None and step == max_steps)
                        ) and not finished[idx]:
                            completed_episode_scores[idx] = scores[idx]
                            finished[idx] = 1
                rewards.append(np.mean(completed_episode_scores))
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit
