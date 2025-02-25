from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from transformers import GenerationConfig

from agilerl.algorithms.core import RLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.base import EvolvableModule
from agilerl.typing import ExperiencesType, GymEnvType


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
        pad_token_id: int,
        hp_config: Optional[HyperparameterConfig] = None,
        index: int = 0,
        batch_size: int = 64,
        beta: float = 0.04,
        lr: float = 2e-5,
        mut: Optional[str] = None,
        clip_coef: float = 0.2,
        update_epochs: int = 4,
        group_size: int = 5,
        temperature: float = 0.9,
        max_sequence_length: int = 512,
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
        self.pad_token_id = pad_token_id

        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_sequence_length,
            pad_token_id=pad_token_id,
        )

        if actor_network is not None:
            if not isinstance(actor_network, EvolvableModule):
                raise TypeError(
                    f"Passed actor network is of type {type(actor_network)}, but must be of type EvolvableModule."
                )
            self.actor = actor_network.to(self.device)

        else:
            raise ValueError(
                "Actor network must be provided to GRPO in the form of a pre-trained huggingface model wrapped with DummyEvolvable"
            )

        # Use optim.W for LLM fine-tuning
        self.optimizer = OptimizerWrapper(
            optim.AdamW, networks=[self.actor], lr=self.lr
        )

        if self.accelerator is not None and wrap:
            self.wrap_models()

        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval=self.actor, policy=True))

        # Create a Generation config e.g. self.generation_config = GenerationConfig

    def get_action(
        self, states: torch.Tensor, return_logits: bool = True
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

        # Update the reference model
        # Duplicate the prompt group_size times
        # Use the generation config to generate (n = group_size) answers
        # No need to tokenize as the environment will take care of tokenization
        import time

        with torch.no_grad():
            self.actor.eval()
            action_masks = []
            completion_ids = []
            completion_logprobs = []
            timing = []
            i = 0
            for state in states:
                # print("Epoch", i)
                # assert False
                state["input_ids"] = (
                    state["input_ids"].repeat(self.group_size, 1).to(self.device)
                )
                # print(state["input_ids"].shape)
                state["attention_mask"] = (
                    state["attention_mask"].repeat(self.group_size, 1).to(self.device)
                )
                t = time.time()
                completion = self.actor.generate(
                    **state,
                    generation_config=self.generation_config,
                    output_scores=return_logits,
                    return_dict_in_generate=return_logits,
                )
                timing.append(time.time() - t)
                completion_ids.append(completion_id := completion.sequences)
                print([len(score) for score in completion.scores])
                completion_logprobs.append(F.log_softmax(completion.scores))
                action_mask = torch.zeros_like(
                    completion_id, dtype=torch.bool, device=self.device
                )
                action_mask[:, state["input_ids"].shape[1] :] = True
                action_mask[completion_id == self.pad_token_id] = False
                action_mask = action_mask[:, 1:]
                action_masks.append(action_mask)
                i += 1
        # print(timing)
        return completion_ids, completion_logprobs, action_mask

    def learn(self, experiences: ExperiencesType) -> float:
        """Updates agent network parameters to learn from experiences.

        :param experience: List of batched states, actions, log_probs, rewards, dones, values, next_state in that order.
        :type experience: Tuple[Union[numpy.ndarray, Dict[str, numpy.ndarray]], ...]
        """

        # Calculate the advantage
        # for each experience, perform the GRPO loss update
        # Do we want to do one experience at a time or a batch of experiences at a time?
        pass

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
        with env.eval():
            # Eval training loop
            pass

    def _calculate_advantage(): ...

    def _grpo_loss(): ...
