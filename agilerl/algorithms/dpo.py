import gc

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from gymnasium import spaces
from peft import LoraConfig
from transformers import PreTrainedModel

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, NetworkGroup
from agilerl.typing import ExperiencesType, LLMObsType
from agilerl.utils.algo_utils import get_experiences_samples
from agilerl.utils.llm_utils import PreferenceGym


class DPO(LLMAlgorithm):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        actor_network: PreTrainedModel,
        pad_token_id: int,
        pad_token: str,
        hp_config: HyperparameterConfig | None = None,
        index: int = 0,
        batch_size: int = 16,
        lr: float = 0.000005,
        beta: float = 0.001,
        max_grad_norm: float = 0.1,
        update_epochs: int = 1,
        calc_position_embeddings: bool = True,
        micro_batch_size_per_gpu: int | None = None,
        reduce_memory_peak: bool = False,
        device: str = "cpu",
        lora_config: LoraConfig | None = None,
        accelerator: Accelerator | None = None,
        wrap: bool = True,
        clone: bool = False,
        use_separate_reference_adapter: bool = False,
        seed: int = 42,
    ):
        device = (
            f"cuda:{accelerator.process_index}"
            if accelerator is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        super().__init__(
            observation_space,
            action_space,
            actor_network,
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
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=None,
            hp_config=hp_config,
            wrap=wrap,
            device=device,
            accelerator=accelerator,
            name="DPO",
        )
        self.beta = beta
        self.temperature = (
            1  # Temperature for logits calculation, DPO does not use temperature
        )
        self.use_vllm = False  # DPO does not use VLLM
        self.update_epochs = update_epochs
        self._initialize_actors(actor_network, not clone)
        # Register network groups for mutations
        self.register_network_group(NetworkGroup(eval_network=self.actor, policy=True))
        if self.wrap:
            self.wrap_models()

    def get_action(
        self, obs: LLMObsType, training: bool = True
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns the action of the agent.

        :param obs: The observation of the agent
        :type obs: LLMObsType
        :param training: Whether the agent is training or not
        :type training: bool
        :return: The action of the agent
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
        raise NotImplementedError(
            "DPO is an offline algorithm and therefore does not require completions to be generated."
        )

    def learn(
        self,
        experiences: ExperiencesType,
        training: bool = True,
    ) -> tuple[float, float, float]:
        """
        Updates agent network parameters to learn from preference data.

        :param experiences: Batched chosen_input_ids, rejected_input_ids, chosen_attention_mask, rejected_attention_mask and rewards
        :type experiences: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        :param training: Whether the agent is training or not
        :type training: bool
        :return: mean loss, mean chosen reward, mean rejected reward
        :rtype: tuple[float, float, float]
        """
        gc.collect()
        torch.cuda.empty_cache()
        # The following tensors are size [batch_size, max_length]
        chosen_input_ids: torch.Tensor = experiences["chosen_input_ids"]
        rejected_input_ids: torch.Tensor = experiences["rejected_input_ids"]
        chosen_attention_mask: torch.Tensor = experiences["chosen_attention_mask"]
        rejected_attention_mask: torch.Tensor = experiences["rejected_attention_mask"]
        # Check first that all tensors have the same max length before calculating the masks
        assert (
            chosen_input_ids.shape[1]
            == rejected_input_ids.shape[1]
            == chosen_attention_mask.shape[1]
            == rejected_attention_mask.shape[1]
        ), "All tensors must have the same max length"
        max_length = chosen_input_ids.shape[1]
        prompt_lengths: list[int] = experiences["prompt_lengths"]
        prompt_masks = LLMAlgorithm.create_prompt_masks(
            prompt_lengths, max_length=max_length
        ).to(self.device)
        # FIXME double check that this is correct

        # Mask has to be shifted by 1 as output log probs dims are 1 shorter than input ids as first token is used to predict the first log prob
        chosen_mask = (prompt_masks * chosen_attention_mask)[:, 1:]
        rejected_mask = (prompt_masks * rejected_attention_mask)[:, 1:]
        num_samples = chosen_input_ids.shape[0]
        batch_size = min(num_samples, self.micro_batch_size_per_gpu)
        batch_idxs = np.arange(num_samples)
        mean_loss, mean_chosen_reward, mean_rejected_reward = 0.0, 0.0, 0.0

        with torch.no_grad():
            ref_rejected_log_probs = self._get_logprobs(
                rejected_input_ids,
                # batch_rejected_attention_mask,
                batch_size,
                use_reference=True,
                eval_mode=True,
            )
            ref_chosen_log_probs = self._get_logprobs(
                chosen_input_ids,
                # batch_chosen_attention_mask,
                batch_size,
                use_reference=True,
                eval_mode=True,
            )

        for _ in range(self.update_epochs):
            for start in range(0, num_samples, batch_size):
                minibatch_idxs = batch_idxs[
                    start : min((start + batch_size), num_samples)
                ]
                (
                    batch_chosen_input_ids,
                    # batch_chosen_attention_mask,
                    batch_rejected_input_ids,
                    # batch_rejected_attention_mask,
                    batch_chosen_mask,
                    batch_rejected_mask,
                    batch_ref_rejected_log_probs,
                    batch_ref_chosen_log_probs,
                ) = get_experiences_samples(
                    minibatch_idxs,
                    chosen_input_ids,
                    # chosen_attention_mask,
                    rejected_input_ids,
                    # rejected_attention_mask,
                    chosen_mask,
                    rejected_mask,
                    ref_rejected_log_probs,
                    ref_chosen_log_probs,
                )
                batch_rejected_log_probs = self._get_logprobs(
                    batch_rejected_input_ids,
                    # batch_rejected_attention_mask,
                    batch_size,
                    use_reference=False,
                    eval_mode=(not training),
                )
                batch_chosen_log_probs = self._get_logprobs(
                    batch_chosen_input_ids,
                    # batch_chosen_attention_mask,
                    batch_size,
                    use_reference=False,
                    eval_mode=(not training),
                )
                loss, chosen_reward, rejected_reward = self._dpo_loss(
                    batch_chosen_log_probs,
                    batch_rejected_log_probs,
                    batch_ref_chosen_log_probs,
                    batch_ref_rejected_log_probs,
                    batch_chosen_mask,
                    batch_rejected_mask,
                )
                if training:
                    self._backward_pass(loss)
                mean_loss += loss.item()
                mean_chosen_reward += chosen_reward.mean().item()
                mean_rejected_reward += rejected_reward.mean().item()
        mean_loss /= num_samples
        mean_chosen_reward /= num_samples
        mean_rejected_reward /= num_samples
        return mean_loss, mean_chosen_reward, mean_rejected_reward

    def _dpo_loss(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor,
        ref_chosen_log_probs: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the DPO loss.xw

        :param chosen_mask: Mask for the prompt and padding tokens of the chosen completions
        :type chosen_mask: torch.Tensor
        :param rejected_mask: Mask for the prompt and padding tokens of the rejected completions
        :type rejected_mask: torch.Tensor
        :param chosen_log_probs: Log probabilities of the chosen completions
        :type chosen_log_probs: torch.Tensor
        :param rejected_log_probs: Log probabilities of the rejected completions
        :type rejected_log_probs: torch.Tensor
        :param ref_chosen_log_probs: Log probabilities of the chosen completions using the reference model

        """
        # Mask and sum the logprobs
        assert (
            chosen_log_probs.shape == chosen_mask.shape
        ), f"Chosen log probabilities and mask must have the same shape, got {chosen_log_probs.shape} and {chosen_mask.shape}"
        chosen_log_probs = (chosen_log_probs * chosen_mask).sum(dim=-1)
        rejected_log_probs = (rejected_log_probs * rejected_mask).sum(dim=-1)
        ref_chosen_log_probs = (ref_chosen_log_probs * chosen_mask).sum(dim=-1)
        ref_rejected_log_probs = (ref_rejected_log_probs * rejected_mask).sum(dim=-1)
        rejected_ratio = rejected_log_probs - ref_rejected_log_probs
        chosen_ratio = chosen_log_probs - ref_chosen_log_probs
        with torch.no_grad():
            implicit_chosen_reward = self._compute_implicit_reward(
                chosen_log_probs, ref_chosen_log_probs
            )
            implicit_rejected_reward = self._compute_implicit_reward(
                rejected_log_probs, ref_rejected_log_probs
            )

        # Clean up intermediate tensors to free up memory
        chosen_log_probs = None
        rejected_log_probs = None
        ref_chosen_log_probs = None
        ref_rejected_log_probs = None
        chosen_mask = None
        rejected_mask = None

        return (
            -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean(),
            implicit_chosen_reward,
            implicit_rejected_reward,
        )

    def _compute_implicit_reward(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the preference reward for the chosen and rejected completions.

        :param log_probs: Log probabilities of the completions
        :type log_probs: torch.Tensor
        :param ref_log_probs: Log probabilities of the completions using the reference model
        :type ref_log_probs: torch.Tensor
        :return: Implicit reward for the chosen and rejected completions
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        implicit_reward = log_probs - ref_log_probs
        return self.beta * implicit_reward

    def test(self, env: PreferenceGym, loop: int = 1) -> torch.Tensor:
        """
        Returns the fitness (test) score tensor of the agent.

        :param env: The environment to be tested in
        :type env: PreferenceGym environment
        :param loop: Number of testing loops/episodes to complete. The returned score is the mean. Defaults to 3
        :type loop: int, optional
        :return: Test score tensor of the agent
        :rtype: torch.Tensor
        """
        with env.eval_mode(), torch.no_grad():
            prompts = env.reset()
            rewards = []
            for _ in range(loop):
                _, chosen_reward, rejected_reward = self.learn(prompts, training=False)
                reward_margin = chosen_reward - rejected_reward
                rewards.append(reward_margin)
                prompts = env.step()
        self.fitness.append(reward_margin)
        return reward_margin
