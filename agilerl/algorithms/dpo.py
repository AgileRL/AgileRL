import torch

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.typing import ExperiencesType, LLMObsType


class DPO(LLMAlgorithm):

    def __init__(self): ...

    def get_action(
        self, obs: LLMObsType, training: bool = True
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raise NotImplementedError(
            "DPO is an offline algorithm and therefore does not require completions to be generated."
        )

    def learn(self, experiences: ExperiencesType) -> tuple[float, float]: ...

    def test(self, env, loop: int = 1) -> torch.Tensor: ...

    def _dpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor: ...
