from agilerl import HAS_LLM_DEPENDENCIES

from .bc_lm import BC_LM, BC_Evaluator, BC_Policy
from .cispo import CISPO
from .cqn import CQN
from .ddpg import DDPG
from .dqn import DQN
from .dqn_rainbow import RainbowDQN
from .grpo import GRPO
from .gspo import GSPO
from .ilql import ILQL
from .ippo import IPPO
from .maddpg import MADDPG
from .matd3 import MATD3
from .neural_ts_bandit import NeuralTS
from .neural_ucb_bandit import NeuralUCB
from .ppo import PPO
from .ppo_llm import PPO as LLMPPO
from .reinforce_llm import REINFORCE as LLMReinforce
from .td3 import TD3

__all__ = [
    "BC_LM",
    "CQN",
    "DDPG",
    "DQN",
    "ILQL",
    "IPPO",
    "MADDPG",
    "MATD3",
    "PPO",
    "TD3",
    "BC_Evaluator",
    "BC_Policy",
    "NeuralTS",
    "NeuralUCB",
    "RainbowDQN",
]

if HAS_LLM_DEPENDENCIES:
    from .cispo import CISPO
    from .dpo import DPO
    from .grpo import GRPO
    from .gspo import GSPO
    from .ppo_llm import PPO as LLMPPO
    from .reinforce_llm import REINFORCE as LLMReinforce
    from .sft import SFT

    __all__ += ["CISPO", "DPO", "GRPO", "GSPO", "LLMPPO", "SFT", "LLMReinforce"]
