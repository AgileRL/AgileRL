# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Field descriptions shared by more than one algorithm.

Defined once so the wording cannot drift between algorithms that expose
the same field.
"""

from __future__ import annotations

BETA = (
    "Weight on the KL penalty holding the policy near the reference "
    "model. 0 disables the penalty entirely."
)
MINI_BATCH = (
    "Trajectories one optimizer step covers, per rank. Must be a whole "
    "number of micro-batches. This is a learning-cadence decision, not "
    "a memory one: it sets how often the policy moves."
)
MICRO_BATCH = (
    "Trajectories per backward pass, per GPU. Affects only memory: it "
    "changes what fits, not what the run learns."
)

TAU = (
    "Polyak coefficient for the soft target-network update. Smaller values "
    "track the online network more slowly and train more stably."
)
LR = "Optimizer learning rate."
LR_ACTOR = "Optimizer learning rate for the actor (policy) network."
LR_CRITIC = "Optimizer learning rate for the critic (value) network."
POLICY_FREQ = "Critic updates between each delayed actor update."
OU_NOISE = (
    "Use temporally-correlated Ornstein-Uhlenbeck exploration noise instead of "
    "uncorrelated Gaussian noise."
)
EXPL_NOISE = "Scale of the exploration noise added to actions."
MEAN_NOISE = "Mean the Ornstein-Uhlenbeck noise reverts towards."
THETA = "Rate at which Ornstein-Uhlenbeck noise reverts to its mean."
DT = "Timestep used when integrating the Ornstein-Uhlenbeck noise process."
VECT_NOISE_DIM = (
    "Independent exploration-noise streams, one per vectorized environment. "
    "Resolved from environment.num_envs."
)
NET_CONFIG = (
    "Resolved from the manifest's network section; setting it here is not required."
)
NET_CONFIG_PER_AGENT = (
    "Resolved from the manifest's network section. May be a single config, or "
    "one per agent id."
)
CLIP_COEF = (
    "PPO-style surrogate clipping range. Caps how far one update may move the "
    "policy from the behaviour that collected the data."
)
ENT_COEF = (
    "Weight on the entropy bonus. Higher values keep the policy exploring for longer."
)
VF_COEF = "Weight on the value-function loss in the combined objective."
GAE_LAMBDA = (
    "Generalized advantage estimation trace decay, trading bias against "
    "variance. 1.0 is plain Monte-Carlo returns."
)
TARGET_KL = (
    "Stop the epoch early once the policy has moved this far in KL. Unset runs "
    "every epoch."
)
UPDATE_EPOCHS = "Optimizer passes over each batch of collected rollouts."
MAX_GRAD_NORM = "Gradients are clipped to this global norm before each step."
ACTION_STD_INIT = "Initial standard deviation for continuous action sampling."
SHARE_ENCODERS = (
    "Share one encoder between actor and critic instead of training separate copies."
)
DOUBLE = (
    "Use double Q-learning: select the next action with the online network and "
    "score it with the target, which curbs value overestimation."
)
TOP_K = "Keep only the K most likely next tokens at each generation step."
TOP_P = (
    "Nucleus sampling: keep the smallest set of tokens whose probability "
    "sums to this value."
)
MIN_P = (
    "Drop tokens whose probability is below this fraction of the most "
    "likely token's probability."
)
REPETITION_PENALTY = "Penalty applied to tokens already generated, to discourage loops."
TEMPERATURE = (
    "Sampling temperature for generation. Higher values diversify rollouts; 0 "
    "is greedy decoding."
)
MAX_OUTPUT_TOKENS = "Hard cap on tokens generated per completion."
MIN_OUTPUT_TOKENS = (
    "Floor on tokens generated per completion. Unset lets the model stop as "
    "soon as it emits an end-of-sequence token."
)
COSINE_LR = (
    "Cosine learning-rate schedule with warmup. Unset holds the learning rate constant."
)
VLLM_CONFIG = "vLLM engine settings for generation."
USE_VLLM = (
    "Run generation through vLLM in the trainer process. Resolved from "
    "training.rollout_mode: async rollout moves generation onto its own "
    "engines and turns this off."
)
IS_CORRECTION = (
    "Correct for the mismatch between the vLLM sampler's log-probabilities and "
    "the trainer's recomputed ones."
)
IS_CAP = "Upper bound applied to the vLLM importance-sampling correction ratio."
ADVANTAGE_GRANULARITY = (
    "Level the advantage is computed at. 'auto' follows the environment: "
    "per-turn for multi-turn, per-trajectory otherwise."
)
ADVANTAGE_GRANULARITY_TOKEN = (
    "Level the advantage is computed at. 'turn' enforces turn-level updates, "
    "'token' token-level, and 'auto' uses token-level only when every sample "
    "is single-turn."
)
IS_LEVEL = (
    "Level the importance-sampling ratio is computed at. Coarser levels are "
    "lower variance but a blunter correction."
)
TURN_RATIO_POOLING = (
    "How per-token ratios combine into a turn ratio: 'sum' in log space "
    "(the product of ratios) or 'mean'."
)
WHITEN_ADVANTAGES = (
    "Normalize advantages to zero mean and unit variance across the batch."
)
GROUP_SIZE = (
    "Completions sampled per prompt. The group is what the advantage is "
    "computed relative to, so this is the main GRPO setting to tune."
)
