# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Field descriptions shared by more than one algorithm or network spec.

Defined once so the wording cannot drift between specs that expose
the same field.
"""

from __future__ import annotations

BETA = (
    "Weight on the KL penalty holding the policy near the reference "
    "model. 0 disables the penalty entirely."
)
REFERENCE_UPDATE_INTERVAL = (
    "Optimizer steps between copies of the policy onto the reference "
    "model the KL penalty holds it near. Unset keeps the reference fixed."
)
KL_CLAMP = (
    "Per-token bound on the K3 KL penalty. 10 is the policy about 2.6 nats "
    "below the reference on a token; tokens further below get no KL "
    "gradient, so one diverged token cannot set the step's gradient norm. "
    "Tokens above the reference keep their KL pull. Unset removes the bound."
)
MINI_BATCH = (
    "Trajectories one optimizer step covers, per rank. Must be a whole "
    "number of micro-batches. This is a learning-cadence decision, not "
    "a memory one: it sets how often the policy moves."
)
MICRO_BATCH = (
    "Trajectories per backward pass, per GPU. Affects only memory: it "
    "changes what fits, not what the run learns. Picked from the memory "
    "estimate as the largest that fits; manifests cannot set it."
)
MICRO_BATCH_TOKENS = (
    "Tokens per packed backward pass, per GPU, when the forward packs "
    "sequences. Rows fill each pass up to this many tokens; a longer row runs "
    "alone. Affects only memory and speed, not what the run learns. Picked "
    "from the memory estimate as the fastest budget that fits; manifests "
    "cannot set it."
)
GRADIENT_CHECKPOINTING = (
    "Recompute block activations during the backward pass instead of "
    "storing them. Affects only memory and speed, not what the run learns. "
    "Picked from the memory estimate together with the micro-batch and "
    "fsdp.checkpoint_every_n_blocks, least recompute that fits; on when the "
    "run has no estimate. Manifests cannot set it."
)

LR = "Optimizer learning rate."
LR_ACTOR = "Optimizer learning rate for the actor (policy) network."
LR_CRITIC = "Optimizer learning rate for the critic (value) network."
CLIP_COEF = (
    "Clipping range of the policy-ratio surrogate. Caps how far one update may move the "
    "policy from the behaviour that collected the data."
)
VF_COEF = "Weight on the value-function loss in the combined objective."
GAE_LAMBDA = (
    "Generalized advantage estimation trace decay, trading bias against "
    "variance. 1.0 is plain Monte-Carlo returns."
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
    "Sampling temperature for generation. Higher values diversify rollouts. "
    "The trainer divides logits by this, so it has to be above 0."
)
MAX_OUTPUT_TOKENS = "Hard cap on tokens generated per completion."
MAX_ROW_TOKENS = (
    "Longest training row, prompt plus completion, that the trainer's memory "
    "estimate sizes for. Unset uses max_model_len. An environment with "
    "segment_prompt_tokens sets it to segment_prompt_tokens plus "
    "max_output_tokens (twice max_output_tokens with answer_continuation), "
    "capped at max_model_len; rollouts never produce a longer row."
)
MIN_OUTPUT_TOKENS = (
    "Floor on tokens generated per completion. Unset lets the model stop as "
    "soon as it emits an end-of-sequence token."
)
COSINE_LR = (
    "Learning-rate schedule stepped once per training step: linear warmup to each "
    "learning rate (actor and critic alike), then cosine decay to min_lr_ratio of it. "
    "The LLMPPO actor's schedule starts after critic_warmup_steps. On the cluster, "
    "unset or null warms up over 5% of training.max_steps and decays to 10%. "
    "For a constant learning rate, set warmup_proportion 0 and min_lr_ratio 1.0."
)
VLLM_CONFIG = "vLLM engine settings for generation."
IS_CAP = (
    "Trainer/vLLM ratio at which the sampling-mismatch metrics count a token "
    "as clipped."
)
VLLM_MAX_LOGPROB_GAP = (
    "Warn when the mean per-token log-prob gap between vLLM and the trainer "
    "exceeds this, in nats. bf16 engines usually sit near 0.01-0.03."
)
VLLM_MAX_CLIP_FRACTION = (
    "Warn when the share of tokens whose trainer/vLLM ratio reaches the "
    "importance-sampling cap exceeds this. bf16 engines usually stay under 0.01."
)
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
    "lower variance but a blunter correction. Turn-level uses a geometric-mean "
    "ratio per reply."
)
OFF_POLICY_TOKEN_MASK = (
    "(low, high) band on each token's ratio of the current policy to the "
    "rollout policy that sampled it. Tokens outside get no policy gradient "
    "(IcePop). Set null to mask nothing."
)
WHITEN_ADVANTAGES = (
    "Normalize advantages to zero mean and unit variance across the batch."
)
GROUP_SIZE = (
    "Completions sampled per prompt. The group is what the advantage is "
    "computed relative to, so this is the main GRPO setting to tune."
)
