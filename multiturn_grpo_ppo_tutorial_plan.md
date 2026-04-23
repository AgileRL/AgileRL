## Multi-turn GRPO vs PPO Tutorial Plan

### Goal
Add one practical tutorial that demonstrates how to run and compare `LLMPPO` and `GRPO` on the same multi-turn environment, with minimal setup and reproducible outputs.

### Recommended Tutorial Shape
- Use a single environment and model setup from `benchmarking/benchmarking_llm_multiturn.py` so only algorithm choice changes.
- Run two short experiments (`ALGO=LLMPPO` and `ALGO=GRPO`) using the same tokenizer/model/env and `finetune_llm_multiturn`.
- End with a compact comparison section explaining when to pick each algorithm.

### Files to Leverage
- `benchmarking/benchmarking_llm_multiturn.py`
  - Reuse the algo registry and env/tokenizer construction path; this already supports both `LLMPPO` and `GRPO`.
- `agilerl/training/train_llm.py`
  - Highlight `finetune_llm_multiturn(...)` as the shared training loop and call out that:
    - `LLMPPO`/`LLMREINFORCE` pass `turn_ids` into `learn(...)`
    - `GRPO` uses grouped rollouts (`group_size`) and batch/group divisibility constraints.
- `configs/training/llm_finetuning/ppo_llm.yaml`
- `configs/training/llm_finetuning/grpo_multiturn.yaml`
  - Use these as starter config examples and document only the key deltas.

### Tutorial Outline
1. Prereqs and install
   - `agilerl[llm]`, GEM env dependency, optional vLLM.
2. Multi-turn data flow (concept)
   - `TokenObservationWrapper` creates tokenized prompt state.
   - `SyncMultiTurnVecEnv` coordinates batch/group trajectories.
   - `collect_rollouts_llm` gathers completions, action masks, turn ids, rewards.
3. Run PPO baseline
   - Use PPO config and command examples.
   - Explain core knobs: `LR_ACTOR`, `LR_CRITIC`, `VF_COEF`, `GAE_LAMBDA`, `MAX_OUTPUT_TOKENS`.
4. Run GRPO baseline
   - Use GRPO multiturn config and command examples.
   - Explain core knobs: `GROUP_SIZE`, `BATCH_SIZE`, `LR`, `BETA`, temperature.
5. Interpret logs
   - Compare `Train/Mean Score`, `Train/Mean KL`, completion length, optional accuracy.
6. Decision guide
   - Choose PPO for stronger value-based credit assignment and controllable policy/value losses.
   - Choose GRPO for group-relative optimization and simpler objective when value-head tuning is less desirable.
7. Troubleshooting
   - Group/batch divisibility errors for GRPO.
   - Context length and `max_model_len` issues.
   - vLLM vs HF generation behavior.

### Suggested Comparison Experiment
- Keep fixed: model, env, max turns, tokenizer, reward function, seed, eval interval.
- Vary only algorithm and required hyperparameters.
- Report:
  - sample efficiency (score vs steps)
  - stability (KL and variance across checkpoints)
  - formatting/completion quality (qualitative examples)

### Deliverables
- One new docs page in LLM tutorials explaining setup, commands, and interpretation.
- Optional companion benchmark script section showing two command invocations and expected outputs.

### Why This Works
It aligns directly with the current multiturn training implementation and tests (already validating `LLMPPO`, `LLMREINFORCE`, and `GRPO` in multiturn mode), so readers learn the real production path rather than a synthetic demo.
