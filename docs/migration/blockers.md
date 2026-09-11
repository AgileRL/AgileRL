# Migration blockers

## Inventory risk #3 — packed expert LoRA under FSDP2

**Status:** open (needs verification)

ZeRO-3 used `set_z3_leaf_modules` on MoE expert wrappers so packed 3D expert
weight reads stayed local during PEFT attach and split-expert LoRA execution.
That DeepSpeed-only leaf API was removed on the FSDP2 path.

**Current mitigation:** `upgrade_moe_param_wrappers` plus explicit
`gather_params(expert_params)` around PEFT attach and MoE wrapper upgrade in
`LLMAlgorithm._initialize_actors`. No FSDP2 leaf analogue is applied.

**Risk:** If per-block `fully_shard` plus explicit gathers are insufficient for
packed 3D expert reads during training or vLLM export, MoE split-LoRA runs may
fail or silently mis-read expert weights. Verify on a MoE model under
`torchrun` with `fsdp_config` before declaring parity.
