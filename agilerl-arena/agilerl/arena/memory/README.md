# GPU memory estimation for LLM RL

Closed-form peak GPU memory for an LLM RL run from model geometry, a GPU,
and the training / generation settings.

```
arena memory estimate manifest.yaml --gpu "NVIDIA L4"
arena memory solve max_model_len --inference --gpu "NVIDIA L4" \
    --model Qwen/Qwen2.5-7B-Instruct
```

`python -m agilerl.arena.memory estimate ...` is the same command. Pass
`--config path/to/config.json` to stay offline. Install `agilerl-arena[hub]`
to resolve a Hugging Face model id (metadata only).

Exit 0 if both phases fit, 3 if either is over budget, 2 on a usage error.

A manifest that leaves `vllm_config.tensor_parallel_size` or `max_num_seqs`
unset (with `rollout_engines_per_agent: auto`) gets them from the node's GPU
count, as the run does. Pass `--num-gpus` with `--gpu` or `--device-gb`; a
resource tier brings its own count.

Training and generation are separate peaks on separate devices. Resource
selection uses the larger bar. The estimate errs high.

`ModelSpec.n_params` is the checkpoint parameter total when the caller has
it (Hub safetensors index). Unparsed geometry (towers, per-layer embeddings)
goes to `ParamCounts.unattributed` so weight bytes stay exact; activation,
KV, and LoRA terms still come from the parsed decoder.

`arena memory solve FIELD` holds the other inputs fixed and searches one
field: a linear scan up from the minimum to the first fit, then bisection
for the top of that fitting run. The scan stops at the first value training
does not fit, since training memory only grows with the field. With a
pinned engine width and no searched split, it also stops at a generation
miss the lean engine shares in both weight-sync modes. Training
uses the same underprediction buffer as `estimate`. Each candidate is sized
as `estimate` sizes it: weight-sync mode, engine width and engines.
`--inference` is a dedicated serving GPU (utilization 0.9, 8 sequences, no
trainer residual, a merged model with no LoRA slot); `--quantization fp8`
sizes vLLM online fp8 weights (`APP_VLLM_QUANTIZATION`) and needs a GPU
with native fp8 (compute capability 8.9+: L4, L40S, H100). Invertible fields:
`max_model_len`, `max_num_seqs`.

Tier search tries node counts fewest first (a tier row may list
`node_counts`; without it, `num_gpus` is the whole allocation). At each
count it tries trainer parallel layouts least parallel first and locks the
first layout that fits. A tier only takes layouts whose expert and tensor x
context groups sit on one of its nodes. `--json` reports it
in a `parallelism` block: `tp`, `ep`, `cp`, `shard_group_size`.

| module | role |
|---|---|
| `specs.py` | `config.json` → geometry; settings and device schemas |
| `manifest.py` | manifest + GPU → `RunConfig` |
| `formulas.py` | parameter counts, activations, tiles |
| `block_activations.py` | one decoder block's recompute, no-grad and stored activations |
| `fsdp_gather.py` | parameters FSDP gathers per block, side unit and prefetch window |
| `kv_cache.py` | vLLM KV cache and hybrid Mamba state pages |
| `estimator.py` | the two phase bars |
| `trainer_residency.py` | frozen base weights and allocator slack on one training GPU |
| `advice.py` | ranked setting changes when a bar is over budget |
| `solver.py` | invert one field: largest value that still fits |
| `resolve/parallel.py` | training layouts (`tp`, `ep`, `cp`, shard group) to try, least parallel first |
| `resolve/run.py` | run-level layout, split and weight-sync search |
| `resources.py` | cheapest Arena resource tier, node count and training layout that fit a manifest |
| `cli.py` | `arena memory estimate` and `arena memory solve` |
