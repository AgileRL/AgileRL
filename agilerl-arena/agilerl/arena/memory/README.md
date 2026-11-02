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
for the top of that fitting run. Training uses the same underprediction
buffer as `estimate`. `--inference` is a dedicated serving GPU
(utilization 0.9, 8 sequences, no trainer residual). Invertible fields:
`max_model_len`, `max_num_seqs`.

| module | role |
|---|---|
| `specs.py` | `config.json` → geometry; settings and device schemas |
| `manifest.py` | manifest + GPU → `RunConfig` |
| `formulas.py` | parameter counts, activations, tiles |
| `fsdp_gather.py` | parameters FSDP gathers per block, side unit and prefetch window |
| `kv_cache.py` | vLLM KV cache and hybrid Mamba state pages |
| `estimator.py` | the two phase bars |
| `trainer_residency.py` | frozen base weights and allocator slack on one training GPU |
| `advice.py` | ranked setting changes when a bar is over budget |
| `solver.py` | invert one field: largest value that still fits |
| `resources.py` | cheapest Arena resource tier whose node fits a manifest |
| `cli.py` | `arena memory estimate` and `arena memory solve` |
