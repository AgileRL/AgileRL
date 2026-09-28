# GPU memory estimation for LLM RL

Closed-form peak GPU memory for an LLM RL run from model geometry, a GPU,
and the training / generation settings. No profiling, no weight download,
no fitted correction.

```
arena memory estimate manifest.yaml --gpu "NVIDIA L4"
```

`python -m agilerl.arena.memory estimate ...` is the same command. Pass
`--config path/to/config.json` to stay offline. Install `agilerl-arena[hub]`
to resolve a Hugging Face model id (metadata only).

Exit 0 if both phases fit, 3 if either is over budget, 2 on a usage error.

Training and generation are separate peaks on separate devices. Resource
selection uses the larger bar. The model prefers to over-predict.

`ModelSpec.n_params` is the checkpoint parameter total when the caller has
it (Hub safetensors index). Unparsed geometry (towers, per-layer embeddings)
goes to `ParamCounts.unattributed` so weight bytes stay exact; activation,
KV, and LoRA terms still come from the parsed decoder.

| module | role |
|---|---|
| `specs.py` | `config.json` → geometry; settings and device schemas |
| `manifest.py` | manifest + GPU → `RunConfig` |
| `formulas.py` | parameter counts, KV, activations, tiles |
| `estimator.py` | the two phase bars |
| `advice.py` | ranked setting changes when a bar is over budget |
| `cli.py` | `arena memory estimate` |

Against the measured corpus: training mean ~4.6% (worst 18% over),
generation mean ~2.9%.
