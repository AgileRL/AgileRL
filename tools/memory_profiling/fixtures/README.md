# Calibration fixtures

One JSON per curated model, produced by `python -m tools.memory_profiling.sweep`
on a reference device. A model is curated exactly when its profile exists here —
`agilerl.memory.curated_models()` is just a listing of this directory.

Each file is a self-contained `ModelProfile`: the model spec (geometry plus
weight variants with realised sizes), the device fingerprint it was measured
on, the framework versions, the fitted residual constants per phase, and the
raw sweep points for refits and audits. Filenames replace `/` in the model id
with `__` (`Qwen/Qwen2.5-3B-Instruct` -> `Qwen__Qwen2.5-3B-Instruct.json`).

These files ship inside the wheel and are the artefact a frontend or backend
port consumes: pure JSON in, `peak(model, device, knobs) -> {component: bytes}`
out.
