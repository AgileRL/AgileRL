# FSDP2 gate experiments

Scaffolding to discharge [`../success_criteria.md`](../success_criteria.md)
without reintroducing DeepSpeed into `migration/fsdp`.

**Do not run the suite yet from this README alone** unless you intend to;
`launch.py` defaults to `--dry-run`.

## Hardware fit (this VM)

Queried on setup:

| Resource | This VM |
| --- | --- |
| GPUs | **2× Tesla T4 16 GiB** (not L4) |
| Host RAM | ~29 GiB |
| Root disk | ~10 GiB, typically **&lt;1 GiB free** |
| Artifact root | `/tmp/agilerl-fsdp-gates` (tmpfs; survives until reboot) |

### Are two GPUs enough to cycle through all gates?

| Gate band | On this box? | Notes |
| --- | --- | --- |
| Gate 0 static | Yes | CPU only |
| Gate 1 smoke + matrix | Yes | Tiny models; 2-GPU torchrun |
| Gate 2 tiny learning (L1–L5) | Yes | Debug configs / tiny actors; run **serially** |
| Gate 2a golden/near-bitwise | Yes if harness ported | Synthetic LM; 2 GPU |
| Gate 3 speed (P1) at 2 GPU | Yes | Same hardware for both backends |
| Gate 3 at 4–8 GPU (P2) | **No** | Need a larger box or waive / remote |
| Gate 4 checkpoint/clone on tiny | Yes | |
| Gate 5 MoE real model (R3) | **Unlikely** | 2×16 GiB + MoE + FSDP usually OOMs; waive or larger GPUs |
| Gate 5 adapter/fused spot checks | Yes on tiny | |
| Gate 6 prod-shaped soak | **No as true prod** | Use a **prod-proxy** (≤0.5B–1B LoRA, short steps) here; full soak elsewhere |
| vLLM colocate (P5) | **Tight / often no** | Trainer + vLLM on 16 GiB is the usual failure mode |

**Bottom line:** two GPUs on this VM are enough to **serially cycle** Gates 0–2 (tiny), Gate 3 at world_size=2, and most of Gate 4/5 spot-checks. They are **not** enough to stand in for multi-node L4 production shapes, MoE, or colocated vLLM without waivers or a bigger machine. Disk pressure is the other hard limit — keep artifacts on `/tmp`, not the repo.

If you meant two **L4** GPUs (24 GiB): same topology answer (world_size=2 is fine for cycling); more headroom for 0.5B–3B LoRA and some vLLM cases, still not a substitute for 8-GPU prod soak.

## Memory / GPU reporting

**Yes — peak-at-end alone is not enough** for Gate 3 memory profiles.

The harness records:

| Signal | Source | Why |
| --- | --- | --- |
| `allocated_bytes` / `reserved_bytes` | `torch.cuda.memory_*` (in-process, FSDP probe) | Allocator view; matches training code |
| `max_allocated_bytes` / `max_reserved_bytes` | `torch.cuda.max_memory_*` | Peak for pass/fail vs DeepSpeed ±10% |
| `nvidia_smi_used_mib` time series | External sampler in `launch.py` | Fair across DeepSpeed **and** FSDP without depending on in-process hooks |
| `step_ms`, `tokens_per_sec` | Probe step timer | Speed ≥0.90× rule |
| `loss` / eval | Probe metrics.jsonl | Learning parity |

Optional later (not required to start): CUDA memory snapshots / `torch.cuda.memory._record_memory_history` for one failing job — too heavy for every gate run.

## Layout

```text
docs/migration/gates/suite.yaml          # job → gate mapping
docs/migration/gates/accelerate_ds_2gpu.yaml  # DeepSpeed baseline launch (wt-zero3)
scripts/fsdp_gates/launch.py             # orchestrator (--dry-run default)
scripts/fsdp_gates/compare.py            # paired pass/fail
scripts/fsdp_gates/metrics.py            # schemas + samplers
scripts/fsdp_gates/run_probe.py          # FSDP / DP instrumented tiny probe
scripts/fsdp_gates/run_probe_ds.py       # DeepSpeed twin (run under wt-zero3)
```

## Usage (when ready to run)

```bash
# Print every command without executing
python scripts/fsdp_gates/launch.py --suite docs/migration/gates/suite.yaml

# Run one job id
python scripts/fsdp_gates/launch.py --ids L1 --execute

# Compare a paired artifact directory after both backends finish
python scripts/fsdp_gates/compare.py /tmp/agilerl-fsdp-gates/L1
```

Environment overrides:

- `AGILERL_GATE_ARTIFACT_ROOT` — default `/tmp/agilerl-fsdp-gates`
- `AGILERL_GATE_ZERO3` — default `/home/mike/wt-zero3`
- `AGILERL_GATE_FSDP` — default `/home/mike/wt-fsdp`
