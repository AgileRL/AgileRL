# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Attribute a torch allocator snapshot to the estimator's components.

The sweep validates predicted *totals*: it compares one measured peak per
configuration against one predicted peak. That leaves the per-component
split unverified — the bars could each be wrong and still sum correctly.

A snapshot closes that gap, because every allocation carries the Python
stack that made it. This module rolls those stacks up into the same
component keys the estimator emits, so predicted and observed splits can be
compared line by line.

Two limits are structural, not incidental:

- **vLLM is invisible.** Its weights and KV pool come from CuMem, outside
  the torch allocator, so a snapshot covers the trainer only. The generation
  bar still needs NVML.
- **It is expensive.** Recording every allocation with stacks is far too
  heavy for a 21-point sweep, so this is for single configurations.

Usage::

    python -m agilerl.memory.profiling.harness --model <id> --snapshot t.pickle ...
    python -m agilerl.memory.profiling.snapshot t.pickle
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from agilerl.memory.specs import GiB, MiB

#: Frame markers mapped to estimator component keys, most specific first.
#: A frame matches if the marker appears in its filename or function name.
ATTRIBUTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # vLLM first: its pool is registered with the torch allocator but is not
    # trainer memory, and after sleep it is not resident at all. Leaving it
    # in would swamp every other component.
    ("vllm_engine", ("vllm/", "cumem", "gpu_worker", "model_runner")),
    # ``_apply``/``convert`` is nn.Module.to() walking parameters, which under
    # use_memory_efficient_params is how the base lands on the device for the
    # backward pass. It has to outrank the forward rules: the move happens
    # inside the training step, not at construction.
    (
        "base_weights",
        ("_apply", "convert", "from_pretrained", "load_state_dict", "move_params"),
    ),
    ("logits_workspace", ("fused_logprobs", "fused_loss", "logsumexp", "lm_head")),
    (
        "grads_optimizer",
        ("optimizer", "adamw", "accumulate_grad", "run_backward"),
    ),
    # Before the adapter rule: a forward through a LoRA-wrapped layer has
    # peft frames in its stack but is allocating activations, not adapter
    # weights. Ordering these first stops activations being booked as
    # adapters — measured at 1.24 GiB mis-attributed on one corner.
    # swiglu/rms_norm/sdpa/masking are the fused-kernel and attention sites
    # that dominate in practice; without them 304 MiB of SwiGLU intermediates
    # and 128 MiB of causal mask fell through to "unattributed".
    (
        "activations",
        (
            "checkpoint",
            "decoder_layer",
            "attention",
            "mlp",
            "swiglu",
            "rms_norm",
            "masking_utils",
            "repeat_kv",
            "embedding",
            "modeling_",
            "linear.py",
            "forward",
        ),
    ),
    ("adapters", ("lora", "peft", "tuners")),
)


#: Frames that carry no attribution meaning and must be stepped over rather
#: than matched. ``torch.compile`` emits generated modules under an inductor
#: cache directory, and vLLM points that cache inside its own tree — so a
#: compiled kernel's innermost frame is a hashed filename under ``.../vllm/``
#: and matched the engine rule, whatever Python actually made the call.
#:
#: It was not a near miss. Every logit tile the fused logprob path allocates
#: is compiled, so all of them were booked to the engine and then dropped from
#: the trainer total (which excludes engine memory by design): the measured
#: ``logits_workspace`` maximum read 7 MiB against tiles of 255.6, 162.3,
#: 127.8 and 81.1 MiB sitting in the trace with
#: ``fused_logprobs.py:_fused_logprob_chunk`` one frame below.
TRANSPARENT_FRAME_MARKERS: tuple[str, ...] = (
    "torchinductor",
    "torch_compile_cache",
    "/inductor/",
    "eval_frame.py",
    "runtime_wrappers.py",
    "_dynamo/",
    "_functorch/",
)


def _is_transparent(filename: str, name: str) -> bool:
    """True for codegen and dispatch frames that should not decide attribution."""
    haystack = f"{filename}::{name}".lower()
    if any(marker in haystack for marker in TRANSPARENT_FRAME_MARKERS):
        return True
    # Inductor names generated modules with a long hash and no path hint that
    # survives ``basename``; the cache directory is the only reliable tell, so
    # fall back to shape: a bare hashed stem invoked as ``call``.
    stem = filename.rsplit("/", 1)[-1].removesuffix(".py")
    return name == "call" and len(stem) > 24 and stem.isalnum()


def classify(frames: list[dict]) -> str:
    """Assign one allocation to a component by its deepest meaningful frame."""
    for frame in frames:
        filename, name = frame.get("filename", ""), frame.get("name", "")
        if _is_transparent(filename, name):
            continue
        haystack = f"{filename}::{name}".lower()
        for component, markers in ATTRIBUTION_RULES:
            if any(marker in haystack for marker in markers):
                return component
    return "unattributed"


def _final_totals(snapshot: dict) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            frames = block.get("frames") or []
            totals[classify(frames)] += int(block.get("size", 0))
    return dict(totals)


def summarise_final(path: Path) -> dict[str, int]:
    """Bytes per component still live when the snapshot was taken.

    Rarely what you want: by the time ``learn`` returns, activations are
    freed and the trainer's parameters are back on the host, so the
    interesting components read as zero.
    """
    with path.open("rb") as handle:
        return _final_totals(pickle.load(handle))


def summarise(path: Path) -> dict[str, int]:
    """Bytes per component at the moment of peak *trainer* allocation.

    Replays the allocation event trace rather than reading the final block
    list, because the peak is what the estimator predicts and it has long
    passed by the time the snapshot is written.

    The peak is taken over trainer bytes only, excluding vLLM's. Recording
    starts before the model exists so the trace spans the whole run, and
    vLLM's pool is registered with the torch allocator — so the *global*
    peak lands mid-generation, when ``use_memory_efficient_params`` has the
    trainer's weights parked on the host. Measured on Qwen2.5-0.5B: the
    global peak reported 0 MiB of base weights against a 942 MiB
    prediction, because at that instant there genuinely were none resident.
    """
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)

    traces = snapshot.get("device_traces") or []
    events = max(traces, key=len) if traces else []
    if not events:
        return _final_totals(snapshot)

    # Block level only. ``segment_alloc`` is the cudaMalloc of a segment and
    # ``alloc`` is a block carved from one, so counting both double-counts
    # the same bytes -- it put the trainer peak at 12.3 GiB against a 6.8 GiB
    # device delta. Blocks are also the level that carries a useful stack;
    # a segment's stack is just whoever happened to trigger the growth.
    allocs = {"alloc"}
    frees = {"free_completed"}

    live: dict[int, int] = {}
    total = 0
    peak = 0
    peak_index = 0
    for index, event in enumerate(events):
        action = event.get("action")
        size = int(event.get("size", 0))
        if action in allocs:
            if classify(event.get("frames") or []) == "vllm_engine":
                continue
            live[event.get("addr")] = size
            total += size
            if total > peak:
                peak, peak_index = total, index
        elif action in frees and event.get("addr") in live:
            total -= live.pop(event.get("addr"))

    totals: dict[str, int] = defaultdict(int)
    live_frames: dict[int, tuple[int, list]] = {}
    for event in events[: peak_index + 1]:
        action = event.get("action")
        if action in allocs:
            live_frames[event.get("addr")] = (
                int(event.get("size", 0)),
                event.get("frames") or [],
            )
        elif action in frees:
            live_frames.pop(event.get("addr"), None)
    for size, frames in live_frames.values():
        totals[classify(frames)] += size
    return dict(totals)


def timeline(
    path: Path, min_drop_bytes: int = 64 * 1024 * 1024
) -> tuple[dict[str, int], list[tuple[int, int, dict[str, int]]]]:
    """Every prominent peak in the trace, not just the global one.

    ``summarise`` answers "what is live at the maximum", which cannot say
    whether a *different* instant was nearly as large, or how big a component
    got at a moment it did not bind. A training step passes through several
    instants and the estimator maximises over models of each, so validating it
    needs all of them measured, not one.

    Returns ``(component_maxima, peaks)``. ``component_maxima`` is the largest
    each component ever gets, whenever that happens -- which is how to measure
    a transient like the logit tile that is freed before the global peak.
    ``peaks`` is ``(event_index, total_bytes, composition)`` for each local
    maximum followed by a drop of at least ``min_drop_bytes``, in order, so
    the step reads as a sequence.
    """
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    traces = snapshot.get("device_traces") or []
    events = max(traces, key=len) if traces else []

    live: dict[int, tuple[int, str]] = {}
    comp_live: dict[str, int] = defaultdict(int)
    comp_max: dict[str, int] = defaultdict(int)
    peaks: list[tuple[int, int, dict[str, int]]] = []
    total = 0
    run_max = 0
    run_max_index = 0
    run_max_comp: dict[str, int] = {}

    for index, event in enumerate(events):
        action = event.get("action")
        if action == "alloc":
            component = classify(event.get("frames") or [])
            if component == "vllm_engine":
                continue
            size = int(event.get("size", 0))
            live[event.get("addr")] = (size, component)
            total += size
            comp_live[component] += size
            comp_max[component] = max(comp_max[component], comp_live[component])
            if total > run_max:
                run_max, run_max_index = total, index
                run_max_comp = dict(comp_live)
        elif action == "free_completed" and event.get("addr") in live:
            size, component = live.pop(event.get("addr"))
            total -= size
            comp_live[component] -= size
            if run_max - total >= min_drop_bytes:
                peaks.append((run_max_index, run_max, run_max_comp))
                run_max, run_max_index = total, index
                run_max_comp = dict(comp_live)
    if run_max:
        peaks.append((run_max_index, run_max, run_max_comp))
    return dict(comp_max), peaks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agilerl.memory.profiling.snapshot", description=__doc__
    )
    parser.add_argument("snapshot", help="Pickle written by --snapshot")
    parser.add_argument(
        "--timeline",
        action="store_true",
        help=(
            "Report every prominent peak and each component's largest "
            "excursion, instead of only what is live at the global maximum"
        ),
    )
    parser.add_argument(
        "--min-drop-mib",
        type=int,
        default=64,
        help="Drop that separates two peaks in --timeline (default 64 MiB)",
    )
    args = parser.parse_args(argv)

    if args.timeline:
        maxima, peaks = timeline(
            Path(args.snapshot), min_drop_bytes=args.min_drop_mib * 1024 * 1024
        )
        print(f"{'component':<22}{'largest excursion MiB':>24}")
        for component, size in sorted(maxima.items(), key=lambda kv: -kv[1]):
            print(f"{component:<22}{size / MiB:>24.0f}")
        print(f"\n{len(peaks)} prominent peaks (drop >= {args.min_drop_mib} MiB):")
        print(f"{'#':>3}{'event':>10}{'total MiB':>11}   composition")
        for i, (index, total, comp) in enumerate(peaks, 1):
            parts = ", ".join(
                f"{k}={v / MiB:.0f}"
                for k, v in sorted(comp.items(), key=lambda kv: -kv[1])
                if v / MiB >= 1
            )
            print(f"{i:>3}{index:>10}{total / MiB:>11.0f}   {parts}")
        return 0

    totals = summarise(Path(args.snapshot))
    if not totals:
        print("No live allocations found in snapshot.", file=sys.stderr)
        return 1
    grand = sum(totals.values())
    print(f"{'component':<22}{'GiB':>8}{'share':>9}")
    for component, size in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{component:<22}{size / GiB:>8.2f}{size / grand:>8.1%}")
    print(f"{'TOTAL (torch only)':<22}{grand / GiB:>8.2f}")
    print("\nvLLM's CuMem allocations are not represented here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
