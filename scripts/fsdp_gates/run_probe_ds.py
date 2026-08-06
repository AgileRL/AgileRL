# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DeepSpeed twin of ``run_probe.py`` — execute under wt-zero3 on PYTHONPATH."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from metrics import (  # noqa: E402
    RunSummary,
    StepMetricsWriter,
    StepRecord,
    git_sha,
    reset_cuda_peak_stats,
)


def _parse_init_hp(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        if raw.lower() in {"true", "false"}:
            out[key] = raw.lower() == "true"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-id", required=True)
    p.add_argument("--backend", default="deepspeed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--algo", type=str, default=None)
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model id or path (e.g. Qwen/Qwen2.5-0.5B). "
        "When set, the real HF model is loaded via create_population; "
        "when omitted, the tiny custom model is used.",
    )
    p.add_argument("--init-hp", action="append", default=[])
    p.add_argument("--zero3-root", type=Path, required=True)
    p.add_argument("--offload", choices=["none", "optim", "full"], default="none",
                   help="DeepSpeed CPU offload: none, optim (optimizer states), full (params+optim)")
    p.add_argument("--zero-stage", type=int, choices=[2, 3], default=3)
    p.add_argument("--warmup-steps", type=int, default=5)
    return p.parse_args()


def main() -> int:
    """Run the tiny probe on zero3's Accelerate + DeepSpeed path."""
    args = parse_args()
    zero3 = args.zero3_root.resolve()
    # Ensure zero3 agilerl wins over any other checkout on sys.path
    sys.path.insert(0, str(zero3))
    sys.path.insert(0, str(zero3 / "demos/llm/debugging"))

    import torch
    from accelerate import Accelerator

    from agilerl.training.llm import finetune_llm_multiturn
    from agilerl.utils.utils import create_population
    from llm_debug_utils import lora_config_from_dict  # type: ignore
    from tiny_model import TinyDigitTokenizer, build_tiny_actor_network  # type: ignore

    from agilerl.llm_envs import TokenObservationWrapper
    from agilerl.utils.probe_envs_llm import ConstantTargetEnv

    cfg = _load_yaml(args.config)
    dbg = dict(cfg.get("DEBUG") or {})
    init_hp = dict(cfg.get("INIT_HP") or {})
    init_hp.update(_parse_init_hp(args.init_hp))
    if args.algo:
        init_hp["ALGO"] = args.algo
    init_hp.setdefault("ALGO", "GRPO")
    init_hp.setdefault("USE_VLLM", False)
    init_hp.setdefault("SEED", args.seed)

    max_steps = int(args.max_steps or dbg.get("max_sample_steps") or 100)
    if args.max_steps is not None:
        max_steps = int(args.max_steps)

    accelerator = Accelerator()
    torch.manual_seed(args.seed + accelerator.process_index)

    target_id = int(dbg.get("target_token_id", 3))
    target_token = str(target_id)
    max_ctx = int(dbg.get("max_context_length", 64))
    max_new = int(dbg.get("max_output_tokens", 1))
    init_hp.setdefault("MAX_MODEL_LEN", max_ctx)
    init_hp.setdefault("MAX_OUTPUT_TOKENS", max_new)
    if "LR" not in init_hp and "LR_ACTOR" in init_hp:
        init_hp["LR"] = init_hp["LR_ACTOR"]

    if args.model_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        init_hp["PAD_TOKEN_ID"] = tokenizer.pad_token_id
        init_hp["PAD_TOKEN"] = tokenizer.pad_token
        actor = None
        model_name = args.model_name
        lora_dict = dict(dbg.get("lora") or {})
        # The debug config ships GPT2-style target_modules (c_attn/c_proj/c_fc)
        # which do not exist in Qwen2.5; target the attention+MLP projections
        # that every Qwen2 dense model exposes instead.
        lora_dict["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        tokenizer = TinyDigitTokenizer()
        actor = build_tiny_actor_network(use_value_head=(init_hp["ALGO"] == "LLMPPO"))
        model_name = None
        lora_dict = dbg.get("lora") or {}
    pop = create_population(
        algo=str(init_hp["ALGO"]),
        net_config=None,
        INIT_HP=init_hp,
        population_size=1,
        tokenizer=tokenizer,
        model_name=model_name,
        actor_network=actor,
        lora_config=lora_config_from_dict(lora_dict),
        accelerator=accelerator,
        device=accelerator.device,
    )
    agent = pop[0]
    out_dir = args.artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if accelerator.is_main_process:
        writer = StepMetricsWriter(
            out_dir,
            RunSummary(
                job_id=args.job_id,
                backend=args.backend,
                seed=args.seed,
                algo=str(init_hp["ALGO"]),
                git_sha=git_sha(zero3),
                worktree=str(zero3),
            ),
        )

    def env_factory() -> TokenObservationWrapper:
        return TokenObservationWrapper(
            ConstantTargetEnv(target_digit=target_token),
            tokenizer,
            1,
            tokenizer.pad_token_id,
            apply_chat_template=False,
            max_model_len=max_ctx,
            max_output_tokens=max_new,
        )

    t0 = time.time()
    status = "ok"
    try:
        warm = min(args.warmup_steps, max(0, max_steps // 5))
        if warm:
            finetune_llm_multiturn(
                pop=[agent],
                max_turns=1,
                init_hp=init_hp,
                max_steps=warm,
                evaluation_interval=10_000,
                wb=False,
                save_elite=False,
                verbose=False,
                env_factory=env_factory,
                accelerator=accelerator,
            )
            if torch.cuda.is_available():
                reset_cuda_peak_stats()

        step_t0 = time.time()
        finetune_llm_multiturn(
            pop=[agent],
            max_turns=1,
            init_hp=init_hp,
            max_steps=max_steps,
            evaluation_interval=max(1, max_steps // 4),
            wb=False,
            save_elite=False,
            verbose=accelerator.is_main_process,
            env_factory=env_factory,
            accelerator=accelerator,
        )
        elapsed_ms = (time.time() - step_t0) * 1000.0
        if writer is not None:
            writer.record(
                StepRecord(
                    step=max_steps,
                    step_ms=elapsed_ms / max(max_steps, 1),
                )
            )
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        if accelerator.is_main_process:
            (out_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    finally:
        if writer is not None:
            writer.finalize(status=status, wall_s=time.time() - t0)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
