# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Instrumented tiny LLM probe for FSDP2 / DP gate runs (wt-fsdp)."""

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
    p.add_argument("--backend", default="fsdp2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--algo", type=str, default=None)
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--init-hp", action="append", default=[])
    p.add_argument("--save-load-roundtrip", action="store_true")
    p.add_argument("--export-pretrained", action="store_true")
    p.add_argument("--clone-check", action="store_true")
    p.add_argument("--resume-roundtrip", action="store_true")
    p.add_argument("--adapter-load-step-check", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=5)
    return p.parse_args()


def main() -> int:
    """Run tiny multiturn/reasoning-style probe with metrics."""
    args = parse_args()
    # Imports after argv parse so --help works without LLM deps
    import torch

    from agilerl.training.llm import finetune_llm_multiturn
    from agilerl.utils.distributed import get_rank, is_main_process
    from agilerl.utils.utils import create_population

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "demos/llm/debugging"))
    from llm_debug_utils import lora_config_from_dict  # type: ignore  # noqa: E402
    from tiny_model import TinyDigitTokenizer, build_tiny_actor_network  # type: ignore  # noqa: E402

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
    if args.fsdp:
        init_hp["FSDP"] = True

    max_steps = int(args.max_steps or dbg.get("max_sample_steps") or 100)
    # Keep gate probes short even if YAML asks for thousands of steps
    max_steps = min(max_steps, int(args.max_steps) if args.max_steps else max_steps)

    torch.manual_seed(args.seed)
    tokenizer = TinyDigitTokenizer()
    target_id = int(dbg.get("target_token_id", 3))
    target_token = str(target_id)
    max_ctx = int(dbg.get("max_context_length", 64))
    max_new = int(dbg.get("max_output_tokens", 1))
    init_hp.setdefault("MAX_MODEL_LEN", max_ctx)
    init_hp.setdefault("MAX_OUTPUT_TOKENS", max_new)
    if "LR" not in init_hp and "LR_ACTOR" in init_hp:
        init_hp["LR"] = init_hp["LR_ACTOR"]

    actor = build_tiny_actor_network(use_value_head=(init_hp["ALGO"] == "LLMPPO"))
    pop = create_population(
        algo=str(init_hp["ALGO"]),
        net_config=None,
        INIT_HP=init_hp,
        population_size=1,
        tokenizer=tokenizer,
        model_name=None,
        actor_network=actor,
        lora_config=lora_config_from_dict(dbg.get("lora") or {}),
    )
    agent = pop[0]
    out_dir = args.artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if is_main_process():
        writer = StepMetricsWriter(
            out_dir,
            RunSummary(
                job_id=args.job_id,
                backend=args.backend,
                seed=args.seed,
                algo=str(init_hp["ALGO"]),
                git_sha=git_sha(repo),
                worktree=str(repo),
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

    # Optional checkpoint / clone probes (rank 0 side effects)
    ckpt_dir = out_dir / "ckpt"
    if args.save_load_roundtrip and is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        agent.save_checkpoint(str(ckpt_dir / "pre.pt"))

    t0 = time.time()
    status = "ok"
    try:
        # Warmup then reset peaks so Gate 3 memory is steady-state
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
            verbose=is_main_process(),
            env_factory=env_factory,
        )
        elapsed_ms = (time.time() - step_t0) * 1000.0
        if writer is not None:
            writer.record(
                StepRecord(
                    step=max_steps,
                    loss=None,
                    step_ms=elapsed_ms / max(max_steps, 1),
                    tokens_per_sec=None,
                )
            )

        if args.export_pretrained and is_main_process():
            export_dir = out_dir / "export"
            export_dir.mkdir(parents=True, exist_ok=True)
            agent.save_checkpoint(str(export_dir))

        if args.clone_check:
            clone = agent.clone()
            del clone

        if args.save_load_roundtrip and is_main_process():
            agent.save_checkpoint(str(ckpt_dir / "post.pt"))
            agent.load_checkpoint(str(ckpt_dir / "post.pt"))

        if args.resume_roundtrip and is_main_process():
            mid = out_dir / "resume"
            mid.mkdir(parents=True, exist_ok=True)
            agent.save_checkpoint(str(mid / "mid.pt"))
            agent.load_checkpoint(str(mid / "mid.pt"))

        if args.adapter_load_step_check and is_main_process():
            # Save adapters, reload, ensure a further step does not crash
            adapt = out_dir / "adapter"
            adapt.mkdir(parents=True, exist_ok=True)
            if hasattr(agent, "save_llm_checkpoint"):
                agent.save_llm_checkpoint(str(adapt))
            finetune_llm_multiturn(
                pop=[agent],
                max_turns=1,
                init_hp=init_hp,
                max_steps=1,
                evaluation_interval=10_000,
                wb=False,
                save_elite=False,
                verbose=False,
                env_factory=env_factory,
            )
    except Exception as exc:  # noqa: BLE001 — gate probe must record failure
        status = "failed"
        if is_main_process():
            (out_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    finally:
        if writer is not None:
            writer.finalize(status=status, wall_s=time.time() - t0)

    # Non-zero ranks exit cleanly after barrier-like join in training
    if get_rank() != 0 and status == "ok":
        return 0
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
