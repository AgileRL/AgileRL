# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Instrumented tiny LLM probe for FSDP2 / DP gate runs (wt-fsdp)."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from metrics import (  # noqa: E402
    StepRecord,
    git_sha,
    reset_cuda_peak_stats,
)
from run_logger import (  # noqa: E402
    RunArtifactCollector,
    patched_init_loggers,
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
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model id or path (e.g. Qwen/Qwen2.5-0.5B). "
        "When set, the real HF model is loaded via create_population; "
        "when omitted, the tiny custom model is used.",
    )
    p.add_argument("--fsdp", action="store_true")
    p.add_argument(
        "--optim-cpu-offload",
        action="store_true",
        help="Enable FSDPConfig.optim_cpu_offload (optimizer states on CPU)",
    )
    p.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable FSDPConfig.cpu_offload (full params+grads+states on CPU)",
    )
    p.add_argument(
        "--use-vllm",
        action="store_true",
        help="Enable colocated vLLM with sleep_mode for rollout generation",
    )
    p.add_argument(
        "--vllm-gpu-mem-util",
        type=float,
        default=0.15,
        help="vLLM gpu_memory_utilization fraction (colocated)",
    )
    p.add_argument("--init-hp", action="append", default=[])
    p.add_argument("--save-load-roundtrip", action="store_true")
    p.add_argument("--export-pretrained", action="store_true")
    p.add_argument("--clone-check", action="store_true")
    p.add_argument("--resume-roundtrip", action="store_true")
    p.add_argument("--adapter-load-step-check", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument(
        "--log-stdout",
        action="store_true",
        help="Tee stdout/stderr to artifact dir and parse loss curves from stdout",
    )
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
    from llm_debug_utils import lora_config_from_dict  # type: ignore
    from tiny_model import (  # type: ignore
        TinyDigitTokenizer,
        build_tiny_actor_network,
    )

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
    if args.use_vllm:
        init_hp["USE_VLLM"] = True
    init_hp.setdefault("SEED", args.seed)
    if args.fsdp:
        if args.cpu_offload:
            init_hp["FSDP"] = {"cpu_offload": True}
        elif args.optim_cpu_offload:
            init_hp["FSDP"] = {"optim_cpu_offload": True}
        else:
            init_hp["FSDP"] = True

    max_steps = int(args.max_steps or dbg.get("max_sample_steps") or 100)
    # Keep gate probes short even if YAML asks for thousands of steps
    max_steps = min(max_steps, int(args.max_steps) if args.max_steps else max_steps)

    torch.manual_seed(args.seed)
    target_id = int(dbg.get("target_token_id", 3))
    target_token = str(target_id)
    max_ctx = int(dbg.get("max_context_length", 64))
    max_new = int(dbg.get("max_output_tokens", 1))
    init_hp.setdefault("MAX_MODEL_LEN", max_ctx)
    init_hp.setdefault("MAX_OUTPUT_TOKENS", max_new)
    if "LR" not in init_hp and "LR_ACTOR" in init_hp:
        init_hp["LR"] = init_hp["LR_ACTOR"]

    vllm_config = None
    if args.use_vllm:
        from agilerl.utils.algo_utils import VLLMConfig

        vllm_config = VLLMConfig(
            sleep_mode=True,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            max_num_seqs=4,
        )

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
        vllm_config=vllm_config,
    )
    agent = pop[0]
    out_dir = args.artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fsdp_cfg = None
    if args.fsdp:
        from agilerl.utils.distributed import FSDPConfig

        cfg_obj = FSDPConfig(
            optim_cpu_offload=args.optim_cpu_offload,
            cpu_offload=args.cpu_offload,
        )
        fsdp_cfg = (
            asdict(cfg_obj) if hasattr(FSDPConfig, "__dataclass_fields__") else None
        )

    collector = RunArtifactCollector(
        out_dir=out_dir,
        repo=repo,
        fsdp_config=fsdp_cfg,
        job_id=args.job_id,
        backend=args.backend,
        seed=args.seed,
        algo=str(init_hp["ALGO"]),
    )
    if args.log_stdout and is_main_process():
        collector.open()
    else:
        collector._run_summary.git_sha = git_sha(repo)
        collector._run_summary.worktree = str(repo)
    writer = collector.step_writer if is_main_process() else None

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
            with patched_init_loggers(collector.jsonl_path):
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
        with patched_init_loggers(collector.jsonl_path):
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
            if is_main_process():
                collector.param_checksum.snapshot("original", agent.actor)
                collector.param_checksum.snapshot("clone", clone.actor)
            del clone

        if args.save_load_roundtrip and is_main_process():
            agent.save_checkpoint(str(ckpt_dir / "post.pt"))
            collector.param_checksum.snapshot("pre_save", agent.actor)
            agent.load_checkpoint(str(ckpt_dir / "post.pt"))
            collector.param_checksum.snapshot("post_load", agent.actor)

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
            with patched_init_loggers(collector.jsonl_path):
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
    except Exception as exc:
        status = "failed"
        if is_main_process():
            (out_dir / "error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
        raise
    finally:
        if args.log_stdout and is_main_process():
            # Feed captured stdout to loss parser as a fallback curve source
            stdout_log = out_dir / "stdout.log"
            if stdout_log.exists() and collector.loss_parser is not None:
                collector.loss_parser.feed_text(stdout_log.read_text(encoding="utf-8"))
            collector.close(status=status)
        elif writer is not None:
            writer.finalize(status=status, wall_s=time.time() - t0)

    # Non-zero ranks exit cleanly after barrier-like join in training
    if get_rank() != 0 and status == "ok":
        return 0
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
