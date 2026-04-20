from __future__ import annotations

import argparse
from copy import deepcopy

import debugging_llm_stage_3 as s3
from config_load import load_debug_config


def main() -> None:
    """Run the stage 3 custom async WIP."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["LLMPPO", "GRPO"], required=True)
    parser.add_argument("--max-sample-steps", type=int, default=2048)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-async", action="store_true")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=16)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-sleep-mode", action="store_true")
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Override LoRA target modules for the debug run.",
    )
    parser.add_argument(
        "--qwen-0p5b-vllm",
        action="store_true",
        help="Shortcut for Qwen2.5-0.5B-Instruct with vLLM async mode.",
    )
    args = parser.parse_args()

    cfg = deepcopy(load_debug_config("grpo_grid_navigation.yaml"))
    cfg["INIT_HP"]["ALGO"] = args.algo
    if args.algo != "GRPO":
        cfg["INIT_HP"]["GROUP_SIZE"] = 1
    if args.qwen_0p5b_vllm:
        args.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        args.use_vllm = True
        args.vllm_async = True
        args.vllm_gpu_memory_utilization = max(
            args.vllm_gpu_memory_utilization,
            0.85,
        )
        if args.lora_target_modules is None:
            args.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ]

    if args.model_name is not None:
        cfg["INIT_HP"]["MODEL_NAME"] = args.model_name
    if (
        args.lora_target_modules is None
        and isinstance(args.model_name, str)
        and "qwen" in args.model_name.lower()
    ):
        args.lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    if args.lora_target_modules is not None:
        cfg.setdefault("DEBUG", {}).setdefault("lora", {})["target_modules"] = list(
            args.lora_target_modules,
        )
    if args.trust_remote_code:
        cfg["INIT_HP"]["TRUST_REMOTE_CODE"] = True
    cfg["INIT_HP"]["USE_VLLM"] = bool(args.use_vllm)
    if args.use_vllm:
        cfg["INIT_HP"]["VLLM_CONFIG"] = {
            "use_async": bool(args.vllm_async),
            "max_num_seqs": args.vllm_max_num_seqs,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "sleep_mode": bool(args.vllm_sleep_mode),
        }
    cfg["DEBUG"]["max_sample_steps"] = args.max_sample_steps
    cfg["DEBUG"]["eval_episodes"] = args.eval_episodes
    cfg["DEBUG"]["seeds"] = [args.seed]
    s3.main(cfg)


if __name__ == "__main__":
    main()
