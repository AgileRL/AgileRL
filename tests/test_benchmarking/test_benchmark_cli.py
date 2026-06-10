"""Unit tests for the shared benchmarking CLI machinery.

These exercise ``benchmarking/benchmark_cli.py`` (generic argparse core +
dataclass bridge) and ``benchmarking/benchmark_cli_llm.py`` (LLM-family
dataclasses + resolution). Both modules are deliberately torch-free, so these
tests run without the LLM stack installed.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKING_DIR = REPO_ROOT / "benchmarking"
LLM_CONFIG_DIR = REPO_ROOT / "configs" / "training" / "llm_finetuning"

# benchmark_cli lives in benchmarking/ (a scripts dir, not a package); add it to
# the path the same way running ``python benchmarking/<script>.py`` would.
if str(BENCHMARKING_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING_DIR))

import benchmark_cli  # noqa: E402
import benchmark_cli_llm  # noqa: E402

CISPO_CFG = str(LLM_CONFIG_DIR / "cispo_quant_bench.yaml")
PPO_CFG = str(LLM_CONFIG_DIR / "ppo_llm_quant_bench.yaml")
REINFORCE_CFG = str(LLM_CONFIG_DIR / "reinforce_quant_bench.yaml")
CISPO_PLAIN_CFG = str(LLM_CONFIG_DIR / "cispo.yaml")
DPO_CFG = str(LLM_CONFIG_DIR / "dpo.yaml")
SFT_CFG = str(REPO_ROOT / "configs" / "training" / "sft.yaml")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _resolve(default_config, argv):
    return benchmark_cli_llm.parse_llm_benchmark_cli(
        default_config=default_config,
        default_model=DEFAULT_MODEL,
        description="test",
        argv=argv,
    )


# Exact-value assertions use a controlled config so they don't break when the
# real benchmark YAMLs evolve; real configs are exercised by
# test_real_llm_configs_resolve (resolve-without-error only).
@pytest.fixture
def grpo_cfg(tmp_path):
    cfg = tmp_path / "grpo.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "INIT_HP": {
                    "ALGO": "CISPO",
                    "GROUP_SIZE": 4,
                    "CLIP_COEF": [0.8, 2.0],
                    "IMPORTANCE_SAMPLING_LEVEL": "token",
                    "ACTION_GRANULARITY": "turn",
                    "MAX_TURNS": 50,
                    "QUANTIZATION": "nf4",
                    "ACTIVATION_OFFLOAD": True,
                    "TARGET_MODULES": ["q_proj", "k_proj"],
                    "POP_SIZE": 1,
                },
                "MUTATION_PARAMS": {"NO_MUT": 0.1, "MIN_GROUP_SIZE": 2},
            }
        )
    )
    return str(cfg)


# --------------------------------------------------------------------------- #
# Generic core
# --------------------------------------------------------------------------- #
def test_parse_key_value_yaml_typing():
    # Key is upper-cased; value is YAML-typed. Note the YAML 1.1 quirk that a
    # float exponent needs a dot ("1.0e-4"); a bare "1e-4" stays a string, so
    # the typed dataclass flags (--lr, type=float) are preferred for scalars.
    assert benchmark_cli.parse_key_value("LR=1.0e-4") == ("LR", 1e-4)
    assert benchmark_cli.parse_key_value("EPOCHS=3") == ("EPOCHS", 3)
    assert benchmark_cli.parse_key_value("target_modules=[a, b]") == (
        "TARGET_MODULES",
        ["a", "b"],
    )
    assert benchmark_cli.parse_key_value("FLAG=true") == ("FLAG", True)


def test_parse_key_value_rejects_malformed():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_cli.parse_key_value("no-equals-sign")


def test_extract_dotted_overrides_parses_prime_rl_forms():
    remaining, overrides = benchmark_cli.extract_dotted_overrides(
        [
            "--config",
            "x.yaml",
            "--init-hp.lr",
            "1.0e-4",
            "--net-config.encoder.hidden=128",
            "--no-init-hp.use-vllm",
            "--init-hp.target-modules",
            "[q_proj, k_proj]",
            "--group-size",
            "8",
        ]
    )
    # Flat flags (no dot in the flag name) are left for argparse to handle.
    assert remaining == ["--config", "x.yaml", "--group-size", "8"]
    assert overrides == [
        ("init-hp.lr", 1e-4),
        ("net-config.encoder.hidden", 128),
        ("init-hp.use-vllm", False),
        ("init-hp.target-modules", ["q_proj", "k_proj"]),
    ]


def test_extract_dotted_overrides_bare_flag_is_true():
    remaining, overrides = benchmark_cli.extract_dotted_overrides(
        ["--init-hp.use-liger-loss", "--no-wandb"]
    )
    assert remaining == ["--no-wandb"]
    assert overrides == [("init-hp.use-liger-loss", True)]


def test_classic_core_overrides_and_wandb(tmp_path):
    cfg = tmp_path / "ppo.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "INIT_HP": {"ALGO": "PPO", "LR": 1e-3, "WANDB": True},
                "MUTATION_PARAMS": {"NO_MUT": 0.1},
                "NET_CONFIG": {"encoder_config": {"hidden": 64}},
            }
        )
    )
    parser = benchmark_cli.build_classic_parser(
        description="classic", default_config=str(cfg)
    )
    config, args = benchmark_cli.resolve_classic(
        parser,
        argv=[
            "--init-hp-override",
            "LR=0.01",
            "--net-config-override",
            "X=1",
            "--no-wandb",
        ],
    )
    assert config["INIT_HP"]["LR"] == 0.01
    assert config["INIT_HP"]["WANDB"] is False
    assert config["NET_CONFIG"]["X"] == 1


def test_classic_dotted_overrides(tmp_path):
    cfg = tmp_path / "ppo.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "INIT_HP": {"ALGO": "PPO", "LR": 1e-3},
                "MUTATION_PARAMS": {"NO_MUT": 0.1},
                "NET_CONFIG": {"encoder_config": {"hidden": 64}},
            }
        )
    )
    parser = benchmark_cli.build_classic_parser(
        description="classic", default_config=str(cfg)
    )
    config, _ = benchmark_cli.resolve_classic(
        parser,
        argv=[
            "--init-hp.lr",
            "0.02",
            "--net-config.encoder-config.hidden",
            "128",
            "--mutation-params.new-key",
            "true",
        ],
    )
    # Dotted, kebab-case paths resolve case-insensitively onto the existing
    # UPPER_SNAKE / snake config keys (prime-rl style).
    assert config["INIT_HP"]["LR"] == 0.02
    assert config["NET_CONFIG"]["encoder_config"]["hidden"] == 128
    assert config["MUTATION_PARAMS"]["new_key"] is True


def test_classic_base_config_inline(capsys):
    # Scripts with an inline config (no YAML) pass base_config; --config absent.
    base = {
        "INIT_HP": {"ALGO": "CQN", "LR": 1e-3},
        "MUTATION_PARAMS": {"NO_MUT": 0.4},
        "NET_CONFIG": {"encoder_config": {"hidden_size": [32, 32]}},
    }
    parser = benchmark_cli.build_classic_parser(
        description="inline", default_config=None
    )
    config, _ = benchmark_cli.resolve_classic(
        parser, argv=["--init-hp-override", "LR=0.5"], base_config=base
    )
    assert config["INIT_HP"]["LR"] == 0.5
    # The caller's literal must not be mutated by the override.
    assert base["INIT_HP"]["LR"] == 1e-3


def test_classic_custom_sections():
    cfg = {
        "INIT_HP": {"ALGO": "IPPO"},
        "MUTATION_PARAMS": {"NO_MUT": 0.1},
        "NET_CONFIG": {"x": 1},
        "DISTRIBUTED_TRAINING": {"ENABLED": False},
    }
    sections = (*benchmark_cli.CLASSIC_SECTIONS, "DISTRIBUTED_TRAINING")
    parser = benchmark_cli.build_classic_parser(
        description="ma", default_config=None, sections=sections
    )
    config, _ = benchmark_cli.resolve_classic(
        parser,
        argv=["--distributed-training-override", "ENABLED=true"],
        sections=sections,
        base_config=cfg,
    )
    assert config["DISTRIBUTED_TRAINING"]["ENABLED"] is True


def test_classic_print_config_exits(tmp_path, capsys):
    cfg = tmp_path / "ppo.yaml"
    cfg.write_text(yaml.safe_dump({"INIT_HP": {"ALGO": "PPO"}, "MUTATION_PARAMS": {}}))
    parser = benchmark_cli.build_classic_parser(
        description="classic", default_config=str(cfg)
    )
    with pytest.raises(SystemExit) as exc:
        benchmark_cli.resolve_classic(parser, argv=["--print-config"])
    assert exc.value.code == 0
    dumped = yaml.safe_load(capsys.readouterr().out)
    assert dumped["INIT_HP"]["ALGO"] == "PPO"


# --------------------------------------------------------------------------- #
# LLM dataclasses
# --------------------------------------------------------------------------- #
def test_select_init_hp_class():
    select = benchmark_cli_llm.select_init_hp_class
    assert select("LLMPPO") is benchmark_cli_llm.PPOInitHP
    assert select("LLMREINFORCE") is benchmark_cli_llm.ReinforceInitHP
    assert select("GRPO") is benchmark_cli_llm.GRPOInitHP
    assert select("CISPO") is benchmark_cli_llm.CISPOInitHP
    assert select("GSPO") is benchmark_cli_llm.GSPOInitHP
    assert select("DPO") is benchmark_cli_llm.DPOInitHP
    assert select("SFT") is benchmark_cli_llm.SFTInitHP
    with pytest.raises(ValueError, match="Unknown algorithm"):
        select("NOPE")


def test_dataclass_hierarchy():
    # DPO/SFT inherit from LLMInitHP but NOT the RL base; the RL family does.
    assert issubclass(benchmark_cli_llm.LLMRLInitHP, benchmark_cli_llm.LLMInitHP)
    assert issubclass(benchmark_cli_llm.GRPOInitHP, benchmark_cli_llm.LLMRLInitHP)
    assert issubclass(benchmark_cli_llm.CISPOInitHP, benchmark_cli_llm.GRPOInitHP)
    assert issubclass(benchmark_cli_llm.GSPOInitHP, benchmark_cli_llm.GRPOInitHP)
    assert issubclass(benchmark_cli_llm.DPOInitHP, benchmark_cli_llm.LLMInitHP)
    assert not issubclass(benchmark_cli_llm.DPOInitHP, benchmark_cli_llm.LLMRLInitHP)
    assert issubclass(benchmark_cli_llm.SFTInitHP, benchmark_cli_llm.LLMInitHP)
    assert not issubclass(benchmark_cli_llm.SFTInitHP, benchmark_cli_llm.LLMRLInitHP)


def test_grpo_config_loads_from_yaml(grpo_cfg):
    resolved = _resolve(grpo_cfg, [])
    hp = resolved.init_hp
    assert resolved.algo == "CISPO"
    assert hp["ALGO"] == "CISPO"
    assert hp["GROUP_SIZE"] == 4
    assert hp["CLIP_COEF"] == [0.8, 2.0]
    assert hp["IMPORTANCE_SAMPLING_LEVEL"] == "token"
    assert hp["ACTION_GRANULARITY"] == "turn"
    # MAX_TURNS is not a modelled flag (it is the rollout cap, overridden by the
    # script's --max-turns), so it passes through from the config verbatim.
    assert hp["MAX_TURNS"] == 50
    assert isinstance(hp["TARGET_MODULES"], list) and hp["TARGET_MODULES"]
    # QUANTIZATION respected from config when --trainer-quantization is unset.
    assert hp["QUANTIZATION"] == "nf4"
    assert hp["ACTIVATION_OFFLOAD"] is True
    assert resolved.mutation_params["NO_MUT"] == 0.1


def test_real_llm_configs_resolve():
    # The real benchmark YAMLs must resolve cleanly through the pipeline (values
    # may evolve, so this asserts structure, not specific numbers).
    rl_configs = {
        "GRPO": "grpo.yaml",
        "GSPO": "gspo.yaml",
        "CISPO": "cispo.yaml",
        "LLMREINFORCE": "reinforce_llm.yaml",
        "LLMPPO": "ppo_llm.yaml",
    }
    for expected_algo, name in rl_configs.items():
        path = str(LLM_CONFIG_DIR / name)
        resolved = _resolve(path, [])
        assert resolved.algo == expected_algo
        assert resolved.init_hp["ALGO"] == expected_algo
    for name, expected_algo in (("dpo.yaml", "DPO"),):
        resolved = _resolve_offline(str(LLM_CONFIG_DIR / name), [])
        assert resolved.algo == expected_algo


def test_grpo_cli_overrides():
    resolved = _resolve(
        CISPO_CFG,
        [
            "--lr",
            "1e-4",
            "--group-size",
            "32",
            "--temperature",
            "0.7",
            "--clip-coef",
            "[0.5, 1.5]",
            "--trainer-quantization",
            "int8",
            "--mut-no-mut",
            "0.5",
            "--no-use-liger-loss",
        ],
    )
    hp = resolved.init_hp
    assert hp["LR"] == 1e-4
    assert hp["GROUP_SIZE"] == 32
    assert hp["TEMPERATURE"] == 0.7
    assert hp["CLIP_COEF"] == [0.5, 1.5]
    assert hp["QUANTIZATION"] == "int8"
    assert hp["USE_LIGER_LOSS"] is False
    assert resolved.mutation_params["NO_MUT"] == 0.5


def test_init_hp_override_escape_hatch():
    resolved = _resolve(
        CISPO_CFG,
        ["--init-hp-override", "GAE_LAMBDA=0.9", "--mutation-override", "RAND_SEED=7"],
    )
    assert resolved.init_hp["GAE_LAMBDA"] == 0.9
    assert resolved.mutation_params["RAND_SEED"] == 7


def test_dotted_overrides_for_llm_sections():
    resolved = _resolve(
        CISPO_CFG,
        [
            "--init-hp.group-size",
            "16",
            "--mutation-params.min-group-size",
            "3",
        ],
    )
    assert resolved.init_hp["GROUP_SIZE"] == 16
    assert resolved.mutation_params["MIN_GROUP_SIZE"] == 3


def test_unmodeled_config_keys_pass_through():
    # cispo.yaml carries COSINE_lR_SCHEDULER, which no dataclass models.
    resolved = _resolve(CISPO_PLAIN_CFG, [])
    assert "COSINE_lR_SCHEDULER" in resolved.init_hp


def test_ppo_schema_selected_for_ppo_config():
    resolved = _resolve(PPO_CFG, ["--lr-actor", "1e-5"])
    hp = resolved.init_hp
    assert resolved.algo == "LLMPPO"
    assert hp["LR_ACTOR"] == 1e-5
    assert "LR_CRITIC" in hp
    assert "GAE_LAMBDA" in hp
    assert "VF_COEF" in hp


def test_ppo_config_rejects_grpo_only_flag():
    # --group-size belongs to the GRPO schema; PPO must not accept it.
    with pytest.raises(SystemExit):
        _resolve(PPO_CFG, ["--group-size", "8"])


def test_algo_flag_overrides_schema_and_value():
    resolved = _resolve(CISPO_CFG, ["--algo", "GSPO"])
    assert resolved.algo == "GSPO"
    assert resolved.init_hp["ALGO"] == "GSPO"


def test_reinforce_schema():
    resolved = _resolve(REINFORCE_CFG, [])
    hp = resolved.init_hp
    assert resolved.algo == "LLMREINFORCE"
    assert hp["LR"] == 5e-5
    assert "GAMMA" in hp
    assert "GROUP_SIZE" not in hp  # REINFORCE has no group schema


# --------------------------------------------------------------------------- #
# Offline LLM family (DPO / SFT)
# --------------------------------------------------------------------------- #
def _resolve_offline(default_config, argv):
    return benchmark_cli_llm.parse_offline_llm_cli(
        default_config=default_config,
        default_model="Qwen/Qwen2.5-0.5B",
        description="test",
        argv=argv,
    )


def test_dpo_offline_resolution():
    resolved = _resolve_offline(DPO_CFG, [])
    hp = resolved.init_hp
    assert resolved.algo == "DPO"
    assert hp["ALGO"] == "DPO"
    assert hp["BETA"] == 0.1
    assert hp["NLL_ALPHA"] == 1.0
    assert hp["LR"] == 5e-6
    assert hp["MAX_CONTEXT_LENGTH"] == 512
    assert hp["USE_SEPARATE_REFERENCE_ADAPTER"] is True
    # W&B + LoRA live inside INIT_HP for DPO/SFT and pass through verbatim.
    assert hp["WANDB"] is False
    assert hp["WANDB_PROJECT"] == "AgileRL"
    assert isinstance(hp["LORA"], dict) and hp["LORA"]["r"] == 16
    # No RL-only keys leaked into the DPO schema.
    assert "GROUP_SIZE" not in hp
    assert "CLIP_COEF" not in hp


def test_sft_offline_resolution():
    resolved = _resolve_offline(SFT_CFG, [])
    hp = resolved.init_hp
    assert resolved.algo == "SFT"
    assert hp["LR"] == 5e-5
    assert hp["USE_LIGER_LOSS"] is True
    assert hp["WANDB_PROJECT"] == "SFT-Benchmarking"
    assert isinstance(hp["LORA"], dict)


def test_offline_cli_overrides():
    resolved = _resolve_offline(
        DPO_CFG,
        [
            "--lr",
            "1e-6",
            "--beta",
            "0.2",
            "--model",
            "my/model",
            "--no-wandb",
            "--wandb-project",
            "proj",
            "--mut-no-mut",
            "0.3",
        ],
    )
    hp = resolved.init_hp
    assert hp["LR"] == 1e-6
    assert hp["BETA"] == 0.2
    assert hp["WANDB"] is False
    assert hp["WANDB_PROJECT"] == "proj"
    assert resolved.args.model == "my/model"
    assert resolved.mutation_params["NO_MUT"] == 0.3


def test_offline_rejects_rl_only_flag():
    # --group-size / --vllm-* belong to the RL schema, not DPO/SFT.
    with pytest.raises(SystemExit):
        _resolve_offline(DPO_CFG, ["--group-size", "8"])


def test_print_config_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        _resolve(CISPO_CFG, ["--print-config"])
    assert exc.value.code == 0
    dumped = yaml.safe_load(capsys.readouterr().out)
    assert dumped["INIT_HP"]["ALGO"] == "CISPO"
    assert "MUTATION_PARAMS" in dumped


# --------------------------------------------------------------------------- #
# vLLM kwargs
# --------------------------------------------------------------------------- #
def test_build_vllm_kwargs_defaults(grpo_cfg):
    resolved = _resolve(grpo_cfg, [])
    kwargs = resolved.build_vllm_kwargs()
    assert kwargs["sleep_mode"] is True  # POP_SIZE == 1
    assert kwargs["weight_sharing"] is False
    assert kwargs["gpu_memory_utilization"] == 0.25
    assert "quantization" not in kwargs


def _add_multiturn_like_args(parser):
    parser.add_argument("--env-name", default="game:Sudoku-v0-hard")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=10)


def test_script_specific_arguments_integrate():
    resolved = benchmark_cli_llm.parse_llm_benchmark_cli(
        default_config=CISPO_CFG,
        default_model=DEFAULT_MODEL,
        description="test",
        add_script_arguments=_add_multiturn_like_args,
        argv=["--env-name", "game:Sudoku-v0-easy", "--max-turns", "12"],
    )
    assert resolved.args.env_name == "game:Sudoku-v0-easy"
    assert resolved.args.max_turns == 12
    assert resolved.args.eval_interval == 10


def test_all_documented_quantization_rst_flags_parse():
    """Every flat flag documented in docs/llm_finetuning/quantization.rst must
    still parse — this is the contract that lets the docs stay unchanged."""
    documented = [
        "--vllm-gpu-memory-utilization",
        "0.4",
        "--vllm-max-num-seqs",
        "2",
        "--vllm-quantization",
        "bitsandbytes",
        "--vllm-dtype",
        "bfloat16",
        "--trainer-quantization",
        "nf4",
        "--trainer-activation-offload",
        "--weight-sharing",
        "--max-steps",
        "100",
        "--max-wall-seconds",
        "60",
        "--max-output-tokens",
        "128",
        "--max-model-len",
        "4096",
        "--max-grad-norm",
        "1.0",
        "--group-size",
        "8",
        "--filter-zero-adv",
        "--adv-norm",
        "mean_std",
        "--use-vllm",
        "--use-liger-loss",
        "--update-epochs",
        "1",
        "--tourn-size",
        "1",
        "--temperature",
        "1.0",
        "--target-modules",
        "q_proj",
        "k_proj",
        "--pop-size",
        "1",
        "--micro-batch-per-gpu",
        "1",
        "--lr",
        "2e-5",
        "--lora-target-scope",
        "language_model",
        "--lora-r",
        "8",
        "--lora-dropout",
        "0.0",
        "--lora-bias",
        "none",
        "--lora-alpha",
        "32",
        "--clip-coef",
        "[0.8, 2.0]",
        "--no-wandb",
        # mutation flags
        "--mut-rl-hp-mut",
        "0.6",
        "--mut-rand-seed",
        "42",
        "--mut-no-mut",
        "0.1",
        "--mut-mut-sd",
        "0.1",
        "--mut-min-lr",
        "1e-7",
        "--mut-min-group-size",
        "2",
        "--mut-min-beta",
        "0.0",
        "--mut-max-lr",
        "2.5e-5",
        "--mut-max-group-size",
        "4",
        "--mut-max-beta",
        "0.01",
        # multiturn script-specific
        "--env-name",
        "game:Sudoku-v0-hard",
        "--max-turns",
        "50",
        "--eval-interval",
        "10",
    ]
    resolved = benchmark_cli_llm.parse_llm_benchmark_cli(
        default_config=CISPO_CFG,
        default_model=DEFAULT_MODEL,
        description="test",
        add_script_arguments=_add_multiturn_like_args,
        argv=documented,
    )
    # Spot-check a few made it through to the resolved config.
    assert resolved.init_hp["GROUP_SIZE"] == 8
    assert resolved.init_hp["ADV_NORM"] == "mean_std"
    assert resolved.init_hp["CLIP_COEF"] == [0.8, 2.0]
    assert resolved.init_hp["QUANTIZATION"] == "nf4"
    assert resolved.mutation_params["MIN_GROUP_SIZE"] == 2


def test_vllm_defaults_per_script(grpo_cfg):
    # reasoning-style throughput defaults flow through to build_vllm_kwargs.
    resolved = benchmark_cli_llm.parse_llm_benchmark_cli(
        default_config=grpo_cfg,
        default_model=DEFAULT_MODEL,
        description="test",
        vllm_defaults={
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 12,
            "sleep_mode": True,
        },
        argv=[],
    )
    kwargs = resolved.build_vllm_kwargs()
    assert kwargs["gpu_memory_utilization"] == 0.8
    assert kwargs["max_num_seqs"] == 12
    assert kwargs["sleep_mode"] is True
    # An explicit flag still overrides the per-script default.
    resolved2 = benchmark_cli_llm.parse_llm_benchmark_cli(
        default_config=grpo_cfg,
        default_model=DEFAULT_MODEL,
        description="test",
        vllm_defaults={"gpu_memory_utilization": 0.8},
        argv=["--vllm-gpu-memory-utilization", "0.3"],
    )
    assert resolved2.build_vllm_kwargs()["gpu_memory_utilization"] == 0.3


def test_build_vllm_kwargs_with_rollout_flags():
    resolved = _resolve(
        CISPO_CFG,
        [
            "--vllm-quantization",
            "bitsandbytes",
            "--vllm-dtype",
            "bfloat16",
            "--weight-sharing",
            "--vllm-gpu-memory-utilization",
            "0.4",
            "--no-vllm-sleep-mode",
        ],
    )
    kwargs = resolved.build_vllm_kwargs()
    assert kwargs["quantization"] == "bitsandbytes"
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["weight_sharing"] is True
    assert kwargs["gpu_memory_utilization"] == 0.4
    assert kwargs["sleep_mode"] is False
