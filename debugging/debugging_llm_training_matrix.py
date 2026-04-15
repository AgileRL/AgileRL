"""Run a tiny LLM loop matrix for quick compatibility checks.

This script checks that LLM loops execute with:
1) population size 1 and no tournaments, and
2) population size >1 with tournament + mutation.

Default behavior:
- runs real tournament/mutation logic
- runs real checkpoint writes

"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any
from unittest.mock import patch

import torch

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    raise ImportError("LLM dependencies are not installed.")

from llm_debug_utils import lora_config_from_dict
from tiny_model import TinyDigitTokenizer, build_tiny_actor_network

from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training import train_llm
from agilerl.training.train_llm import (
    finetune_llm_multiturn,
    finetune_llm_preference,
    finetune_llm_reasoning,
)
from agilerl.utils.probe_envs_llm import ConditionalTargetEnv
from agilerl.utils.utils import create_population
from agilerl.wrappers.multiturn_wrappers import TokenObservationWrapper


@dataclass
class MatrixCase:
    """Single matrix case."""

    loop_name: str
    algo: str
    population_size: int
    with_tournament: bool

    @property
    def case_id(self) -> str:
        """Return a stable identifier for logs."""
        evo = "tournament" if self.with_tournament else "no_tournament"
        return f"{self.loop_name}:{self.algo}:pop{self.population_size}:{evo}"


class TinyReasoningEnv:
    """Tiny env compatible with `finetune_llm_reasoning`."""

    def __init__(
        self,
        tokenizer: TinyDigitTokenizer,
        data_batch_size_per_gpu: int,
        dataset_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_batch_size_per_gpu = data_batch_size_per_gpu
        self.name = "tiny_reasoning_debug"
        self.num_epochs = 0

        self._cursor = 0
        self._questions = [f"{idx % 5}{(idx + 1) % 5}" for idx in range(dataset_size)]
        self._answers = [
            str((idx % 5 + (idx + 1) % 5) % 5) for idx in range(dataset_size)
        ]
        self._last_prompts: dict[str, Any] | None = None

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._questions)

    def reset(self, reset_dataloaders: bool = False) -> dict[str, Any]:
        """Reset cursor (optional) and return the next prompt batch."""
        if reset_dataloaders:
            self._cursor = 0
            self.num_epochs = 0
        self._last_prompts = self._next_batch()
        return self._last_prompts

    def step(
        self, completions: list[torch.Tensor]
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Score completions and return the next prompt batch."""
        if self._last_prompts is None:
            msg = "reset() must be called before step()."
            raise RuntimeError(msg)

        prompt_len = int(self._last_prompts["input_ids"].shape[1])
        answers: list[str] = self._last_prompts["answer"]
        rewards: list[list[float]] = []

        for completion_group, answer in zip(completions, answers, strict=False):
            if completion_group.dim() == 1:
                completion_group = completion_group.unsqueeze(0)
            decoded = self.tokenizer.batch_decode(
                completion_group[:, prompt_len:],
                skip_special_tokens=True,
            )
            rewards.append([1.0 if answer in text else 0.0 for text in decoded])

        self._last_prompts = self._next_batch()
        return self._last_prompts, torch.tensor(rewards, dtype=torch.float32)

    def _next_batch(self) -> dict[str, Any]:
        """Return the next tokenized mini-batch."""
        start = self._cursor
        end = start + self.data_batch_size_per_gpu
        total = len(self._questions)

        if end <= total:
            idxs = list(range(start, end))
            self._cursor = end
            if self._cursor == total:
                self._cursor = 0
                self.num_epochs += 1
        else:
            idxs = list(range(start, total))
            remaining = end - total
            idxs.extend(range(remaining))
            self._cursor = remaining
            self.num_epochs += 1

        questions = [self._questions[i] for i in idxs]
        answers = [self._answers[i] for i in idxs]
        tokenized = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "question": questions,
            "answer": answers,
        }


class TinyPreferenceEnv:
    """Tiny env compatible with `finetune_llm_preference`."""

    def __init__(
        self,
        tokenizer: TinyDigitTokenizer,
        data_batch_size_per_gpu: int,
        dataset_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_batch_size_per_gpu = data_batch_size_per_gpu
        self.name = "tiny_preference_debug"
        self.num_epochs = 0

        self._cursor = 0
        self._prompts = [f"{idx % 5}{(idx + 2) % 5}" for idx in range(dataset_size)]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._prompts)

    def reset(self, reset_dataloaders: bool = False) -> dict[str, Any]:
        """Reset cursor (optional) and return the next preference batch."""
        if reset_dataloaders:
            self._cursor = 0
            self.num_epochs = 0
        return self._next_batch()

    def step(self) -> dict[str, Any]:
        """Return the next preference batch."""
        return self._next_batch()

    def _next_batch(self) -> dict[str, Any]:
        """Build one preference mini-batch."""
        start = self._cursor
        end = start + self.data_batch_size_per_gpu
        total = len(self._prompts)

        if end <= total:
            idxs = list(range(start, end))
            self._cursor = end
            if self._cursor == total:
                self._cursor = 0
                self.num_epochs += 1
        else:
            idxs = list(range(start, total))
            remaining = end - total
            idxs.extend(range(remaining))
            self._cursor = remaining
            self.num_epochs += 1

        prompts = [self._prompts[i] for i in idxs]
        chosen = ["1" for _ in idxs]
        rejected = ["2" for _ in idxs]
        prompt_lengths = [len(self.tokenizer.encode(prompt)) for prompt in prompts]

        chosen_ids, chosen_mask = self._encode_pairs(prompts, chosen)
        rejected_ids, rejected_mask = self._encode_pairs(prompts, rejected)

        return {
            "prompt": prompts,
            "prompt_lengths": prompt_lengths,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }

    def _encode_pairs(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt/response pairs and pad tensors."""
        encoded = [
            self.tokenizer.encode(prompt + response)
            for prompt, response in zip(prompts, responses, strict=False)
        ]
        max_len = max(len(tokens) for tokens in encoded)
        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for token_ids in encoded:
            pad_len = max_len - len(token_ids)
            padded.append(token_ids + [self.tokenizer.pad_token_id] * pad_len)
            masks.append([1] * len(token_ids) + [0] * pad_len)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )


def build_init_hp(
    algo: str,
    seed: int,
    batch_size: int,
    max_model_len: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build minimal hyperparameters for debug runs."""
    init_hp: dict[str, Any] = {
        "ALGO": algo,
        "BATCH_SIZE": batch_size,
        "UPDATE_EPOCHS": 1,
        "USE_VLLM": False,
        "GRADIENT_CHECKPOINTING": False,
        "MAX_MODEL_LEN": max_model_len,
        "MAX_OUTPUT_TOKENS": max_output_tokens,
        "SEED": seed,
    }
    if algo == "GRPO":
        init_hp |= {
            "LR": 1e-4,
            "BETA": 0.0,
            "GROUP_SIZE": 2,
            "TEMPERATURE": 0.7,
        }
    elif algo == "LLMPPO":
        init_hp |= {
            "LR_ACTOR": 1e-4,
            "LR_CRITIC": 1e-4,
            "BETA": 0.0,
            "VF_COEF": 0.2,
            "TEMPERATURE": 0.7,
        }
    elif algo == "LLMReinforce":
        init_hp |= {
            "LR": 1e-4,
            "BETA": 0.0,
            "TEMPERATURE": 0.7,
        }
    elif algo == "DPO":
        init_hp |= {"LR": 1e-4, "BETA": 0.1}
    return init_hp


def build_population(
    algo: str,
    population_size: int,
    seed: int,
    batch_size: int,
    max_model_len: int,
    max_output_tokens: int,
) -> tuple[list[Any], TinyDigitTokenizer, dict[str, Any]]:
    """Create a tiny LLM population with `create_population`."""
    tokenizer = TinyDigitTokenizer()
    init_hp = build_init_hp(
        algo=algo,
        seed=seed,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_output_tokens=max_output_tokens,
    )
    pop: list[Any] = []
    for idx in range(population_size):
        init_hp_single = dict(init_hp)
        init_hp_single["SEED"] = seed + idx
        single_pop = create_population(
            algo=algo,
            net_config=None,
            INIT_HP=init_hp_single,
            population_size=1,
            accelerator=None,
            tokenizer=tokenizer,
            model_name=None,
            actor_network=build_tiny_actor_network(use_value_head=(algo == "LLMPPO")),
            lora_config=lora_config_from_dict(
                {
                    "r": 4,
                    "lora_alpha": 8,
                    "target_modules": ["c_attn", "c_proj", "c_fc"],
                }
            ),
        )
        agent = single_pop[0]
        agent.index = idx
        pop.append(agent)

    # Tournament selection expects at least one fitness value.
    for agent in pop:
        if len(agent.fitness) == 0:
            agent.fitness.append(0.0)
        if not hasattr(agent, "micro_batch_size_per_gpu"):
            agent.micro_batch_size_per_gpu = batch_size

    return pop, tokenizer, init_hp


def build_evolution_components(
    population_size: int,
) -> tuple[TournamentSelection, Mutations]:
    """Build tournament and mutation objects for matrix runs."""
    tournament = TournamentSelection(
        tournament_size=min(2, population_size),
        elitism=True,
        population_size=population_size,
        eval_loop=1,
    )
    mutation = Mutations(
        no_mutation=1.0,
        architecture=0.0,
        new_layer_prob=0.0,
        parameters=0.0,
        activation=0.0,
        rl_hp=0.0,
        mutate_elite=False,
        rand_seed=0,
    )
    return tournament, mutation


def run_reasoning_case(
    case: MatrixCase,
    args: argparse.Namespace,
) -> None:
    """Run one reasoning case."""
    pop, tokenizer, _ = build_population(
        algo=case.algo,
        population_size=case.population_size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
    )
    env = TinyReasoningEnv(
        tokenizer=tokenizer,
        data_batch_size_per_gpu=args.batch_size,
        dataset_size=args.reasoning_dataset_size,
    )

    tournament = mutation = None
    evo_steps = None
    if case.with_tournament:
        tournament, mutation = build_evolution_components(case.population_size)
        evo_steps = args.evo_steps

    finetune_llm_reasoning(
        pop=pop,
        env=env,
        wb=False,
        save_elite=False,
        verbose=False,
        max_steps=args.reasoning_steps,
        evaluation_interval=args.evaluation_interval,
        evo_steps=evo_steps,
        tournament=tournament,
        mutation=mutation,
        accelerator=None,
        checkpoint_steps=args.checkpoint_steps,
    )


def run_preference_case(
    case: MatrixCase,
    args: argparse.Namespace,
) -> None:
    """Run one preference case."""
    pop, tokenizer, _ = build_population(
        algo=case.algo,
        population_size=case.population_size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
    )
    env = TinyPreferenceEnv(
        tokenizer=tokenizer,
        data_batch_size_per_gpu=args.batch_size,
        dataset_size=args.preference_dataset_size,
    )

    tournament = mutation = None
    evo_steps = None
    if case.with_tournament:
        tournament, mutation = build_evolution_components(case.population_size)
        evo_steps = args.evo_steps

    finetune_llm_preference(
        pop=pop,
        env=env,
        wb=False,
        save_elite=False,
        verbose=False,
        max_steps=args.preference_steps,
        evaluation_interval=args.evaluation_interval,
        evo_steps=evo_steps,
        tournament=tournament,
        mutation=mutation,
        accelerator=None,
        checkpoint_steps=args.checkpoint_steps,
    )


def run_multiturn_case(
    case: MatrixCase,
    args: argparse.Namespace,
) -> None:
    """Run one multiturn case."""
    pop, tokenizer, init_hp = build_population(
        algo=case.algo,
        population_size=case.population_size,
        seed=args.seed,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
    )
    rng = Random(args.seed)

    def env_factory() -> TokenObservationWrapper:
        """Create one tiny token observation environment."""
        return TokenObservationWrapper(
            ConditionalTargetEnv(seed=rng.randint(0, 2**31)),
            tokenizer,
            1,
            tokenizer.pad_token_id,
            apply_chat_template=False,
            max_model_len=args.max_model_len,
            max_output_tokens=args.max_output_tokens,
        )

    tournament = mutation = None
    evo_steps = None
    if case.with_tournament:
        tournament, mutation = build_evolution_components(case.population_size)
        evo_steps = args.evo_steps

    finetune_llm_multiturn(
        pop=pop,
        max_turns=1,
        env_factory=env_factory,
        init_hp=init_hp,
        max_steps=args.multiturn_steps,
        wb=False,
        save_elite=False,
        verbose=False,
        evaluation_interval=args.evaluation_interval,
        eval_fn=None,
        evo_steps=evo_steps,
        tournament=tournament,
        mutation=mutation,
        checkpoint_steps=args.checkpoint_steps,
        accelerator=None,
    )


def should_validate_checkpoint(case: MatrixCase, args: argparse.Namespace) -> bool:
    """Return whether checkpoint calls should be validated."""
    if args.skip_checkpoint_validation:
        return False
    return not case.with_tournament


def should_validate_tournament(case: MatrixCase, args: argparse.Namespace) -> bool:
    """Return whether tournament calls should be validated."""
    return case.with_tournament


def case_runner(case: MatrixCase) -> Callable[[argparse.Namespace], None]:
    """Map a matrix case to its runner."""
    if case.loop_name == "reasoning":
        return lambda args: run_reasoning_case(case, args)
    if case.loop_name == "preference":
        return lambda args: run_preference_case(case, args)
    if case.loop_name == "multiturn":
        return lambda args: run_multiturn_case(case, args)
    msg = f"Unsupported loop: {case.loop_name}"
    raise ValueError(msg)


def run_case(case: MatrixCase, args: argparse.Namespace) -> tuple[bool, str]:
    """Execute one case and return pass/fail with message."""
    checkpoint_calls = 0
    tournament_calls = 0
    run_impl = case_runner(case)
    original_save_checkpoint = train_llm.save_llm_checkpoint
    start = time.time()

    def checkpoint_wrapper(agent: Any, checkpoint_path: str | None) -> None:
        """Count checkpoint calls and optionally write a real checkpoint."""
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if args.mock_checkpoints:
            return
        target_root = Path(args.checkpoint_dir)
        target_path = str(target_root / case.case_id)
        original_save_checkpoint(agent, target_path)

    def tournament_wrapper(*call_args: Any, **kwargs: Any) -> list[Any]:
        """Count tournament calls and run real or stubbed logic."""
        nonlocal tournament_calls
        tournament_calls += 1
        if args.stub_tournament_selection:
            if "population" in kwargs:
                return kwargs["population"]
            return call_args[0]
        return original_tournament(*call_args, **kwargs)

    try:
        if not args.mock_checkpoints:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        original_tournament = train_llm.tournament_selection_and_mutation
        with (
            patch(
                "agilerl.training.train_llm.save_llm_checkpoint",
                side_effect=checkpoint_wrapper,
                autospec=True,
            ),
            patch(
                "agilerl.training.train_llm.tournament_selection_and_mutation",
                side_effect=tournament_wrapper,
                autospec=True,
            ),
        ):
            run_impl(args)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        return False, f"{type(exc).__name__}: {exc} ({elapsed:.2f}s)"

    if should_validate_checkpoint(case, args) and checkpoint_calls == 0:
        return False, "Expected checkpoint invocation but observed 0 calls."
    if should_validate_tournament(case, args) and tournament_calls == 0:
        return False, "Expected tournament invocation but observed 0 calls."

    elapsed = time.time() - start
    return (
        True,
        f"ok ({elapsed:.2f}s, ckpt_calls={checkpoint_calls}, tourn_calls={tournament_calls})",
    )


def build_cases() -> list[MatrixCase]:
    """Build the full loop/algo/population matrix."""
    scenarios = [
        {"population_size": 1, "with_tournament": False},
        {"population_size": 2, "with_tournament": True},
    ]
    cases: list[MatrixCase] = []

    for algo in ("GRPO", "LLMPPO", "LLMReinforce"):
        for scenario in scenarios:
            cases.append(MatrixCase(loop_name="reasoning", algo=algo, **scenario))

    for scenario in scenarios:
        cases.append(MatrixCase(loop_name="preference", algo="DPO", **scenario))

    for algo in ("GRPO", "LLMPPO", "LLMReinforce"):
        for scenario in scenarios:
            cases.append(MatrixCase(loop_name="multiturn", algo=algo, **scenario))

    return cases


def filter_cases_by_algo(cases: list[MatrixCase], algo: str | None) -> list[MatrixCase]:
    """Filter matrix cases to one algorithm, when requested."""
    if algo is None:
        return cases
    return [case for case in cases if case.algo == algo]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM population matrix debug checks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["GRPO", "LLMPPO", "LLMReinforce", "DPO"],
        help=("Run only one algorithm across compatible loops. Default runs all."),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32)
    parser.add_argument("--max-output-tokens", type=int, default=2)

    parser.add_argument("--reasoning-steps", type=int, default=2)
    parser.add_argument("--preference-steps", type=int, default=2)
    parser.add_argument("--multiturn-steps", type=int, default=3)

    parser.add_argument("--reasoning-dataset-size", type=int, default=12)
    parser.add_argument("--preference-dataset-size", type=int, default=12)

    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    parser.add_argument("--checkpoint-steps", type=int, default=1)
    parser.add_argument("--evo-steps", type=int, default=1)

    parser.add_argument(
        "--skip-checkpoint-validation",
        action="store_true",
        help="Disable default checkpoint invocation assertions.",
    )
    parser.add_argument(
        "--mock-checkpoints",
        action="store_true",
        help="Do not write real checkpoints; only count checkpoint invocations.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./saved_checkpoints/debug_matrix",
        help="Directory used when --real-checkpoints is enabled.",
    )
    parser.add_argument(
        "--stub-tournament-selection",
        action="store_true",
        help=("Use a lightweight tournament stub (counts calls only)."),
    )
    return parser.parse_args()


def main() -> None:
    """Run all matrix cases and exit non-zero on failure."""
    args = parse_args()
    cases = filter_cases_by_algo(build_cases(), args.algo)
    if len(cases) == 0:
        print(f"No matrix cases found for algo={args.algo}.")
        raise SystemExit(1)
    failures: list[tuple[str, str]] = []

    print(f"Running {len(cases)} LLM matrix checks...")
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx:02d}/{len(cases)}] {case.case_id} ... ", end="", flush=True)
        ok, message = run_case(case, args)
        if ok:
            print(f"PASS ({message})")
        else:
            print(f"FAIL ({message})")
            failures.append((case.case_id, message))

    if failures:
        print("\nFailures:")
        for case_id, message in failures:
            print(f"- {case_id}: {message}")
        raise SystemExit(1)

    print("\nAll matrix checks passed.")


if __name__ == "__main__":
    main()
