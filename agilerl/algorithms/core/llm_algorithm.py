# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import logging
import shutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
)

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from typing_extensions import Self

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.configs import (
    LLMSetup,
    PopulationIndex,
)
from agilerl.algorithms.core.llm_init import prepare_llm_setup
from agilerl.llm_envs import RolloutHarness
from agilerl.metrics import AgentMetrics
from agilerl.typing import (
    ActionResult,
    DeviceType,
    ExperiencesT,
    LLMObsType,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    CosineLRScheduleConfig,
    DummyOptimizer,
    clone_llm,
    create_warmup_cosine_scheduler,
)

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import SequentialLR

# Make imports visible to typechecker and import when required
if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:

    from agilerl.utils.algo_utils import clone_llm
    from agilerl.utils.llm_utils import (
        align_deepspeed_lr,
        assert_no_activation_checkpointing_config,
        get_model_name_or_path,
        get_state_dict,
        is_rollout_prompt,
        resolve_llm_device,
    )
    from agilerl.utils.zero3_patches import install_model_patches



from agilerl.algorithms.core.evolvable_algorithm import EvolvableAlgorithm
from agilerl.algorithms.core.llm_actors import LLMActorMixin
from agilerl.algorithms.core.llm_checkpoint import LLMCheckpointMixin
from agilerl.algorithms.core.llm_forward import LLMForwardMixin
from agilerl.algorithms.core.llm_vllm import LLMVLLMMixin

logger = logging.getLogger(__name__)


def _kwargs_with_cloned_actor(
    kwargs: dict[str, Any],
    actor_network: object,
) -> dict[str, Any]:
    """Copy *kwargs* with *actor_network* on the grouped llm model, or as a flat key."""
    clone_kwargs = dict(kwargs)
    llm = clone_kwargs.get("llm")
    if is_dataclass(llm) and not isinstance(llm, type):
        clone_kwargs["llm"] = replace(
            llm,
            model=replace(llm.model, actor_network=actor_network),
        )
        return clone_kwargs
    clone_kwargs.pop("actor_network", None)
    clone_kwargs["actor_network"] = actor_network
    return clone_kwargs


class LLMAlgorithm(
    LLMCheckpointMixin,
    LLMActorMixin,
    LLMForwardMixin,
    LLMVLLMMixin,
    EvolvableAlgorithm[ExperiencesT],
    ABC,
    Generic[ExperiencesT],
):
    """Base object for all LLM algorithms in the AgileRL framework.

    :param llm: Grouped model, train, vLLM, generation, and device settings.
    :type llm: LLMSetup
    :param member: Population index and hyperparameter config.
    :type member: PopulationIndex | None
    """

    _allowed_adapters = frozenset({"actor", "reference", "critic"})
    # Adapter exported to vLLM for rollout — always the actor (also the
    # ``LoRARequest`` name the llm_utils helpers default to).
    _vllm_rollout_adapter = "actor"
    _mini_batch_size_default: ClassVar[Literal["micro_batch", "batch"]] = "batch"

    # Runtime handles that cross HF/PEFT/DeepSpeed/vLLM wrapper boundaries;
    # their concrete types depend on the launch configuration (Accelerate
    # prepare(), DeepSpeed engines, DummyEvolvable, colocated vLLM), so they
    # are deliberately duck-typed. ``clean_up`` resets them to None.
    actor: Any
    optimizer: Any
    llm: Any
    tp_group: Any
    lr: float

    temperature: float
    repetition_penalty: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    max_model_len: int
    max_output_tokens: int | None
    min_output_tokens: int | None

    def __init__(
        self,
        llm: LLMSetup,
        member: PopulationIndex | None = None,
    ) -> None:
        member = member or PopulationIndex()
        llm = prepare_llm_setup(llm)
        runtime = llm.runtime
        device = resolve_llm_device(runtime.accelerator, runtime.device)
        llm = replace(llm, runtime=replace(runtime, device=device))
        super().__init__(
            member.index,
            member.hp_config,
            device,
            llm.runtime.accelerator,
            llm.runtime.torch_compiler,
            llm.runtime.name,
        )
        self.mut = member.mut
        self._bind_llm_setup(llm)

    def _bind_llm_setup(self, llm: LLMSetup) -> None:
        """Copy grouped LLM settings onto instance attributes."""
        train = llm.train
        self._store_llm_model_fields(llm)
        self._configure_batch_size_per_process(
            train.batch_size,
            train.micro_batch_size_per_gpu,
            train.mini_batch_size,
        )
        self.batch_size = train.batch_size
        self.lr = align_deepspeed_lr(float(train.lr), self.accelerator)
        self.lr_critic = train.lr_critic
        self.seed = llm.model.seed
        rng_seed = self._apply_accelerator_llm_config(llm)
        self._store_llm_runtime_flags(llm)
        self._init_llm_rollout_state(llm, rng_seed)

    def _store_llm_model_fields(self, llm: LLMSetup) -> None:
        """Bind model identity, tokenizer pads, and quantization onto ``self``."""
        model = llm.model
        train = llm.train
        self.gradient_checkpointing = train.gradient_checkpointing
        self.use_liger_loss = train.use_liger_loss
        self.zero_stage = None
        self.reference_update_tracker = 0
        self.calc_position_embeddings = model.calc_position_embeddings
        self.pad_token_id = model.pad_token_id
        self.pad_token = model.pad_token
        self.pretrained_model_name_or_path = (
            model.model_name
            if model.model_name is not None
            else get_model_name_or_path(model.actor_network)
        )
        quantization_config = model.quantization_config
        if quantization_config is not None:
            # Fused logprob and Liger paths matmul lm_head outside the quantized
            # forward, so lm_head must stay unquantized.
            skip = list(
                getattr(quantization_config, "llm_int8_skip_modules", None) or []
            )
            if "lm_head" not in skip:
                skip.append("lm_head")
                quantization_config.llm_int8_skip_modules = skip
        self.quantization_config = quantization_config
        self.activation_offload = train.activation_offload
        self.use_sequence_packing = bool(train.use_sequence_packing)
        self.lora_target_scope = model.lora_target_scope
        model_config = model.model_config
        if isinstance(model_config, dict):
            model_dict = {
                k: v
                for k, v in dict(model_config).items()
                if k != "lora_target_scope"
            }
            if quantization_config is not None:
                model_dict.setdefault("quantization_config", quantization_config)
            model_config = model_dict
        elif model_config is None and quantization_config is not None:
            model_config = {"quantization_config": quantization_config}
        self.model_config = model_config

    def _apply_accelerator_llm_config(self, llm: LLMSetup) -> int:
        """Apply DeepSpeed plugin settings and return the per-rank RNG seed."""
        train = llm.train
        seed = llm.model.seed
        self.cosine_lr_schedule_config = train.cosine_lr_schedule_config
        if self.accelerator is None:
            return seed
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is not None:
            ds_config = ds_plugin.deepspeed_config
            assert_no_activation_checkpointing_config(
                ds_config,
                source="the DeepSpeed plugin config",
            )
            if train.max_grad_norm is not None:
                if self.accelerator.is_main_process:
                    warnings.warn(
                        "Argument 'max_grad_norm' will overwrite the equivalent value set for 'gradient_clipping' in the deepspeed config.",
                        stacklevel=2,
                    )
                ds_config["gradient_clipping"] = train.max_grad_norm
            if (
                self.cosine_lr_schedule_config is not None
                and self.accelerator.is_main_process
            ):
                warnings.warn(
                    "Cannot specify the optimizer in the DeepSpeed config and use AgileRL's LR scheduler. "
                    "If you want to use LR scheduling, please specify in the DeepSpeed config. "
                    "Setting LR scheduler to None.",
                    stacklevel=2,
                )
                self.cosine_lr_schedule_config = None
            self.register_mutation_hook(self._sync_deepspeed_gradient_clipping)
            self.zero_stage = ds_config["zero_optimization"]["stage"]
            if (
                self.zero_stage is not None
                and self.zero_stage > 2
                and self.accelerator.is_main_process
            ):
                warnings.warn(
                    "DeepSpeed ZeRO Stage 3 is nascent and may not work as expected, proceed with caution when using this feature.",
                    stacklevel=2,
                )
            install_model_patches(
                self.zero_stage,
                ds_config,
                model_name_or_path=self.pretrained_model_name_or_path,
                model=llm.model.actor_network,
                num_partitions=int(self.accelerator.num_processes),
            )
        if self.accelerator.num_processes > 1:
            seed = broadcast_object_list([seed], from_process=0)[0]
        seed += self.accelerator.process_index
        set_seed(seed)
        return seed

    def _store_llm_runtime_flags(self, llm: LLMSetup) -> None:
        """Bind LoRA, vLLM, wrap, and importance-sampling flags."""
        model = llm.model
        train = llm.train
        vllm = llm.vllm
        use_memory_efficient_params = vllm.use_memory_efficient_params
        self.lora_config = model.lora_config
        self.use_vllm = vllm.use_vllm
        self.vllm_config = vllm.vllm_config
        self.max_grad_norm = train.max_grad_norm
        # ZeRO-3 shards params in place; naive ``.to("cpu")`` breaks DeepSpeed
        # parameter status and leaves weights on CPU for the next forward.
        if self.zero_stage == 3 and use_memory_efficient_params:
            warnings.warn(
                "Memory efficient params is not compatible with DeepSpeed "
                "ZeRO-3; disabling trainer CPU offload for this run.",
                stacklevel=2,
            )
            use_memory_efficient_params = False
        self.use_memory_efficient_params = use_memory_efficient_params
        self.memory_efficient_params_context = (
            self._memory_efficient_params
            if use_memory_efficient_params
            else nullcontext
        )
        self.wrap = llm.runtime.wrap
        self.use_separate_reference_adapter = model.use_separate_reference_adapter
        self.cast_logprobs_to_fp32 = train.cast_logprobs_to_fp32
        if train.chunk_rows is not None and train.chunk_rows <= 0:
            msg = f"chunk_rows must be a positive int or None, got {train.chunk_rows}."
            raise ValueError(msg)
        self.chunk_rows = train.chunk_rows
        if vllm.vllm_importance_sampling_cap <= 0:
            msg = "vllm_importance_sampling_cap must be > 0."
            raise ValueError(msg)
        self.vllm_importance_sampling_correction = bool(
            vllm.vllm_importance_sampling_correction
        )
        self.vllm_importance_sampling_cap = float(vllm.vllm_importance_sampling_cap)
        self._is_correction_liger_warned = False
        self._liger_non_token_warned = False
        self._frozen_reference_warned = False

    def _init_llm_rollout_state(self, llm: LLMSetup, rng_seed: int) -> None:
        """Select adapters and initialize colocated vLLM / metrics state."""
        model = llm.model
        train = llm.train
        selected_adapters = ("actor",)
        if model.use_separate_reference_adapter:
            selected_adapters += ("reference",)
        if train.use_value_head:
            selected_adapters += ("critic",)
        self.selected_adapters = selected_adapters
        self.use_value_head = train.use_value_head
        self._uses_deepspeed = (
            self.accelerator is not None
            and getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self._vllm_awake = self.use_vllm and not (
            self.vllm_config is not None and self.vllm_config.sleep_mode
        )
        self._vllm_moved = False
        self._vllm_lora_loaded = False
        self._vllm_lora_staging_dir: Path | None = None
        self._vllm_lora_staging_dir_is_temp = True
        self._vllm_rollout_lora_request: Any | None = None
        # Colocated vLLM keeps a separate base from the HF trainer; only LoRA
        # adapters sync per rollout. The in-process engine is single-GPU.
        if self.use_vllm and self.vllm_config is not None:
            tp = getattr(self.vllm_config, "tensor_parallel_size", 1)
            if tp != 1:
                msg = (
                    "Colocated vLLM requires tensor_parallel_size==1 (the "
                    f"in-process external_launcher engine is single-GPU), got "
                    f"{tp}. Use a non-colocated / async rollout for "
                    "tensor-parallel generation (colocated TP support is planned)."
                )
                raise ValueError(msg)
        self.rng = np.random.RandomState(rng_seed)
        self.metrics = AgentMetrics()

    def preprocess_observation(self, observation: TorchObsType) -> TorchObsType:
        """Preprocess observations (dummy) for forward pass through neural network.

        :param observation: Observations of environment
        :type observation: torch.Tensor[float] or dict[str, torch.Tensor[float]]

        :return: Preprocessed observations
        :rtype: torch.Tensor[float] or dict[str, torch.Tensor[float]] or tuple[torch.Tensor[float], ...]
        """
        # Dummy pass-through: LLM observations are already batched tensors.
        return observation

    @abstractmethod
    def get_action(
        self,
        obs: LLMObsType,
        training: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> ActionResult:
        """Generate completions for each prompt in ``obs``.

        :param obs: A single prompt dict or a list of HF-style prompt dicts.
        :type obs: LLMObsType
        :param training: Whether the rollout is a training rollout.
        :type training: bool
        :return: The generated completions and their masks.
        :rtype: ActionResult
        """

    def test(
        self,
        env: RolloutHarness,
        loop: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return fitness (test) score of the llm on the test sub-set.

        :param env: Tokenized rollout episode environment (single- or multi-turn).
        :type env: RolloutHarness
        :param loop: Number of outer test iterations (episodes).
        :type loop: int
        :return: Zero-dimensional array holding the mean per-step reward,
            which is also recorded in the agent's fitness history.
        :rtype: np.ndarray
        """
        if not isinstance(env, RolloutHarness):
            msg = f"env must be a RolloutHarness; got {type(env).__name__}"
            raise TypeError(msg)
        rewards: list[float] = []
        with env.eval_mode():
            for _ in range(loop):
                prompt, _info = env.reset()
                while not env.done:
                    # ``current_prompt`` is only empty once the env is done,
                    # so inside this loop it always carries token ids.
                    if not is_rollout_prompt(prompt):
                        msg = "an active env always holds a prompt"
                        raise TypeError(msg)
                    token_ids = self.get_action(
                        [prompt],
                        training=False,
                    ).token_ids
                    prompt, reward, _terminated, _truncated, _info = env.step(
                        token_ids[0],
                    )
                    rewards.append(float(reward))
        if rewards:
            mean_fit = float(np.mean(rewards))
        else:
            warnings.warn(
                "test() collected no turns (every reset was already done, e.g. "
                "over-budget prompts); recording fitness 0.0.",
                UserWarning,
                stacklevel=2,
            )
            mean_fit = 0.0
        self.metrics.add_fitness(mean_fit)
        if self.accelerator is not None:
            # Episodes early-exit at their own turn counts, so ranks reach here
            # out of step; sync once before training resumes.
            self.accelerator.wait_for_everyone()
        return np.array(mean_fit)


    @classmethod
    def population(
        cls,
        size: int,
        accelerator: Accelerator | None = None,
        device: DeviceType = "cpu",
        resume_from_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> list[Self]:
        """Create a population of LLM algorithms.

        Builds agent 0 fully (loading the model from disk), then clones the actor
        network for agents 1..N using :func:`clone_llm`. Each agent beyond the
        first receives a fresh ``Accelerator`` instance to avoid sharing the same
        DeepSpeed distributed context.

        :param size: The size of the population.
        :type size: int
        :param accelerator: HuggingFace ``Accelerator`` instance for agent 0.
        :type accelerator: Accelerator | None
        :param device: Torch device. Defaults to ``"cpu"``.
        :type device: DeviceType
        :param resume_from_checkpoint: Path to checkpoint to resume from.
        :type resume_from_checkpoint: str | None
        :param kwargs: Additional keyword arguments to pass to the algorithm constructor.
        :type kwargs: Any
        :return: A list of LLM algorithms.
        :rtype: list[LLMAlgorithm]
        """
        agent_0 = cls(index=0, accelerator=accelerator, device=device, **kwargs)
        if resume_from_checkpoint is not None:
            agent_0.load_checkpoint(resume_from_checkpoint)
            agent_0.index = 0

        population: list[Self] = [agent_0]
        for i in range(1, size):
            agent_accelerator = Accelerator() if accelerator is not None else None
            cloned_actor = clone_llm(
                agent_0.actor,
                zero_stage=0,
                state_dict=(
                    agent_0.actor.state_dict()
                    if accelerator is None
                    else get_state_dict(agent_0.actor)
                ),
            )
            clone_kwargs = _kwargs_with_cloned_actor(kwargs, cloned_actor)
            agent = cls(
                index=i,
                accelerator=agent_accelerator,
                device=device,
                **clone_kwargs,
            )
            if resume_from_checkpoint is not None:
                agent.load_checkpoint(resume_from_checkpoint)
                agent.index = i
            population.append(agent)

        return population

    def _gradient_checkpointing_kwargs(self) -> dict[str, bool]:
        """Kwargs for HF/PEFT ``gradient_checkpointing_enable``.

        ZeRO-3 re-partitions frozen params in place during activation
        recompute; reentrant checkpointing skips the metadata check that
        rejects those empty shards under LoRA.
        """
        return {"use_reentrant": self.zero_stage == 3}

    def wrap_models(self) -> None:
        """Wrap the models in the accelerator, DeepSpeed objects must be wrapped at the same time,
        not individually.
        """
        if self.accelerator is not None:
            assert self.optimizer is not None, (
                "Optimizer is set to None, please check that the optimizer is correctly defined."
            )
            # The below is true when an optimizer is defined in the deepspeed config.
            is_dummy_optimizer = isinstance(self.optimizer.optimizer, DummyOptimizer)
            self._restore_adapter_trainability(["actor", "critic"])

            # When prepare is called on the dummy optimizer, it is returned as a DummyOptimizer object.
            # In the cases where self.optimizer.optimizer is an optim.Adam object, it is returned as DeepSpeedOptimizer
            self.actor, optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.actor,
                self.optimizer.optimizer,
                self.lr_scheduler,
            )
            # If optimizer is a dummy optimizer, then the deepspeed engine has been initialized with
            # an optimizer in the config and the optimizer is therefore part of the engine. We point the
            # optimizer attribute of the OptimizerWrapper to the active optimizer.
            self.optimizer.optimizer = (
                optimizer if not is_dummy_optimizer else self.actor.optimizer
            )

            # Again, we retrospectively set the optimizer class to the type of the optimizer as returned by prepare.
            self.optimizer.optimizer_cls = (
                type(optimizer)
                if not is_dummy_optimizer
                else type(self.actor.optimizer)
            )
            if self.gradient_checkpointing:
                self._get_unwrapped_actor().gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=self._gradient_checkpointing_kwargs(),
                )
        else:
            assert self.actor is not None, (
                "Actor is set to None, please check that the actor is defined."
            )
            self.actor = self.actor.to(self.device)
            if self.gradient_checkpointing:
                self.actor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=self._gradient_checkpointing_kwargs(),
                )

    def clean_up(self) -> None:
        """Clean up the algorithm."""
        if self.accelerator is not None:
            # Free up GPU memory occupied by parameters
            if hasattr(self.actor, "empty_partition_cache"):
                self.actor.empty_partition_cache()
            if hasattr(self.actor, "destroy"):
                self.actor.destroy()
            (
                self.actor,
                self.optimizer,
                self.lr_scheduler,
            ) = self.accelerator.free_memory(
                self.actor,
                self.optimizer,
                self.lr_scheduler,
            )
            self.accelerator.wait_for_everyone()
        else:
            (
                self.actor,
                self.optimizer,
                self.lr_scheduler,
            ) = (
                None,
                None,
                None,
            )
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
        staging_dir = getattr(self, "_vllm_lora_staging_dir", None)
        if (
            staging_dir is not None
            and getattr(self, "_vllm_lora_staging_dir_is_temp", True)
            and staging_dir.is_dir()
        ):
            # Only remove staging dirs we created; a user-configured
            # ``VLLMConfig.lora_staging_dir`` is left in place.
            shutil.rmtree(staging_dir, ignore_errors=True)
        self._vllm_lora_staging_dir = None
        self._vllm_lora_loaded = False
        self._vllm_rollout_lora_request = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        elif torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()


    @staticmethod
    def update_lr(
        optimizer: torch.optim.Optimizer,  # Deepspeed optimizers are subclasses of torch.optim.Optimizer
        lr: float | tuple[float, float],
        accelerator: Accelerator | None = None,
        scheduler_config: CosineLRScheduleConfig | None = None,
    ) -> tuple[Accelerator | None, SequentialLR | None]:
        """Update the learning rate of the optimizer.

        :param optimizer: Optimizer
        :type optimizer: Optimizer
        :param lr: Learning rate value, or actor/critic pair.
        :type lr: float | tuple[float, float]
        :param accelerator: Accelerator
        :type accelerator: Accelerator | None
        :param scheduler_config: Scheduler configuration
        :type scheduler_config: CosineLRScheduleConfig | None

        :return: Tuple of accelerator and scheduler
        :return: Accelerator
        :rtype: tuple[Accelerator | None, SequentialLR | None]
        """
        if isinstance(lr, tuple):
            lr_actor, lr_critic = lr
            lr = lr_actor
        else:
            lr_critic = None

        split = lr_critic is not None and any(
            "group" in pg for pg in optimizer.param_groups
        )
        if split:
            for param_group in optimizer.param_groups:
                g = param_group.get("group")
                if g == "critic":
                    param_group["lr"] = lr_critic
                elif g == "actor":
                    param_group["lr"] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if accelerator is None:
            scheduler = (
                create_warmup_cosine_scheduler(optimizer, scheduler_config, 1e-8, lr)
                if scheduler_config is not None
                else None
            )
            return accelerator, scheduler

        ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is None:
            scheduler = (
                create_warmup_cosine_scheduler(optimizer, scheduler_config, 1e-8, lr)
                if scheduler_config is not None
                else None
            )
            return accelerator, scheduler

        ds_config = getattr(ds_plugin, "deepspeed_config", None)
        if ds_config is None:
            return accelerator, None

        if ds_config.get("scheduler", None) is not None:
            ds_config["scheduler"]["params"]["warmup_max_lr"] = lr

        if ds_config.get("optimizer", None) is not None:
            ds_config["optimizer"]["params"]["lr"] = lr

        return accelerator, None

    def set_reference_policy(self, reference_update_tracker: int) -> None:
        """Update the reference policy when the tracker advances past the stored value.

        Base weights are immutable in AgileRL's LoRA-only training: with
        ``use_separate_reference_adapter=True`` the actor adapter is copied onto
        the ``reference`` adapter; without one the implicit reference (the base
        model with adapters disabled) cannot move, so the update request is
        acknowledged with a one-time warning and the KL anchor stays the initial
        policy.

        :param reference_update_tracker: The reference policy update tracker
        :type reference_update_tracker: int
        """
        assert reference_update_tracker >= self.reference_update_tracker, (
            "Reference policy update tracker should be greater than or equal to the current reference policy update tracker."
        )
        if reference_update_tracker > self.reference_update_tracker:
            if self.use_separate_reference_adapter:
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                self._copy_adapter_weights(
                    source_adapter="actor", target_adapter="reference"
                )
            elif not self._frozen_reference_warned:
                warnings.warn(
                    "A reference-policy update was requested but "
                    "use_separate_reference_adapter is False, so the reference "
                    "stays the initial base policy. Set "
                    "use_separate_reference_adapter=True for an updating "
                    "reference.",
                    stacklevel=2,
                    category=UserWarning,
                )
                self._frozen_reference_warned = True
            self.reference_update_tracker += 1

    def use_adapter(self, adapter_name: str) -> None:
        """Switch the active PEFT adapter, handling all side-effects.

        For "reference": switches adapter and freezes reference params (never trained).
        For all others: switches adapter and restores requires_grad=True on all
        training adapter LoRA params so that DeepSpeed ZeRO-2 gradient bucket hooks
        keep firing correctly.

        :param adapter_name: Name of the adapter to activate ("actor", "critic", "reference").
        :type adapter_name: str
        """
        peft_model = self._peft_model
        if adapter_name == "reference":
            if self.use_separate_reference_adapter:
                peft_model.set_adapter("reference")
                for name, param in self.actor.named_parameters():
                    if param is not None and "reference" in name:
                        param.requires_grad = False
            else:
                peft_model.base_model.disable_adapter_layers()
        else:
            if self.use_separate_reference_adapter:
                peft_model.set_adapter(adapter_name)
            else:
                peft_model.base_model.enable_adapter_layers()
        self._restore_adapter_trainability(["actor", "critic"])

    @contextmanager
    def select_adapter(self, adapter_name: str) -> Generator[None, None, None]:
        """Temporarily switch adapter; restores the actor adapter on exit.

        :param adapter_name: Name of the adapter to activate ("actor", "critic", "reference").
        :type adapter_name: str
        """
        self.use_adapter(adapter_name)
        try:
            yield
        finally:
            self.use_adapter("actor")
