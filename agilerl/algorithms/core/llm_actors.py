# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES

if HAS_LIGER_KERNEL:
    from liger_kernel.transformers import _apply_liger_kernel_to_instance
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.architectures.nemotron_h import register_nemotron_h_liger
from agilerl.modules.dummy import DummyEvolvable
from agilerl.protocols import (
    PeftModelProtocol,
)
from agilerl.utils.algo_utils import (
    DummyOptimizer,
    create_warmup_cosine_scheduler,
)
from agilerl.utils.evolvable_networks import (
    compile_model,
)

# Make imports visible to typechecker and import when required
if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import (
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    from agilerl.algorithms.core.llm_ops.fused_lora import (
        patch_lora_for_fused_forward,
    )
    from agilerl.algorithms.core.llm_ops.moe_lora import (
        mark_expert_wrappers_as_zero3_leaves,
        upgrade_moe_param_wrappers,
    )
    from agilerl.utils.llm_utils import (
        adapt_lora_config_for_model,
        create_model_from_name_or_path,
        log_cuda_memory_snapshot,
        offload_colocated_trainer_from_gpu,
        zero3_full_shape_views,
    )




logger = logging.getLogger(__name__)


class LLMActorMixin:
    """Actor construction and colocated vLLM init for :class:`LLMAlgorithm`."""

    def _setup_actors(
        self,
        actor_network: object | None,
        *,
        clone: bool,
    ) -> None:
        """Build the actor(s), routing through the colocated-vLLM path when enabled.

        ``clone=True`` with a pre-built PEFT ``actor_network`` reuses it (no new
        adapters). Quantized clones pass ``actor_network=None`` so the base is
        reloaded with ``BitsAndBytesConfig``; adapters must still be attached
        before the LoRA-only DeepSpeed / PEFT weight restore.
        """
        # Rebuild-from-pretrained (actor_network is None) always needs adapters,
        # including when clone=True for the quantized path.
        add_adapters = (not clone) or (actor_network is None)
        if self.use_vllm:
            self._initialize_colocated_vllm_and_actors(
                actor_network, add_adapters=add_adapters, clone=clone
            )
        else:
            self._initialize_actors(actor_network, add_adapters)

    def _initialize_colocated_vllm_and_actors(
        self,
        base_model: object | None,
        add_adapters: bool = True,
        *,
        clone: bool = False,
    ) -> None:
        """Initialize a colocated vLLM rollout engine + the HF/PEFT trainer.

        vLLM and the trainer each hold their OWN base (no zero-copy aliasing).
        Across the rollout<->train cycle vLLM round-trips its base CPU<->GPU via
        native sleep/wake (vLLM >= 0.22 restores both dense and bnb 4-bit
        losslessly) and the trainer base is offloaded to CPU during rollout
        (``use_memory_efficient_params``), so the two bases never coexist on the
        GPU. Only LoRA adapters are synced to vLLM per rollout (see
        :meth:`_move_lora_to_vllm`).

        **Init ordering is CUDA-safe.** A bitsandbytes trainer quantizes on the
        GPU during ``from_pretrained``; starting vLLM first (even slept) can
        leave the CUDA allocator in a state where the trainer's bnb device
        copies segfault. So for a fresh bnb trainer under ``sleep_mode`` the
        trainer is built first, offloaded to CPU, then vLLM starts. A dense
        trainer (or ``sleep_mode`` off, or a clone) is CUDA-safe vLLM-first.

        ``base_model`` is ``None`` for a fresh algorithm (the trainer base is
        loaded from ``pretrained_model_name_or_path``) or a fully-built,
        already-adapted actor copy for a clone (``add_adapters=False``), reused
        as-is.

        **Sleep-mode clones** do not construct a second ``LLM``: CuMem is
        process-global (one sleep-mode engine per process). The parent's engine
        is transferred in :meth:`_copy_clone_attributes` after construction.
        """
        # Sleep-mode CuMem forbids a second in-process engine. Build the
        # trainer only; ``clone()`` moves the parent's ``llm`` onto this instance.
        if clone and self.vllm_config is not None and self.vllm_config.sleep_mode:
            self.llm = None
            self._initialize_actors(base_model, add_adapters)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            return

        main = self.accelerator is None or self.accelerator.process_index == 0
        if self._trainer_should_load_before_vllm(base_model):
            if main:
                warnings.warn(
                    "colocated init: trainer-first order (bnb trainer + vLLM "
                    "sleep mode); trainer built, offloaded to CPU, then vLLM "
                    "starts. vLLM cycles its base via native sleep/wake.",
                    stacklevel=2,
                )
            self._initialize_actors(base_model, add_adapters)
            self._offload_trainer_to_cpu_for_colocated_vllm()
            self._configure_vllm()
        else:
            if main:
                warnings.warn(
                    "colocated init: vLLM-first order (dense trainer or "
                    "sleep_mode off); each side holds its own base, vLLM cycles "
                    "via native sleep/wake, trainer offloaded during rollout.",
                    stacklevel=2,
                )
            self._configure_vllm()
            self._initialize_actors(base_model, add_adapters)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _trainer_should_load_before_vllm(
        self,
        base_model: object | None,
    ) -> bool:
        """Whether the HF trainer must be built before colocated vLLM starts.

        bitsandbytes runs GPU quantization during ``from_pretrained``. Starting
        vLLM first (even in sleep mode) can leave the CUDA allocator in a state
        where subsequent device copies during the trainer's bnb load segfault,
        so a fresh quantized trainer under sleep mode is loaded first.
        """
        return (
            self.use_vllm
            and self.vllm_config is not None
            and self.vllm_config.sleep_mode
            and self.quantization_config is not None
            and base_model is None
        )

    def _offload_trainer_to_cpu_for_colocated_vllm(self) -> None:  # pragma: no cover
        """Move the HF trainer off the GPU before colocated vLLM ``LLM()`` init.

        Trainer-side bitsandbytes quantization runs on the GPU during
        ``from_pretrained`` even with ``device_map="cpu"``. Offloading after
        load keeps the trainer-first ordering (which avoids post-vLLM bnb
        segfaults) while freeing the GPU for vLLM startup (profile / compile /
        CUDA-graph capture).
        """
        if not getattr(self, "actor", None):
            warnings.warn(
                "colocated init: trainer CPU offload skipped (actor not set)",
                stacklevel=2,
            )
            return
        main = self.accelerator is None or self.accelerator.process_index == 0
        if main:
            log_cuda_memory_snapshot("colocated init: before trainer CPU offload")
        remaining_cuda_bytes = offload_colocated_trainer_from_gpu(
            self._get_unwrapped_actor()
        )
        if main:
            log_cuda_memory_snapshot("colocated init: after trainer CPU offload")
            if remaining_cuda_bytes > 0:
                warnings.warn(
                    f"colocated init: trainer still has "
                    f"{remaining_cuda_bytes / (1024**2):.2f} MiB on CUDA after "
                    "offload",
                    stacklevel=2,
                )

    def _initialize_actors(
        self,
        base_model: Any | None,  # noqa: ANN401 -- base HF model or trl-style value-head wrapper, dereferenced dynamically
        add_adapters: bool = True,
    ) -> None:
        """Initialize the actor network.

        A user-supplied :class:`~peft.PeftModel` is rejected (with
        ``add_adapters`` True): AgileRL manages its own adapters on an immutable
        base, so pass the base model instead. The clone path (``add_adapters``
        False) passes through the model unchanged.

        :param base_model: Base model
        :type base_model: PreTrainedModelProtocol
        :param add_adapters: Flag to indicate if adapters should be added to the model, defaults to True
        :type add_adapters: bool, optional
        """
        if base_model is None:
            model_config = (
                dict(self.model_config) if isinstance(self.model_config, dict) else None
            )
            # Colocated sleep mode: keep HF weights on CPU until accelerator.prepare().
            # When trainer bnb quant is enabled, _initialize_colocated_vllm_and_actors
            # loads the trainer before vLLM so bnb kernels do not run after vLLM sleep.
            if (
                self.use_vllm
                and self.vllm_config is not None
                and self.vllm_config.sleep_mode
            ):
                if model_config is None:
                    model_config = {}
                model_config.setdefault("device_map", "cpu")
            if (
                self.use_vllm
                and getattr(self, "llm", None) is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            base_model = create_model_from_name_or_path(
                self.pretrained_model_name_or_path,
                model_config=model_config,
                add_value_head=self.use_value_head,
                use_accelerator=self.accelerator is not None,
            )

        if add_adapters:
            if isinstance(base_model, PeftModelProtocol):
                msg = (
                    "actor_network: a PeftModel was passed, but AgileRL manages its "
                    "own LoRA adapters on an immutable base model. Pass the "
                    "base model instead (merge your adapters first via PEFT's "
                    "merge_and_unload() if you want to keep their effect)."
                )
                raise ValueError(msg)
            if self.use_value_head and isinstance(
                getattr(base_model, "pretrained_model", None), PeftModelProtocol
            ):
                msg = (
                    "actor_network.pretrained_model: a PeftModel was passed, but AgileRL manages its "
                    "own LoRA adapters on an immutable base model. Pass the "
                    "base model instead (merge your adapters first via PEFT's "
                    "merge_and_unload() if you want to keep their effect)."
                )
                raise ValueError(msg)

            # The PEFT target crosses HF/PEFT wrapper types; keep it untyped.
            peft_target: Any = (
                base_model.pretrained_model if self.use_value_head else base_model
            )
            # A bitsandbytes-quantized base must go through PEFT's kbit
            # preprocessing before adapters are attached: it casts layernorms
            # (and lm_head) to fp32 for training stability and registers the
            # input-embedding hook that gradient checkpointing relies on.
            # Detected via the transformers bnb load flags so a user-supplied
            # quantized actor_network is handled even without quantization_config.
            quantized_base = (
                getattr(peft_target, "is_loaded_in_8bit", False)
                or getattr(peft_target, "is_loaded_in_4bit", False)
                or getattr(peft_target, "is_quantized", False)
            )
            if quantized_base:
                peft_target = prepare_model_for_kbit_training(
                    peft_target,
                    use_gradient_checkpointing=self.gradient_checkpointing,
                    gradient_checkpointing_kwargs=self._gradient_checkpointing_kwargs(),
                )
            # Gemma 4 etc.: LoRA must target inner ``.linear`` inside *ClippableLinear.
            lora_config = adapt_lora_config_for_model(
                peft_target,
                self.lora_config,
                lora_target_scope=self.lora_target_scope,
            )
            self.lora_config = lora_config
            expert_target_parameters = getattr(lora_config, "target_parameters", None)
            if not isinstance(expert_target_parameters, (list, tuple)):
                expert_target_parameters = None
            if expert_target_parameters:
                if lora_config.lora_dropout:
                    msg = (
                        "lora_config.target_parameters (packed-experts LoRA) "
                        "requires lora_dropout=0.0: PEFT's parameter-level "
                        "LoRA cannot factor dropout out of the low-rank "
                        "product."
                    )
                    raise ValueError(msg)
                extra_adapters = [a for a in self.selected_adapters if a != "actor"]
                if extra_adapters:
                    msg = (
                        "lora_config.target_parameters (packed-experts LoRA) "
                        "supports only the 'actor' adapter — PEFT allows one "
                        "adapter per model with target_parameters, but "
                        f"selected_adapters also lists {extra_adapters}. Use "
                        "use_separate_reference_adapter=False and no value "
                        "head with expert LoRA."
                    )
                    raise ValueError(msg)
            keep_adapter_base_dtype = self.zero_stage == 3 and not quantized_base
            # PEFT reads only the targeted parameters' shapes when attaching
            # expert wrappers; under zero.Init they are partitioned
            # placeholders, so expose zero-storage full-shape views rather
            # than all-gathering the experts, whose full set can exceed
            # device memory on large MoEs.
            expert_params: list[torch.Tensor] = []
            attach_ctx: AbstractContextManager[Any] = nullcontext()
            if expert_target_parameters and self.zero_stage == 3:
                expert_params = [
                    param
                    for name, param in peft_target.named_parameters()
                    if any(name.endswith(target) for target in expert_target_parameters)
                ]
                attach_ctx = zero3_full_shape_views(expert_params)
            with attach_ctx:
                peft_target = get_peft_model(
                    peft_target,
                    lora_config,
                    adapter_name="actor",
                    autocast_adapter_dtype=not keep_adapter_base_dtype,
                )

            # Add every adapter listed in ``selected_adapters`` beyond ``actor`` as a fresh
            # LoRA initialised from ``self.lora_config``. Downstream loads can overwrite
            # these (with padding for rank-mutation) via :meth:`_load_adapter_weights`.
            for name in self.selected_adapters:
                if name == "actor":
                    continue
                if name not in peft_target.peft_config:
                    peft_target.add_adapter(
                        adapter_name=name,
                        peft_config=self.lora_config,
                        autocast_adapter_dtype=not keep_adapter_base_dtype,
                    )

            # Drop any adapters we don't own (e.g. from a user-supplied PEFT model).
            for stray in list(peft_target.peft_config.keys()):
                if stray not in self.selected_adapters:
                    warnings.warn(
                        f"Adapter '{stray}' found in the model but is not listed in "
                        f"`selected_adapters={self.selected_adapters!r}`. It will be removed "
                        "and any weights will be lost.",
                        stacklevel=2,
                    )
                    peft_target.delete_adapter(stray)

            if keep_adapter_base_dtype:
                for name, param in peft_target.named_parameters():
                    if "lora" in name and param.dtype != torch.bfloat16:
                        param.data = param.data.to(torch.bfloat16)

            if expert_target_parameters:
                # The upgrade's convention checks read the packed weights'
                # shapes; the same zero-storage views serve them under ZeRO-3.
                with zero3_full_shape_views(expert_params):
                    n_expert_lora = upgrade_moe_param_wrappers(peft_target)
                logger.info(
                    "Split expert-LoRA execution enabled on %d packed-experts modules.",
                    n_expert_lora,
                )
                if n_expert_lora and self.zero_stage == 3:
                    mark_expert_wrappers_as_zero3_leaves(peft_target)

            # Apply Liger Kernel optimizations (fused RMSNorm, RoPE, SwiGLU,
            # CrossEntropy) to the inner causal-LM *after* PEFT wrapping.
            # PeftModel is not a PreTrainedModel, so AutoLigerKernelForCausalLM
            # cannot be used at load time; instance-level patching works on the
            # already-constructed modules instead.
            if HAS_LIGER_KERNEL:
                inner_model = (
                    peft_target.base_model.model
                    if hasattr(peft_target, "base_model")
                    else peft_target
                )
                # Skip if already patched (by us or upstream) — double-patching
                # is undefined behavior.
                already_patched = getattr(
                    inner_model, "_agilerl_liger_patched", False
                ) or any(
                    type(m).__module__.startswith("liger_kernel")
                    for m in inner_model.modules()
                )
                if already_patched:
                    logger.info(
                        "Liger Kernel patches already present on %s; skipping.",
                        type(inner_model).__name__,
                    )
                try:
                    if not already_patched:
                        register_nemotron_h_liger()
                        _apply_liger_kernel_to_instance(model=inner_model)
                        inner_model._agilerl_liger_patched = True
                        logger.info(
                            "Liger Kernel instance-level patches applied to %s.",
                            type(inner_model).__name__,
                        )
                except (KeyError, AttributeError, TypeError):
                    logger.warning(
                        "Liger Kernel does not support %s; "
                        "falling back to stock HF modules.",
                        type(inner_model).__name__,
                    )

            # The actor crosses HF/PEFT/compile/DummyEvolvable representations
            # below, so build it in an untyped local before storing it.
            actor: Any
            if self.use_value_head:
                # The value-head wrapper is duck-typed (trl-style).
                vh_wrapper: Any = base_model
                vh_wrapper.pretrained_model = peft_target
                vh_wrapper.is_peft_model = True
                actor = vh_wrapper
            else:
                actor = peft_target
        else:
            actor = base_model

        self.actor = actor
        self.use_adapter("actor")
        patch_lora_for_fused_forward(actor)

        if self.torch_compiler:
            if self._uses_deepspeed:
                warnings.warn(
                    "torch_compiler is not yet compatible with DeepSpeed; "
                    "compilation skipped for this run.",
                    stacklevel=2,
                )
            else:
                if self.gradient_checkpointing:
                    warnings.warn(
                        "torch_compiler is incompatible with gradient_checkpointing; "
                        "disabling gradient checkpointing for this run.",
                        stacklevel=2,
                    )
                    self.gradient_checkpointing = False
                actor = compile_model(actor, self.torch_compiler)
                self.actor = actor

        if self.accelerator is None:
            actor = DummyEvolvable(module=actor, device=str(self.device))
            self.actor = actor

        # If an optimizer is defined in the deepspeed config, then the optimizer is part of the engine when
        # accelerator.prepare() is called. Since we are yet to wrap the model, we pass a dummy optimizer to the OptimizerWrapper.
        # In all other cases optim.Adam is used.
        optim_class = self._select_optim_class()

        self.optimizer = OptimizerWrapper(
            optim_class,
            networks=[self.actor],
            lr=self.lr,
            lr_critic=self.lr_critic,
            is_llm_optimizer=True,
            network_names=["actor"],
            lr_name="lr" if self.lr_critic is None else ("lr_actor", "lr_critic"),
        )

        if self.cosine_lr_schedule_config is not None:
            if self.optimizer.optimizer_cls != DummyOptimizer:
                sched_optimizer = self.optimizer._single_optimizer()
            else:
                # With a DummyOptimizer, DeepSpeed owns the real optimizer and
                # attaches it to the actor engine at wrap time.
                sched_optimizer = actor.optimizer  # pragma: no cover - needs a live DeepSpeed engine to attach actor.optimizer
            # ``actor.optimizer`` resolves through ``nn.Module.__getattr__`` to a
            # dynamic member; both branches yield a real optimizer at runtime.
            assert isinstance(sched_optimizer, torch.optim.Optimizer)
            self.lr_scheduler = create_warmup_cosine_scheduler(
                sched_optimizer,
                self.cosine_lr_schedule_config,
                1e-8,
                self.lr,
            )
        else:
            self.lr_scheduler = None
