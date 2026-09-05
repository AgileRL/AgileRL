# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import pickle
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)

import dill
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from typing_extensions import Self

from agilerl import HAS_DEEPSPEED, HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.typing import (
    DeviceType,
)
from agilerl.utils.algo_utils import (
    clone_llm,
)
from agilerl.utils.constructor_kwargs import (
    constructor_kwargs_from_obj,
)

# Make imports visible to typechecker and import when required
if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:

    from agilerl.utils.algo_utils import clone_llm
    from agilerl.utils.llm_utils import (
        adapter_checkpoint_params,
        gather_if_zero3,
    )

if TYPE_CHECKING or HAS_DEEPSPEED:
    from deepspeed.checkpoint.utils import clone_tensors_for_torch_save


from agilerl.algorithms.core.evolvable_checkpoint import (
    EvolvableCheckpointMixin,
    get_checkpoint_dict,
)
from agilerl.algorithms.core.evolvable_helpers import _is_readonly_property

logger = logging.getLogger(__name__)


class LLMCheckpointMixin:
    """Save, load, and clone for :class:`LLMAlgorithm`."""

    def save_checkpoint(
        self,
        path: str,
        lora_only: bool = True,
        save_optimizer: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save adapter weights and algorithm state to a directory.

        AgileRL never persists base-model weights when ``lora_only=True`` for
        LLM algorithms: a checkpoint is a directory containing

          * ``<adapter>/adapter_model.safetensors`` + ``adapter_config.json`` —
            one subdirectory per adapter in :attr:`selected_adapters` (always
            ``actor``, plus ``reference`` / ``critic`` when those adapters are
            configured). Written only when ``lora_only=True``.
          * ``attributes.pt`` — algorithm hyperparameters, plus (optionally)
            the actor state dict and/or optimizer state dict depending on the
            cell below. Always present.
          * ``save_checkpoint/`` — DeepSpeed ZeRO \u2265 2 sharded-checkpoint
            output. Present only when an :class:`~accelerate.Accelerator` is
            attached and ``save_optimizer=True``.

        Behaviour per cell of the ``(lora_only, save_optimizer, deepspeed)``
        grid:

        **Plain (no accelerator):**

        - ``lora_only=T, save_optimizer=T`` -- PEFT adapter dirs on disk +
          optimizer state in ``attributes.pt``
        - ``lora_only=T, save_optimizer=F`` -- PEFT adapter dirs only
        - ``lora_only=F, save_optimizer=T`` -- full actor state_dict +
          optimizer state in ``attributes.pt``
        - ``lora_only=F, save_optimizer=F`` -- full actor state_dict in
          ``attributes.pt``

        **DeepSpeed:**

        - ``lora_only=T, save_optimizer=T`` -- engine tag dir (frozen params
          excluded) + PEFT adapter dirs
        - ``lora_only=T, save_optimizer=F`` -- PEFT adapter dirs only
        - ``lora_only=F, save_optimizer=T`` -- engine tag dir (frozen params
          included)
        - ``lora_only=F, save_optimizer=F`` -- gathered (ZeRO-3 aware) actor
          state_dict injected into ``attributes.pt``

        :param path: Directory to write the checkpoint into.
        :type path: str
        :param lora_only: If ``True`` (default) only adapter weights are
            written to disk via ``save_pretrained``; the base model is shared
            across checkpoints and not serialised. If ``False``, the full
            actor state dict is persisted (into ``attributes.pt`` on the plain
            path, or into the DeepSpeed engine's tag dir / gathered dict on
            the distributed path).
        :type lora_only: bool
        :param save_optimizer: If ``True`` (default) also persist the
            optimizer and LR scheduler state so training can resume. On
            DeepSpeed ZeRO \u2265 2 this writes a sharded checkpoint into
            ``<path>/save_checkpoint``; otherwise optimizer state is included
            in ``attributes.pt``.
        :type save_optimizer: bool
        """
        if "weights_only" in kwargs:
            warnings.warn(
                "weights_only is deprecated and will be removed in a future release. Use lora_only instead.",
                stacklevel=2,
                category=DeprecationWarning,
            )
            lora_only = kwargs["weights_only"]

        Path(path).mkdir(parents=True, exist_ok=True)

        # omit_actor_info: actor state goes into attributes.pt only when we
        # want a full-model torch save on the plain (non-deepspeed) path.
        #   * lora_only=True  → adapter weights saved via PEFT on disk; no actor in attrs.pt.
        #   * deepspeed        → actor state either lives in the engine's tag dir
        #                        (save_optimizer=True) or is gathered and injected
        #                        via the manual state_dict path below (F, F).
        #   * plain + lora_only=False → full state_dict round-trips through attrs.pt.
        omit_actor_info = lora_only or self.accelerator is not None
        omit_optimizer_info = True
        state_dict = {}
        if save_optimizer:
            if self.accelerator is not None:
                # Save deepspeed checkpoint with lora_only=True
                self._save_distributed_actor(
                    path, tag="save_checkpoint", lora_only=lora_only
                )
            else:
                omit_optimizer_info = False

        if lora_only:
            model_ref = self._get_unwrapped_actor()
            with gather_if_zero3(self.zero_stage, adapter_checkpoint_params(model_ref)):
                model_ref.save_pretrained(
                    save_directory=path,
                    selected_adapters=self.selected_adapters,
                    is_main_process=self.accelerator is None
                    or self.accelerator.is_main_process,
                )

        elif self._uses_deepspeed and not save_optimizer:
            # (lora_only=False, save_optimizer=False, deepspeed): the ZeRO-3
            # shards aren't materialised in the default module loop, so gather
            # manually and inject the state_dict into attributes.pt.
            model_ref = self._get_unwrapped_actor()
            with gather_if_zero3(self.zero_stage, list(model_ref.parameters())):
                module_cls = model_ref.__class__
                state_dict = {
                    "actor_cls": module_cls,
                    "actor_init_dict": None,
                    "actor_state_dict": model_ref.state_dict(),
                    "actor_module_dict_cls": None,
                }

        # Build the checkpoint payload saved alongside adapter weights.
        checkpoint_dict = get_checkpoint_dict(
            self,
            omit_actor_info=omit_actor_info,
            omit_optimizer_info=omit_optimizer_info,
        )
        checkpoint_dict.pop("llm", None)
        checkpoint_dict.pop("tp_group", None)
        checkpoint_dict["_lora_only"] = lora_only
        if state_dict:
            checkpoint_dict["network_info"] = {}
            checkpoint_dict["network_info"]["modules"] = {}
            checkpoint_dict["network_info"]["modules"] = state_dict

        # Persist non-model attributes to ``attributes.pt``.
        # In distributed runs only the main process writes the file.
        if self.accelerator is None or self.accelerator.is_main_process:
            checkpoint_path = Path(path) / "attributes.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                checkpoint_dict,
                str(checkpoint_path),
                pickle_module=dill,
            )

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def load_weights(
        self,
        path: str,
        overwrite_reference_adapter: bool | None = None,
        overwrite_critic_adapter: bool = False,
    ) -> None:
        """Load only the LoRA adapters (and value head) from a checkpoint directory.

        :param path: Directory containing a checkpoint written by
            :meth:`save_checkpoint`.
        :type path: str
        :param overwrite_reference_adapter: See :meth:`load_checkpoint`.
        :type overwrite_reference_adapter: bool | None
        :param overwrite_critic_adapter: See :meth:`load_checkpoint`.
        :type overwrite_critic_adapter: bool
        """
        checkpoint: dict[str, Any] = torch.load(
            str(Path(path) / "attributes.pt"),
            weights_only=False,
            pickle_module=dill if self.accelerator is None else pickle,
        )
        lora_only = checkpoint.get("_lora_only", False) or checkpoint.get(
            "_weights_only", False
        )
        if lora_only:
            self._load_lora_checkpoint(
                path,
                overwrite_reference_adapter,
                overwrite_critic_adapter,
            )
        elif self._uses_deepspeed:
            self._load_full_model_checkpoint(path, checkpoint)
        else:
            # A full-model checkpoint keeps the actor's weights in ``attributes.pt``.
            self._load_torch_checkpoint(checkpoint)

    def _load_full_model_checkpoint(
        self, path: str, checkpoint: dict[str, Any]
    ) -> None:
        """Load only the actor weights of a DeepSpeed full-model checkpoint.

        :param path: Checkpoint directory written by :meth:`save_checkpoint`.
        :type path: str
        :param checkpoint: Deserialized ``attributes.pt`` payload.
        :type checkpoint: dict[str, Any]
        """
        actor_state_dict = (
            checkpoint.get("network_info", {})
            .get("modules", {})
            .get("actor_state_dict")
        )
        if actor_state_dict is None:
            self._load_distributed_actor(
                path,
                tag="save_checkpoint",
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
        else:
            model_ref = self._get_unwrapped_actor()
            params_to_gather = [
                p
                for name, p in model_ref.named_parameters()
                if name in actor_state_dict
            ]
            with gather_if_zero3(
                self.zero_stage,
                params_to_gather,
                modifier_rank=0,
            ):
                model_ref.load_state_dict(actor_state_dict)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = False,
        overwrite_reference_adapter: bool | None = None,
        overwrite_critic_adapter: bool = False,
    ) -> None:
        """Load adapter weights and algorithm state from a checkpoint directory.

        Restores full training state (adapters, optimizer, LR schedule,
        hyperparameters) to resume a run; :meth:`load_weights` takes adapters only.

        Adapter roles restored on load:

          * ``actor``     — the trained policy. Always loaded.
          * ``reference`` — the fixed policy used for KL / comparison. Loaded from
            the checkpoint's ``reference/`` adapter when it has one, so a resumed
            run keeps the anchor it was training against. When it has none, the
            checkpoint's ``actor`` is copied onto ``reference`` instead, so
            SFT -> DPO -> GRPO chains work out of the box: the stage-N actor
            becomes the stage-N+1 reference.
          * ``critic``    — optional value head. Loaded from the checkpoint's
            ``critic/`` adapter when it has one, otherwise left at its fresh LoRA
            init (all-zero ``lora_B``), i.e. a critic that starts from the base
            model. Set ``overwrite_critic_adapter`` to seed it from the actor
            instead.

        The checkpoint's LoRA config must match the live algorithm's config;
        a mismatch raises ``ValueError`` (re-create the agent with the
        checkpoint's LoRA config to load it).

        **No DeepSpeed:**

        - ``lora_only=T, load_optimizer=T`` -- PEFT adapter load + optimizer
          state from ``attributes.pt``
        - ``lora_only=T, load_optimizer=F`` -- PEFT adapter load only
        - ``lora_only=F, load_optimizer=T`` -- torch load of actor +
          optimizer from ``attributes.pt``
        - ``lora_only=F, load_optimizer=F`` -- torch load of actor only

        **DeepSpeed:**

        - ``lora_only=T, load_optimizer=T`` -- DeepSpeed engine load from
          ``<path>/save_checkpoint``
        - ``lora_only=T, load_optimizer=F`` -- PEFT adapter load
        - ``lora_only=F, load_optimizer=T`` -- DeepSpeed engine load from
          ``<path>/save_checkpoint``
        - ``lora_only=F, load_optimizer=F`` -- ``actor.load_state_dict(...)``
          from ``attributes.pt``

        When ``load_optimizer=True`` but the checkpoint contains no optimizer
        state (e.g. it was saved with ``save_optimizer=False``), a
        ``UserWarning`` is emitted and a freshly-initialised optimizer is
        used.

        :param path: Directory containing a checkpoint written by
            :meth:`save_checkpoint`.
        :type path: str
        :param load_optimizer: If ``True`` (default) also load the optimizer
            and LR scheduler state so training can resume. On DeepSpeed ZeRO
            \u2265 2 this reads a sharded checkpoint from
            ``<path>/save_checkpoint``; otherwise optimizer state is read
            from ``attributes.pt``.
        :type load_optimizer: bool
        """
        pickle_module = dill if self.accelerator is None else pickle
        checkpoint = torch.load(
            str(Path(path) / "attributes.pt"),
            weights_only=False,
            pickle_module=pickle_module,
        )

        lora_only = checkpoint.pop("_lora_only", False) or checkpoint.pop(
            "_weights_only", False
        )
        if self._uses_deepspeed:
            if load_optimizer:
                self._load_distributed_actor(path, tag="save_checkpoint")
                # DeepSpeed restore resumes actor/optimizer shards. For LoRA-only
                # checkpoints also load adapter dirs so reference/critic adapters
                # are refreshed from PEFT artifacts.
                if lora_only:
                    self._load_lora_checkpoint(
                        path,
                        overwrite_reference_adapter,
                        overwrite_critic_adapter,
                    )
            elif lora_only:
                self._load_lora_checkpoint(
                    path,
                    overwrite_reference_adapter,
                    overwrite_critic_adapter,
                )
            else:
                self._load_full_model_checkpoint(path, checkpoint)

            self._restore_checkpoint_attributes(checkpoint)

        else:
            # ``get_checkpoint_dict`` always emits a ``network_info.optimizers``
            # key — empty dict means "no optimizer state was saved". Check
            # truthiness, not key presence.
            if (
                not checkpoint.get("network_info", {}).get("optimizers")
                and load_optimizer
            ):
                warnings.warn(
                    "Optimizer state not found in checkpoint. Training will proceed using a NEW optimizer instance with random/initial default state. ",
                    stacklevel=2,
                )
            if lora_only:
                self._load_lora_checkpoint(
                    path,
                    overwrite_reference_adapter,
                    overwrite_critic_adapter,
                )
            # ``super().load_checkpoint`` restores every attribute from the
            # checkpoint, which would clobber the live ``lora_config`` /
            # ``selected_adapters``. Stash and restore, mirroring the deepspeed
            # branch's ``_restore_checkpoint_attributes`` skip-list.
            live_lora_config = self.lora_config
            live_selected_adapters = self.selected_adapters
            super().load_checkpoint(path + "/attributes.pt")
            self.lora_config = live_lora_config
            self.selected_adapters = live_selected_adapters

        if "lr_scheduler" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def _load_lora_checkpoint(
        self,
        path: str,
        overwrite_reference_adapter: bool | None = None,
        overwrite_critic_adapter: bool = False,
    ) -> None:
        """Restore LoRA adapter weights from a checkpoint directory.

        Each selected adapter is loaded from its own subdirectory; a LoRA-config
        mismatch raises ``ValueError``.

        :param path: Checkpoint directory path.
        :type path: str
        :param overwrite_reference_adapter: Seed the reference adapter from the
            actor. ``None`` (the default) decides from the checkpoint: seed when it
            carries no ``reference/`` adapter of its own, otherwise keep the one
            just loaded from disk.
        :type overwrite_reference_adapter: bool | None
        :param overwrite_critic_adapter: Seed the critic adapter from the actor.
            Defaults to ``False``: a critic absent from the checkpoint keeps its
            fresh LoRA init and so starts from the base model.
        :type overwrite_critic_adapter: bool
        """
        if overwrite_reference_adapter is None:
            overwrite_reference_adapter = not (Path(path) / "reference").exists()

        ckpt_lora_config = self._load_checkpoint_lora_config(path)
        if ckpt_lora_config is not None:
            if self.lora_config is None or self._lora_configs_equivalent(
                self.lora_config, ckpt_lora_config
            ):
                self.lora_config = ckpt_lora_config
            else:
                raise ValueError(
                    self._format_lora_config_mismatch_error(
                        self.lora_config, ckpt_lora_config
                    )
                )

        for adapter in self.selected_adapters:
            if (Path(path) / adapter).exists():
                self._load_adapter_weights(path, adapter)

        if "reference" in self.selected_adapters and overwrite_reference_adapter:
            self._copy_adapter_weights(
                source_adapter="actor", target_adapter="reference"
            )

        if "critic" in self.selected_adapters and overwrite_critic_adapter:
            self._copy_adapter_weights(source_adapter="actor", target_adapter="critic")

        # The value head (PPO's ``v_head`` Linear) is a non-LoRA module saved
        # alongside the adapters; the adapter load above never touches it.
        if self.use_value_head:
            self._restore_value_head(path)

        self._refresh_deepspeed_master_weights()

    def _refresh_deepspeed_master_weights(self) -> None:
        """Point the optimizer's fp32 master weights at the parameters on the model.

        DeepSpeed snapshots fp32 master copies of the parameters when the engine is
        built and writes those copies back over the model's shards on every
        optimizer step. Weights written into a live engine therefore survive only
        until the first step unless the master copies are refreshed to match.
        """
        optimizer = getattr(self.actor, "optimizer", None)
        refresh = getattr(optimizer, "refresh_fp32_params", None)
        if callable(refresh):
            refresh()

    def _restore_value_head(self, path: str) -> None:
        """Restore the ``v_head`` weights saved next to the LoRA adapters.

        ``AutoModelForCausalLMWithValueHead.save_pretrained`` writes the value
        head into ``pytorch_model.bin`` (the ``v_head.*`` keys of its combined
        state dict), but the LoRA-adapter load path never reads it back, so the
        value head would otherwise stay at its fresh init after
        :meth:`load_checkpoint`. Mirrors what ``from_pretrained`` does on
        construction (``post_init``), for the load-into-existing-agent path.

        :param path: Checkpoint directory written by :meth:`save_checkpoint`.
        :type path: str
        """
        wrapper = self._get_unwrapped_actor()
        loader = getattr(type(wrapper), "_maybe_load_resume_state_dict", None)
        if loader is None or not hasattr(wrapper, "post_init"):
            return
        resume_sd = loader(path)
        if resume_sd is not None and any(k.startswith("v_head.") for k in resume_sd):
            wrapper.post_init(resume_sd)

    def _restore_checkpoint_attributes(self, checkpoint: dict[str, Any]) -> None:
        """Restore algorithm attributes from payload.

        ``lora_config`` and ``selected_adapters`` are intentionally skipped \u2014 the current
        algorithm's values are authoritative, and any LoRA-shape reconciliation is done
        inside :meth:`_load_lora_checkpoint`. ``device`` is skipped for the same reason:
        the live agent owns the device its models sit on. Read-only properties are
        skipped because they are derived (e.g. GRPO's ``aux_metric_name``) and cannot
        be assigned via ``setattr``.

        :param checkpoint: Loaded attribute payload.
        :type checkpoint: dict[str, Any]
        :param checkpoint_type: The checkpoint type.
        :type checkpoint_type: Literal["peft", "deepspeed", "torch"]
        """
        skip_attrs = {
            "lr_scheduler",
            "lora_config",
            "selected_adapters",
            "device",
        }
        for attr, value in checkpoint.items():
            if attr in skip_attrs:
                continue
            if _is_readonly_property(self, attr):
                continue
            setattr(self, attr, value)

    def _rebuild_optimizer_after_load(self) -> None:
        """Recreate the optimizer wrapper after distributed checkpoint load.

        Distributed load restores model weights/engine state first, then this
        method rebuilds the wrapper metadata used by training paths.
        """
        self.optimizer = OptimizerWrapper(
            optimizer_cls=self._select_optim_class(),
            networks=[self.actor],
            network_names=["actor"],
            lr=self.lr,
            lr_critic=self.lr_critic,
            is_llm_optimizer=True,
            lr_name="lr" if self.lr_critic is None else ("lr_actor", "lr_critic"),
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: DeviceType = "cpu",
        accelerator: Accelerator | None = None,
    ) -> NoReturn:
        msg = (
            "The load class method is not supported for this algorithm class. "
            "To load a saved LLM, please load the model as follows, and then re-instantiate the GRPO/DPO/SFT "
            "class, using the pre-trained model.\n\n"
            "base_model = AutoModelForCausalLM.from_pretrained(\n"
            '    "Qwen/Qwen2.5-3B",\n'
            "    torch_dtype=torch.bfloat16,\n"
            '    device_map="auto"\n'
            ")\n"
            'tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")\n'
            "model = PeftModelProtocol.from_pretrained(base_model, path)\n"
            "where 'path' is the directory containing the saved LoRA adapter weights."
        )
        raise NotImplementedError(
            msg,
        )

    def clone(self, index: int | None = None, wrap: bool = True) -> Self:
        """Create a clone of the algorithm.

        :param index: The index of the clone, defaults to None
        :type index: int | None, optional
        :param wrap: If True, wrap the models in the clone with the accelerator, defaults to False
        :type wrap: bool, optional

        :return: A clone of the algorithm
        :rtype: EvolvableAlgorithm
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = self._resolve_clone_work_dir(temp_dir)
            self._save_clone_distributed_actor_state(work_dir)
            clone = self._create_clone_instance()
            clone.mutation_hook()
            clone = self._copy_clone_attributes(clone)
            self._restore_clone_optimizer_and_scheduler(clone)

            # Set the index
            if index is not None:
                clone.index = index

            clone.wrap_models()
            self._load_clone_distributed_actor_state(clone, work_dir)

            return clone

    def _resolve_clone_work_dir(self, temp_dir: str) -> str:
        """Resolve a clone workspace path visible to all ranks.

        :param temp_dir: Local temporary directory path.
        :type temp_dir: str
        :return: Shared working directory path for clone artifacts.
        :rtype: str
        """
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            return broadcast_object_list([temp_dir], from_process=0)[0]
        return temp_dir

    def _uses_quantized_clone_rebuild(self) -> bool:
        """Whether clone should rebuild the actor from pretrained + BitsAndBytes.

        QLoRA / bitsandbytes bases store packed ``Params4bit`` tensors. A dense
        ``clone_llm`` shell cannot load those shapes from a DeepSpeed checkpoint,
        so quantized clones reload the base via ``from_pretrained`` and transfer
        only adapter (and optimizer) state.
        """
        return self.quantization_config is not None

    def _save_clone_adapter_weights(self, work_dir: str) -> None:
        """Persist PEFT adapters for a quantized rebuild-from-pretrained clone.

        Used when ZeRO-2/3 DeepSpeed sharding is not available (or not needed)
        to move adapter weights onto a freshly loaded quantized base.

        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        adapter_dir = f"{work_dir}/adapters"
        model_ref = self._get_unwrapped_actor()
        with gather_if_zero3(self.zero_stage, adapter_checkpoint_params(model_ref)):
            model_ref.save_pretrained(
                save_directory=adapter_dir,
                selected_adapters=self.selected_adapters,
                is_main_process=self.accelerator is None
                or self.accelerator.is_main_process,
            )
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _load_clone_adapter_weights(self, work_dir: str) -> None:
        """Load PEFT adapters saved by :meth:`_save_clone_adapter_weights`.

        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        adapter_dir = f"{work_dir}/adapters"
        for adapter_name in self.selected_adapters:
            self._load_adapter_weights(adapter_dir, adapter_name)

    def _save_clone_distributed_actor_state(self, work_dir: str) -> None:
        """Save distributed actor state for ZeRO-2/3 clone workflows.

        Quantized clones also write PEFT adapters when ZeRO stage is below 2 so
        the rebuild-from-pretrained path can restore LoRA weights without a
        DeepSpeed module load of packed nf4 base tensors.

        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        quant_rebuild = self._uses_quantized_clone_rebuild()
        if quant_rebuild and (
            self.accelerator is None or self.zero_stage is None or self.zero_stage < 2
        ):
            self._save_clone_adapter_weights(work_dir)

        if self.accelerator is None or self.zero_stage is None or self.zero_stage < 2:
            return

        self.accelerator.wait_for_everyone()
        # Quantized: exclude frozen base so DeepSpeed never round-trips packed
        # Params4bit into a freshly loaded nf4 shell (adapters + opt only).
        self._save_distributed_actor(
            f"{work_dir}/agent_{self.index}",
            lora_only=quant_rebuild,
        )
        self.accelerator.wait_for_everyone()

    def _create_clone_instance(self) -> Self:
        """Instantiate a clone with cloned actor weights and runtime args.

        Quantized clones pass ``actor_network=None`` so init reloads the base
        with ``BitsAndBytesConfig`` and re-attaches adapters; adapter weights
        are restored after ``wrap_models`` via DeepSpeed (ZeRO≥2) or PEFT files.

        :return: Newly constructed clone instance.
        :rtype: Self
        """
        kwargs = constructor_kwargs_from_obj(self)
        llm = kwargs["llm"]
        model = llm.model
        if self._uses_quantized_clone_rebuild():
            actor_network = None
            model_name = self.pretrained_model_name_or_path
        else:
            actor_network = self._clone_actor_network()
            model_name = model.model_name
        accelerator = Accelerator() if self.accelerator is not None else None
        kwargs["llm"] = replace(
            llm,
            model=replace(
                model,
                actor_network=actor_network,
                model_name=model_name,
            ),
            train=replace(llm.train, clone=True),
            runtime=replace(llm.runtime, wrap=False, accelerator=accelerator),
        )
        return type(self)(**kwargs)

    def _clone_actor_network(self) -> Any:  # noqa: ANN401 -- returns a heterogeneous HF/PEFT/value-head actor wrapper
        """Clone actor network while preserving value-head state when enabled.

        :return: Cloned actor network (HF/PEFT/value-head wrapper) suitable
            for clone instantiation.
        :rtype: Any
        """
        actor = self._get_unwrapped_actor()

        if self.use_value_head:
            value_head_model = actor
            inner_peft = value_head_model.pretrained_model
            inner_sd = None
            if self.zero_stage is None or self.zero_stage < 2:
                inner_sd = clone_tensors_for_torch_save(inner_peft.state_dict())
            cloned_inner = clone_llm(inner_peft, self.zero_stage, state_dict=inner_sd)
            cloned_model = type(value_head_model)(cloned_inner)
            v_head_params = list(value_head_model.v_head.parameters())
            with gather_if_zero3(self.zero_stage, v_head_params):
                cloned_model.v_head.load_state_dict(
                    value_head_model.v_head.state_dict()
                )
            cloned_model.is_peft_model = True
            return cloned_model

        actor_state_dict = None
        if self.zero_stage is None or self.zero_stage < 2:
            actor_state_dict = clone_tensors_for_torch_save(actor.state_dict())
        return clone_llm(actor, self.zero_stage, state_dict=actor_state_dict)

    def _copy_clone_attributes(self, clone: Self) -> Self:
        """Copy non-network attributes while preserving clone runtime members.

        Keeps clone-owned accelerator/scheduler (and vLLM handles when used)
        intact while copying remaining algorithm attributes.

        :param clone: Clone instance to mutate.
        :type clone: Self
        :return: Updated clone instance.
        :rtype: Self
        """
        accelerator = clone.accelerator
        cloned_lr_scheduler = clone.lr_scheduler
        original_lr_scheduler = self.lr_scheduler

        clone.lr_scheduler = None
        self.lr_scheduler = None
        sleep_mode = bool(
            self.use_vllm
            and self.vllm_config is not None
            and self.vllm_config.sleep_mode
        )
        if self.use_vllm:
            original_llm = self.llm
            cloned_llm = clone.llm
            clone.llm = None
            self.llm = None

        clone = EvolvableCheckpointMixin.copy_attributes(self, clone)
        clone.accelerator = accelerator
        clone.lr_scheduler = cloned_lr_scheduler
        self.lr_scheduler = original_lr_scheduler

        if self.use_vllm:
            if sleep_mode:
                # CuMem is process-global: transfer the single sleep-mode engine
                # to the clone. Tournament selection cleans up the parent next.
                clone.llm = original_llm
                self.llm = None
                for attr in (
                    "_vllm_awake",
                    "_vllm_moved",
                    "_vllm_lora_loaded",
                    "_vllm_lora_staging_dir",
                    "_vllm_lora_staging_dir_is_temp",
                    "_vllm_rollout_lora_request",
                    "_vllm_rollout_adapter",
                    "tp_group",
                ):
                    if hasattr(self, attr):
                        setattr(clone, attr, getattr(self, attr))
                # Prevent parent ``clean_up`` from deleting the staging dir the
                # clone still owns.
                self._vllm_lora_staging_dir = None
                self._vllm_lora_loaded = False
                self._vllm_rollout_lora_request = None
            else:
                clone.llm = cloned_llm
                self.llm = original_llm
        return clone

    def _restore_clone_optimizer_and_scheduler(self, clone: Self) -> None:
        """Restore optimizer/scheduler state for non-accelerated clones.

        :param clone: Clone instance receiving optimizer/scheduler states.
        :type clone: Self
        """
        if self.accelerator is not None:
            return

        clone.optimizer.optimizer.load_state_dict(
            state_dict=self.optimizer.optimizer.state_dict(),
        )
        if self.lr_scheduler is not None and clone.lr_scheduler is not None:
            clone.lr_scheduler.load_state_dict(self.lr_scheduler.state_dict())

    def _load_clone_distributed_actor_state(self, clone: Self, work_dir: str) -> None:
        """Load saved distributed actor state into clone for ZeRO-2/3.

        Quantized clones without ZeRO≥2 restore adapters from the PEFT files
        written by :meth:`_save_clone_distributed_actor_state`.

        :param clone: Clone instance receiving distributed actor state.
        :type clone: Self
        :param work_dir: Shared clone workspace directory.
        :type work_dir: str
        """
        if self.zero_stage is not None and self.zero_stage >= 2:
            assert clone.accelerator is not None  # ZeRO >= 2 implies an accelerator
            clone.accelerator.wait_for_everyone()
            clone._load_distributed_actor(f"{work_dir}/agent_{self.index}")
            clone.accelerator.wait_for_everyone()
        elif self._uses_quantized_clone_rebuild():
            clone._load_clone_adapter_weights(work_dir)
            if self.use_value_head:
                clone_actor = clone._get_unwrapped_actor()
                parent_actor = self._get_unwrapped_actor()
                v_head_params = list(parent_actor.v_head.parameters())
                with gather_if_zero3(self.zero_stage, v_head_params):
                    clone_actor.v_head.load_state_dict(parent_actor.v_head.state_dict())
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        elif self.accelerator is not None:
            self.accelerator.wait_for_everyone()
