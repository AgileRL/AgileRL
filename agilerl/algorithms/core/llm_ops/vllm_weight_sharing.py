"""Zero-copy base-weight sharing between a colocated vLLM engine and the
HF/PEFT LoRA trainer.

After vLLM loads the base model (bitsandbytes-quantized or dense bf16/fp16),
its weight tensors — and, when quantized, their bnb ``QuantState`` objects —
are extracted *by reference* and grafted into an HF model skeleton, so the
trainer and vLLM share the same GPU storage. Extraction is a generic module
walk over the live text decoder (fused projections split via
``packed_modules_mapping``), not a per-architecture mapping. The shared base
is frozen and read-only on both sides; only the LoRA adapters (and, for PPO, a
trainer-only value head) differ per side. See
``docs/llm_finetuning/quantization.rst`` for the full picture.
"""

from __future__ import annotations

import gc
import itertools
import os
import warnings
from collections import OrderedDict
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "assert_shared_storage",
    "build_shared_hf_model",
    "extract_vllm_bnb_state_dict",
    "get_vllm_internal_model",
    "patch_vllm_lora_keep_resident",
    "patch_vllm_standby_sleep_mode",
    "patch_vllm_strip_multimodal_towers",
    "prepare_shared_base_for_kbit_training",
]


class _StrippedTower:
    """Falsy placeholder left in place of a tower freed by
    :func:`patch_vllm_strip_multimodal_towers`; any use of it raises.
    """

    def __init__(self, path: str) -> None:
        self._stripped_path = path

    def __bool__(self) -> bool:
        # Falsy so ``if self.vision_tower:``-style guards short-circuit.
        return False

    def _error(self, detail: str) -> str:
        return (
            f"Stripped multimodal tower '{self._stripped_path}' ({detail}). "
            "The tower was freed to save GPU memory for text-only RL training; "
            "this code path should not run. Set "
            "VLLMConfig(strip_multimodal_towers=False) (or omit "
            "--vllm-strip-multimodal-towers) to keep the towers loaded."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(self._error("called as a forward path"))

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(self._error(f"attribute '{name}' accessed"))


def patch_vllm_strip_multimodal_towers(
    llm: Any,
    tower_attrs: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    """Free the GPU memory held by a multimodal base's unused towers.

    Text-only RL rollouts never execute the vision/audio towers, so each tower
    attribute found on the model (or one level down, on ``model.model``) is
    replaced with a :class:`_StrippedTower` and its parameter storage freed.
    Must be called **after** ``LLM(...)`` returns (vLLM's init memory profile
    may touch the towers); idempotent. Checkpoints are unaffected — only the
    LoRA adapter is saved.

    :param llm: A constructed in-process ``vllm.LLM`` (external_launcher).
    :type llm: vllm.LLM
    :param tower_attrs: Attribute names to strip; ``None`` uses the standard
        HF names.
    :type tower_attrs: tuple[str, ...] | list[str] | None
    :return: Mapping ``{tower_path: param_count_freed}``; empty if there is
        nothing to strip or the model could not be reached.
    :rtype: dict[str, int]
    """
    if tower_attrs is None:
        # Standard HF names for *ForConditionalGeneration wrappers, plus
        # Gemma-4-style per-modality embedders.
        tower_attrs = (
            "vision_tower",
            "audio_tower",
            "multi_modal_projector",
            "embed_vision",
            "embed_audio",
        )
    try:
        model = get_vllm_internal_model(llm)
    except Exception:
        return {}

    holders = [(model, "")]
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        holders.append((inner, "model."))

    freed: dict[str, int] = {}
    for holder, prefix in holders:
        for attr in tower_attrs:
            sub = getattr(holder, attr, None)
            if sub is None or isinstance(sub, _StrippedTower):
                continue
            try:
                n_params = sum(int(p.numel()) for p in sub.parameters())
            except Exception:
                n_params = 0
            if isinstance(holder, torch.nn.Module):
                # nn.Module.__setattr__ refuses to replace a registered child
                # with a non-Module; drop the registration first.
                holder._modules.pop(attr, None)
            path = f"{prefix}{attr}"
            setattr(holder, attr, _StrippedTower(path))
            freed[path] = n_params

    if freed:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return freed


def patch_vllm_lora_keep_resident(llm: Any) -> int:
    """Keep vLLM's LoRA slot weights resident by neutralizing ``reset_lora``.

    vLLM (V1) zeroes a LoRA layer's GPU slot via ``reset_lora`` whenever a
    no-LoRA/dummy batch runs, but never re-copies the adapter afterwards
    (activation early-returns once the id is active) — the rollout adapter then
    silently contributes nothing. AgileRL drives a single persistent rollout
    adapter gated per token by vLLM's Punica index mapping, so the slot never
    needs clearing; a genuine adapter switch still overwrites it via
    ``set_lora``. Call once after the engine is constructed; idempotent.

    :param llm: A constructed in-process ``vllm.LLM`` (external_launcher).
    :type llm: Any
    :return: Number of LoRA layers neutralized (0 if LoRA is disabled or the
        model is unreachable).
    :rtype: int
    """
    try:
        model = get_vllm_internal_model(llm)
    except Exception:
        return 0

    count = 0
    for module in model.modules():
        # A LoRA-wrapped layer exposes both reset_lora and the stacked GPU slot.
        if (
            hasattr(module, "reset_lora")
            and hasattr(module, "lora_b_stacked")
            and not getattr(module, "_agilerl_lora_resident", False)
        ):
            module.reset_lora = lambda *args, **kwargs: None
            module._agilerl_lora_resident = True
            count += 1
    return count


def _expandable_segments_enabled() -> bool:
    """Whether ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is set."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return "expandable_segments:true" in conf.replace(" ", "").lower()


def patch_vllm_standby_sleep_mode() -> None:
    """Patch vLLM's ``CuMemAllocator`` so sleep keeps the base weights resident.

    Native sleep moves the base off the GPU (level 1 offloads to host RAM,
    level 2 discards and expects a reload — impossible in-place for a bnb
    4-bit base). Standby keeps allocations tagged ``"weights"`` physically
    resident across sleep/wake and frees only the KV cache and other
    recomputable allocations, so the single shared base copy never moves and
    generation after wake is bit-identical. Idempotent; no-op when vLLM is not
    installed.

    :raises RuntimeError: If ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
        is set (incompatible with standby sleep).
    """
    try:
        from vllm.device_allocator.cumem import (
            CuMemAllocator,
            create_and_map,
            libcudart,
            unmap_and_release,
        )
    except Exception:
        return

    if getattr(CuMemAllocator, "_agilerl_standby_patched", False):
        return

    if _expandable_segments_enabled():
        msg = (
            "vLLM standby sleep (weight-sharing) is incompatible with "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Unset it before "
            "enabling weight_sharing."
        )
        raise RuntimeError(msg)

    try:
        from vllm.utils import is_pin_memory_available
    except Exception:
        try:
            from vllm.utils.platform_utils import is_pin_memory_available
        except Exception:

            def is_pin_memory_available() -> bool:  # type: ignore[misc]
                return False

    def sleep(self, offload_tags=None) -> None:
        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        for ptr, data in self.pointer_to_data.items():
            # Keep base weights resident — never offload or discard them.
            if data.tag == "weights":
                continue
            handle = data.handle
            if data.tag in offload_tags:
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )
                libcudart.cudaMemcpy(cpu_backup_tensor.data_ptr(), ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags=None) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        for ptr, data in self.pointer_to_data.items():
            # Weights were never released; nothing to recreate for them.
            if data.tag == "weights":
                continue
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    size_in_bytes = (
                        cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                    )
                    libcudart.cudaMemcpy(
                        ptr, cpu_backup_tensor.data_ptr(), size_in_bytes
                    )
                    data.cpu_backup_tensor = None

    CuMemAllocator.sleep = sleep
    CuMemAllocator.wake_up = wake_up
    CuMemAllocator._agilerl_standby_patched = True


def get_vllm_internal_model(llm: Any) -> nn.Module:
    """Return the live ``nn.Module`` inside an in-process vLLM ``LLM``,
    trying the attribute layouts of the known vLLM versions / engine cores.

    :param llm: A constructed ``vllm.LLM`` instance.
    :type llm: Any
    :return: The underlying model module.
    :rtype: nn.Module
    :raises RuntimeError: If the model cannot be located.
    """
    engine = getattr(llm, "llm_engine", getattr(llm, "engine", llm))
    candidates = []
    core = getattr(engine, "engine_core", None)
    if core is not None:
        candidates.append(getattr(core, "engine_core", core))
    candidates.append(engine)

    def _model_from(base: Any) -> nn.Module | None:
        try:
            return base.model_executor.driver_worker.model_runner.model
        except AttributeError:
            return None

    for base in candidates:
        model = _model_from(base)
        if model is not None:
            return model
    msg = (
        "Could not locate the vLLM internal model. Weight sharing requires an "
        "in-process engine (distributed_executor_backend='external_launcher')."
    )
    raise RuntimeError(msg)


def _resolve_dtype(value: Any) -> torch.dtype:
    """Resolve a torch dtype from a dtype, ``"torch.x"`` string, or bare name."""
    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, str(value).removeprefix("torch."))


def _bnb_linear_kwargs(bnb_config: Any) -> dict[str, Any]:
    """Map a ``BitsAndBytesConfig`` to ``Linear4bit`` / ``Params4bit`` kwargs."""
    return {
        "compress_statistics": bnb_config.bnb_4bit_use_double_quant,
        "quant_type": bnb_config.bnb_4bit_quant_type,
        "quant_storage": _resolve_dtype(bnb_config.bnb_4bit_quant_storage),
    }


def _truncate_vocab(weight: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Drop vLLM's padded-vocab rows (HF expects exactly ``vocab_size``)."""
    if vocab_size and weight.shape[0] > vocab_size:
        return weight[:vocab_size]
    return weight


# Sentinel distinguishing "attribute absent" from a stored None.
_MISSING = object()


def _navigate(root: Any, dotted: str) -> Any:
    """Walk ``root.<dotted>``, supporting ``ModuleList`` integer indices."""
    cur = root
    if dotted:
        for part in dotted.split("."):
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _navigate_safe(root: Any, dotted: str) -> Any:
    """Like :func:`_navigate` but return :data:`_MISSING` if any hop fails."""
    try:
        return _navigate(root, dotted)
    except (AttributeError, IndexError, KeyError, TypeError):
        return _MISSING


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace ``model.<name>`` with ``new_module`` (supports ModuleList idx)."""
    parent_path, _, leaf = name.rpartition(".")
    parent = _navigate(model, parent_path)
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Strip a tensor subclass to a plain ``Tensor`` view of the same storage
    (vLLM's parameter subclasses break ``nn.Parameter(...)``).
    """
    if type(tensor) is torch.Tensor:
        return tensor
    return tensor.as_subclass(torch.Tensor)


def _graft_param(
    model: nn.Module, name: str, tensor: torch.Tensor, requires_grad: bool = False
) -> None:
    """Set ``model.<name>`` to a ``Parameter`` aliasing ``tensor``."""
    parent_path, _, leaf = name.rpartition(".")
    parent = _navigate(model, parent_path)
    param = torch.nn.Parameter(_plain_tensor(tensor), requires_grad=requires_grad)
    setattr(parent, leaf, param)


def _target_exists(model: nn.Module, name: str) -> bool:
    """Whether ``model.<name>`` resolves to an existing attribute / index."""
    return _navigate_safe(model, name) is not _MISSING


def _quant_target(model: nn.Module, owner: str) -> str | None:
    """Graft target for a shared linear: ``owner.linear`` when the HF module
    is a clipping-style wrapper around a plain ``nn.Linear`` (e.g.
    ``Gemma4ClippableLinear``), ``owner`` otherwise, or ``None`` when the HF
    model omits the module (e.g. k/v projections on KV-shared layers).
    """
    module = _navigate_safe(model, owner)
    if module is _MISSING:
        return None
    if isinstance(getattr(module, "linear", None), torch.nn.Linear):
        return f"{owner}.linear"
    return owner


def _override_to(module: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
    """No-op ``.to``: a re-cast would break bnb 4-bit weights or silently
    copy (un-alias) a dense shared base.
    """
    return module


def _language_model_layout(internal: nn.Module) -> tuple[Any, str, Any]:
    """Locate the text decoder inside a (possibly multimodal) vLLM model.

    :return: ``(decoder_root, hf_prefix, lm_head_or_None)`` where ``hf_prefix``
        is ``"model"`` for a plain causal-LM or ``"model.language_model"`` for
        a conditional-generation wrapper.
    """
    if hasattr(internal, "language_model"):
        lm = internal.language_model
        root = getattr(lm, "model", lm)
        return root, "model.language_model", getattr(lm, "lm_head", None)
    if hasattr(internal, "model"):
        return internal.model, "model", getattr(internal, "lm_head", None)
    msg = (
        "Could not locate the language-model decoder in the vLLM model "
        f"({type(internal).__name__}); weight sharing supports text decoders "
        "(optionally wrapped for conditional generation)."
    )
    raise RuntimeError(msg)


def _output_size_offsets(module: Any, subs: list[str], pname: str) -> list[int]:
    """Per-shard row offsets of a fused projection from vLLM's ``output_sizes``
    (used for dense fused weights and for fused biases, which are dense even on
    a quantized base and never carried in ``bnb_shard_offsets``).
    """
    output_sizes = getattr(module, "output_sizes", None)
    if output_sizes is None or len(output_sizes) != len(subs):
        msg = (
            f"Cannot split the fused parameter {pname!r} into {len(subs)} "
            f"shards: the vLLM module exposes output_sizes={output_sizes!r}, "
            f"which does not match packed_modules_mapping={subs!r}."
        )
        raise RuntimeError(msg)
    return list(itertools.accumulate([0, *output_sizes]))


def _extract_with_layout(
    llm: Any,
    hf_config: Any,
) -> tuple[OrderedDict[str, Any], nn.Module, Any, str]:
    """:func:`extract_vllm_bnb_state_dict` body that also returns the resolved
    ``(state_dict, internal, decoder_root, hf_prefix)`` layout, so callers
    needing both don't walk the vLLM tree twice.
    """
    internal = get_vllm_internal_model(llm)
    root, hf_prefix, lm_head = _language_model_layout(internal)
    packed = getattr(internal, "packed_modules_mapping", None) or {}
    text_config = getattr(hf_config, "text_config", hf_config)
    vocab_size = text_config.vocab_size

    out: OrderedDict[str, Any] = OrderedDict()

    for pname, param in root.named_parameters(recurse=True):
        # Skip vLLM's LoRA adapter slots; only the frozen base is shared.
        if "lora" in pname.lower():
            continue
        # vLLM LoRA-wraps each linear under ``.base_layer``; HF is un-nested.
        name = pname.replace(".base_layer.", ".")
        if name.endswith(".weight"):
            owner = name[: -len(".weight")]
            leaf = owner.rsplit(".", 1)[-1]
            quant_states = getattr(param, "bnb_quant_state", None)
            if quant_states is not None:
                raw = getattr(param, "bnb_shard_offsets", None)
                offsets = (
                    [int(x) for x in raw]
                    if raw is not None
                    else [0, int(param.shape[0])]
                )
                n_shards = len(offsets) - 1
                subs = packed.get(leaf)
                if subs is not None and len(subs) == n_shards and n_shards > 1:
                    parent_hf = f"{hf_prefix}.{owner.rsplit('.', 1)[0]}"
                    for i, sub in enumerate(subs):
                        out[f"{parent_hf}.{sub}.weight"] = param[
                            offsets[i] : offsets[i + 1]
                        ]
                        out[f"{parent_hf}.{sub}.weight.quant_state"] = quant_states[i]
                elif n_shards == 1:
                    hf_name = f"{hf_prefix}.{owner}"
                    out[f"{hf_name}.weight"] = param[offsets[0] : offsets[1]]
                    out[f"{hf_name}.weight.quant_state"] = quant_states[0]
                else:
                    msg = (
                        f"Cannot name the {n_shards} fused shards of {pname!r}: no "
                        f"matching packed_modules_mapping entry for {leaf!r}."
                    )
                    raise RuntimeError(msg)
            else:
                # Dense weight; a fused projection still splits into its HF
                # sub-modules, with shard sizes from vLLM's ``output_sizes``.
                param.requires_grad_(False)
                subs = packed.get(leaf)
                if subs is not None and len(subs) > 1:
                    module = _navigate_safe(root, pname[: -len(".weight")])
                    offsets = _output_size_offsets(module, subs, pname)
                    parent_hf = f"{hf_prefix}.{owner.rsplit('.', 1)[0]}"
                    for i, sub in enumerate(subs):
                        out[f"{parent_hf}.{sub}.weight"] = param[
                            offsets[i] : offsets[i + 1]
                        ]
                else:
                    weight = param
                    if leaf == "embed_tokens":
                        weight = _truncate_vocab(param, vocab_size)
                    out[f"{hf_prefix}.{owner}.weight"] = weight
        elif name.endswith(".bias"):
            # Fused biases (dense even on a quantized base) split like dense
            # weights — dropping them silently breaks e.g. Qwen2 attention.
            param.requires_grad_(False)
            owner = name[: -len(".bias")]
            leaf = owner.rsplit(".", 1)[-1]
            subs = packed.get(leaf)
            if subs is not None and len(subs) > 1:
                module = _navigate_safe(root, pname[: -len(".bias")])
                offsets = _output_size_offsets(module, subs, pname)
                parent_hf = f"{hf_prefix}.{owner.rsplit('.', 1)[0]}"
                for i, sub in enumerate(subs):
                    out[f"{parent_hf}.{sub}.bias"] = param[offsets[i] : offsets[i + 1]]
            else:
                out[f"{hf_prefix}.{owner}.bias"] = param
        else:
            # Raw parameter (scales, router coefficients, ...): carried 1:1.
            param.requires_grad_(False)
            out[f"{hf_prefix}.{name}"] = param

    # Separate (untied) lm_head lives outside the decoder root.
    if not getattr(text_config, "tie_word_embeddings", False):
        if lm_head is not None and getattr(lm_head, "weight", None) is not None:
            lm_head.weight.requires_grad_(False)
            out["lm_head.weight"] = _truncate_vocab(lm_head.weight, vocab_size)

    return out, internal, root, hf_prefix


def extract_vllm_bnb_state_dict(
    llm: Any,
    hf_config: Any,
) -> OrderedDict[str, Any]:
    """Extract an HF-named, zero-copy view of vLLM's language-model weights.

    Every value aliases vLLM's storage. A bnb-quantized ``.weight`` is emitted
    with a companion ``"{name}.weight.quant_state"`` entry; fused projections
    (``qkv_proj``/``gate_up_proj``) are split into their HF sub-modules via
    ``packed_modules_mapping`` (per-shard ``bnb_shard_offsets`` when quantized,
    the vLLM linear's ``output_sizes`` when dense); biases and raw parameters
    are carried through 1:1. Keys follow HF naming (including the
    ``model.language_model.`` prefix for a multimodal wrapper) so they graft
    directly onto an HF skeleton.

    :param llm: Constructed ``vllm.LLM`` (in-process engine).
    :type llm: Any
    :param hf_config: The HF ``PretrainedConfig`` for the model.
    :type hf_config: Any
    :return: Ordered dict of HF-named shared tensors / quant states.
    :rtype: OrderedDict[str, Any]
    :raises RuntimeError: If a fused module's shards cannot be named.
    """
    return _extract_with_layout(llm, hf_config)[0]


def build_shared_hf_model(
    llm: Any,
    hf_config: Any,
    compute_dtype: torch.dtype,
    bnb_config: Any,
    share_towers: bool = False,
    attn_implementation: str | None = None,
    add_value_head: bool = False,
) -> nn.Module:
    """Build an HF causal-LM whose language weights alias vLLM's tensors.

    Constructs an empty (meta-device) skeleton from ``hf_config`` and grafts in
    the shared weights from :func:`extract_vllm_bnb_state_dict`: bnb
    ``Linear4bit`` modules (aliased ``Params4bit`` + ``QuantState``) for a
    quantized base, shared ``nn.Linear`` weights for a dense one; embeddings /
    norms / raw params are shared either way. For a multimodal base the full
    skeleton is built (so LoRA adapter names match vLLM's layout) but the
    non-language towers are frozen, uninitialised placeholders never executed
    in a text forward.

    :param llm: Constructed ``vllm.LLM`` (in-process engine, base loaded).
    :type llm: Any
    :param hf_config: The HF ``PretrainedConfig`` for the model.
    :type hf_config: Any
    :param compute_dtype: Compute dtype for the bnb ``Linear4bit`` modules and
        the value head (e.g. ``torch.bfloat16``).
    :type compute_dtype: torch.dtype
    :param bnb_config: The trainer's ``BitsAndBytesConfig`` when the base is
        quantized, or ``None`` for a dense base.
    :type bnb_config: Any
    :param share_towers: Reserved toggle for sharing vision/audio towers too;
        not implemented (text-only).
    :type share_towers: bool
    :param add_value_head: Wrap in ``AutoModelForCausalLMWithValueHead`` (PPO).
        The value head is a small trainer-only module, never aliased or used
        by rollout.
    :type add_value_head: bool
    :return: An ``nn.Module`` ready for PEFT wrapping.
    :rtype: nn.Module
    :raises NotImplementedError: If ``share_towers`` is True.
    """
    if share_towers:
        msg = (
            "share_towers=True (multimodal tower sharing) is not implemented; "
            "only the language model is shared."
        )
        raise NotImplementedError(msg)

    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    from agilerl.utils.llm_utils import (
        patch_flex_attention_kernel_options,
        resolve_attn_implementation,
    )

    shared, _internal, vllm_root, hf_prefix = _extract_with_layout(llm, hf_config)
    device = torch.device("cuda", torch.cuda.current_device())
    is_quantized = bnb_config is not None
    bnb_kwargs = _bnb_linear_kwargs(bnb_config) if is_quantized else {}
    text_config = getattr(hf_config, "text_config", hf_config)

    # Meta-device skeleton: params hold no storage, buffers are real. Prefer
    # flash attention when available — SDPA's masked fallback materialises the
    # full TxT score matrix and OOMs on long RL contexts.
    attn_impl = resolve_attn_implementation(attn_implementation)
    if attn_impl == "flex_attention":
        patch_flex_attention_kernel_options()
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            hf_config, attn_implementation=attn_impl
        )

    # 1. Quantized linears -> shared bnb Linear4bit (aliased Params4bit).
    handled: set[str] = set()
    skipped = 0
    quant_owners = sorted(
        k[: -len(".weight.quant_state")]
        for k in shared
        if k.endswith(".weight.quant_state")
    )
    if quant_owners:
        from bitsandbytes.nn.modules import Linear4bit, Params4bit
    for owner in quant_owners:
        handled.update(
            {f"{owner}.weight", f"{owner}.weight.quant_state", f"{owner}.bias"}
        )
        target = _quant_target(model, owner)
        if target is None:  # vLLM-only module (e.g. k/v on a KV-shared layer)
            skipped += 1
            continue
        quant_state = shared[f"{owner}.weight.quant_state"]
        weight = shared[f"{owner}.weight"]
        bias = shared.get(f"{owner}.bias")
        layer = Linear4bit(
            0,
            0,
            device=device,
            bias=bias is not None,
            compute_dtype=compute_dtype,
            **bnb_kwargs,
        )
        layer.in_features = quant_state.shape[1]
        layer.out_features = quant_state.shape[0]
        layer.weight = Params4bit(data=weight, requires_grad=False, **bnb_kwargs)
        layer.weight.quant_state = quant_state
        if bias is not None:
            layer.bias = torch.nn.Parameter(_plain_tensor(bias), requires_grad=False)
        # bnb errors if a 4-bit weight is re-cast; freeze .to to no-ops.
        layer.to = partial(_override_to, layer)
        layer.weight.to = partial(_override_to, layer.weight)
        _set_submodule(model, target, layer)

    # 2. Everything else: embeddings, norms, raw params, untied lm_head and —
    # for a dense base — every linear weight/bias. ``_quant_target`` applies
    # the clipping-wrapper redirection and the vLLM-only skip uniformly.
    for key, val in shared.items():
        if key in handled:
            continue
        grafted = False
        for suffix in (".weight", ".bias"):
            if key.endswith(suffix):
                target = _quant_target(model, key[: -len(suffix)])
                if target is not None:
                    _graft_param(model, f"{target}{suffix}", val, requires_grad=False)
                    grafted = True
                break
        else:  # raw parameter (no .weight/.bias suffix)
            if _target_exists(model, key):
                _graft_param(model, key, val, requires_grad=False)
                grafted = True
        if not grafted:
            skipped += 1

    if skipped:
        warnings.warn(
            f"weight_sharing: skipped {skipped} vLLM params with no HF target "
            "(e.g. k/v projections + norms on KV-shared layers, which HF reuses "
            "from a paired layer).",
            stacklevel=2,
        )

    # 3. Persistent checkpoint buffers (e.g. per-layer scalars) are loaded, not
    # computed — overwrite the skeleton's __init__ defaults with vLLM's values.
    for bname, buf in vllm_root.named_buffers(recurse=True):
        if "lora" in bname.lower():
            continue
        hf_name = f"{hf_prefix}.{bname.replace('.base_layer.', '.')}"
        parent_path, _, leaf = hf_name.rpartition(".")
        parent = _navigate_safe(model, parent_path)
        if parent is not _MISSING and leaf in getattr(parent, "_buffers", {}):
            parent._buffers[leaf] = buf.detach()

    # 4. Tie lm_head to the grafted embeddings; move the remaining buffers
    # (computed correctly at from_config time, e.g. rotary inv_freq) to GPU.
    if getattr(text_config, "tie_word_embeddings", False):
        model.tie_weights()
    _move_buffers_to_device(model, device)

    # 5. Leftover meta params: under the language prefix it is a coverage bug;
    # the rest are towers, materialised as frozen empties (unused in text RL).
    lang_leftover: list[str] = []
    for name, param in model.named_parameters():
        if param.device.type != "meta":
            continue
        if name.startswith(hf_prefix + ".") or name == "lm_head.weight":
            lang_leftover.append(name)
        else:
            _graft_param(
                model,
                name,
                torch.empty(param.shape, dtype=param.dtype, device=device),
                requires_grad=False,
            )
    if lang_leftover:
        msg = (
            f"Weight sharing left {len(lang_leftover)} language params un-grafted "
            f"(still on meta), e.g. {lang_leftover[:8]}. The vLLM/HF naming for "
            "this model is not fully covered by the generic walk."
        )
        raise RuntimeError(msg)
    n_towers = _materialise_meta_buffers(model, device)
    if n_towers:
        warnings.warn(
            f"weight_sharing: materialised {n_towers} non-language buffers as "
            "frozen empties (vision/audio towers; unused in text rollouts).",
            stacklevel=2,
        )

    if is_quantized:
        # Flag as bnb 4-bit so PEFT's kbit prep / get_peft_model treat it as
        # QLoRA; a dense shared base sets none of these.
        model.is_loaded_in_4bit = True
        model.is_loaded_in_8bit = False
        model.is_quantized = True
        model.quantization_method = "bitsandbytes"
        model.config.quantization_config = bnb_config
    # Freeze ``.to`` so accelerate can't re-cast (un-alias) the shared base.
    model.to = partial(_override_to, model)
    model.eval()

    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    if add_value_head:
        from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

        # The inner causal LM keeps its no-op ``.to``; the wrapper's ``.to``
        # still moves the (trainer-only) head.
        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.v_head.to(device=device, dtype=compute_dtype)
        return wrapped
    return model


def _rewrite_buffers(model: nn.Module, fn: Any) -> int:
    """Replace each non-None buffer with ``fn(buf)``; ``fn`` returning ``None``
    leaves the buffer unchanged. Returns the number replaced.
    """
    count = 0
    for module in model.modules():
        for buf_name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            new = fn(buf)
            if new is not None:
                module._buffers[buf_name] = new
                count += 1
    return count


def _move_buffers_to_device(model: nn.Module, device: torch.device) -> None:
    """Move every (real) buffer onto ``device`` without touching params."""

    def _to_device(buf: torch.Tensor) -> torch.Tensor | None:
        if buf.device.type not in ("meta", device.type):
            return buf.to(device)
        return None

    _rewrite_buffers(model, _to_device)


def _materialise_meta_buffers(model: nn.Module, device: torch.device) -> int:
    """Replace leftover meta buffers (towers) with frozen empties; count them."""

    def _empty(buf: torch.Tensor) -> torch.Tensor | None:
        if buf.device.type == "meta":
            return torch.empty(buf.shape, dtype=buf.dtype, device=device)
        return None

    return _rewrite_buffers(model, _empty)


def prepare_shared_base_for_kbit_training(
    model: nn.Module,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """kbit-prep for a *shared* frozen base — like peft's but with no fp32 upcast.

    ``peft.prepare_model_for_kbit_training`` upcasts every non-``Params4bit``
    bf16/fp16 param to fp32, which both allocates huge transient copies (OOM at
    high ``gpu_memory_utilization``) and un-aliases the shared base from vLLM.
    The base is frozen and only LoRA trains, so this replicates just the parts
    that matter — freeze the base, enable gradient checkpointing — keeping it
    bf16 and aliased.

    :param model: The shared HF base model (pre-PEFT).
    :type model: nn.Module
    :param use_gradient_checkpointing: Enable gradient checkpointing.
    :type use_gradient_checkpointing: bool
    :param gradient_checkpointing_kwargs: Forwarded to
        ``gradient_checkpointing_enable`` (e.g. ``{"use_reentrant": False}``).
    :type gradient_checkpointing_kwargs: dict[str, Any] | None
    :return: The same model, frozen and (optionally) checkpointing-enabled.
    :rtype: nn.Module
    """
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}
    model.requires_grad_(False)
    if use_gradient_checkpointing:
        # use_reentrant=True needs grad-requiring inputs so the checkpointed
        # graph can backprop into the LoRA adapters (matches peft's logic).
        if gradient_checkpointing_kwargs.get("use_reentrant", True):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def _make_inputs_require_grad(_module, _inp, out):
                    out.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    _make_inputs_require_grad
                )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
    return model


def assert_shared_storage(llm: Any, hf_model: nn.Module) -> None:
    """Spot-check that the HF trainer base aliases vLLM's storage: a handful of
    grafted parameters' ``data_ptr()`` must equal the corresponding vLLM
    tensor's.

    :param llm: The vLLM ``LLM`` whose tensors were shared.
    :type llm: Any
    :param hf_model: The model returned by :func:`build_shared_hf_model`
        (pass the unwrapped base if already PEFT-wrapped).
    :type hf_model: nn.Module
    :raises RuntimeError: If a checked parameter does not alias vLLM's storage.
    """
    shared = extract_vllm_bnb_state_dict(llm, hf_model.config)
    checked = 0
    mismatches: list[str] = []
    for name, module in hf_model.named_modules():
        key = f"{name}.weight"
        if key not in shared:
            continue
        # The real weight may be nested in a clipping-linear wrapper.
        holder = (
            module if hasattr(module, "weight") else getattr(module, "linear", None)
        )
        hf_weight = getattr(holder, "weight", None)
        if hf_weight is None:
            continue
        if hf_weight.data_ptr() != shared[key].data_ptr():
            mismatches.append(name)
        checked += 1
        if checked >= 8 and not mismatches:
            break
    if mismatches:
        msg = (
            "Weight sharing failed: these HF base params do not alias vLLM's "
            f"storage (a copy was made): {mismatches[:8]}"
        )
        raise RuntimeError(msg)
    if checked == 0:
        msg = "Weight sharing self-check found no shared parameters to verify."
        raise RuntimeError(msg)
