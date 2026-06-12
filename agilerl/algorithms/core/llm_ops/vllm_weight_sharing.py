"""Zero-copy base-weight sharing between a colocated vLLM engine and the
HF/PEFT LoRA trainer (bitsandbytes-quantized or dense bf16/fp16).

The rollout engine (vLLM) and the trainer (HF + PEFT) load the *same* base
model; this module removes the trainer's separate copy. After vLLM loads the
base, its weight tensors (and, when quantized, their bnb ``QuantState`` objects)
are extracted *by reference* and grafted into an HF model skeleton, so the
trainer's linears and vLLM share the exact same GPU storage. For a quantized
base the grafted modules are bnb ``Linear4bit``; for a dense base the shared
``Params`` are grafted onto the skeleton's ``nn.Linear`` directly. Only the LoRA
adapters (and, for PPO, a small trainer-only value head) differ per side.
Combined with :func:`patch_vllm_standby_sleep_mode`, the shared base never
leaves the GPU and never needs reloading.

Scope (v1): the **language model only**. RL rollouts are text-only, so for a
multimodal base the trainer skeleton is still the full model (so LoRA adapter
names match vLLM's layout) but the non-language towers are materialised as
frozen, uninitialised placeholders — never executed in a text forward. Sharing
the towers too is a future toggle (``share_towers``).

Extraction is a generic module-walk over vLLM's live text decoder: it splits
fused projections via ``packed_modules_mapping`` and takes everything else 1:1,
rather than hardcoding per-architecture layer-name lists.

Aliasing-safety invariant: the shared base weights are frozen and read-only on
both sides (the trainer only updates LoRA deltas), so neither side ever mutates
the shared tensors.
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


# Default submodule attribute names to free for text-only training of a
# multimodal base. ``vision_tower`` / ``audio_tower`` / ``multi_modal_projector``
# are the standard HF transformers names for *ForConditionalGeneration wrappers;
# ``embed_vision`` / ``embed_audio`` cover Gemma-4-style per-modality embedders.
# Pass a custom list via ``VLLMConfig(strip_multimodal_towers=[...])`` for
# models that mount their towers under other attribute names.
_STRIPPABLE_TOWER_ATTRS: tuple[str, ...] = (
    "vision_tower",
    "audio_tower",
    "multi_modal_projector",
    "embed_vision",
    "embed_audio",
)


def _stripped_error(path: str, detail: str) -> str:
    """Build the shared "tower was stripped" message for ``_StrippedTower``."""
    return (
        f"Stripped multimodal tower '{path}' ({detail}). The tower was freed to "
        "save GPU memory for text-only RL training; this code path should not "
        "run. Set VLLMConfig(strip_multimodal_towers=False) (or omit "
        "--vllm-strip-multimodal-towers) to keep the towers loaded."
    )


class _StrippedTower:
    """Placeholder left in place of a freed multimodal tower.

    Not an ``nn.Module`` on purpose — any code path that iterates the model's
    children and then invokes a tower fails loudly (``__call__``/``__getattr__``
    raise) rather than silently dispatching to an empty sub-module. The owning
    model keeps the attribute name (so ``hasattr`` checks pass) while the
    tower's parameters are gone from GPU memory; ``__bool__`` is falsy so
    ``if self.vision_tower is not None`` branches skip cleanly.
    """

    __slots__ = ("_stripped_path",)

    def __init__(self, stripped_path: str) -> None:
        self._stripped_path = stripped_path

    def __repr__(self) -> str:
        return f"_StrippedTower({self._stripped_path!r})"

    def __bool__(self) -> bool:
        return False

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            _stripped_error(self._stripped_path, "called as a forward path")
        )

    def __getattr__(self, name: str) -> Any:
        if name == "_stripped_path":
            raise AttributeError(name)
        raise AttributeError(
            _stripped_error(self._stripped_path, f"attribute '{name}' accessed")
        )


def _walk_tower_holders(
    model: nn.Module,
    tower_attrs: tuple[str, ...] = _STRIPPABLE_TOWER_ATTRS,
) -> Any:
    """Yield ``(holder_module, attr_name, full_path)`` for every strippable
    tower attribute reachable on ``model`` or ``model.model``.

    Only the immediate-parent module is yielded so the caller can ``setattr``
    to swap the tower out and drop the GPU reference. Deeper traversal isn't
    needed because multimodal wrappers put towers at the top or one level under
    ``model.``; expanding the search to arbitrary depth would risk catching
    something not intended for stripping.
    """
    holders = [(model, "")]
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        holders.append((inner, "model."))
    for holder, prefix in holders:
        for attr in tower_attrs:
            sub = getattr(holder, attr, None)
            if sub is None:
                continue
            if isinstance(sub, _StrippedTower):
                continue
            yield holder, attr, f"{prefix}{attr}"


def patch_vllm_strip_multimodal_towers(
    llm: Any,
    tower_attrs: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    """Free GPU memory used by the multimodal towers of a text-only-RL base.

    Multimodal ``*ForConditionalGeneration`` classes (Gemma-4-MM and similar)
    load vision + audio + connector submodules into GPU memory at engine init,
    even when the RL rollout never feeds an image/audio token. Those towers
    are dead weight on a tight colocated budget — for Gemma-4-E4B the vision
    tower alone is a SigLIP-style encoder of ~16 layers and the audio tower
    is a similarly sized USM-style encoder, together typically 1-3 GiB.

    ``tower_attrs`` overrides the default attribute names
    (:data:`_STRIPPABLE_TOWER_ATTRS` — the standard HF naming convention) for
    models that mount unwanted modalities under different attributes.

    This walks the live in-process model, replaces each tower attribute with a
    :class:`_StrippedTower` placeholder (so falsy checks like
    ``if vision_tower is not None`` still short-circuit cleanly), and frees
    the underlying parameter storage. After ``gc.collect`` +
    ``torch.cuda.empty_cache`` the GPU pool is reclaimed.

    Must be called **after** ``LLM(...)`` returns — vLLM's init memory profile
    runs *during* construction and may touch the towers. Once init is done,
    text-only rollouts never invoke them; the stub's ``__call__`` raises a
    precise error if a future code path ever tries.

    Checkpoints are unaffected: AgileRL saves only the LoRA adapter; the base
    (including towers) is referenced by ``pretrained_model_name_or_path`` so
    downstream loads pick up the original towers from the HF Hub model.

    :param llm: A constructed in-process ``vllm.LLM`` (external_launcher).
    :type llm: vllm.LLM
    :param tower_attrs: Attribute names to strip; ``None`` uses the standard
        HF names in :data:`_STRIPPABLE_TOWER_ATTRS`.
    :type tower_attrs: tuple[str, ...] | list[str] | None
    :return: Mapping ``{tower_full_path: param_count_freed}``. Empty if the
        model has no strippable towers or it could not be reached.
    :rtype: dict[str, int]
    """
    try:
        model = get_vllm_internal_model(llm)
    except Exception:
        return {}

    attrs = _STRIPPABLE_TOWER_ATTRS if tower_attrs is None else tuple(tower_attrs)
    freed: dict[str, int] = {}
    for holder, attr, full_path in _walk_tower_holders(model, attrs):
        sub = getattr(holder, attr)
        try:
            n_params = sum(int(p.numel()) for p in sub.parameters())
        except Exception:
            n_params = 0
        if isinstance(holder, torch.nn.Module):
            # A registered child module can't be replaced by a non-Module via
            # plain attribute assignment (``nn.Module.__setattr__`` raises
            # TypeError); drop the registration first so the placeholder lands
            # as a plain instance attribute.
            holder._modules.pop(attr, None)
        setattr(holder, attr, _StrippedTower(full_path))
        freed[full_path] = n_params

    if freed:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return freed


def _noop_reset_lora(*args: Any, **kwargs: Any) -> None:
    """Replacement for a LoRA layer's ``reset_lora`` that leaves the slot intact."""
    return


def patch_vllm_lora_keep_resident(llm: Any) -> int:
    """Keep each LoRA slot's weights resident by neutralizing ``reset_lora``.

    vLLM (V1) zeroes a LoRA layer's GPU slot via ``reset_lora`` whenever a
    no-LoRA / dummy batch runs, but on the next LoRA forward it takes the
    "already loaded" path and never re-copies the adapter into the slot
    (``LoRAModelManager.activate_adapter`` early-returns once the id is active,
    and the worker only re-activates on first load / ``load_inplace``). The slot
    therefore stays zero and the rollout adapter contributes nothing — vLLM runs
    the LoRA Punica kernels on empty buffers, so generation is bit-identical to
    the base model no matter how much the adapter has trained.

    AgileRL drives a *single persistent* rollout adapter and selects it per
    request, so per-token application is gated by vLLM's Punica index mapping,
    not by clearing slots. Neutralizing ``reset_lora`` keeps the adapter weights
    resident across no-LoRA batches; a genuine adapter switch still overwrites
    the slot via ``set_lora``, and unmapped tokens never read the slot. This is
    what makes the trained adapter actually affect rollout generation.

    Must be called once, after the in-process engine is constructed (the LoRA
    layers only exist after ``LLM(...)``). Idempotent per layer. Safe to call
    when LoRA is disabled or the model can't be reached (returns 0).

    :param llm: A constructed in-process ``vllm.LLM`` (external_launcher).
    :type llm: Any
    :return: Number of LoRA layers neutralized.
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
            module.reset_lora = _noop_reset_lora
            module._agilerl_lora_resident = True
            count += 1
    return count


def _expandable_segments_enabled() -> bool:
    """Whether ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is set.

    Standby sleep is incompatible with expandable segments (the released
    physical pages are not returned in a way the standby path can rely on),
    so callers must guard against it.
    """
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return "expandable_segments:true" in conf.replace(" ", "").lower()


def patch_vllm_standby_sleep_mode() -> None:
    """Patch vLLM's ``CuMemAllocator`` for "standby" sleep.

    vLLM's native sleep modes both move the base weights off the GPU: level 1
    offloads them to host RAM (and the freed virtual mapping is not reclaimed
    from PyTorch's allocator perspective), level 2 discards them and expects
    the caller to reload on wake. Reloading a bnb 4-bit model in-place is not
    supported by vLLM and produces garbage logits.

    Standby sidesteps both by keeping anything tagged ``"weights"`` physically
    resident across sleep/wake and freeing only the KV cache (and other
    recomputable, non-weight allocations). The base weights never move, so no
    reload is needed and generation after wake is bit-identical to before
    sleep. This is what makes :func:`build_shared_hf_model` viable: the single
    shared base copy stays put.

    Idempotent: applying twice is a no-op (guarded by a sentinel attr). Safe to
    call when vLLM isn't installed (no-op).

    :raises RuntimeError: If ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
        is set, which is incompatible with standby sleep.
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

    if _expandable_segments_enabled():  # pragma: no cover - needs CUDA allocator
        msg = (
            "vLLM standby sleep (weight-sharing) is incompatible with "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Unset it before "
            "enabling weight_sharing."
        )
        raise RuntimeError(msg)

    try:  # pragma: no cover - vLLM runtime only
        from vllm.utils import is_pin_memory_available
    except Exception:
        try:
            from vllm.utils.platform_utils import is_pin_memory_available
        except Exception:

            def is_pin_memory_available() -> bool:  # type: ignore[misc]
                return False

    def sleep(
        self, offload_tags=None
    ) -> None:  # pragma: no cover - drives the CUDA mem pool
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

    def wake_up(self, tags=None) -> None:  # pragma: no cover - drives the CUDA mem pool
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
    """Return the live ``nn.Module`` inside a colocated vLLM ``LLM``.

    AgileRL runs vLLM with ``distributed_executor_backend="external_launcher"``
    (in-process), so the model is directly reachable on the driver worker. The
    attribute path differs slightly across vLLM versions / engine cores; this
    tries the known layouts in order.

    :param llm: A constructed ``vllm.LLM`` instance.
    :type llm: Any
    :return: The underlying model module (e.g. a ``*ForCausalLM`` or
        ``*ForConditionalGeneration``).
    :rtype: nn.Module
    :raises RuntimeError: If the model cannot be located.
    """
    engine = getattr(llm, "llm_engine", getattr(llm, "engine", llm))
    candidates = []
    core = getattr(engine, "engine_core", None)
    if core is not None:
        inner = getattr(core, "engine_core", core)
        candidates.append(inner)
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
    """Resolve a torch dtype from a dtype, ``torch.x`` string, or bare name."""
    if isinstance(value, torch.dtype):
        return value
    name = str(value)
    name = name.removeprefix("torch.")
    return getattr(torch, name)


def _bnb_linear_kwargs(bnb_config: Any) -> dict[str, Any]:
    """Map a ``BitsAndBytesConfig`` to ``Linear4bit`` / ``Params4bit`` kwargs."""
    return {
        "compress_statistics": bnb_config.bnb_4bit_use_double_quant,
        "quant_type": bnb_config.bnb_4bit_quant_type,
        "quant_storage": _resolve_dtype(bnb_config.bnb_4bit_quant_storage),
    }


def _truncate_vocab(weight: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Drop vLLM's padded-vocab rows from an embedding / lm_head weight.

    vLLM pads the vocab dimension up to a hardware-friendly multiple (the real
    rows come first); HF expects exactly ``vocab_size``.
    """
    if vocab_size and weight.shape[0] > vocab_size:
        return weight[:vocab_size]
    return weight


# Sentinel distinguishing "attribute absent" from a stored None in
# ``_navigate_safe`` lookups.
_MISSING = object()


def _navigate(root: Any, dotted: str) -> Any:
    """Walk ``root.<dotted>`` supporting ``ModuleList`` integer indices."""
    cur = root
    if dotted:
        for part in dotted.split("."):
            cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _navigate_safe(root: Any, dotted: str) -> Any:
    """Like :func:`_navigate` but return :data:`_MISSING` if any hop fails.

    Centralises the ``getattr``/index walk and the ``(AttributeError,
    IndexError, KeyError, TypeError)`` guard used everywhere we probe an HF
    target that vLLM may expose but the HF model omits (e.g. k/v projections /
    norms on KV-shared layers, which HF reuses from a paired layer).
    """
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
    """Strip a tensor *subclass* to a plain ``Tensor`` view (shares storage).

    vLLM wraps loaded weights in tensor subclasses (e.g. dense weights are
    ``ModelWeightParameter`` with a custom ``__torch_dispatch__``).
    ``nn.Parameter(subclass)`` raises because the subclass's ``detach()`` returns
    a plain ``Tensor`` of a different type. ``as_subclass`` returns a view that
    aliases the same storage (so zero-copy sharing is preserved) but whose type
    is exactly ``torch.Tensor``, which ``nn.Parameter`` accepts.
    """
    if type(tensor) is torch.Tensor:
        return tensor
    return tensor.as_subclass(torch.Tensor)


def _graft_param(
    model: nn.Module, name: str, tensor: torch.Tensor, requires_grad: bool = False
) -> None:
    """Set ``model.<name>`` to a ``Parameter`` wrapping ``tensor`` in place."""
    parent_path, _, leaf = name.rpartition(".")
    parent = _navigate(model, parent_path)
    param = torch.nn.Parameter(_plain_tensor(tensor), requires_grad=requires_grad)
    setattr(parent, leaf, param)


def _target_exists(model: nn.Module, name: str) -> bool:
    """Whether ``model.<name>`` resolves to an existing attribute / index.

    Used to graft defensively: vLLM may expose modules the HF model does not
    have (e.g. per-layer k/v projections on KV-shared layers, which HF omits
    because they reuse a paired layer's KV). Such vLLM-only weights are skipped
    rather than grafted onto a non-existent target.
    """
    return _navigate_safe(model, name) is not _MISSING


def _quant_target(model: nn.Module, owner: str) -> str | None:
    """Where to graft a shared bnb linear, handling clipping-linear wrappers.

    vLLM exposes the projection as a plain ``Linear`` (``owner``), but the HF
    model may wrap it (e.g. ``Gemma4ClippableLinear``) with the real ``Linear``
    nested at ``owner.linear``. Grafting into ``.linear`` preserves the wrapper
    (so its clipping behaviour is kept) and matches the existing AgileRL
    ``*.linear`` LoRA-target convention.

    :return: The dotted path to replace with the shared ``Linear4bit``, or
        ``None`` if the HF model has no such module (e.g. a KV-shared layer's
        k/v projection).
    """
    module = _navigate_safe(model, owner)
    if module is _MISSING:
        return None
    inner = getattr(module, "linear", None)
    if isinstance(inner, torch.nn.Linear):
        return f"{owner}.linear"
    return owner


def _override_to(module: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
    """No-op ``.to`` so accelerate/DeepSpeed can't re-cast bnb tensors."""
    return module


def _language_model_layout(internal: nn.Module) -> tuple[Any, str, Any, bool]:
    """Locate the text decoder inside a (possibly multimodal) vLLM model.

    :return: ``(decoder_root, hf_prefix, lm_head_module_or_None, multimodal)``
        where ``hf_prefix`` is the HF-side dotted prefix of the decoder
        (``"model"`` for a plain causal-LM, ``"model.language_model"`` for a
        conditional-generation wrapper).
    """
    if hasattr(internal, "language_model"):
        lm = internal.language_model
        root = getattr(lm, "model", lm)
        return root, "model.language_model", getattr(lm, "lm_head", None), True
    if hasattr(internal, "model"):
        return internal.model, "model", getattr(internal, "lm_head", None), False
    msg = (
        "Could not locate the language-model decoder in the vLLM model "
        f"({type(internal).__name__}); weight sharing supports text decoders "
        "(optionally wrapped for conditional generation)."
    )
    raise RuntimeError(msg)


def _emit_quant(
    out: OrderedDict[str, Any],
    hf_name: str,
    packed_weight: torch.Tensor,
    offsets: list[int],
    idx: int,
    quant_states: Any,
) -> None:
    """Record one (possibly fused) bnb shard as an aliased HF weight + state."""
    out[f"{hf_name}.weight"] = packed_weight[offsets[idx] : offsets[idx + 1]]
    out[f"{hf_name}.weight.quant_state"] = quant_states[idx]


def _emit_dense(
    out: OrderedDict[str, Any],
    hf_name: str,
    packed_weight: torch.Tensor,
    offsets: list[int],
    idx: int,
) -> None:
    """Record one (possibly fused) dense shard as an aliased HF weight slice.

    Dense counterpart of :func:`_emit_quant`: the un-quantized fused projection
    is split into HF sub-modules by slicing rows (no ``QuantState`` to carry).
    """
    out[f"{hf_name}.weight"] = packed_weight[offsets[idx] : offsets[idx + 1]]


def _output_size_offsets(module: Any, subs: list[str], pname: str) -> list[int]:
    """Per-shard row offsets for a fused projection, from vLLM ``output_sizes``.

    Used to split a dense fused **weight** and any fused **bias** (which is dense
    even on a quantized base, so it is never carried in ``bnb_shard_offsets``).
    ``QKVParallelLinear`` / ``MergedColumnParallelLinear`` expose ``output_sizes``
    (e.g. ``[q, k, v]`` / ``[gate, up]``); at TP=1 these are the full per-shard
    sizes. Raises if they are absent or don't match the packed sub-module count.
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
    """:func:`extract_vllm_bnb_state_dict` that also returns the resolved layout.

    Walks the vLLM tree once and hands back ``(state_dict, internal, root,
    hf_prefix)`` so callers that also need the layout (build/assert) don't
    re-walk it. See :func:`extract_vllm_bnb_state_dict` for the state-dict
    semantics.
    """
    internal = get_vllm_internal_model(llm)
    root, hf_prefix, lm_head, _multimodal = _language_model_layout(internal)
    packed = getattr(internal, "packed_modules_mapping", None) or {}
    text_config = getattr(hf_config, "text_config", hf_config)
    vocab_size = text_config.vocab_size

    out: OrderedDict[str, Any] = OrderedDict()

    for pname, param in root.named_parameters(recurse=True):
        # Skip vLLM's LoRA adapter slots; we only share the frozen base.
        if "lora" in pname.lower():
            continue
        # vLLM LoRA-wraps each linear, nesting the real weight under
        # ``.base_layer``; the HF model has it un-nested.
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
                        _emit_quant(
                            out, f"{parent_hf}.{sub}", param, offsets, i, quant_states
                        )
                elif n_shards == 1:
                    _emit_quant(
                        out, f"{hf_prefix}.{owner}", param, offsets, 0, quant_states
                    )
                else:
                    msg = (
                        f"Cannot name the {n_shards} fused shards of {pname!r}: no "
                        f"matching packed_modules_mapping entry for {leaf!r}."
                    )
                    raise RuntimeError(msg)
            else:
                # Dense (unquantized) weight. A fused projection
                # (``qkv_proj`` / ``gate_up_proj``) must still be split into its
                # HF sub-modules, but the per-shard sizes come from the vLLM
                # linear's ``output_sizes`` (there is no ``bnb_shard_offsets``).
                param.requires_grad_(False)
                subs = packed.get(leaf)
                if subs is not None and len(subs) > 1:
                    module = _navigate_safe(root, pname[: -len(".weight")])
                    offsets = _output_size_offsets(module, subs, pname)
                    parent_hf = f"{hf_prefix}.{owner.rsplit('.', 1)[0]}"
                    for i, sub in enumerate(subs):
                        _emit_dense(out, f"{parent_hf}.{sub}", param, offsets, i)
                else:
                    weight = param
                    if leaf == "embed_tokens":
                        weight = _truncate_vocab(param, vocab_size)
                    out[f"{hf_prefix}.{owner}.weight"] = weight
        elif name.endswith(".bias"):
            # A fused projection's bias must be split into its HF sub-modules
            # exactly like the weight, using ``output_sizes`` — for BOTH dense
            # and quantized bases (the bias is dense either way and is not in
            # ``bnb_shard_offsets``). Without this, models with attention biases
            # (e.g. Qwen2) lose their q/k/v biases and the forward is wrong.
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
            # Raw parameter (scales, router coefficients, per-layer parameters);
            # carried through unchanged.
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

    Generic module-walk over vLLM's live text decoder, handling both
    bitsandbytes-quantized and dense (bf16/fp16) bases. For each parameter:

    * a bnb-quantized ``.weight`` (has ``bnb_quant_state``) becomes
      ``"{hf}.weight"`` (a slice/view of vLLM storage; shared) plus
      ``"{hf}.weight.quant_state"`` (the bnb ``QuantState`` object). Fused
      projections (``qkv_proj``/``gate_up_proj``) are split into their HF
      sub-modules using vLLM's ``packed_modules_mapping`` and the per-shard
      ``bnb_shard_offsets`` / ``bnb_quant_state``.
    * a dense ``.weight`` (no ``bnb_quant_state``) becomes ``"{hf}.weight"`` (a
      shared view). A dense fused projection is split the same way but using the
      vLLM linear's ``output_sizes`` for the per-shard boundaries.
    * any other bias / raw parameter (embeddings, norms, scales, per-layer
      params) is carried through 1:1 (also shared).

    Names are emitted in HF convention (e.g. ``model.layers.0.self_attn.q_proj``
    or ``model.language_model.layers.0...`` for a multimodal wrapper) so they
    graft directly onto an HF skeleton.

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

    Constructs an empty (meta-device) HF model from ``hf_config`` then grafts
    in vLLM's shared weights (via :func:`extract_vllm_bnb_state_dict`). The
    language decoder holds **no** base weight storage of its own: for a
    quantized base every ``Linear4bit`` (aliased ``Params4bit`` + ``QuantState``)
    and for a dense base every ``nn.Linear.weight`` points at vLLM's GPU memory;
    embeddings / norms / raw params are shared either way. For a multimodal base
    the full skeleton is built (so LoRA adapter names match vLLM's layout) but
    the non-language towers are materialised as frozen, uninitialised
    placeholders (never executed in a text forward).

    :param llm: Constructed ``vllm.LLM`` (in-process engine, base loaded).
    :type llm: Any
    :param hf_config: The HF ``PretrainedConfig`` for the model.
    :type hf_config: Any
    :param compute_dtype: Compute dtype for the bnb ``Linear4bit`` modules and
        the value head (e.g. ``torch.bfloat16``); should match the trainer recipe.
    :type compute_dtype: torch.dtype
    :param bnb_config: The trainer's ``BitsAndBytesConfig`` when the base is
        quantized (used to construct matching ``Linear4bit`` / ``Params4bit``),
        or ``None`` for a dense base.
    :type bnb_config: Any
    :param share_towers: Reserved toggle for sharing vision/audio towers too;
        not implemented in v1 (text-only).
    :type share_towers: bool
    :param add_value_head: Wrap the shared causal-LM in
        ``AutoModelForCausalLMWithValueHead`` (PPO). The base stays shared and
        frozen; the value head is a small trainer-only module (not aliased,
        never used by vLLM rollout).
    :type add_value_head: bool
    :return: An ``nn.Module`` ready for PEFT wrapping.
    :rtype: nn.Module
    :raises NotImplementedError: If ``share_towers`` is True.
    """
    if share_towers:
        msg = (
            "share_towers=True (multimodal tower sharing) is not implemented "
            "yet; v1 shares the language model only."
        )
        raise NotImplementedError(msg)

    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    from agilerl.utils.llm_utils import (
        patch_flex_attention_kernel_options,
        resolve_attn_implementation,
    )

    # Walk the vLLM tree once: the state dict and the decoder layout together.
    shared, _internal, vllm_root, hf_prefix = _extract_with_layout(llm, hf_config)
    device = torch.device("cuda", torch.cuda.current_device())
    is_quantized = bnb_config is not None
    bnb_kwargs = _bnb_linear_kwargs(bnb_config) if is_quantized else {}
    text_config = getattr(hf_config, "text_config", hf_config)

    # Empty skeleton: params live on meta (zero storage); buffers are real so
    # things like Gemma's embed_scale keep their values. Prefer FlashAttention
    # for the long contexts RL rollouts produce — SDPA falls back to the math
    # backend for masked attention and materialises the full TxT score matrix
    # (OOMs at ~30k tokens); flash is O(T). Falls back to SDPA when flash-attn
    # is not installed.
    attn_impl = resolve_attn_implementation(attn_implementation)
    if attn_impl == "flex_attention":  # pragma: no cover - resolved only on CUDA hosts
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
    if quant_owners:  # pragma: no cover - bnb 4-bit path needs CUDA quant states
        from bitsandbytes.nn.modules import Linear4bit, Params4bit
    for (
        owner
    ) in quant_owners:  # pragma: no cover - bnb 4-bit path needs CUDA quant states
        handled.update(
            {f"{owner}.weight", f"{owner}.weight.quant_state", f"{owner}.bias"}
        )
        target = _quant_target(model, owner)
        if target is None:
            # vLLM exposes a quantized module the HF model omits (e.g. k/v on a
            # KV-shared layer); HF reuses a paired layer's KV, so skip it.
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

    # 2. Everything else: embeddings, norms, raw params, untied lm_head, and —
    # for a dense base — every shared linear weight/bias, grafted onto the
    # skeleton's existing nn.Linear. ``.weight``/``.bias`` route through
    # ``_quant_target`` so the ``.linear`` clipping-wrapper redirection (e.g.
    # Gemma4ClippableLinear) and the "HF omits this module" skip apply uniformly.
    for key, val in shared.items():
        if (
            key in handled
        ):  # pragma: no cover - handled keys exist only on the quantized path
            continue
        grafted = False
        for suffix in (".weight", ".bias"):
            if key.endswith(suffix):
                target = _quant_target(model, key[: -len(suffix)])
                if target is not None:
                    _graft_param(model, f"{target}{suffix}", val, requires_grad=False)
                    grafted = True
                break
        else:
            # Raw parameter (no .weight/.bias suffix).
            if _target_exists(
                model, key
            ):  # pragma: no cover - raw extra params occur on GPU multimodal models
                _graft_param(model, key, val, requires_grad=False)
                grafted = True
        if not grafted:
            # vLLM-only parameter the HF model omits (e.g. k/v projections +
            # norms on KV-shared layers); skip rather than graft onto a missing
            # target.
            skipped += 1

    if skipped:
        warnings.warn(
            f"weight_sharing: skipped {skipped} vLLM params with no HF target "
            "(e.g. k/v projections + norms on KV-shared layers, which HF reuses "
            "from a paired layer).",
            stacklevel=2,
        )

    # 2b. Graft persistent checkpoint buffers (e.g. gemma3n per-layer scalars)
    # from vLLM. These are loaded weights, not computed, so the empty skeleton's
    # ``__init__`` defaults are wrong; only the buffers vLLM actually carries are
    # overwritten (computed ones like rotary inv_freq already match).
    for bname, buf in vllm_root.named_buffers(recurse=True):
        if "lora" in bname.lower():
            continue
        hf_name = f"{hf_prefix}.{bname.replace('.base_layer.', '.')}"
        parent_path, _, leaf = hf_name.rpartition(".")
        parent = _navigate_safe(model, parent_path)
        if parent is _MISSING:
            continue
        if leaf in getattr(parent, "_buffers", {}):
            parent._buffers[leaf] = buf.detach()

    # 3. Tie lm_head to the (grafted) embeddings when the config says so.
    if getattr(text_config, "tie_word_embeddings", False):
        model.tie_weights()

    # 4. Move buffers (incl. rotary ``inv_freq``) to the GPU. They were computed
    # correctly at ``from_config`` time (real config, ``include_buffers=False``)
    # and just live on CPU, so a device move suffices — no re-init needed.
    _move_buffers_to_device(model, device)

    # 5. Resolve leftover meta params. Anything still on meta under the language
    # prefix is an un-grafted base param (a coverage bug); the rest are the
    # vision/audio towers, materialised as frozen empties (unused in text RL).
    lang_leftover: list[str] = []
    for name, param in model.named_parameters():
        if param.device.type != "meta":
            continue
        if name.startswith(hf_prefix + ".") or name == "lm_head.weight":
            lang_leftover.append(name)
        else:  # pragma: no cover - non-language meta params occur on multimodal models
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
    if n_towers:  # pragma: no cover - multimodal towers need a GPU model
        warnings.warn(
            f"weight_sharing: materialised {n_towers} non-language buffers as "
            "frozen empties (vision/audio towers; unused in text rollouts).",
            stacklevel=2,
        )

    # Flag as bnb-4-bit so PEFT's kbit prep + get_peft_model treat it as QLoRA.
    # A dense shared base sets none of these (it is not quantized).
    if is_quantized:  # pragma: no cover - bnb 4-bit flags need the CUDA quant path
        model.is_loaded_in_4bit = True
        model.is_loaded_in_8bit = False
        model.is_quantized = True
        model.quantization_method = "bitsandbytes"
        model.config.quantization_config = bnb_config
    # Freeze ``.to`` so accelerate/DeepSpeed can't re-cast the shared base: bnb
    # errors on a 4-bit re-cast, and a dense bf16 ``.to(dtype)`` would copy and
    # silently un-alias it from vLLM.
    model.to = partial(_override_to, model)
    model.eval()

    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    if add_value_head:
        # Share the base; add a small trainer-only value head (PPO critic). The
        # head is not aliased and never needed by vLLM rollout. The inner causal
        # LM keeps its no-op ``.to``; the wrapper's ``.to`` still moves the head.
        from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

        wrapped = AutoModelForCausalLMWithValueHead(model)
        wrapped.v_head.to(device=device, dtype=compute_dtype)
        return wrapped
    return model


def _rewrite_buffers(model: nn.Module, fn: Any) -> int:
    """Walk every (real) buffer and replace it with ``fn(buf)``, when given.

    :param fn: Callback ``(buf) -> tensor | None``; ``None`` leaves the buffer
        unchanged. Never invoked for ``None`` buffer slots.
    :type fn: Any
    :return: Number of buffers actually replaced.
    :rtype: int
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
    bf16/fp16 parameter to fp32. For weight sharing that is harmful: it
    allocates ~14 GiB of transient fp32 copies (OOMs at high
    ``gpu_memory_utilization``, where vLLM already reserves most of the GPU) and
    un-aliases the shared base from vLLM. The base here is frozen and only LoRA
    is trained, so it needs no fp32 master, and bf16 matches vLLM's forward
    (better RL parity). So we replicate the parts that matter — freeze the base
    and enable gradient checkpointing — and skip the upcast entirely, keeping
    the base bf16 and aliased.

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
        # With use_reentrant=True the input embeddings must require grad so the
        # checkpointed graph can backprop into the LoRA adapters; with
        # use_reentrant=False this hack is unnecessary (matches peft's logic).
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
    """Assert the HF trainer base aliases vLLM's storage (no second copy).

    Spot-checks a handful of grafted parameters: their ``data_ptr()`` must
    equal the corresponding vLLM tensor's. Raises if any copy slipped in.

    :param llm: The vLLM ``LLM`` whose tensors were shared.
    :type llm: Any
    :param hf_model: The model returned by :func:`build_shared_hf_model`
        (before PEFT wrapping; pass the unwrapped base if already wrapped).
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
