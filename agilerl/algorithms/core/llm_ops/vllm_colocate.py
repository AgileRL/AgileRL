"""Helpers for a colocated vLLM rollout engine (rollout + HF/PEFT trainer in
one process, sharing the GPU via vLLM native sleep/wake).

vLLM and the trainer each hold their own base; only LoRA adapters are synced
per rollout. These helpers are independent of how the base is loaded:

* :func:`patch_vllm_lora_keep_resident` stops vLLM from zeroing the single
  persistent rollout-adapter slot between forwards.
* :func:`patch_vllm_strip_multimodal_towers` frees the GPU memory held by a
  multimodal base's unused vision/audio towers (text-only RL never runs them).
* :func:`get_vllm_internal_model` reaches the live ``nn.Module`` inside an
  in-process (``external_launcher``) engine.

See ``docs/llm_finetuning/quantization.rst`` for the full colocated picture.
"""

from __future__ import annotations

import gc
import os
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "get_vllm_internal_model",
    "patch_vllm_lora_copy_path",
    "patch_vllm_lora_keep_resident",
    "patch_vllm_strip_multimodal_towers",
]

_COPY_DEBUG_ENV = "AGILERL_VLLM_LORA_COPY_DEBUG"
_DEVICE_SAFE_COPY_ENV = "AGILERL_VLLM_LORA_DEVICE_SAFE_COPY"
_SYNC_RETRY_ENV = "AGILERL_VLLM_LORA_COPY_SYNC_FALLBACK"
_COPY_DEBUG_LIMIT_ENV = "AGILERL_VLLM_LORA_COPY_DEBUG_LIMIT"
_COPY_PATCH_WARNED = False
_COPY_DEBUG_LINES_EMITTED = 0


def _copy_debug_enabled() -> bool:
    return os.environ.get(_COPY_DEBUG_ENV) == "1"


def _device_safe_copy_enabled() -> bool:
    return os.environ.get(_DEVICE_SAFE_COPY_ENV) == "1"


def _sync_retry_enabled() -> bool:
    return os.environ.get(_SYNC_RETRY_ENV) == "1"


def _copy_debug_limit() -> int | None:
    raw = os.environ.get(_COPY_DEBUG_LIMIT_ENV)
    if raw is None:
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    return max(limit, 0)


def _safe_cuda_current_device() -> int | str:
    try:
        if torch.cuda.is_initialized():
            return torch.cuda.current_device()
    except Exception:
        return "error"
    return "n/a"


def _extract_lora_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    for key in ("lora_id", "adapter_id", "lora_int_id"):
        value = kwargs.get(key)
        if value is not None:
            return str(value)
    for arg in args:
        for attr in ("lora_int_id", "adapter_id", "lora_id"):
            value = getattr(arg, attr, None)
            if value is not None:
                return str(value)
    return "n/a"


def _is_tracked_lora_dst(dst: torch.Tensor, module: Any) -> bool:
    tracked = (
        getattr(module, "lora_a_stacked", None),
        getattr(module, "lora_b_stacked", None),
    )
    for tensor in tracked:
        if not torch.is_tensor(tensor):
            continue
        try:
            if dst.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
                return True
        except Exception:
            continue
    return False


def _copy_debug_line(
    module: Any,
    lora_id: str,
    src: Any,
    dst: torch.Tensor,
) -> None:
    global _COPY_DEBUG_LINES_EMITTED
    if not _copy_debug_enabled():
        return
    limit = _copy_debug_limit()
    if limit is not None and _COPY_DEBUG_LINES_EMITTED >= limit:
        return
    pid = os.getpid()
    msg = (
        f"[AGILERL_VLLM_LORA_COPY_DEBUG pid={pid}] "
        f"rank={os.environ.get('RANK', 'unset')} "
        f"local_rank={os.environ.get('LOCAL_RANK', 'unset')} "
        f"world_size={os.environ.get('WORLD_SIZE', 'unset')} "
        f"cuda_current_device={_safe_cuda_current_device()} "
        f"lora_id={lora_id} "
        f"module={type(module).__name__} "
        f"src_device={getattr(src, 'device', 'n/a')} "
        f"src_dtype={getattr(src, 'dtype', 'n/a')} "
        f"src_shape={tuple(src.shape) if torch.is_tensor(src) else 'n/a'} "
        f"src_contig={src.is_contiguous() if torch.is_tensor(src) else 'n/a'} "
        f"dst_device={dst.device} "
        f"dst_dtype={dst.dtype} "
        f"dst_shape={tuple(dst.shape)} "
        f"dst_contig={dst.is_contiguous()}"
    )
    print(msg, flush=True)
    _COPY_DEBUG_LINES_EMITTED += 1


def _copy_debug_failure_line(
    lora_id: str,
    src: Any,
    dst: torch.Tensor,
    exc: Exception,
    non_blocking: Any,
) -> None:
    if not _copy_debug_enabled():
        return

    stream_repr: str = "n/a"
    try:
        if torch.cuda.is_initialized():
            stream = torch.cuda.current_stream()
            stream_id = getattr(stream, "cuda_stream", None)
            stream_repr = (
                f"{stream} cuda_stream={stream_id}"
                if stream_id is not None
                else repr(stream)
            )
    except Exception as stream_exc:
        stream_repr = f"error:{stream_exc}"

    src_is_pinned = "n/a"
    if torch.is_tensor(src):
        is_pinned = getattr(src, "is_pinned", None)
        if callable(is_pinned):
            try:
                src_is_pinned = is_pinned()
            except Exception:
                src_is_pinned = "error"

    msg = (
        f"[AGILERL_VLLM_LORA_COPY_DEBUG_FAIL pid={os.getpid()}] "
        f"rank={os.environ.get('RANK', 'unset')} "
        f"local_rank={os.environ.get('LOCAL_RANK', 'unset')} "
        f"world_size={os.environ.get('WORLD_SIZE', 'unset')} "
        f"cuda_current_device={_safe_cuda_current_device()} "
        f"lora_id={lora_id} "
        f"exception_type={type(exc).__name__} "
        f"exception_message={exc} "
        f"non_blocking={non_blocking} "
        f"src_device={getattr(src, 'device', 'n/a')} "
        f"src_dtype={getattr(src, 'dtype', 'n/a')} "
        f"src_shape={tuple(src.shape) if torch.is_tensor(src) else 'n/a'} "
        f"src_contig={src.is_contiguous() if torch.is_tensor(src) else 'n/a'} "
        f"src_is_pinned={src_is_pinned} "
        f"dst_device={dst.device} "
        f"dst_dtype={dst.dtype} "
        f"dst_shape={tuple(dst.shape)} "
        f"dst_contig={dst.is_contiguous()} "
        f"dst_stride={tuple(dst.stride())} "
        f"dst_storage_offset={dst.storage_offset()} "
        f"dst_is_leaf={dst.is_leaf} "
        f"dst_requires_grad={dst.requires_grad} "
        f"cuda_stream={stream_repr}"
    )
    print(msg, flush=True)


def _copy_debug_recovered_line(
    lora_id: str,
    src: Any,
    dst: torch.Tensor,
    first_exc: Exception,
) -> None:
    if not _copy_debug_enabled():
        return
    msg = (
        f"[AGILERL_VLLM_LORA_COPY_DEBUG_RECOVERED pid={os.getpid()}] "
        f"rank={os.environ.get('RANK', 'unset')} "
        f"local_rank={os.environ.get('LOCAL_RANK', 'unset')} "
        f"world_size={os.environ.get('WORLD_SIZE', 'unset')} "
        f"cuda_current_device={_safe_cuda_current_device()} "
        f"lora_id={lora_id} "
        f"first_exception_type={type(first_exc).__name__} "
        f"first_exception_message={first_exc} "
        f"retry_non_blocking=False "
        f"src_device={getattr(src, 'device', 'n/a')} "
        f"dst_device={dst.device}"
    )
    print(msg, flush=True)


def _copy_debug_retry_fail_line(
    lora_id: str,
    src: Any,
    dst: torch.Tensor,
    retry_exc: Exception,
) -> None:
    if not _copy_debug_enabled():
        return
    msg = (
        f"[AGILERL_VLLM_LORA_COPY_DEBUG_RETRY_FAIL pid={os.getpid()}] "
        f"rank={os.environ.get('RANK', 'unset')} "
        f"local_rank={os.environ.get('LOCAL_RANK', 'unset')} "
        f"world_size={os.environ.get('WORLD_SIZE', 'unset')} "
        f"cuda_current_device={_safe_cuda_current_device()} "
        f"lora_id={lora_id} "
        f"retry_exception_type={type(retry_exc).__name__} "
        f"retry_exception_message={retry_exc} "
        f"retry_non_blocking=False "
        f"src_device={getattr(src, 'device', 'n/a')} "
        f"dst_device={dst.device}"
    )
    print(msg, flush=True)


@contextmanager
def _patch_tensor_copy_for_set_lora(
    module: Any,
    lora_id: str,
):
    original_copy = torch.Tensor.copy_

    def wrapped_copy(dst: torch.Tensor, src: Any, *args: Any, **kwargs: Any) -> Any:
        # Log all tensor copy operations executed inside set_lora so vLLM slice/
        # view-based destinations are captured too (Ray colocated rank>0 debug).
        _copy_debug_line(module, lora_id, src, dst)
        if _device_safe_copy_enabled() and torch.is_tensor(src):
            if src.device != dst.device:
                src = src.to(dst.device, non_blocking=True)
        non_blocking = kwargs.get("non_blocking", args[0] if args else False)
        try:
            return original_copy(dst, src, *args, **kwargs)
        except Exception as exc:
            _copy_debug_failure_line(lora_id, src, dst, exc, non_blocking)
            if not _sync_retry_enabled():
                raise

            retry_src = src
            if torch.is_tensor(retry_src) and retry_src.device != dst.device:
                retry_src = retry_src.to(dst.device, non_blocking=False)
            try:
                result = original_copy(dst, retry_src, non_blocking=False)
            except Exception as retry_exc:
                _copy_debug_retry_fail_line(lora_id, retry_src, dst, retry_exc)
                raise
            _copy_debug_recovered_line(lora_id, retry_src, dst, exc)
            return result

    torch.Tensor.copy_ = wrapped_copy
    try:
        yield
    finally:
        torch.Tensor.copy_ = original_copy


def patch_vllm_lora_copy_path(llm: Any) -> int:
    """Patch vLLM LoRA ``set_lora`` copy path for colocated multi-GPU debug/safety.

    Temporary diagnostics/mitigation for Ray colocated rank>0 LoRA activation
    device mismatches where ``set_lora`` can copy CPU chunks into CUDA buffers
    on a different device. Env-gated and idempotent.
    """
    global _COPY_PATCH_WARNED
    debug = _copy_debug_enabled()
    safe_copy = _device_safe_copy_enabled()
    sync_retry = _sync_retry_enabled()
    if not debug and not safe_copy and not sync_retry:
        return 0
    try:
        model = get_vllm_internal_model(llm)
    except Exception:
        if not _COPY_PATCH_WARNED:
            warnings.warn(
                "colocated init: vLLM LoRA copy patch skipped (model unavailable).",
                stacklevel=2,
            )
            _COPY_PATCH_WARNED = True
        return 0

    patched = 0
    for module in model.modules():
        if not hasattr(module, "set_lora"):
            continue
        if not hasattr(module, "lora_a_stacked") and not hasattr(
            module, "lora_b_stacked"
        ):
            continue
        if getattr(module, "_agilerl_set_lora_copy_patched", False):
            continue
        original = getattr(module, "set_lora", None)
        if original is None or not callable(original):
            continue

        @wraps(original)
        def wrapped_set_lora(
            *args: Any,
            _original: Any = original,
            _module: Any = module,
            **kwargs: Any,
        ) -> Any:
            lora_id = _extract_lora_id(args, kwargs)
            with _patch_tensor_copy_for_set_lora(_module, lora_id):
                return _original(*args, **kwargs)

        module.set_lora = wrapped_set_lora
        module._agilerl_set_lora_copy_patched = True
        patched += 1

    if patched == 0:
        if not _COPY_PATCH_WARNED:
            warnings.warn(
                "colocated init: vLLM LoRA copy patch found no compatible set_lora modules; vLLM internals may have changed.",
                stacklevel=2,
            )
            _COPY_PATCH_WARNED = True

    return patched


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


def get_vllm_internal_model(llm: Any) -> nn.Module:
    """Return the live ``nn.Module`` inside an in-process vLLM ``LLM``.

    The other colocated patches need to mutate vLLM's running model in place —
    :func:`patch_vllm_lora_keep_resident` neutralizes ``reset_lora`` on its LoRA
    layers and :func:`patch_vllm_strip_multimodal_towers` frees its tower
    submodules. vLLM exposes no public accessor, so this walks the known
    engine-core / executor attribute layouts to reach ``...model_runner.model``.

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
        "Could not locate the vLLM internal model. Colocated vLLM requires an "
        "in-process engine (distributed_executor_backend='external_launcher')."
    )
    raise RuntimeError(msg)
