"""Context-parallel primitives for AgileRL LLM training.

Sequence is sharded across a ``cp`` process-group dimension. Attention is
patched (Ulysses or ring) so each rank still sees full-sequence semantics
while activations stay local. FSDP shards parameters over ``dp_shard × cp``
when ``cp > 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

CPStyle = Literal["ulysses", "ring"]


def get_cu_seqlens_from_seq_lens(
    seq_lens: torch.Tensor, total_tokens: int | None = None
) -> tuple[torch.Tensor, int]:
    """Build ``cu_seqlens`` / ``max_seqlen`` from per-document lengths."""
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be 1D, got shape={tuple(seq_lens.shape)}")
    if seq_lens.numel() == 0:
        raise ValueError("seq_lens must not be empty")
    if bool((seq_lens <= 0).any().item()):
        raise ValueError(f"seq_lens must be positive, got {seq_lens.tolist()}")
    if total_tokens is not None and int(seq_lens.sum().item()) != total_tokens:
        raise ValueError(
            f"seq_lens sum must equal sequence length: "
            f"{seq_lens.tolist()} vs {total_tokens}"
        )

    seq_lens_i32 = seq_lens.to(dtype=torch.int32)
    cu_seqlens = torch.empty(
        seq_lens_i32.numel() + 1, dtype=torch.int32, device=seq_lens_i32.device
    )
    cu_seqlens[0] = 0
    cu_seqlens[1:] = seq_lens_i32.cumsum(dim=0, dtype=torch.int32)
    return cu_seqlens, int(seq_lens_i32.max().item())


def shard_for_cp(
    t: torch.Tensor, cp_rank: int, cp_world_size: int, seq_dim: int = 1
) -> torch.Tensor:
    """Shard ``t`` along the sequence dim for context parallelism.

    Requires ``B == 1`` when ``seq_dim == 1`` and ``S % cp_world_size == 0``.
    """
    if seq_dim == 1 and t.shape[0] != 1:
        raise ValueError(
            f"For CP, tensor must have batch dimension 1, got shape={tuple(t.shape)}"
        )
    if t.shape[seq_dim] % cp_world_size != 0:
        raise ValueError(
            f"CP requires sequence dimension {seq_dim} to be divisible by cp size: "
            f"shape={tuple(t.shape)}, cp_size={cp_world_size}; "
            "uneven shards deadlock CP collectives (e.g. ulysses all-to-all)"
        )
    return torch.chunk(t, cp_world_size, dim=seq_dim)[cp_rank]


def shard_position_ids_for_cp(
    position_ids: torch.Tensor, cp_rank: int, cp_world_size: int
) -> torch.Tensor:
    """Shard position ids (2D or packed 3D) for CP."""
    if position_ids.ndim == 3:
        return shard_for_cp(
            position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size, seq_dim=2
        )
    return shard_for_cp(
        position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size, seq_dim=1
    )


def gather_for_cp(t: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """Differentiable all-gather + cat along sequence dim 1."""
    gathered_t = dist_nn.all_gather(t, group=cp_group)
    return torch.cat(gathered_t, dim=1)


def gather_for_cp_wo_grad(
    t: torch.Tensor, cp_world_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Non-differentiable all-gather + cat along sequence dim 1."""
    empty_like_t = [torch.empty_like(t) for _ in range(cp_world_size)]
    dist.all_gather(empty_like_t, t, group=cp_group)
    return torch.cat(empty_like_t, dim=1)


def shift_labels_for_cp(
    input_ids: torch.Tensor,
    *,
    pad_token_id: int = -100,
) -> torch.Tensor:
    """Next-token labels on the *full* sequence before ``shard_for_cp``.

    ``labels[t] = input_ids[t + 1]`` with the final position set to
    ``pad_token_id``. Callers must shard the result; never shift a CP shard
    alone (that drops the boundary token between shards).
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"shift_labels_for_cp expects shape (1, S), got {tuple(input_ids.shape)}"
        )
    labels = torch.empty_like(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_token_id
    return labels


def pad_seq_to_cp_multiple(
    t: torch.Tensor,
    cp_world_size: int,
    *,
    seq_dim: int = 1,
    pad_value: int | float = 0,
    ring_zigzag: bool = False,
) -> torch.Tensor:
    """Pad sequence length to a multiple of ``cp`` (or ``2 * cp`` for ring zigzag)."""
    divisor = 2 * cp_world_size if ring_zigzag else cp_world_size
    length = t.shape[seq_dim]
    remainder = length % divisor
    if remainder == 0:
        return t
    pad_len = divisor - remainder
    pad_shape = list(t.shape)
    pad_shape[seq_dim] = pad_len
    pad = t.new_full(pad_shape, pad_value)
    return torch.cat([t, pad], dim=seq_dim)


def _has_linear_attn_layer(model: nn.Module) -> bool:
    """True if the model contains any non-softmax (linear/SSM) attention layer."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    layers = getattr(inner, "layers", None)
    if layers is None:
        return False
    layer_modules = layers.modules() if isinstance(layers, nn.Module) else layers
    for layer in layer_modules:
        if getattr(layer, "layer_type", None) == "linear_attention":
            return True
        if hasattr(layer, "mamba"):
            return True
    return False


def assert_cp_style_supports_model(cp_style: CPStyle, model: nn.Module) -> None:
    """Refuse ``cp_style='ring'`` on linear-attn / Mamba hybrids."""
    if cp_style == "ring" and _has_linear_attn_layer(model):
        raise ValueError(
            "cp_style='ring' is not supported for models with linear-attention "
            "or Mamba/SSM layers. Use cp_style='ulysses' instead."
        )


def assert_ulysses_head_divisibility(
    *,
    num_attention_heads: int,
    num_key_value_heads: int | None,
    cp: int,
) -> None:
    """Validate Ulysses head / GQA constraints for ``cp``."""
    if num_attention_heads % cp != 0:
        raise ValueError(
            f"Ulysses CP requires num_attention_heads ({num_attention_heads}) "
            f"divisible by cp ({cp})"
        )
    h_kv = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
    if h_kv % cp != 0 and cp % h_kv != 0:
        raise ValueError(
            f"Ulysses GQA requires num_key_value_heads ({h_kv}) % cp ({cp}) == 0 "
            f"or cp % num_key_value_heads == 0 (KV replicate path); got neither"
        )


def flash_attn_available() -> bool:
    """Return True when FA2's varlen entrypoint is importable."""
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        return True
    except ImportError:
        return False


def ring_flash_attn_available() -> bool:
    """Return True when ``ring_flash_attn`` is importable (after compat shim)."""
    try:
        # Compat must run before ring_flash_attn imports transformers helpers.
        import agilerl.utils.ring_attn_compat  # noqa: F401
        import ring_flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def validate_cp_config(
    *,
    cp: int,
    cp_style: str,
    fsdp_config: Any | None,
    world_size: int,
    use_liger_loss: bool = False,
    packing_mode: str | None = None,
    attn_implementation: str | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    check_flash_attn: bool = True,
) -> CPStyle:
    """Fail loud on illegal CP configurations before NCCL work.

    Returns the normalized ``cp_style`` (ignored when ``cp == 1``).
    """
    if cp < 1:
        raise ValueError(f"cp must be >= 1, got {cp}")
    if cp == 1:
        return "ulysses"

    if fsdp_config is None:
        raise ValueError(
            "cp > 1 requires fsdp_config. Context parallel without FSDP would "
            "keep a full weight replica on every rank."
        )
    if world_size % cp != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by cp ({cp})"
        )
    if cp_style not in ("ulysses", "ring"):
        raise ValueError(
            f"cp_style must be 'ulysses' or 'ring', got {cp_style!r}"
        )
    style: CPStyle = cp_style  # type: ignore[assignment]

    if use_liger_loss:
        raise ValueError(
            "Liger fused CE is not supported with cp > 1 until logprob "
            "gather / shape contracts are defined."
        )
    if packing_mode == "flex":
        raise ValueError(
            "flex-attention packing is not supported with cp > 1; use FA2 packing "
            "or disable packing."
        )
    if check_flash_attn and not flash_attn_available():
        raise ValueError(
            "cp > 1 requires flash-attn (FA2). Install the flash-attn package "
            "compatible with your torch/CUDA build."
        )
    if attn_implementation is not None and attn_implementation not in (
        "flash_attention_2",
        "auto",
    ):
        raise ValueError(
            f"cp > 1 requires attn_implementation='flash_attention_2', "
            f"got {attn_implementation!r}"
        )
    if style == "ring" and not ring_flash_attn_available():
        raise ValueError(
            "cp_style='ring' requires ring-flash-attn>=0.1.8. "
            "Install it or use cp_style='ulysses'."
        )
    if style == "ulysses" and num_attention_heads is not None:
        assert_ulysses_head_divisibility(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            cp=cp,
        )
    return style


def setup_cp_attention_params(
    *,
    seq_lens: torch.Tensor,
    total_tokens: int,
    cp_group: dist.ProcessGroup,
    cp_style: CPStyle = "ulysses",
    device: torch.device | None = None,
) -> None:
    """Publish full-sequence attention params for the active CP style."""
    cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
        seq_lens.to(device=device) if device is not None else seq_lens,
        total_tokens=total_tokens,
    )
    if cp_style == "ulysses":
        from agilerl.utils.ulysses_attn import update_ulysses_params

        update_ulysses_params(cu_seqlens, max_seqlen)
    elif cp_style == "ring":
        import agilerl.utils.ring_attn_compat  # noqa: F401
        from ring_flash_attn import update_ring_flash_attn_params
        from ring_flash_attn.adapters.hf_adapter import use_ring_attn

        update_ring_flash_attn_params(cu_seqlens, cp_group)
        # Train forwards publish params then enable; generate keeps this False.
        use_ring_attn(True)
    else:
        raise ValueError(f"Unknown cp_style: {cp_style}")


def disable_cp_attention_params(cp_style: CPStyle = "ulysses") -> None:
    """Clear / disable CP attention so the next forward uses stock FA2."""
    if cp_style == "ulysses":
        from agilerl.utils.ulysses_attn import clear_ulysses_params

        clear_ulysses_params()
    elif cp_style == "ring":
        try:
            import agilerl.utils.ring_attn_compat  # noqa: F401
            from ring_flash_attn.adapters.hf_adapter import use_ring_attn

            use_ring_attn(False)
        except ImportError:
            pass
    else:
        raise ValueError(f"Unknown cp_style: {cp_style}")


def setup_cp_params(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
    cp_group: dist.ProcessGroup,
    *,
    seq_lens: torch.Tensor,
    cp_style: CPStyle = "ulysses",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish attention params and return sequence-sharded ids / positions."""
    total_tokens = int(position_ids.shape[-1])
    setup_cp_attention_params(
        seq_lens=seq_lens,
        total_tokens=total_tokens,
        cp_group=cp_group,
        cp_style=cp_style,
        device=position_ids.device,
    )
    input_ids = shard_for_cp(input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    position_ids = shard_position_ids_for_cp(
        position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size
    )
    return input_ids, position_ids


def substitute_cp_attention(cp_style: CPStyle, cp_group: dist.ProcessGroup) -> None:
    """Install the HF attention monkeypatch for ``cp_style`` (once per process)."""
    if cp_style == "ulysses":
        from agilerl.utils.ulysses_attn import substitute_hf_ulysses_attn

        substitute_hf_ulysses_attn(cp_group)
    elif cp_style == "ring":
        import agilerl.utils.ring_attn_compat  # noqa: F401
        from ring_flash_attn import substitute_hf_flash_attn
        from ring_flash_attn.adapters.hf_adapter import use_ring_attn

        # llama3-style ring FA2; heads_k_stride=1 matches Prime-RL HF path.
        substitute_hf_flash_attn(cp_group, heads_k_stride=1)
        # Default off until ``setup_cp_attention_params`` publishes DATA_PARAMS
        # (rollouts / HF generate stay on stock FA2).
        use_ring_attn(False)
    else:
        raise ValueError(f"Unknown cp_style: {cp_style}")


@dataclass
class ParallelDims:
    """Minimal CP/FSDP mesh planner (no EP/PP/dp_replicate).

    Topology: ``dp_shard * cp == world_size``.
    """

    dp_shard: int
    cp: int
    world_size: int
    _world_mesh: DeviceMesh | None = field(default=None, repr=False, compare=False)
    _submeshes: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.cp < 1:
            raise ValueError(f"cp must be >= 1, got {self.cp}")
        if self.dp_shard < 1:
            raise ValueError(f"dp_shard must be >= 1, got {self.dp_shard}")
        if self.dp_shard * self.cp != self.world_size:
            raise ValueError(
                f"Invalid parallel dims: dp_shard({self.dp_shard}) * cp({self.cp}) "
                f"!= WORLD_SIZE({self.world_size})"
            )

    @classmethod
    def from_world(cls, world_size: int, cp: int = 1) -> ParallelDims:
        """Build dims with ``dp_shard = world_size // cp``."""
        if cp < 1:
            raise ValueError(f"cp must be >= 1, got {cp}")
        if world_size % cp != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by cp ({cp})"
            )
        return cls(dp_shard=world_size // cp, cp=cp, world_size=world_size)

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def dp_size(self) -> int:
        """Batch / sampler world (excludes CP)."""
        return self.dp_shard

    @property
    def seq_len_divisor(self) -> int:
        """Preferred pad multiple: ``cp`` for Ulysses, ``2 * cp`` for ring zigzag."""
        return self.cp * 2 if self.cp > 1 else 1

    def build_mesh(self, device_type: str | None = None) -> DeviceMesh | None:
        """Build a ``DeviceMesh`` when ``cp > 1``; return ``None`` for the flat path."""
        if not self.cp_enabled:
            return None
        from torch._utils import _get_available_device_type
        from torch.distributed.device_mesh import init_device_mesh

        dtype = device_type or _get_available_device_type() or "cuda"
        mesh = init_device_mesh(
            dtype, (self.dp_shard, self.cp), mesh_dim_names=("dp_shard", "cp")
        )
        self._submeshes["dp"] = mesh["dp_shard"]
        self._submeshes["cp"] = mesh["cp"]
        self._submeshes["dp_shard_cp"] = mesh._flatten(mesh_dim_name="dp_shard_cp")
        self._submeshes["dp_cp"] = self._submeshes["dp_shard_cp"]
        self._submeshes["hsdp"] = self._submeshes["dp_shard_cp"]
        self._world_mesh = mesh
        return mesh

    @property
    def world_mesh(self) -> DeviceMesh | None:
        if self._world_mesh is None and self.cp_enabled:
            self.build_mesh()
        return self._world_mesh

    def get_mesh(self, name: str) -> DeviceMesh:
        if self.world_mesh is None:
            raise RuntimeError("No DeviceMesh when cp == 1 (flat process-group path)")
        if name in self._submeshes:
            return self._submeshes[name]
        return self.world_mesh[name]

    def cp_group(self) -> dist.ProcessGroup:
        return self.get_mesh("cp").get_group()

    def cp_rank(self) -> int:
        return dist.get_rank(self.cp_group())
