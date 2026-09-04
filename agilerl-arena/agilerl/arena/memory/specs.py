# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Input specifications for the GPU memory estimator.

Everything in this module is plain data: pydantic models that describe the
model architecture, the device, and the user-facing settings. The specs are
JSON-serializable so the same objects can back the Arena widget, the CLI, and the arena pre-submission gate. Nothing here imports
torch — the calculation core must stay portable to a browser runtime.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

GiB = 1024**3
MiB = 1024**2

#: Fallback CUDA context for a device whose own has not been measured. Set
#: between the two devices measured so far (A100 501 MiB, L4 226 MiB) rather
#: than at either, so an unknown device is wrong by less in both directions.
#: Prefer ``DeviceSpec.cuda_context_bytes``.
CUDA_CONTEXT_BYTES_DEFAULT = 384 * MiB

#: Contexts measured per device, keyed by ``torch.cuda.get_device_name``.
#: Read from NVML either side of a one-element allocation, torch 2.11/cu130.
#: Compute capability does not predict this — the newer L4 (sm_89) costs less
#: than half the older A100 (sm_80) — so it is a lookup, not a formula.
MEASURED_CUDA_CONTEXT_BYTES: dict[str, int] = {
    "NVIDIA A100-SXM4-40GB": 501 * MiB,
    "NVIDIA A100-SXM4-80GB": 501 * MiB,
    "NVIDIA L4": 226 * MiB,
}

#: Bytes per element for the dtypes the estimator reasons about. Sub-byte
#: quantization formats (nf4) are handled via bytes-per-param factors in
#: :mod:`agilerl.arena.memory.formulas`, not through this table.
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
}

WeightDtype = Literal["fp32", "bf16", "fp16"]
AttnImplementation = Literal[
    "auto", "eager", "sdpa", "flash_attention_2", "flex_attention"
]
KVCacheDtype = Literal["auto", "fp8", "int8"]
#: How vLLM caches the recurrent state of a hybrid model's Mamba layers.
MambaCacheMode = Literal["none", "align", "all"]
QuantizationMethod = Literal["none", "nf4", "int8", "awq", "gptq"]
#: What bitsandbytes can apply to the trainer copy, a subset of the engine-side
#: methods (AWQ and GPTQ are load-time formats, not runtime quantizers).
TrainerQuantization = Literal["none", "nf4", "int8"]
Algorithm = Literal["grpo", "gspo", "cispo", "ppo", "reinforce", "dpo", "sft"]
LoraTargetScope = Literal["all-linear", "attention-only"]
PackedMoeDispatch = Literal["materialized", "contracted"]
DistributedBackend = Literal["none", "deepspeed"]


def _encoder_params(cfg: dict[str, Any]) -> int:
    """Rough parameter count of a transformer encoder from its config."""
    h = int(cfg.get("hidden_size") or 0)
    layers = int(cfg.get("num_hidden_layers") or 0)
    if not h or not layers:
        return 0
    heads = int(cfg.get("num_attention_heads") or 1)
    head_dim = int(cfg.get("head_dim") or (h // heads if heads else 0))
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    intermediate = int(cfg.get("intermediate_size") or 4 * h)
    attn = h * heads * head_dim + 2 * h * kv_heads * head_dim + heads * head_dim * h
    per_layer = attn + 2 * h * intermediate + 2 * h
    return layers * per_layer + int(cfg.get("position_embedding_size") or 0) * h


def multimodal_tower_params(config: dict[str, Any]) -> int:
    """Approximate parameters in vision and audio towers.

    A known lower bound: the formula assumes plain transformer encoders,
    while real towers are convolutional (~45% short on Gemma 4). A profiled
    ``realised_bytes`` supersedes this whenever one exists.
    """
    return sum(
        _encoder_params(config[key])
        for key in ("vision_config", "audio_config")
        if isinstance(config.get(key), dict)
    )


def _sliding_layer_fraction(text_cfg: dict[str, Any]) -> float:
    """Fraction of layers using windowed attention.

    Hybrid models interleave local and global attention and declare it per
    layer (Gemma 4 E2B: 28 sliding to 7 full, i.e. 0.8). Only the windowed
    layers cap their KV growth, so the ratio is a first-order term on the
    largest inference cost. ``sliding_window_pattern`` is the older integer
    form: one full-attention layer every N.
    """
    layer_types = text_cfg.get("layer_types")
    if layer_types:
        windowed = sum(1 for entry in layer_types if "sliding" in str(entry))
        return windowed / len(layer_types)
    pattern = text_cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 1:
        return (pattern - 1) / pattern
    return 1.0


def _layer_type_list(text_cfg: dict[str, Any]) -> list[str]:
    """The per-layer type list, under either of its two spellings.

    ``layer_types`` is the transformers convention (Granite 4 H);
    Nemotron 3.5 writes the same list as ``layers_block_type``.
    """
    entries = text_cfg.get("layer_types") or text_cfg.get("layers_block_type") or []
    return [str(entry).lower() for entry in entries]


def _layer_mix(text_cfg: dict[str, Any], n_layers: int) -> tuple[int | None, int]:
    """(attention layers, recurrent layers) for a hybrid state-space model.

    Three spellings in the wild, plus a fourth layout. ``layer_types`` (or
    Nemotron 3.5's ``layers_block_type``) lists them per layer; Nemotron-H's
    ``hybrid_override_pattern`` is a string where ``*`` is attention, ``M`` a
    Mamba mixer and ``-`` a plain MLP block; Qwen3-Next declares the ratio as
    ``full_attention_interval``.

    Falcon-H1 is the fourth: it states no layout at all because it has none —
    attention and the SSM run in *parallel* inside every block, so both counts
    are ``n_layers`` and the model pays for both a KV cache and a Mamba state.
    """
    entries = _layer_type_list(text_cfg)
    if entries:
        recurrent = sum(1 for e in entries if "mamba" in e or "linear" in e)
        if recurrent:
            return sum(1 for e in entries if "attention" in e), recurrent

    pattern = text_cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str) and "M" in pattern:
        return pattern.count("*"), pattern.count("M")

    every = text_cfg.get("full_attention_interval")
    if isinstance(every, int) and every > 1:
        attention = n_layers // every
        return attention, n_layers - attention

    if (
        any(key.startswith("mamba_") for key in text_cfg)
        or "ssm_state_size" in text_cfg
    ):
        return n_layers, n_layers
    return None, 0


def _ffn_layer_mix(text_cfg: dict[str, Any]) -> tuple[int | None, int | None]:
    """(dense-MLP layers, MoE layers) when a layer is its FFN *instead of* a
    mixer.

    Most stacks put an FFN in every block alongside the mixer, and both counts
    stay ``None`` (= every layer). Nemotron-H is block-exclusive: each layer is
    exactly one of Mamba, attention, MLP or MoE, declared per layer in
    ``layers_block_type`` ("moe" / "mlp" entries) or as ``-`` in
    ``hybrid_override_pattern``. Only those spellings set explicit counts —
    a layer list of mixers alone (Granite 4 H) says nothing about the FFNs.
    """
    entries = _layer_type_list(text_cfg)
    if entries:
        moe = sum(1 for e in entries if e == "moe")
        mlp = sum(1 for e in entries if e in ("mlp", "-"))
        if moe or mlp:
            return mlp, moe

    pattern = text_cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str) and "M" in pattern and "-" in pattern:
        return pattern.count("-"), None
    return None, None


class MambaGeometry(TypedDict, total=False):
    """The recurrent-state keyword arguments ``_mamba_state_geometry`` fills."""

    mamba_d_state: int
    mamba_d_conv: int
    mamba_n_groups: int
    mamba_n_heads: int
    mamba_d_head: int
    mamba_conv_dim: int


def _mamba_state_geometry(text_cfg: dict[str, Any], hidden: int) -> MambaGeometry:
    """Mamba-2 / gated-delta-net state dimensions, normalised to one shape.

    vLLM stores two tensors per recurrent layer per slot: a causal-conv window
    of ``(conv_dim, d_conv - 1)`` and a recurrent state of
    ``(n_heads, d_head, d_state)``. The two families disagree only on how
    ``conv_dim`` is built, so that is resolved here and everything downstream
    uses the same arithmetic.
    """
    if text_cfg.get("linear_key_head_dim"):  # gated-delta-net (Qwen3-Next)
        k_dim = int(text_cfg["linear_key_head_dim"])
        v_dim = int(text_cfg["linear_value_head_dim"])
        k_heads = int(text_cfg["linear_num_key_heads"])
        v_heads = int(text_cfg["linear_num_value_heads"])
        return {
            "mamba_d_state": k_dim,
            "mamba_d_conv": int(text_cfg.get("linear_conv_kernel_dim") or 0),
            "mamba_n_groups": 0,
            "mamba_n_heads": v_heads,
            "mamba_d_head": v_dim,
            "mamba_conv_dim": 2 * k_heads * k_dim + v_heads * v_dim,
        }

    d_state = int(
        text_cfg.get("mamba_d_state")
        or text_cfg.get("ssm_state_size")
        or text_cfg.get("mamba_state_dim")
        or 0
    )
    n_groups = int(
        text_cfg.get("mamba_n_groups")
        or text_cfg.get("mamba_num_groups")
        or text_cfg.get("n_groups")
        or 0
    )
    n_heads = int(text_cfg.get("mamba_n_heads") or text_cfg.get("mamba_num_heads") or 0)
    d_head = int(text_cfg.get("mamba_d_head") or text_cfg.get("mamba_head_dim") or 0)
    d_inner = int(
        text_cfg.get("mamba_d_ssm")
        or n_heads * d_head
        or int(text_cfg.get("mamba_expand") or text_cfg.get("expand") or 2) * hidden
    )
    return {
        "mamba_d_state": d_state,
        "mamba_d_conv": int(
            text_cfg.get("mamba_d_conv") or text_cfg.get("conv_kernel") or 0
        ),
        "mamba_n_groups": n_groups,
        "mamba_n_heads": n_heads,
        "mamba_d_head": d_head,
        "mamba_conv_dim": d_inner + 2 * n_groups * d_state,
    }


class ModelArch(BaseModel):
    """Dense (or MoE) decoder-only transformer geometry.

    Field names follow HF ``config.json`` semantics; use
    :meth:`from_hf_config` to build one from a raw config dict.
    """

    model_config = ConfigDict(frozen=True)

    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool = False
    #: Sliding-window size when the model uses windowed attention. Caps KV
    #: growth per sequence on the windowed layers.
    sliding_window: int | None = None
    #: Fraction of layers using the sliding window (1.0 for fully windowed
    #: models, e.g. 0.75 for hybrid layouts). Ignored when
    #: ``sliding_window`` is None.
    sliding_window_layer_fraction: float = 1.0
    #: Whether attention projections carry biases (Qwen2-style qkv bias).
    attn_bias: bool = False
    #: Gated MLP (SwiGLU-style: gate + up + down). All mainstream decoder
    #: models qualify; ungated MLPs use two matrices instead of three.
    gated_mlp: bool = True

    # MoE geometry — None for dense models. Weights/optimizer scale with the
    # total expert count; activations scale with the active (routed) count.
    n_experts: int | None = None
    n_experts_per_tok: int | None = None
    expert_intermediate_size: int | None = None
    n_shared_experts: int = 0
    #: Shared-expert FFN width when it differs from the routed experts'
    #: (Nemotron 3.5: 3712 against 1856). ``None`` falls back to
    #: ``expert_intermediate_size``.
    shared_expert_intermediate_size: int | None = None

    # --- Block-exclusive layer layout (Nemotron-H) -------------------------
    # In most stacks every block carries a mixer *and* an FFN; Nemotron-H
    # layers are exactly one of Mamba, attention, MLP or MoE. ``None`` keeps
    # the every-layer default; explicit counts (from ``layers_block_type`` or
    # ``hybrid_override_pattern``'s ``-`` entries) confine each parameter
    # group to the layers that actually hold it.
    #: Layers carrying a dense MLP, when the FFN is a block of its own.
    n_mlp_layers: int | None = None
    #: Layers carrying the routed experts, when MoE is a block of its own.
    n_moe_layers: int | None = None

    #: Parameter count of multimodal towers (vision/audio encoders and
    #: projectors) when the checkpoint carries them. Tower geometry varies too
    #: much to model analytically; profiled realised sizes refine this.
    multimodal_tower_params: int = 0

    # --- Gemma-4-style geometry -------------------------------------------
    # Three features that break the "one width per layer" assumption the rest
    # of this class makes. Each was measured to matter; see the formulas.
    #: Head dim on the *full*-attention layers when it differs from the
    #: sliding layers' (Gemma 4: 512 against 256), which widens q/k/v on
    #: exactly the layers that are not windowed.
    global_head_dim: int | None = None
    #: Trailing layers that reuse an earlier layer's KV instead of storing
    #: their own, so they add nothing to the cache.
    n_kv_shared_layers: int = 0
    #: Whether the KV-sharing layers carry a 2x-wide MLP (Gemma 4 E2B). Note
    #: this applies only to those layers, not the whole model.
    double_wide_mlp: bool = False
    #: Per-Layer Embeddings: a per-layer input vector of this width, looked up
    #: per token and fed into every layer. Both a large parameter block
    #: (``vocab_size_per_layer_input x n_layers x this``) and a live
    #: activation of ``tokens x n_layers x this`` — which for Gemma 4 E2B is
    #: 8960 per token, wider than hidden_size itself.
    per_layer_input_dim: int = 0
    per_layer_input_vocab: int = 0

    # Hybrid SSM: most attention layers are Mamba-2 mixers. A Mamba
    # layer's per-sequence state is constant; attention KV grows with
    # context. Nemotron-Nano-9B-v2 has 4 attention layers of 56, so
    # treating them all as attention overstates KV 14x.
    #: Attention layers, when fewer than ``n_layers``. None means all of them.
    n_attention_layers: int | None = None
    #: Mamba-2 mixer layers.
    n_mamba_layers: int = 0
    #: Mamba-2 state geometry (vLLM ``mamba2_state_shape``).
    mamba_d_state: int = 0
    mamba_d_conv: int = 0
    mamba_n_groups: int = 0
    mamba_n_heads: int = 0
    mamba_d_head: int = 0
    #: Dtype of the recurrent (SSM) state. vLLM resolves this *per model*, not
    #: from the config: Nemotron-H keeps it in fp32 while Falcon-H1 keeps it in
    #: the model dtype, and the two were confirmed by asking vLLM's own
    #: ``get_mamba_state_dtype_from_config``. fp32 is the default because it is
    #: the larger of the two and a memory estimate must not read low.
    mamba_ssm_state_dtype: WeightDtype = "fp32"
    #: Width of the mixer's causal conv, which is family-specific: Mamba-2 uses
    #: ``d_inner + 2 * n_groups * d_state``, gated-delta-net (Qwen3-Next)
    #: ``2 * k_heads * k_dim + v_heads * v_dim``. Stored rather than derived so
    #: one state formula covers both.
    mamba_conv_dim: int = 0

    @property
    def is_hybrid_ssm(self) -> bool:
        return self.n_mamba_layers > 0

    @property
    def attention_layers(self) -> int:
        """Layers that actually hold a KV cache."""
        if self.n_attention_layers is not None:
            return self.n_attention_layers
        return self.n_layers

    @property
    def mlp_width_factor(self) -> float:
        """Mean intermediate width relative to ``intermediate_size``.

        Gemma 4 E2B doubles the MLP on its KV-shared layers only, so the flat
        ``intermediate_size`` understates the model: 15 layers at 6144 and 20
        at 12288 average 9655, a factor of 1.57.
        """
        if not self.double_wide_mlp or not self.n_kv_shared_layers:
            return 1.0
        wide = min(self.n_kv_shared_layers, self.n_layers)
        return (self.n_layers + wide) / self.n_layers

    @property
    def peak_mlp_width_factor(self) -> float:
        """The *widest single block's* intermediate width relative to
        ``intermediate_size``.

        Per-block transients — the checkpoint recompute, one block's fp32
        LoRA input casts, the engine's layer-at-a-time forward — peak at the
        widest block, not the mean one. On Gemma 4 E2B the 20 KV-shared
        layers carry a 2x MLP (``use_double_wide_mlp``), and four of them
        (19/24/29/34 in ``layer_types``) are also global-attention layers, so
        a block realising both maxima exists. Parameter totals keep using the
        mean (:attr:`mlp_width_factor`); the sum over layers is what they
        integrate.
        """
        if not self.double_wide_mlp or not self.n_kv_shared_layers:
            return 1.0
        return 2.0

    @property
    def mean_qkv_dim(self) -> int:
        """q+k+v width per token, averaged over layer types."""
        narrow = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        if not self.global_head_dim or self.global_head_dim == self.head_dim:
            return narrow
        wide = (self.n_heads + 2 * self.n_kv_heads) * self.global_head_dim
        # ``sliding_window_layer_fraction`` is the share using the narrow head.
        f = self.sliding_window_layer_fraction if self.sliding_window else 0.0
        return int(f * narrow + (1.0 - f) * wide)

    @property
    def peak_qkv_dim(self) -> int:
        """q+k+v width of the widest block — the global-attention head width
        wherever ``global_head_dim`` exceeds ``head_dim``. The per-block
        counterpart of :attr:`mean_qkv_dim`, for the same reason as
        :attr:`peak_mlp_width_factor`.
        """
        if not self.global_head_dim or self.global_head_dim == self.head_dim:
            return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        return (self.n_heads + 2 * self.n_kv_heads) * self.global_head_dim

    @property
    def is_moe(self) -> bool:
        return self.n_experts is not None and self.n_experts > 1

    @property
    def block_exclusive_layers(self) -> bool:
        """Whether each layer is one block kind rather than mixer + FFN."""
        return self.n_mlp_layers is not None or self.n_moe_layers is not None

    @property
    def moe_layers(self) -> int:
        """Layers carrying the routed experts (all of them by default)."""
        if self.n_moe_layers is not None:
            return self.n_moe_layers
        return self.n_layers

    @property
    def mlp_layers(self) -> int:
        """Layers carrying a dense MLP.

        Every layer by default; 0 on a MoE model unless a block-exclusive
        layout says otherwise, matching how the counting always treated MoE
        stacks (routed experts replace the dense FFN).
        """
        if self.n_mlp_layers is not None:
            return self.n_mlp_layers
        return 0 if self.is_moe else self.n_layers

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> Self:
        """Build a :class:`ModelArch` from a raw HF ``config.json`` dict.

        Multimodal configs that nest the language model under ``text_config``
        are unwrapped; the tower parameter count is left at 0 (profiling or an
        explicit override fills it in).
        """
        text_cfg = config.get("text_config", config)
        n_heads = int(text_cfg["num_attention_heads"])
        hidden = int(text_cfg["hidden_size"])
        n_kv = int(text_cfg.get("num_key_value_heads") or n_heads)
        moe_experts = (
            text_cfg.get("num_experts")
            or text_cfg.get("num_local_experts")
            or text_cfg.get("n_routed_experts")
        )
        n_layers = int(text_cfg["num_hidden_layers"])
        n_attention, n_mamba = _layer_mix(text_cfg, n_layers)
        n_mlp, n_moe = _ffn_layer_mix(text_cfg)
        mamba: MambaGeometry = (
            _mamba_state_geometry(text_cfg, hidden) if n_mamba else {}
        )
        # Gatedness decides the FFN matrix count (SwiGLU: gate+up+down;
        # Nemotron-H's squared-ReLU: up+down). Only relu-family activations
        # in the wild are ungated.
        act = str(
            text_cfg.get("mlp_hidden_act") or text_cfg.get("hidden_act") or ""
        ).lower()
        gated = act not in {"relu", "relu2", "relu_squared", "squared_relu"}
        return cls(
            n_layers=n_layers,
            n_attention_layers=n_attention,
            n_mamba_layers=n_mamba,
            n_mlp_layers=n_mlp,
            n_moe_layers=n_moe,
            gated_mlp=gated,
            **mamba,
            hidden_size=hidden,
            intermediate_size=int(text_cfg["intermediate_size"]),
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=int(text_cfg.get("head_dim") or hidden // n_heads),
            vocab_size=int(text_cfg["vocab_size"]),
            tied_embeddings=bool(text_cfg.get("tie_word_embeddings", False)),
            sliding_window=(
                text_cfg.get("sliding_window")
                if text_cfg.get("use_sliding_window", True)
                else None
            ),
            sliding_window_layer_fraction=_sliding_layer_fraction(text_cfg),
            attn_bias=bool(
                text_cfg.get("attention_bias", False) or text_cfg.get("qkv_bias", False)
            ),
            global_head_dim=text_cfg.get("global_head_dim") or None,
            n_kv_shared_layers=int(text_cfg.get("num_kv_shared_layers") or 0),
            double_wide_mlp=bool(text_cfg.get("use_double_wide_mlp", False)),
            per_layer_input_dim=int(text_cfg.get("hidden_size_per_layer_input") or 0),
            per_layer_input_vocab=int(text_cfg.get("vocab_size_per_layer_input") or 0),
            multimodal_tower_params=multimodal_tower_params(config),
            n_experts=int(moe_experts) if moe_experts else None,
            n_experts_per_tok=(
                int(text_cfg["num_experts_per_tok"])
                if text_cfg.get("num_experts_per_tok")
                else None
            ),
            expert_intermediate_size=(
                int(text_cfg["moe_intermediate_size"])
                if text_cfg.get("moe_intermediate_size")
                else None
            ),
            n_shared_experts=int(text_cfg.get("n_shared_experts") or 0),
            shared_expert_intermediate_size=(
                int(text_cfg["moe_shared_expert_intermediate_size"])
                if text_cfg.get("moe_shared_expert_intermediate_size")
                else None
            ),
        )


class WeightVariant(BaseModel):
    """One loadable form of a model's weights.

    Quantization and multimodal-tower stripping produce discrete variants of
    the same checkpoint with different realised in-memory sizes. The analytic
    size is always computable; ``realised_bytes`` is the profiled ground truth
    and takes precedence when present (scales, zero points, and
    higher-precision held-out layers all cut into the nominal saving).
    """

    model_config = ConfigDict(frozen=True)

    name: str = "base"
    quantization: QuantizationMethod = "none"
    stripped_multimodal: bool = False
    realised_bytes: int | None = None


class ModelSpec(BaseModel):
    """A curated model: geometry plus its profiled weight variants."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    arch: ModelArch
    checkpoint_dtype: WeightDtype = "bf16"
    #: Exact parameter count when known (e.g. from safetensors metadata).
    #: ``None`` falls back to the analytic count from the geometry.
    n_params: int | None = None
    variants: tuple[WeightVariant, ...] = (WeightVariant(),)

    def variant(self, name: str) -> WeightVariant:
        for v in self.variants:
            if v.name == name:
                return v
        msg = (
            f"Model {self.model_id!r} has no weight variant {name!r}; "
            f"available: {[v.name for v in self.variants]}"
        )
        raise KeyError(msg)


class DeviceSpec(BaseModel):
    """A device abstracted as capacity plus the capability flags that change
    which formula applies (not just which constant).
    """

    model_config = ConfigDict(frozen=True)

    total_bytes: int
    #: Bytes actually usable by the run. CUDA context + driver reserve
    #: consume memory before any tensor is allocated, so this is lower than
    #: ``total_bytes``; the default carve-out matches a typical ~0.75 GiB
    #: context.
    available_bytes: int | None = None
    name: str | None = None
    #: fp8 KV cache / fp8 weights need compute capability >= 8.9 (Ada/Hopper).
    supports_fp8: bool = True
    #: Whether the device can run flash-style attention kernels at all
    #: (Ampere and later). Necessary but not sufficient — which backend is
    #: actually used also depends on the installed packages and the model's
    #: masking, see :func:`agilerl.arena.memory.formulas.resolve_attn_implementation`.
    has_flash_attention: bool = True
    #: Whether the ``flash_attn`` package is importable in the target
    #: environment. Defaults to ``False`` because it is *not* part of the
    #: ``llm`` extra, so a stock install resolves ``auto`` to SDPA rather
    #: than FlashAttention-2.
    flash_attn_installed: bool = False
    #: Bytes a bare CUDA context costs, before the process allocates anything.
    #: Device-dependent by a factor of two (see
    #: :data:`MEASURED_CUDA_CONTEXT_BYTES`), so a single constant biases whole
    #: fleets in opposite directions. ``None`` falls back to
    #: :data:`CUDA_CONTEXT_BYTES_DEFAULT`.
    cuda_context_bytes: int | None = None

    @property
    def context_bytes(self) -> int:
        """This device's CUDA context, measured if known."""
        if self.cuda_context_bytes is not None:
            return self.cuda_context_bytes
        return CUDA_CONTEXT_BYTES_DEFAULT

    @property
    def usable_bytes(self) -> int:
        """Capacity the predicted peak is checked against.

        The CUDA context is *not* deducted here: it is already a component on
        the demand side, matching what NVML reports as device-used memory, so
        carving it out again would double-count it. What this does reserve is
        a fragmentation band — the caching allocator fails to serve a request
        somewhat below nominal capacity, so a run predicted at 100% does not
        actually fit.
        """
        if self.available_bytes is not None:
            return self.available_bytes
        return int(self.total_bytes * 0.95)

    @classmethod
    def from_compute_capability(
        cls, total_bytes: int, major: int, minor: int, name: str | None = None
    ) -> Self:
        cc = major + minor / 10
        return cls(
            total_bytes=total_bytes,
            name=name,
            supports_fp8=cc >= 8.9,
            has_flash_attention=cc >= 8.0,
            cuda_context_bytes=MEASURED_CUDA_CONTEXT_BYTES.get(name),
        )


class TrainingKnobs(BaseModel):
    """The training-side settings that genuinely exist in the framework.

    Notably absent, because the framework does not expose them:

    - Full fine-tuning — AgileRL trains LoRA adapters only, so gradients and
      optimizer state scale with adapter parameters, never with the base.
    - Optimizer choice — ``torch.optim.AdamW`` (or a DeepSpeed-config-owned
      optimizer) is the only path.
    - ``beta`` — the KL coefficient is memory-neutral: the reference forward
      runs (and the reference adapter exists) regardless of beta, so it is a
      loss setting, not a memory setting.
    """

    model_config = ConfigDict(frozen=True)

    algorithm: Algorithm = "grpo"
    #: Completion rows per gradient minibatch; with ``group_size`` it also
    #: proxies the (megabyte-scale) held rollout-tensor row count.
    batch_size: int = 16
    #: Rows per gradient forward/backward. ``None`` falls back to
    #: ``batch_size`` (matching ``LLMAlgorithm`` behaviour).
    micro_batch_size_per_gpu: int | None = None
    group_size: int = 8
    #: Completion rows per ``learn`` call (``prompts x group_size``).
    #: ``None`` assumes one prompt group.
    trajectories_per_update: int | None = None
    #: Full context budget (prompt + completion), i.e. the worst-case
    #: sequence length for activation and logprob tensors.
    max_model_len: int = 1024
    lora_rank: int = 16
    lora_target_scope: LoraTargetScope = "all-linear"
    #: Packed MoE expert matrices adapted via PEFT ``target_parameters``
    #: (e.g. ``mixer.experts.up_proj``). Each targeted matrix adds a
    #: per-expert rank decomposition on every MoE layer — on a 128-expert
    #: stack that is tens of millions of adapter parameters, an order of
    #: magnitude beyond the module-targeted adapters.
    lora_packed_target_matrices: int = 0
    #: How the packed-expert adapter path executes. ``"materialized"`` is
    #: PEFT ParamWrapper building the full effective weight per targeted
    #: matrix, saved for backward. ``"contracted"`` models the delta by
    #: contraction with chunked dispatch. Use the default unless those
    #: kernels are in the run.
    packed_moe_dispatch: PackedMoeDispatch = "materialized"
    #: KL coefficient. Memory-relevant only through the reference forward:
    #: at ``beta=0`` the KL term drops out of the loss, so the reference row
    #: of the fused no-grad pass has nothing to feed and can be skipped.
    beta: float = 0.001
    #: A frozen copy of the actor adapter used for reference logprobs.
    #: ``False`` routes reference rows through the (immutable) base instead —
    #: this removes the second adapter's weights but not the reference
    #: forward itself.
    use_separate_reference_adapter: bool = True
    weight_dtype: WeightDtype = "bf16"
    #: Trainer-side base quantization (QLoRA). ``lm_head`` stays unquantized
    #: and norms + lm_head are upcast to fp32 by k-bit preparation.
    quantization: TrainerQuantization = "none"
    #: Attention backend. ``auto`` mirrors the framework's resolution:
    #: FlashAttention-2 when the ``flash_attn`` package is importable,
    #: otherwise SDPA. Left explicit because the choice decides whether the
    #: S x S score matrix is materialised, which dominates long-context
    #: activation memory.
    attn_implementation: AttnImplementation = "auto"
    gradient_checkpointing: bool = True
    #: Save backward-saved activations to pinned host RAM instead of GPU.
    activation_offload: bool = False
    #: Fused-logprob tile rows. ``None`` auto-tunes to a ~256 MiB fp32 logit
    #: workspace, clamped to [128, 4096].
    chunk_rows: int | None = None
    #: Offload the trainer's base weights to host RAM across the rollout (the
    #: framework default). ``False`` keeps the whole trainer resident on the
    #: device *during generation* — which is how PPO currently has to run,
    #: because its value head is not moved back before the Triton forward — so
    #: the generation bar gains the trainer's base weights and adapters.
    use_memory_efficient_params: bool = True
    distributed: DistributedBackend = "none"
    #: Data-parallel trainer GPUs per population member
    #: (``training.training_gpus_per_agent``). The estimate stays per GPU;
    #: this decides what DeepSpeed shards across the group.
    n_training_gpus: int = 1
    #: DeepSpeed ZeRO stage when ``distributed="deepspeed"``. Stage 2 shards
    #: optimizer state and gradients; stage 3 also shards parameters —
    #: including the frozen base, which is where the memory goes.
    zero_stage: Literal[2, 3] = 2

    @property
    def dp_world_size(self) -> int:
        """GPUs anything is actually sharded across.

        The framework's multi-GPU LLM path is DeepSpeed-under-accelerate;
        without it every device holds the whole state, however many there are.
        """
        return self.n_training_gpus if self.distributed == "deepspeed" else 1

    @property
    def trajectories(self) -> int:
        """Completion rows one learner GPU sees per ``learn`` call.

        ``None`` assumes a single prompt group. The framework processes
        ``prompts x group_size`` rows per update, chunked into micro-batches;
        under data parallelism the replay buffer shards the update across the
        learner GPUs, so each rank works through its share.
        """
        total = self.trajectories_per_update or self.group_size
        return max(-(-total // self.dp_world_size), 1)

    @property
    def grad_rows(self) -> int:
        """Rows in one gradient micro-batch.

        Capped by the trajectories actually available: the framework uses
        ``min(num_samples, micro_batch_size_per_gpu)``, so asking for a
        micro-batch larger than the update has rows does not cost anything.
        """
        requested = self.micro_batch_size_per_gpu or self.batch_size
        return max(min(requested, self.trajectories), 1)

    @property
    def n_micro_batches(self) -> int:
        return max(-(-self.trajectories // self.grad_rows), 1)

    @property
    def uses_reference(self) -> bool:
        """Whether a reference policy is consulted at all.

        SFT has none. DPO always has one — its beta is the preference
        temperature, not a KL coefficient, so it never switches the reference
        off. Elsewhere the reference exists to supply the KL term, so at
        ``beta=0`` there is nothing for it to feed.

        Note this models the beta=0 short-circuit as *implemented*: the fused
        no-grad pass currently builds the reference row unconditionally, so a
        run at beta=0 pays for it until that is optimised. The estimate says
        so in its warnings rather than silently assuming the saving.
        """
        if self.algorithm == "sft":
            return False
        if self.algorithm == "dpo":
            return True
        return self.beta != 0.0

    @property
    def uses_critic(self) -> bool:
        """PPO trains a critic: its own LoRA adapter plus a value head."""
        return self.algorithm == "ppo"

    @property
    def uses_generation_engine(self) -> bool:
        """Whether the run starts a vLLM engine at all.

        SFT and DPO train from a fixed dataset — they hard-set
        ``use_vllm=False`` — so there is no engine to size and no sleeping
        engine to leave a residual in the training bar.
        """
        return self.algorithm not in ("sft", "dpo")

    @property
    def has_nograd_pass(self) -> bool:
        """Whether a no-grad logprob pass exists as its own instant.

        SFT is plain supervised learning: no old logprobs, no reference, so
        the no-grad instant never happens.
        """
        return self.algorithm != "sft"

    @property
    def grad_graph_rows(self) -> int:
        """Rows whose autograd graphs are live together at the loss backward.

        DPO forwards the chosen and rejected sequences separately, but one
        loss depends on both, so both graphs' checkpoint boundaries, hidden
        states and fp32 input casts are resident until the backward completes
        — twice the micro-batch. Everything else backpropagates one graph.
        """
        pairs = 2 if self.algorithm == "dpo" else 1
        return self.grad_rows * pairs

    @property
    def n_adapter_rows(self) -> int:
        """Row multiplier of the fused no-grad forward.

        The rollout algorithms fuse one row per consulted adapter —
        reference, actor and (PPO) critic — by repeating the batch, so this
        multiplies the no-grad activation footprint directly. DPO does not
        fuse: its reference logprobs are computed in separate sequential
        passes of one tensor each, so its multiplier is 1.
        """
        if self.algorithm == "dpo":
            return 1
        rows = 1  # the actor's own logprobs
        if self.uses_reference:
            rows += 1
        if self.uses_critic:
            rows += 1
        return rows

    @property
    def lora_casts_recompute_only(self) -> bool:
        """Whether PEFT's fp32 input casts exist only in checkpoint recompute.

        ``_amp_ctx`` wraps the shared forward helpers (``_fused_model_pass``,
        ``_get_logprobs``) in bf16 autocast and disables ``cast_input_dtype``
        on every LoRA layer for the duration (``_lora_input_cast_ctx``) — but
        only for the *original* forward. The gradient-checkpoint recompute
        runs during backward, after the context has restored the flag, so one
        block's casts reappear then: sized by the rows of the one graph
        autograd is recomputing (``grad_rows``), never by DPO's two live
        graphs, and absent at the loss instant.

        SFT is the exception by code path: ``_sft_loss`` calls
        ``actor.forward`` outside ``_amp_ctx``, so its forward casts survive
        (and its measured loss-instant peaks carry them).
        """
        return self.algorithm != "sft"

    @property
    def n_trained_adapters(self) -> int:
        """Adapters carrying gradients and optimizer state.

        The reference adapter is frozen; PPO additionally trains a critic.
        """
        return 2 if self.uses_critic else 1

    @property
    def n_resident_adapters(self) -> int:
        """Adapter copies held on the device.

        The reference adapter stays resident whenever it exists, including at
        beta=0 — it is created at init, so only its *forward* is skippable.
        """
        n = 1
        if self.use_separate_reference_adapter and self.algorithm != "sft":
            n += 1
        if self.uses_critic:
            n += 1
        return n


class GenerationKnobs(BaseModel):
    """The vLLM-side settings, mirroring ``VLLMConfig`` plus the workload shape."""

    model_config = ConfigDict(frozen=True)

    gpu_memory_utilization: float = 0.3
    max_num_seqs: int = 8
    max_model_len: int = 1024
    #: Worst-case prompt length. Prefill is the memory-relevant part of
    #: generation — decode adds one token per sequence per step — so the
    #: resident activation peak tracks prompt tokens in flight, not the full
    #: context budget. ``None`` assumes the worst case (all context is
    #: prompt).
    max_prompt_len: int | None = None
    #: ``None`` uses the framework's resolution rule:
    #: ``min(max_num_seqs * max_model_len, max(max_model_len, max_num_seqs * 8192))``.
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: KVCacheDtype = "auto"
    #: Pin the KV pool size instead of letting vLLM derive it from
    #: ``gpu_memory_utilization``.
    kv_cache_memory_bytes: int | None = None
    #: ``True`` skips CUDA-graph capture, at some decode-throughput cost.
    #: ``None``/``False`` keeps graphs on.
    enforce_eager: bool | None = None
    #: Recurrent-state cache strategy for a hybrid model's Mamba layers.
    #: vLLM defaults to ``"none"`` (one state per sequence) with prefix caching
    #: off, and to ``"all"`` -- one state per *block*, so the cache scales with
    #: context again -- when it is on. ``"align"`` bounds it at two.
    mamba_cache_mode: MambaCacheMode = "none"
    max_lora_rank: int = 16
    max_loras: int = 1
    weight_dtype: WeightDtype = "bf16"
    #: Name of the :class:`WeightVariant` the engine loads (vLLM may load a
    #: different variant than the trainer, e.g. an AWQ export).
    weight_variant: str = "base"
    #: Worst-case concurrent requests, ``prompts_in_flight * group_size``.
    #: ``None`` assumes the schedule limit (``max_num_seqs``) is saturated.
    concurrent_requests: int | None = None

    @property
    def concurrency(self) -> int:
        if self.concurrent_requests is None:
            return self.max_num_seqs
        return min(self.concurrent_requests, self.max_num_seqs)

    @property
    def prompt_len(self) -> int:
        return min(self.max_prompt_len or self.max_model_len, self.max_model_len)

    @model_validator(mode="after")
    def _check_utilization(self) -> Self:
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            msg = (
                "gpu_memory_utilization must be in (0, 1], got "
                f"{self.gpu_memory_utilization}"
            )
            raise ValueError(msg)
        return self


class RunConfig(BaseModel):
    """A complete sizing question: model + devices + settings + placement."""

    model_config = ConfigDict(frozen=True)

    model: ModelSpec
    train_device: DeviceSpec
    #: ``None`` means colocated: generation shares ``train_device`` and the
    #: cross-phase residuals (offloaded trainer state during rollout, sleeping
    #: engine during training) are included in each phase's bar.
    gen_device: DeviceSpec | None = None
    training: TrainingKnobs = Field(default_factory=TrainingKnobs)
    generation: GenerationKnobs = Field(default_factory=GenerationKnobs)
    #: Whether the run executes under Ray orchestration (every Arena
    #: submission does). Charges the measured worker overhead per process.
    orchestrated: bool = False

    @property
    def colocated(self) -> bool:
        return self.gen_device is None

    @property
    def generation_device(self) -> DeviceSpec:
        return self.gen_device or self.train_device
