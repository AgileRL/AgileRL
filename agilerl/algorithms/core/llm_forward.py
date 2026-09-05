# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.typing import (
    OptimizerType,
)
from agilerl.utils.algo_utils import (
    DummyOptimizer,
)
from agilerl.utils.llm_packing import (
    pack_padded_batch,
    unpack_logprobs,
    unpack_values,
)

# Make imports visible to typechecker and import when required
if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:

    from agilerl.algorithms.core.llm_ops.fused_lora import (
        adapter_aligned_chunks,
        get_cached_lora_layers,
        set_fused_adapter_routing,
        unset_fused_adapter_routing,
    )
    from agilerl.utils.llm_utils import (
        allreduce_minmax_int,
        attention_mask_from_padded_ids,
        fill_outside_mask,
        gather_if_ds_param,
    )




logger = logging.getLogger(__name__)


class LLMForwardMixin:
    """Fused forward, logprobs, and backward for :class:`LLMAlgorithm`."""

    def _select_optim_class(self) -> type[OptimizerType | DummyOptimizer]:
        """Select the optimizer class based on the accelerator and deepspeed config.

        :return: Optimizer class
        :rtype: type[torch.optim.Optimizer] | type[DummyOptimizer]
        """
        if (
            self.accelerator is not None
            and self.accelerator.state.deepspeed_plugin is not None
            and self.accelerator.state.deepspeed_plugin.deepspeed_config.get(
                "optimizer",
                None,
            )
            is not None
        ):
            return DummyOptimizer
        return AdamW

    def _save_distributed_actor(
        self,
        path: str,
        tag: str = "intermediate_checkpoint",
        lora_only: bool = False,
    ) -> None:
        """Save actor/optimizer/scheduler state via DeepSpeed checkpointing.

        :param path: Output directory to save the checkpoint at
        :type path: str
        """
        if self.accelerator is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            assert self.actor is not None, (
                "Actor is not defined, please check that the actor is defined."
            )
            # Keep reference adapter frozen in DeepSpeed checkpoints so frozen
            # param fragments are emitted consistently on save/load roundtrips.
            trainable_adapters = [
                name for name in self.selected_adapters if name != "reference"
            ]
            self._restore_adapter_trainability(trainable_adapters)
            self.actor.save_checkpoint(
                path, tag=tag, exclude_frozen_parameters=lora_only
            )
            self.use_adapter("actor")
        else:
            warnings.warn(
                "Distributed actor save not supported for non-distributed training.",
                stacklevel=2,
            )

    def _load_distributed_actor(
        self,
        path: str,
        tag: str = "intermediate_checkpoint",
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> None:
        """Override the load_checkpoint method to provide guidance on the correct method to use.

        :param path: Output directory to load the checkpoint from
        :type path: str
        """
        if self.accelerator is not None:
            deepspeed_dirs = sorted(Path(path).glob(tag))
            try:
                assert len(deepspeed_dirs) > 0
                load_path, _ = self.actor.load_checkpoint(
                    str(path),
                    tag=tag,
                    load_module_strict=False,
                    load_optimizer_states=load_optimizer_states,
                    load_lr_scheduler_states=load_lr_scheduler_states,
                )
                if load_path is None:
                    msg = (
                        "Load path is returned as None from deepspeed load_checkpoint."
                    )
                    raise ValueError(
                        msg,
                    )
                self.use_adapter("actor")

            except Exception as e:
                msg = f"Deepspeed failed to resume from checkpoint {path}"
                raise ValueError(
                    msg,
                ) from e
        else:
            warnings.warn(
                "Distributed actor load not supported for non-distributed training.",
                stacklevel=2,
            )

    @staticmethod
    def _position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """Left-padding-safe ``position_ids`` from an attention mask.

        Cumulative real-token count minus one, with padded positions pinned to
        ``1`` so the rotary embedding sees a valid (ignored) index.

        :param mask: ``(B, T)`` attention mask (1 = real token, 0 = pad).
        :type mask: torch.Tensor
        :return: ``(B, T)`` position ids in ``long``.
        :rtype: torch.Tensor
        """
        position_ids = mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(mask == 0), value=1)
        return position_ids

    def _fused_logprob_fn_and_head(
        self,
    ) -> tuple[Callable, torch.Tensor, torch.Tensor | None]:
        """Resolve the fused per-token-logprob fn and live lm_head tensors.

        Fused-linear-logprob path (the only path): the lm_head is identity-
        patched for the forward (which returns the last hidden state), then
        per-token logprobs are computed via a chunked matmul over the lm_head
        weight, never materializing ``(B, T, V)``. Under grad the matmul is
        routed through a gradient-checkpointed autograd Function so the backward
        stays bounded too; under no_grad the lighter static method is used.

        Under ZeRO-3 the fused kernel gathers ``lm_head`` for the matmul only
        (not across ``actor.forward``); see ``gather_if_ds_param``.
        """
        fused_fn = (
            self._logprobs_from_hidden_fused_grad
            if torch.is_grad_enabled()
            else self._logprobs_from_hidden_fused
        )
        lm_head = self._get_lm_head()
        return fused_fn, lm_head.weight, lm_head.bias

    def _liger_head_gather(self) -> AbstractContextManager[None]:
        """Gather ``lm_head`` for one Liger fused-loss call.

        Liger computes its gradients inside ``forward`` and saves them, so the
        weight is read only during ``apply``. Must wrap the ``apply`` call
        alone, never ``actor.forward``, whose post-forward hooks re-partition
        the param.
        """
        lm_head = self._get_lm_head()
        return gather_if_ds_param(lm_head.weight, lm_head.bias)

    def _warn_liger_non_token_is(
        self,
        level: str,
        algo_name: str,
        *,
        once_attr: str = "_liger_non_token_warned",
    ) -> None:
        """Warn once that Liger + non-token importance sampling is not memory-bounded.

        The combination (``use_liger_loss=True`` with a turn-/trajectory-level
        importance-sampling ``level``) is permitted but unbounded in memory: the
        token-flatten trick only applies at token level, so the fused kernel
        processes one whole sequence per chunk. Guards on a per-instance flag
        (``once_attr``) so repeated calls emit the canonical message at most once.

        :param level: the importance-sampling level (e.g. ``"turn"``/``"trajectory"``).
        :type level: str
        :param algo_name: human-readable algorithm name for the message prefix.
        :type algo_name: str
        :param once_attr: per-instance attribute used to suppress duplicates.
        :type once_attr: str, optional
        """
        if getattr(self, once_attr, False):
            return
        warnings.warn(
            f"{algo_name} with use_liger_loss=True at importance_sampling_level="
            f"'{level}' is permitted but NOT memory-bounded: the fused kernel "
            "processes one whole sequence per chunk and materializes a "
            "(seq_len, vocab) logits tensor per trajectory (the token-flatten "
            "trick only applies at token level). For bounded memory set "
            "use_liger_loss=False — the standard path is always memory-bounded "
            "via the fused-linear-logprob path.",
            stacklevel=2,
        )
        setattr(self, once_attr, True)

    def _align_sampling_logprobs(
        self,
        sampling_logps: list[torch.Tensor | None] | None,
        action_masks: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, int]:
        """Scatter per-row flat vLLM logprobs onto the ``(B, T-1)`` action frame.

        ``sampling_logps`` is one 1-D tensor per row (the generated-token
        logprobs in order; single-turn = one rollout, multi-turn = concatenated
        across turns), parallel to the stacked ``token_ids`` rows. Each is
        scattered onto the ``True`` positions of that row's action mask. Rows
        whose token count doesn't match the mask (e.g. env truncation) keep the
        ``old_log_probs`` value there, so their importance ratio is 1 (no
        correction) instead of crashing.

        :param sampling_logps: Per-row flat logprobs, or ``None``.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param old_log_probs: ``(B, T-1)`` trainer old-policy logprobs (the
            fallback where data is missing → unit ratio).
        :type old_log_probs: torch.Tensor
        :return: ``(aligned (B, T-1) or None, n_rows_skipped)``.
        :rtype: tuple[torch.Tensor | None, int]
        """
        if not sampling_logps:
            return None, 0
        out = old_log_probs.clone()
        mask_bool = action_masks.to(torch.bool)
        n_rows = out.shape[0]
        n_skipped = 0
        for b in range(n_rows):
            flat = sampling_logps[b] if b < len(sampling_logps) else None
            if flat is None:
                n_skipped += 1
                continue
            pos = mask_bool[b].nonzero(as_tuple=True)[0]
            if pos.numel() != flat.numel():
                n_skipped += 1
                continue
            out[b, pos] = flat.to(device=out.device, dtype=out.dtype)
        return out, n_skipped

    def _sampling_mismatch_metrics(
        self,
        old_log_probs: torch.Tensor,
        sampling_log_probs: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> dict[str, float]:
        """Summarise the vLLM-vs-trainer logprob divergence over action tokens.

        ``vllm_is_delta_mean`` is ``mean|old - sampling|``; the ratio stats
        describe ``clamp(exp(old - sampling), max=cap)`` (mean, p95, and the
        fraction hitting the clamp). All are detached batch-level diagnostics,
        computed regardless of whether the correction is applied to the loss.
        """
        with torch.no_grad():
            mask = action_masks.to(torch.bool)
            denom = mask.sum().clamp(min=1.0).to(torch.float32)
            log_diff = fill_outside_mask(
                (old_log_probs - sampling_log_probs).to(torch.float32),
                mask,
            )
            delta_mean = (log_diff.abs().sum() / denom).item()
            ratio = torch.exp(log_diff).clamp(max=self.vllm_importance_sampling_cap)
            sel = ratio[mask]
            metrics = {"vllm_is_delta_mean": delta_mean}
            if sel.numel() > 0:
                metrics["vllm_is_ratio_mean"] = sel.mean().item()
                metrics["vllm_is_ratio_p95"] = torch.quantile(sel.float(), 0.95).item()
                metrics["vllm_is_frac_clamped"] = (
                    (sel >= self.vllm_importance_sampling_cap).float().mean().item()
                )
            return metrics

    def _aligned_sampling_logprobs_and_metrics(
        self,
        sampling_logps: list[torch.Tensor | None] | None,
        action_masks: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Align captured vLLM sampling logprobs and summarise the mismatch.

        Aligns the per-row logprobs captured at rollout onto the ``(B, T-1)``
        action frame (:meth:`_align_sampling_logprobs`), computes the
        vLLM-vs-trainer mismatch metrics whenever logprobs were captured —
        independently of whether the correction is applied to the loss — and
        warns when rows were skipped for a token-count mismatch. Shared by the
        GRPO/PPO/REINFORCE ``learn`` implementations.

        :param sampling_logps: Per-row vLLM sampling logprobs from rollout, or
            ``None`` when none were captured.
        :type sampling_logps: list[torch.Tensor | None] | None
        :param action_masks: ``(B, T-1)`` action-token mask.
        :type action_masks: torch.Tensor
        :param old_log_probs: ``(B, T-1)`` frozen-policy logprobs.
        :type old_log_probs: torch.Tensor
        :return: ``(aligned_logprobs_or_None, metrics)``.
        :rtype: tuple[torch.Tensor | None, dict[str, float]]
        """
        is_metrics: dict[str, float] = {}
        sampling_log_probs, n_skipped = self._align_sampling_logprobs(
            sampling_logps, action_masks, old_log_probs
        )
        if sampling_log_probs is not None:
            is_metrics = self._sampling_mismatch_metrics(
                old_log_probs, sampling_log_probs, action_masks
            )
            if n_skipped:
                is_metrics["vllm_is_rows_skipped"] = float(n_skipped)
                warnings.warn(
                    f"{n_skipped}/{action_masks.shape[0]} rows had a token-count "
                    "mismatch between captured vLLM logprobs and the action "
                    "mask; their importance ratio defaults to 1 (no "
                    "correction). Check rollout/trainer tokenisation if this "
                    "is large.",
                    stacklevel=2,
                )
        return sampling_log_probs, is_metrics

    @contextmanager
    def _amp_ctx(self) -> Generator[None, None, None]:
        """Yield a ``torch.amp.autocast`` context when running without an accelerator.

        When an ``Accelerator`` is present it already manages mixed-precision
        via its own autocast wrapper, so this is a no-op in that case.
        """
        if self.accelerator is not None:
            yield
        else:
            device_type = torch.device(self.device).type
            if device_type == "cuda" and torch.cuda.is_bf16_supported():
                with (
                    torch.amp.autocast(device_type, dtype=torch.bfloat16),
                    self._lora_input_cast_ctx(),
                ):
                    yield
            else:
                yield

    @contextmanager
    def _lora_input_cast_ctx(self) -> Generator[None, None, None]:
        """Skip PEFT's fp32 cast of every LoRA input while autocast is active."""
        if self.actor is None:
            yield
            return

        layers = get_cached_lora_layers(self.actor)
        previous = [layer.cast_input_dtype_enabled for layer in layers]
        for layer in layers:
            layer.cast_input_dtype_enabled = False
        try:
            yield
        finally:
            for layer, was_enabled in zip(layers, previous, strict=True):
                layer.cast_input_dtype_enabled = was_enabled

    @contextmanager
    def _activation_offload_ctx(self) -> Generator[None, None, None]:
        """Offload tensors saved for backward to pinned host RAM.

        When ``activation_offload`` is set, the training forward pass is run
        inside :func:`torch.autograd.graph.save_on_cpu`, so the activations
        kept between forward and backward (with gradient checkpointing, the
        checkpoint-boundary tensors) live in host memory instead of GPU
        memory. This trades PCIe bandwidth for GPU memory; the win grows with
        sequence length, which makes it the lever for long-context training.

        A no-op when offload is disabled or grads are inactive (rollout /
        reference forwards save nothing for backward). Purely trainer-side, so
        it composes with a co-located or a decoupled rollout engine alike.
        """
        if self.activation_offload and torch.is_grad_enabled():
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                yield
        else:
            yield

    def _fused_model_pass(
        self,
        fused_ids: torch.Tensor,
        fused_mask: torch.Tensor,
        routing: list[str],
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model on a fused batch with per-sample adapter routing.

        When *batch_size* is ``None`` the full batch is processed in a single
        ``model.forward`` call — required when gradients are active so that
        gradient-checkpoint recomputation sees the same routing.  When set,
        the batch is iterated in micro-batches (safe under ``no_grad``).

        :return: ``(log_probs, values)`` where *log_probs* has shape
            ``(fused_ids.shape[0], seq_len - 1)`` and *values* matches that
            batch dimension when ``use_value_head`` is set, else ``None``.
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
        unwrapped = self._get_unwrapped_actor()
        total = fused_ids.shape[0]
        seq_len_out = fused_ids.shape[1] - 1

        position_ids = None
        if self.calc_position_embeddings:
            position_ids = self._position_ids_from_mask(fused_mask)

        # Micro-batches never straddle an adapter run: packed-experts LoRA
        # layers see expert-sorted rows and can only apply one adapter per
        # forward call.
        chunks = (
            [(0, total)]
            if batch_size is None
            else adapter_aligned_chunks(routing, batch_size)
        )

        fused_fn, lm_head_weight, lm_head_bias = self._fused_logprob_fn_and_head()

        def _process_chunk(
            start: int, end: int
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            set_fused_adapter_routing(unwrapped, routing[start:end])
            model_kwargs: dict = {
                "input_ids": fused_ids[start:end],
                "attention_mask": fused_mask[start:end],
                "use_cache": False,
            }
            if position_ids is not None:
                model_kwargs["position_ids"] = position_ids[start:end]

            with (
                self._patch_lm_head_to_identity(),
                self._amp_ctx(),
                self._activation_offload_ctx(),
            ):
                output = self.actor.forward(**model_kwargs)

            if isinstance(output, tuple):
                # Value-head models may return (loss, logits, value, ...); Peft/causal
                # paths may return shorter tuples — only index when present. With
                # lm_head identity-patched, output[0] is the last hidden state.
                first = output[0]
                value = output[2] if len(output) > 2 else None
            else:
                first = output.logits
                value = None
            del output

            chunk_lp = fused_fn(
                first[:, :-1],
                lm_head_weight,
                lm_head_bias,
                fused_ids[start:end, 1:],
                temperature=self.temperature,
                cast_to_fp32=self.cast_logprobs_to_fp32,
                chunk_rows=self.chunk_rows,
            )
            del first

            chunk_v = (
                value[:, :-1] if (self.use_value_head and value is not None) else None
            )
            return chunk_lp, chunk_v

        # Single-chunk fast path: skip the buffer + copy entirely.
        if len(chunks) == 1:
            return _process_chunk(0, total)

        # Multi-chunk path: pre-allocate output buffers once and write each
        # chunk in place via copy_(). Avoids holding the full list of chunk
        # tensors plus the concatenated buffer in memory at the same time
        # (which doubles peak memory in the torch.cat path).
        logprobs_out: torch.Tensor | None = None
        values_out: torch.Tensor | None = None

        for start, end in chunks:
            chunk_lp, chunk_v = _process_chunk(start, end)

            # Lazy-allocate on the first chunk so we inherit dtype/device
            # from the model output rather than guessing up front.
            if logprobs_out is None:
                logprobs_out = torch.empty(
                    (total, seq_len_out),
                    dtype=chunk_lp.dtype,
                    device=chunk_lp.device,
                )
            logprobs_out[start:end].copy_(chunk_lp)
            del chunk_lp

            if chunk_v is not None:
                if values_out is None:
                    values_out = torch.empty(
                        (total, seq_len_out),
                        dtype=chunk_v.dtype,
                        device=chunk_v.device,
                    )
                values_out[start:end].copy_(chunk_v)
                del chunk_v

        assert logprobs_out is not None  # chunks is never empty
        return logprobs_out, values_out

    def _fused_forward(
        self,
        ids: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Actor log-probs, and optionally critic values, in one forward.

        When ``use_value_head`` is set, the input is doubled (actor slice then
        critic slice) and routed so the base model runs once. Otherwise only
        the actor slice is run.

        The doubled batch (value-head path) is always processed in one
        ``model.forward`` call to preserve gradient-checkpoint correctness.

        .. note::

           The routing is **not** cleared here — it must remain active until
           after ``backward()`` completes (for gradient checkpoint
           recomputation).  Callers must call
           ``unset_fused_adapter_routing`` after the backward pass.

           Callers are responsible for ensuring the model is in training
           mode and adapter trainability is restored before entering the
           minibatch loop (see ``learn()`` in ``ppo_llm.py``).

        :param ids: Token IDs ``(B, seq_len)``.
        :type ids: torch.Tensor
        :param batch_size: Unused (kept for API symmetry).
        :type batch_size: int
        :param attention_mask: Optional attention mask matching *ids*.
        :type attention_mask: torch.Tensor | None, optional
        :return: ``(actor_log_probs, critic_values)`` with shapes ``(B, seq_len-1)``;
            *critic_values* is ``None`` when no value head is used.
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
        B = ids.shape[0]
        if attention_mask is None:
            attention_mask = attention_mask_from_padded_ids(ids, self.pad_token_id)

        # Packed path for the gradient forward only; no-grad passes stay padded.
        if torch.is_grad_enabled() and self._packing_mode() is not None:
            return self._fused_packed_forward(ids, attention_mask)

        if self.use_value_head:
            fused_ids = ids.repeat(2, 1)
            fused_mask = attention_mask.repeat(2, 1)
            routing = ["actor"] * B + ["critic"] * B
        else:
            fused_ids = ids
            fused_mask = attention_mask
            routing = ["actor"] * B

        log_probs, values = self._fused_model_pass(
            fused_ids,
            fused_mask,
            routing,
        )
        if self.use_value_head:
            assert values is not None  # value-head models return values
            return log_probs[:B], values[B:]
        return log_probs, None

    def _fused_packed_forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Padding-free packed variant of the gradient :meth:`_fused_forward`.

        Actor and critic see identical token ids, so the ``(B, T)`` batch is
        packed **once** into a single padding-free row, then that row is
        repeated per active adapter (``actor`` [+ ``critic``]) to form an
        ``(n_adapters, N)`` batch. Per-row fused LoRA routing applies the actor
        adapter to row 0 and the critic adapter to the last row; both rows carry
        the same per-segment ``position_ids`` (``attention_mask=None``), so
        transformers builds a block-diagonal varlen / blockmask forward — no
        cross-sequence attention, sliding windows preserved (see
        :meth:`_packing_mode`). A **single** ``model.forward`` keeps
        gradient-checkpoint recomputation consistent with one persistent routing
        (the routing is *not* cleared here — see the :meth:`_fused_forward`
        note).

        :param ids: Token IDs ``(B, seq_len)``.
        :type ids: torch.Tensor
        :param attention_mask: Mask matching *ids* (non-zero marks real tokens).
        :type attention_mask: torch.Tensor
        :return: ``(actor_log_probs, critic_values)`` each ``(B, seq_len - 1)``;
            *critic_values* is ``None`` when no value head is used.
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
        unwrapped = self._get_unwrapped_actor()
        packed = pack_padded_batch(ids, attention_mask)

        adapters = ["actor"] + (["critic"] if self.use_value_head else [])
        n_adapters = len(adapters)
        fused_ids = packed.input_ids.repeat(n_adapters, 1)  # (n_adapters, N)
        fused_position_ids = packed.position_ids.repeat(n_adapters, 1)
        set_fused_adapter_routing(unwrapped, adapters)

        fused_fn, lm_head_weight, lm_head_bias = self._fused_logprob_fn_and_head()
        with (
            self._patch_lm_head_to_identity(),
            self._amp_ctx(),
            self._activation_offload_ctx(),
        ):
            output = self.actor.forward(
                input_ids=fused_ids,
                position_ids=fused_position_ids,
                use_cache=False,
            )

        if isinstance(output, tuple):
            hidden = output[0]
            value = output[2] if len(output) > 2 else None
        else:
            hidden = output.logits
            value = None

        # Actor log-probs from row 0 (actor adapter): the fused matmul
        # consumes the (1, N, H) hidden + (1, N-1) next-token targets exactly as
        # the padded path does; unpack scatters back to the (B, T-1) frame and
        # drops the cross-segment boundary prediction.
        packed_lp = fused_fn(
            hidden[:1][:, :-1],
            lm_head_weight,
            lm_head_bias,
            packed.input_ids[:, 1:],
            temperature=self.temperature,
            cast_to_fp32=self.cast_logprobs_to_fp32,
            chunk_rows=self.chunk_rows,
        )
        log_probs = unpack_logprobs(packed_lp, packed)

        values = None
        if self.use_value_head and value is not None:
            # Critic values from the last row (critic adapter): per-token scalars
            # mapped back to the (B, T-1) frame (no boundary drop, see
            # unpack_values). Pad positions are zero and masked downstream.
            values = unpack_values(value[-1], packed)

        return log_probs, values

    def _fused_forward_no_grad(
        self,
        ids: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute reference log-probs, actor log-probs, and critic values in
        one forward pass (under ``torch.no_grad``).

        The batch is tripled (reference / actor / critic rows) and routed per
        row, so the frozen base runs in a single fused pass.  When
        ``use_separate_reference_adapter`` is ``True`` the reference rows use
        the ``"reference"`` adapter; when ``False`` they are routed to PEFT's
        reserved ``"__base__"`` name, which applies no LoRA delta (the frozen
        base is the reference policy).

        This method micro-batches because no gradient checkpoint recomputation
        is involved.

        :param ids: Token IDs ``(B, seq_len)``.
        :type ids: torch.Tensor
        :param batch_size: Micro-batch size for memory-bounded iteration.
        :type batch_size: int
        :param attention_mask: Optional attention mask matching *ids*.
        :type attention_mask: torch.Tensor | None, optional
        :return: ``(reference_log_probs, actor_log_probs, critic_values)``
            each of shape ``(B, seq_len - 1)``.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        """
        B = ids.shape[0]
        if attention_mask is None:
            attention_mask = attention_mask_from_padded_ids(ids, self.pad_token_id)

        self.actor.eval()

        with torch.no_grad():
            reference_adapter = (
                "reference" if self.use_separate_reference_adapter else "__base__"
            )
            adapters = [reference_adapter, "actor"]
            if self.use_value_head:
                adapters.append("critic")

            N = len(adapters)
            fused_ids = ids.repeat(N, 1)
            fused_mask = attention_mask.repeat(N, 1)
            routing: list[str] = []
            for adapter in adapters:
                routing.extend([adapter] * B)

            log_probs, values = self._fused_model_pass(
                fused_ids,
                fused_mask,
                routing,
                batch_size=batch_size,
            )
            unset_fused_adapter_routing(self._get_unwrapped_actor())
            ref_logprobs = log_probs[:B]
            actor_logprobs = log_probs[B : 2 * B]
            if self.use_value_head:
                assert values is not None  # value-head models return values
                critic_values = values[2 * B :]
            else:
                critic_values = None

        return ref_logprobs, actor_logprobs, critic_values

    def _resolve_attn_implementation(self) -> str | None:
        """Best-effort read of the actor's active attention backend."""
        try:
            cfg = getattr(self._get_unwrapped_actor(), "config", None)
            impl = getattr(cfg, "_attn_implementation", None)
            if impl is not None:
                return impl
        except Exception:
            # Best-effort probe: introspecting a wrapped actor's config can fail
            # in exotic setups (accelerate/DeepSpeed wrappers, slotted modules).
            # Non-fatal — fall back to model_config / None below.
            logger.debug("attn-implementation probe failed", exc_info=True)
        model_config = getattr(self, "model_config", None)
        if isinstance(model_config, dict):
            return model_config.get("attn_implementation")
        return None

    def _packing_mode(self) -> str | None:
        """Resolve how (if at all) to pack the forward, given the backend.

        Packing passes the model only per-sequence ``position_ids`` (no mask);
        transformers detects the packed format and AND-composes a block-diagonal
        constraint onto each layer's native (causal / sliding-window) mask. We
        only gate on backends where that composed mask stays sparse:

        * ``"varlen"`` — FlashAttention-2 turns the packed ``position_ids`` into
          ``cu_seqlens`` (+ per-layer ``window_size``); no mask is materialized.
          Memory/throughput-optimal and handles dynamic shapes with no recompile.
        * ``"blockmask"`` — FlexAttention builds a sparse block-diagonal
          ``BlockMask`` (active tiles + compiled ``mask_mod``, windowed on
          sliding layers). Needs no ``flash_attn`` build, but recompiles per
          packed length.

        Both preserve sliding-window attention per layer, so packing is correct
        for SWA models (e.g. gemma), not just full-attention ones. Dense backends
        (SDPA / eager) would build a *dense* ``O(N^2)`` block-diagonal mask —
        correct but defeating the memory win — so packing is **not** enabled on
        them and falls back to padding (warning once). Returns ``None`` when
        packing is off or the backend is unsupported.
        """
        if not getattr(self, "use_sequence_packing", False):
            return None
        impl = self._resolve_attn_implementation()
        if impl == "flash_attention_2":
            return "varlen"
        if impl == "flex_attention":
            return "blockmask"
        if not getattr(self, "_packing_backend_warned", False):
            warnings.warn(
                "use_sequence_packing=True needs a varlen/block-sparse attention "
                "backend (flash_attention_2 or flex_attention); got "
                f"{impl!r}, which has no sparse path. Falling back to the padded "
                "forward (no packing).",
                stacklevel=2,
            )
            self._packing_backend_warned = True
        return None

    def _sequence_packing_active(self) -> bool:
        """``True`` when sequence packing should drive this forward."""
        return self._packing_mode() is not None

    def _get_logprobs(
        self,
        ids: torch.Tensor,
        batch_size: int,
        use_reference: bool = False,
        eval_mode: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Find the log probabilities for a set of previously generated ids.

        :param ids: Completion IDs.
        :type ids: torch.Tensor
        :param batch_size: Batch size.
        :type batch_size: int
        :param use_reference: Flag to indicate to use reference policy, defaults to False
        :type use_reference: bool, optional
        :param eval_mode: Flag to indicate setting policy network to evaluation mode, defaults to False
        :type eval_mode: bool, optional
        :param attention_mask: Attention mask.
        :type attention_mask: torch.Tensor, optional
        :return: Log probabilities of the completion IDs.
        :rtype: torch.Tensor
        """
        grad_enabled = torch.is_grad_enabled()
        with self.select_adapter("reference" if use_reference else "actor"):
            self.actor.train(mode=not eval_mode)
            num_samples = ids.shape[0]
            if attention_mask is None:
                attention_mask = attention_mask_from_padded_ids(ids, self.pad_token_id)
            if self.calc_position_embeddings:
                position_ids = self._position_ids_from_mask(attention_mask)

            fused_fn, lm_head_weight, lm_head_bias = self._fused_logprob_fn_and_head()
            # Pack only the gradient forward (the per-epoch hot path). The
            # no-grad old/reference passes stay padded so they are mutually
            # consistent; the packed-vs-padded gap for the current policy is the
            # tiny varlen/dense-vs-padded numerical difference.
            packing_mode = self._packing_mode() if grad_enabled else None

            # Split the sample into batches
            log_probs = []
            for batch in range(0, num_samples, batch_size):
                end_idx = min((batch + batch_size), num_samples)
                batch_ids = ids[batch:end_idx, :]
                batch_attention_mask = attention_mask[batch:end_idx, :]

                packed = None
                if packing_mode is not None:
                    packed = pack_padded_batch(batch_ids, batch_attention_mask)

                if packed is not None:
                    # Per-sequence position_ids (no mask): transformers detects
                    # the packed format and keeps sequences attention-isolated
                    # per layer (sliding-window safe).
                    batch_model_kwargs = {
                        "input_ids": packed.input_ids,
                        "position_ids": packed.position_ids,
                        "use_cache": False,
                    }
                else:
                    batch_model_kwargs = {
                        "input_ids": batch_ids,
                        "attention_mask": batch_attention_mask,
                        "use_cache": False,
                    }
                    if self.calc_position_embeddings:
                        batch_model_kwargs["position_ids"] = position_ids[
                            batch:end_idx, :
                        ]

                with (
                    self._patch_lm_head_to_identity(),
                    self._amp_ctx(),
                    self._activation_offload_ctx(),
                ):
                    output = self.actor.forward(**batch_model_kwargs)
                first = output[0] if isinstance(output, tuple) else output.logits

                if packed is not None:
                    packed_lp = fused_fn(
                        first[:, :-1],
                        lm_head_weight,
                        lm_head_bias,
                        packed.input_ids[:, 1:],
                        temperature=self.temperature,
                        cast_to_fp32=self.cast_logprobs_to_fp32,
                        chunk_rows=self.chunk_rows,
                    )
                    # Map back to the dense (mb, T-1) frame so the loss path is
                    # unchanged; cross-segment boundary predictions are dropped.
                    # unpack_logprobs reshapes the packed logprobs internally.
                    log_prob = unpack_logprobs(packed_lp, packed)
                else:
                    log_prob = fused_fn(
                        first[:, :-1],
                        lm_head_weight,
                        lm_head_bias,
                        batch_ids[:, 1:],
                        temperature=self.temperature,
                        cast_to_fp32=self.cast_logprobs_to_fp32,
                        chunk_rows=self.chunk_rows,
                    )

                first = None
                batch_model_kwargs = None
                log_probs.append(log_prob)
            return torch.cat(log_probs, dim=0)

    def _raise_if_loss_not_finite_on_any_rank(self, loss: torch.Tensor) -> None:
        """Raise when ``loss`` is non-finite on this rank or any DP peer.

        Under multi-process training a local-only raise leaves peers waiting in
        ZeRO-3 / NCCL collectives; the allreduce makes every rank raise together.

        :param loss: Scalar loss about to enter :meth:`_backward_pass`.
        :type loss: torch.Tensor
        :return: None
        :rtype: None
        """
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            nonfinite_flag = 0 if loss.isfinite().item() else 1
            _, max_flag = allreduce_minmax_int(nonfinite_flag, self.accelerator)
            if max_flag > 0:
                msg = f"Loss is not finite: {loss}"
                raise ValueError(msg)
            return
        if not loss.isfinite():
            msg = f"Loss is not finite: {loss}"
            raise ValueError(msg)

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Perform a backward pass and optimizer step.

        :param loss: Combined loss.
        :type loss: torch.Tensor
        """
        if self._uses_deepspeed:
            assert self.accelerator is not None  # _uses_deepspeed implies one
            self.accelerator.backward(loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.lr = float(self.lr_scheduler.get_last_lr()[0])
        else:
            loss.backward()

            for group in self.optimizer.optimizer.param_groups:
                clip_grad_norm_(group["params"], self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.lr = float(self.lr_scheduler.get_last_lr()[0])

    @property
    def _peft_model(self) -> Any:  # noqa: ANN401 -- PeftModel lives at a wrapper-specific attribute; concrete type varies
        """The PeftModel managing LoRA adapters.

        When ``use_value_head=True`` the PeftModel lives inside the
        value-head wrapper at ``self.actor.pretrained_model``.
        Otherwise ``self.actor`` itself is the PeftModel.
        """
        if self.use_value_head:
            return self.actor.pretrained_model
        return self.actor

    def _restore_adapter_trainability(self, selected_adapters: list[str]) -> None:
        """Restore requires_grad=True for all trainable parameters of specified adapters.

        PEFT's set_adapter() sets requires_grad=False on all non-active adapter
        weights. Under DeepSpeed ZeRO Stage 2, gradient bucket hooks are registered
        once at accelerator.prepare() time based on the requires_grad snapshot at
        that moment. If set_adapter() later toggles requires_grad=False on params
        that ZeRO-2 registered hooks for, those hooks never fire, the bucket never
        completes, and reduce-scatter never runs - the optimizer sees zero gradients.

        :param selected_adapters: LoRA adapter names whose params should be trainable.
        :type selected_adapters: list[str]
        """
        key = tuple(sorted(selected_adapters))
        cache = getattr(self, "_trainable_params_cache", None)
        if cache is not None and cache[0] == key:
            for param in cache[1]:
                param.requires_grad_(True)
            return

        model = self.actor.module if hasattr(self.actor, "module") else self.actor
        params: list[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            for adapter in selected_adapters:
                if adapter in name and "lora" in name:
                    params.append(param)
                    break
        for param in params:
            param.requires_grad_(True)
        self._trainable_params_cache = (key, params)
