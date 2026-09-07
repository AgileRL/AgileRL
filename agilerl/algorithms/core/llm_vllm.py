# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch

from agilerl import HAS_LLM_DEPENDENCIES, HAS_VLLM
from agilerl.algorithms.core.llm_ops.fused_logprobs import (
    FusedLinearLogProbsFunction,
    fused_linear_logprobs_chunked,
)
from agilerl.modules.dummy import DummyEvolvable
from agilerl.typing import (
    RolloutPrompt,
)
from agilerl.utils.algo_utils import (
    VLLMConfig,
    stack_and_pad_experiences,
)
from agilerl.utils.evolvable_networks import (
    compile_model,
)

# Make imports visible to typechecker and import when required
if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import (
        LoraConfig,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    from agilerl.algorithms.core.llm_ops.vllm_colocate import (
        patch_vllm_3d_moe_lora_flag,
        patch_vllm_lora_keep_resident,
        patch_vllm_strip_multimodal_towers,
    )
    from agilerl.utils.llm_utils import (
        adapter_checkpoint_params,
        build_completion_mask,
        build_vllm_llm_init_kwargs,
        build_vllm_rollout_lora_request,
        expert_lora_vllm_key_map,
        gather_if_zero3,
        log_cuda_memory_snapshot,
        move_params_to_cpu,
        move_params_to_gpu,
        save_peft_adapter_for_vllm_rollout,
    )


if TYPE_CHECKING:
    from vllm import LLM, CompletionOutput, SamplingParams
elif HAS_VLLM:
    from vllm import LLM, CompletionOutput, SamplingParams
else:
    LLM = CompletionOutput = SamplingParams = None


logger = logging.getLogger(__name__)


def _vllm_sampled_token_logprobs(output: CompletionOutput) -> list[float]:
    """Per-token logprob of the *sampled* token from a vLLM ``CompletionOutput``.

    With ``SamplingParams(logprobs=0)`` vLLM returns, per generated position, a
    dict that always contains the sampled token. Missing or non-finite entries
    fall back to ``0.0`` (yielding a unit importance-sampling ratio for that
    token, since the correction multiplies the loss by ``exp(old - sampling)``).

    :param output: A vLLM ``CompletionOutput`` (``token_ids`` + ``logprobs``).
    :type output: CompletionOutput
    :return: One sampled-token logprob per generated token.
    :rtype: list[float]
    """
    token_ids = output.token_ids
    logprobs = getattr(output, "logprobs", None)
    if not logprobs:
        return [0.0] * len(token_ids)
    out: list[float] = []
    for tok, lp_dict in zip(token_ids, logprobs, strict=False):
        entry = lp_dict.get(tok) if lp_dict else None
        val = float(entry.logprob) if entry is not None else 0.0
        # Only finite logprobs count (rejects NaN and ±inf).
        out.append(val if np.isfinite(val) else 0.0)
    return out


class LLMVLLMMixin:
    """Colocated vLLM rollout and adapter sync for :class:`LLMAlgorithm`."""

    def _get_peft_model_for_vllm_sync(self) -> Any:  # noqa: ANN401 -- unwrapped PEFT model type varies (value-head wrapper vs bare PeftModel)
        """Unwrapped PEFT model used for vLLM weight / adapter sync."""
        model_ref = self._get_unwrapped_actor()
        return model_ref.pretrained_model if self.use_value_head else model_ref

    def _ensure_vllm_lora_staging_dir(self) -> Path:
        """Resolve (once) the dir the rollout LoRA adapter is exported to.

        The staging dir is always process-private: each rank exports its own
        adapter copy and reads it back locally, so ranks never race on shared
        files. Honours ``VLLMConfig.lora_staging_dir`` when set — e.g. a known
        path that orchestrated deployments expect the adapter under — staging
        in a ``rank_<process_index>`` subdirectory of that root when
        distributed. The dir is created (parents included) and marked
        non-temporary so ``clean_up`` never deletes it. Otherwise falls back
        to a process-private ``mkdtemp`` that ``clean_up`` removes.
        Idempotent: both the colocated init (``_configure_vllm``) and every
        adapter sync (``_move_lora_to_vllm``) call this, so the same directory
        is used throughout the agent's life.

        :return: The resolved staging directory.
        :rtype: pathlib.Path
        """
        if self._vllm_lora_staging_dir is None:
            configured = getattr(self.vllm_config, "lora_staging_dir", None)
            if configured is not None:
                staging_dir = Path(configured)
                if self.accelerator is not None and self.accelerator.num_processes > 1:
                    staging_dir = staging_dir / f"rank_{self.accelerator.process_index}"
                staging_dir.mkdir(parents=True, exist_ok=True)
                self._vllm_lora_staging_dir = staging_dir
                self._vllm_lora_staging_dir_is_temp = False
            else:
                self._vllm_lora_staging_dir = Path(
                    tempfile.mkdtemp(prefix="agilerl_vllm_lora_")
                )
                self._vllm_lora_staging_dir_is_temp = True
        return self._vllm_lora_staging_dir

    def _move_lora_to_vllm(self) -> None:
        """Export the actor LoRA adapter to disk and register it with vLLM.

        Adapter-only sync (colocated vLLM always serves LoRA): vLLM keeps its
        own base and only the LoRA delta is synced per rollout via
        ``llm_engine.add_lora``. Compatible with vLLM-side weight quantization
        (e.g. ``bitsandbytes`` for QLoRA rollouts).

        **Does not touch base weights.** vLLM owns its base across the
        native sleep/wake cycle; the trainer holds (and offloads) its own.
        """
        peft_ref = self._get_peft_model_for_vllm_sync()
        peft_ref.set_adapter(self._vllm_rollout_adapter)

        staging_dir = self._ensure_vllm_lora_staging_dir()
        with gather_if_zero3(self.zero_stage, adapter_checkpoint_params(peft_ref)):
            if self.lora_config is None:
                msg = "lora_config is required for vLLM LoRA adapter export."
                raise ValueError(msg)
            target_modules = self.lora_config.target_modules
            target_parameters = getattr(self.lora_config, "target_parameters", None)
            if not isinstance(target_parameters, (list, tuple)):
                target_parameters = None
            assert target_modules is not None or target_parameters, (
                "lora_config.target_modules or target_parameters is required "
                "for vLLM LoRA adapter export."
            )
            expert_key_map = (
                expert_lora_vllm_key_map(peft_ref) if target_parameters else None
            )
            adapter_path = save_peft_adapter_for_vllm_rollout(
                peft_ref,
                staging_dir,
                self._vllm_rollout_adapter,
                target_modules=target_modules,
                expert_key_map=expert_key_map,
            )
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        # One-shot refresh of the resident slot. ``load_inplace`` forces vLLM to
        # re-read the (updated) adapter weights from disk; required from the second
        # sync onward, when the slot already holds the previous step's adapter.
        refresh_request = build_vllm_rollout_lora_request(
            adapter_path,
            load_inplace=self._vllm_lora_loaded,
        )
        lora_device = torch.device(self.device)
        if lora_device.type == "cuda":
            # Pin the CUDA context to this agent's device: vLLM's LoRA copy
            # kernels otherwise launch on the process-default device.
            with torch.cuda.device(lora_device):
                loaded = self.llm.llm_engine.add_lora(refresh_request)
        else:
            loaded = self.llm.llm_engine.add_lora(refresh_request)
        if not loaded:
            msg = (
                "vLLM failed to load LoRA adapter from "
                f"{adapter_path}. Check max_lora_rank / target module "
                "names match the trainer."
            )
            raise RuntimeError(msg)

        # The request handed to ``generate()`` must NOT carry ``load_inplace``:
        # vLLM re-evaluates active LoRAs every decode step, and load_inplace would
        # reparse the full adapter from disk each step (disk-bound rollouts). The
        # one-shot add_lora above already refreshed the resident slot.
        self._vllm_rollout_lora_request = build_vllm_rollout_lora_request(
            adapter_path,
            load_inplace=False,
        )
        self._vllm_lora_loaded = True

    def _sync_actor_to_vllm(self) -> None:
        """Sync the trainer's actor LoRA adapter into the colocated vLLM engine.

        Colocated vLLM keeps its own base and always serves LoRA via
        ``add_lora``, so only the adapter is synced — see
        :meth:`_move_lora_to_vllm`. The bases are not shared.
        Idempotent within a rollout cycle: gated by ``self._vllm_moved``, which
        the wake path clears.
        """
        if self._vllm_moved:
            return
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        self._move_lora_to_vllm()

        self.llm.reset_prefix_cache()
        self._vllm_moved = True

    def _generate_with_vllm_colocate(
        self,
        prompts: Sequence[RolloutPrompt],
        group_size: int,
        temperature: float | None,
        capture_sampling_logps: bool = False,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None] | None
    ]:
        """Generate completions with colocated vLLM for GRPO/LLMPPO-style batches.

        Each entry in ``prompts`` is repeated ``group_size`` times so vLLM
        receives a flat list of length ``len(prompts) * group_size``
        (e.g. GRPO groups). Action masks use the full prompt length from
        ``input_ids``.

        :param prompts: Length-``N`` sequence of prompt mappings for this rank.
        :type prompts: Sequence[RolloutPrompt]
        :param group_size: Repeat factor per prompt (1 for plain PPO).
        :type group_size: int
        :param temperature: Temperature for sampling.
        :type temperature: float | None
        :return: Per-prompt completion token tensors and matching action masks.
        :rtype: tuple[list[torch.Tensor], list[torch.Tensor]]
        """
        if SamplingParams is None:
            msg = "vLLM is required when use_vllm=True. Install AgileRL with vLLM support for this platform: `pip install agilerl[llm]`."
            raise ImportError(msg)
        vllm_config = self.vllm_config
        assert vllm_config is not None, (
            "vllm_config must be configured for colocated vLLM generation."
        )

        max_token_cap = (
            self.max_output_tokens
            if self.max_output_tokens is not None
            else self.max_model_len
        )

        def _token_prompt_for_vllm(ids: torch.Tensor) -> dict[str, list[int]]:
            return {"prompt_token_ids": ids.squeeze(0).tolist()}

        def _vllm_max_new_tokens(model_prompt_len: int) -> int:
            room = self.max_model_len - model_prompt_len
            if room <= 0:
                error_msg = f"Model prompt length ({model_prompt_len}) is greater than the model length ({self.max_model_len})"
                raise ValueError(error_msg)
            max_out = min(max_token_cap, room)
            if self.min_output_tokens is not None:
                max_out = max(max_out, min(self.min_output_tokens, room))
            return min(max_out, room)

        # Compute the per-prompt work once per *unique* prompt (N items),
        # then alias by reference across each group (N·G items)
        unique_ids = [prompt["input_ids"] for prompt in prompts]
        unique_tokens = [_token_prompt_for_vllm(ids) for ids in unique_ids]
        unique_max = [_vllm_max_new_tokens(int(ids.shape[1])) for ids in unique_ids]

        # Replicate by reference for the flat vLLM batch. Entries within a
        # group of `group_size` are aliased references to the same tensor / dict
        # — safe because downstream use is read-only w.r.t. these objects.
        # Do not introduce in-place ops on these aliases.
        prompts_ids = [ids for ids in unique_ids for _ in range(group_size)]
        token_prompts = [tp for tp in unique_tokens for _ in range(group_size)]
        max_output_tokens = [m for m in unique_max for _ in range(group_size)]

        if vllm_config.tensor_parallel_size > 1:
            orig_size = len(token_prompts)

            gathered_prompts_ids: list[Any] = [
                None for _ in range(vllm_config.tensor_parallel_size)
            ]
            gathered_token_prompts: list[Any] = [
                None
            ] * vllm_config.tensor_parallel_size
            gathered_max_output_tokens: list[Any] = [
                None
            ] * vllm_config.tensor_parallel_size

            for gathered, obj in zip(
                (
                    gathered_prompts_ids,
                    gathered_token_prompts,
                    gathered_max_output_tokens,
                ),
                (prompts_ids, token_prompts, max_output_tokens),
                strict=True,
            ):
                torch.distributed.all_gather_object(gathered, obj, group=self.tp_group)

            all_prompts_ids = [
                prompt_id for sublist in gathered_prompts_ids for prompt_id in sublist
            ]
            all_token_prompts = [
                prompt for sublist in gathered_token_prompts for prompt in sublist
            ]
            all_max_output_tokens = [
                max_out for sublist in gathered_max_output_tokens for max_out in sublist
            ]
        else:
            all_token_prompts = token_prompts
            all_prompts_ids = prompts_ids
            all_max_output_tokens = max_output_tokens

        generation_kwargs: dict[str, Any] = {
            "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
            "repetition_penalty": self.repetition_penalty,
            "temperature": temperature,
            "top_p": self.top_p,
            "top_k": -1 if (self.top_k is None or self.top_k == 0) else self.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "min_tokens": (
                0 if self.min_output_tokens is None else self.min_output_tokens
            ),
            "presence_penalty": vllm_config.presence_penalty,
            "frequency_penalty": vllm_config.frequency_penalty,
        }
        if capture_sampling_logps:
            # logprobs=0 → vLLM returns the sampled token's logprob only.
            generation_kwargs["logprobs"] = 0
        if vllm_config.stop_sequences:
            generation_kwargs["stop"] = vllm_config.stop_sequences
        sampling_params = [
            SamplingParams(**generation_kwargs, max_tokens=max_output_token)
            for max_output_token in all_max_output_tokens
        ]

        generate_kwargs: dict[str, Any] = {
            "sampling_params": sampling_params,
            "use_tqdm": False,
        }
        if self.vllm_config is not None and self._vllm_rollout_lora_request is not None:
            generate_kwargs["lora_request"] = self._vllm_rollout_lora_request

        all_outputs = self.llm.generate(all_token_prompts, **generate_kwargs)

        generated_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        # Flat per-completion lists of the sampled tokens' logprobs (one float
        # per generated token), parallel to ``generated_ids``. Empty when not
        # capturing.
        sampling_logps_flat: list[list[float]] = (
            [
                _vllm_sampled_token_logprobs(output)
                for outputs in all_outputs
                for output in outputs.outputs
            ]
            if capture_sampling_logps
            else []
        )
        if vllm_config.tensor_parallel_size > 1:
            # Slice completions for this rank within its TP group.
            # Each rank generates all outputs — we keep only our share.
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(
                local_rank_in_group * orig_size,
                (local_rank_in_group + 1) * orig_size,
            )
            generated_ids = generated_ids[tp_slice]
            prompts_ids = all_prompts_ids[tp_slice]
            if capture_sampling_logps:
                sampling_logps_flat = sampling_logps_flat[tp_slice]

        # Transfer fromn host-to-device once per unique prompt, then re-alias across the group.
        unique_prompts_ids_dev = [
            prompts_ids[group_size * i].to(self.device, non_blocking=True)
            for i in range(len(prompts))
        ]
        prompts_ids = [ids for ids in unique_prompts_ids_dev for _ in range(group_size)]

        token_ids_list = [
            torch.cat(
                [
                    torch.cat(
                        prompts_ids[group_size * i : group_size * (i + 1)],
                        dim=0,
                    ),
                    stack_and_pad_experiences(
                        generated_ids[group_size * i : group_size * (i + 1)],
                        padding_values=[self.pad_token_id],
                        device=self.device,
                    )[0],
                ],
                dim=1,
            )
            for i in range(len(prompts))
        ]

        sampling_logps: list[torch.Tensor | None] | None = (
            [
                torch.tensor(lp, dtype=torch.float32, device=self.device)
                for lp in sampling_logps_flat
            ]
            if capture_sampling_logps
            else None
        )

        num_input_tokens = [
            int(prompts[i]["input_ids"].shape[1]) for i in range(len(prompts))
        ]
        completion_masks = [
            build_completion_mask(token_ids, num_input_tokens[i], self.pad_token_id)
            for i, token_ids in enumerate(token_ids_list)
        ]

        return token_ids_list, completion_masks, sampling_logps

    @staticmethod
    def _logprobs_from_logits(
        logits: torch.Tensor,
        index: torch.Tensor,
        cast_to_fp32: bool = True,
        chunk_rows: int = 1,
    ) -> torch.Tensor:
        """Calculate log probabilities for previously generated token ids.

        Processes ``chunk_rows`` rows at a time so peak memory stays bounded to
        ``(chunk_rows, seq_len, vocab_size)`` rather than the full batch, avoiding
        OOM on large-vocabulary models. Default ``chunk_rows=1`` minimizes the
        fp32 workspace at the cost of more kernel launches; raise to amortize
        launch overhead when memory headroom allows.

        With ``cast_to_fp32=True``, the per-chunk reduction (``amax`` /
        ``gather`` / ``logsumexp``) runs in fp32 then casts the
        ``(B, seq_len)`` output back to *logits* dtype. Matches the precision
        of ``F.log_softmax`` over the same inputs to within the final bf16
        cast. With ``cast_to_fp32=False`` the reduction stays in *logits*
        dtype throughout — faster and lower peak (no fp32 workspace) at the
        cost of bf16-quantisation error in the reduction.

        Logits are max-centered per row before ``logsumexp``, matching
        ``F.log_softmax`` stability either way.

        :param logits: Logits of shape ``(B, seq_len, vocab_size)``.
        :type logits: torch.Tensor
        :param index: Token IDs of shape ``(B, seq_len)``.
        :type index: torch.Tensor
        :param cast_to_fp32: Promote each chunk to fp32 before the reduction.
        :type cast_to_fp32: bool
        :return: Log probabilities of the completion IDs, shape ``(B, seq_len)``.
        :rtype: torch.Tensor
        """
        orig_dtype = logits.dtype
        B = logits.shape[0]

        def _logprobs_chunk(lg: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            if cast_to_fp32:
                lg = lg.float()
            max_lg = lg.amax(dim=-1, keepdim=True)
            shifted = lg - max_lg
            target = shifted.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
            log_z = torch.logsumexp(shifted, dim=-1)
            result = target - log_z
            return result.to(orig_dtype) if cast_to_fp32 else result

        if B <= chunk_rows:
            return _logprobs_chunk(logits, index)

        per_token_logps = []
        for start in range(0, B, chunk_rows):
            end = min(start + chunk_rows, B)
            per_token_logps.append(
                _logprobs_chunk(logits[start:end], index[start:end]),
            )
        return torch.cat(per_token_logps, dim=0)

    @staticmethod
    def _resolve_fused_chunk_rows(vocab_size: int, explicit: int | None = None) -> int:
        """Rows per fused ``(chunk_rows, vocab)`` logit tile.

        Shared by the fused-linear-logprob (standard) path and the Liger
        fused-loss path so both bound their per-chunk logit workspace
        identically. A positive ``explicit`` overrides; ``None``
        auto-tunes to a ~256 MB fp32 logit workspace (fewer rows at larger
        vocab), clamped to ``[128, 4096]``.

        :param vocab_size: lm_head output dim (rows of the logit tile's V axis).
        :type vocab_size: int
        :param explicit: Explicit override, or ``None`` to auto-tune.
        :type explicit: int | None
        :return: Rows per chunk.
        :rtype: int
        """
        if explicit is not None:
            return explicit
        workspace_bytes = 256 * 1024 * 1024
        return min(max(workspace_bytes // max(1, vocab_size * 4), 128), 4096)

    @staticmethod
    def _logprobs_from_hidden_fused(
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float = 1.0,
        cast_to_fp32: bool = True,
        chunk_rows: int | None = None,
    ) -> torch.Tensor:
        """Per-token target logprobs without materializing the full ``(B, T, V)``
        logits tensor.

        Tiles flat over ``(B*T)`` with workspace bounded to ``(chunk_rows, V)``
        per iteration. Counterpart of :meth:`_logprobs_from_logits` for
        callers that hold hidden states and the lm_head separately. **No-grad
        only** — gradients won't flow to ``lm_head_weight`` from this fn. The
        gradient-aware counterpart is :meth:`_logprobs_from_hidden_fused_grad`.

        Numerical contract matches :meth:`_logprobs_from_logits` when fed
        equivalent inputs (``logits = (hidden @ Wᵀ + b) / T``): same
        ``cast_to_fp32`` semantics, same final-cast-back-to-input-dtype, same
        max-shift ``gather - logsumexp`` formulation. Default ``cast_to_fp32=True``
        keeps the two paths bit-comparable.

        :param hidden: ``(B, T, H)`` last-hidden-state.
        :type hidden: torch.Tensor
        :param lm_head_weight: ``(V, H)``.
        :type lm_head_weight: torch.Tensor
        :param lm_head_bias: ``(V,)`` or ``None``.
        :type lm_head_bias: torch.Tensor | None
        :param target_ids: ``(B, T)`` (caller does the ``[:, :-1]``/``[:, 1:]``
            shift before calling).
        :type target_ids: torch.Tensor
        :param temperature: scalar; logits divided by this before log_softmax
            (skipped when ``1.0``).
        :type temperature: float, optional
        :param cast_to_fp32: when True (default), run the per-chunk reduction
            in fp32 then cast back. Same semantics as
            :meth:`_logprobs_from_logits`.
        :type cast_to_fp32: bool, optional
        :param chunk_rows: rows of the flattened ``(B*T)`` workspace per
            iteration; trades launch count vs ``chunk_rows * V`` peak. When
            ``None`` (default) it is resolved from the vocab size via
            a ~256 MB fp32 workspace heuristic.
        :type chunk_rows: int | None, optional
        :return: ``(B, T)`` per-token logprobs in ``hidden.dtype``.
        :rtype: torch.Tensor
        """
        chunk_rows = LLMVLLMMixin._resolve_fused_chunk_rows(
            getattr(lm_head_weight, "ds_shape", lm_head_weight.shape)[0],
            chunk_rows,
        )
        return fused_linear_logprobs_chunked(
            hidden,
            lm_head_weight,
            lm_head_bias,
            target_ids,
            temperature=temperature,
            cast_to_fp32=cast_to_fp32,
            chunk_rows=chunk_rows,
        )

    @staticmethod
    def _logprobs_from_hidden_fused_grad(
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float = 1.0,
        cast_to_fp32: bool = True,
        chunk_rows: int | None = None,
    ) -> torch.Tensor:
        """Gradient-aware version of :meth:`_logprobs_from_hidden_fused`.

        Routes through :class:`FusedLinearLogProbsFunction` so the per-token
        logprobs are differentiable w.r.t. ``hidden`` (and ``lm_head_weight`` /
        bias when they require grad) while never materializing the full
        ``(B, T, V)`` logits tensor in the forward *or* backward pass — the
        lm_head matmul is gradient-checkpointed and recomputed chunk-by-chunk.

        Forward values are bit-comparable to :meth:`_logprobs_from_hidden_fused`
        (and hence :meth:`_logprobs_from_logits`); the gradient equals the exact
        ``log_softmax`` gradient.

        :param hidden: ``(B, T, H)`` last-hidden-state (typically requires grad).
        :type hidden: torch.Tensor
        :param lm_head_weight: ``(V, H)``.
        :type lm_head_weight: torch.Tensor
        :param lm_head_bias: ``(V,)`` or ``None``.
        :type lm_head_bias: torch.Tensor | None
        :param target_ids: ``(B, T)`` (caller does the shift before calling).
        :type target_ids: torch.Tensor
        :param temperature: logits divided by this before log_softmax.
        :type temperature: float, optional
        :param cast_to_fp32: run the per-chunk reduction in fp32.
        :type cast_to_fp32: bool, optional
        :param chunk_rows: rows of the flattened ``(B*T)`` workspace per chunk.
            When ``None`` (default) it is resolved from the vocab size via
            a ~256 MB fp32 workspace heuristic.
        :type chunk_rows: int | None, optional
        :return: ``(B, T)`` per-token logprobs in ``hidden.dtype``.
        :rtype: torch.Tensor
        """
        chunk_rows = LLMVLLMMixin._resolve_fused_chunk_rows(
            getattr(lm_head_weight, "ds_shape", lm_head_weight.shape)[0],
            chunk_rows,
        )
        return FusedLinearLogProbsFunction.apply(
            hidden,
            lm_head_weight,
            lm_head_bias,
            target_ids,
            temperature,
            cast_to_fp32,
            chunk_rows,
        )

    def _configure_batch_size_per_process(
        self,
        batch_size: int,
        micro_batch_size_per_gpu: int | None,
        mini_batch_size: int | None,
    ) -> None:
        if mini_batch_size is not None and mini_batch_size < 1:
            msg = f"mini_batch_size must be a positive integer; got {mini_batch_size}."
            raise ValueError(msg)
        if self.accelerator is None:
            self.batch_size_per_process = batch_size
            if micro_batch_size_per_gpu is not None:
                self.micro_batch_size_per_gpu = int(micro_batch_size_per_gpu)
            else:
                self.micro_batch_size_per_gpu = batch_size
            if (
                mini_batch_size is not None
                and int(mini_batch_size) != self.micro_batch_size_per_gpu
            ):
                msg = (
                    f"mini_batch_size ({mini_batch_size}) requires a DeepSpeed "
                    "engine to accumulate gradients across micro-batches; "
                    "without one every micro-batch of "
                    f"{self.micro_batch_size_per_gpu} takes its own optimizer "
                    "step."
                )
                raise ValueError(msg)
            self.mini_batch_size = self.micro_batch_size_per_gpu
            return

        ds_plugin = self.accelerator.state.deepspeed_plugin
        if ds_plugin is None:
            err_msg = """DeepSpeed plugin is not initialized. If using an accelerator,
            ensure to launch your training script with `accelerate launch --num_processes <your_script.py>`."""
            raise ValueError(err_msg)
        ds_config = ds_plugin.deepspeed_config

        if batch_size % self.accelerator.num_processes != 0:
            msg = f"Batch size ({batch_size}) must be divisible by the number of processes ({self.accelerator.num_processes})."
            raise ValueError(
                msg,
            )

        self.batch_size_per_process = int(batch_size / self.accelerator.num_processes)

        if micro_batch_size_per_gpu is None and mini_batch_size is None:
            if (
                self.batch_size_per_process
                % ds_config.get("gradient_accumulation_steps", 1)
                != 0
            ):
                msg = (
                    f"Batch size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and gradient accumulation steps ({ds_config.get('gradient_accumulation_steps', 1)})."
                    "Gradient accumulation steps can be updated in the deepspeed config by changing the 'gradient_accumulation_steps' parameter."
                )
                raise ValueError(
                    msg,
                )

            gradient_accumulation_steps = ds_config.get(
                "gradient_accumulation_steps", 1
            )
            self.micro_batch_size_per_gpu = (
                self.batch_size_per_process // gradient_accumulation_steps
            )
            self.mini_batch_size = (
                self.micro_batch_size_per_gpu * gradient_accumulation_steps
            )

            prev_micro = ds_config.get("train_micro_batch_size_per_gpu")
            if prev_micro is not None:
                warnings.warn(
                    "Overwriting DeepSpeed config train_micro_batch_size_per_gpu "
                    f"from {prev_micro!r} to {self.micro_batch_size_per_gpu} "
                    f"(batch_size_per_process={self.batch_size_per_process} "
                    f"// gradient_accumulation_steps={gradient_accumulation_steps}).",
                    stacklevel=2,
                )
            ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
            return

        if micro_batch_size_per_gpu == 0:
            msg = (
                "micro_batch_size_per_gpu is equal to zero, which is not allowed. "
                "Please set micro_batch_size_per_gpu to a positive integer."
            )
            raise ValueError(msg)

        if micro_batch_size_per_gpu is not None:
            self.micro_batch_size_per_gpu = int(micro_batch_size_per_gpu)
            if (
                batch_size
                % (self.micro_batch_size_per_gpu * self.accelerator.num_processes)
                != 0
            ):
                msg = f"When specifying micro_batch_size_per_gpu, batch_size ({batch_size}) must be divisible by the product of the number of processes ({self.accelerator.num_processes}) and micro_batch_size_per_gpu ({self.micro_batch_size_per_gpu})."
                raise ValueError(
                    msg,
                )
        elif mini_batch_size is not None:
            self.micro_batch_size_per_gpu = int(mini_batch_size)

        if mini_batch_size is not None:
            self.mini_batch_size = int(mini_batch_size)
        elif self._mini_batch_size_default == "micro_batch":
            self.mini_batch_size = self.micro_batch_size_per_gpu
        else:
            self.mini_batch_size = self.batch_size_per_process
        if self.mini_batch_size % self.micro_batch_size_per_gpu != 0:
            msg = (
                f"mini_batch_size ({self.mini_batch_size}) must be divisible by "
                f"micro_batch_size_per_gpu ({self.micro_batch_size_per_gpu}): "
                "gradient_accumulation_steps = mini_batch_size / "
                "micro_batch_size_per_gpu must be a whole number of backward "
                "passes."
            )
            raise ValueError(msg)

        prev_micro = ds_config.get("train_micro_batch_size_per_gpu")
        if prev_micro is not None:
            warnings.warn(
                "Overwriting DeepSpeed config train_micro_batch_size_per_gpu "
                f"from {prev_micro!r} to {self.micro_batch_size_per_gpu} ",
                stacklevel=2,
            )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size_per_gpu
        gradient_accumulation_steps = (
            self.mini_batch_size // self.micro_batch_size_per_gpu
        )
        prev_accumulation = ds_config.get("gradient_accumulation_steps")
        if (
            prev_accumulation not in (None, "auto")
            and int(prev_accumulation) != gradient_accumulation_steps
        ):
            warnings.warn(
                "Overwriting DeepSpeed config gradient_accumulation_steps from "
                f"{prev_accumulation!r} to {gradient_accumulation_steps} "
                f"(mini_batch_size={self.mini_batch_size} // "
                f"micro_batch_size_per_gpu={self.micro_batch_size_per_gpu}).",
                stacklevel=2,
            )
        ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        return

    def recompile(self) -> None:
        """Recompile evolvable modules with ``torch.compile``.

        Iterates over ``evolvable_attributes`` and compiles each one.
        Skipped when DeepSpeed is active because ``DeepSpeedEngine`` is not
        compatible with ``OptimizedModule`` wrapping.
        """
        if self.torch_compiler is None or self._uses_deepspeed:
            return
        for name, obj in self.evolvable_attributes(networks_only=True).items():
            setattr(self, name, compile_model(obj, self.torch_compiler))

    def _update_existing_adapter(
        self,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        """Overwrite weights of an existing adapter in-place without creating new parameters.

        :param checkpoint_dir: Checkpoint directory
        :type checkpoint_dir: str
        :param adapter_name: Adapter name
        :type adapter_name: str.

        :return: None
        :rtype: None
        """
        unwrapped = self._get_unwrapped_actor()
        peft_model = unwrapped.pretrained_model if self.use_value_head else unwrapped

        adapter_path = f"{checkpoint_dir}/{adapter_name}/adapter_model.safetensors"
        adapter_state = load_file(adapter_path, device=str(self.device))

        with gather_if_zero3(
            self.zero_stage,
            adapter_checkpoint_params(unwrapped),
            modifier_rank=0,
        ):
            with torch.no_grad():
                set_peft_model_state_dict(
                    peft_model,
                    adapter_state,
                    adapter_name=adapter_name,
                )
            peft_model.set_adapter(adapter_name)

            for name, param in unwrapped.named_parameters():
                if "reference" in name:
                    param.requires_grad = False
                elif "actor" in name or "critic" in name:
                    param.requires_grad = True

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _copy_adapter_weights(self, source_adapter: str, target_adapter: str) -> None:
        """Copy LoRA weights from source adapter to target adapter."""
        source_params = {}
        target_params = {}
        for name, param in self.actor.named_parameters():
            if "lora" not in name:
                continue
            if f".{source_adapter}." in name:
                key = name.replace(f".{source_adapter}.", ".", 1)
                source_params[key] = param
            elif f".{target_adapter}." in name:
                key = name.replace(f".{target_adapter}.", ".", 1)
                target_params[key] = param

        if not source_params:
            msg = f"No LoRA tensors found for source adapter '{source_adapter}'."
            raise ValueError(
                msg,
            )
        if not target_params:
            msg = f"No LoRA tensors found for target adapter '{target_adapter}'."
            raise ValueError(
                msg,
            )

        missing = [key for key in source_params if key not in target_params]
        if missing:
            msg = (
                f"Target adapter '{target_adapter}' is missing {len(missing)} LoRA tensors "
                f"present in source adapter '{source_adapter}'."
            )
            raise ValueError(
                msg,
            )

        lora_params = list(source_params.values()) + list(target_params.values())
        with gather_if_zero3(self.zero_stage, lora_params, modifier_rank=0):
            for key, src_param in source_params.items():
                target_params[key].data.copy_(src_param.data)

    @staticmethod
    def _load_checkpoint_lora_config(path: str) -> LoraConfig | None:
        """Load the ``actor`` adapter's LoRA config from a checkpoint directory, if present.

        :param path: Directory previously written by :meth:`save_checkpoint`.
        :type path: str
        :return: The ``LoraConfig`` stored alongside the actor adapter, or ``None`` if
            the checkpoint does not contain one (legacy checkpoint, or no ``actor/`` subdir).
        :rtype: peft.LoraConfig | None
        """
        config_path = Path(path) / "actor" / "adapter_config.json"
        if not config_path.is_file():
            return None
        return LoraConfig.from_pretrained(str(config_path.parent))

    @staticmethod
    def _format_lora_config_mismatch_error(
        current: LoraConfig,
        checkpoint: LoraConfig,
    ) -> str:
        """Format a user-facing error for mismatched LoRA configs.

        :param current: LoRA config from the live loading agent.
        :type current: peft.LoraConfig
        :param checkpoint: LoRA config persisted in the checkpoint.
        :type checkpoint: peft.LoraConfig
        :return: Error string with mismatch context and remediation.
        :rtype: str
        """

        def summarize(cfg: LoraConfig) -> dict[str, Any]:
            """Summarize key LoRA config fields for mismatch messages."""
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
            summary_keys = (
                "r",
                "lora_alpha",
                "target_modules",
                "modules_to_save",
                "bias",
                "task_type",
            )
            summary = {key: cfg_dict.get(key) for key in summary_keys}
            for key in ("target_modules", "modules_to_save"):
                value = summary.get(key)
                if isinstance(value, (set, tuple)):
                    summary[key] = sorted(value)
            return summary

        current_summary = summarize(current)
        checkpoint_summary = summarize(checkpoint)
        return (
            "LoRA configs differ; refusing to load the checkpoint.\n"
            f"Current config: {current_summary}\n"
            f"Checkpoint config: {checkpoint_summary}\n"
            "Resolution: re-create the agent with the checkpoint's LoRA config "
            "before calling load_checkpoint."
        )

    @staticmethod
    def _lora_configs_equivalent(a: LoraConfig, b: LoraConfig) -> bool:
        """Structural equality for two ``LoraConfig`` instances.

        List/tuple/set-typed fields (``target_modules`` etc.) are normalised to sorted
        lists before comparison so insertion order does not matter.

        :param a: First config.
        :type a: peft.LoraConfig
        :param b: Second config.
        :type b: peft.LoraConfig
        :return: ``True`` iff every keyword field is equal after normalisation.
        :rtype: bool
        """
        ignore_keys = {"inference_mode"}
        ordered_keys = ("target_modules", "modules_to_save", "exclude_modules")
        a_dict = a.to_dict() if hasattr(a, "to_dict") else dict(vars(a))
        b_dict = b.to_dict() if hasattr(b, "to_dict") else dict(vars(b))
        for key in ordered_keys:
            for d in (a_dict, b_dict):
                val = d.get(key)
                if isinstance(val, (list, tuple, set)):
                    d[key] = sorted(val)
        for key in ignore_keys:
            a_dict.pop(key, None)
            b_dict.pop(key, None)
        return a_dict == b_dict

    def _load_adapter_weights(
        self,
        checkpoint_dir: str,
        adapter_name: str,
    ) -> None:
        """Overwrite a live adapter's weights from disk.

        :param checkpoint_dir: Directory written by :meth:`save_checkpoint`; must contain
            ``<adapter_name>/adapter_model.safetensors``.
        :type checkpoint_dir: str
        :param adapter_name: Name of the adapter to overwrite (must already exist on the
            live PEFT model).
        :type adapter_name: str
        :return: None. Mutates the live adapter's parameters in place.
        :rtype: None
        """
        unwrapped = self._get_unwrapped_actor()
        peft_model = unwrapped.pretrained_model if self.use_value_head else unwrapped

        adapter_path = f"{checkpoint_dir}/{adapter_name}/adapter_model.safetensors"
        adapter_state = load_file(adapter_path, device=str(self.device))

        with gather_if_zero3(
            self.zero_stage,
            adapter_checkpoint_params(unwrapped),
            modifier_rank=0,
        ):
            with torch.no_grad():
                set_peft_model_state_dict(
                    peft_model, adapter_state, adapter_name=adapter_name
                )
            peft_model.set_adapter(adapter_name)

            for name, param in unwrapped.named_parameters():
                if "reference" in name:
                    param.requires_grad = False

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @staticmethod
    def _create_prompt_masks(
        prompt_lengths: list[int], max_length: int
    ) -> torch.Tensor:
        """Create a mask for the prompts based on the prompt lengths (vectorized).

        :param prompt_lengths: List of prompt lengths
        :type prompt_lengths: list[int]
        :param max_length: Maximum length of the prompts
        :type max_length: int
        :return: Mask tensor [batch_size, max_length]
        :rtype: torch.Tensor
        """
        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
        positions = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        # The first response token sits AT index prompt_length, so the mask must
        # include it — a strict ``>`` silently drops it from every loss.
        return positions >= prompt_lengths_tensor.unsqueeze(1)

    def _configure_vllm(self) -> None:
        """Configure vLLM for efficient inference during generation in 'get_action'."""
        if LLM is None:
            msg = "vLLM is required when use_vllm=True. Install AgileRL with vLLM support for this platform: `pip install agilerl[llm]`."
            raise ImportError(msg)
        if self.vllm_config is None:
            warnings.warn(
                "No VLLM config provided. Using default VLLM configuration for generation.",
                stacklevel=2,
            )
            self.vllm_config = VLLMConfig()
        num_processes = (
            self.accelerator.num_processes if self.accelerator is not None else 1
        )
        process_index = (
            self.accelerator.process_index if self.accelerator is not None else 0
        )
        local_process_index = (
            self.accelerator.local_process_index if self.accelerator is not None else 0
        )
        if num_processes % self.vllm_config.tensor_parallel_size != 0:
            msg = f"Tensor parallel size {self.vllm_config.tensor_parallel_size} must be a multiple of the number of processes {num_processes}."
            raise ValueError(
                msg,
            )

        if self.vllm_config.tensor_parallel_size > 1:
            # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
            # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
            self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                [
                    list(
                        range(
                            i * self.vllm_config.tensor_parallel_size,
                            (i + 1) * self.vllm_config.tensor_parallel_size,
                        ),
                    )
                    for i in range(
                        num_processes // self.vllm_config.tensor_parallel_size,
                    )
                ],
            )

        # vLLM requires the environment variables to be set for distributed training.
        os.environ["RANK"] = str(process_index)
        os.environ["LOCAL_RANK"] = str(local_process_index)
        os.environ["WORLD_SIZE"] = str(num_processes)
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

        lora_rank = getattr(self.lora_config, "r", None) if self.lora_config else None
        llm_kwargs = build_vllm_llm_init_kwargs(
            self.vllm_config,
            trainer_model_name_or_path=self.pretrained_model_name_or_path,
            max_model_len=self.max_model_len,
            process_index=process_index,
            lora_rank=lora_rank,
        )
        self._ensure_vllm_lora_staging_dir()
        if self.accelerator is None or self.accelerator.process_index == 0:
            warnings.warn(
                f"colocated init: starting vLLM LLM() with "
                f"max_num_batched_tokens={llm_kwargs.get('max_num_batched_tokens')} "
                f"(max_num_seqs={self.vllm_config.max_num_seqs} "
                f"max_model_len={self.max_model_len})",
                stacklevel=2,
            )

        if getattr(self.lora_config, "target_parameters", None) and (
            patch_vllm_3d_moe_lora_flag(llm_kwargs["model"])
        ):
            logger.info(
                "Marked vLLM %s as taking stacked-3D MoE LoRA adapters.",
                llm_kwargs["model"],
            )

        try:
            self.llm = LLM(**llm_kwargs)
        except ValueError as err:
            backend_env = os.environ.get("VLLM_ATTENTION_BACKEND")
            if backend_env is not None and "backend" in str(err).lower():
                msg = (
                    "vLLM initialization failed due to unsupported "
                    f"VLLM_ATTENTION_BACKEND={backend_env!r}. "
                    "Please unset VLLM_ATTENTION_BACKEND or set it to a backend "
                    "supported by your installed vLLM build."
                )
                raise ValueError(msg) from err
            raise

        # Keep the persistent rollout-adapter slot resident (vLLM V1 otherwise
        # zeroes it on dummy batches and never re-copies it, so the trained
        # adapter would contribute nothing); see ``patch_vllm_lora_keep_resident``.
        # Must run after the in-process engine (and its LoRA layers) exist.
        patched = patch_vllm_lora_keep_resident(self.llm)
        if (
            self.accelerator is None or self.accelerator.process_index == 0
        ) and patched:
            warnings.warn(
                f"colocated init: kept {patched} vLLM LoRA slots resident "
                "(works around vLLM zeroing the rollout adapter slot).",
                stacklevel=2,
            )

        strip_towers = getattr(self.vllm_config, "strip_multimodal_towers", False)
        if strip_towers:
            # Free unused vision/audio towers on multimodal bases (text-only RL
            # never runs them); see ``patch_vllm_strip_multimodal_towers``.
            freed = patch_vllm_strip_multimodal_towers(
                self.llm,
                tower_attrs=strip_towers if isinstance(strip_towers, list) else None,
            )
            if (
                self.accelerator is None or self.accelerator.process_index == 0
            ) and freed:
                total_params = sum(freed.values())
                detail = ", ".join(
                    f"{path}={count / 1e6:.1f}M" for path, count in freed.items()
                )
                warnings.warn(
                    f"colocated init: stripped multimodal towers "
                    f"({total_params / 1e6:.1f}M params freed: {detail}).",
                    stacklevel=2,
                )

        if self.vllm_config.sleep_mode:
            # Native sleep: back the base up to CPU and free the KV cache, so
            # the trainer's own base can use the GPU during the training step.
            self._sleep_vllm_after_init()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def _sleep_vllm_after_init(self) -> None:
        """Put the colocated engine to sleep once after construction.

        Native ``sleep(level=sleep_mode_level)``: vLLM cycles its allocator
        state based on the configured sleep level; ``wake_up()`` restores the
        engine allocations.
        """
        assert self.vllm_config is not None  # _configure_vllm guarantees a config
        self.llm.sleep(level=self.vllm_config.sleep_mode_level)
        self._vllm_awake = False
        if self.accelerator is None or self.accelerator.is_main_process:
            log_cuda_memory_snapshot("vLLM sleep complete")

    def _sync_deepspeed_gradient_clipping(self) -> None:
        """Synchronize max_grad_norm with DeepSpeed gradient_clipping config.
        Registered as a mutation hook to ensure consistency after mutations.
        """
        if self.accelerator is None:
            return

        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is None:
            return

        ds_config = ds_plugin.deepspeed_config
        if "gradient_clipping" not in ds_config:
            return

        if ds_config["gradient_clipping"] != self.max_grad_norm:
            ds_config["gradient_clipping"] = self.max_grad_norm

        if hasattr(self.actor, "optimizer"):
            if hasattr(self.actor.optimizer, "grad_clip"):
                self.actor.optimizer.grad_clip = self.max_grad_norm
            if hasattr(self.actor.optimizer, "clip_grad"):
                self.actor.optimizer.clip_grad = self.max_grad_norm

    def _get_lm_head_parent(self) -> tuple[Any, str]:
        """Locate the parent module owning ``lm_head`` (or ``embed_out``).

        Walks through value-head, PEFT, and LoRA wrappers to the inner
        causal-LM that exposes the language-model head as an attribute.
        Returned so that callers can both read the head (``getattr(parent,
        attr)``) and replace it temporarily (``setattr(parent, attr, ...)``)
        — the latter is used by the no-grad fused-linear-logprob path.

        :return: ``(parent_module, attr_name)``.
        :rtype: tuple[Any, str]
        :raises AttributeError: If no lm_head can be found.
        """
        model = self.actor
        if self.use_value_head and hasattr(model, "pretrained_model"):
            # Value-head wrapper (e.g. AutoModelForCausalLMWithValueHead) →
            # the PEFT/causal-LM inner model.
            model = model.pretrained_model
        if hasattr(model, "base_model"):  # PeftModel → LoraModel
            model = model.base_model
        if hasattr(model, "model"):  # LoraModel → CausalLM
            model = model.model
        for attr in ("lm_head", "embed_out"):
            if hasattr(model, attr):
                return model, attr
        err_msg = (
            f"Cannot find lm_head (or embed_out) in {type(self.actor).__name__}. "
            "The fused-linear-logprob path needs the output embedding layer to "
            "compute per-token log-probs without materializing full logits."
        )
        raise AttributeError(err_msg)

    def _get_lm_head(self) -> torch.nn.Linear:
        """Locate the lm_head module, handling value-head, PEFT and LoRA wrappers.

        :return: The lm_head (or embed_out) linear layer.
        :rtype: torch.nn.Linear
        :raises AttributeError: If no lm_head can be found.
        """
        parent, attr = self._get_lm_head_parent()
        return getattr(parent, attr)

    @contextmanager
    def _patch_lm_head_to_identity(self) -> Generator[torch.nn.Module, None, None]:
        """Temporarily replace ``lm_head`` with ``nn.Identity``.

        With the head identity-patched, the model's ``output.logits`` becomes
        the post-final-norm hidden state ``(B, T, H)`` instead of the full
        ``(B, T, V)`` logits — which is what the no-grad fused-linear-logprob
        kernel consumes directly. The original module is always restored,
        even if the wrapped block raises.
        """
        model, attr = self._get_lm_head_parent()
        original = getattr(model, attr)
        setattr(model, attr, torch.nn.Identity())
        try:
            yield original
        finally:
            setattr(model, attr, original)

    def _get_unwrapped_actor(self) -> Any:  # noqa: ANN401 -- actor spans PEFT/DeepSpeed/value-head/DummyEvolvable wrappers
        """Return actor unwrapped from Accelerate and DummyEvolvable layers."""
        actor = (
            self.accelerator.unwrap_model(self.actor)
            if self.accelerator is not None
            else self.actor
        )
        while isinstance(actor, DummyEvolvable):
            actor = actor.module
        return actor

    @contextmanager
    def _memory_efficient_params(
        self,
    ) -> Generator[None, None, None]:  # pragma: no cover
        """Hold the trainer base on GPU only for the wrapped (training) block.

        Used by the colocated path (``use_memory_efficient_params``): the
        trainer's own base normally rests on CPU so the rollout engine owns the
        GPU; this moves it onto the GPU for the forward/backward and back to CPU
        afterwards, so the two bases never coexist on the GPU (no 2x-base peak).
        Disabled under DeepSpeed ZeRO-3 (params are already sharded; see init).
        """
        if self.zero_stage == 3:
            yield
            return
        unwrapped_model = self._get_unwrapped_actor()
        move_params_to_gpu(unwrapped_model, torch.device(self.device))
        try:
            yield
        finally:
            # Always move the base back on CPU on error
            move_params_to_cpu(unwrapped_model)

    def _prepare_vllm_for_training(self) -> None:
        """Prepare vLLM for learning."""
        if not self.use_vllm:
            return
        assert self.vllm_config is not None  # _configure_vllm guarantees a config
        # Every rank holds its own colocated engine (external_launcher), so
        # every rank must sleep it — not just the main process.
        if self.vllm_config.sleep_mode and self._vllm_awake:
            torch.cuda.empty_cache()
            self.llm.sleep(level=self.vllm_config.sleep_mode_level)
            self._vllm_awake = False

        self._vllm_moved = False

    def _prepare_vllm_for_generation(self) -> None:
        assert self.vllm_config is not None  # _configure_vllm guarantees a config
        if self.use_memory_efficient_params and self.zero_stage != 3:
            # Colocated: park the trainer's own base on CPU *before* waking vLLM
            # so the rollout engine owns the GPU and the two bases never coexist
            # on-device. The training step brings it back via
            # ``memory_efficient_params_context``. Skipped under ZeRO-3 — params
            # are already sharded and must not be moved with ``.to("cpu")``.
            moved = move_params_to_cpu(self._get_unwrapped_actor())
            if moved and (self.accelerator is None or self.accelerator.is_main_process):
                log_cuda_memory_snapshot(
                    "trainer base offloaded to CPU (before vLLM wake)"
                )
        # Every rank holds its own colocated engine, and _sleep_vllm_after_init
        # slept them all; we wake them all here.
        if self.vllm_config.sleep_mode and not self._vllm_awake:
            torch.cuda.empty_cache()
            self.llm.wake_up()
            self._vllm_awake = True
            if self.accelerator is None or self.accelerator.is_main_process:
                log_cuda_memory_snapshot("vLLM base restored on GPU (after wake)")
        self._sync_actor_to_vllm()
