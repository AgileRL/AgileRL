# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel ``forward`` for packed experts: dispatch, run local experts, combine.

The wrapped ``forward`` keeps ``agilerl.lora.moe`` EP-blind: it remaps ids to
the local expert range and runs the existing kernel on this rank's rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from types import MethodType
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.func import functional_call
from typing_extensions import Self

from agilerl.distributed.ep_dispatch import (
    TokenDispatchState,
    all_to_all_single_autograd,
    dispatch_state,
    exchange_expert_counts,
    token_combine,
    token_dispatch,
    unpermute_from_local_expert_major,
)
from agilerl.distributed.ep_sharding import expert_local_tensor, module_ep_degree
from agilerl.distributed.replicated_rows import (
    GatherReplicatedRows,
    SliceReplicatedRows,
    replicated_row_span,
)
from agilerl.lora.fused import ROUTING_STATE
from agilerl.lora.moe.adapters import mixed_routing
from agilerl.lora.moe.grouped_gemm import expert_row_counts
from agilerl.lora.moe.layouts import (
    is_routed_experts_module,
    is_sorted_experts_module,
    routed_projection_names,
)
from agilerl.lora.moe.routed import ROUTED_EXPERT_CHUNK_BYTES
from agilerl.lora.moe.wrappers import RoutedExpertsLoraWrapper


@contextmanager
def _gathered_expert_block(module: nn.Module) -> Iterator[bool]:
    """Gather leftover-dp expert shards for the duration of the local kernel.

    The EP-local block is split across the data-parallel pair. The token
    dispatch and the packed-expert kernel both index that whole block.
    Yields whether an FSDP gather happened.
    """
    base = module
    get_base = getattr(module, "get_base_layer", None)
    if callable(get_base):
        base = get_base()
    unshard = getattr(base, "unshard", None)
    reshard = getattr(base, "reshard", None)
    if not callable(unshard) or not callable(reshard):
        yield False
        return
    unshard()
    try:
        yield True
    finally:
        reshard()


class TokenAdapterIds(NamedTuple):
    """Adapter names under mixed fused routing and each token's index into them."""

    names: list[str]
    ids: torch.Tensor


def _token_adapter_ids(
    routing: list[str], n_tokens: int, device: torch.device
) -> TokenAdapterIds:
    """Expand fused routing (one adapter per batch row) to per-token adapter ids.

    :param routing: Adapter name per batch row.
    :param n_tokens: Leading dimension of the experts input.
    :param device: Device for the id tensor.
    :return: Adapter names and each token's index into them.
    """
    names = list(dict.fromkeys(routing))
    name_to_id = {name: index for index, name in enumerate(names)}
    factor, remainder = divmod(n_tokens, len(routing))
    if remainder:
        msg = (
            f"Fused adapter routing covers {len(routing)} rows but the "
            f"experts input's leading dimension is {n_tokens}."
        )
        raise ValueError(msg)
    ids = torch.tensor(
        [name_to_id[name] for name in routing for _ in range(factor)],
        device=device,
        dtype=torch.long,
    )
    return TokenAdapterIds(names, ids)


def _dispatch_adapter_ids(
    sorted_ids: torch.Tensor, state: TokenDispatchState
) -> torch.Tensor:
    """Send expert-sorted adapter ids with the tokens; local-expert-major result."""
    dispatched = sorted_ids.new_empty((sum(state.output_splits),))
    dist.all_to_all_single(
        dispatched,
        sorted_ids.contiguous(),
        state.output_splits,
        state.input_splits,
        group=state.ep_group,
    )
    return dispatched[state.permute_indices]


def _ep_param_grad_hook(param: nn.Parameter) -> Callable[[torch.Tensor], None]:
    """Accumulate a local shard grad onto an EP ``DTensor`` parameter.

    Local views of a ``DTensor`` cannot carry ``.grad`` back to it, so the
    detached leaf used for the local kernel copies its grad here as a
    same-placed ``DTensor``. Grad accumulation across micro-batches sums.
    """

    def hook(grad: torch.Tensor) -> None:
        assert isinstance(param, DTensor)
        local = param.to_local()
        if grad.stride() != local.stride():
            # Fused AdamW needs grad strides to match the param's. This hook
            # skips AccumulateGrad, which would otherwise copy into that layout.
            grad = torch.empty_like(local).copy_(grad)
        shard = DTensor.from_local(
            grad, param.device_mesh, param.placements, run_check=False
        )
        param.grad = shard if param.grad is None else param.grad + shard

    return hook


def _local_param_dict(
    module: nn.Module, gathered: bool = False
) -> dict[str, torch.Tensor]:
    """Plain-tensor views of a module's params for the EP-blind local kernel.

    Frozen ``DTensor`` params pass as ``to_local()`` views. When ``gathered``,
    ``reshard()`` frees their storage before backward reads it, so they are
    copied. Trainable ones become detached leaf copies; a hook copies each
    leaf grad back onto the sharded parameter (see
    :func:`_ep_param_grad_hook`). Dense params pass through untouched.

    :param module: Module whose parameters the local kernel reads.
    :param gathered: Whether the parameters are an FSDP-gathered block.
    :return: Parameter and buffer tensors keyed by name.
    """
    params: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if not isinstance(param, DTensor):
            params[name] = param
        elif param.requires_grad:
            leaf = param.to_local().detach().clone().requires_grad_(True)
            leaf.register_hook(_ep_param_grad_hook(param))
            params[name] = leaf
        elif gathered:
            params[name] = param.to_local().clone()
        else:
            params[name] = param.to_local()
    params.update(dict(module.named_buffers()))
    return params


def _call_with_local_params(
    module: nn.Module,
    original: Callable[..., torch.Tensor],
    params: dict[str, torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run ``original`` as ``module.forward`` on ``params`` from :func:`_local_param_dict`."""
    saved = module.forward
    # Module.__call__ does not inject self into ``forward``; keep the bound method.
    module.forward = original
    try:
        return functional_call(module, params, args, kwargs)
    finally:
        module.forward = saved


def scatter_scaled_expert_rows(
    combined: torch.Tensor,
    router_weights: torch.Tensor,
    token_idx: torch.Tensor,
    hidden_states: torch.Tensor,
    chunk_bytes: int,
) -> torch.Tensor:
    """Scale expert-sorted rows and scatter them to token order in fp32.

    :param chunk_bytes: Size of one fp32 chunk of scaled rows.
    """
    result = torch.zeros_like(hidden_states, dtype=torch.float32)
    _add_scaled_expert_rows(result, combined, router_weights, token_idx, chunk_bytes)
    return result.to(dtype=hidden_states.dtype)


def _add_scaled_expert_rows(
    result: torch.Tensor,
    combined: torch.Tensor,
    router_weights: torch.Tensor,
    token_idx: torch.Tensor,
    chunk_bytes: int,
) -> None:
    """Add router-scaled expert rows into the fp32 ``result`` at ``token_idx``."""
    router_weights = router_weights.to(dtype=torch.float32)
    # A full scaled copy of the combine buffer does not fit beside it.
    row_bytes = combined.shape[-1] * result.element_size()
    chunk_rows = max(1, chunk_bytes // row_bytes)
    # One split node concatenates the chunk grads in backward. Empty
    # ``combined`` still yields one piece, so its all-to-all runs in backward.
    pieces = zip(
        combined.split(chunk_rows),
        router_weights.split(chunk_rows),
        token_idx.split(chunk_rows),
        strict=True,
    )
    for rows, weights, index in pieces:
        # The fp32 weights promote the product to fp32.
        result.index_add_(0, index, rows * weights)


@contextmanager
def _on_comm_stream(comm: torch.cuda.Stream | None) -> Iterator[None]:
    """Run the body on ``comm`` after the current stream's queued work."""
    if comm is None:
        yield
        return
    comm.wait_stream(torch.cuda.current_stream(comm.device))
    with torch.cuda.stream(comm):
        yield


def _hand_over(
    comm: torch.cuda.Stream | None,
    sent: Sequence[torch.Tensor],
    received: Sequence[torch.Tensor],
) -> torch.cuda.Event | None:
    """Tie ``sent`` to ``comm`` and ``received`` to the current stream.

    :return: Event that marks ``comm``'s queued work, or ``None`` without a side stream.
    """
    if comm is None:
        return None
    for tensor in sent:
        tensor.record_stream(comm)
    current = torch.cuda.current_stream(comm.device)
    for tensor in received:
        tensor.record_stream(current)
    return comm.record_event()


def _wait_for(event: torch.cuda.Event | None) -> None:
    if event is not None:
        torch.cuda.current_stream().wait_event(event)


class DispatchedBlock(NamedTuple):
    """A token block sent to the ranks that own its experts."""

    token_idx: torch.Tensor
    state: TokenDispatchState
    rows: torch.Tensor
    adapter_ids: torch.Tensor | None
    arrived: torch.cuda.Event | None


class CombinedBlock(NamedTuple):
    """A token block's expert output, back on this rank in expert-sorted order."""

    rows: torch.Tensor
    token_idx: torch.Tensor
    order: torch.Tensor
    arrived: torch.cuda.Event | None


@dataclass(frozen=True)
class LocalExperts:
    """An EP-blind expert ``forward`` bound to this rank's local parameters."""

    module: nn.Module
    forward: Callable[..., torch.Tensor]
    params: dict[str, torch.Tensor]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class TokenExchange:
    """How routed tokens move between EP ranks in one forward."""

    ep_group: dist.ProcessGroup
    ep_degree: int
    token_blocks: int
    # Side stream for the token all-to-alls, or ``None`` to run them in order.
    comm: torch.cuda.Stream | None
    # Size of one fp32 chunk of router-scaled rows in the combine scatter.
    chunk_bytes: int


class RoutedCounts(NamedTuple):
    """One routed EP call's ``[blocks, experts]`` row counts, on device and host."""

    send: torch.Tensor
    received: torch.Tensor
    send_lists: list[list[int]]
    received_lists: list[list[int]]


class RoutedCountsScope:
    """Record each routed EP call's counts in a checkpointed forward, or replay them in its recompute.

    The recompute reruns the checkpointed block from its saved input, so it
    makes the same routed EP calls in the same order with the same routing.
    """

    def __init__(self, calls: list[RoutedCounts], replay: bool) -> None:
        self.calls = calls
        self.replay = replay
        self.position = 0
        self.tokens: list[Token[RoutedCountsScope | None]] = []

    def __enter__(self) -> Self:
        self.position = 0
        self.tokens.append(ROUTED_COUNTS_SCOPE.set(self))
        return self

    def __exit__(self, *_exc: object) -> None:
        ROUTED_COUNTS_SCOPE.reset(self.tokens.pop())


ROUTED_COUNTS_SCOPE: ContextVar[RoutedCountsScope | None] = ContextVar(
    "routed_counts_scope", default=None
)


def routed_counts_contexts() -> tuple[RoutedCountsScope, RoutedCountsScope]:
    """``checkpoint`` ``context_fn``: the recompute reuses the forward's routed EP counts.

    :return: Forward (record) and recompute (replay) contexts sharing one log.
    """
    calls: list[RoutedCounts] = []
    return RoutedCountsScope(calls, replay=False), RoutedCountsScope(calls, replay=True)


def _routed_counts(send: torch.Tensor, exchange: TokenExchange) -> RoutedCounts:
    """Swap ``send`` with the EP peers and copy both to host, or replay the forward's copy.

    :param send: ``[blocks, experts]`` rows this rank sends, from this call's routing.
    :param exchange: EP group and its size.
    :return: Device and host counts for the call.
    """
    scope = ROUTED_COUNTS_SCOPE.get()
    if scope is not None and scope.replay:
        recorded = scope.calls[scope.position]
        scope.position += 1
        # Checked on device, so the recompute issues no host sync.
        torch._assert_async(
            torch.eq(send, recorded.send).all(),
            "Recompute routed tokens differently from the checkpointed forward",
        )
        return recorded
    received = exchange_expert_counts(send, exchange.ep_group, exchange.ep_degree)
    send_lists, received_lists = torch.stack((send, received)).tolist()
    counts = RoutedCounts(send, received, send_lists, received_lists)
    if scope is not None:
        scope.calls.append(counts)
    return counts


def _base_experts(module: nn.Module) -> nn.Module:
    """The packed-experts module under a PEFT wrapper, or ``module`` itself."""
    get_base = getattr(module, "get_base_layer", None)
    return get_base() if callable(get_base) else module


def _routed_up_weight(module: nn.Module) -> torch.Tensor:
    experts = _base_experts(module)
    projections = routed_projection_names(experts)
    if projections is None:
        msg = "Routed experts module does not match a supported packed layout."
        raise RuntimeError(msg)
    return getattr(experts, projections[0])


def _sorted_weight(module: nn.Module) -> torch.Tensor:
    weight = getattr(_base_experts(module), "weight", None)
    if not isinstance(weight, torch.Tensor):
        msg = "Sorted experts module has no stacked weight."
        raise RuntimeError(msg)
    return weight


def _sort_token_blocks(
    top_k_index: torch.Tensor, n_tokens: int, token_blocks: int, num_experts: int
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Expert-sorted flat-row order and ``[blocks, experts]`` row counts of each token block."""
    top_k = top_k_index.shape[-1]
    flat_experts = top_k_index.reshape(-1)
    bounds = [n_tokens * block // token_blocks for block in range(token_blocks + 1)]
    orders: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for start, stop in pairwise(bounds):
        block_experts = flat_experts[start * top_k : stop * top_k]
        orders.append(torch.argsort(block_experts, stable=True) + start * top_k)
        counts.append(expert_row_counts(block_experts, num_experts))
    return orders, torch.stack(counts).to(torch.long)


def _dispatch_block(
    hidden_states: torch.Tensor,
    order: torch.Tensor,
    top_k: int,
    state: TokenDispatchState,
    adapter_ids: torch.Tensor | None,
    comm: torch.cuda.Stream | None,
) -> DispatchedBlock:
    """All-to-all one block's routed rows, and their adapter ids, to the expert owners."""
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    routed = hidden_states[token_idx]
    sent = [routed]
    local_ids = None
    with _on_comm_stream(comm):
        rows = all_to_all_single_autograd(
            routed, state.output_splits, state.input_splits, state.ep_group
        )
        if adapter_ids is not None:
            sorted_ids = adapter_ids[token_idx]
            sent += [sorted_ids, state.permute_indices]
            local_ids = _dispatch_adapter_ids(sorted_ids, state)
    arrived = [rows] if local_ids is None else [rows, local_ids]
    return DispatchedBlock(
        token_idx, state, rows, local_ids, _hand_over(comm, sent, arrived)
    )


@contextmanager
def _routing_override(module: nn.Module, routing: list[str] | None) -> Iterator[None]:
    """Route ``module``'s adapters by ``routing`` for the body; ``None`` keeps its routing."""
    if routing is None:
        yield
        return
    previous = ROUTING_STATE.get(module)
    ROUTING_STATE[module] = routing
    try:
        yield
    finally:
        if previous is None:
            ROUTING_STATE.pop(module, None)
        else:
            ROUTING_STATE[module] = previous


def _run_local_experts(
    experts: LocalExperts,
    state: TokenDispatchState,
    rows: torch.Tensor,
    local_ids: torch.Tensor | None,
    adapter_names: list[str],
    index_dtype: torch.dtype,
) -> torch.Tensor:
    """Run this rank's experts on a dispatched block, one expert per row in local-expert order."""
    local_rows = rows[state.permute_indices]
    n_rows = local_rows.shape[0]
    if n_rows == 0:
        # Peers with rows run the combine all-to-all in backward; tie
        # this rank's empty output to the same grad inputs so it joins.
        trainable = [p.sum() for p in experts.params.values() if p.requires_grad]
        return local_rows + 0 * sum(trainable, local_rows.sum())
    expert_index = torch.repeat_interleave(
        torch.arange(state.num_local_experts, device=rows.device, dtype=index_dtype),
        state.num_tokens_per_local_expert,
        output_size=n_rows,
    ).unsqueeze(-1)
    ones = torch.ones(n_rows, 1, dtype=rows.dtype, device=rows.device)
    # Mixed routing (e.g. PPO actor + critic rows) is per batch row; the
    # local wrapper needs routing aligned to its local rows.
    routing = (
        None if local_ids is None else [adapter_names[i] for i in local_ids.tolist()]
    )
    with _routing_override(experts.module, routing):
        return _call_with_local_params(
            experts.module,
            experts.forward,
            experts.params,
            local_rows,
            expert_index,
            ones,
            **experts.kwargs,
        )


def _scatter_block(
    result: torch.Tensor,
    combined: CombinedBlock,
    flat_weights: torch.Tensor,
    chunk_bytes: int,
) -> None:
    """Add a combined block's router-scaled rows into ``result`` once they arrive."""
    _wait_for(combined.arrived)
    router_weights = flat_weights[combined.order].unsqueeze(-1)
    _add_scaled_expert_rows(
        result, combined.rows, router_weights, combined.token_idx, chunk_bytes
    )


def _routed_ep_blocks(
    module: nn.Module,
    inner: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    adapter_ids: TokenAdapterIds | None,
    exchange: TokenExchange,
    expert_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Dispatch, run local experts, and combine ``hidden_states`` in token blocks.

    One all-to-all swaps every block's expert counts; a checkpoint recompute
    under :func:`routed_counts_contexts` reuses its forward's. Block ``b + 1``'s
    dispatch runs on ``exchange.comm`` while block ``b``'s experts run on the
    current stream, and block ``b - 1`` scatters while block ``b``'s combine is
    in flight. With ``exchange.comm=None`` every step runs in order.

    :param module: EP-sharded routed experts, or their outer LoRA wrapper.
    :param inner: The module's own forward, run on this rank's expert rows.
    :param hidden_states: ``[tokens, hidden]`` experts input.
    :param top_k_index: ``[tokens, top_k]`` global expert ids.
    :param top_k_weights: ``[tokens, top_k]`` router weights.
    :param adapter_ids: Adapter names and per-token ids under mixed fused routing.
    :param exchange: EP group, its size, token blocks per call, and side stream.
    :param expert_kwargs: Extra keyword arguments for ``inner``.
    :return: ``[tokens, hidden]`` routed expert output.
    """
    with _gathered_expert_block(module) as gathered:
        local_e = expert_local_tensor(_routed_up_weight(module)).shape[0]
        top_k = top_k_index.shape[-1]
        orders, send = _sort_token_blocks(
            top_k_index,
            hidden_states.shape[0],
            exchange.token_blocks,
            local_e * exchange.ep_degree,
        )
        counts = _routed_counts(send, exchange)
        experts = LocalExperts(
            module, inner, _local_param_dict(module, gathered), expert_kwargs
        )
        token_ids = None if adapter_ids is None else adapter_ids.ids
        names = [] if adapter_ids is None else adapter_ids.names
        flat_weights = top_k_weights.reshape(-1)
        result = torch.zeros_like(hidden_states, dtype=torch.float32)

        def dispatch(block: int) -> DispatchedBlock:
            state = dispatch_state(
                counts.received[block],
                counts.send_lists[block],
                counts.received_lists[block],
                exchange.ep_group,
                exchange.ep_degree,
                local_e,
            )
            return _dispatch_block(
                hidden_states, orders[block], top_k, state, token_ids, exchange.comm
            )

        pending = dispatch(0)
        previous: CombinedBlock | None = None
        for block in range(exchange.token_blocks):
            token_idx, state, rows, local_ids, arrived = pending
            if block + 1 < exchange.token_blocks:
                pending = dispatch(block + 1)
            _wait_for(arrived)
            expert_out = _run_local_experts(
                experts, state, rows, local_ids, names, top_k_index.dtype
            )
            del rows
            unpermuted = unpermute_from_local_expert_major(
                expert_out, state.permute_indices, sum(state.output_splits)
            )
            del expert_out
            with _on_comm_stream(exchange.comm):
                combined = all_to_all_single_autograd(
                    unpermuted,
                    state.input_splits,
                    state.output_splits,
                    exchange.ep_group,
                )
            returned = _hand_over(exchange.comm, [unpermuted], [combined])
            del unpermuted
            if previous is not None:
                _scatter_block(result, previous, flat_weights, exchange.chunk_bytes)
            previous = CombinedBlock(combined, token_idx, orders[block], returned)
        assert previous is not None
        _scatter_block(result, previous, flat_weights, exchange.chunk_bytes)
        return result.to(dtype=hidden_states.dtype)


def _comm_stream(
    streams: dict[torch.device, torch.cuda.Stream], hidden_states: torch.Tensor
) -> torch.cuda.Stream | None:
    """Side stream for the token all-to-alls on CUDA inputs, one per device."""
    if not hidden_states.is_cuda:
        return None
    device = hidden_states.device
    if device not in streams:
        streams[device] = torch.cuda.Stream(device)
    return streams[device]


def _route_replicated_span(
    run_blocks: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    adapter_ids: TokenAdapterIds | None,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Route this rank's :func:`replicated_row_span` of tokens and all-gather every span's output."""
    n_tokens = hidden_states.shape[0]
    start, stop = replicated_row_span(n_tokens, tp_group)
    if adapter_ids is not None:
        adapter_ids = TokenAdapterIds(adapter_ids.names, adapter_ids.ids[start:stop])
    part = run_blocks(
        SliceReplicatedRows.apply(hidden_states, tp_group),
        top_k_index[start:stop],
        SliceReplicatedRows.apply(top_k_weights, tp_group),
        adapter_ids,
    )
    return GatherReplicatedRows.apply(part, n_tokens, tp_group)


def _install_routed_ep_forward(
    module: nn.Module, token_blocks: int, tp_group: dist.ProcessGroup | None
) -> None:
    """Dispatch → local experts → combine, in ``token_blocks`` token blocks.

    :param module: Routed experts module, or its outer LoRA wrapper.
    :param token_blocks: Token blocks per call; see :func:`_routed_ep_blocks`.
    :param tp_group: Ranks holding the same tokens, or ``None``. Each routes
        only its :func:`replicated_row_span` and the outputs are all-gathered.
    """
    if getattr(module, "_agilerl_ep_forward", False):
        return
    inner = module.forward
    # Local rows reach the kernel one expert per row, in expert order.
    expert_kwargs: dict[str, Any] = (
        {"already_grouped": True}
        if isinstance(module, RoutedExpertsLoraWrapper)
        else {}
    )
    comm_streams: dict[torch.device, torch.cuda.Stream] = {}

    def ep_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        ep_degree = module_ep_degree(self)
        ep_group = getattr(self, "_ep_group", None)
        if (
            args
            or kwargs
            or hidden_states.dim() != 2
            or ep_degree <= 1
            or ep_group is None
        ):
            return inner(hidden_states, top_k_index, top_k_weights, *args, **kwargs)
        routing = mixed_routing(self)
        adapter_ids = (
            None
            if routing is None
            else _token_adapter_ids(
                routing, hidden_states.shape[0], hidden_states.device
            )
        )
        comm = _comm_stream(comm_streams, hidden_states) if token_blocks > 1 else None
        chunk_bytes = (
            self.chunk_bytes
            if isinstance(self, RoutedExpertsLoraWrapper)
            else ROUTED_EXPERT_CHUNK_BYTES
        )
        run_blocks = partial(
            _routed_ep_blocks,
            self,
            inner,
            exchange=TokenExchange(
                ep_group, ep_degree, token_blocks, comm, chunk_bytes
            ),
            expert_kwargs=expert_kwargs,
        )
        if tp_group is None:
            return run_blocks(hidden_states, top_k_index, top_k_weights, adapter_ids)
        return _route_replicated_span(
            run_blocks, hidden_states, top_k_index, top_k_weights, adapter_ids, tp_group
        )

    module.forward = MethodType(ep_forward, module)
    object.__setattr__(module, "_agilerl_ep_forward", True)


def _span_expert_counts(counts: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Per-expert row counts of the expert-sorted rows ``[start, stop)``."""
    ends = counts.cumsum(0)
    return ends.clamp(start, stop) - (ends - counts).clamp(start, stop)


def _install_sorted_ep_forward(
    module: nn.Module, tp_group: dist.ProcessGroup | None
) -> None:
    """Dispatch sorted rows → existing grouped kernel with local counts → combine.

    :param module: Sorted experts module, or its outer LoRA wrapper.
    :param tp_group: Ranks holding the same rows, or ``None``. Each dispatches
        only its :func:`replicated_row_span` and the outputs are all-gathered.
    """
    if getattr(module, "_agilerl_ep_forward", False):
        return
    inner = module.forward

    def dispatch_rows(
        self: nn.Module,
        inputs: torch.Tensor,
        counts: torch.Tensor,
        ep_group: dist.ProcessGroup,
        ep_degree: int,
    ) -> torch.Tensor:
        with _gathered_expert_block(self) as gathered:
            local_e = expert_local_tensor(_sorted_weight(self)).shape[0]
            local_rows, local_counts, state = token_dispatch(
                inputs,
                counts,
                ep_group=ep_group,
                ep_degree=ep_degree,
                num_local_experts=local_e,
            )
            params = _local_param_dict(self, gathered=gathered)
            return token_combine(
                _call_with_local_params(self, inner, params, local_rows, local_counts),
                state,
            )

    def ep_forward(
        self: nn.Module,
        inputs: torch.Tensor,
        expert_size: Sequence[int] | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        ep_degree = module_ep_degree(self)
        ep_group = getattr(self, "_ep_group", None)
        if ep_degree <= 1 or ep_group is None:
            return inner(inputs, expert_size, *args, **kwargs)
        if isinstance(expert_size, torch.Tensor):
            counts = expert_size.to(dtype=torch.long)
        else:
            counts = torch.as_tensor(list(expert_size), dtype=torch.long)
        if tp_group is None:
            return dispatch_rows(self, inputs, counts, ep_group, ep_degree)
        n_rows = inputs.shape[0]
        start, stop = replicated_row_span(n_rows, tp_group)
        part = dispatch_rows(
            self,
            SliceReplicatedRows.apply(inputs, tp_group),
            _span_expert_counts(counts, start, stop),
            ep_group,
            ep_degree,
        )
        return GatherReplicatedRows.apply(part, n_rows, tp_group)

    module.forward = MethodType(ep_forward, module)
    object.__setattr__(module, "_agilerl_ep_forward", True)


def install_ep_forward(
    module: nn.Module, token_blocks: int, tp_group: dist.ProcessGroup | None
) -> None:
    experts = _base_experts(module)
    if is_routed_experts_module(experts):
        _install_routed_ep_forward(module, token_blocks, tp_group)
    elif is_sorted_experts_module(experts):
        _install_sorted_ep_forward(module, tp_group)
