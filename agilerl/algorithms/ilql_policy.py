# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from agilerl.data.language_environment import Language_Observation
from agilerl.data.rl_data import DataPoint
from agilerl.data.tokenizer import Tokenizer
from agilerl.utils.sampling_utils import (
    always_terminate,
    map_all_kvs,
    pad_sequence,
    process_logits,
    update_kvs,
)

if TYPE_CHECKING:
    from agilerl.algorithms.ilql import ILQL

def _decode_str(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    """Decode one flat id sequence to a string."""
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

class ILQL_Policy:
    def __init__(self, iql_model: ILQL, kind: str, **generation_kwargs: Any) -> None:
        super().__init__()
        self.iql_model = iql_model
        assert kind in {"beam", "sample"}
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.kls_all = []
        self.logprobs_all = []

    def sample_raw(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        state_idxs: torch.Tensor,
        action_idxs: torch.Tensor,
        termination_condition: Callable[[str], bool],
        num_generations: int = 1,
        max_generation_len: int | None = None,
        temp: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        exp_adv: bool = False,
        adv_weight: float = 0.0,
        adv_clip: float | None = None,
        include_logits: bool = True,
        include_adv: bool = True,
        rerank_log_prob_weight: float = 0.0,
        rerank_advantage_weight: float = 0.0,
        prefix_embs: torch.Tensor | None = None,
        prefix_attn_mask: torch.Tensor | None = None,
        remove_prefix_position_embs: bool = False,
    ) -> tuple[list[tuple[str, list[str]]], torch.Tensor, torch.Tensor]:
        assert include_logits or include_adv

        tokenizer = self.iql_model.dataset.tokenizer
        max_length = self.iql_model.dataset.max_len
        if max_length is None:
            max_length = self.iql_model.model.block_size
        max_length = min(max_length, self.iql_model.model.block_size)
        device = self.iql_model.device
        bsize = tokens.shape[0]
        n = bsize * num_generations
        if max_generation_len is None:
            max_generation_len = max_length + 1
        input_strs = [
            _decode_str(
                tokenizer,
                tokens[i, :][: attn_mask[i, :].sum().long()].tolist(),
            )
            for i in range(len(tokens))
        ]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        model_outputs = self.iql_model(
            tokens,
            state_idxs,
            action_idxs,
            attn_mask,
            prefix_embs=prefix_embs,
            prefix_attn_mask=prefix_attn_mask,
            remove_prefix_position_embs=remove_prefix_position_embs,
            qv_kwargs={"is_causal": False},
            target_kwargs={"is_causal": False},
            policy_kwargs={"is_causal": False},
        )["model_outputs"]
        kvs = {"qv": model_outputs["qv_model_outputs"]["past_key_values"]}
        if self.iql_model.actor_target is not None:
            kvs["target"] = model_outputs["target_model_outputs"]["past_key_values"]
        if self.iql_model.actor is not None:
            kvs["policy"] = model_outputs["policy_model_outputs"]["past_key_values"]
        dialogue_lens = attn_mask.sum(dim=1)
        tokens = pad_sequence(
            torch.repeat_interleave(tokens, num_generations, dim=0),
            max_length,
            tokenizer.pad_token_id,
            device,
            1,
        )
        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)
        kvs["qv"] = map_all_kvs(
            lambda x: pad_sequence(
                torch.repeat_interleave(x, num_generations, dim=0),
                max_length,
                0.0,
                device,
                2,
            ),
            kvs["qv"],
        )
        if "target" in kvs:
            kvs["target"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, num_generations, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["target"],
            )
        if "policy" in kvs:
            kvs["policy"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, num_generations, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["policy"],
            )
        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        kls = torch.full(
            (dialogue_lens.shape[0],),
            math.log(num_generations) - ((num_generations - 1) / num_generations),
        ).to(device)
        advantages = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(device)
        state_idxs_temp, action_idxs_temp = (
            torch.zeros(
                (
                    dialogue_lens.shape[0],
                    1,
                ),
            )
            .long()
            .to(device),
            torch.zeros(
                (
                    dialogue_lens.shape[0],
                    1,
                ),
            )
            .long()
            .to(
                device,
            ),
        )
        t = int(dialogue_lens.min().item())
        base_logits = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        while termination_mask.sum() > 0 and (t + prefix_t) < max_length:
            curr_token = tokens[:, t - 1].unsqueeze(1)
            _t = t
            curr_kvs = map_all_kvs(
                lambda x, _t=_t: x[:, :, : (_t + prefix_t) - 1, :],
                kvs["qv"],
            )
            curr_target_kvs, curr_policy_kvs = curr_kvs, curr_kvs
            if "target" in kvs:
                map_all_kvs(
                    lambda x, _t=_t: x[:, :, : (_t + prefix_t) - 1, :], kvs["target"]
                )
            if "policy" in kvs:
                map_all_kvs(
                    lambda x, _t=_t: x[:, :, : (_t + prefix_t) - 1, :], kvs["policy"]
                )
            iql_outputs = self.iql_model(
                curr_token,
                state_idxs_temp,
                action_idxs_temp,
                None,
                qv_kwargs={"past_key_values": curr_kvs},
                policy_kwargs={"past_key_values": curr_policy_kvs},
                target_kwargs={"past_key_values": curr_target_kvs},
            )
            model_outputs, logits = iql_outputs["model_outputs"], iql_outputs["logits"]

            logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1,
                float("-inf"),
                1e7,
            )
            logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens,
                1e7,
            )
            edited_logits = process_logits(
                logits.clone(),
                temp=temp,
                top_k=top_k,
                top_p=top_p,
            )

            vs, qs = iql_outputs["target_vs"], iql_outputs["target_qs"]
            if exp_adv:
                adv_logits = adv_weight * (qs - vs.unsqueeze(2))
            else:
                adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
                adv_logits = adv_weight * adv_sign + (1 - adv_weight) * (1 - adv_sign)
                adv_logits = torch.log(adv_logits)
            if adv_clip is not None:
                adv_logits = torch.clip(adv_logits, max=adv_clip)
            adv_logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1,
                float("-inf"),
                1e7,
            )
            adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens,
                1e7,
            )

            full_logits = (
                (edited_logits if include_logits else 0.0)
                + (adv_logits if include_adv else 0.0)
                + base_logits.unsqueeze(1).unsqueeze(2)
            )

            cat_dist = torch.distributions.categorical.Categorical(
                logits=full_logits[:, 0],
            )
            original_cat_dist = torch.distributions.categorical.Categorical(
                logits=logits[:, 0],
            )

            new_tokens = cat_dist.sample()
            log_probs += cat_dist.log_prob(new_tokens)
            kls += cat_dist.log_prob(new_tokens) - original_cat_dist.log_prob(
                new_tokens,
            )
            qs_chosen = torch.gather(
                qs.squeeze(1),
                dim=1,
                index=new_tokens.unsqueeze(1),
            ).squeeze(1)
            advantages += qs_chosen - vs.squeeze(1)
            tokens[:, t] = new_tokens
            kvs["qv"] = update_kvs(
                kvs["qv"],
                model_outputs["qv_model_outputs"]["past_key_values"],
                torch.arange(0, n).to(device),
                (t + prefix_t) - 1,
            )
            if "target" in kvs:
                kvs["target"] = update_kvs(
                    kvs["target"],
                    model_outputs["target_model_outputs"]["past_key_values"],
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            if "policy" in kvs:
                kvs["policy"] = update_kvs(
                    kvs["policy"],
                    model_outputs["policy_model_outputs"]["past_key_values"],
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= 1 - int(
                        termination_condition(
                            _decode_str(tokenizer, tokens[idx, :].tolist()),
                        ),
                    )
            t += 1
            termination_mask *= ((t - dialogue_lens) < max_generation_len).int()

        scores = (
            (advantages * rerank_advantage_weight)
            + (log_probs * rerank_log_prob_weight)
        ).reshape(-1, num_generations)
        order = torch.argsort(-scores, dim=1)
        output_strs = [
            _decode_str(tokenizer, tokens[i, :].tolist()) for i in range(len(tokens))
        ]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(num_generations):
                processed_str = output_strs[i * num_generations + order[i, x]][
                    len(input_strs[i]) :
                ].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.pad_token_id),
                        )
                    ].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.eoa_token_id),
                        )
                    ].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        log_probs = torch.gather(
            log_probs.reshape(-1, num_generations),
            dim=1,
            index=order,
        )
        kls = torch.gather(kls.reshape(-1, num_generations), dim=1, index=order)
        return (
            list(zip(input_strs, processed_outputs, strict=False)),
            log_probs.reshape(-1, num_generations),
            kls,
        )

    def beam_raw(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        state_idxs: torch.Tensor,
        action_idxs: torch.Tensor,
        termination_condition: Callable[[str], bool],
        max_generation_len: int | None = None,
        beam_width: int = 1,
        temp: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        exp_adv: bool = False,
        adv_weight: float = 0.0,
        adv_clip: float | None = None,
        include_logits: bool = True,
        include_adv: bool = True,
        prefix_embs: torch.Tensor | None = None,
        prefix_attn_mask: torch.Tensor | None = None,
        remove_prefix_position_embs: bool = False,
    ) -> tuple[list[tuple[str, list[str]]], torch.Tensor, torch.Tensor]:
        tokenizer = self.iql_model.dataset.tokenizer
        max_length = self.iql_model.dataset.max_len
        if max_length is None:
            max_length = self.iql_model.model.block_size
        max_length = min(max_length, self.iql_model.model.block_size)
        device = self.iql_model.device
        bsize, vocab_size = tokens.shape[0], tokenizer.num_tokens()
        n = bsize * beam_width
        if max_generation_len is None:
            max_generation_len = max_length + 1
        input_strs = [
            _decode_str(
                tokenizer,
                tokens[i, :][: attn_mask[i, :].sum().long()].tolist(),
            )
            for i in range(len(tokens))
        ]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        model_outputs = self.iql_model(
            tokens,
            state_idxs,
            action_idxs,
            attn_mask,
            prefix_embs=prefix_embs,
            prefix_attn_mask=prefix_attn_mask,
            remove_prefix_position_embs=remove_prefix_position_embs,
            qv_kwargs={"is_causal": False},
            target_kwargs={"is_causal": False},
            policy_kwargs={"is_causal": False},
        )["model_outputs"]
        kvs = {"qv": model_outputs["qv_model_outputs"]["past_key_values"]}
        if self.iql_model.actor_target is not None:
            kvs["target"] = model_outputs["target_model_outputs"]["past_key_values"]
        if self.iql_model.actor is not None:
            kvs["policy"] = model_outputs["policy_model_outputs"]["past_key_values"]
        original_dialogue_lens = attn_mask.sum(dim=1)
        batch_indicator = torch.stack(
            beam_width * [torch.arange(0, bsize).to(device)],
            dim=1,
        )

        tokens = pad_sequence(
            torch.repeat_interleave(tokens, beam_width, dim=0),
            max_length,
            tokenizer.pad_token_id,
            device,
            1,
        )
        dialogue_lens = torch.repeat_interleave(
            original_dialogue_lens,
            beam_width,
            dim=0,
        )
        kvs["qv"] = map_all_kvs(
            lambda x: pad_sequence(
                torch.repeat_interleave(x, beam_width, dim=0),
                max_length,
                0.0,
                device,
                2,
            ),
            kvs["qv"],
        )
        if "target" in kvs:
            kvs["target"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, beam_width, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["target"],
            )
        if "policy" in kvs:
            kvs["policy"] = map_all_kvs(
                lambda x: pad_sequence(
                    torch.repeat_interleave(x, beam_width, dim=0),
                    max_length,
                    0.0,
                    device,
                    2,
                ),
                kvs["policy"],
            )
        curr_scores = torch.zeros(bsize, beam_width).to(device)  # (batch, k)
        logit_scores = torch.zeros(bsize, beam_width).to(device)  # (batch, k)
        termination_mask = torch.full((n,), 1).to(device)
        state_idxs_temp, action_idxs_temp = (
            torch.zeros(
                (
                    dialogue_lens.shape[0],
                    1,
                ),
            )
            .long()
            .to(device),
            torch.zeros(
                (
                    dialogue_lens.shape[0],
                    1,
                ),
            )
            .long()
            .to(
                device,
            ),
        )
        t = int(dialogue_lens.min().item())
        base_logits = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        while termination_mask.sum() > 0 and (t + prefix_t) < max_length:
            curr_token = tokens[:, t - 1].unsqueeze(1)
            _t = t
            curr_kvs = map_all_kvs(
                lambda x, _t=_t: x[:, :, : (_t + prefix_t) - 1, :],
                kvs["qv"],
            )
            curr_target_kvs, curr_policy_kvs = curr_kvs, curr_kvs
            if "target" in kvs:
                map_all_kvs(
                    lambda x, _t=_t: x[:, :, : (_t + prefix_t) - 1, :], kvs["target"]
                )
            if "policy" in kvs:
                map_all_kvs(
                    lambda x, _t=_t: x[:, :, : (_t + prefix_t) - 1, :], kvs["policy"]
                )
            iql_outputs = self.iql_model(
                curr_token,
                state_idxs_temp,
                action_idxs_temp,
                None,
                qv_kwargs={"past_key_values": curr_kvs},
                policy_kwargs={"past_key_values": curr_policy_kvs},
                target_kwargs={"past_key_values": curr_target_kvs},
                is_causal=False,
            )
            model_outputs, logits = iql_outputs["model_outputs"], iql_outputs["logits"]

            logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1,
                float("-inf"),
                1e7,
            )
            logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens,
                1e7,
            )
            edited_logits = process_logits(
                logits.clone(),
                temp=temp,
                top_k=top_k,
                top_p=top_p,
            )

            vs, qs = iql_outputs["target_vs"], iql_outputs["target_qs"]
            if exp_adv:
                adv_logits = adv_weight * (qs - vs.unsqueeze(2))
            else:
                adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
                adv_logits = adv_weight * adv_sign + (1 - adv_weight) * (1 - adv_sign)
                adv_logits = torch.log(adv_logits)
            if adv_clip is not None:
                adv_logits = torch.clip(adv_logits, max=adv_clip)
            adv_logits[:, 0, tokenizer.pad_token_id] = torch.where(
                termination_mask == 1,
                float("-inf"),
                1e7,
            )
            adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ] = adv_logits[
                torch.arange(0, n).to(device),
                torch.full((n,), 0).to(device),
                tokens[:, t],
            ].masked_fill_(
                t < dialogue_lens,
                1e7,
            )

            full_logits = (
                (edited_logits if include_logits else 0.0)
                + (adv_logits if include_adv else 0.0)
                + base_logits.unsqueeze(1).unsqueeze(2)
            )

            scores = (
                (
                    torch.log(F.softmax(full_logits, dim=-1))
                    .reshape(1, bsize, beam_width, -1)
                    .permute(3, 0, 1, 2)
                    + curr_scores
                )
                .permute(1, 2, 3, 0)
                .reshape(1, bsize, -1)
            )  # (time, batch, k*vocab)
            scores[0, :, vocab_size:] = scores[0, :, vocab_size:].masked_fill_(
                (original_dialogue_lens == t)
                .unsqueeze(1)
                .repeat(1, scores.shape[2] - vocab_size),
                float("-inf"),
            )
            curr_scores, top_k_ = torch.topk(
                scores[0, :, :],
                k=beam_width,
                dim=1,
            )  # (batch, k), (batch, k)
            tokens = tokens[
                (batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1),
                :,
            ]
            logits = logits[
                (batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1),
                :,
                :,
            ]
            logit_scores += (
                torch.gather(
                    torch.log(F.softmax(logits, dim=-1)).squeeze(1),
                    dim=1,
                    index=(top_k_.reshape(-1) % vocab_size).unsqueeze(1),
                )
                .squeeze(1)
                .reshape(-1, beam_width)
            )
            tokens[:, t] = top_k_.reshape(-1) % vocab_size  # (batch*k,)
            _top_k = top_k_
            fixed_kvs = map_all_kvs(
                lambda x, _top_k=_top_k: x[
                    (
                        batch_indicator * beam_width
                        + torch.div(_top_k, vocab_size, rounding_mode="trunc")
                    ).reshape(-1),
                    :,
                    :,
                    :,
                ],
                model_outputs["qv_model_outputs"]["past_key_values"],
            )
            kvs["qv"] = map_all_kvs(
                lambda x, _top_k=_top_k: x[
                    (
                        batch_indicator * beam_width
                        + torch.div(_top_k, vocab_size, rounding_mode="trunc")
                    ).reshape(-1),
                    :,
                    :,
                    :,
                ],
                kvs["qv"],
            )
            kvs["qv"] = update_kvs(
                kvs["qv"],
                fixed_kvs,
                torch.arange(0, n).to(device),
                (t + prefix_t) - 1,
            )
            if "target" in kvs:
                fixed_target_kvs = map_all_kvs(
                    lambda x, _top_k=_top_k: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(_top_k, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    model_outputs["target_model_outputs"]["past_key_values"],
                )
                kvs["target"] = map_all_kvs(
                    lambda x, _top_k=_top_k: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(_top_k, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    kvs["target"],
                )
                kvs["target"] = update_kvs(
                    kvs["target"],
                    fixed_target_kvs,
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            if "policy" in kvs:
                fixed_policy_kvs = map_all_kvs(
                    lambda x, _top_k=_top_k: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(_top_k, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    model_outputs["policy_model_outputs"]["past_key_values"],
                )
                kvs["policy"] = map_all_kvs(
                    lambda x, _top_k=_top_k: x[
                        (
                            batch_indicator * beam_width
                            + torch.div(_top_k, vocab_size, rounding_mode="trunc")
                        ).reshape(-1),
                        :,
                        :,
                        :,
                    ],
                    kvs["policy"],
                )
                kvs["policy"] = update_kvs(
                    kvs["policy"],
                    fixed_policy_kvs,
                    torch.arange(0, n).to(device),
                    (t + prefix_t) - 1,
                )
            termination_mask = termination_mask[
                (batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1)
            ]
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= 1 - int(
                        termination_condition(
                            _decode_str(tokenizer, tokens[idx, :].tolist()),
                        ),
                    )
            t += 1
            termination_mask *= ((t - dialogue_lens) < max_generation_len).int()

        output_strs = [_decode_str(tokenizer, tokens[i, :].tolist()) for i in range(n)]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(beam_width):
                processed_str = output_strs[i * beam_width + x][
                    len(input_strs[i]) :
                ].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.pad_token_id),
                        )
                    ].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[
                        : processed_str.find(
                            tokenizer.id_to_token(tokenizer.eoa_token_id),
                        )
                    ].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return (
            list(zip(input_strs, processed_outputs, strict=False)),
            curr_scores,
            -logit_scores,
        )

    def generate(
        self,
        items: list[DataPoint] | dict[str, torch.Tensor],
        termination_condition: Callable[[str], bool],
        **kwargs: Any,
    ) -> tuple[list[tuple[str, list[str]]], Any, torch.Tensor]:
        prepared_inputs = self.iql_model.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs["tokens"], prepared_inputs["attn_mask"]
        state_idxs, action_idxs = (
            prepared_inputs["state_idxs"],
            prepared_inputs["action_idxs"],
        )
        if self.kind == "beam":
            method = self.beam_raw
        elif self.kind == "sample":
            method = self.sample_raw
        else:
            raise NotImplementedError
        generations, info, kls = method(
            tokens,
            attn_mask,
            state_idxs,
            action_idxs,
            termination_condition,
            **kwargs,
        )
        return generations, info, kls

    def act(self, obs: Language_Observation) -> str:
        item = DataPoint.from_obs(
            obs,
            self.iql_model.dataset.tokenizer,
            self.iql_model.dataset.token_reward,
        )
        generations, logprobs, kls = self.generate(
            [item],
            always_terminate,
            **self.generation_kwargs,
        )
        self.kls_all.append(kls[0, 0].item())
        self.logprobs_all.append(logprobs[0, 0].item())
        return generations[0][1][0]

    def train(self) -> None:
        self.iql_model.train()

    def eval(self) -> None:
        self.iql_model.eval()
