from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GenerationConfig
from trl.experimental.ppo.modeling_value_head import AutoModelForCausalLMWithValueHead

TINY_VOCAB_SIZE = 7
TINY_TARGET_TOKEN_ID = 6
# Keep this aligned with (or above) the PPO max_model_len you pass from benchmarking.
# A too-small n_positions triggers CUDA device-side asserts from embedding/index ops.
TINY_MAX_CONTEXT_LENGTH = 1024
TINY_MAX_OUTPUT_TOKENS = 64


class TinyDigitTokenizer:
    """Minimal tokenizer with 7-token vocabulary: digits '0'-'4' + PAD + EOS.

    PAD (5) and EOS (6) are separate from data tokens so digit embeddings
    aren't conflated with special-token semantics.
    """

    NUM_DIGITS = 5

    def __init__(self) -> None:
        self.vocab = {str(i): i for i in range(self.NUM_DIGITS)}
        self.inv_vocab = {i: str(i) for i in range(self.NUM_DIGITS)}
        self.pad_token_id = 5
        self.pad_token = "<PAD>"
        self.eos_token_id = TINY_TARGET_TOKEN_ID
        self.eos_token = "<EOS>"
        self.vocab_size = TINY_VOCAB_SIZE

    def apply_chat_template(
        self,
        conversation_template: list[dict[str, str]],
        tokenize: bool = False,
        continue_final_message: bool = True,
    ):
        del continue_final_message
        text = "".join(message["content"] for message in conversation_template)
        return self.encode(text) if tokenize else text

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        del args, kwargs
        encoded = [self.vocab[ch] for ch in text if ch in self.vocab]
        return encoded if encoded else [self.pad_token_id]

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        special = {self.pad_token_id, self.eos_token_id}
        decoded = []
        for idx in token_ids:
            idx = int(idx)
            if skip_special_tokens and idx in special:
                continue
            decoded.append(self.inv_vocab.get(idx, ""))
        return "".join(decoded)

    def batch_decode(
        self,
        token_batch: list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> list[str]:
        if isinstance(token_batch, torch.Tensor):
            token_batch = token_batch.tolist()
        return [
            self.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for token_ids in token_batch
        ]

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "pt",
        padding: bool | str = True,
        padding_side: str = "left",
        return_attention_mask: bool = True,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del args, kwargs
        if return_tensors != "pt":
            msg = "TinyDigitTokenizer only supports return_tensors='pt'."
            raise ValueError(msg)
        encoded = [self.encode(text) for text in texts]
        max_len = max(len(ids) for ids in encoded) if padding else None

        padded_ids = []
        masks = []
        for ids in encoded:
            if max_len is None:
                padded_ids.append(ids)
                masks.append([1] * len(ids))
                continue
            pad_len = max_len - len(ids)
            if padding_side == "left":
                padded = [self.pad_token_id] * pad_len + ids
                mask = [0] * pad_len + [1] * len(ids)
            else:
                padded = ids + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            padded_ids.append(padded)
            masks.append(mask)

        output = {"input_ids": torch.tensor(padded_ids, dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor(masks, dtype=torch.long)
        return output


def _make_tiny_config() -> GPT2Config:
    config = GPT2Config(
        vocab_size=TINY_VOCAB_SIZE,
        n_positions=TINY_MAX_CONTEXT_LENGTH,
        n_ctx=TINY_MAX_CONTEXT_LENGTH,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=TINY_TARGET_TOKEN_ID,
        pad_token_id=5,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    if hasattr(config, "summary_first_dropout"):
        config.summary_first_dropout = 0.0
    return config


def _make_tiny_generation_config() -> GenerationConfig:
    return GenerationConfig(
        do_sample=True,
        max_length=TINY_MAX_CONTEXT_LENGTH,
        max_new_tokens=TINY_MAX_OUTPUT_TOKENS,
        pad_token_id=5,
        eos_token_id=TINY_TARGET_TOKEN_ID,
    )


def build_tiny_actor_network(SEPARATE_CRITIC: bool = False) -> GPT2LMHeadModel:
    """Create a tiny causal LM (no value head) suitable for PPO actor debugging."""
    base_model = GPT2LMHeadModel(_make_tiny_config())
    generation_config = _make_tiny_generation_config()
    base_model.generation_config = generation_config
    base_model.name_or_path = "tiny-debug-transformer"
    if not SEPARATE_CRITIC:
        actor_network = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model, **{"summary_dropout_prob": 0.0}
        )
        actor_network.generation_config = generation_config
        return actor_network
    return base_model


def build_tiny_critic_network() -> AutoModelForCausalLMWithValueHead:
    """Create a tiny value-head causal LM suitable for PPO critic debugging."""
    base_model = GPT2LMHeadModel(_make_tiny_config())
    generation_config = _make_tiny_generation_config()
    base_model.generation_config = generation_config
    critic_network = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, **{"summary_dropout_prob": 0.0}
    )
    critic_network.generation_config = generation_config
    if hasattr(critic_network, "pretrained_model"):
        critic_network.pretrained_model.generation_config = generation_config
    critic_network.name_or_path = "tiny-debug-transformer"
    return critic_network
