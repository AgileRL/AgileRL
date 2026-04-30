from __future__ import annotations

import torch
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel

from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

TINY_VOCAB_SIZE = 7
TINY_TARGET_TOKEN_ID = 6
TINY_MAX_CONTEXT_LENGTH = 1024
TINY_MAX_OUTPUT_TOKENS = 64


class TinyDigitTokenizer:
    """Digits 0–4 plus PAD (5) and EOS (6)."""  # noqa: RUF002

    NUM_DIGITS = 5

    def __init__(self) -> None:
        self.vocab = {str(i): i for i in range(self.NUM_DIGITS)}
        self.inv_vocab = {i: str(i) for i in range(self.NUM_DIGITS)}
        self.pad_token_id = 5
        self.pad_token = "<PAD>"  # noqa: S105
        self.eos_token_id = TINY_TARGET_TOKEN_ID
        self.eos_token = "<EOS>"  # noqa: S105
        self.vocab_size = TINY_VOCAB_SIZE

    def apply_chat_template(
        self,
        conversation_template: list[dict[str, str]],
        tokenize: bool = False,
        continue_final_message: bool = True,
    ):
        """Apply a chat template to a conversation.

        :param conversation_template: The conversation template.
        :type conversation_template: list[dict[str, str]]
        :param tokenize: Whether to tokenize the conversation.
        :type tokenize: bool
        :param continue_final_message: Whether to continue the final message.
        :type continue_final_message: bool
        :return: The encoded conversation.
        :rtype: list[int]
        """
        del continue_final_message
        text = "".join(message["content"] for message in conversation_template)
        return self.encode(text) if tokenize else text

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        """Encode a text into a list of token IDs.

        :param text: The text to encode.
        :type text: str
        :param args: Additional arguments.
        :type args: Any
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        :return: The encoded token IDs.
        :rtype: list[int]
        """
        del args, kwargs
        encoded = [self.vocab[ch] for ch in text if ch in self.vocab]
        return encoded or [self.pad_token_id]

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """Decode a list of token IDs.

        :param token_ids: The list of token IDs to decode.
        :type token_ids: list[int] | torch.Tensor
        :param skip_special_tokens: Whether to skip special tokens.
        :type skip_special_tokens: bool
        :param clean_up_tokenization_spaces: Whether to clean up tokenization spaces.
        :type clean_up_tokenization_spaces: bool
        :return: The decoded text.
        :rtype: str
        """
        del clean_up_tokenization_spaces
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        special = {self.pad_token_id, self.eos_token_id}
        decoded = []
        for idx in token_ids:
            int_idx = int(idx)
            if skip_special_tokens and int_idx in special:
                continue
            decoded.append(self.inv_vocab.get(int_idx, ""))
        return "".join(decoded)

    def batch_decode(
        self,
        token_batch: list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> list[str]:
        """Decode a batch of token IDs.

        :param token_batch: The batch of token IDs to decode.
        :type token_batch: list[list[int]] | torch.Tensor
        :param skip_special_tokens: Whether to skip special tokens.
        :type skip_special_tokens: bool
        :param clean_up_tokenization_spaces: Whether to clean up tokenization spaces.
        :type clean_up_tokenization_spaces: bool
        :return: The decoded texts.
        :rtype: list[str]
        """
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
        """Tokenize a list of texts.

        :param texts: The texts to tokenize.
        :type texts: list[str]
        :param return_tensors: The type of tensors to return.
        :type return_tensors: str
        :param padding: Whether to pad the texts.
        :type padding: bool | str
        :param padding_side: The side to pad the texts.
        :type padding_side: str
        :param return_attention_mask: Whether to return the attention mask.
        :type return_attention_mask: bool
        :param args: Additional arguments.
        :type args: Any
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        :return: The tokenized texts.
        :rtype: dict[str, torch.Tensor]
        """
        del args, kwargs
        if return_tensors != "pt":
            error_msg = "TinyDigitTokenizer only supports return_tensors='pt'."
            raise ValueError(error_msg)
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


def build_tiny_actor_network(use_value_head: bool = False) -> GPT2LMHeadModel:
    """Build a tiny actor network for debugging.

    :param use_value_head: Whether to use a value head.
    :type use_value_head: bool
    :return: The actor network.
    :rtype: GPT2LMHeadModel
    """
    base_model = GPT2LMHeadModel(_make_tiny_config())
    generation_config = _make_tiny_generation_config()
    base_model.generation_config = generation_config
    base_model.name_or_path = "tiny-debug-transformer"
    if use_value_head:
        actor_network = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model, summary_dropout_prob=0.0
        )
        actor_network.generation_config = generation_config
        return actor_network
    return base_model
