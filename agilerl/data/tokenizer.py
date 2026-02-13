from abc import ABC, abstractmethod
from typing import Any


class Tokenizer(ABC):
    def __init__(
        self,
        pad_token_id: int,
        eos_token_id: int,
        eoa_token_id: int,
        bos_token_id: int,
        boa_token_id: int,
        eod_token_id: int,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.eoa_token_id = eoa_token_id
        self.bos_token_id = bos_token_id
        self.boa_token_id = boa_token_id
        self.eod_token_id = eod_token_id

    @abstractmethod
    def encode(
        self,
        str_: str | list[str],
        **kwargs,
    ) -> tuple[list[int] | list[list[int]], list[int] | list[list[int]]]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int] | list[list[int]], **kwargs) -> str | list[str]:
        pass

    @abstractmethod
    def num_tokens(self) -> int:
        pass

    @abstractmethod
    def id_to_token(self, id_: int) -> str:
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def get_vocab(self) -> Any:
        pass
