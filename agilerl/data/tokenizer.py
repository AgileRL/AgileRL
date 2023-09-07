from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union


class Tokenizer(ABC):
    def __init__(
        self,
        pad_token_id,
        eos_token_id,
        eoa_token_id,
        bos_token_id,
        boa_token_id,
        eod_token_id,
    ):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.eoa_token_id = eoa_token_id
        self.bos_token_id = bos_token_id
        self.boa_token_id = boa_token_id
        self.eod_token_id = eod_token_id

    @abstractmethod
    def encode(
        self, str_: Union[str, List[str]], **kwargs
    ) -> Tuple[Union[List[int], List[List[int]]], Union[List[int], List[List[int]]]]:
        pass

    @abstractmethod
    def decode(
        self, tokens: Union[List[int], List[List[int]]], **kwargs
    ) -> Union[str, List[str]]:
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
