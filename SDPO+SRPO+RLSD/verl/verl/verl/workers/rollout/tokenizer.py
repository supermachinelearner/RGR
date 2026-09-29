
from abc import ABC, abstractmethod

import numpy as np
import torch

__all__ = ["HybridEngineBaseTokenizer"]


class HybridEngineBaseTokenizer(ABC):
    """the tokenizer property and function name should align with HF's to meet vllm requirement"""

    @property
    @abstractmethod
    def vocab_size(self):
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        pass

    @property
    @abstractmethod
    def all_special_ids(self) -> list[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        pass

    @property
    @abstractmethod
    def all_special_tokens(self) -> list[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        pass

    @abstractmethod
    def encode(self, text):
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers.

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers.
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:

        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
  
        pass

    @abstractmethod
    def get_added_vocab(self) -> dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        pass

    @property
    def is_fast(self):
        return False
