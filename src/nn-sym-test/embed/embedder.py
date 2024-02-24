from abc import ABC, abstractmethod
from typing import Callable, Optional
from collections.abc import Collection

# SUPPORTED_BACKENDS = ['numpy', 'torch']


class Embedder(ABC):
    def __init__(self, precision):
        self.precision = precision
        # if backend not in SUPPORTED_BACKENDS:
        #     raise ValueError(f"Backend {backend} not supported, currently "
        #                      f"supporing {','.join(SUPPORTED_BACKENDS)}.")

    @classmethod
    def max_exponent_wo_overflow(cls, data: Collection) -> int:
        """
        Calculate the maximum exponent of 10 with which data can be
        multiplied without overflow of any element in data.

        :param data: array of data embeddings
        :return: maximum exponent of 10 that will not cause overflow
        """
        raise NotImplementedError

    @abstractmethod
    def round_and_cast(self, data: Collection) -> Collection:
        """
        Multiply data embeddings by an exponent of 10, round and cast it
        to an integer array.

        :param data: array of data embeddings
        :return: integer array of data embeddings
        """
        raise NotImplementedError

    def embed(self, data: Collection,
              embed_fun: Callable[[Collection], Collection],
              agg_fun: Optional[Callable[[Collection], Collection]] = None
              ) -> Collection:
        """
        Embed, round, and optionally aggregate the data.

        :param data:
        :param embed_fun:
        :param agg_fun:
        :return:
        """
        data_embedding = embed_fun(data)
        data_embedding = self.round_and_cast(data_embedding)
        if agg_fun is not None:
            data_embedding = agg_fun(data_embedding)
        return data_embedding
