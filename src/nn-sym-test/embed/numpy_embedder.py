import numpy as np

from . import Embedder


class NumPyEmbedder(Embedder):
    def __init__(self, precision):
        super().__init__(precision)

    @classmethod
    def max_exponent_wo_overflow(cls, data: np.ndarray[np.floating]) -> int:
        """
        Calculate the maximum exponent of 10 with which data can be
        multiplied without overflow of any element in data.

        :param data: NumPy array of data embeddings
        :return: maximum exponent of 10 that will not cause overflow
        """
        max_val = np.max(data)
        max_exp = np.floor(np.log10(np.iinfo(data.dtype).max / max_val))
        return int(max_exp)

    def round_and_cast(self, data: np.ndarray[np.floating]
                       ) -> np.ndarray[np.integer]:
        """
        Multiply data embeddings by an exponent of 10, round and cast it
        to an integer array.

        :param data: NumPy ndarray of data embeddings
        :return: NumPy integer ndarray of data embeddings
        """
        return np.rint(data * 10 ** self.precision).astype(int)
