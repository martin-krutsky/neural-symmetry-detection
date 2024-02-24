import torch

from . import Embedder


class TorchEmbedder(Embedder):
    def __init__(self, precision):
        super().__init__(precision)

    @classmethod
    def max_exponent_wo_overflow(cls, data: torch.Tensor) -> int:
        """
        Calculate the maximum exponent of 10 with which data can be
        multiplied without overflow of any element in data.

        :param data: PyTorch tensor of data embeddings
        :return: maximum exponent of 10 that will not cause overflow
        """
        max_val = torch.max(data)
        max_exp = torch.floor(
            torch.log10(torch.iinfo(data.dtype).max / max_val))
        return int(max_exp)

    def round_and_cast(self, data: torch.Tensor) -> torch.Tensor:
        """
        Multiply data embeddings by an exponent of 10, round and cast it
        to an integer array.

        :param data: PyTorch tensor of data embeddings
        :return: PyTorch integer tensor of data embeddings
        """
        return torch.round(data * 10 ** self.precision).int()
