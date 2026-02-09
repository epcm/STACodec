import math
import typing as tp

import torch
from dataclasses import dataclass, field

from audiocraft.quantization.base import BaseQuantizer, QuantizedResult
from past.quantization.core_vq import ResidualVectorQuantization


@dataclass
class QuantizedResultCentroid(QuantizedResult):
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)
    rvq_centroid: tp.Optional[torch.Tensor] = None
    embedding: tp.Optional[torch.Tensor] = None


class ResidualVectorQuantizer(BaseQuantizer):
    """Residual Vector Quantizer.

    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        q_dropout (bool): Random quantizer drop out at train time.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider.
            for orthogonal regularization.
    """

    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        q_dropout: bool = False,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: tp.Optional[int] = None,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            orthogonal_reg_weight=self.orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=self.orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=self.orthogonal_reg_max_codes,
            channels_last=False,
        )

    def forward(self, x: torch.Tensor, frame_rate: int, ssl_labels: torch.Tensor = None):
        n_q = self.n_q
        if self.training and self.q_dropout:
            n_q = int(torch.randint(1, self.n_q + 1, (1,)).item())
        bw_per_q = math.log2(self.bins) * frame_rate / 1000
        quantized, codes, commit_loss, out_quantized = self.vq(x, n_q=n_q, ssl_labels=ssl_labels)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResultCentroid(quantized, codes, bw, penalty=torch.mean(commit_loss), rvq_centroid=out_quantized, embedding=x)

    def encode(self, x: torch.Tensor, ssl_labels: torch.Tensor = None) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.n_q
        quantized, codes, commit_loss, out_quantized = self.vq(x, n_q=n_q, ssl_labels=ssl_labels)
        codes = codes.transpose(0, 1)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert n > 0 and n <= self.max_n_q
        self.n_q = n
