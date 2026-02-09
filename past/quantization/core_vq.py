import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from audiocraft.quantization.core_vq import ResidualVectorQuantization as ResidualVectorQuantizationAudiocraft
from audiocraft.quantization.core_vq import VectorQuantization as VectorQuantizationAudiocraft
from audiocraft.quantization.core_vq import EuclideanCodebook as EuclideanCodebookAudiocraft
from audiocraft.quantization.core_vq import orthogonal_loss_fn, exists, ema_inplace, laplace_smoothing

class EuclideanCodebook(EuclideanCodebookAudiocraft):
    def forward(self, x, ssl_labels: tp.Optional[torch.Tensor] = None):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        self.init_embed_(x)

        embed_ind = self.quantize(x)
        # print("ssl_labels shape:", ssl_labels.shape)
        # raise NotImplementedError("The quantizer should be implemented in the subclass.")
        if ssl_labels is not None:
            embed_ind = ssl_labels.view(-1)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind

    
class VectorQuantization(VectorQuantizationAudiocraft):
    """Vector quantization layer.

    Args:
        num_quantizers (int): Number of quantizers to use.
        **kwargs: Additional arguments for the base class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # print(kwargs)
        # def default(val: tp.Any, d: tp.Any) -> tp.Any:
        #     return val if exists(val) else d
        _codebook_dim = kwargs['dim']
        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=kwargs['codebook_size'],
                                           kmeans_init=kwargs['kmeans_init'], kmeans_iters=kwargs['kmeans_iters'],
                                           decay=kwargs['decay'], epsilon=self.epsilon,
                                           threshold_ema_dead_code=kwargs['threshold_ema_dead_code'])

    def forward(self, x, ssl_labels: tp.Optional[torch.Tensor] = None):
        device = x.device
        x = self._preprocess(x)

        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x, ssl_labels=ssl_labels)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)

        return quantize, embed_ind, loss


class ResidualVectorQuantization(ResidualVectorQuantizationAudiocraft):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__(num_quantizers=num_quantizers, **kwargs)
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None, ssl_labels: torch.Tensor = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []
        all_centroids = []

        n_q = n_q or len(self.layers)

        layer = self.layers[0]
        quantized, indices, loss = layer(residual, ssl_labels=ssl_labels)
        all_centroids.append(residual + (quantized - residual).detach())
        quantized = quantized.detach()
        residual = residual - quantized
        quantized_out = quantized_out + quantized
        all_indices.append(indices)
        all_losses.append(loss)

        for layer in self.layers[1:n_q]:
            quantized, indices, loss = layer(residual)
            all_centroids.append(residual + (quantized - residual).detach())
            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        if self.training:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            quantized_out = x + (quantized_out - x).detach()

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        out_centroids = torch.stack(all_centroids, dim=1)
        # see if the embedding is fixed
        # print(self.layers[0]._codebook.embed.data[0][:10])
        # import ipdb; ipdb.set_trace()
        return quantized_out, out_indices, out_losses, out_centroids

