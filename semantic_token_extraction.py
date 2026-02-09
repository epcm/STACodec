# Class for extracting semantic tokens from wav file
# Use WavLM Large/HuBERT Base + k-means model to extract semantic tokens

import os
import joblib
import torch
import torch.nn as nn
import numpy as np

from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT


# Default paths for kmeans models
KMEANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmeans_models")

# Supported SSL model configurations
SSL_CONFIGS = {
    "wavlm_large_l23": {
        "ssl_class": WavLM,
        "ssl_hub": "microsoft/wavlm-large",
        "layer": 23,
        "num_clusters": 1000,
        "kmeans_filename": "LibriSpeech_wavlm_k1000_L23.pt",
    },
    "hubert_base_l9": {
        "ssl_class": HuBERT,
        "ssl_hub": "facebook/hubert-base-ls960",
        "layer": 9,
        "num_clusters": 1024,
        "kmeans_filename": "LibriSpeech_hubert_k1024_L9.ckpt",
    },
}


class SemanticTokenExtractor(nn.Module):
    """Extract semantic tokens from audio using an SSL model + k-means quantization.

    Supports WavLM Large (layer 23, 1000 clusters) and HuBERT Base (layer 9, 1024 clusters).
    The SSL model produces continuous hidden-state features; a pre-trained k-means model
    maps each frame to the nearest cluster index, yielding a sequence of discrete tokens.

    Arguments
    ---------
    ssl_model_type : str
        Which SSL model to use. One of ``"wavlm_large_l23"`` or ``"hubert_base_l9"``.
    kmeans_dir : str, optional
        Directory containing the k-means checkpoint files.
        Defaults to ``<this_file>/kmeans_models``.
    cache_dir : str, optional
        Directory for caching downloaded SSL weights.
        Defaults to ``<this_file>/cache/<ssl_model_type>``.
    freeze_ssl : bool
        Whether to freeze the SSL model parameters (default ``True``).
    device : str
        Device for computation (default ``"cpu"``).

    Example
    -------
    >>> import torch
    >>> extractor = SemanticTokenExtractor("hubert_base_l9")
    >>> wav = torch.randn(2, 16000)          # batch of 2, 1 second at 16 kHz
    >>> tokens = extractor.extract(wav)       # [B, T]
    >>> tokens.shape
    torch.Size([2, 49])
    """

    def __init__(
        self,
        ssl_model_type,
        kmeans_dir=None,
        cache_dir=None,
        freeze_ssl=True,
        device="cpu",
    ):
        super().__init__()

        if ssl_model_type not in SSL_CONFIGS:
            raise ValueError(
                f"Unknown ssl_model_type '{ssl_model_type}'. "
                f"Supported: {list(SSL_CONFIGS.keys())}"
            )

        cfg = SSL_CONFIGS[ssl_model_type]
        self.layer = cfg["layer"]
        self.num_clusters = cfg["num_clusters"]
        self.device = device

        # --- Load SSL model via SpeechBrain ---
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "cache",
                ssl_model_type,
            )

        self.ssl_model = cfg["ssl_class"](
            source=cfg["ssl_hub"],
            output_norm=False,
            freeze=freeze_ssl,
            freeze_feature_extractor=freeze_ssl,
            output_all_hiddens=True,
            save_path=cache_dir,
        )

        # --- Load k-means model ---
        if kmeans_dir is None:
            kmeans_dir = KMEANS_DIR
        kmeans_path = os.path.join(kmeans_dir, cfg["kmeans_filename"])
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(
                f"K-means model not found at {kmeans_path}. "
                f"Please place the checkpoint in {kmeans_dir}."
            )
        self.kmeans_model = joblib.load(kmeans_path)
        # Ensure cluster centers are float32 so sklearn's Cython code
        # doesn't raise a dtype mismatch ("expected float, got double").
        self.kmeans_model.cluster_centers_ = self.kmeans_model.cluster_centers_.astype(np.float32)
        self.vocabulary = self.kmeans_model.cluster_centers_  # (num_clusters, feat_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(self, wav, wav_lens=None):
        """Extract semantic token ids from a batch of waveforms.

        Arguments
        ---------
        wav : torch.Tensor
            Raw waveform tensor of shape ``[B, T_samples]`` (16 kHz).
        wav_lens : torch.Tensor, optional
            Relative lengths in ``[0, 1]`` with shape ``[B]``.

        Returns
        -------
        tokens : torch.Tensor
            Integer token ids of shape ``[B, N]`` where ``N`` is the number
            of frames produced by the SSL model.
        """
        self.ssl_model.eval()
        feats = self.ssl_model.extract_features(wav, wav_lens)  # list/tensor indexed by layer
        layer_feat = feats[self.layer]  # (B, N, D)

        B, N, D = layer_feat.shape
        flat_feat = layer_feat.reshape(-1, D).cpu().to(torch.float32).numpy()
        flat_feat = flat_feat.astype("float32", copy=False) 
        token_ids = self.kmeans_model.predict(flat_feat)  # (B*N,)
        tokens = torch.tensor(
            token_ids.reshape(B, N), dtype=torch.long, device=wav.device,
        )
        return tokens

    @torch.no_grad()
    def extract_with_embeddings(self, wav, wav_lens=None):
        """Extract both token ids and their corresponding cluster-center embeddings.

        Arguments
        ---------
        wav : torch.Tensor
            Raw waveform tensor of shape ``[B, T_samples]`` (16 kHz).
        wav_lens : torch.Tensor, optional
            Relative lengths in ``[0, 1]`` with shape ``[B]``.

        Returns
        -------
        tokens : torch.Tensor
            Integer token ids of shape ``[B, N]``.
        embeddings : torch.Tensor
            Cluster-center embeddings of shape ``[B, N, D]``.
        """
        self.ssl_model.eval()
        feats = self.ssl_model.extract_features(wav, wav_lens)
        layer_feat = feats[self.layer]  # (B, N, D)

        B, N, D = layer_feat.shape
        flat_feat = layer_feat.reshape(-1, D).cpu().to(torch.float32).numpy()
        flat_feat = flat_feat.astype("float32", copy=False)
        token_ids = self.kmeans_model.predict(flat_feat)  # (B*N,)

        embs = self.vocabulary[token_ids]  # (B*N, D_emb)
        tokens = torch.tensor(
            token_ids.reshape(B, N), dtype=torch.long, device=wav.device,
        )
        embeddings = torch.tensor(
            embs.reshape(B, N, -1), dtype=torch.float32, device=wav.device,
        )
        return tokens, embeddings

    def forward(self, wav, wav_lens=None):
        """Alias for :meth:`extract`."""
        return self.extract(wav, wav_lens)


# ---------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing SemanticTokenExtractor ===\n")

    for model_type in ["hubert_base_l9", "wavlm_large_l23"]:
        print(f"--- {model_type} ---")
        extractor = SemanticTokenExtractor(model_type, device="cpu")
        wav = torch.randn(2, 16000)  # 2 utterances, 1 second each
        tokens = extractor.extract(wav)
        print(f"  tokens shape : {tokens.shape}")
        print(f"  tokens range : [{tokens.min().item()}, {tokens.max().item()}]")
        print(f"  num_clusters : {extractor.num_clusters}")

        tokens2, embs = extractor.extract_with_embeddings(wav)
        print(f"  embeddings   : {embs.shape}")
        assert torch.equal(tokens, tokens2)
        print()

    print("All tests passed.")