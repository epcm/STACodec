"""
STACodec Inference Class

A clean class wrapper for STACodec audio codec inference, supporting
encoding audio to discrete tokens and decoding tokens back to audio.
"""

import torch
import torchaudio
import typing as tp
from pathlib import Path

from past.models.past_model import PastModel
from semantic_token_extraction import SemanticTokenExtractor


class STACodecInference:
    """
    A class for STACodec audio codec inference.

    Supports encoding audio to discrete tokens and decoding tokens back to audio.
    Optionally uses semantic token extraction via SSL models (HuBERT/WavLM).

    Arguments
    ---------
    model_path : str
        Path to the model checkpoint or HuggingFace repo ID.
    ssl_model_type : str
        SSL model type for semantic token extraction. One of "hubert_base_l9"
        or "wavlm_large_l23". Set to None to disable semantic extraction.
    spd : bool
        Whether to use the model's semantic prediction for SSL tokens instead
        of extracting them externally.
    device : str, optional
        Device for computation (default: "cuda" if available, else "cpu").

    Example
    -------
    >>> # From local checkpoint
    >>> codec = STACodecInference("path/to/model.th", ssl_model_type="hubert_base_l9", spd=False)
    >>> # From HuggingFace (auto download)
    >>> codec = STACodecInference.from_pretrained("username/stacodec")
    >>> # Encode audio file to tokens
    >>> codes, scale = codec.encode_file("audio.wav")
    >>> # Decode tokens back to audio
    >>> reconstructed = codec.decode(codes, scale)
    >>> codec.save_audio(reconstructed, "output.wav")
    """

    def __init__(
        self,
        model_path: str,
        ssl_model_type: str,
        spd: bool,
        device: str = None,
        kmeans_dir: str = None,
    ):
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.spd = spd

        # Load model
        self.model = PastModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.sample_rate = self.model.sample_rate

        # Initialize semantic token extractor if needed
        self.semantic_extractor = None
        if ssl_model_type and not spd:
            self.semantic_extractor = SemanticTokenExtractor(
                ssl_model_type, kmeans_dir=kmeans_dir, device=str(self.device)
            )
            self.semantic_extractor.ssl_model = self.semantic_extractor.ssl_model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        model_id: str = None,
        device: str = None,
    ):
        """Instantiate STACodecInference from a local directory or HuggingFace repo.

        The directory / repo should be organized as::

            repo_id/
            ├── <model_id>/
            │   ├── model.th
            │   ├── config.json   # {"ssl_model_type": "wavlm_large_l23", "spd": false}
            │   └── kmeans_models/
            │       └── <kmeans_file>
            ├── <another_model_id>/
            │   └── ...

        If ``model_id`` is None, files are expected at the root.

        Arguments
        ---------
        repo_id : str
            Local directory path or HuggingFace repo ID (e.g. "username/stacodec").
            If the path exists on disk, loads locally; otherwise downloads from HuggingFace.
        model_id : str, optional
            Model variant name (subfolder) within the repo.
            E.g. "stacodec_wavlm", "stacodec_hubert".
        device : str, optional
            Device for computation.

        Example
        -------
        >>> # Local
        >>> codec = STACodecInference.from_pretrained("./hf_repo", model_id="stacodec_wavlm")
        >>> # HuggingFace
        >>> codec = STACodecInference.from_pretrained("username/stacodec", model_id="stacodec_wavlm")
        """
        import json
        import os
        from semantic_token_extraction import SSL_CONFIGS

        is_local = os.path.isdir(repo_id)

        if is_local:
            base_dir = Path(repo_id) / model_id if model_id else Path(repo_id)

            def resolve(filename):
                p = base_dir / filename
                assert p.exists(), f"File not found: {p}"
                return p
        else:
            from past.models.builders import download_file_from_huggingface
            prefix = f"{model_id}/" if model_id else ""

            def resolve(filename):
                return download_file_from_huggingface(repo_id, f"{prefix}{filename}")

        # Read config
        config_path = resolve("config.json")
        with open(config_path) as f:
            config = json.load(f)

        ssl_model_type = config.get("ssl_model_type", None)
        spd = config.get("spd", False)

        # Resolve kmeans model if needed
        kmeans_dir = None
        if ssl_model_type and not spd:
            ssl_cfg = SSL_CONFIGS[ssl_model_type]
            kmeans_filename = ssl_cfg["kmeans_filename"]
            kmeans_path = resolve(f"kmeans_models/{kmeans_filename}")
            kmeans_dir = str(kmeans_path.parent)

        # Resolve model checkpoint
        checkpoint_path = resolve("model.th")

        return cls(
            model_path=str(checkpoint_path),
            ssl_model_type=ssl_model_type,
            spd=spd,
            device=device,
            kmeans_dir=kmeans_dir,
        )

    def load_audio(self, path: str) -> torch.Tensor:
        """
        Load audio from file and preprocess for the model.

        Arguments
        ---------
        path : str
            Path to the audio file.

        Returns
        -------
        wav : torch.Tensor
            Audio tensor of shape [1, 1, T] on the model's device.
        """
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        if wav.shape[0] == 2:
            wav = wav[:1]  # Convert stereo to mono
        return wav.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode(
        self,
        wav: torch.Tensor,
        ssl_labels: torch.Tensor = None
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio waveform to discrete codes.

        Arguments
        ---------
        wav : torch.Tensor
            Audio tensor of shape [B, 1, T] or [1, 1, T].
        ssl_labels : torch.Tensor, optional
            Pre-computed SSL labels. If None and semantic_extractor is available,
            labels will be extracted automatically.

        Returns
        -------
        codes : torch.Tensor
            Discrete codes from the encoder.
        scale : torch.Tensor
            Scale factor for reconstruction.
        """
        # Ensure wav is on correct device and has correct shape
        if wav.device != self.device:
            wav = wav.to(self.device)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)  # [1, T] -> [1, 1, T]

        # Extract SSL labels if needed
        if ssl_labels is None and self.semantic_extractor is not None and not self.spd:
            ssl_labels = self.semantic_extractor.extract(wav.squeeze(1))  # [B, N]
        
        # import numpy as np
        # ssl_labels = np.random.randint(0, 1000, size=ssl_labels.shape)
        # print("Random ssl_labels shape:", ssl_labels.shape)
        # ssl_labels = torch.tensor(ssl_labels, device=self.device)

        # Encode
        codes, scale = self.model.encode(
            wav,
            ssl_labels=ssl_labels if not self.spd else None,
            use_prediction=self.spd
        )
        # print codes in layer 1
        # print("layer 0 codes:", codes[:, 0, :])
        # print("layer 1 codes:", codes[:, 1, :])
        return codes, scale

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete codes back to audio waveform.

        Arguments
        ---------
        codes : torch.Tensor
            Discrete codes from the encoder.
        scale : torch.Tensor
            Scale factor from encoding.

        Returns
        -------
        wav : torch.Tensor
            Reconstructed audio waveform of shape [B, 1, T].
        """
        return self.model.decode(codes, scale)

    @torch.no_grad()
    def encode_file(self, path: str) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and encode an audio file to discrete codes.

        Arguments
        ---------
        path : str
            Path to the audio file.

        Returns
        -------
        codes : torch.Tensor
            Discrete codes from the encoder.
        scale : torch.Tensor
            Scale factor for reconstruction.
        """
        wav = self.load_audio(path)
        return self.encode(wav)

    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Encode and decode audio (full reconstruction pipeline).

        Arguments
        ---------
        wav : torch.Tensor
            Audio tensor of shape [B, 1, T] or [1, 1, T].

        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed audio waveform.
        """
        codes, scale = self.encode(wav)
        return self.decode(codes, scale)

    @torch.no_grad()
    def reconstruct_file(self, path: str) -> torch.Tensor:
        """
        Load, encode, and decode an audio file.

        Arguments
        ---------
        path : str
            Path to the audio file.

        Returns
        -------
        reconstructed : torch.Tensor
            Reconstructed audio waveform.
        """
        wav = self.load_audio(path)
        return self.reconstruct(wav)

    def save_audio(
        self,
        wav: torch.Tensor,
        path: str,
        rescale: bool = False
    ) -> None:
        """
        Save audio tensor to file.

        Arguments
        ---------
        wav : torch.Tensor
            Audio tensor of shape [B, 1, T] or [1, T].
        path : str
            Output file path.
        rescale : bool, optional
            Whether to rescale audio to avoid clipping (default: False).
        """
        limit = 0.99
        wav = wav.detach().cpu()
        if wav.dim() == 3:
            wav = wav.squeeze(0)  # [1, 1, T] -> [1, T]

        mx = wav.abs().max()
        if rescale:
            wav = wav * min(limit / mx, 1)
        else:
            wav = wav.clamp(-limit, limit)

        path = str(Path(path).with_suffix('.wav'))
        torchaudio.save(path, wav, sample_rate=self.sample_rate, encoding='PCM_S', bits_per_sample=16)

    def codes_to_list(self, codes: torch.Tensor) -> list:
        """
        Convert codes tensor to a Python list for JSON serialization.

        Arguments
        ---------
        codes : torch.Tensor
            Discrete codes tensor.

        Returns
        -------
        list
            Codes as nested Python list.
        """
        return codes[0].cpu().numpy().tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STACodec Inference Example")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--ssl-model-type", type=str, required=True,
                        choices=["hubert_base_l9", "wavlm_large_l23", "none"],
                        help="SSL model type for semantic extraction")
    parser.add_argument("--spd", type=bool, required=True,
                        help="Use semantic prediction instead of external SSL")
    args = parser.parse_args()

    ssl_type = None if args.ssl_model_type == "none" else args.ssl_model_type

    print(f"Loading model from {args.model_path}...")
    codec = STACodecInference(
        model_path=args.model_path,
        ssl_model_type=ssl_type,
        spd=args.spd,
        device=args.device,
    )

    print(f"Processing {args.input}...")
    codes, scale = codec.encode_file(args.input)
    print(f"Encoded to codes with shape: {codes.shape}")

    reconstructed = codec.decode(codes, scale)
    print(f"Decoded to audio with shape: {reconstructed.shape}")

    codec.save_audio(reconstructed, args.output)
    print(f"Saved to {args.output}")
