# STACodec

[![arXiv](https://img.shields.io/badge/arXiv-2602.06180-b31b1b.svg)](https://arxiv.org/abs/2602.06180)

[ICASSP 2026] STACodec: Semantic Token Assignment for Balancing Acoustic Fidelity and Semantic Information in Audio Codecs

## Installation

```bash
conda create -n stacodec python=3.10 -y
conda activate stacodec
conda install -y -c conda-forge "ffmpeg=6.1.*" "pkg-config"
pip install -r requirements.txt
```

## Quick Start

```python
from stacodec_inference import STACodecInference

codec = STACodecInference.from_pretrained(
    repo_id="kaiyuanzhang0808/stacodec",
    model_id="stacodec_wavlm",  # stacodec_wavlm | stacodec_hubert | stacodec_wavlm_spd
    device="cuda:0",
)

# Encode audio to discrete codes
codes, scale = codec.encode_file("audio.wav")
print("Codes shape:", codes.shape)  # [B, N_codebooks, T]

# Decode codes back to audio
reconstructed = codec.decode(codes, scale)
codec.save_audio(reconstructed, "reconstructed.wav")
```

## Acknowledgements

We borrowed a lot of code from [PAST](https://github.com/slp-rl/PAST).
