from stacodec_inference import STACodecInference

codec = STACodecInference.from_pretrained(
    repo_id="kaiyuanzhang0808/stacodec",
    model_id="stacodec_wavlm", # stacodec_wavlm, stacodec_hubert, stacodec_wavlm_spd
    device="cuda:0",
)

# Encode
codes, scale = codec.encode_file("./assets/1089-134686-0004.flac")
print("Encoded codes shape:", codes.shape)

# Decode
reconstructed = codec.decode(codes, scale)
codec.save_audio(reconstructed, "reconstructed.wav")
