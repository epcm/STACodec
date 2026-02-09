import logging
import typing as tp
from pathlib import Path
import os

import torch
from torch import nn

from audiocraft.models.encodec import EncodecModel
from audiocraft import quantization as qt
from audiocraft.utils import checkpoint

logger = logging.getLogger()


class PastModel(EncodecModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: qt.BaseQuantizer,
        frame_rate: int,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        renormalize: bool = False,
        auxiliary_tasks_models: tp.List[nn.Module] = None,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            frame_rate=frame_rate,
            sample_rate=sample_rate,
            channels=channels,
            causal=causal,
            renormalize=renormalize,
        )
        self.auxiliary_tasks_models = auxiliary_tasks_models

    @classmethod
    def from_pretrained(cls, model_cp: str, device: tp.Union[torch.device, str] = None,
                        checkpoint_filename: str = "model.th"):
        """Instantiate a CompressionModel from a local checkpoint path or a HuggingFace repo ID.

        Args:
            model_cp: Local path to a checkpoint file, or a HuggingFace repo ID
                (e.g., "username/stacodec"). When a repo ID is given, the checkpoint
                is downloaded automatically.
            device: Device on which the model is loaded.
            checkpoint_filename: Filename of the checkpoint inside the HuggingFace
                repo (default: "model.th"). Ignored when model_cp is a local path.
        """
        from past.models.builders import get_compression_model, get_model_cp_from_huggingface

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if os.path.exists(model_cp):
            checkpoint_path = Path(model_cp)
            logger.info(f"Loading local compression model from checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = get_model_cp_from_huggingface(model_cp, filename=checkpoint_filename)
            logger.info(f"Checkpoint downloaded from HuggingFace: {checkpoint_path}")
        assert checkpoint_path is not None and Path(checkpoint_path).exists(), f"Checkpoint not found: {checkpoint_path}"
        state = checkpoint.load_checkpoint(checkpoint_path)
        assert state is not None and 'xp.cfg' in state, f"Could not load compression model from ckpt: {checkpoint_path}"
        cfg = state['xp.cfg']
        cfg.device = device
        compression_model = get_compression_model(cfg).to(device)
        assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"
        assert 'best_state' in state and state['best_state'] != {}
        compression_model.load_state_dict(state['best_state']['model'])
        compression_model.eval()
        logger.info("Compression model loaded!")
        return compression_model

    @property
    def has_axiliary_task(self):
        return self.auxiliary_tasks_models is not None and len(self.auxiliary_tasks_models) > 0

    @property
    def device(self):
        """Return the device on which the model is loaded."""
        return next(self.parameters()).device

    @staticmethod
    def calc_frame_valid_mask(audio, audio_tokens, hop_length):
        valid_mask = torch.ones_like(audio_tokens[:, 0], dtype=torch.bool, device=audio_tokens.device)
        T_audio = audio.shape[2]
        T_tokens = audio_tokens.shape[2]
        last_non_zero_audio_index = T_audio - (torch.flip(audio, [-1]) != 0).float().argmax(dim=-1)
        last_non_zero_tokens_index = torch.ceil(last_non_zero_audio_index / hop_length)
        padding_cond = last_non_zero_tokens_index < torch.arange(end=T_tokens, device=audio.device).unsqueeze(0)
        valid_mask[padding_cond] = 0
        return valid_mask

    def forward(self, x: torch.Tensor, targets: dict = None, use_mask: bool = True) -> qt.QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x, return_cnn_features=False)
        valid_frame_mask = self.calc_frame_valid_mask(x, emb, self.encoder.hop_length)

        metrics = {}
        loss = torch.tensor(0.0, device=emb.device)
        if targets:
            ssl_labels = torch.stack(targets['gt_tokens'], dim=0) # B*T*C
            # to device
            ssl_labels = ssl_labels.to(emb.device)
            if self.has_axiliary_task:
                for aux_model in self.auxiliary_tasks_models:
                    loss_, metrics_, predictions = aux_model(emb, targets, valid_frame_mask, use_mask=use_mask)
                    loss += loss_ * aux_model.weight
                    metrics.update(metrics_)
                q_res = self.quantizer(emb, self.frame_rate, ssl_labels=predictions.to(emb.device))
            else:
                q_res = self.quantizer(emb, self.frame_rate, ssl_labels=ssl_labels)
        else:
            q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res, loss, metrics

    def encode(self, x: torch.Tensor, ssl_labels: torch.Tensor = None, use_prediction: bool = False) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        # print params of the current model, sort by size
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        if use_prediction:
            emb = self.encoder(x)
            print("Using predictions for encoding", "has_axiliary_task:", self.has_axiliary_task)
            if self.has_axiliary_task:
                targets = {'gt_tokens': [ssl_labels]}
                valid_frame_mask = self.calc_frame_valid_mask(x, emb, self.encoder.hop_length)
                for aux_model in self.auxiliary_tasks_models:
                    loss_, metrics_, predictions = aux_model(emb, targets, valid_frame_mask)
            labels_used = predictions.to(emb.device)
        else:
            print("Using ssl_labels for encoding")
            emb = self.encoder(x)
            labels_used = ssl_labels.to(emb.device) if ssl_labels is not None else None
            if ssl_labels == None:
                print("SSL labels None")
        for attempt in range(3):
            try:
                # print("Attempting to encode with quantizer")
                codes = self.quantizer.encode(emb, ssl_labels=labels_used)
                break
            except Exception as e:
                # print(f"!!!Error in quantization: {e}, attempt {attempt}")
                emb = self.encoder(x[..., :-320*(attempt+1)])
        return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None, return_latent: bool = False):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        if return_latent:
            return emb
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out
