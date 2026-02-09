import torch
from torch import nn
import numpy as np
from past.modules.asr_lstm import LSTM
from past.auxiliary_tasks.base_task import BaseTask
import logging

logger = logging.getLogger(__name__)


class TokenPredictLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1000, bidirectional=True):
        super(TokenPredictLSTM, self).__init__()
        self.linar_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.enc: nn.Module = LSTM(
            hidden_size=hidden_dim,
            input_shape=(None, None, hidden_dim),
            num_layers=1,
            bias=True,
            dropout=0.2,
            re_init=True,
            bidirectional=bidirectional,
        )

        input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.ce_lin: nn.Module = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # print(f"TokenPredictLSTM input shape: {x.shape}")
        x = self.linar_layer(x)  # (B, T, emb_dim) -> (B, T, H)
        y, _ = self.enc(x)  # (B, T, emb_dim) -> (B, T, H*2)
        logits = self.ce_lin(y)  # (B, T, H*2) -> (B, T, 1000)
        return logits


class TokenPredTask(BaseTask):
    def __init__(self, cfg, weight, connect_point, name, hidden_dim, probs_num=None, mode='lstm', bidirectional=True):
        super(TokenPredTask, self).__init__(cfg, weight, connect_point, name, probs_num)
        assert connect_point != 'tokens', 'tokens not supported for TokenPred yet'
        
        if mode == 'lstm':
            self.linar_prob_models: nn.ModuleList = nn.ModuleList(
                [
                    TokenPredictLSTM(
                        self.input_dim,
                        hidden_dim,
                        1000,  # Fixed vocab size of 1000
                        bidirectional=bidirectional,
                    )
                    for _ in range(probs_num or 1)
                ]
            )
        elif mode == 'simple':
            self.linar_prob_models: nn.ModuleList = nn.ModuleList([nn.Linear(self.input_dim, 1000) for _ in range(probs_num or 1)])
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def forward(self, q_res, targets, valid_frame_mask):
        return self.forward_helper(q_res, targets, valid_frame_mask, 'ce_loss+')

    def _forward_single_model(self, x, targets, valid_frame_mask, model, i):
        if self.run_on_codebooks:
            x = x[:, i]
        # if self.connect_point != "transformer_encoder":
        x = x.permute(0, 2, 1)  # B, CB, C, T -> B, T, C
        # print("_forward_single_model input shape:", x.shape)
        logits = model(x)

        # Assume targets['tokens'] contains token IDs for each frame
        # Shape: (B, T) with values in range [0, 999]

        valid_logits = logits.permute(0, 2, 1) # [B， 1000， T]
        valid_targets = torch.stack(targets['gt_tokens'], dim=0)[:, :, -1] # [B, T]
        valid_targets = valid_targets.to(valid_logits.device)

        # print("valid_logits shape:", valid_logits.shape)
        # print("valid_targets shape:", valid_targets.shape)
        # print("valid_logits:", valid_logits)
        # print("valid_targets:", valid_targets)
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(valid_logits, valid_targets, reduction="mean")
        # print(f"TokenPredTask loss: {loss.item()}")
        
        # Calculate accuracy
        predictions = torch.argmax(valid_logits, dim=1)
        accuracy = (predictions == valid_targets).float().mean()
        # print(f"TokenPredTask accuracy: {accuracy.item()}")
        # raise ValueError("The quantizer should be implemented in the subclass.")
        
        metrics = {'pred_acc': accuracy.item(), 'ce_loss': loss.item()}
        return loss, metrics, predictions
