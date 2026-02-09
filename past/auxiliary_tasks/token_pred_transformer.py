import torch
from torch import nn
import numpy as np
from past.auxiliary_tasks.base_task import BaseTask
import logging
from past.modules.embedding_transformer import EmbeddingTransformer
import random

logger = logging.getLogger(__name__)


class TokenPredictTransformer(nn.Module):

    def __init__(self, transformer_params, mask_params, output_dim=1000, bidirectional=True):
        super(TokenPredictTransformer, self).__init__()
        self.enc: nn.Module = EmbeddingTransformer(**transformer_params)
        enc_dim = transformer_params['enc_dim']
        self.ce_lin: nn.Module = nn.Linear(enc_dim, output_dim)
        self.use_mask = mask_params.get('use_mask', False) if mask_params else False
        if self.use_mask:
            self.mask_time_prob = mask_params.get('mask_time_prob', 0.065)
            self.num_time_masks = mask_params.get('num_time_masks', 2)
            self.mask_time_length = mask_params.get('mask_time_length', 10)
            self.mask_feature_prob = mask_params.get('mask_feature_prob', 0.5)
            self.num_feature_masks = mask_params.get('num_feature_masks', 2)
            self.mask_feature_length = mask_params.get('mask_feature_length', 32)

    def _generate_mask(self, x):
        # x: [batch_size, seq_len, feat_dim]
        batch_size, seq_len, feat_dim = x.size()

        # 3D boolean mask
        mask = torch.zeros(batch_size, seq_len, feat_dim, dtype=torch.bool, device=x.device)
        if self.use_mask:
            for i in range(batch_size):
                num_spans_time = self.num_time_masks
                num_spans_feat = self.num_feature_masks
                if random.random() > self.mask_time_prob:
                    num_spans_time = 0
                if random.random() > self.mask_feature_prob:
                    num_spans_feat = 0
                if num_spans_time > 0:
                    valid_starts = np.arange(seq_len - self.mask_time_length + 1)
                    starts = np.random.choice(valid_starts, size=num_spans_time, replace=False)
                    for s in starts:
                        mask[i, s : s + self.mask_time_length, :] = True

                if num_spans_feat > 0:
                    valid_fstarts = np.arange(feat_dim - self.mask_feature_length + 1)
                    fstarts = np.random.choice(valid_fstarts, size=num_spans_feat, replace=False)
                    for f in fstarts:
                        mask[i, :, f : f + self.mask_feature_length] = True
        return mask

    def forward(self, x):
        if self.use_mask:
            mask = self._generate_mask(x)
            x = x.masked_fill(mask, 0.0)
        y = self.enc(x.permute(0, 2, 1))  # (B, T, emb_dim) -> (B, T, H*2)
        y = y.permute(0, 2, 1)  # (B, T, H*2) -> (B, H*2, T)
        logits = self.ce_lin(y)  # (B, H*2, T) -> (B, 1000, T)
        return logits


class TokenPredTask(BaseTask):
    def __init__(self, cfg, weight, connect_point, name, hidden_dim, probs_num=None, mode='lstm', mask_params=None, transformer_params=None):
        super(TokenPredTask, self).__init__(cfg, weight, connect_point, name, probs_num)
        assert connect_point != 'tokens', 'tokens not supported for TokenPred yet'
        if mode == 'transformer':
            self.linar_prob_models: nn.ModuleList = nn.ModuleList(
                [
                    TokenPredictTransformer(transformer_params, mask_params, output_dim=1000)
                    for _ in range(probs_num or 1)
                ]
            )
        elif mode == 'simple':
            self.linar_prob_models: nn.ModuleList = nn.ModuleList([nn.Linear(self.input_dim, 1000) for _ in range(probs_num or 1)])
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def forward(self, q_res, targets, valid_frame_mask, use_mask=False):
        org_mask = self.linar_prob_models[0].use_mask
        self.linar_prob_models[0].use_mask = use_mask
        res = self.forward_helper(q_res, targets, valid_frame_mask, 'ce_loss+')
        self.linar_prob_models[0].use_mask = org_mask
        return res

    def _forward_single_model(self, x, targets, valid_frame_mask, model, i):
        if self.run_on_codebooks:
            x = x[:, i]
        x = x.permute(0, 2, 1)  # B, CB, C, T -> B, T, C
        logits = model(x)
        if targets['gt_tokens'][0] is None:
            print("No targets provided, returning predictions only.")
            valid_logits = logits.permute(0, 2, 1)
            predictions = torch.argmax(valid_logits, dim=1)
            return torch.tensor(0.0, device=logits.device), {'pred_acc': torch.tensor(0.0, device=logits.device), 'ce_loss': torch.tensor(0.0, device=logits.device)}, predictions


        valid_logits = logits.permute(0, 2, 1) # [B， 1000， T]
        valid_targets = torch.stack(targets['gt_tokens'], dim=0)[:, :, -1] # [B, T]
        valid_targets = valid_targets.to(valid_logits.device)
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(valid_logits, valid_targets, reduction="mean")
        # print(f"TokenPredTask loss: {loss.item()}")
        
        # Calculate accuracy
        predictions = torch.argmax(valid_logits, dim=1)
        accuracy = (predictions == valid_targets).float().mean()
        
        metrics = {'pred_acc': accuracy.item(), 'ce_loss': loss.item()}
        return loss, metrics, predictions
