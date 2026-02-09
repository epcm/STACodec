import torch
from torch import nn
from torch.nn import functional as F

from past.auxiliary_tasks.base_task import BaseTask
from past.data.phone_utils import phones2idx, chars2idx


class LinarProb(BaseTask):
    def __init__(self, cfg, dataset, weight, connect_point, name, probs_num=None, loss_type='ce'):
        super(LinarProb, self).__init__(cfg, weight, connect_point, name, probs_num)
        assert connect_point != 'tokens', 'tokens not supported for LinarProb yet'
        self.loss_type = loss_type
        self.dataset = dataset

        self.linar_prob_models: nn.ModuleList = nn.ModuleList([self._get_model(self._get_vocab_size(), self.input_dim) for _ in range(probs_num or 1)])

    def _get_vocab_size(self):
        if self.dataset == 'phones':
            vocab_size = len(phones2idx)
        elif self.dataset == 'chars':
            vocab_size = len(chars2idx)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        return vocab_size

    def _get_model(self, vocab_size, emb_dim):
        return torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, q_res, targets, valid_frame_mask):
        return self.forward_helper(q_res, targets, valid_frame_mask, 'acc-')

    def _forward_single_model(self, x, targets, valid_mask, model, i):
        if self.run_on_codebooks:
            x = x[:, i].permute(0, 2, 1)  # B, CB, C, T -> B, T, C

        prob = model(x)
        target_prob = targets['phone_mat'].to(prob.device)
        use_sample_mask = (target_prob != 0).any(dim=-1).any(dim=-1)
        metrics = {}

        if use_sample_mask.any():
            target_prob_use = target_prob[use_sample_mask]
            valid_mask_use = valid_mask[use_sample_mask]
            pred_prob_use = prob[use_sample_mask]

            curr_acc_sum = ((torch.argmax(pred_prob_use, dim=-1) == target_prob_use.argmax(dim=-1)).float() * valid_mask_use).sum()
            curr_acc_num = valid_mask_use.sum()

            metrics['acc'] = curr_acc_sum / max(curr_acc_num, 1e-9)

            valid_mask_reshape = valid_mask_use.reshape(-1)
            pred_prob_reshape = pred_prob_use.reshape(-1, pred_prob_use.shape[-1])[valid_mask_reshape]
            target_mat_reshape = target_prob_use.reshape(-1, target_prob_use.shape[-1])[valid_mask_reshape]
            metrics['ce'] = nn.functional.cross_entropy(pred_prob_reshape, target_mat_reshape)
        else:
            metrics['acc'] = 0
            metrics['ce'] = prob.mean() * 1e-9
        return metrics[self.loss_type], metrics
