from abc import ABC, abstractmethod
from torch import nn
import torch


class BaseTask(ABC, nn.Module):
    def __init__(self, cfg, weight, connect_point, name, probs_num=None):
        super(BaseTask, self).__init__()
        self._weight = weight
        self.connect_point = connect_point
        self.name = name
        self.cfg = cfg
        self.probs_num = probs_num

        self.input_dim = self.get_input_dim()
        self.run_on_codebooks = self.connect_point not in ["embeddings", "transformer_encoder"]

    def get_input_dim(self):
        if self.connect_point == 'rvq_centroid':
            return self.cfg.seanet.dimension
        elif self.connect_point == "transformer_encoder":
            return self.cfg.seanet.dimension
        raise ValueError(f"Unsupported connect_point: {self.connect_point}")

    def get_input(self, q_res):
        if self.connect_point == "rvq_centroid":
            return q_res.rvq_centroid
        if self.connect_point == "transformer_encoder":
            return q_res
        raise ValueError(f"Unsupported connect_point: {self.connect_point}")

    @property
    def weight(self):
        return self._weight

    @abstractmethod
    def forward(self, q_res, targets, valid_frame_mask): ...

    @abstractmethod
    def _forward_single_model(self, x, targets, valid_frame_mask, model, i): ...

    def forward_helper(self, q_res, targets, valid_frame_mask, semantic_field):
        device = q_res.device
        x = self.get_input(q_res)

        loss = torch.tensor(0.0, device=device)
        semantic_error = torch.tensor(0.0, device=device)
        metrics = {}
        for i, model in enumerate(self.linar_prob_models):
            _loss, curr_metrics, predictions = self._forward_single_model(x, targets, valid_frame_mask, model, i)
            loss += _loss
            s = curr_metrics[semantic_field[:-1]]
            semantic_error += s if semantic_field[-1] == '+' else -s
            for key, value in curr_metrics.items():
                metrics[f'{self.name}_{key}_{i}' if self.run_on_codebooks else f'{self.name}_{key}'] = value
        metrics[f'semantic_error'] = semantic_error
        metrics[f'loss_{self.name}'] = loss
        return loss, metrics, predictions
