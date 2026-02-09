import torch
import torch.nn as nn

from transformers import BertModel, BertConfig


class EmbeddingTransformer(torch.nn.Module):
    def __init__(
        self,
        enc_dim,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        max_frames_in_trans,
        hidden_dropout_prob,
        overlap_frames,
        causal=False,
        **kwargs,
    ):
        super().__init__()
        self.max_frames_in_trans = max_frames_in_trans
        self.overlap_frames = overlap_frames
        self.num_attention_heads = num_attention_heads
        self.causal = causal

        self.in_projection: nn.Module = torch.nn.Linear(enc_dim, hidden_size)
        self.out_projection: nn.Module = torch.nn.Linear(hidden_size, enc_dim)
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_frames_in_trans,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.transformer_encoder: nn.Module = BertModel(config)

        self.do_skip_connection = kwargs.get('do_skip_connection', False)
        self.drop_transformer_prob = kwargs.get('drop_transformer_prob', 0.0)
        self.drop_skip_connection_prob = kwargs.get('drop_skip_connection_prob', 0.0)
        assert self.drop_transformer_prob + self.drop_skip_connection_prob <= 1.0, "Sum of drop probabilities should be less than 1"

    def _apply_transformer(self, embeddings, att_mask=None):
        # embeddings: (N * num_heads, target_seq_len, source_seq_len)
        if self.causal:
            causal_mask = torch.triu(torch.ones(embeddings.size(1), embeddings.size(1), device=embeddings.device), diagonal=1).bool()
            if att_mask is not None:
                encoder_attention_mask = (causal_mask.unsqueeze(0) & (~att_mask.unsqueeze(1))).float()
            else:
                encoder_attention_mask = causal_mask.float()
        else:
            encoder_attention_mask = (~att_mask).float() if att_mask is not None else None
        emb_bert = self.transformer_encoder(inputs_embeds=embeddings, encoder_attention_mask=encoder_attention_mask)
        emb_after_transformer = emb_bert.last_hidden_state
        return emb_after_transformer

    def run_transformer_in_chunks(self, embeddings):
        att_mask = torch.zeros_like(embeddings[:, :, 0]).bool()
        B, T, C = embeddings.size()
        M = self.max_frames_in_trans
        step = M - self.overlap_frames
        device = embeddings.device
        embeddings_windows, mask_windows = [], []
        for start in range(0, T, step):
            embeddings_windows.append(embeddings[:, start : start + M, :])
            mask_windows.append(att_mask[:, start : start + M])
            last_len = embeddings_windows[-1].size(1)
            if last_len < M:
                emb_pad_vector = torch.zeros(B, M - last_len, C, device=device)
                embeddings_windows[-1] = torch.cat([embeddings_windows[-1], emb_pad_vector], dim=1)
                mask_pad_vector = torch.ones(B, M - last_len, device=device).bool()
                mask_windows[-1] = torch.cat([mask_windows[-1], mask_pad_vector], dim=1)
        embeddings_len = torch.stack(embeddings_windows, dim=1)  # Shape: (batch_size, num_windows, M, feature_dim)
        embeddings_len = embeddings_len.view(-1, M, C)  # Shape: (batch_size * num_windows, M, feature_dim)

        att_mask_len = torch.stack(mask_windows, dim=1)  # Shape: (batch_size, num_windows, M)
        att_mask_len = att_mask_len.view(-1, M)  # Shape: (batch_size * num_windows, M)

        emb_after_transformer = self._apply_transformer(embeddings_len, att_mask_len)
        emb_after = emb_after_transformer.view(B, -1, M, C)

        merged_signal = torch.zeros((B, T, C), device=device)
        count = torch.zeros((B, T, C), device=device)
        for i, start in enumerate(range(0, T, step)):
            merged_signal[:, start : start + M] += emb_after[:, i, : (M if start + M <= T else T - start)]
            count[:, start : start + M] += 1
        emb_after = merged_signal / count  # Normalize overlapping areas
        return emb_after

    def calc_embeddings_after_drop_and_mask(self, embeddings, trans_output):
        # Explanation table:
        #  p    |   [0, drop_skip]  | [drop_skip, 1-drop_trans]         |               [1-drop_trans, 1]
        #  -    |   -               |  -                                |       masked              | not masked
        # line1 | 0.5 trans_output  | 0.5 embeddings                    | 0 (embeddings afte fill)  | 0.5 embeddings
        # line2 | 0.5 trans_output  | 0.5 trans_output                  | 0.5 * 2 *trans_output     | 0.5 embeddings
        # total | trans_output      | 0.5 embeddings + 0.5 trans_output | trans_output              | embeddings
        p = torch.rand(embeddings.size(0), 1, 1, device=embeddings.device)
        new_e = torch.zeros_like(embeddings)
        new_e = new_e + torch.where(p < self.drop_skip_connection_prob, trans_output, embeddings) * 0.5
        embeddings = new_e + torch.where((1 - p) < self.drop_transformer_prob, embeddings, trans_output) * 0.5
        return embeddings

    def forward(self, embeddings):
        embeddings = self.in_projection(embeddings.transpose(1, 2))
        if embeddings.size(1) > self.max_frames_in_trans:
            assert not self.training, "Training with input length > max_frames"
            trans_output = self.run_transformer_in_chunks(embeddings)
        else:
            trans_output = self._apply_transformer(embeddings)

        if not self.do_skip_connection:
            embeddings = trans_output
        elif self.training:
            embeddings = self.calc_embeddings_after_drop_and_mask(embeddings, trans_output)
        else:  # Inference + do_skip_connection
            embeddings = 0.5 * embeddings + 0.5 * trans_output

        embeddings = self.out_projection(embeddings)
        return embeddings.transpose(1, 2)
