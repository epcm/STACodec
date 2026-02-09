import torch
from torch import nn
import numpy as np
from jiwer import wer, cer
import speechbrain as sb

from past.modules.ctc_encoder import CTCTextEncoder
from past.modules.asr_lstm import LSTM
from past.auxiliary_tasks.base_task import BaseTask
import logging

logger = logging.getLogger(__name__)


class AsrCtcLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=30, bidirectional=True):
        super(AsrCtcLSTM, self).__init__()
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
        self.ctc_lin: nn.Module = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linar_layer(x)  # (B, T, emb_dim) -> (B, T, H)
        y, _ = self.enc(x)  # (B, T, emb_dim) -> (B, T, H*2)
        logits = self.ctc_lin(y)  # (B, T, H*2) -> (B, T, 30)
        return logits


class AsrTask(BaseTask):
    def __init__(self, cfg, weight, connect_point, name, hidden_dim, ctc_encoder_path=None, probs_num=None, mode='lstm', bidirectional=True):
        super(AsrTask, self).__init__(cfg, weight, connect_point, name, probs_num)
        assert connect_point != 'tokens', 'tokens not supported for LinarProb yet'
        self.ctc_encoder = CTCTextEncoder.from_saved(ctc_encoder_path)
        if mode == 'lstm':
            self.linar_prob_models: nn.ModuleList = nn.ModuleList(
                [
                    AsrCtcLSTM(
                        self.input_dim,
                        hidden_dim,
                        len(self.ctc_encoder.ind2lab),
                        bidirectional=bidirectional,
                    )
                    for _ in range(probs_num or 1)
                ]
            )
        elif mode == 'simple':
            self.linar_prob_models: nn.ModuleList = nn.ModuleList([nn.Linear(self.input_dim, len(self.ctc_encoder.ind2lab)) for _ in range(probs_num or 1)])
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def forward(self, q_res, targets, valid_frame_mask):
        return self.forward_helper(q_res, targets, valid_frame_mask, 'wer+')

    def _forward_single_model(self, x, targets, valid_frame_mask, model, i):
        if self.run_on_codebooks:
            x = x[:, i]
        if self.connect_point != "transformer_encoder":
            x = x.permute(0, 2, 1)  # B, CB, C, T -> B, T, C
        logits = model(x)

        x_len = valid_frame_mask.sum(dim=-1)
        t_len = []
        encode_sequences = []
        for transcription in targets['transcription']:
            encode_sequences.append(self.ctc_encoder.encode_sequence_torch(transcription))
            t_len.append(encode_sequences[-1].shape[0])
        t_len = torch.tensor(t_len)
        encode_sequences = torch.nn.utils.rnn.pad_sequence(encode_sequences, batch_first=True)

        blank_index = 0
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        input_lens_rel = x_len / x_len.max()
        p_tokens = sb.decoders.ctc_greedy_decode(log_probs, input_lens_rel, blank_id=blank_index)
        loss = ctc_loss(log_probs, encode_sequences, x_len, t_len, blank_index=blank_index, reduction="mean")

        target_words = [''.join(transcription) for transcription in targets['transcription']]
        predicted_words = ["".join(self.ctc_encoder.decode_ndim(utt_seq)) for utt_seq in p_tokens]
        try:
            wer_score = wer(target_words, predicted_words)
            cer_score = cer(target_words, predicted_words)
        except ValueError as e:
            logger.debug(f'WER or CER faild with error: {e}')
            wer_score = 1.0
            cer_score = 1.0

        metrics = {'wer': wer_score, 'cer': cer_score, 'ctc_loss': loss}
        return loss, metrics


def ctc_loss(log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"):
    """CTC loss.

    Arguments
    ---------
    log_probs : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'batch',
        'batchmean', 'none'.
        See pytorch for 'mean', 'sum', 'none'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Returns
    -------
    The computed CTC loss.
    """
    # input_lens = (input_lens * log_probs.shape[1]).round().int()
    # target_lens = (target_lens * targets.shape[1]).round().int()
    log_probs = log_probs.transpose(0, 1)

    if reduction == "batchmean":
        reduction_loss = "sum"
    elif reduction == "batch":
        reduction_loss = "none"
    else:
        reduction_loss = reduction
    loss = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        zero_infinity=True,
        reduction=reduction_loss,
    )

    if reduction == "batchmean":
        return loss / targets.shape[0]
    elif reduction == "batch":
        N = loss.size(0)
        return loss.view(N, -1).sum(1) / target_lens.view(N, -1).sum(1)
    else:
        return loss
