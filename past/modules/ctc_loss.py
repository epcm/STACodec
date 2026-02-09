import torch
from torch.nn import Module
import speechbrain as sb
from jiwer import wer, cer
from past.modules.ctc_encoder import CTCTextEncoder

# def ctc_loss(log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"):
#     """CTC loss.

#     Arguments
#     ---------
#     log_probs : torch.Tensor
#         Predicted tensor, of shape [batch, time, chars].
#     targets : torch.Tensor
#         Target tensor, without any blanks, of shape [batch, target_len]
#     input_lens : torch.Tensor
#         Length of each utterance.
#     target_lens : torch.Tensor
#         Length of each target sequence.
#     blank_index : int
#         The location of the blank symbol among the character indexes.
#     reduction : str
#         What reduction to apply to the output. 'mean', 'sum', 'batch',
#         'batchmean', 'none'.
#         See pytorch for 'mean', 'sum', 'none'. The 'batch' option returns
#         one loss per item in the batch, 'batchmean' returns sum / batch size.

#     Returns
#     -------
#     The computed CTC loss.
#     """
#     # input_lens = (input_lens * log_probs.shape[1]).round().int()
#     # target_lens = (target_lens * targets.shape[1]).round().int()
#     log_probs = log_probs.transpose(0, 1)

#     if reduction == "batchmean":
#         reduction_loss = "sum"
#     elif reduction == "batch":
#         reduction_loss = "none"
#     else:
#         reduction_loss = reduction
#     loss = torch.nn.functional.ctc_loss(
#         log_probs,
#         targets,
#         input_lens,
#         target_lens,
#         blank_index,
#         zero_infinity=True,
#         reduction=reduction_loss,
#     )

#     if reduction == "batchmean":
#         return loss / targets.shape[0]
#     elif reduction == "batch":
#         N = loss.size(0)
#         return loss.view(N, -1).sum(1) / target_lens.view(N, -1).sum(1)
#     else:
#         return loss
