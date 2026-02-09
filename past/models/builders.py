"""
All the functions to build the relevant models and modules
from the Hydra config.
"""

import omegaconf
import torch
import logging
from pathlib import Path

from audiocraft.quantization import DummyQuantizer, BaseQuantizer
from audiocraft.utils.utils import dict_from_config
from audiocraft.models.builders import get_compression_model as get_compression_model_audiocraft

from past.quantization.vq import ResidualVectorQuantizer
from past.models.past_model import PastModel
from past.auxiliary_tasks.linar_prob import LinarProb
from past.auxiliary_tasks.asr_ctc_lstm import AsrTask
from past.auxiliary_tasks.token_pred_transformer import TokenPredTask
from past.modules.seanet import SEANetEncoder, SEANetDecoder

logger = logging.getLogger()


def get_quantizer(quantizer: str, cfg: omegaconf.DictConfig, dimension: int) -> BaseQuantizer:
    klass = {'no_quant': DummyQuantizer, 'rvq': ResidualVectorQuantizer}[quantizer]
    kwargs = dict_from_config(getattr(cfg, quantizer))
    if quantizer != 'no_quant':
        kwargs['dimension'] = dimension
    return klass(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    if encoder_name == 'seanet':
        kwargs = dict_from_config(getattr(cfg, 'seanet'))
        transformer_params = kwargs.pop('transformer_params', dict())
        encoder_override_kwargs = kwargs.pop('encoder')
        decoder_override_kwargs = kwargs.pop('decoder')
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        if transformer_params.pop('do_dec', False):
            decoder_kwargs['transformer_params'] = transformer_params
        if transformer_params.pop('do_enc', False):
            encoder_kwargs['transformer_params'] = transformer_params
        encoder = SEANetEncoder(**encoder_kwargs)
        decoder = SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_compression_model(cfg: omegaconf.DictConfig) -> PastModel:
    """Instantiate a compression model."""
    if 'past' not in cfg.compression_model:
        return get_compression_model_audiocraft(cfg)
    kwargs = dict_from_config(getattr(cfg, 'encodec'))
    encoder_name = kwargs.pop('autoencoder')
    quantizer_name = kwargs.pop('quantizer')
    encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
    quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
    frame_rate = kwargs['sample_rate'] // encoder.hop_length
    renormalize = kwargs.pop('renormalize', False)
    # deprecated params
    kwargs.pop('renorm', None)
    if hasattr(cfg, 'auxiliary_tasks'):
        auxiliary_tasks_models = []
        for name, one_task_params in cfg.auxiliary_tasks.items():
            params = dict_from_config(one_task_params)
            task_type = params.pop('task_type')
            if not params.pop('apply', False):
                continue
            if task_type == 'linar_probing':
                auxiliary_tasks_models.append(LinarProb(cfg, name=name, **params))
            elif task_type == 'asr':
                auxiliary_tasks_models.append(AsrTask(cfg, name=name, **params))
            elif task_type == 'token_prediction':
                auxiliary_tasks_models.append(
                    TokenPredTask(cfg, name=name, **params)
                )
            else:
                raise ValueError(f"Unsupported auxiliary task: {one_task_params.task_type}")

        kwargs['auxiliary_tasks_models'] = torch.nn.ModuleList(auxiliary_tasks_models)
    return PastModel(encoder, decoder, quantizer, frame_rate=frame_rate, renormalize=renormalize, **kwargs).to(cfg.device)


def get_model_cp_from_huggingface(repo_id: str, filename: str = "model.th") -> Path:
    """Download a model checkpoint from HuggingFace hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "username/stacodec").
        filename: Checkpoint filename in the repo (default: "model.th").
    """
    from huggingface_hub import hf_hub_download

    model_cp_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    logger.info(f"Model checkpoint downloaded to: {model_cp_path}")
    return model_cp_path


def download_file_from_huggingface(repo_id: str, filename: str) -> Path:
    """Download a single file from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))
