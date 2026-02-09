"""
All the functions to build the relevant solvers and used objects
from the Hydra config.
"""

from enum import Enum
import logging
import typing as tp

import flashy
import omegaconf
import torch

# LRScheduler was renamed in some torch versions
try:
    from torch.optim.lr_scheduler import LRScheduler  # type: ignore
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from audiocraft.solvers.base import StandardSolver
from audiocraft.solvers.builders import get_solver as get_solver_audiocraft
from audiocraft import data, optim
from audiocraft.utils.utils import dict_from_config, get_loader
# from past.data.phone_speech_dataset import PhoneSpeechDataset
from past.data.ssl_phone_speech_dataset import SSLSpeechDataset
from past.data.phone_speech_dataset import PhoneSpeechDataset


logger = logging.getLogger(__name__)


class DatasetType(Enum):
    AUDIO = "audio"
    MUSIC = "music"
    SOUND = "sound"
    SPEECH = "speech"
    PHONE_SPEECH = "phone_speech"


def get_solver(cfg: omegaconf.DictConfig) -> StandardSolver:
    """Instantiate solver from config."""
    from past.solvers.compression_asr_phone import CompressionAsrPhoneSolver

    solvers = {
        'compression_asr_phone': CompressionAsrPhoneSolver,
    }
    if cfg.solver not in solvers:
        return get_solver_audiocraft(cfg)
    klass = solvers[cfg.solver]
    return klass(cfg)  # type: ignore


def get_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: omegaconf.DictConfig, total_updates: int) -> tp.Optional[LRScheduler]:
    """Build torch learning rate scheduler from config and associated optimizer.
    Supported learning rate schedulers: ExponentialLRScheduler, PlateauLRScheduler

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        cfg (DictConfig): Schedule-related configuration.
        total_updates (int): Total number of updates.
    Returns:
        torch.optim.Optimizer.
    """
    if 'lr_scheduler' not in cfg:
        raise KeyError("LR Scheduler not found in config")

    lr_sched: tp.Optional[LRScheduler] = None
    if cfg.lr_scheduler == 'step':
        lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, **cfg.step)
    elif cfg.lr_scheduler == 'exponential':
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.exponential)
    elif cfg.lr_scheduler == 'cosine':
        kwargs = dict_from_config(cfg.cosine)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.CosineLRScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_updates, **kwargs)
    elif cfg.lr_scheduler == 'polynomial_decay':
        kwargs = dict_from_config(cfg.polynomial_decay)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.PolynomialDecayLRScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_updates, **kwargs)
    elif cfg.lr_scheduler == 'inverse_sqrt':
        kwargs = dict_from_config(cfg.inverse_sqrt)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.InverseSquareRootLRScheduler(optimizer, warmup_steps=warmup_steps, **kwargs)
    elif cfg.lr_scheduler == 'linear_warmup':
        kwargs = dict_from_config(cfg.linear_warmup)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.LinearWarmupLRScheduler(optimizer, warmup_steps=warmup_steps, **kwargs)
    elif cfg.lr_scheduler is not None:
        raise ValueError(f"Unsupported LR Scheduler: {cfg.lr_scheduler}")
    return lr_sched


def get_audio_datasets(cfg: omegaconf.DictConfig, dataset_type: DatasetType = DatasetType.AUDIO) -> tp.Dict[str, torch.utils.data.DataLoader]:
    """Build AudioDataset from configuration.

    Args:
        cfg (omegaconf.DictConfig): Configuration.
        dataset_type: The type of dataset to create.
    Returns:
        dict[str, torch.utils.data.DataLoader]: Map of dataloader for each data split.
    """
    dataloaders: dict = {}

    sample_rate = cfg.sample_rate
    channels = cfg.channels
    seed = cfg.seed
    max_sample_rate = cfg.datasource.max_sample_rate
    max_channels = cfg.datasource.max_channels

    assert cfg.dataset is not None, "Could not find dataset definition in config"

    dataset_cfg = dict_from_config(cfg.dataset)
    splits_cfg: dict = {}
    splits_cfg['train'] = dataset_cfg.pop('train')
    splits_cfg['valid'] = dataset_cfg.pop('valid')
    splits_cfg['evaluate'] = dataset_cfg.pop('evaluate')
    splits_cfg['generate'] = dataset_cfg.pop('generate')
    execute_only_stage = cfg.get('execute_only', None)

    assert cfg.sample_rate <= max_sample_rate, f"Expecting a max sample rate of {max_sample_rate} for datasource but {sample_rate} found."
    assert cfg.channels <= max_channels, f"Expecting a max number of channels of {max_channels} for datasource but {channels} found."

    for split, paths in cfg.datasource.items():
        if not isinstance(paths, omegaconf.DictConfig):
            continue  # skipping this as not a path
        if execute_only_stage is not None and split != execute_only_stage:
            continue
        logger.info(f"Loading audio data split {split}")

        split_cfg = splits_cfg[split]
        split_kwargs = {k: v for k, v in split_cfg.items()}
        kwargs = {**dataset_cfg, **split_kwargs}  # split kwargs overrides default dataset_cfg
        kwargs['sample_rate'] = sample_rate
        kwargs['channels'] = channels

        if kwargs.get('permutation_on_files') and cfg.optim.updates_per_epoch:
            kwargs['num_samples'] = flashy.distrib.world_size() * cfg.dataset.batch_size * cfg.optim.updates_per_epoch

        num_samples = kwargs['num_samples']
        shuffle = kwargs['shuffle']

        return_info = kwargs.pop('return_info')
        batch_size = kwargs.pop('batch_size', None)
        num_workers = kwargs.pop('num_workers')

        if dataset_type == DatasetType.PHONE_SPEECH:
            assert return_info, "PhoneSpeechDataset requires return_info=True"
            kwargs['do_phone'] = cfg.auxiliary_tasks.phone_quant.apply
            kwargs['do_char'] = cfg.auxiliary_tasks.asr_quant.apply

            librispeech_path = paths.get('librispeech', None)
            timit_path = paths.get('timit', None)
            if librispeech_path is not None and timit_path is not None:
                timit_probability = cfg.datasource.timit_p
            elif librispeech_path is not None:
                timit_probability = 0.0
            elif timit_path is not None:
                timit_probability = 1.0
            else:
                raise ValueError("At least one of librispeech or timit paths must be provided for PhoneSpeechDataset")

            kwargs['ssl_token_file'] = paths.get('tokens', None)
            print("ssl_token_file:", kwargs['ssl_token_file'])

            dataset = SSLSpeechDataset.from_meta(librispeech_path, timit_path, timit_probability, **kwargs)
        else:
            raise ValueError(f"Dataset type is unsupported: {dataset_type}")

        loader = get_loader(
            dataset,
            num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            collate_fn=dataset.collater if return_info else None,
            shuffle=shuffle,
        )
        dataloaders[split] = loader

    return dataloaders
