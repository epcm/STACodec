import logging

logger = logging.getLogger(__name__)

import multiprocessing
import typing as tp
from pathlib import Path

import flashy
import torch
from flashy.distrib import rank_zero_only

try:
    import wandb
except ModuleNotFoundError:
    wandb = None
    logger.warning("wandb not found, logging will not be available.")

# Add imports for audio quality metrics
try:
    from pesq import pesq
except ImportError:
    pesq = None
    logger.warning("pesq not found, PESQ metric will not be available.")

try:
    from pystoi import stoi
except ImportError:
    stoi = None
    logger.warning("pystoi not found, STOI metric will not be available.")

try:
    from past.solvers.pseudo_mos import pseudo_mos_setup, pseudo_mos_metric
except ImportError:
    pseudo_mos_setup = None
    pseudo_mos_metric = None
    logger.warning("pseudo_mos not found, UTMOS metric will not be available.")

from audiocraft.utils import checkpoint
from audiocraft.solvers.builders import get_optimizer, get_lr_scheduler, get_optim_parameter_groups
from audiocraft import quantization
from audiocraft.utils.samples.manager import SampleManager
from audiocraft.utils.utils import get_pool_executor
from audiocraft.solvers.compression import CompressionSolver, evaluate_audio_reconstruction
from past.data.ssl_phone_speech_dataset import SSLSpeechDataset
from past.data.phone_speech_dataset import PhoneSpeechDataset
from past.solvers.builders import DatasetType, get_audio_datasets
from past.models.builders import get_compression_model, get_model_cp_from_huggingface


# -------------------------------------------------------------------
# Module‐level cache for UTMOS predictors
# -------------------------------------------------------------------
_utmos_predictor_dict: tp.Optional[dict] = None
_utmos_predictor_fs: tp.Optional[dict] = None

def _get_utmos_predictors(cfg):
    """
    Initialize UTMOS once per worker. Returns (predictor_dict, predictor_fs).
    Subsequent calls will reuse the cached predictors.
    """
    global _utmos_predictor_dict, _utmos_predictor_fs

    if pseudo_mos_setup is None:
        return None, None

    if _utmos_predictor_dict is None:
        try:
            _utmos_predictor_dict, _utmos_predictor_fs = pseudo_mos_setup(
                ["utmos"],
                predictor_args={},
                use_gpu=torch.cuda.is_available()
            )
            logger.info("UTMOS predictor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize UTMOS model: {e}")
            _utmos_predictor_dict, _utmos_predictor_fs = None, None

    return _utmos_predictor_dict, _utmos_predictor_fs


class CompressionAsrPhoneSolver(CompressionSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Don't initialize UTMOS here to avoid pickling issues

    def build_model(self):
        self.model = get_compression_model(self.cfg).to(self.device)
        self.optimizer = get_optimizer(get_optim_parameter_groups(self.model), self.cfg.optim)
        if getattr(self.cfg, 'schedule', None):
            self.lr_scheduler = get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
        else:
            self.lr_scheduler = None
        self.register_stateful('model', 'optimizer', 'lr_scheduler')
        self.register_best_state('model')
        self.register_ema('model')

        for dataset in self.dataloaders.values():
            d = dataset
            while not isinstance(d, SSLSpeechDataset) and not isinstance(d, PhoneSpeechDataset):
                d = d.dataset
            d.set_hop_length(self.model.encoder.hop_length)

    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = get_audio_datasets(self.cfg, DatasetType.PHONE_SPEECH)

    def load_state_dict(self, state):
        for name, sub_state in state.items():
            if name == 'model':
                model = getattr(self.stateful.sources[name].owner, name)
                incompat_keys = model.load_state_dict(sub_state, strict=False)
                if len(incompat_keys.unexpected_keys) > 0:
                    if all(k.startswith('auxiliary') for k in incompat_keys.unexpected_keys):
                        logger.warning(f"Found auxiliary tasks in model: {incompat_keys.unexpected_keys}")
                        logger.warning("Skipping auxiliary tasks loading")
                    else:
                        raise ValueError(f"Unexpected keys in model: {incompat_keys.unexpected_keys}")
                if len(incompat_keys.missing_keys) > 0:
                    if all(k.startswith('auxiliary') for k in incompat_keys.missing_keys):
                        logger.warning(f"Auxiliary tasks not found in model: {incompat_keys.missing_keys}")
                        logger.warning("Skipping auxiliary tasks loading")
                    else:
                        raise ValueError(f"Missing keys in model: {incompat_keys.missing_keys}")
            else:
                self.stateful.sources[name].load_state_dict(sub_state)

    def load_checkpoints(self, load_best: bool = False, ignore_state_keys: tp.List[str] = []) -> tp.Optional[dict]:
        """Load last checkpoint or the one specified in continue_from."""
        load_from_path: tp.Optional[Path] = None
        checkpoint_source: tp.Optional[checkpoint.CheckpointSource] = None

        if load_best:
            self.logger.info("Trying to load state_dict from best state.")

        state: tp.Optional[dict] = None
        rank0_checkpoint_path = self.checkpoint_path(use_fsdp=False)
        current_checkpoint_path = self.checkpoint_path()
        if rank0_checkpoint_path.exists():
            self.logger.info(f"Loading existing checkpoint: {current_checkpoint_path}")
            load_from_path = current_checkpoint_path
            checkpoint.check_sharded_checkpoint(current_checkpoint_path, rank0_checkpoint_path)
            checkpoint_source = checkpoint.CheckpointSource.CURRENT_XP
        elif self.cfg.continue_from:
            self.logger.info(f"Continuing from provided checkpoint: {self.cfg.continue_from}")
            if Path(self.cfg.continue_from).exists():
                load_from_path = Path(self.cfg.continue_from)
            elif self.cfg.continue_from.startswith('PAST'):
                self.logger.info(f"Resolving continue_from from HuggingFace: {self.cfg.continue_from}")
                load_from_path = get_model_cp_from_huggingface(self.cfg.continue_from)
            if load_from_path is None:
                self.logger.error('Could not resolve continue_from checkpoint %s', self.cfg.continue_from)
                raise RuntimeError(f'Could not resolve continue_from checkpoint {self.cfg.continue_from}')
            checkpoint_source = checkpoint.CheckpointSource.OTHER

        if load_from_path is not None:
            state = checkpoint.load_checkpoint(load_from_path)

        if checkpoint_source is not None and checkpoint_source != checkpoint.CheckpointSource.CURRENT_XP:
            assert state is not None
            self.logger.info("Loading best state from external checkpoint.")
            load_best = True
            state = {k: state[k] for k in self._continue_best_source_keys if k in state}

        if state is not None:
            if load_best:
                self.logger.info("Ignoring keys when loading best %r", ignore_state_keys)
                for key in set(ignore_state_keys):
                    state.pop(key, None)
                assert 'best_state' in state, "No 'best_state' in checkpoint"
                self.logger.info("Loading best_state from checkpoint.")
                state = state['best_state']
            self.load_state_dict(state)

            if load_best:
                self.logger.info("Swapping to best and reinitializing EMA.")
                for name in self.best_state.states.keys():
                    source = self._get_state_source(name)
                    self.best_state.update(name, source)
                if self.ema is not None:
                    self.initialize_ema()

        return state

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        if self.model.has_axiliary_task:
            return 'semantic_error'
        return 'sisnr'

    # ... your existing run_step, generate, etc. unchanged ...
    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        x, targets = batch
        x = x.to(self.device)
        y = x.clone()

        if self.is_training:
            use_mask = True
        else:
            use_mask = False
        qres, auxiliary_tasks_loss, auxiliary_tasks_metrics = self.model(x, targets=targets, use_mask=use_mask)
        assert isinstance(qres, quantization.QuantizedResult)
        valid_mask = self.calc_valid_mask(x)
        y_pred = qres.x
        y_pred[~valid_mask] = 0
        # Log bandwidth in kb/s
        metrics['bandwidth'] = qres.bandwidth.mean()

        if self.is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr'] * 1000
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss = adversary.train_adv(y_pred, y)
                    d_losses[f'd_{adv_name}'] = disc_loss
                metrics['d_loss'] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses: dict = {}
        other_losses: dict = {}

        # penalty from quantization
        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty  # penalty term from the quantizer

        # adversarial losses
        for adv_name, adversary in self.adv_losses.items():
            adv_loss, feat_loss = adversary(y_pred, y)
            balanced_losses[f'adv_{adv_name}'] = adv_loss
            balanced_losses[f'feat_{adv_name}'] = feat_loss

        # auxiliary losses
        for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

        # axiliary tasks losses
        if self.model.has_axiliary_task:
            other_losses['auxiliary_tasks_loss'] = auxiliary_tasks_loss
        if auxiliary_tasks_metrics:
            metrics.update(auxiliary_tasks_metrics)

        # weighted losses
        metrics.update(balanced_losses)
        metrics.update(other_losses)
        metrics.update(qres.metrics)

        if self.is_training:
            # backprop losses that are not handled by balancer
            other_loss = torch.tensor(0.0, device=self.device)
            if 'penalty' in other_losses:
                other_loss += other_losses['penalty']
            if 'auxiliary_tasks_loss' in other_losses:
                other_loss += other_losses['auxiliary_tasks_loss']
            if other_loss.requires_grad:
                other_loss.backward(retain_graph=True)
                ratio1 = sum(p.grad.data.norm(p=2).pow(2) for p in self.model.parameters() if p.grad is not None)
                assert isinstance(ratio1, torch.Tensor)
                metrics['ratio1'] = ratio1.sqrt()

            # balancer losses backward, returns effective training loss
            # with effective weights at the current batch.
            metrics['g_loss'] = self.balancer.backward(balanced_losses, y_pred)
            # add metrics corresponding to weight ratios
            metrics.update(self.balancer.metrics)
            ratio2 = sum(p.grad.data.norm(p=2).pow(2) for p in self.model.parameters() if p.grad is not None)
            assert isinstance(ratio2, torch.Tensor)
            metrics['ratio2'] = ratio2.sqrt()

            # optim
            flashy.distrib.sync_model(self.model)
            if self.cfg.optim.max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_norm)
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # informative losses only
        info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(y_pred, y)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

        # aggregated GAN losses: this is useful to report adv and feat across different adversarial loss setups
        adv_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('adv')]
        if len(adv_losses) > 0:
            metrics['adv'] = torch.sum(torch.stack(adv_losses))
        feat_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('feat')]
        if len(feat_losses) > 0:
            metrics['feat'] = torch.sum(torch.stack(feat_losses))

        return metrics

    def evaluate(self):
        """Evaluate stage. Runs audio reconstruction evaluation."""
        self.model.eval()
        loader = self.dataloaders['evaluate']
        total_batches = len(loader)
        pbar = self.log_progress(f'{self.current_stage} inference', loader,
                                 total=total_batches, updates=self.log_updates)
        
        # create an averager (callable) to accumulate per‐batch metrics
        average = flashy.averager()

        pendings = []
        ctx = multiprocessing.get_context('spawn')
        with get_pool_executor(self.cfg.evaluate.num_workers, mp_context=ctx) as pool:
            for data in pbar:
                batch = data[0]
                ssl_labels = data[-1]
                x = batch.to(self.device)
                with torch.no_grad():
                    qres = self.model(x, ssl_labels, use_mask=False)[0]

                y_pred = qres.x.cpu()
                y = batch.cpu()
                pendings.append(
                    pool.submit(evaluate_audio_with_all_metrics, y_pred, y, self.cfg)
                )

            metrics = None
            # collect and accumulate
            for future in self.log_progress(f'{self.current_stage} metrics',
                                            pendings, total=total_batches, updates=self.log_updates):
                batch_metrics = future.result()
                # average(batch_metrics)
                metrics = average(batch_metrics)

        # *** HERE’S THE FIX: call the averager to get your aggregated metrics ***
        # local_metrics = average()  
        # then average across all ranks
        # metrics = flashy.distrib.average_metrics(local_metrics, total_batches)
        metrics = flashy.distrib.average_metrics(metrics, total_batches)
        return metrics

    @staticmethod
    def calc_valid_mask(x):
        valid_mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
        T_audio = x.shape[-1]
        last_non_zero_audio_index = T_audio - (torch.flip(x, [-1]) != 0).float().argmax(dim=-1)
        padding_cond = last_non_zero_audio_index.unsqueeze(1) < torch.arange(end=T_audio, device=x.device).unsqueeze(0)
        valid_mask[padding_cond] = 0
        return valid_mask

    def generate(self):
        """Generate stage."""
        self.model.eval()
        sample_manager = SampleManager(self.xp, map_reference_to_sample_id=True)
        generate_stage_name = str(self.current_stage)

        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        for batch in lp:
            reference = batch[0]
            ssl_labels = batch[-1]
            reference = reference.to(self.device)
            with torch.no_grad():
                qres = self.model(reference, ssl_labels, use_mask=False)[0]
            assert isinstance(qres, quantization.QuantizedResult)

            reference = reference.cpu()
            estimate = qres.x.cpu()
            samples = sample_manager.add_samples(estimate, self.epoch, ground_truth_wavs=reference)
            self._log_samples(samples)

        flashy.distrib.barrier()

    @rank_zero_only
    def _log_samples(self, samples: tp.List[tp.Dict[str, tp.Any]]):
        if wandb is None or not self.cfg.logging.log_wandb:
            return
        table = wandb.Table(columns=['estimate', 'reference', 'epoch'])
        for sample in samples:
            table.add_data(wandb.Audio(sample.path), wandb.Audio(sample.reference.path), self.epoch)
        wandb.log({str(self.current_stage): table}, step=self.epoch)

def evaluate_audio_with_all_metrics(y_pred: torch.Tensor, y: torch.Tensor, cfg):
    """Evaluate audio with reconstruction metrics and additional quality metrics."""
    # Base reconstruction metrics
    metrics = evaluate_audio_reconstruction(y_pred, y, cfg)

    # audio quality metrics
    sample_rate = cfg.sample_rate if hasattr(cfg, 'sample_rate') else 24000
    pesq_scores, stoi_scores, utmos_scores = [], [], []

    # get (or reuse) our UTMOS predictors
    utmos_predictor_dict, utmos_predictor_fs = _get_utmos_predictors(cfg)

    batch_size = y_pred.shape[0]
    for i in range(batch_size):
        pred_audio = y_pred[i].squeeze().numpy()
        ref_audio = y[i].squeeze().numpy()

        # PESQ
        if pesq is not None:
            try:
                pesq_scores.append(pesq(sample_rate, ref_audio, pred_audio, 'wb'))
            except Exception as e:
                logger.warning(f"Failed to compute PESQ: {e}")

        # STOI
        if stoi is not None:
            try:
                stoi_scores.append(stoi(ref_audio, pred_audio, sample_rate, extended=False))
            except Exception as e:
                logger.warning(f"Failed to compute STOI: {e}")

        # UTMOS
        if utmos_predictor_dict is not None and pseudo_mos_metric is not None:
            try:
                scores = pseudo_mos_metric(
                    pred_audio,
                    fs=sample_rate,
                    predictor_dict=utmos_predictor_dict,
                    predictor_fs=utmos_predictor_fs,
                    use_gpu=torch.cuda.is_available()
                )
                if 'utmos' in scores:
                    utmos_scores.append(scores['utmos'])
            except Exception as e:
                logger.warning(f"Failed to compute UTMOS: {e}")

    # average and add to metrics
    if pesq_scores:
        metrics['pesq'] = torch.tensor(sum(pesq_scores) / len(pesq_scores), device='cpu')
    if stoi_scores:
        metrics['stoi'] = torch.tensor(sum(stoi_scores) / len(stoi_scores), device='cpu')
    if utmos_scores:
        metrics['utmos'] = torch.tensor(sum(utmos_scores) / len(utmos_scores), device='cpu')

    return metrics