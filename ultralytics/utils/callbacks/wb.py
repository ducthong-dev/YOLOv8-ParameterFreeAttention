# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
import re

from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers


def _wandb_disabled():
    return os.getenv('WANDB_DISABLED', '').lower() in {'1', 'true', 'yes', 'on'}


def _project_name(project):
    project = str(project or 'YOLOv8')
    project = re.sub(r'[/\\,#?%:]+', '-', project).strip('-')
    return project or 'YOLOv8'


try:
    import wandb as wb

    assert hasattr(wb, '__version__')
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['wandb'] is True  # verify integration is enabled
    assert not _wandb_disabled()  # allow Colab/local scripts to disable W&B without editing settings
except (ImportError, AssertionError):
    wb = None

_processed_plots = {}


def _log_plots(plots, step):
    for name, params in plots.items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """Initiate and start project if module is present."""
    wb.run or wb.init(project=_project_name(trainer.args.project), name=trainer.args.name, config=vars(trainer.args))


def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact at end of training."""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=['best'])


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if wb else {}
