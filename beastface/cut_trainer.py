"""Drives Contrastive Unpaired Translation (CUT) training as a subprocess.

Saves a `params.json` in the experiment dir so a later resume can prefill
the GUI. On resume passes `--continue_train` and an auto-detected
`--epoch_count` derived from `<n>_net_G.pth` files in the checkpoint dir.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

from . import paths
from .jobs import Job

CUT_DIR = Path(paths.LIB) / 'CUT'

PARAMS_FILE = 'params.json'

# (form name, CLI flag, kind, default). kind is one of:
#   - python type (int/float/str)
#   - "flag": store_true CLI flag controlled by a bool
HPARAMS = [
    ('netG',           '--netG',           str,   'resnet_9blocks'),
    ('netD',           '--netD',           str,   'basic'),
    ('ngf',            '--ngf',            int,   64),
    ('ndf',            '--ndf',            int,   64),
    ('normG',          '--normG',          str,   'instance'),
    ('normD',          '--normD',          str,   'instance'),
    ('direction',      '--direction',      str,   'AtoB'),
    ('preprocess',     '--preprocess',     str,   'resize_and_crop'),
    ('load_size',      '--load_size',      int,   286),
    ('crop_size',      '--crop_size',      int,   256),
    ('batch_size',     '--batch_size',     int,   1),
    ('n_epochs',       '--n_epochs',       int,   200),
    ('n_epochs_decay', '--n_epochs_decay', int,   200),
    ('lr',             '--lr',             float, 0.0002),
    ('beta1',          '--beta1',          float, 0.5),
    ('beta2',          '--beta2',          float, 0.999),
    ('gan_mode',       '--gan_mode',       str,   'lsgan'),
    ('lr_policy',      '--lr_policy',      str,   'linear'),
    ('lr_decay_iters', '--lr_decay_iters', int,   50),
    ('lambda_GAN',     '--lambda_GAN',     float, 1.0),
    ('lambda_NCE',     '--lambda_NCE',     float, 1.0),
    ('nce_layers',     '--nce_layers',     str,   '0,4,8,12,16'),
    ('num_patches',    '--num_patches',    int,   256),
    ('save_latest_freq', '--save_latest_freq', int, 5000),
    ('save_epoch_freq',  '--save_epoch_freq',  int, 5),
    ('print_freq',     '--print_freq',     int,   100),
    ('display_id',     '--display_id',     int,   0),
    ('no_flip',        '--no_flip',        'flag', False),
    ('no_antialias',   '--no_antialias',   'flag', False),
    ('no_antialias_up','--no_antialias_up','flag', False),
]

NETG_CHOICES = ('resnet_9blocks', 'resnet_6blocks', 'resnet_4blocks',
                'unet_128', 'unet_256')
NETD_CHOICES = ('basic', 'n_layers', 'pixel')
NORM_CHOICES = ('instance', 'batch', 'none')
DIRECTION_CHOICES = ('AtoB', 'BtoA')
PREPROCESS_CHOICES = ('resize_and_crop', 'crop', 'scale_width',
                      'scale_width_and_crop', 'none')
GAN_MODE_CHOICES = ('lsgan', 'vanilla', 'wgangp')
LR_POLICY_CHOICES = ('linear', 'step', 'plateau', 'cosine')


def _runs_root() -> Path:
    d = Path(paths.ROOT) / 'experiments' / 'cut-runs'
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_params() -> dict:
    return {name: default for name, _, _, default in HPARAMS}


def load_params(experiment_dir: Path) -> dict | None:
    f = Path(experiment_dir) / PARAMS_FILE
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def save_params(experiment_dir: Path, params: dict) -> None:
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    (Path(experiment_dir) / PARAMS_FILE).write_text(json.dumps(params, indent=2))


def _coerce(value, kind, default):
    if kind == 'flag':
        return bool(value)
    try:
        return kind(value)
    except Exception:
        return default


def _detect_epoch_count(experiment_dir: Path) -> int:
    """Find the highest numbered checkpoint and return that + 1.

    CUT saves `<epoch>_net_G.pth` per save_epoch_freq, plus `latest_net_G.pth`.
    Returns 1 if only `latest_net_*` is present (no numbered checkpoint).
    """
    pattern = re.compile(r'^(\d+)_net_G\.pth$')
    max_epoch = 0
    if experiment_dir.is_dir():
        for f in experiment_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_epoch = max(max_epoch, int(m.group(1)))
    return max_epoch + 1 if max_epoch > 0 else 1


def prepare_run(
    *,
    dataset_path: str | Path,
    experiment_dir: Path | None = None,
    params: dict,
    resume: bool = False,
) -> tuple[Job, Path]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not (dataset_path / 'trainA').is_dir() or not (dataset_path / 'trainB').is_dir():
        raise FileNotFoundError(
            f'Dataset directory must contain trainA/ and trainB/: {dataset_path}'
        )

    if experiment_dir is None:
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        experiment_dir = _runs_root() / f'cut-{ts}'
    experiment_dir = Path(experiment_dir).resolve()

    if resume and not experiment_dir.is_dir():
        raise FileNotFoundError(f'Resume dir does not exist: {experiment_dir}')

    coerced = {
        name: _coerce(params.get(name, default), kind, default)
        for name, _, kind, default in HPARAMS
    }
    save_params(experiment_dir, {'dataset_path': str(dataset_path), **coerced})

    parent = experiment_dir.parent
    name = experiment_dir.name

    cmd = [sys.executable, str(CUT_DIR / 'train.py'),
           '--dataroot', str(dataset_path),
           '--name', name,
           '--checkpoints_dir', str(parent),
           '--model', 'cut']
    for field, flag, kind, default in HPARAMS:
        value = coerced[field]
        if kind == 'flag':
            if value:
                cmd.append(flag)
        else:
            cmd += [flag, str(value)]

    if resume:
        cmd.append('--continue_train')
        epoch_count = _detect_epoch_count(experiment_dir)
        cmd += ['--epoch_count', str(epoch_count)]
        cmd += ['--epoch', 'latest']

    job = Job(cmd, cwd=CUT_DIR)
    return job, experiment_dir
