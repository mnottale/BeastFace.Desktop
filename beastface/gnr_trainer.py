"""Drives GANs'n Roses (GNR) training as a subprocess.

Saves the run's hyperparameters as `params.json` in the experiment dir.
On resume the existing dir is reused and `--ckpt=<dir>/checkpoint/ck.pt`
is appended so train.py picks up where it left off.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from . import paths
from .jobs import Job

GNR_DIR = Path(paths.LIB) / 'GNR'

PARAMS_FILE = 'params.json'

# Field name -> (CLI flag, type, default). The order is preserved for the
# GUI so the form rows match the spec's grouping.
HPARAMS = [
    ('iter',              '--iter',              int,   300000),
    ('batch',             '--batch',             int,   4),
    ('size',              '--size',              int,   256),
    ('lr',                '--lr',                float, 2e-3),
    ('num_down',          '--num_down',          int,   3),
    ('latent_dim',        '--latent_dim',        int,   8),
    ('n_res',             '--n_res',             int,   1),
    ('lr_mlp',            '--lr_mlp',            float, 0.01),
    ('mixing',            '--mixing',            float, 0.9),
    ('r1',                '--r1',                float, 10.0),
    ('lambda_cycle',      '--lambda_cycle',      int,   1),
    ('d_reg_every',       '--d_reg_every',       int,   16),
    ('g_reg_every',       '--g_reg_every',       int,   4),
    ('path_regularize',   '--path_regularize',   float, 2.0),
    ('path_batch_shrink', '--path_batch_shrink', int,   2),
    ('n_sample',          '--n_sample',          int,   64),
]


def _runs_root() -> Path:
    d = Path(paths.ROOT) / 'experiments' / 'gnr-runs'
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


def _coerce(name: str, value, default):
    typ = type(default)
    try:
        if typ is bool:
            return bool(value)
        return typ(value)
    except Exception:
        return default


def prepare_run(
    *,
    dataset_path: str | Path,
    experiment_dir: Path | None = None,
    params: dict,
    resume: bool = False,
) -> tuple[Job, Path]:
    """Lay out the experiment dir, persist params, and build the Job."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not (dataset_path / 'trainA').is_dir() or not (dataset_path / 'trainB').is_dir():
        raise FileNotFoundError(
            f'Dataset directory must contain trainA/ and trainB/: {dataset_path}'
        )

    if experiment_dir is None:
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        experiment_dir = _runs_root() / f'gnr-{ts}'
    experiment_dir = Path(experiment_dir).resolve()

    if resume:
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f'Resume dir does not exist: {experiment_dir}')

    # Coerce params with the canonical defaults so the dict is always typed.
    coerced = {
        name: _coerce(name, params.get(name, default), default)
        for name, _, _, default in HPARAMS
    }

    save_params(experiment_dir, {'dataset_path': str(dataset_path), **coerced})

    parent = experiment_dir.parent
    name = experiment_dir.name

    cmd = [sys.executable, str(GNR_DIR / 'train.py'),
           '--name', name, '--d_path', str(dataset_path)]
    for field, flag, _typ, default in HPARAMS:
        cmd += [flag, str(coerced[field])]

    if resume:
        ckpt = experiment_dir / 'checkpoint' / 'ck.pt'
        if ckpt.exists():
            cmd += ['--ckpt', str(ckpt)]
        else:
            raise FileNotFoundError(
                f'Resume requested but no checkpoint at {ckpt}'
            )

    # cwd=parent so GNR's `./{name}` save_path resolves to experiment_dir.
    # PYTHONPATH lets `from dataset import ImageFolder` find GNR's module.
    job = Job(cmd, cwd=parent, env={'PYTHONPATH': str(GNR_DIR)})
    return job, experiment_dir
