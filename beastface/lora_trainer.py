"""Drives a kohya-ss/sd-scripts LoRA training run as a subprocess.

`prepare_run()` lays out a per-job working directory containing the
`<repeats>_<trigger> <class>` images dir and a config.toml, then returns
a `Job` ready to start. The caller (GUI) controls start/stop/streaming.
"""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

from . import paths
from .jobs import Job


def _project_root() -> Path:
    return Path(paths.ROOT)


def _experiments_dir() -> Path:
    d = _project_root() / 'experiments' / 'lora'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sd_scripts_dir() -> Path:
    """Resolve the sd-scripts checkout (submodule or env override).

    Looks first at $BEASTFACE_SDSCRIPTS, then <project>/sd-scripts, then
    libraries/sd-scripts. Raises if none exists so the GUI can show a
    helpful error.
    """
    import os
    env = os.environ.get('BEASTFACE_SDSCRIPTS')
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        _project_root() / 'sd-scripts',
        _project_root() / 'libraries' / 'sd-scripts',
    ]
    for p in candidates:
        if (p / 'train_network.py').exists():
            return p
    raise FileNotFoundError(
        'sd-scripts checkout not found. Expected at <project>/sd-scripts '
        'or <project>/libraries/sd-scripts (kohya-ss/sd-scripts). '
        'Set BEASTFACE_SDSCRIPTS to override.'
    )


def _sanitize(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s).strip('_') or 'x'


def prepare_run(
    *,
    base_model: str,
    images_dir: str,
    class_word: str,
    trigger: str,
    repeats: int = 20,
    resolution: int = 512,
    max_train_steps: int = 2000,
    network_dim: int = 16,
    network_alpha: int = 8,
    learning_rate: float = 1e-4,
    train_batch_size: int = 1,
    mixed_precision: str = 'fp16',
    extra_args: list[str] | None = None,
) -> tuple[Job, Path]:
    """Build the dataset dir + config.toml and return (Job, output_path).

    The Job is configured but not started; caller calls .start().
    """
    base_model = str(Path(base_model).expanduser().resolve())
    images_src = Path(images_dir).expanduser().resolve()
    if not images_src.is_dir():
        raise FileNotFoundError(f'Images dir does not exist: {images_src}')

    sd_scripts = _sd_scripts_dir()
    out_dir = _experiments_dir()

    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    base_stem = Path(base_model).stem
    name = f'{trigger}-{_sanitize(base_stem)}-{_sanitize(class_word)}-{ts}'
    work_dir = _project_root() / 'experiments' / 'lora-runs' / name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Dataset directory: kohya parses subfolder names as "<repeats>_<concept>"
    concept_dir = work_dir / 'dataset' / f'{repeats}_{trigger} {class_word}'
    concept_dir.mkdir(parents=True, exist_ok=True)
    image_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    n_images = 0
    for entry in images_src.iterdir():
        if entry.is_file() and entry.suffix.lower() in image_exts:
            shutil.copy2(entry, concept_dir / entry.name)
            n_images += 1
    if n_images == 0:
        raise FileNotFoundError(f'No images found under {images_src}')

    # Config TOML for train_network.py
    output_name = name
    config_path = work_dir / 'config.toml'
    train_data_dir = (work_dir / 'dataset').as_posix()
    config_path.write_text(
        '[model_arguments]\n'
        f'pretrained_model_name_or_path = "{base_model}"\n'
        '\n'
        '[network_arguments]\n'
        'network_module = "networks.lora"\n'
        f'network_dim = {network_dim}\n'
        f'network_alpha = {network_alpha}\n'
        '\n'
        '[optimizer_arguments]\n'
        'optimizer_type = "AdamW8bit"\n'
        f'learning_rate = {learning_rate}\n'
        'lr_scheduler = "cosine_with_restarts"\n'
        'lr_warmup_steps = 0\n'
        '\n'
        '[dataset_arguments]\n'
        f'train_data_dir = "{train_data_dir}"\n'
        f'resolution = "{resolution},{resolution}"\n'
        f'train_batch_size = {train_batch_size}\n'
        'caption_extension = ".txt"\n'
        'shuffle_caption = false\n'
        '\n'
        '[training_arguments]\n'
        f'output_dir = "{out_dir.as_posix()}"\n'
        f'output_name = "{output_name}"\n'
        'save_model_as = "safetensors"\n'
        f'max_train_steps = {max_train_steps}\n'
        f'mixed_precision = "{mixed_precision}"\n'
        'gradient_checkpointing = true\n'
        'xformers = true\n'
        'cache_latents = true\n'
        'save_every_n_epochs = 1\n'
        'logging_dir = "logs"\n'
    )

    cmd = [
        sys.executable,
        str(sd_scripts / 'train_network.py'),
        f'--config_file={config_path}',
    ]
    if extra_args:
        cmd += list(extra_args)

    job = Job(cmd, cwd=sd_scripts)
    final = out_dir / f'{output_name}.safetensors'
    return job, final
