"""Drives a kohya gen_img.py many-shots generation as a subprocess.

Renders N prompts from a JSON template and writes them to a per-line
prompt file with `--n` negative-prompt directives, then constructs the
`gen_img.py` command line. The actual run is launched via `Job`.
"""

from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path

from . import paths, prompt_template
from .jobs import Job


def _project_root() -> Path:
    return Path(paths.ROOT)


def _sd_scripts_dir() -> Path:
    # Reuses the locator from lora_trainer; duplicated here only to avoid an
    # import cycle when training_gui imports both modules.
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
        if (p / 'gen_img.py').exists():
            return p
    raise FileNotFoundError(
        'sd-scripts checkout not found. Expected at <project>/sd-scripts.'
    )


def prepare_run(
    *,
    base_model: str,
    lora_path: str,
    template_path: str,
    resolution: int = 512,
    n_images: int = 100,
    network_mul: float = 1.0,
    steps: int = 30,
    sampler: str = 'k_euler_a',
    seed: int | None = None,
    extra_args: list[str] | None = None,
) -> tuple[Job, Path]:
    """Build the prompts file and return (Job, output_dir)."""
    base_model = str(Path(base_model).expanduser().resolve())
    lora_path = str(Path(lora_path).expanduser().resolve())
    template = prompt_template.load_template(template_path)

    rng = random.Random(seed)
    sd_scripts = _sd_scripts_dir()

    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = _project_root() / 'experiments' / f'sd-many-{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = out_dir / 'prompts.txt'
    with open(prompts_path, 'w', encoding='utf-8') as fd:
        for _ in range(int(n_images)):
            pos, neg = prompt_template.render(template, rng)
            line = pos.strip()
            if neg.strip():
                line += f' --n {neg.strip()}'
            fd.write(line + '\n')

    # Run gen_img through a tiny shim that patches a function the local
    # kohya checkout's gen_img expects but train_util has since dropped.
    shim = Path(__file__).resolve().parent / '_kohya_gen_img_shim.py'
    cmd = [
        sys.executable,
        str(shim),
        '--ckpt', base_model,
        '--outdir', str(out_dir),
        '--from_file', str(prompts_path),
        '--W', str(resolution),
        '--H', str(resolution),
        '--steps', str(steps),
        '--sampler', sampler,
        '--batch_size', '1',
        '--xformers',
        '--network_module', 'networks.lora',
        '--network_weights', lora_path,
        '--network_mul', str(network_mul),
    ]
    if seed is not None:
        cmd += ['--seed', str(seed)]
    if extra_args:
        cmd += list(extra_args)

    job = Job(cmd, cwd=sd_scripts)
    return job, out_dir
