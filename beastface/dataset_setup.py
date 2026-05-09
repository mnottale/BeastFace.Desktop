"""Build the trainA/trainB/testA/testB layout that GNR and CUT expect.

Domain A is a fixed directory (`<project>/dataset-a` by default; override
with $BEASTFACE_DATASET_A). Domain B comes from the user. trainA/trainB
are symlinked to those source dirs (or junctioned/copied as a fallback);
testA/testB hold 8 random images each.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

from . import paths

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

DATASET_A_DIR = Path(
    os.environ.get('BEASTFACE_DATASET_A')
    or (Path(paths.ROOT) / 'dataset-a')
)


def _link_dir(src: Path, dst: Path) -> None:
    """Make `dst` point at `src`. Symlink, junction, or copy — whichever works."""
    src = Path(src).resolve()
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == 'nt':
        # Directory junction: same effect as a dir symlink, no admin required.
        try:
            subprocess.check_call(
                ['cmd', '/c', 'mklink', '/J', str(dst), str(src)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass
    shutil.copytree(src, dst)


def _sample_images(src: Path, dst: Path, n: int, rng: random.Random) -> None:
    if dst.exists() and any(dst.iterdir()):
        return  # already populated, leave the existing test set alone
    dst.mkdir(parents=True, exist_ok=True)
    candidates = [
        f for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    if not candidates:
        raise FileNotFoundError(f'No images in {src}')
    pick = rng.sample(candidates, min(n, len(candidates)))
    for img in pick:
        shutil.copy2(img, dst / img.name)


def assemble(
    domain_b: str | os.PathLike,
    target: str | os.PathLike,
    *,
    n_test: int = 8,
    rng_seed: int | None = None,
) -> Path:
    """Lay out trainA/trainB/testA/testB under `target` and return it."""
    domain_b = Path(domain_b).expanduser().resolve()
    if not domain_b.is_dir():
        raise FileNotFoundError(f'Domain B folder not found: {domain_b}')
    if not DATASET_A_DIR.is_dir():
        raise FileNotFoundError(
            f'Domain A dataset not found at {DATASET_A_DIR}. '
            'Set $BEASTFACE_DATASET_A to override.'
        )
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    _link_dir(DATASET_A_DIR, target / 'trainA')
    _link_dir(domain_b, target / 'trainB')
    rng = random.Random(rng_seed)
    _sample_images(DATASET_A_DIR, target / 'testA', n=n_test, rng=rng)
    _sample_images(domain_b, target / 'testB', n=n_test, rng=rng)
    return target
