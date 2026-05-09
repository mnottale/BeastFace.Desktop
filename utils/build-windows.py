#!/usr/bin/env python3
"""Assemble a self-contained Windows zip of BeastFace.Desktop.

Pure-Linux build (no wine). Downloads Astral's python-build-standalone
Windows tarball (a portable Python with Tcl/Tk, pip, and the full stdlib),
pip-installs Windows wheels into its Lib/site-packages, fetches a static
ffmpeg.exe and freeglut.dll, copies the project source, and zips it.

Usage:
    utils/build-windows.py                  # default: cu128 CUDA build
    utils/build-windows.py --cpu            # CPU-only torch
    utils/build-windows.py --cuda cu126     # different CUDA tag
    utils/build-windows.py --keep           # leave dist/ unzipped

Resulting layout in the zip:
    run.bat
    ffmpeg.exe
    python\\python.exe
    python\\freeglut.dll
    python\\Lib\\site-packages\\...
    BeastFace.Desktop\\app.py
    BeastFace.Desktop\\beastface\\...
    BeastFace.Desktop\\libraries\\...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PY_VER_NODOTS = '311'
PY_PLAT = 'win_amd64'

# Astral's python-build-standalone publishes self-contained, portable Python
# tarballs that include Tcl/Tk and pip. The "install_only" variant is the
# ready-to-use one (the alternative is a build-environment tarball). The
# tarball root is "python/", which becomes our dist/python/ directly.
PYTHON_URL = (
    'https://github.com/astral-sh/python-build-standalone/releases/download/'
    '20260508/cpython-3.11.15%2B20260508-'
    'x86_64-pc-windows-msvc-install_only.tar.gz'
)

# gyan.dev essentials build of FFmpeg includes libtheora -> .ogv encoding works.
FFMPEG_URL = (
    'https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/'
    'ffmpeg-7.1.1-essentials_build.zip'
)

# conda-forge ships a Windows freeglut package as a plain bzip2 tarball
# (no extra deps to extract, unlike MSYS2's .pkg.tar.zst). The DLL is at
# Library/bin/freeglut.dll inside the archive. Override via --freeglut-url
# if conda-forge ever rebuilds and removes this specific build.
FREEGLUT_URL = (
    'https://conda.anaconda.org/conda-forge/win-64/'
    'freeglut-3.2.2-h0e60522_1.tar.bz2'
)

DEFAULT_CUDA = 'cu128'
TORCH_PACKAGES = ('torch', 'torchvision')

EXCLUDE_DIR_NAMES = {
    '.git', '__pycache__', '.pytest_cache', 'recordings',
    'dist', 'build', 'node_modules', '.idea', '.vscode', '.cache',
}
EXCLUDE_FILE_SUFFIXES = ('.pyc',)

CACHE_DIR = PROJECT_ROOT / '.cache' / 'build-windows'


# ---------------------------------------------------------------------- net

def _cache_name(url: str) -> str:
    from urllib.parse import unquote, urlparse
    name = unquote(urlparse(url).path.rsplit('/', 1)[-1])
    if not name:
        import hashlib
        name = hashlib.sha256(url.encode()).hexdigest()
    return name


def cached_download(url: str, *, force: bool = False) -> Path:
    """Download `url` into CACHE_DIR/<filename>, return the cached Path.

    Skips the GET if the file already exists and is non-empty. Atomic via
    a .partial file so an interrupted run doesn't poison the cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / _cache_name(url)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f'  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)')
        return dest
    print(f'  GET {url}')
    tmp = dest.with_suffix(dest.suffix + '.partial')
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    return dest


def extract_tar_subtree(tar_path: Path, prefix: str, target_dir: Path) -> None:
    """Extract members of tar_path whose name starts with `prefix` into
    target_dir, with `prefix` itself stripped from the path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:*') as tf:
        for member in tf.getmembers():
            if not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix):]
            if not rel:
                continue
            out = target_dir / rel
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst)


def extract_zip_one(zip_path: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(suffix):
                with zf.open(name) as src:
                    return src.read()
    raise RuntimeError(f'{suffix} not found in {zip_path}')


def extract_tar_one(tar_path: Path, suffix: str) -> bytes:
    with tarfile.open(tar_path, 'r:*') as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith(suffix):
                f = tf.extractfile(member)
                if f is not None:
                    return f.read()
    raise RuntimeError(f'{suffix} not found in {tar_path}')


# ----------------------------------------------------------------------- pip

def pip_install(site_packages: Path, packages: list[str], *,
                index_url: str | None = None,
                extra_index_url: str | None = None) -> None:
    # Specify multiple --platform and --abi values so the tag generator
    # accepts both the binary CPython wheels (cp311-cp311-win_amd64) and
    # the universal pure-Python wheels (py3-none-any) that come in as
    # transitive dependencies (typing-extensions, packaging, etc.).
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        '--cache-dir', str(CACHE_DIR / 'pip'),
        '--target', str(site_packages),
        '--platform', PY_PLAT,
        '--platform', 'any',
        '--only-binary=:all:',
        '--python-version', PY_VER_NODOTS,
        '--implementation', 'cp',
        '--abi', f'cp{PY_VER_NODOTS}',
        '--abi', 'abi3',
        '--abi', 'none',
        '--upgrade',
    ]
    if index_url:
        cmd += ['--index-url', index_url]
    if extra_index_url:
        cmd += ['--extra-index-url', extra_index_url]
    cmd += list(packages)
    print('  $', ' '.join(cmd))
    subprocess.check_call(cmd)


def parse_requirements(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.split('#', 1)[0].strip()
        if s:
            out.append(s)
    return out


# -------------------------------------------------------------------- copy

def _ignore(directory: str, names: list[str]) -> list[str]:
    skip = []
    for n in names:
        full = Path(directory) / n
        if n in EXCLUDE_DIR_NAMES:
            skip.append(n)
        elif full.is_file() and n.endswith(EXCLUDE_FILE_SUFFIXES):
            skip.append(n)
    return skip


def copy_project(out_root: Path) -> None:
    target = out_root / 'BeastFace.Desktop'
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in PROJECT_ROOT.iterdir():
        if entry.name in EXCLUDE_DIR_NAMES:
            continue
        dest = target / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, ignore=_ignore)
        else:
            shutil.copy2(entry, dest)


def write_run_bat(out_root: Path) -> None:
    bat = out_root / 'run.bat'
    bat.write_bytes(
        b'@echo off\r\n'
        b'setlocal\r\n'
        b'set HERE=%~dp0\r\n'
        b'set PATH=%HERE%;%HERE%python;%PATH%\r\n'
        b'cd /d "%HERE%BeastFace.Desktop"\r\n'
        b'"%HERE%python\\python.exe" app.py %*\r\n'
        b'if errorlevel 1 pause\r\n'
    )


def fix_permissions(out_root: Path) -> None:
    """Add +x to every file and directory so Wine (and any unzipper that
    honors POSIX modes) can execute the .exe / .dll bits. Native Windows
    ignores Unix mode bits, so this is harmless there."""
    out_root.chmod(out_root.stat().st_mode | 0o111)
    for entry in out_root.rglob('*'):
        try:
            entry.chmod(entry.stat().st_mode | 0o111)
        except OSError:
            pass


def patch_pth(py_dir: Path) -> None:
    """Make sure Lib/site-packages is on sys.path even with the embed-style
    `pythonNNN._pth` if NuGet ever ships with one."""
    for pth in py_dir.glob('python*._pth'):
        text = pth.read_text()
        if 'Lib\\site-packages' in text and 'import site' in text and not text.startswith('#'):
            continue
        if 'import site' not in text:
            text = text.rstrip() + '\nimport site\n'
        if 'Lib\\site-packages' not in text:
            text = 'Lib\\site-packages\n' + text
        pth.write_text(text)


# -------------------------------------------------------------------- zip

def zip_dir(src: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    base = src.parent
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED,
                         allowZip64=True, compresslevel=6) as zf:
        for f in src.rglob('*'):
            if f.is_file():
                zf.write(f, f.relative_to(base))


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default='dist',
                    help='Build directory (will be (re)created)')
    ap.add_argument('--zip', default='BeastFace.Desktop-windows.zip',
                    help='Output zip path; ignored with --keep')
    ap.add_argument('--cuda', default=DEFAULT_CUDA,
                    help=f'PyTorch CUDA tag (default {DEFAULT_CUDA})')
    ap.add_argument('--cpu', action='store_true', help='CPU-only torch')
    ap.add_argument('--keep', action='store_true', help="Don't zip; leave the build dir")
    ap.add_argument('--no-cache', action='store_true',
                    help='Force re-downloading Python/ffmpeg/freeglut and bypass pip cache')
    args = ap.parse_args()
    if args.no_cache:
        # Wipe the cache so pip and our cached_download both start fresh.
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)

    out_root = Path(args.out).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    work_dir = Path(tempfile.mkdtemp(prefix='bf-build-'))
    try:
        print('[1/7] Downloading Windows Python (python-build-standalone)')
        py_tar = cached_download(PYTHON_URL, force=args.no_cache)

        py_dir = out_root / 'python'
        py_dir.mkdir()
        # python-build-standalone tarball layout: python/<full tree>
        extract_tar_subtree(py_tar, 'python/', py_dir)
        patch_pth(py_dir)

        site_packages = py_dir / 'Lib' / 'site-packages'
        site_packages.mkdir(parents=True, exist_ok=True)

        # Split requirements into torch-related vs the rest.
        reqs = parse_requirements(PROJECT_ROOT / 'requirements.txt')

        def pkg_name(spec: str) -> str:
            for sep in ('==', '>=', '<=', '~=', '>', '<'):
                if sep in spec:
                    return spec.split(sep, 1)[0].strip().lower()
            return spec.strip().lower()

        torch_specs = [r for r in reqs if pkg_name(r) in TORCH_PACKAGES]
        other_specs = [r for r in reqs if pkg_name(r) not in TORCH_PACKAGES]
        if not torch_specs:
            torch_specs = list(TORCH_PACKAGES)

        print('[2/7] Installing torch / torchvision')
        # Use --extra-index-url (not --index-url) so PyPI stays primary.
        # That way torch's pure-Python deps (typing-extensions, sympy, ...)
        # come from PyPI rather than pytorch.org's mirror, which sometimes
        # serves wheels with inconsistent metadata that pip rejects.
        # The CUDA wheel still wins via PEP 440 ordering: 2.x.y+cu128 > 2.x.y.
        if args.cpu:
            pip_install(site_packages, torch_specs,
                        extra_index_url='https://download.pytorch.org/whl/cpu')
        else:
            pip_install(site_packages, torch_specs,
                        extra_index_url=f'https://download.pytorch.org/whl/{args.cuda}')

        print('[3/7] Installing remaining wheels')
        if other_specs:
            pip_install(site_packages, other_specs)

        print('[4/7] Fetching ffmpeg.exe')
        ffmpeg_zip = cached_download(FFMPEG_URL, force=args.no_cache)
        (out_root / 'ffmpeg.exe').write_bytes(extract_zip_one(ffmpeg_zip, '/bin/ffmpeg.exe'))

        print('[5/7] Fetching freeglut.dll')
        freeglut_archive = cached_download(FREEGLUT_URL, force=args.no_cache)
        (py_dir / 'freeglut.dll').write_bytes(
            extract_tar_one(freeglut_archive, 'Library/bin/freeglut.dll')
        )

        print('[6/7] Copying project source')
        copy_project(out_root)
        write_run_bat(out_root)
        fix_permissions(out_root)

        if args.keep:
            print(f'[7/7] Done. Build at: {out_root}')
            return

        print('[7/7] Zipping')
        zip_path = Path(args.zip).resolve()
        zip_dir(out_root, zip_path)
        size_mb = zip_path.stat().st_size / 1e6
        print(f'      -> {zip_path}  ({size_mb:.1f} MB)')
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
