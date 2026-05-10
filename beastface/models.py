"""Model loaders for GNR and CUT checkpoints with hyperparameter inference."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from . import paths

paths.setup()


GNR_CHANNELS = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256,
                128: 128, 256: 64, 512: 32, 1024: 16}
# Reverse map for sizes >= 64 (unique mapping).
GNR_SIZE_FROM_CHANNELS = {GNR_CHANNELS[s]: s for s in (64, 128, 256, 512, 1024)}


def _strip_module_prefix(sd):
    if any(k.startswith('module.') for k in sd.keys()):
        return {k[len('module.'):]: v for k, v in sd.items()}
    return sd


def _stem_indices(sd, prefix):
    out = set()
    for k in sd.keys():
        if k.startswith(prefix):
            tail = k[len(prefix):]
            try:
                out.add(int(tail.split('.', 1)[0]))
            except ValueError:
                pass
    return out


def detect_gnr(state_dict):
    """Infer GNR Generator hyperparameters from a state_dict.

    Returns a dict or None if the dict doesn't look like a GNR generator.
    """
    sd = state_dict
    if 'mapping.0.weight' not in sd or 'encoder.stem.0.0.weight' not in sd:
        return None

    latent_dim = int(sd['mapping.0.weight'].shape[1])

    n_mlp = 0
    while f'mapping.{n_mlp}.weight' in sd:
        n_mlp += 1

    first_conv = sd['encoder.stem.0.0.weight']
    out_ch = int(first_conv.shape[0])
    size = GNR_SIZE_FROM_CHANNELS.get(out_ch, 256)

    stem_idxs = sorted(i for i in _stem_indices(sd, 'encoder.stem.') if i != 0)
    num_down = sum(
        1 for i in stem_idxs
        if any(k.startswith(f'encoder.stem.{i}.skip.') for k in sd.keys())
    )
    n_res = len(stem_idxs) - num_down

    return dict(
        size=size,
        latent_dim=latent_dim,
        n_mlp=n_mlp,
        num_down=num_down,
        n_res=max(n_res, 1),
        channel_multiplier=1,
    )


def detect_cut(state_dict):
    """Infer CUT ResnetGenerator hyperparameters. Returns None if it's not a
    plain CUT generator state dict."""
    sd = state_dict
    if 'mapping.0.weight' in sd:
        return None
    w1 = sd.get('model.1.weight')
    if w1 is None or w1.dim() != 4 or w1.shape[2] != 7:
        return None

    ngf = int(w1.shape[0])
    input_nc = int(w1.shape[1])

    last_conv_w = None
    last_idx = -1
    for k, v in sd.items():
        if (k.startswith('model.') and k.endswith('.weight')
                and v.dim() == 4 and v.shape[2] == 7):
            try:
                idx = int(k.split('.')[1])
            except ValueError:
                continue
            if idx > last_idx:
                last_idx = idx
                last_conv_w = v
    output_nc = int(last_conv_w.shape[0]) if last_conv_w is not None else 3

    rb_indices = set()
    for k in sd.keys():
        if '.conv_block.' in k and k.startswith('model.'):
            try:
                rb_indices.add(int(k.split('.')[1]))
            except ValueError:
                pass
    n_blocks = len(rb_indices)
    if n_blocks not in (4, 6, 9):
        # fall back to nearest standard
        n_blocks = min((4, 6, 9), key=lambda n: abs(n - n_blocks))

    has_bn = any(k.endswith('running_mean') for k in sd.keys())
    norm = 'batch' if has_bn else 'instance'

    # Antialias Downsample/Upsample modules register a 'filt' buffer.
    # Detect down/up antialias separately by checking if filt buffers occur
    # before or after the resnet block range.
    rb_min = min(rb_indices) if rb_indices else 0
    rb_max = max(rb_indices) if rb_indices else 0
    has_down_filt = False
    has_up_filt = False
    for k in sd.keys():
        if k.endswith('.filt') and '.conv_block.' not in k:
            try:
                idx = int(k.split('.')[1])
            except ValueError:
                continue
            if idx < rb_min:
                has_down_filt = True
            elif idx > rb_max:
                has_up_filt = True

    return dict(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        norm=norm,
        no_antialias=not has_down_filt,
        no_antialias_up=not has_up_filt,
    )


def _load_raw(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ('G_A2B_ema', 'G_A2B', 'state_dict'):
            if key in obj and isinstance(obj[key], dict):
                return obj, _strip_module_prefix(obj[key])
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, _strip_module_prefix(obj)
    raise ValueError(f'Unrecognized checkpoint format: {path}')


class _ModelBase:
    type: str = ''
    size: int = 256

    def roll(self):
        """Re-roll style. No-op for non-GNR models."""
        return None

    def to(self, device):
        self.device = device
        self.net.to(device)
        return self

    @staticmethod
    def _bgr_to_tensor(img_bgr, size):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != size or rgb.shape[1] != size:
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        return t.unsqueeze(0)

    @staticmethod
    def _tensor_to_bgr(t):
        x = t.detach().cpu()[0].permute(1, 2, 0).numpy()
        x = (x * 0.5 + 0.5)
        x = np.clip(x, 0, 1) * 255.0
        return cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2BGR)


class GNRModel(_ModelBase):
    type = 'GNR'

    @staticmethod
    def styles_path_for(model_path):
        from pathlib import Path
        return Path(model_path).with_suffix('.styles.json')

    def __init__(self, path, device='cpu'):
        self.path = path
        self.device = device
        full, sd = _load_raw(path, device)
        params = detect_gnr(sd)
        if params is None:
            raise ValueError(f'Not a GNR checkpoint: {path}')
        self.params = params
        self.size = params['size']
        self.latent_dim = params['latent_dim']

        from model import Generator  # GNR's model.py
        self.net = Generator(
            params['size'],
            params['num_down'],
            params['latent_dim'],
            params['n_mlp'],
            n_res=params['n_res'],
            channel_multiplier=params['channel_multiplier'],
            lr_mlp=0.01,
        )
        # Filter state dict keys to only those expected by the model.
        # (Some checkpoints carry additional helper buffers we don't need.)
        self.net.load_state_dict(sd, strict=False)
        self.net.eval()
        self.net.to(device)
        self.style = torch.randn(1, self.latent_dim, device=device)
        print(f'[GNR] loaded {path} on {next(self.net.parameters()).device}')

    def roll(self):
        self.style = torch.randn(1, self.latent_dim, device=self.device)
        return self.style

    def to(self, device):
        self.device = device
        self.net.to(device)
        self.style = self.style.to(device)
        return self

    # ---- saved-style management (persisted as <model>.styles.json) ----

    def styles_path(self):
        return GNRModel.styles_path_for(self.path)

    def list_styles(self):
        import json
        p = self.styles_path()
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text())
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def save_style(self, name):
        import json
        if not name or not name.strip():
            raise ValueError('Style name cannot be empty.')
        name = name.strip()
        existing = self.list_styles()
        existing[name] = self.style.detach().cpu().reshape(-1).tolist()
        p = self.styles_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(existing, indent=2))
        return name

    def load_style(self, name):
        styles = self.list_styles()
        if name not in styles:
            raise KeyError(f'Style not found: {name}')
        values = styles[name]
        if len(values) != self.latent_dim:
            raise ValueError(
                f'Saved style has dim {len(values)}, model expects {self.latent_dim}'
            )
        self.style = torch.tensor(values, dtype=torch.float32,
                                  device=self.device).view(1, self.latent_dim)
        return self.style

    def delete_style(self, name):
        import json
        styles = self.list_styles()
        if name not in styles:
            return False
        del styles[name]
        p = self.styles_path()
        if styles:
            p.write_text(json.dumps(styles, indent=2))
        else:
            try:
                p.unlink()
            except OSError:
                p.write_text('{}')
        return True

    def __call__(self, img_bgr):
        x = self._bgr_to_tensor(img_bgr, self.size).to(self.device)
        with torch.no_grad():
            content, _ = self.net.encode(x)
            out = self.net.decode(content, self.style)
        return self._tensor_to_bgr(out)


class CUTModel(_ModelBase):
    type = 'CUT'

    def __init__(self, path, device='cpu'):
        self.path = path
        self.device = device
        sd = _strip_module_prefix(torch.load(path, map_location=device))
        params = detect_cut(sd)
        if params is None:
            raise ValueError(f'Not a CUT generator checkpoint: {path}')
        self.params = params
        self.size = 256

        from models.networks import define_G  # CUT's networks.py
        self.net = define_G(
            input_nc=params['input_nc'],
            output_nc=params['output_nc'],
            ngf=params['ngf'],
            netG=f"resnet_{params['n_blocks']}blocks",
            norm=params['norm'],
            use_dropout=False,
            init_type='xavier',
            init_gain=0.02,
            no_antialias=params['no_antialias'],
            no_antialias_up=params['no_antialias_up'],
            gpu_ids=[],
            opt=None,
        )
        self.net.load_state_dict(sd, strict=False)
        self.net.eval()
        self.net.to(device)
        print(f'[CUT] loaded {path} on {next(self.net.parameters()).device}')

    def __call__(self, img_bgr):
        x = self._bgr_to_tensor(img_bgr, self.size).to(self.device)
        with torch.no_grad():
            out = self.net(x)
        return self._tensor_to_bgr(out)


def load_model(path, device='cpu'):
    """Auto-detect and load a GNR or CUT model from a .pt/.pth/.ckpt file."""
    obj = torch.load(path, map_location='cpu')
    sd = obj
    if isinstance(obj, dict):
        for key in ('G_A2B_ema', 'G_A2B', 'state_dict'):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
    sd = _strip_module_prefix(sd) if isinstance(sd, dict) else sd
    if detect_gnr(sd) is not None:
        return GNRModel(path, device=device)
    if detect_cut(sd) is not None:
        return CUTModel(path, device=device)
    raise ValueError(f'Could not identify model architecture for {path}')
