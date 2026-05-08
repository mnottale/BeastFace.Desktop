#!/usr/bin/env python3
"""Strip a GNR training checkpoint down to just the A2B generator weights.

Usage:
    gnr-strip.py <input.pt> <last|ema> <output.pt>

Discards G_B2A weights, discriminators, optimizers and any other training
state. Keeps the chosen G_A2B variant under its original key so the loader
in beastface/models.py picks it up unchanged.
"""

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('input', help='source GNR checkpoint (.pt)')
    ap.add_argument('which', choices=('last', 'ema'),
                    help='which copy of G_A2B to keep')
    ap.add_argument('output', help='destination .pt file')
    args = ap.parse_args()

    src_key = 'G_A2B_ema' if args.which == 'ema' else 'G_A2B'

    ckpt = torch.load(args.input, map_location='cpu')
    if not isinstance(ckpt, dict) or src_key not in ckpt:
        sys.exit(f'error: {args.input} has no "{src_key}" entry '
                 f'(found keys: {sorted(ckpt) if isinstance(ckpt, dict) else type(ckpt).__name__})')

    state = ckpt[src_key]
    if not isinstance(state, dict):
        sys.exit(f'error: "{src_key}" is not a state_dict (got {type(state).__name__})')

    torch.save({src_key: state}, args.output)
    n_params = sum(t.numel() for t in state.values() if torch.is_tensor(t))
    print(f'wrote {args.output}  [{src_key}, {len(state)} tensors, {n_params:,} params]')


if __name__ == '__main__':
    main()
