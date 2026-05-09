#!/usr/bin/env python3
"""Wrapper that patches kohya-ss/sd-scripts gen_img.py for the case where
the local train_util refactor dropped `load_tokenizer` while gen_img still
calls it on the v1/v2 SD path.

Run from inside the sd-scripts directory (cwd) so kohya's relative imports
resolve. Forwards sys.argv to gen_img's argument parser unchanged.
"""

from __future__ import annotations

import sys


def _patch_train_util():
    from library import train_util  # type: ignore[import-not-found]
    if hasattr(train_util, 'load_tokenizer'):
        return
    from library import strategy_sd  # type: ignore[import-not-found]

    def load_tokenizer(args):
        return strategy_sd.SdTokenizeStrategy(
            v2=getattr(args, 'v2', False),
            max_length=getattr(args, 'max_token_length', None),
            tokenizer_cache_dir=getattr(args, 'tokenizer_cache_dir', None),
        ).tokenizer

    train_util.load_tokenizer = load_tokenizer  # type: ignore[attr-defined]


def main() -> None:
    _patch_train_util()
    import gen_img  # type: ignore[import-not-found]
    parser = gen_img.setup_parser()
    args = parser.parse_args()
    gen_img.main(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
