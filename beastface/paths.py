"""Configures sys.path so the GNR/CUT/leanftm code in libraries/ can be imported."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, 'libraries')
GNR_DIR = os.path.join(LIB, 'GNR')
CUT_DIR = os.path.join(LIB, 'CUT')


def setup():
    for p in (LIB, GNR_DIR, CUT_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
