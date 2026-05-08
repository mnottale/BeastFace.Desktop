#!/usr/bin/env python3
"""BeastFace Desktop - portable GUI app for face-to-sona DNN visualisation."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from beastface.gui import main

if __name__ == '__main__':
    main()
