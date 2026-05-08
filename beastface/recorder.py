"""Recording outputs to disk: PNG for stills, OGV (Theora) for video."""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2


def has_ffmpeg():
    return shutil.which('ffmpeg') is not None


class PNGRecorder:
    """Single-shot recorder: write one PNG and finish."""

    def __init__(self, path):
        self.path = path
        self._wrote = False

    def write(self, frame_bgr):
        if self._wrote:
            return
        cv2.imwrite(self.path, frame_bgr)
        self._wrote = True

    def close(self):
        return self.path if self._wrote else None


class OGVRecorder:
    """Encode raw BGR frames into a Theora .ogv via ffmpeg subprocess."""

    def __init__(self, path, fps, width, height):
        self.path = path
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.width = width
        self.height = height
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', f'{self.fps}',
            '-i', '-',
            '-c:v', 'libtheora', '-q:v', '7',
            path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr):
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def close(self):
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        return self.path
