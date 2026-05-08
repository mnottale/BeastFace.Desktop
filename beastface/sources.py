"""Frame sources: still image, video file, webcam."""

from __future__ import annotations

import platform
import sys

import cv2


class Source:
    fps: float = 30.0
    is_live: bool = False
    is_image: bool = False

    def read(self):
        """Return the next BGR frame as numpy array, or None on end."""
        raise NotImplementedError

    def close(self):
        pass


class ImageSource(Source):
    is_image = True

    def __init__(self, path):
        self.path = path
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f'Could not load image: {path}')
        self.img = img

    def read(self):
        return self.img.copy()


class VideoSource(Source):
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f'Could not open video: {path}')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    def close(self):
        self.cap.release()


class WebcamSource(Source):
    is_live = True

    def __init__(self, index, width=640, height=480):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise ValueError(f'Could not open webcam {index}')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        self.cap.release()


def list_webcams(max_index=10):
    """Probe /dev/video* (Linux) or sequential indices to discover webcams."""
    found = []
    if platform.system() == 'Linux':
        import glob
        for dev in sorted(glob.glob('/dev/video*')):
            try:
                idx = int(dev[len('/dev/video'):])
            except ValueError:
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append((idx, dev))
                cap.release()
    else:
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append((idx, f'Camera {idx}'))
                cap.release()
    return found
