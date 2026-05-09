"""Optional virtual-camera output sink via pyvirtualcam."""

from __future__ import annotations

import threading

import cv2

try:
    import pyvirtualcam
    AVAILABLE = True
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import-time only
    pyvirtualcam = None
    AVAILABLE = False
    IMPORT_ERROR = str(exc)


class VirtualCam:
    """Lazily-managed pyvirtualcam.Camera. Recreated when frame size changes."""

    def __init__(self):
        self._cam = None
        self._size = None
        self._lock = threading.Lock()
        self.last_error = None

    @property
    def device(self):
        return getattr(self._cam, 'device', None) if self._cam is not None else None

    def push(self, frame_bgr, fps=30):
        if not AVAILABLE:
            self.last_error = f'pyvirtualcam unavailable: {IMPORT_ERROR}'
            return False
        h, w = frame_bgr.shape[:2]
        with self._lock:
            if self._cam is None or self._size != (w, h):
                self._open(w, h, fps)
                if self._cam is None:
                    return False
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                self._cam.send(rgb)
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self._close_locked()
                return False

    def close(self):
        with self._lock:
            self._close_locked()

    def _open(self, w, h, fps):
        self._close_locked()
        try:
            self._cam = pyvirtualcam.Camera(
                width=w, height=h, fps=int(fps) if fps and fps > 0 else 30,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
            self._size = (w, h)
            self.last_error = None
        except Exception as exc:
            self._cam = None
            self._size = None
            self.last_error = str(exc)

    def _close_locked(self):
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
        self._cam = None
        self._size = None
