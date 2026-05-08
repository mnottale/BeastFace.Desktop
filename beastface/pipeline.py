"""Background render thread that ties the source, preprocessor and models together."""

from __future__ import annotations

import threading
import time
import traceback

import cv2
import numpy as np


def hcat(images):
    """Horizontally stack a list of BGR images, resizing them to share height."""
    images = [im for im in images if im is not None]
    if not images:
        return None
    h = max(im.shape[0] for im in images)
    out = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            new_w = int(round(im.shape[1] * scale))
            im = cv2.resize(im, (new_w, h), interpolation=cv2.INTER_AREA)
        out.append(im)
    return np.concatenate(out, axis=1)


class RenderPipeline:
    """Continuous render loop driven by a background thread.

    Holds a single 'latest frame' that the GUI can poll; if the GUI is slower
    than the pipeline, intermediate frames are simply dropped.
    """

    def __init__(self, app):
        self.app = app
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None  # most recent rendered BGR composite
        self._latest_seq = 0
        self._error = None
        self._last_no_face_log = 0.0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self):
        with self._lock:
            return self._latest, self._latest_seq, self._error

    def _publish(self, frame, err=None):
        with self._lock:
            self._latest = frame
            self._latest_seq += 1
            self._error = err

    def _run(self):
        from . import paths
        paths.setup()
        from leanftm import face_to_model  # noqa: WPS433 (deferred)
        last_face_img = None
        next_deadline = time.monotonic()
        while not self._stop.is_set():
            loop_start = time.monotonic()
            src = self.app.source
            if src is None:
                time.sleep(0.05)
                continue
            try:
                frame = src.read()
            except Exception as exc:  # pragma: no cover - source errors
                self._publish(None, err=str(exc))
                time.sleep(0.1)
                continue
            if frame is None:
                if src.is_image:
                    time.sleep(0.1)
                    continue
                # video / webcam returned nothing for this tick
                time.sleep(0.01)
                continue

            target_res = self.app.vertical_resolution
            try:
                face_img = face_to_model(frame, target_res, render_3d=True, rpmode=True)
            except Exception:
                traceback.print_exc()
                face_img = None

            if face_img is None:
                # Reuse last successfully-detected face render so the model
                # output doesn't blank out on a transient detection miss.
                face_img = last_face_img
            else:
                last_face_img = face_img

            # face_to_model returns OpenGL-ordered bytes (R,G,B per pixel),
            # but the GNR/CUT models were trained against this array treated
            # as BGR (see viimp.py). We follow the same convention end-to-end:
            # use the array as-is for both display and model input.
            face_bgr = face_img

            tiles = []
            if self.app.show_source.get() and frame is not None:
                tiles.append(self._fit(frame, target_res))
            if self.app.show_model.get() and face_bgr is not None:
                tiles.append(self._fit(face_bgr, target_res))

            model_input = face_bgr if face_bgr is not None else self._fit(frame, target_res)
            if model_input is not None:
                with self.app.models_lock:
                    active = [m for m in self.app.models if m.enabled]
                for m in active:
                    try:
                        out = m.model(model_input)
                        out = self._fit(out, target_res)
                        tiles.append(out)
                    except Exception:
                        traceback.print_exc()

            composite = hcat(tiles)
            if composite is not None:
                self._publish(composite)
                rec = self.app.recorder
                if rec is not None:
                    try:
                        rec.write(composite)
                        if src.is_image:
                            self.app.finalize_recording()
                    except Exception:
                        traceback.print_exc()

            # Pace the loop to roughly match source fps for video files,
            # accounting for the time we already spent rendering this frame.
            if not src.is_live and not src.is_image and src.fps > 0:
                period = 1.0 / src.fps
                next_deadline = max(next_deadline + period, loop_start)
                remaining = next_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            else:
                next_deadline = time.monotonic()

    @staticmethod
    def _fit(img, target_h):
        if img is None:
            return None
        if img.shape[0] == target_h:
            return img
        scale = target_h / img.shape[0]
        new_w = int(round(img.shape[1] * scale))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
