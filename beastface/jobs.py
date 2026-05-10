"""Background subprocess jobs with line-by-line output streaming.

A Job spawns a subprocess, captures its merged stdout/stderr line by line
into a thread-safe queue, and exposes `running`, `stop()`, and `drain()`
helpers for the GUI.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading


class Job:
    def __init__(self, cmd, cwd=None, env=None):
        self.cmd = list(cmd)
        self.cwd = str(cwd) if cwd is not None else None
        self.env = env
        self._proc = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = None
        self.returncode = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError('Job already started')
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env.setdefault('PYTHONUNBUFFERED', '1')
        self._proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                self._lines.put(line.rstrip('\n'))
        finally:
            self._proc.wait()
            self.returncode = self._proc.returncode
            self._lines.put(None)  # sentinel

    def drain(self):
        """Yield any output lines available right now (non-blocking)."""
        while True:
            try:
                item = self._lines.get_nowait()
            except queue.Empty:
                return
            yield item

    def stop(self, timeout: float = 3.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        try:
            self._proc.send_signal(signal.SIGTERM)
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
