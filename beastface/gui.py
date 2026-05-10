"""Tkinter GUI for the BeastFace visualiser."""

from __future__ import annotations

import datetime
import os
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import torch
from PIL import Image, ImageTk

from . import paths, sources, recorder, virtualcam
from .models import load_model
from .pipeline import RenderPipeline

paths.setup()

RESOLUTIONS = (256, 512, 768, 1024)


class LoadedModel:
    """A model the user has added to the visualiser, plus its UI state."""

    __slots__ = ('model', 'name', 'enabled', '_style_combo')

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.enabled = True
        self._style_combo = None


class BeastFaceApp:
    def __init__(self, root):
        self.root = root
        root.title('BeastFace Desktop')
        root.geometry('1280x800')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_var = tk.StringVar(value=self.device)
        self.show_source = tk.BooleanVar(value=False)
        self.show_model = tk.BooleanVar(value=True)
        self.resolution_var = tk.IntVar(value=512)

        self.source = None
        self.models: list[LoadedModel] = []
        self.models_lock = threading.Lock()
        self.recorder = None
        self._recording_path = None
        self._displayed_seq = -1
        self._photo = None  # keep a ref so Tk doesn't GC it

        self.virtualcam = virtualcam.VirtualCam()
        self.virtualcam_enabled = tk.BooleanVar(value=False)

        self._build_ui()

        self.pipeline = RenderPipeline(self)
        self.pipeline.start()
        self.root.after(33, self._tick_display)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text='Source:').pack(side=tk.LEFT)
        self.source_kind = tk.StringVar(value='image')
        kind_box = ttk.Combobox(
            top, textvariable=self.source_kind,
            values=('image', 'video', 'webcam'),
            state='readonly', width=10,
        )
        kind_box.pack(side=tk.LEFT, padx=(4, 4))
        kind_box.bind('<<ComboboxSelected>>', self._on_source_kind)

        self.webcam_var = tk.StringVar()
        self.webcam_box = ttk.Combobox(
            top, textvariable=self.webcam_var, state='readonly', width=20,
        )
        self.webcam_box.pack(side=tk.LEFT, padx=(0, 4))
        self.webcam_box.bind('<<ComboboxSelected>>', lambda _e: self._open_webcam())

        ttk.Button(top, text='Open…', command=self._pick_source).pack(side=tk.LEFT, padx=(0, 4))
        self.source_label = ttk.Label(top, text='(none)')
        self.source_label.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text='Device:').pack(side=tk.LEFT)
        device_box = ttk.Combobox(
            top, textvariable=self.device_var,
            values=('cuda', 'cpu') if torch.cuda.is_available() else ('cpu',),
            state='readonly', width=6,
        )
        device_box.pack(side=tk.LEFT, padx=(4, 12))
        device_box.bind('<<ComboboxSelected>>', self._on_device_change)

        ttk.Label(top, text='Vertical res:').pack(side=tk.LEFT)
        res_box = ttk.Combobox(
            top, textvariable=self.resolution_var,
            values=RESOLUTIONS, state='readonly', width=6,
        )
        res_box.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Checkbutton(top, text='show source', variable=self.show_source).pack(side=tk.LEFT)
        ttk.Checkbutton(top, text='show model', variable=self.show_model).pack(side=tk.LEFT)

        self.vcam_check = ttk.Checkbutton(
            top, text='virtual cam', variable=self.virtualcam_enabled,
            command=self._on_virtualcam_toggle,
        )
        self.vcam_check.pack(side=tk.LEFT, padx=(8, 0))
        if not virtualcam.AVAILABLE:
            self.vcam_check.state(['disabled'])

        self.record_btn = ttk.Button(top, text='● Record', command=self._toggle_record)
        self.record_btn.pack(side=tk.RIGHT)

        # Models row
        mid = ttk.Frame(self.root, padding=6)
        mid.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(mid, text='+ Add model', command=self._add_model).pack(side=tk.LEFT)
        self.status_label = ttk.Label(mid, text='')
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.cards_frame = ttk.Frame(self.root, padding=4)
        self.cards_frame.pack(side=tk.TOP, fill=tk.X)

        # Display area
        self.canvas = tk.Canvas(self.root, bg='#222', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Populate webcam list
        self._refresh_webcams()
        self.webcam_box.pack_forget()

    def _refresh_webcams(self):
        try:
            cams = sources.list_webcams()
        except Exception:
            cams = []
        items = [f'{idx} {name}' for idx, name in cams]
        self._webcams = cams
        self.webcam_box['values'] = items
        if items:
            self.webcam_box.current(0)

    def _on_source_kind(self, _event=None):
        kind = self.source_kind.get()
        if kind == 'webcam':
            self._refresh_webcams()
            self.webcam_box.pack(side=tk.LEFT, padx=(0, 4))
        else:
            self.webcam_box.pack_forget()

    def _on_virtualcam_toggle(self):
        if not self.virtualcam_enabled.get():
            self.virtualcam.close()
            return
        if not virtualcam.AVAILABLE:
            self.virtualcam_enabled.set(False)
            messagebox.showerror(
                'Virtual camera unavailable',
                f'pyvirtualcam could not be loaded:\n\n{virtualcam.IMPORT_ERROR}',
            )

    def _on_device_change(self, _event=None):
        new_dev = self.device_var.get()
        self.device = new_dev
        with self.models_lock:
            for lm in self.models:
                try:
                    lm.model.to(new_dev)
                except Exception:
                    traceback.print_exc()

    # ------------------------------------------------------------ source

    def _pick_source(self):
        kind = self.source_kind.get()
        if kind == 'image':
            path = filedialog.askopenfilename(
                title='Select an image',
                filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp'), ('All', '*.*')],
            )
            if not path:
                return
            self._set_source(sources.ImageSource(path), os.path.basename(path))
        elif kind == 'video':
            path = filedialog.askopenfilename(
                title='Select a video',
                filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv *.webm *.ogv'), ('All', '*.*')],
            )
            if not path:
                return
            self._set_source(sources.VideoSource(path), os.path.basename(path))
        else:
            self._open_webcam()

    def _open_webcam(self):
        if not getattr(self, '_webcams', None):
            messagebox.showinfo('No webcams', 'No webcams detected.')
            return
        sel = self.webcam_box.current()
        if sel < 0:
            sel = 0
        idx, name = self._webcams[sel]
        try:
            src = sources.WebcamSource(idx)
        except Exception as exc:
            messagebox.showerror('Webcam error', str(exc))
            return
        self._set_source(src, name)

    def _set_source(self, src, label):
        old = self.source
        self.source = src
        self.source_label.config(text=label)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

    # ------------------------------------------------------------- models

    def _add_model(self):
        path = filedialog.askopenfilename(
            title='Select a model checkpoint',
            filetypes=[('Checkpoints', '*.pt *.pth *.ckpt'), ('All', '*.*')],
        )
        if not path:
            return
        self.status_label.config(text=f'Loading {os.path.basename(path)}…')
        self.root.update_idletasks()

        def worker():
            try:
                model = load_model(path, device=self.device)
                lm = LoadedModel(model, os.path.basename(path))
                self.root.after(0, lambda: self._on_model_loaded(lm, path))
            except Exception as exc:
                traceback.print_exc()
                self.root.after(0, lambda: self._on_model_failed(exc, path))

        threading.Thread(target=worker, daemon=True).start()

    def _on_model_loaded(self, lm, path):
        with self.models_lock:
            self.models.append(lm)
        params = getattr(lm.model, 'params', {}) or {}
        param_str = ', '.join(f'{k}={v}' for k, v in params.items())
        try:
            actual = str(next(lm.model.net.parameters()).device)
        except Exception:
            actual = lm.model.device
        self.status_label.config(
            text=f'Loaded {os.path.basename(path)} [{lm.model.type} on {actual}] {param_str}'
        )
        self._rebuild_cards()

    def _on_model_failed(self, exc, path):
        self.status_label.config(text='')
        messagebox.showerror('Failed to load model', f'{path}\n\n{exc}')

    def _rebuild_cards(self):
        for child in list(self.cards_frame.children.values()):
            child.destroy()

        col = 0
        with self.models_lock:
            models_snapshot = list(self.models)
        for lm in models_snapshot:
            card = ttk.LabelFrame(self.cards_frame, text=f'{lm.model.type}')
            card.grid(row=0, column=col, padx=4, pady=4, sticky='nw')
            col += 1
            ttk.Label(card, text=lm.name, wraplength=180).pack(anchor='w', padx=4, pady=2)
            enabled_var = tk.BooleanVar(value=lm.enabled)

            def make_toggle(target=lm, var=enabled_var):
                def _t():
                    target.enabled = var.get()
                return _t

            ttk.Checkbutton(card, text='enabled', variable=enabled_var,
                            command=make_toggle()).pack(anchor='w', padx=4)

            row = ttk.Frame(card)
            row.pack(anchor='w', padx=4, pady=2)
            ttk.Button(row, text='Remove',
                       command=lambda target=lm: self._remove_model(target)).pack(side=tk.LEFT)
            if lm.model.type == 'GNR':
                ttk.Button(row, text='Roll style',
                           command=lambda target=lm: target.model.roll()).pack(side=tk.LEFT, padx=(4, 0))
                ttk.Button(row, text='Save style as…',
                           command=lambda target=lm: self._save_style(target)).pack(side=tk.LEFT, padx=(4, 0))

                styles_row = ttk.Frame(card)
                styles_row.pack(anchor='w', padx=4, pady=2)
                ttk.Label(styles_row, text='Saved style:').pack(side=tk.LEFT)
                style_var = tk.StringVar()
                combo = ttk.Combobox(styles_row, textvariable=style_var,
                                     state='readonly', width=18,
                                     values=sorted(lm.model.list_styles().keys()))
                combo.pack(side=tk.LEFT, padx=(4, 0))
                combo.bind(
                    '<<ComboboxSelected>>',
                    lambda _e, target=lm, var=style_var:
                        self._load_style(target, var.get()),
                )
                lm._style_combo = combo  # so _save_style can refresh values

    def _remove_model(self, lm):
        with self.models_lock:
            if lm in self.models:
                self.models.remove(lm)
        self._rebuild_cards()

    # ----------------------------------------------------- GNR styles

    def _save_style(self, lm):
        name = simpledialog.askstring(
            'Save style', 'Name for this style:', parent=self.root,
        )
        if not name:
            return
        try:
            lm.model.save_style(name)
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror('Save style failed', str(exc))
            return
        combo = getattr(lm, '_style_combo', None)
        if combo is not None:
            combo['values'] = sorted(lm.model.list_styles().keys())
            combo.set(name)
        self.status_label.config(
            text=f'Saved style "{name}" → {lm.model.styles_path()}'
        )

    def _load_style(self, lm, name):
        if not name:
            return
        try:
            lm.model.load_style(name)
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror('Load style failed', str(exc))
            return
        self.status_label.config(text=f'Loaded style "{name}"')

    # ----------------------------------------------------------- recording

    def _toggle_record(self):
        if self.recorder is not None:
            self.finalize_recording()
            return
        if self.source is None:
            messagebox.showinfo('Recording', 'Pick a source first.')
            return

        rec_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'recordings')
        os.makedirs(rec_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        if self.source.is_image:
            path = os.path.join(rec_dir, f'beastface-{ts}.png')
            self.recorder = recorder.PNGRecorder(path)
            self._recording_path = path
            self.record_btn.config(text='● Saving…')
            return

        # Video/webcam: need to know frame size. Wait for one composite.
        latest, _, _ = self.pipeline.latest()
        if latest is None:
            messagebox.showinfo('Recording', 'Waiting for first frame; try again in a moment.')
            return
        h, w = latest.shape[:2]
        if not recorder.has_ffmpeg():
            messagebox.showerror('ffmpeg missing', 'ffmpeg is required to encode .ogv videos.')
            return
        path = os.path.join(rec_dir, f'beastface-{ts}.ogv')
        self.recorder = recorder.OGVRecorder(path, fps=self.source.fps, width=w, height=h)
        self._recording_path = path
        self.record_btn.config(text='■ Stop')

    def finalize_recording(self):
        rec = self.recorder
        if rec is None:
            return
        self.recorder = None
        try:
            path = rec.close()
        except Exception:
            traceback.print_exc()
            path = self._recording_path
        self.record_btn.config(text='● Record')
        self.status_label.config(text=f'Saved: {path}')

    # ------------------------------------------------------------ display

    @property
    def vertical_resolution(self):
        return int(self.resolution_var.get())

    def _tick_display(self):
        try:
            frame, seq, err = self.pipeline.latest()
            if err:
                self.status_label.config(text=f'Error: {err}')
            if frame is not None and seq != self._displayed_seq:
                self._displayed_seq = seq
                self._draw(frame)
        finally:
            self.root.after(33, self._tick_display)

    def _draw(self, frame_bgr):
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        h, w = frame_bgr.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo)

    def _on_close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        try:
            if self.source is not None:
                self.source.close()
        except Exception:
            pass
        try:
            if self.recorder is not None:
                self.recorder.close()
        except Exception:
            pass
        try:
            self.virtualcam.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    BeastFaceApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
