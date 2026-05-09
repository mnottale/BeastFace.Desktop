"""Tkinter GUI for the BeastFace training pipeline.

Step 1: LoRA training (kohya-ss/sd-scripts train_network.py)
Step 2: SD many-shots generation (kohya gen_img.py with the trained LoRA)
"""

from __future__ import annotations

import os
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

from . import lora_trainer, paths, sd_gen

paths.setup()


class _Step:
    """Base class for a single training-pipeline step shown as a tab.

    Each subclass builds its own form on `self.frame`, knows how to launch
    its job via `_launch_job()`, and inherits the shared log pane / Run /
    Stop wiring from this class.
    """

    title = ''

    def __init__(self, parent: tk.Misc, app: 'TrainingApp'):
        self.app = app
        self.frame = ttk.Frame(parent, padding=8)
        self.job = None

    # subclasses fill this in
    def _build_form(self, parent):  # pragma: no cover - UI
        raise NotImplementedError

    def _launch_job(self):  # pragma: no cover - UI
        raise NotImplementedError

    def build(self):
        form = ttk.Frame(self.frame)
        form.pack(side=tk.TOP, fill=tk.X)
        self._build_form(form)

        bar = ttk.Frame(self.frame)
        bar.pack(side=tk.TOP, fill=tk.X, pady=(8, 4))
        self.run_btn = ttk.Button(bar, text='Run', command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(bar, text='Stop', command=self._on_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.status = ttk.Label(bar, text='Idle')
        self.status.pack(side=tk.LEFT, padx=(12, 0))

        log_frame = ttk.LabelFrame(self.frame, text='Output')
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = tk.Text(log_frame, wrap=tk.NONE, height=20,
                           background='#101010', foreground='#cfcfcf',
                           insertbackground='#cfcfcf')
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scroll.set)

    # ---------------- helpers shared by subclasses ----------------

    def _row(self, parent, label):
        row = ttk.Frame(parent)
        row.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=20, anchor='w').pack(side=tk.LEFT)
        return row

    def _file_entry(self, parent, label, var, filetypes):
        row = self._row(parent, label)
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            row, text='Browse…',
            command=lambda: self._pick_file(var, filetypes),
        ).pack(side=tk.LEFT, padx=(4, 0))

    def _dir_entry(self, parent, label, var):
        row = self._row(parent, label)
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text='Browse…',
                   command=lambda: self._pick_dir(var)).pack(side=tk.LEFT, padx=(4, 0))

    def _text_entry(self, parent, label, var, width=20):
        row = self._row(parent, label)
        ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT)

    def _spin(self, parent, label, var, from_, to, increment=1):
        row = self._row(parent, label)
        ttk.Spinbox(row, from_=from_, to=to, increment=increment,
                    textvariable=var, width=10).pack(side=tk.LEFT)

    def _combo(self, parent, label, var, values):
        row = self._row(parent, label)
        ttk.Combobox(row, textvariable=var, values=values,
                     state='readonly', width=10).pack(side=tk.LEFT)

    def _pick_file(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _pick_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # ---------------- run / stop / log ----------------

    def _append_log(self, text):
        self.log.insert(tk.END, text + '\n')
        self.log.see(tk.END)

    def _on_run(self):
        if self.job is not None and self.job.running:
            return
        self.log.delete('1.0', tk.END)
        try:
            self._launch_job()
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror('Cannot start', str(exc))
            return
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status.config(text='Running…')
        self._append_log('$ ' + ' '.join(self.job.cmd))
        self.job.start()
        self._poll()

    def _on_stop(self):
        if self.job is None:
            return
        self.status.config(text='Stopping…')
        self.job.stop()

    def _poll(self):
        if self.job is None:
            return
        for line in self.job.drain():
            if line is None:
                self._on_finished()
                return
            self._append_log(line)
        if not self.job.running:
            # process exited but the sentinel hasn't been seen yet; another tick
            pass
        self.app.root.after(100, self._poll)

    def _on_finished(self):
        rc = self.job.returncode if self.job else None
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.config(text=f'Done (rc={rc})' if rc == 0 else f'Failed (rc={rc})')
        self._on_complete(rc)
        self.job = None

    def _on_complete(self, rc):
        pass


class LoraStep(_Step):
    title = '1: LoRA training'

    def _build_form(self, parent):
        self.base_model = tk.StringVar()
        self.images_dir = tk.StringVar()
        self.class_word = tk.StringVar()
        self.trigger = tk.StringVar()
        self.repeats = tk.IntVar(value=20)
        self.resolution = tk.IntVar(value=512)
        self.steps = tk.IntVar(value=2000)
        self.dim = tk.IntVar(value=16)
        self.alpha = tk.IntVar(value=8)
        self.lr = tk.StringVar(value='1e-4')

        self._file_entry(parent, 'Base SD model:', self.base_model,
                         [('SD checkpoint', '*.safetensors *.ckpt'), ('All', '*')])
        self._dir_entry(parent, 'Few-shot images dir:', self.images_dir)
        self._text_entry(parent, 'Class word:', self.class_word)
        self._text_entry(parent, 'Trigger word:', self.trigger)
        self._spin(parent, 'Repeats:', self.repeats, 1, 1000)
        self._combo(parent, 'Resolution:', self.resolution, (256, 384, 512, 640, 768, 1024))
        self._spin(parent, 'Max train steps:', self.steps, 100, 100000, increment=100)
        self._spin(parent, 'Network dim:', self.dim, 1, 256)
        self._spin(parent, 'Network alpha:', self.alpha, 1, 256)
        self._text_entry(parent, 'Learning rate:', self.lr, width=10)

    def _launch_job(self):
        self._lora_output_path = None
        if not self.base_model.get().strip():
            raise ValueError('Pick a base SD model.')
        if not self.images_dir.get().strip():
            raise ValueError('Pick the few-shot images directory.')
        if not self.class_word.get().strip():
            raise ValueError('Class word is required (e.g. "wolf").')
        if not self.trigger.get().strip():
            raise ValueError('Trigger word is required (e.g. "cpc").')
        try:
            lr = float(self.lr.get())
        except ValueError as exc:
            raise ValueError(f'Invalid learning rate: {exc}') from exc

        job, output_path = lora_trainer.prepare_run(
            base_model=self.base_model.get().strip(),
            images_dir=self.images_dir.get().strip(),
            class_word=self.class_word.get().strip(),
            trigger=self.trigger.get().strip(),
            repeats=int(self.repeats.get()),
            resolution=int(self.resolution.get()),
            max_train_steps=int(self.steps.get()),
            network_dim=int(self.dim.get()),
            network_alpha=int(self.alpha.get()),
            learning_rate=lr,
        )
        self.job = job
        self._lora_output_path = output_path

    def _on_complete(self, rc):
        if rc == 0 and self._lora_output_path is not None:
            self._append_log(f'\n✓ saved {self._lora_output_path}')
            # Pre-populate the SD generation step's lora field for convenience.
            try:
                self.app.sd_step.lora_path.set(str(self._lora_output_path))
            except Exception:
                pass


class SdGenStep(_Step):
    title = '2: SD many shots'

    def _build_form(self, parent):
        self.base_model = tk.StringVar()
        self.lora_path = tk.StringVar()
        self.template_path = tk.StringVar(
            value=os.path.join(paths.ROOT, 'sd-prompts.json')
        )
        self.resolution = tk.IntVar(value=512)
        self.n_images = tk.IntVar(value=200)
        self.network_mul = tk.StringVar(value='1.0')
        self.steps = tk.IntVar(value=30)

        self._file_entry(parent, 'Base SD model:', self.base_model,
                         [('SD checkpoint', '*.safetensors *.ckpt'), ('All', '*')])
        self._file_entry(parent, 'LoRA:', self.lora_path,
                         [('LoRA', '*.safetensors'), ('All', '*')])
        self._file_entry(parent, 'Prompt template:', self.template_path,
                         [('JSON', '*.json'), ('All', '*')])
        self._combo(parent, 'Resolution:', self.resolution, (256, 384, 512, 640, 768, 1024))
        self._spin(parent, 'Number of images:', self.n_images, 1, 100000, increment=10)
        self._text_entry(parent, 'Network mul:', self.network_mul, width=8)
        self._spin(parent, 'Sampling steps:', self.steps, 5, 200)

    def _launch_job(self):
        if not self.base_model.get().strip():
            raise ValueError('Pick a base SD model.')
        if not self.lora_path.get().strip():
            raise ValueError('Pick a LoRA file.')
        if not self.template_path.get().strip():
            raise ValueError('Pick a prompt template JSON.')
        try:
            mul = float(self.network_mul.get())
        except ValueError as exc:
            raise ValueError(f'Invalid network mul: {exc}') from exc

        job, out_dir = sd_gen.prepare_run(
            base_model=self.base_model.get().strip(),
            lora_path=self.lora_path.get().strip(),
            template_path=self.template_path.get().strip(),
            resolution=int(self.resolution.get()),
            n_images=int(self.n_images.get()),
            network_mul=mul,
            steps=int(self.steps.get()),
        )
        self.job = job
        self._out_dir = out_dir

    def _on_complete(self, rc):
        if rc == 0:
            self._append_log(f'\n✓ saved to {self._out_dir}')


class TrainingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('BeastFace Training')
        root.geometry('1100x780')

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)

        self.lora_step = LoraStep(nb, self)
        self.sd_step = SdGenStep(nb, self)
        for step in (self.lora_step, self.sd_step):
            step.build()
            nb.add(step.frame, text=step.title)

        root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self):
        for step in (self.lora_step, self.sd_step):
            try:
                if step.job is not None:
                    step.job.stop()
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
    TrainingApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
