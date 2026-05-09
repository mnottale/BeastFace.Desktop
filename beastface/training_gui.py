"""Tkinter GUI for the BeastFace training pipeline.

Step 1: LoRA training (kohya-ss/sd-scripts train_network.py)
Step 2: SD many-shots generation (kohya gen_img.py with the trained LoRA)
Step 3: GNR training (libraries/GNR/train.py)
Step 4: CUT training (libraries/CUT/train.py)
"""

from __future__ import annotations

import os
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import cut_trainer, gnr_trainer, lora_trainer, paths, sd_gen

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


class _ResumableStep(_Step):
    """Base for steps with a `Resume from` directory and a hyperparameter
    form auto-built from a list of (name, flag, kind, default) tuples.

    Subclasses set `trainer_module` (which must expose `HPARAMS`,
    `default_params`, `load_params`, `prepare_run`) plus a few choice maps
    used to render combobox-typed fields.
    """

    trainer_module = None
    combo_choices: dict = {}

    def _build_form(self, parent):
        self.dataset_path = tk.StringVar()
        self.resume_dir = tk.StringVar()

        self._dir_entry(parent, 'Dataset (trainA/B):', self.dataset_path)

        # Resume row with explicit Load button so picking a dir doesn't surprise
        # the user by overwriting form values until they ask for it.
        row = self._row(parent, 'Resume from (opt.):')
        ttk.Entry(row, textvariable=self.resume_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text='Browse…',
                   command=lambda: self._pick_dir(self.resume_dir)).pack(
            side=tk.LEFT, padx=(4, 0))
        ttk.Button(row, text='Load params',
                   command=self._load_resume_params).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=4)

        # Render hyperparams in two columns to keep the form short.
        hp_frame = ttk.Frame(parent)
        hp_frame.pack(fill=tk.X)
        left = ttk.Frame(hp_frame)
        right = ttk.Frame(hp_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.hparam_vars: dict[str, tk.Variable] = {}
        hparams = list(self.trainer_module.HPARAMS)
        half = (len(hparams) + 1) // 2
        for i, (name, _flag, kind, default) in enumerate(hparams):
            var = self._make_var(kind, default)
            self.hparam_vars[name] = var
            target = left if i < half else right
            label = name.replace('_', ' ') + ':'
            if kind == 'flag':
                row = self._row(target, label)
                ttk.Checkbutton(row, variable=var).pack(side=tk.LEFT)
            elif name in self.combo_choices:
                self._combo(target, label, var, self.combo_choices[name])
            elif kind in (int, float):
                self._spin_for(target, label, var, kind)
            else:
                self._text_entry(target, label, var, width=20)

    @staticmethod
    def _make_var(kind, default):
        if kind == 'flag':
            v = tk.BooleanVar()
            v.set(bool(default))
            return v
        if kind is int:
            v = tk.IntVar()
            v.set(int(default))
            return v
        if kind is float:
            v = tk.DoubleVar()
            v.set(float(default))
            return v
        v = tk.StringVar()
        v.set(str(default))
        return v

    def _spin_for(self, parent, label, var, kind):
        row = self._row(parent, label)
        if kind is int:
            ttk.Spinbox(row, from_=0, to=10_000_000, increment=1,
                        textvariable=var, width=12).pack(side=tk.LEFT)
        else:
            ttk.Spinbox(row, from_=0.0, to=1_000_000.0, increment=0.001,
                        textvariable=var, width=12).pack(side=tk.LEFT)

    def _load_resume_params(self):
        d = self.resume_dir.get().strip()
        if not d:
            messagebox.showinfo('Resume', 'Pick a resume directory first.')
            return
        path = Path(d)
        params = self.trainer_module.load_params(path)
        if params is None:
            messagebox.showwarning(
                'Resume',
                f'No params.json found in {path}.\n'
                'Form left as-is; you can still resume but architectural '
                'params must match the saved checkpoint.',
            )
            return
        if 'dataset_path' in params and not self.dataset_path.get().strip():
            self.dataset_path.set(params['dataset_path'])
        for name, _flag, _kind, _default in self.trainer_module.HPARAMS:
            if name in params and name in self.hparam_vars:
                try:
                    self.hparam_vars[name].set(params[name])
                except Exception:
                    pass
        self._append_log(f'loaded params from {path}')

    def _gather_params(self) -> dict:
        out = {}
        for name, _flag, kind, default in self.trainer_module.HPARAMS:
            try:
                v = self.hparam_vars[name].get()
            except Exception:
                v = default
            out[name] = v
        return out

    def _launch_job(self):
        if not self.dataset_path.get().strip():
            raise ValueError('Pick a dataset directory (must contain trainA/ and trainB/).')

        resume_dir = self.resume_dir.get().strip()
        if resume_dir:
            experiment_dir = Path(resume_dir)
            resume = True
        else:
            experiment_dir = None
            resume = False

        job, exp = self.trainer_module.prepare_run(
            dataset_path=self.dataset_path.get().strip(),
            experiment_dir=experiment_dir,
            params=self._gather_params(),
            resume=resume,
        )
        self.job = job
        self._experiment_dir = exp
        self._append_log(f'experiment dir: {exp}')

    def _on_complete(self, rc):
        if rc == 0:
            self._append_log(f'\n✓ checkpoints in {self._experiment_dir}')


class GnrStep(_ResumableStep):
    title = '3: GNR training'
    trainer_module = gnr_trainer


class CutStep(_ResumableStep):
    title = '4: CUT training'
    trainer_module = cut_trainer
    combo_choices = {
        'netG': cut_trainer.NETG_CHOICES,
        'netD': cut_trainer.NETD_CHOICES,
        'normG': cut_trainer.NORM_CHOICES,
        'normD': cut_trainer.NORM_CHOICES,
        'direction': cut_trainer.DIRECTION_CHOICES,
        'preprocess': cut_trainer.PREPROCESS_CHOICES,
        'gan_mode': cut_trainer.GAN_MODE_CHOICES,
        'lr_policy': cut_trainer.LR_POLICY_CHOICES,
    }


class TrainingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('BeastFace Training')
        root.geometry('1100x780')

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)

        self.lora_step = LoraStep(nb, self)
        self.sd_step = SdGenStep(nb, self)
        self.gnr_step = GnrStep(nb, self)
        self.cut_step = CutStep(nb, self)
        self._steps = (self.lora_step, self.sd_step, self.gnr_step, self.cut_step)
        for step in self._steps:
            step.build()
            nb.add(step.frame, text=step.title)

        root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self):
        for step in self._steps:
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
