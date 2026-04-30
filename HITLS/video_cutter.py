#!/usr/bin/env python3
"""
video_cutter.py  –  Fast non-destructive video trimmer
Uses ffmpeg stream-copy (no re-encoding → saves in seconds).

Usage
-----
    python video_cutter.py [video_file]

Controls
--------
  Space / click Play  –  Play / Pause
  C                   –  Set IN-point  (discard everything before here)
  T                   –  Set OUT-point (discard everything after here)
  ←  /  →             –  Step ±5 seconds
  ,  /  .             –  Step ±1 frame
  Enter               –  Save trimmed clip (ffmpeg stream-copy)
  R                   –  Reset in/out to full video
  Slider drag         –  Scrub to any position
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
except ImportError:
    sys.exit("ERROR: opencv-python is required.  Run: pip install opencv-python")

try:
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("ERROR: Pillow is required.  Run: pip install Pillow")


# ─── Palette ───────────────────────────────────────────────────────────────────
C_BG       = '#16161e'
C_BAR      = '#1e1e2e'
C_IN       = '#40ff80'
C_OUT      = '#ff4848'
C_ACTIVE   = '#1e3030'
C_INACTIVE = '#2a2a3a'
C_FG       = '#cdd6f4'
C_DIM      = '#6e738d'
C_BTN_BG   = '#313244'
C_BTN_FG   = '#cdd6f4'
C_MARK_BG  = '#1e2030'


# ─── Helpers ───────────────────────────────────────────────────────────────────
def fmt_time(sec: float) -> str:
    """Return MM:SS.cc"""
    sec = max(0.0, sec)
    m   = int(sec // 60)
    s   = sec % 60
    return f'{m:02d}:{s:05.2f}'


def btn(parent, text, command, bg=C_BTN_BG, fg=C_BTN_FG, **kw):
    kw.setdefault('font', ('sans-serif', 10))
    return tk.Button(parent, text=text, command=command,
                     bg=bg, fg=fg, activebackground='#45475a',
                     activeforeground='#ffffff', relief=tk.FLAT,
                     padx=8, pady=3,
                     cursor='hand2', **kw)


# ─── Main application ──────────────────────────────────────────────────────────
class VideoCutter:
    def __init__(self, root: tk.Tk, video_path: str | None = None):
        self.root = root
        root.title('Video Cutter')
        root.configure(bg=C_BG)
        root.geometry('1024x640')
        root.minsize(640, 440)

        # ── Video state ──────────────────────────────────────────────────
        self.cap:          cv2.VideoCapture | None = None
        self.video_path:   Path | None             = None
        self.fps:          float                   = 25.0
        self.total_frames: int                     = 0
        self.duration:     float                   = 0.0
        self.cur_frame:    int                     = 0   # 0-based index

        # ── Edit marks ───────────────────────────────────────────────────
        self.in_frame:  int = 0      # inclusive
        self.out_frame: int = 0      # inclusive, 0 = not set yet (= last frame)

        # ── Playback ─────────────────────────────────────────────────────
        self.playing:      bool       = False
        self._play_job:    str | None = None
        self._last_tick:   float      = 0.0
        self._photo:       ImageTk.PhotoImage | None = None
        self._slider_drag: bool       = False   # True while user holds slider

        self._build_ui()
        self._bind_keys()

        if video_path:
            self._load(video_path)

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        r = self.root

        # ── Video canvas ─────────────────────────────────────────────────
        self.canvas = tk.Canvas(r, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda _e: self._redisplay())

        # ── Bottom controls ───────────────────────────────────────────────
        bottom = tk.Frame(r, bg=C_BAR)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        # ── Time row ─────────────────────────────────────────────────────
        trow = tk.Frame(bottom, bg=C_BAR)
        trow.pack(fill=tk.X, padx=8, pady=(6, 0))

        self.lbl_cur = tk.Label(trow, text='--:--.-', font=('monospace', 11),
                                fg=C_FG, bg=C_BAR)
        self.lbl_cur.pack(side=tk.LEFT)

        self.lbl_dur = tk.Label(trow, text='/ --:--.-', font=('monospace', 11),
                                fg=C_DIM, bg=C_BAR)
        self.lbl_dur.pack(side=tk.LEFT, padx=(4, 0))

        self.lbl_marks = tk.Label(trow, text='In: --  |  Out: --',
                                  font=('monospace', 11), fg='#80c0ff', bg=C_BAR)
        self.lbl_marks.pack(side=tk.RIGHT)

        # ── Slider ───────────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Cut.Horizontal.TScale',
                         troughcolor='#313244', background=C_BAR,
                         sliderlength=16, sliderrelief='flat')

        self.slider_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(bottom, from_=0, to=1000,
                                orient=tk.HORIZONTAL,
                                variable=self.slider_var,
                                style='Cut.Horizontal.TScale',
                                command=self._on_slider_move)
        self.slider.pack(fill=tk.X, padx=8, pady=4)
        self.slider.bind('<ButtonPress-1>',   self._slider_press)
        self.slider.bind('<ButtonRelease-1>', self._slider_release)

        # ── Mark bar (in/out visual) ──────────────────────────────────────
        self.mark_canvas = tk.Canvas(bottom, height=6, bg=C_MARK_BG,
                                     highlightthickness=0)
        self.mark_canvas.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.mark_canvas.bind('<Configure>', lambda _e: self._redraw_marks())

        # ── Buttons ───────────────────────────────────────────────────────
        brow = tk.Frame(bottom, bg=C_BAR)
        brow.pack(fill=tk.X, padx=8, pady=(2, 8))

        btn(brow, '📂 Open',    self._open_file).pack(side=tk.LEFT, padx=(0, 6))

        self.btn_play = btn(brow, '▶  Play',    self._toggle_play,
                            bg='#2a5e3a', fg='#a6e3a1')
        self.btn_play.pack(side=tk.LEFT, padx=2)

        btn(brow, '← 5s',    lambda: self._step(-5)).pack(side=tk.LEFT, padx=2)
        btn(brow, '5s →',    lambda: self._step(+5)).pack(side=tk.LEFT, padx=2)

        sep = tk.Frame(brow, width=1, bg=C_DIM)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        btn(brow, '[C]  Set In',  self._set_in,  bg='#2a4030', fg=C_IN
            ).pack(side=tk.LEFT, padx=2)
        btn(brow, '[T]  Set Out', self._set_out, bg='#402a2a', fg=C_OUT
            ).pack(side=tk.LEFT, padx=2)
        btn(brow, 'R  Reset',     self._reset_marks,
            bg=C_BTN_BG, fg=C_DIM).pack(side=tk.LEFT, padx=2)

        btn(brow, '↵  Save', self._save,
            bg='#1a5e1a', fg='#a6e3a1',
            font=('sans-serif', 11)).pack(side=tk.RIGHT, padx=(6, 0))

        # ── Status bar ────────────────────────────────────────────────────
        self.status = tk.Label(r,
            text='  Open a video  ·  C = set in-point  ·  T = set out-point'
                 '  ·  Enter = save  ·  ← → = ±5 s  ·  , . = ±1 frame',
            font=('sans-serif', 9), fg=C_DIM, bg='#0e0e16',
            anchor=tk.W, padx=6, pady=3)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # ─────────────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────────────
    def _bind_keys(self):
        r = self.root
        r.bind('<space>',  lambda _e: self._toggle_play())
        r.bind('<c>',      lambda _e: self._set_in())
        r.bind('<C>',      lambda _e: self._set_in())
        r.bind('<t>',      lambda _e: self._set_out())
        r.bind('<T>',      lambda _e: self._set_out())
        r.bind('<r>',      lambda _e: self._reset_marks())
        r.bind('<R>',      lambda _e: self._reset_marks())
        r.bind('<Return>',  lambda _e: self._save())
        r.bind('<Left>',   lambda _e: self._step(-5))
        r.bind('<Right>',  lambda _e: self._step(+5))
        r.bind('<comma>',  lambda _e: self._step_frames(-1))
        r.bind('<period>', lambda _e: self._step_frames(+1))

    # ─────────────────────────────────────────────────────────────────────────
    # File loading
    # ─────────────────────────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title='Open video',
            filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv *.webm *.ts'),
                       ('All files', '*.*')])
        if path:
            self._load(path)

    def _load(self, path: str):
        was_playing = self.playing
        self._stop_playback()

        if self.cap:
            self.cap.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror('Cannot open', f'Could not open:\n{path}')
            return

        self.cap          = cap
        self.video_path   = Path(path)
        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration     = self.total_frames / self.fps
        self.cur_frame    = 0
        self.in_frame     = 0
        self.out_frame    = self.total_frames - 1

        self.root.title(f'Video Cutter  —  {self.video_path.name}')
        self._seek(0)
        self._update_labels()
        self._redraw_marks()
        self._set_status(f'Loaded: {self.video_path.name}  '
                         f'({fmt_time(self.duration)}  '
                         f'{self.fps:.3f} fps  '
                         f'{self.total_frames} frames)')

    # ─────────────────────────────────────────────────────────────────────────
    # Playback
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if not self.cap:
            return
        if self.playing:
            self._stop_playback()
        else:
            # If already at/past out-point, rewind to in-point first
            if self.cur_frame >= self.out_frame:
                self._seek(self.in_frame)
            self.playing      = True
            self._last_tick   = time.monotonic()
            self.btn_play.config(text='⏸  Pause')
            self._tick()

    def _stop_playback(self):
        self.playing = False
        self.btn_play.config(text='▶  Play')
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _tick(self):
        if not self.playing or not self.cap:
            return

        # Stop at out-point
        if self.cur_frame >= self.out_frame:
            self._stop_playback()
            return

        ret, frame = self.cap.read()
        if not ret:
            self._stop_playback()
            return

        self.cur_frame += 1
        now            = time.monotonic()
        elapsed        = now - self._last_tick
        self._last_tick = now
        interval        = 1.0 / self.fps
        delay_ms        = max(1, int((interval - elapsed) * 1000))

        self._display(frame)
        if not self._slider_drag:
            self._update_slider_pos()
        self._update_labels()

        self._play_job = self.root.after(delay_ms, self._tick)

    # ─────────────────────────────────────────────────────────────────────────
    # Seeking
    # ─────────────────────────────────────────────────────────────────────────
    def _seek(self, frame_idx: int):
        if not self.cap:
            return
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.cur_frame = frame_idx
            self._display(frame)
        self._update_slider_pos()
        self._update_labels()

    def _step(self, seconds: float):
        if not self.cap:
            return
        target = self.cur_frame + int(seconds * self.fps)
        self._seek(target)

    def _step_frames(self, delta: int):
        if not self.cap:
            return
        self._seek(self.cur_frame + delta)

    # ─────────────────────────────────────────────────────────────────────────
    # Slider interaction
    # ─────────────────────────────────────────────────────────────────────────
    def _slider_press(self, _event):
        self._slider_drag = True
        was_playing       = self.playing
        self._stop_playback()
        self._was_playing = was_playing

    def _slider_release(self, _event):
        self._slider_drag = False
        # Commit seek
        frac       = self.slider_var.get() / 1000.0
        frame_idx  = int(frac * (self.total_frames - 1))
        self._seek(frame_idx)
        if getattr(self, '_was_playing', False):
            self._toggle_play()

    def _on_slider_move(self, val):
        """Called continuously while slider is dragged."""
        if not self.cap or not self._slider_drag:
            return
        frac      = float(val) / 1000.0
        frame_idx = int(frac * (self.total_frames - 1))
        # Lightweight seek: just read one frame without triggering full _seek
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.cur_frame = frame_idx
            self._display(frame)
            self._update_labels()

    def _update_slider_pos(self):
        if not self.total_frames:
            return
        frac = self.cur_frame / max(1, self.total_frames - 1)
        self.slider_var.set(frac * 1000.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Frame display
    # ─────────────────────────────────────────────────────────────────────────
    def _display(self, frame):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 8 or ch < 8:
            cw, ch = 960, 540

        h, w    = frame.shape[:2]
        scale   = min(cw / w, ch / h)
        nw, nh  = max(1, int(w * scale)), max(1, int(h * scale))

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb)
        photo   = ImageTk.PhotoImage(img)

        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
        self._photo = photo   # prevent GC

    def _redisplay(self):
        """Re-draw current frame after window resize."""
        if not self.cap or self.playing:
            return
        frame_idx = max(0, self.cur_frame - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self._display(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # In / Out marks
    # ─────────────────────────────────────────────────────────────────────────
    def _set_in(self):
        if not self.cap:
            return
        self.in_frame = self.cur_frame
        self._update_labels()
        self._redraw_marks()
        self._set_status(f'In-point set at  {fmt_time(self.in_frame / self.fps)}')

    def _set_out(self):
        if not self.cap:
            return
        self.out_frame = self.cur_frame
        self._update_labels()
        self._redraw_marks()
        self._set_status(f'Out-point set at  {fmt_time(self.out_frame / self.fps)}')

    def _reset_marks(self):
        if not self.cap:
            return
        self.in_frame  = 0
        self.out_frame = self.total_frames - 1
        self._update_labels()
        self._redraw_marks()
        self._set_status('Marks reset to full video')

    # ─────────────────────────────────────────────────────────────────────────
    # Mark bar drawing
    # ─────────────────────────────────────────────────────────────────────────
    def _redraw_marks(self):
        mc = self.mark_canvas
        mc.update_idletasks()
        w = mc.winfo_width()
        h = mc.winfo_height()
        if w < 4 or not self.total_frames:
            return
        tf = self.total_frames - 1

        mc.delete('all')

        # Background
        mc.create_rectangle(0, 0, w, h, fill=C_MARK_BG, outline='')

        # Kept region (green tint)
        xi = int(self.in_frame  / tf * w)
        xo = int(self.out_frame / tf * w)
        mc.create_rectangle(xi, 0, xo, h, fill='#1a3022', outline='')

        # Discarded tails (dark red)
        if xi > 0:
            mc.create_rectangle(0, 0, xi, h, fill='#2a1616', outline='')
        if xo < w:
            mc.create_rectangle(xo, 0, w,  h, fill='#2a1616', outline='')

        # In-point line (green)
        mc.create_line(xi, 0, xi, h, fill=C_IN,  width=2)
        # Out-point line (red)
        mc.create_line(xo, 0, xo, h, fill=C_OUT, width=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Label updates
    # ─────────────────────────────────────────────────────────────────────────
    def _update_labels(self):
        cur_s = self.cur_frame / self.fps
        self.lbl_cur.config(text=fmt_time(cur_s))
        self.lbl_dur.config(text=f'/ {fmt_time(self.duration)}')
        in_s  = self.in_frame  / self.fps
        out_s = self.out_frame / self.fps
        self.lbl_marks.config(
            text=f'In: {fmt_time(in_s)}  |  Out: {fmt_time(out_s)}'
                 f'  |  Keep: {fmt_time(out_s - in_s)}'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────
    def _save(self):
        if not self.cap or not self.video_path:
            return

        in_s  = self.in_frame  / self.fps
        out_s = self.out_frame / self.fps

        if in_s >= out_s:
            messagebox.showerror('Invalid range',
                                 f'In-point ({fmt_time(in_s)}) must be '
                                 f'before out-point ({fmt_time(out_s)}).')
            return

        # Default filename
        stem     = self.video_path.stem
        suffix   = self.video_path.suffix or '.mp4'

        def tag(s):
            m = int(s // 60)
            sec = int(s % 60)
            return f'{m:02d}m{sec:02d}s'

        default_name = f'{stem}_cut_{tag(in_s)}-{tag(out_s)}{suffix}'

        save_path = filedialog.asksaveasfilename(
            title='Save trimmed clip',
            initialdir=str(self.video_path.parent),
            initialfile=default_name,
            defaultextension=suffix,
            filetypes=[('MP4', '*.mp4'), ('MKV', '*.mkv'),
                       ('AVI', '*.avi'),  ('All files', '*.*')])
        if not save_path:
            return

        self._stop_playback()
        self._set_status('Saving … (stream-copy, should take < 1 s per GB)')
        self.root.update_idletasks()

        # ffmpeg stream-copy: -ss before -i for fast seek
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(in_s),
            '-to', str(out_s),
            '-i', str(self.video_path),
            '-c', 'copy',
            '-avoid_negative_ts', '1',
            save_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            messagebox.showerror('Timeout', 'ffmpeg took too long (> 10 min).')
            return

        if result.returncode == 0:
            size_mb  = Path(save_path).stat().st_size / 1_048_576
            kept_dur = out_s - in_s
            self._set_status(
                f'Saved: {Path(save_path).name}  ({size_mb:.1f} MB  '
                f'{fmt_time(kept_dur)})')
            messagebox.showinfo(
                'Saved',
                f'{Path(save_path).name}\n\n'
                f'Duration : {fmt_time(kept_dur)}\n'
                f'File size: {size_mb:.1f} MB')
        else:
            tail = result.stderr.strip().splitlines()
            msg  = '\n'.join(tail[-8:]) if tail else '(no output)'
            messagebox.showerror('ffmpeg error', msg)
            self._set_status('Save failed — see error dialog')

    # ─────────────────────────────────────────────────────────────────────────
    # Status bar
    # ─────────────────────────────────────────────────────────────────────────
    def _set_status(self, msg: str):
        self.status.config(text=f'  {msg}')


# ─── Entry point ───────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = VideoCutter(root, video_path)
    try:
        root.mainloop()
    finally:
        app._stop_playback()
        if app.cap:
            app.cap.release()


if __name__ == '__main__':
    main()
