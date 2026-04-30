"""
transcript_crosscheck.py — Audio-synchronized transcript editor with vim motions.

Usage:
    python transcript_crosscheck.py HITLS/P02/P02_itw.m4a
    python transcript_crosscheck.py HITLS/P02/P02_itw.m4a --json HITLS/P02/P02_itw_transcript.json

The script auto-detects the JSON sidecar (<stem>_transcript.json) next to the audio file.
Saves edits back to the JSON and regenerates the .txt on quit (Ctrl+S or Q).

────────────────────────────────────────────────────────────────────
KEYBINDINGS  (Normal mode unless noted)
────────────────────────────────────────────────────────────────────
Playback
  Space           pause / resume
  ]  /  [         speed up / slow down  (0.25x steps, range 0.25–3.0)
  r               rewind to start of current segment
  g               seek to segment under cursor

Navigation (vim motions, accept numeric prefix e.g. 3j)
  h / l           cursor left / right (char)
  j / k           move to next / previous segment
  w               forward one word
  b               backward one word
  0               start of segment text
  $               end of segment text
  /               start forward search  (Enter confirms, Esc cancels)
  n               next search match
  N               previous search match

Editing
  i               enter Insert mode  (cursor stays in place)
  a               enter Insert mode after cursor char
  Esc             back to Normal mode  (saves segment text)
  s               substitute char under cursor (enter Insert after deleting 1 char)
  x               delete char under cursor
  dw / daw / diw  delete word / around word / inside word
  das / dis       delete around/inside sentence (whole segment text)
  caw / cis       change around word / inside sentence (delete + Insert)
  p / P           paste after / before cursor  (from internal yank register)
  dj / dk         delete segment below / above current
  dd              delete current segment entirely
  u               undo last change (per-segment, single level)

App
  Ctrl+S          save JSON + regenerate TXT (also done on quit)
  q / Q           save and quit
"""

from __future__ import annotations

import argparse
import copy
import json
import queue
import re
import subprocess
import sys
import threading
import time
try:
    import msvcrt as _msvcrt
    def _flush_kbd() -> None:
        """Drain any buffered keystrokes so they aren't fed into the TUI."""
        while _msvcrt.kbhit():
            _msvcrt.getwch()
except ImportError:
    def _flush_kbd() -> None:  # type: ignore[misc]
        pass  # non-Windows: nothing to drain
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.markup import escape as markup_escape
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.color import Color
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static


# ─── constants ───────────────────────────────────────────────────────────────

SPEED_MIN   = 0.25
SPEED_MAX   = 3.0
SPEED_STEP  = 0.25
TICK_MS     = 80          # how often the sync timer fires (ms)

SPEAKER_COLORS: dict[str, str] = {
    "INTERVIEWER":  "cyan",
    "PARTICIPANT":  "green",
    "SPEAKER_00":   "cyan",
    "SPEAKER_01":   "green",
    "SPEAKER_02":   "yellow",
    "SPEAKER_03":   "magenta",
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def fmt_time(s: float) -> str:
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def speaker_color(spk: str) -> str:
    return SPEAKER_COLORS.get(spk, "white")


def word_boundaries(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets of each word in text."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def sentence_boundaries(text: str) -> tuple[int, int]:
    """Return (start, end) of the whole non-whitespace content (segment = one sentence)."""
    stripped = text.strip()
    if not stripped:
        return 0, 0
    start = text.index(stripped[0])
    return start, start + len(stripped)


# ─── audio backend ───────────────────────────────────────────────────────────

class AudioPlayer:
    """ffmpeg -> sounddevice player with real-time speed control via atempo."""

    SAMPLE_RATE = 44100
    CHANNELS    = 2
    BLOCKSIZE   = 2048  # frames per callback

    def __init__(self, path: Path) -> None:
        self._path        = path
        self._speed       = 1.0
        self._paused      = True
        self._offset      = 0.0
        self._duration    = self._probe_duration(path)
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stream: sd.OutputStream | None = None
        self._stop_evt    = threading.Event()
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=60)
        self._pcm_buf     = bytearray()
        self._play_wall   = 0.0
        self._play_offset = 0.0

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def position(self) -> float:
        if self._paused:
            return self._offset
        return min(self._play_offset + (time.monotonic() - self._play_wall) * self._speed,
                   self._duration)

    def play(self) -> None:
        if self._paused:
            self._paused = False
            self._start(self._offset)

    def pause(self) -> None:
        if not self._paused:
            self._offset = self.position
            self._paused = True
            self._stop()

    def toggle(self) -> None:
        self.pause() if not self._paused else self.play()

    def seek(self, seconds: float) -> None:
        seconds = max(0.0, min(seconds, self._duration))
        was_playing = not self._paused
        self._offset = seconds
        if was_playing:
            self._stop()
            self._paused = False
            self._start(seconds)

    def set_speed(self, speed: float) -> None:
        speed = round(max(SPEED_MIN, min(SPEED_MAX, speed)), 2)
        if speed == self._speed:
            return
        pos = self.position
        self._speed = speed
        self._offset = pos
        if not self._paused:
            self._stop()
            self._start(pos)

    def _atempo(self, speed: float) -> str:
        """Chain atempo filters; each individual filter must be 0.5-2.0."""
        parts: list[str] = []
        s = speed
        while s > 2.0:
            parts.append("atempo=2.0")
            s /= 2.0
        while s < 0.5:
            parts.append("atempo=0.5")
            s *= 2.0
        parts.append(f"atempo={s:.6f}")
        return ",".join(parts)

    def _start(self, offset: float) -> None:
        self._stop_evt.clear()
        self._pcm_buf.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        cmd = [
            "ffmpeg", "-ss", f"{offset:.3f}", "-i", str(self._path),
            "-af", self._atempo(self._speed),
            "-f", "s16le", "-ac", str(self.CHANNELS), "-ar", str(self.SAMPLE_RATE), "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._play_wall   = time.monotonic()
        self._play_offset = offset

        stop = self._stop_evt
        q    = self._queue
        proc = self._proc
        READ = self.BLOCKSIZE * self.CHANNELS * 2

        def _reader() -> None:
            while not stop.is_set():
                data = proc.stdout.read(READ)
                if not data:
                    break
                try:
                    q.put(data, timeout=1.0)
                except queue.Full:
                    pass

        self._reader = threading.Thread(target=_reader, daemon=True)
        self._reader.start()

        buf  = self._pcm_buf
        ch   = self.CHANNELS
        q_cb = self._queue

        def _callback(outdata: np.ndarray, frames: int, _t, _s) -> None:
            needed = frames * ch * 2
            while len(buf) < needed:
                try:
                    buf.extend(q_cb.get_nowait())
                except queue.Empty:
                    break
            if len(buf) >= needed:
                outdata[:] = np.frombuffer(bytes(buf[:needed]), dtype=np.int16).reshape(frames, ch)
                del buf[:needed]
            else:
                have = len(buf) // (ch * 2)
                if have:
                    outdata[:have] = np.frombuffer(
                        bytes(buf[:have * ch * 2]), dtype=np.int16).reshape(have, ch)
                outdata[have:] = 0
                buf.clear()

        self._stream = sd.OutputStream(
            samplerate=self.SAMPLE_RATE, channels=self.CHANNELS,
            dtype="int16", blocksize=self.BLOCKSIZE, callback=_callback,
        )
        self._stream.start()

    def _stop(self) -> None:
        self._stop_evt.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._reader and self._reader.is_alive():
            self._reader.join(timeout=1.0)
        self._stop_evt.clear()

    @staticmethod
    def _probe_duration(path: Path) -> float:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=10)
            return float(r.stdout.strip())
        except Exception:
            return 9999.0


# ─── segment data model ──────────────────────────────────────────────────────

class Segment:
    def __init__(self, raw: dict) -> None:
        self.start:   float       = raw.get("start", 0.0)
        self.end:     float       = raw.get("end", 0.0)
        self.text:    str         = raw.get("text", "")
        self.speaker: str         = raw.get("speaker", "")
        self.words:   list[dict]  = raw.get("words", [])
        self._undo:   str | None  = None   # single-level undo buffer

    def to_dict(self) -> dict:
        d: dict = {"start": self.start, "end": self.end, "text": self.text}
        if self.speaker:
            d["speaker"] = self.speaker
        if self.words:
            d["words"] = self.words
        return d

    def save_undo(self) -> None:
        self._undo = self.text

    def undo(self) -> bool:
        if self._undo is not None:
            self.text, self._undo = self._undo, None
            return True
        return False


# ─── vim state machine ────────────────────────────────────────────────────────

class VimState:
    NORMAL  = "NORMAL"
    INSERT  = "INSERT"
    SEARCH  = "SEARCH"
    PENDING = "PENDING"   # waiting for motion after operator (d/c)

    def __init__(self) -> None:
        self.mode:      str      = self.NORMAL
        self.col:       int      = 0         # char cursor within segment text
        self.count_buf: str      = ""        # numeric prefix accumulator
        self.operator:  str      = ""        # pending operator: d / c
        self.yank_reg:  str      = ""        # internal clipboard
        self.search_q:  str      = ""        # last search query
        self.search_buf: str     = ""        # in-progress search input

    def count(self, default: int = 1) -> int:
        try:
            return max(1, int(self.count_buf)) if self.count_buf else default
        except ValueError:
            return default

    def clear_count(self) -> None:
        self.count_buf = ""

    def clamp_col(self, text: str) -> None:
        self.col = max(0, min(self.col, max(0, len(text) - 1)))


# ─── main TUI widget ──────────────────────────────────────────────────────────

class TranscriptView(Widget):
    """Scrollable list of segments; handles all vim key processing."""

    COMPONENT_CLASSES = {"segment--active", "segment--normal"}
    DEFAULT_CSS = """
    TranscriptView {
        overflow-y: scroll;
        overflow-x: hidden;
        height: 1fr;
        padding: 0 1;
    }
    TranscriptView Label {
        width: 100%;
        text-wrap: wrap;
    }
    """

    # reactive index of the "cursor" segment
    cursor: reactive[int] = reactive(0, layout=True)

    def __init__(self, segments: list[Segment], **kwargs) -> None:
        super().__init__(**kwargs)
        self.segments   = segments
        self.vim        = VimState()
        self._dirty     = False
        self._search_matches: list[int] = []
        self._match_idx: int = 0
        self._active_word: int = -1   # word index within cursor segment during playback
        self._struct_undo: list = []   # undo records for structural ops (dd, enter-split)
        self._last_action: tuple | None = None   # for dot-repeat (.)
        self._in_dot_repeat: bool = False

    # ── rendering ────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Not used — we use compose children instead."""
        return ""

    def compose(self) -> ComposeResult:
        for i, seg in enumerate(self.segments):
            yield self._make_label(i)

    def _build_body(self, idx: int, blink_on: bool = True) -> str:
        """Return Rich-markup body string for a segment label."""
        seg  = self.segments[idx]
        text = seg.text
        vim  = self.vim

        if idx != self.cursor:
            return markup_escape(text)

        # Active word karaoke highlight (playback in NORMAL mode)
        if self._active_word >= 0 and vim.mode == VimState.NORMAL and seg.words:
            wi = self._active_word
            # Walk words in order to find each word's char position,
            # advancing the search offset so duplicate words resolve correctly.
            offset = 0
            wstart = -1
            wend   = -1
            for i, w in enumerate(seg.words):
                wtext = w.get("word", "")
                pos = text.find(wtext, offset) if wtext else -1
                if pos == -1:
                    continue
                if i == wi:
                    wstart = pos
                    wend   = pos + len(wtext)
                    break
                offset = pos + len(wtext)
            if wstart == -1:
                return markup_escape(text)
            return (markup_escape(text[:wstart])
                    + f"[bold underline]{markup_escape(text[wstart:wend])}[/bold underline]"
                    + markup_escape(text[wend:]))

        # INSERT mode — blinking underline cursor (no extra width)
        if vim.mode == VimState.INSERT:
            col = max(0, min(vim.col, len(text)))
            if blink_on:
                if col < len(text):
                    return (markup_escape(text[:col])
                            + f"[underline]{markup_escape(text[col])}[/underline]"
                            + markup_escape(text[col + 1:]))
                else:
                    return markup_escape(text) + "[underline] [/underline]"
            return markup_escape(text)

        # NORMAL mode — block cursor
        col = max(0, min(vim.col, max(0, len(text) - 1)))
        if text:
            char = markup_escape(text[col]) if col < len(text) else " "
            return (markup_escape(text[:col])
                    + f"[reverse]{char}[/reverse]"
                    + markup_escape(text[col+1:]))
        return ""

    def _make_label(self, idx: int) -> Label:
        seg   = self.segments[idx]
        spk   = seg.speaker or "?"
        color = speaker_color(spk)
        ts    = f"[dim]{fmt_time(seg.start)} → {fmt_time(seg.end)}[/dim]"
        tag   = f"[bold {color}]{spk}[/bold {color}]"
        body  = self._build_body(idx)
        bg    = "on dark_blue" if idx == self.cursor else ""
        line  = f"{ts}  {tag}  [{color} {bg}]{body}[/{color} {bg}]"
        return Label(line, id=f"seg_{idx}", markup=True)

    def _refresh_label(self, idx: int, blink_on: bool = True) -> None:
        try:
            lbl: Label = self.query_one(f"#seg_{idx}", Label)
            seg   = self.segments[idx]
            spk   = seg.speaker or "?"
            color = speaker_color(spk)
            ts    = f"[dim]{fmt_time(seg.start)} → {fmt_time(seg.end)}[/dim]"
            tag   = f"[bold {color}]{spk}[/bold {color}]"
            body  = self._build_body(idx, blink_on=blink_on)
            bg    = "on dark_blue" if idx == self.cursor else ""
            lbl.update(f"{ts}  {tag}  [{color} {bg}]{body}[/{color} {bg}]")
        except NoMatches:
            pass

    # ── cursor movement ───────────────────────────────────────────────────────

    def _move_segment(self, delta: int) -> None:
        old = self.cursor
        self.cursor = max(0, min(len(self.segments) - 1, self.cursor + delta))
        self.vim.col = 0
        self._active_word = -1
        self._refresh_label(old)
        self._refresh_label(self.cursor)
        self._scroll_to_cursor()

    def _scroll_to_cursor(self) -> None:
        try:
            lbl = self.query_one(f"#seg_{self.cursor}", Label)
            lbl.scroll_visible(animate=False)
        except NoMatches:
            pass

    # ── search ───────────────────────────────────────────────────────────────

    def _build_matches(self) -> None:
        q = self.vim.search_q.lower()
        self._search_matches = [
            i for i, s in enumerate(self.segments) if q in s.text.lower()
        ]

    def _goto_match(self, forward: bool) -> None:
        if not self._search_matches:
            return
        if forward:
            candidates = [i for i in self._search_matches if i > self.cursor]
            self._match_idx = (self._search_matches.index(candidates[0])
                               if candidates else 0)
        else:
            candidates = [i for i in self._search_matches if i < self.cursor]
            self._match_idx = (self._search_matches.index(candidates[-1])
                               if candidates else len(self._search_matches) - 1)
        old = self.cursor
        self.cursor = self._search_matches[self._match_idx]
        self.vim.col = 0
        self._refresh_label(old)
        self._refresh_label(self.cursor)
        self._scroll_to_cursor()

    # ── text editing helpers ──────────────────────────────────────────────────

    def _seg(self) -> Segment:
        return self.segments[self.cursor]

    def _word_at_col(self) -> tuple[int, int] | None:
        words = word_boundaries(self._seg().text)
        col   = self.vim.col
        for start, end in words:
            if start <= col < end:
                return start, end
        return None

    def _word_around_col(self) -> tuple[int, int] | None:
        """Include surrounding whitespace (daw behaviour)."""
        text  = self._seg().text
        words = word_boundaries(text)
        col   = self.vim.col
        for i, (start, end) in enumerate(words):
            if start <= col < end:
                # include trailing space, or leading space if last word
                ws_end = end
                while ws_end < len(text) and text[ws_end] == " ":
                    ws_end += 1
                if ws_end == end and start > 0:
                    ws_start = start
                    while ws_start > 0 and text[ws_start - 1] == " ":
                        ws_start -= 1
                    return ws_start, end
                return start, ws_end
        return None

    def _delete_range(self, start: int, end: int) -> None:
        seg = self._seg()
        seg.save_undo()
        self.vim.yank_reg = seg.text[start:end]
        seg.text = seg.text[:start] + seg.text[end:]
        self.vim.col = max(0, min(start, len(seg.text) - 1))
        self._dirty = True
        self._refresh_label(self.cursor)

    def _delete_segment(self, idx: int) -> None:
        if len(self.segments) <= 1:
            return
        self._struct_undo.append(("del_seg", idx, self.segments[idx], self.cursor))
        self.segments.pop(idx)
        self.cursor = max(0, min(self.cursor, len(self.segments) - 1))
        self._dirty = True
        self._rebuild_all_labels()

    def _rebuild_all_labels(self) -> None:
        """Full re-mount of all labels after structural changes (deferred to next cycle)."""
        self.call_after_refresh(self._async_rebuild)

    async def _async_rebuild(self) -> None:
        await self.query("Label").remove()
        for i in range(len(self.segments)):
            await self.mount(self._make_label(i))
        self._scroll_to_cursor()

    # ── key handling (Normal mode) ────────────────────────────────────────────

    def handle_key_normal(self, key: str) -> bool:
        """Return True if key was consumed."""
        vim  = self.vim
        seg  = self._seg()
        text = seg.text
        n    = vim.count()

        # ── numeric prefix ──────────────────────────────────────────────────
        if key.isdigit() and (key != "0" or vim.count_buf):
            vim.count_buf += key
            return True

        # ── pending operator + motion ────────────────────────────────────────
        if vim.operator:
            op = vim.operator
            vim.operator = ""
            vim.clear_count()
            consumed = self._handle_operator_motion(op, key, n)
            if consumed and not self._in_dot_repeat and not vim.operator and op[0] in ("d", "c"):
                self._last_action = ("op_motion", op, key)
            if consumed:
                return True
            # not a valid motion — cancel silently
            return True

        # ── mode transitions ─────────────────────────────────────────────────
        if key == "i":
            vim.mode = VimState.INSERT
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "a":
            vim.col = min(vim.col + 1, len(text))
            vim.mode = VimState.INSERT
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "/":
            vim.mode = VimState.SEARCH
            vim.search_buf = ""
            vim.clear_count()
            self.app.query_one("#status_bar", Label).update(
                "[bold yellow]SEARCH:[/bold yellow] ")
            return True

        # ── motion ───────────────────────────────────────────────────────────
        if key == "h":
            vim.col = max(0, vim.col - n)
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "l":
            vim.col = min(len(text) - 1, vim.col + n) if text else 0
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "j":
            self._move_segment(n)
            vim.clear_count()
            return True
        if key == "k":
            self._move_segment(-n)
            vim.clear_count()
            return True
        if key in ("0", "zero"):
            vim.col = 0
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key in ("$", "dollar_sign"):
            vim.col = max(0, len(text) - 1)
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "w":
            words = word_boundaries(text)
            ahead = [s for s, e in words if s > vim.col]
            for _ in range(n):
                if ahead:
                    vim.col = ahead.pop(0)
                else:
                    # no more words in this segment — cross to next
                    if self.cursor < len(self.segments) - 1:
                        old = self.cursor
                        self.cursor += 1
                        next_words = word_boundaries(self.segments[self.cursor].text)
                        vim.col = next_words[0][0] if next_words else 0
                        self._active_word = -1
                        self._refresh_label(old)
                        self._refresh_label(self.cursor)
                        self._scroll_to_cursor()
                        ahead = []  # consumed the cross; stop repeating
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "b":
            words = word_boundaries(text)
            before = [s for s, e in words if s < vim.col]
            if before:
                for _ in range(n):
                    if before:
                        vim.col = before.pop()
            else:
                # at beginning of segment — jump to last word of previous segment
                if self.cursor > 0:
                    old = self.cursor
                    self.cursor -= 1
                    prev_text = self.segments[self.cursor].text
                    prev_words = word_boundaries(prev_text)
                    vim.col = prev_words[-1][0] if prev_words else 0
                    self._refresh_label(old)
                    self._refresh_label(self.cursor)
                    self._scroll_to_cursor()
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "e":
            # move to last char of current or next word
            words = word_boundaries(text)
            ends  = [e - 1 for s, e in words if e - 1 > vim.col]
            for _ in range(n):
                if ends:
                    vim.col = ends.pop(0)
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True

        # ── search navigation ─────────────────────────────────────────────────
        if key == "n":
            self._goto_match(forward=True)
            vim.clear_count()
            return True
        if key == "N":
            self._goto_match(forward=False)
            vim.clear_count()
            return True

        # ── operators / two-key sequences ─────────────────────────────────────
        if key in ("d", "c", "g"):
            vim.operator = key
            vim.clear_count()
            return True

        # ── single-key edits ─────────────────────────────────────────────────
        if key == "x":
            if text and vim.col < len(text):
                seg.save_undo()
                vim.yank_reg = text[vim.col]
                seg.text = text[:vim.col] + text[vim.col + 1:]
                vim.col = max(0, min(vim.col, len(seg.text) - 1))
                self._dirty = True
                self._refresh_label(self.cursor)
                if not self._in_dot_repeat:
                    self._last_action = ("key", "x")
            vim.clear_count()
            return True
        if key == "s":
            if text and vim.col < len(text):
                seg.save_undo()
                vim.yank_reg = text[vim.col]
                seg.text = text[:vim.col] + text[vim.col + 1:]
                vim.col = max(0, min(vim.col, len(seg.text)))
                self._dirty = True
            vim.mode = VimState.INSERT
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "p":
            if vim.yank_reg:
                seg.save_undo()
                ins = vim.col + 1
                seg.text = seg.text[:ins] + vim.yank_reg + seg.text[ins:]
                vim.col = ins + len(vim.yank_reg) - 1
                self._dirty = True
                self._refresh_label(self.cursor)
            vim.clear_count()
            return True
        if key == "P":
            if vim.yank_reg:
                seg.save_undo()
                seg.text = seg.text[:vim.col] + vim.yank_reg + seg.text[vim.col:]
                vim.col = vim.col + len(vim.yank_reg) - 1
                self._dirty = True
                self._refresh_label(self.cursor)
            vim.clear_count()
            return True
        if key == "u":
            if self._struct_undo:
                record = self._struct_undo.pop()
                if record[0] == "del_seg":
                    _, idx, seg_obj, old_cursor = record
                    self.segments.insert(idx, seg_obj)
                    self.cursor = min(old_cursor, len(self.segments) - 1)
                    self._dirty = True
                    self._rebuild_all_labels()
                elif record[0] == "enter_split":
                    _, seg_idx, orig_text, orig_end, orig_col = record
                    if seg_idx + 1 < len(self.segments):
                        self.segments.pop(seg_idx + 1)
                    self.segments[seg_idx].text = orig_text
                    self.segments[seg_idx].end  = orig_end
                    self.cursor  = seg_idx
                    vim.col      = orig_col
                    self._dirty  = True
                    self._rebuild_all_labels()
            elif seg.undo():
                vim.col = 0
                self._dirty = True
                self._refresh_label(self.cursor)
            vim.clear_count()
            return True
        if key in (".", "period"):
            if self._last_action and not self._in_dot_repeat:
                self._in_dot_repeat = True
                la = self._last_action
                try:
                    if la[0] == "key":
                        self.handle_key_normal(la[1])
                    elif la[0] == "op_motion":
                        # For two-char ops (da/ca/di/ci), call directly
                        self._handle_operator_motion(la[1], la[2], 1)
                finally:
                    self._in_dot_repeat = False
            vim.clear_count()
            return True
        # D — delete from cursor to end of segment
        if key == "D":
            if vim.col < len(text):
                seg.save_undo()
                vim.yank_reg = text[vim.col:]
                seg.text = text[:vim.col]
                self._dirty = True
                self._refresh_label(self.cursor)
                if not self._in_dot_repeat:
                    self._last_action = ("key", "D")
            vim.clear_count()
            return True
        # C — delete from cursor to end, enter INSERT mode
        if key == "C":
            seg.save_undo()
            vim.yank_reg = text[vim.col:]
            seg.text = text[:vim.col]
            vim.mode = VimState.INSERT
            self._dirty = True
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "o":
            # go to beginning of next segment in INSERT mode
            target = min(self.cursor + 1, len(self.segments) - 1)
            if target != self.cursor:
                old = self.cursor
                self.cursor = target
                self._refresh_label(old)
            vim.col = 0
            vim.mode = VimState.INSERT
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "O":
            # go to end of previous segment in INSERT mode
            target = max(self.cursor - 1, 0)
            if target != self.cursor:
                old = self.cursor
                self.cursor = target
                self._refresh_label(old)
            vim.col = len(self.segments[self.cursor].text)
            vim.mode = VimState.INSERT
            vim.clear_count()
            self._refresh_label(self.cursor)
            return True
        if key == "tab":
            self._cycle_speaker()
            if not self._in_dot_repeat:
                self._last_action = ("key", "tab")
            vim.clear_count()
            return True

        vim.clear_count()
        return False

    def _handle_operator_motion(self, op: str, motion: str, n: int) -> bool:
        vim  = self.vim
        seg  = self._seg()

        # dd — delete whole segment
        if op == "d" and motion == "d":
            self._delete_segment(self.cursor)
            return True

        # dj — delete segment below
        if op == "d" and motion == "j":
            target = min(self.cursor + 1, len(self.segments) - 1)
            if target != self.cursor:
                self._delete_segment(target)
            return True

        # dk — delete segment above
        if op == "d" and motion == "k":
            target = max(self.cursor - 1, 0)
            if target != self.cursor:
                self._delete_segment(target)
            return True

        # dw — delete from cursor to start of next word (including spaces)
        if motion == "w":
            text  = seg.text
            col   = vim.col
            words = word_boundaries(text)
            nexts = [s for s, e in words if s > col]
            if nexts:
                end = nexts[0]          # start of next word
            else:
                end = len(text)         # to end of line
            self._delete_range(col, end)
            if op == "c":
                vim.mode = VimState.INSERT
            return True

        # ge — move to end of previous word (motion only, works with g operator)
        if op == "g" and motion == "e":
            words = word_boundaries(seg.text)
            ends  = [e - 1 for s, e in words if e - 1 < vim.col]
            if ends:
                vim.col = ends[-1]
            self._refresh_label(self.cursor)
            return True

        # g + anything else — treat as seek-to-segment-start (app fallback)
        if op == "g":
            # return False so the app-level handler never fires; just ignore
            return True

        # db — delete from start of current/previous word back to cursor
        if motion == "b":
            text  = seg.text
            col   = vim.col
            words = word_boundaries(text)
            # find the word that starts at or before col
            candidates = [s for s, e in words if s <= col]
            if candidates:
                start = candidates[-1]
                if start == col:
                    # cursor is exactly at a word start — go one word further back
                    start = candidates[-2] if len(candidates) >= 2 else 0
            else:
                start = 0
            self._delete_range(start, col)
            if op == "c":
                vim.mode = VimState.INSERT
            return True

        # daw / caw
        if motion == "a":
            # consume next char: w / s
            # We can't peek ahead here — handle via double-char sequences below
            # Treat 'a' as trigger and wait for next key in a temporary state
            vim.operator = op + "a"
            return True

        # second char of daw/caw/das/cas — called when operator == "da"/"ca"/"di"/"ci"
        if op in ("da", "ca", "di", "ci"):
            real_op = op[0]
            style   = op[1]   # 'a' = around, 'i' = inside
            if motion == "w":
                wb = (self._word_around_col() if style == "a"
                      else self._word_at_col())
                if wb:
                    self._delete_range(*wb)
                if real_op == "c":
                    vim.mode = VimState.INSERT
                return True
            if motion == "s":
                start, end = sentence_boundaries(seg.text)
                if style == "a":
                    self._delete_range(0, len(seg.text))
                else:
                    self._delete_range(start, end)
                if real_op == "c":
                    vim.col = start
                    vim.mode = VimState.INSERT
                return True

        # di / ci (inside)
        if motion == "i":
            vim.operator = op + "i"
            return True

        # das / dis — direct shortcuts
        if op == "d" and motion == "s":
            self._delete_range(*sentence_boundaries(seg.text))
            return True

        return False

    # ── key handling (Insert mode) ────────────────────────────────────────────

    def handle_key_insert(self, key: str, char: str = "") -> bool:
        vim = self.vim
        seg = self._seg()
        # Use the actual character for printable input; fall back to key for
        # named keys (escape, backspace, ctrl+*, etc.)
        ch = char if (char and char.isprintable() and len(char) == 1) else None
        if key == "backslash":
            return True  # muted — too close to ' on keyboard, crashes markup
        if key in ("escape", "ctrl+c"):
            vim.mode = VimState.NORMAL
            vim.col  = max(0, min(vim.col, len(seg.text) - 1))
            self._refresh_label(self.cursor)
            return True
        if key == "backspace":
            if vim.col > 0:
                seg.text = seg.text[:vim.col - 1] + seg.text[vim.col:]
                vim.col -= 1
                self._dirty = True
                self._refresh_label(self.cursor)
            return True
        if key == "ctrl+w":  # delete previous word (ctrl+backspace sends same code as backspace in most terminals)
            # delete from start of current/previous word up to cursor
            if vim.col > 0:
                words = word_boundaries(seg.text)
                candidates = [s for s, e in words if s < vim.col]
                start = candidates[-1] if candidates else 0
                seg.save_undo()
                seg.text = seg.text[:start] + seg.text[vim.col:]
                vim.col = start
                self._dirty = True
                self._refresh_label(self.cursor)
            return True
        if key == "delete":
            if vim.col < len(seg.text):
                seg.text = seg.text[:vim.col] + seg.text[vim.col + 1:]
                self._dirty = True
                self._refresh_label(self.cursor)
            return True
        if key == "ctrl+delete":
            # delete from cursor to start of next word (including spaces)
            if vim.col < len(seg.text):
                words = word_boundaries(seg.text)
                nexts = [s for s, e in words if s > vim.col]
                end = nexts[0] if nexts else len(seg.text)
                seg.save_undo()
                seg.text = seg.text[:vim.col] + seg.text[end:]
                self._dirty = True
                self._refresh_label(self.cursor)
            return True
        if key == "left":
            vim.col = max(0, vim.col - 1)
            self._refresh_label(self.cursor)
            return True
        if key == "right":
            vim.col = min(len(seg.text), vim.col + 1)
            self._refresh_label(self.cursor)
            return True
        if key == "up":
            if self.cursor > 0:
                old = self.cursor
                self.cursor -= 1
                vim.col = min(vim.col, len(self.segments[self.cursor].text))
                self._refresh_label(old)
                self._refresh_label(self.cursor)
                self._scroll_to_cursor()
            return True
        if key == "down":
            if self.cursor < len(self.segments) - 1:
                old = self.cursor
                self.cursor += 1
                vim.col = min(vim.col, len(self.segments[self.cursor].text))
                self._refresh_label(old)
                self._refresh_label(self.cursor)
                self._scroll_to_cursor()
            return True
        if key == "tab":
            self._cycle_speaker()
            return True
        # printable character insertion
        if ch is not None:
            seg.text = seg.text[:vim.col] + ch + seg.text[vim.col:]
            vim.col += 1
            self._dirty = True
            self._refresh_label(self.cursor)
            return True
        if key == "enter":
            # split segment at cursor: left part stays, right part becomes new segment below
            left  = seg.text[:vim.col].rstrip()
            right = seg.text[vim.col:].lstrip()
            self._struct_undo.append(("enter_split", self.cursor, seg.text, seg.end, vim.col))
            seg.text = left
            # interpolate timestamps: midpoint
            mid = (seg.start + seg.end) / 2
            new_seg = Segment({
                "start":   mid,
                "end":     seg.end,
                "text":    right,
                "speaker": seg.speaker,
            })
            seg.end = mid
            self.segments.insert(self.cursor + 1, new_seg)
            self.cursor += 1
            vim.col = 0
            self._dirty = True
            self._rebuild_all_labels()
            return True
        return False

    # ── speaker cycling ───────────────────────────────────────────────────────

    def _cycle_speaker(self) -> None:
        """Rotate the speaker of the cursor segment through all speakers in the file."""
        speakers = list(dict.fromkeys(
            s.speaker for s in self.segments if s.speaker
        ))
        if not speakers:
            return
        seg = self._seg()
        cur = seg.speaker
        if cur in speakers:
            seg.speaker = speakers[(speakers.index(cur) + 1) % len(speakers)]
        else:
            seg.speaker = speakers[0]
        self._dirty = True
        self._refresh_label(self.cursor)

    # ── key handling (Search mode) ────────────────────────────────────────────

    def handle_key_search(self, key: str) -> bool:
        vim = self.vim
        if key == "escape":
            vim.mode = VimState.NORMAL
            vim.search_buf = ""
            try:
                self.app.query_one("#status_bar", Label).update("")
            except NoMatches:
                pass
            return True
        if key == "enter":
            vim.search_q  = vim.search_buf
            vim.search_buf = ""
            vim.mode = VimState.NORMAL
            self._build_matches()
            self._goto_match(forward=True)
            try:
                self.app.query_one("#status_bar", Label).update(
                    f"[dim]/{vim.search_q}[/dim]")
            except NoMatches:
                pass
            return True
        if key == "backspace":
            vim.search_buf = vim.search_buf[:-1]
        elif len(key) == 1:
            vim.search_buf += key
        try:
            self.app.query_one("#status_bar", Label).update(
                f"[bold yellow]SEARCH:[/bold yellow] {vim.search_buf}")
        except NoMatches:
            pass
        return True

    # ── unified key handler (called by app) ──────────────────────────────────

    def process_key(self, key: str, char: str = "") -> bool:
        mode = self.vim.mode
        if mode == VimState.INSERT:
            return self.handle_key_insert(key, char)
        if mode == VimState.SEARCH:
            return self.handle_key_search(key)
        # NORMAL or PENDING (operator accumulated in operator field)
        return self.handle_key_normal(key)

    # ── external sync (called by timer) ──────────────────────────────────────

    def sync_to_position(self, pos: float) -> None:
        """Highlight the segment (and word) whose time range contains pos."""
        for i, seg in enumerate(self.segments):
            if seg.start <= pos < seg.end:
                seg_changed = (i != self.cursor)
                if seg_changed:
                    old = self.cursor
                    self.cursor = i
                    self.vim.col = 0
                    self._active_word = -1
                    self._refresh_label(old)
                    self._scroll_to_cursor()
                # find active word within segment
                new_word = -1
                for wi, w in enumerate(seg.words):
                    ws = w.get("start", seg.start)
                    we = w.get("end",   seg.end)
                    if ws <= pos < we:
                        new_word = wi
                        break
                    elif pos < ws:
                        # between words — keep previous
                        new_word = max(wi - 1, 0)
                        break
                else:
                    if seg.words:
                        new_word = len(seg.words) - 1
                if seg_changed or new_word != self._active_word:
                    self._active_word = new_word
                    self._refresh_label(self.cursor)
                return


# ─── selector screens ───────────────────────────────────────────────────────

SELECTOR_CSS = """
Screen {
    align: center middle;
}
#selector_box {
    width: 60;
    height: auto;
    max-height: 80vh;
    border: double $primary;
    padding: 1 2;
}
#selector_title {
    text-align: center;
    text-style: bold;
    color: $accent;
    margin-bottom: 1;
}
#selector_hint {
    text-align: center;
    color: $text-muted;
    margin-top: 1;
}
ListView {
    height: auto;
    max-height: 60vh;
    border: solid $panel;
}
ListItem {
    padding: 0 1;
}
ListItem:hover {
    background: $primary 30%;
}
ListItem.--highlight {
    background: $primary 50%;
    color: $text;
}
"""


class ParticipantScreen(Screen):
    """Step 1 — choose a participant folder."""

    CSS = SELECTOR_CSS

    def __init__(self, base: Path) -> None:
        super().__init__()
        self.base = base
        self._dirs = sorted(
            p for p in base.iterdir()
            if p.is_dir() and re.match(r"P\d+$", p.name)
            and any(p.glob("*itw*.m4a"))   # only dirs that have audio
        )

    def compose(self) -> ComposeResult:
        with Static(id="selector_box"):
            yield Label("Select participant", id="selector_title")
            yield ListView(
                *[ListItem(Label(d.name), id=f"p_{d.name}") for d in self._dirs],
                id="lv"
            )
            yield Label("↑↓ navigate  Enter select  Esc quit", id="selector_hint")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # id is "p_P02", "p_P03", etc.
        name = event.item.id[2:]   # strip leading "p_"
        chosen = next((d for d in self._dirs if d.name == name), None)
        if chosen:
            self.app.push_screen(FileScreen(chosen))

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.app.exit()
            event.stop()


class FileScreen(Screen):
    """Step 2 — choose an interview file within the selected participant folder."""

    CSS = SELECTOR_CSS

    def __init__(self, participant_dir: Path) -> None:
        super().__init__()
        self.participant_dir = participant_dir
        # Only show files that have BOTH .m4a and _transcript.json
        self._pairs: list[tuple[Path, Path]] = []
        for m4a in sorted(participant_dir.glob("*itw*.m4a")):
            json_p = participant_dir / f"{m4a.stem}_transcript.json"
            self._pairs.append((m4a, json_p))

    def compose(self) -> ComposeResult:
        with Static(id="selector_box"):
            yield Label(
                f"[bold]{self.participant_dir.name}[/bold] — select interview",
                id="selector_title", markup=True
            )
            items = []
            for m4a, json_p in self._pairs:
                has_json = "✓" if json_p.exists() else "[red]✗ no transcript[/red]"
                size_mb  = m4a.stat().st_size / 1_048_576
                items.append(ListItem(
                    Label(f"{m4a.name}  [dim]{size_mb:.1f} MB[/dim]  {has_json}",
                          markup=True),
                    id=f"f_{self._pairs.index((m4a, json_p))}"
                ))
            yield ListView(*items, id="lv")
            yield Label("↑↓ navigate  Enter select  Esc back", id="selector_hint")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = int(event.item.id.split("_", 1)[1])
        m4a, json_p = self._pairs[idx]
        if not json_p.exists():
            self.app.query_one("#selector_hint", Label).update(
                "[bold red]No transcript yet — run transcription.py first[/bold red]"
            )
            return
        # Replace this selector app's result and switch to editor
        self.app._launch_editor(m4a, json_p)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.app.pop_screen()
            event.stop()


# ─── launcher app (wraps selector + editor) ──────────────────────────────────

class LauncherApp(App):
    """Thin shell that hosts the selector screens, then swaps to CrosscheckApp."""

    def __init__(self, base: Path) -> None:
        super().__init__()
        self.base = base

    def on_mount(self) -> None:
        self.push_screen(ParticipantScreen(self.base))

    def _launch_editor(self, audio_path: Path, json_path: Path) -> None:
        self.exit(result=(audio_path, json_path))


# ─── main app ─────────────────────────────────────────────────────────────────

class CrosscheckApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #toolbar {
        height: 1;
        background: $panel;
        padding: 0 1;
        color: $text;
    }
    #status_bar {
        height: 1;
        background: $panel-darken-2;
        padding: 0 1;
        color: $text;
    }
    TranscriptView {
        border: solid $primary;
    }
    """

    BINDINGS = [
        Binding("ctrl+s",      "save",  "Save",  show=True),
        # ctrl+delete / ctrl+w routed to insert-mode word-delete via on_key
        Binding("ctrl+delete", "noop",  show=False, priority=True),
        Binding("ctrl+w",      "noop",  show=False, priority=True),
        # tab is handled in on_key (prevent_default stops focus-cycling)
        # 'q' is handled manually in on_key (only quits in NORMAL mode)
        # Override Textual's built-in ctrl+q (would show a confirm dialog
        # and conflicts with VS Code's own ctrl+q shortcut).
        Binding("ctrl+q",      "noop",  show=False),
    ]

    def action_noop(self) -> None:  # noqa: D102
        pass

    def __init__(self, audio_path: Path, json_path: Path) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.json_path  = json_path
        self._meta: dict = {}

        with json_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        self._meta     = {k: v for k, v in raw.items() if k != "segments"}
        self.segments  = [Segment(s) for s in raw.get("segments", [])]

        self.player    = AudioPlayer(audio_path)
        self._sync_running = False
        self._blink_on: bool = True

    # ── layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Label("", id="toolbar")
        yield TranscriptView(self.segments, id="tv")
        yield Label("", id="status_bar")

    def on_mount(self) -> None:
        self._update_toolbar()
        self.set_interval(TICK_MS / 1000, self._tick)

    # ── timer ────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        pos = self.player.position
        tv: TranscriptView = self.query_one("#tv", TranscriptView)
        if not self.player.paused and tv.vim.mode == VimState.NORMAL:
            tv.sync_to_position(pos)
        # blink the insert cursor
        if tv.vim.mode == VimState.INSERT:
            self._blink_on = not self._blink_on
            tv._refresh_label(tv.cursor, blink_on=self._blink_on)
        else:
            self._blink_on = True
        self._update_toolbar()

    # ── toolbar ──────────────────────────────────────────────────────────────

    def _update_toolbar(self) -> None:
        try:
            lbl: Label = self.query_one("#toolbar", Label)
            tv: TranscriptView = self.query_one("#tv", TranscriptView)
        except NoMatches:
            return
        pos      = self.player.position
        dur      = self.player.duration
        speed    = self.player.speed
        mode     = tv.vim.mode
        paused   = "⏸ PAUSED" if self.player.paused else "▶ PLAYING"
        modified = "[bold red]*[/bold red]" if tv._dirty else ""
        lbl.update(
            f"{modified}[bold]{paused}[/bold]  "
            f"{fmt_time(pos)} / {fmt_time(dur)}  "
            f"[cyan]speed {speed:.2f}x[/cyan]  "
            f"[yellow]{mode}[/yellow]  "
            f"[dim]{self.audio_path.name}[/dim]"
        )

    # ── keyboard ─────────────────────────────────────────────────────────────

    def on_key(self, event) -> None:
        key = event.key
        tv: TranscriptView = self.query_one("#tv", TranscriptView)

        # Global keys (work in any vim mode)
        if key == "space":
            if tv.vim.mode == VimState.INSERT:
                # type a space into the text
                tv.process_key(" ", " ")
                event.stop()
                return
            if self.player.paused:
                # Resume from cursor position: use word timestamp if available
                seg  = self.segments[tv.cursor]
                col  = tv.vim.col
                seek_t = seg.start
                if seg.words:
                    # find the word whose char range covers vim.col
                    offset = 0
                    for w in seg.words:
                        wtext = w.get("word", "")
                        wstart_char = seg.text.find(wtext, offset)
                        if wstart_char == -1:
                            offset += len(wtext)
                            continue
                        wend_char = wstart_char + len(wtext)
                        if wstart_char <= col < wend_char or col < wstart_char:
                            seek_t = w.get("start", seg.start)
                            break
                        offset = wend_char
                    else:
                        # col is past all words — use last word start
                        seek_t = seg.words[-1].get("start", seg.start)
                self.player.seek(seek_t)
                self.player.play()
            else:
                self.player.pause()
                # move block cursor to the active (last spoken) word
                seg = self.segments[tv.cursor]
                wi  = tv._active_word
                if wi >= 0 and wi < len(seg.words):
                    # walk in order so duplicate words resolve to correct occurrence
                    off = 0
                    for i, w in enumerate(seg.words):
                        wtext = w.get("word", "")
                        pos = seg.text.find(wtext, off) if wtext else -1
                        if pos == -1:
                            continue
                        if i == wi:
                            tv.vim.col = pos
                            break
                        off = pos + len(wtext)
                tv._active_word = -1
                tv._refresh_label(tv.cursor)
            event.stop()
            return
        if key == "ctrl+s":
            self.action_save()
            event.stop()
            return
        if key == "q" and tv.vim.mode == VimState.NORMAL:
            self.action_quit_save()
            event.stop()
            return
        if key in ("]", "right_square_bracket") and tv.vim.mode != VimState.INSERT:
            self.player.set_speed(self.player.speed + SPEED_STEP)
            event.stop()
            return
        if key in ("[", "left_square_bracket") and tv.vim.mode != VimState.INSERT:
            self.player.set_speed(self.player.speed - SPEED_STEP)
            event.stop()
            return
        if key == "r" and tv.vim.mode == VimState.NORMAL:
            seg = tv.segments[tv.cursor]
            self.player.seek(seg.start)
            event.stop()
            return
        # tab cycles speaker (prevent_default stops Textual's focus-cycling)
        if key == "tab":
            event.prevent_default()
            tv._cycle_speaker()
            event.stop()
            return
        # 'g' is now handled inside handle_key_normal (ge, etc.)

        consumed = tv.process_key(key, event.character or "")
        if consumed:
            event.stop()

    # ── actions ──────────────────────────────────────────────────────────────

    def action_save(self) -> None:
        self._write_json()
        self._write_txt()
        tv: TranscriptView = self.query_one("#tv", TranscriptView)
        tv._dirty = False
        try:
            self.query_one("#status_bar", Label).update(
                f"[green]Saved → {self.json_path.name}[/green]")
        except NoMatches:
            pass

    def action_quit_save(self) -> None:
        tv: TranscriptView = self.query_one("#tv", TranscriptView)
        if tv._dirty:
            self._write_json()
            self._write_txt()
        self.exit()

    # ── persistence ──────────────────────────────────────────────────────────

    def _write_json(self) -> None:
        payload = dict(self._meta)
        payload["segments"] = [s.to_dict() for s in self.segments]
        payload["last_edited"] = datetime.now(timezone.utc).isoformat()
        with self.json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _write_txt(self) -> None:
        txt_path = self.json_path.with_name(
            self.json_path.stem.replace("_transcript", "") + "_transcript.txt"
        )
        with txt_path.open("w", encoding="utf-8") as fh:
            current_speaker = None
            for seg in self.segments:
                speaker = seg.speaker or "UNKNOWN"
                text    = seg.text.strip()
                if not text:
                    continue
                if speaker != current_speaker:
                    fh.write(f"\n[{fmt_time(seg.start)} → {fmt_time(seg.end)}]  {speaker}\n")
                    current_speaker = speaker
                fh.write(f"{text}\n")


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio-synchronized transcript editor with vim motions.")
    parser.add_argument("audio", nargs="?", default=None,
                        help="Path to .m4a audio file. Omit to use interactive selector.")
    parser.add_argument("--json", default=None,
                        help="Path to transcript JSON (default: <audio_stem>_transcript.json)")
    parser.add_argument("--base", default=None,
                        help="Base HITLS directory for the interactive selector "
                             "(default: folder containing this script)")
    args = parser.parse_args()

    if args.audio:
        # Direct mode — open a specific file
        audio_path = Path(args.audio).resolve()
        if not audio_path.exists():
            print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            json_path = Path(args.json).resolve()
        else:
            json_path = audio_path.parent / f"{audio_path.stem}_transcript.json"
        if not json_path.exists():
            print(f"Error: transcript JSON not found: {json_path}", file=sys.stderr)
            print("Run transcription.py first, or pass --json explicitly.", file=sys.stderr)
            sys.exit(1)
        _flush_kbd()
        CrosscheckApp(audio_path, json_path).run()
    else:
        # Interactive selector mode
        base = Path(args.base).resolve() if args.base else Path(__file__).parent
        if not base.exists():
            print(f"Error: base directory not found: {base}", file=sys.stderr)
            sys.exit(1)
        _flush_kbd()
        launcher = LauncherApp(base)
        result = launcher.run()
        if result:
            audio_path, json_path = result
            _flush_kbd()
            CrosscheckApp(audio_path, json_path).run()


if __name__ == "__main__":
    main()
