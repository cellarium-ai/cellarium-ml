# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Progress lines for the long phases of a conversion: what is running, how far along it is and how long it will take.

A :class:`Progress` reports on one phase at a time (scanning the files, writing the tiles, ...). The code that does the work
calls :meth:`Progress.advance`; a background thread prints a status line every ``interval`` seconds (the first one after ten
seconds), so there is output even when nothing completes for a while. The rate, and from it the time left, is measured over the
last few minutes, so it follows the speed the phase has now and not the average since it started.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

DEFAULT_INTERVAL = 60.0
_FIRST_LINE_AFTER = 10.0  # seconds into a phase: the first status line, so that there is a sign of life soon


def say_flushed(message: str) -> None:
    """``print`` that flushes, so that the lines show up at once when the output goes to a file or a pipe."""
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    """``h:mm:ss``."""
    s = int(round(max(seconds, 0)))
    hours, rest = divmod(s, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _format_rate(rate: float) -> str:
    return f"{rate:.0f}" if rate >= 10 else f"{rate:.1f}" if rate >= 1 else f"{rate:.2f}"


class Progress:
    """Status lines for one phase at a time, through ``say`` (a function that takes a line of text).

    Args:
        say: Where the lines go. ``None`` makes everything silent (and starts no thread).
        interval: Seconds between status lines; ``0`` turns the periodic lines off (the first and last line of each phase are
            still printed).
        clock: The time source (seconds), for tests.
    """

    def __init__(
        self,
        say: Callable[[str], None] | None,
        interval: float = DEFAULT_INTERVAL,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval < 0:
            raise ValueError("the progress interval must not be negative")
        self._say = say
        self.interval = interval
        self._clock = clock
        self._lock = threading.RLock()
        self._name: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._next_line = float("inf")

    # --- the phases ---

    def start(
        self,
        name: str,
        total: int | None = None,
        unit: str = "items",
        *,
        header: str | None = None,
        bytes_label: str | None = None,
        detail: Callable[[], str] | None = None,
    ) -> None:
        """Begin a phase and print its first line (``header``, or ``name`` followed by an ellipsis).

        Args:
            name: The phase, at the start of every line.
            total: How many ``unit`` s there are to do, if known (gives the percentage and the time left).
            bytes_label: If given, :meth:`advance` is also told the bytes done, and the lines show the rate (``MB/s <label>``).
            detail: Called for each status line; what it returns is appended (for state that the phase knows about).
        """
        with self._lock:
            self._name, self._total, self._unit = name, total, unit
            self._bytes_label, self._detail = bytes_label, detail
            self._done, self._bytes = 0, 0
            self._t0 = self._clock()
            self._window = max(300.0, 5 * self.interval)  # seconds over which the rate is measured
            self._samples: deque[tuple[float, int, int]] = deque([(self._t0, 0, 0)])
            self._next_line = self._t0 + min(_FIRST_LINE_AFTER, self.interval) if self.interval > 0 else float("inf")
            self._emit(header if header is not None else f"{name} ...")
            self._ensure_thread()

    def advance(self, n: int = 1, nbytes: int = 0) -> None:
        """``n`` more ``unit`` s are done (and ``nbytes`` more bytes, for a phase with ``bytes_label``)."""
        with self._lock:
            if self._name is None:
                return
            self._done += n
            self._bytes += nbytes
            now = self._clock()
            self._samples.append((now, self._done, self._bytes))
            while len(self._samples) > 2 and now - self._samples[1][0] > self._window:
                self._samples.popleft()

    def finish(self, summary: str = "") -> float:
        """End the phase with a line saying what was done and how long it took. Returns the seconds it took."""
        with self._lock:
            if self._name is None:
                return 0.0
            elapsed = self._clock() - self._t0
            what = f"{self._done:,} {self._unit} in " if self._total is not None else "done in "
            self._emit(f"{self._name}: {what}{format_duration(elapsed)}{summary}")
            self._name = None
            self._next_line = float("inf")
            return elapsed

    def close(self) -> None:
        """Stop the timer thread (the current phase, if any, is left without a closing line)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._thread = None
        with self._lock:
            self._name = None
            self._next_line = float("inf")

    # --- the status lines ---

    def status(self) -> str | None:
        """The status line for the phase that is running now (``None`` if there is none)."""
        with self._lock:
            if self._name is None:
                return None
            now = self._clock()
            elapsed = now - self._t0
            first = self._samples[0]
            span = now - first[0]
            rate = (self._done - first[1]) / span if span >= 1 else None
            parts = []
            if self._total is not None:
                parts.append(
                    f"{self._done:,}/{self._total:,} {self._unit} ({100 * self._done / max(self._total, 1):.0f}%)"
                )
            elif self._done:
                parts.append(f"{self._done:,} {self._unit}")
            else:
                parts.append("still running")
            if rate is not None and self._total is not None and self._done:
                parts.append(f"{_format_rate(rate)} {self._unit}/s")
            if self._bytes_label and span >= 1:
                parts.append(f"{(self._bytes - first[2]) / span / 1e6:.1f} MB/s {self._bytes_label}")
            parts.append(f"elapsed {format_duration(elapsed)}")
            if self._total is not None and rate:
                parts.append(f"ETA {format_duration((self._total - self._done) / rate)}")
            elif self._total is not None:
                parts.append("ETA unknown yet")
            line = f"{self._name}: " + ", ".join(parts)
            if self._detail is not None:
                extra = self._detail()
                if extra:
                    line += f" | {extra}"
            return line

    def tick(self) -> None:
        """Print the status line now (the timer thread calls this when one is due)."""
        with self._lock:  # held while printing, so that a line never comes after the closing line of its phase
            line = self.status()
            if line is not None:
                self._emit(line)
            self._next_line = self._clock() + self.interval if self.interval > 0 else float("inf")

    # --- internals ---

    def _emit(self, line: str) -> None:
        if self._say is not None:
            self._say(line)

    def _ensure_thread(self) -> None:
        if self._say is None or self.interval <= 0 or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="deltacells-progress", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        poll = min(0.5, self.interval / 4)
        while not self._stop.wait(poll):
            with self._lock:
                due = self._name is not None and self._clock() >= self._next_line
            if due:
                try:
                    self.tick()
                except Exception:  # a failing log function must not kill the conversion (or this thread's usefulness)
                    pass
