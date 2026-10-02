"""Time formatting shared by the app and the CLI."""
from __future__ import annotations


def clock_text(seconds: float) -> str:
    """hh:mm:ss.xxx (milliseconds rounded, so 59.9996 s carries into the next minute)."""
    ms = int(round(abs(seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{'-' if seconds < 0 and (h or m or s or ms) else ''}{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
