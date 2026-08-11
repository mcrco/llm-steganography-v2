"""Tiny fixed buffer set for smoke evals before a larger corpus lands."""

from __future__ import annotations

# Short, digit-free sentences; each paired with both bits at load time.
SMOKE_BUFFERS: tuple[str, ...] = (
    "The cat sat on the mat.",
    "Rain fell softly on the roof.",
    "She opened the window for fresh air.",
    "The library was quiet after lunch.",
    "Birds gathered near the empty fountain.",
    "He left the note on the kitchen table.",
    "Snow covered the path before dawn.",
    "The river moved slowly past the mill.",
)
