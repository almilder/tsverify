"""Spectrometer descriptions used by the case catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    """One spectrometer channel: pixel centres, IRF width, fit ranges and an optional notch."""

    channel: str
    lam_range: tuple[float, float]
    pixels: int
    points_per_pixel: int
    sigma: float
    amp_ranges: tuple[tuple[float, float], ...]
    notch: tuple[float, float, float] | None = None
    amplitude: float = 100.0
