"""Spectrometer descriptions used by the case catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    """One spectrometer channel: pixel centres, Gaussian IRF width, fit ranges, an optional notch
    ``(center, width, optical_density)`` and the windows in which the conformance tests compare peak positions.
    ``channel`` labels the tolerances that apply; ``points_per_pixel`` is a sampling hint for codes that
    oversample the pixel axis."""

    channel: str
    lam_range: tuple[float, float]
    pixels: int
    points_per_pixel: int
    sigma: float
    amp_ranges: tuple[tuple[float, float], ...]
    notch: tuple[float, float, float] | None = None
    amplitude: float = 100.0
    peak_windows: tuple[tuple[float, float], ...] = ()
