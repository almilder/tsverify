"""Tolerance tiers for the conformance tests. ``REFERENCE`` holds the values tsadar meets (tsadar PR 164)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tolerances:
    spectral_density: dict = field(
        default_factory=lambda: {"A1": (1.0e-3, 5.0e-4), "A3": (2.5e-3, 1.5e-3), "A5": (1.0e-3, 5.0e-4)}
    )
    detector_linf: dict = field(default_factory=lambda: {"EPW": 1.5e-3, "IAW": 1.0e-3})
    detector_l1: dict = field(default_factory=lambda: {"EPW": 1.0e-3, "IAW": 1.0e-3})
    peak_nm: dict = field(default_factory=lambda: {"EPW": 5.0e-3, "IAW": 5.0e-4})
    zprime_abs: float = 2.5e-4
    high_z_linf: float = 1.0e-2
    high_z_peak_nm: float = 5.0e-5
    sweep_linf: float = 2.0e-3
    relativistic_shape: float = 3.0
    relativistic_peak: float = 1.2
    fit_rtol: float = 1.0e-3
    fit_atol: dict = field(default_factory=lambda: {"Va": 1.0e-3, "ud": 1.0e-2})
    cramer_rao_self: float = 0.01
    cramer_rao_oracle: float = 0.05
    zero_gradient: float = 1.0e-12
    rewritten_plasma: float = 1.0e-10


REFERENCE = Tolerances()
