"""Detector-depth comparison of a code's spectrum with the oracle's."""

import numpy as np

from . import oracle
from .tolerances import REFERENCE

NOTCH_EDGE_SIGMAS = 2.0


def oracle_detector(det, centers, spectrum):
    """The analytic detector spectrum of ``spectrum`` (a callable of wavelength in nm) on ``det``."""
    transmission = oracle.rectangular_notch(*det.notch) if det.notch else None
    values = oracle.detector_spectrum(spectrum, centers, det.sigma, transmission=transmission)
    return oracle.normalize_in_ranges(centers, values, det.amp_ranges, det.amplitude)


def compared_pixels(det, centers):
    """Pixels inside the fit ranges, excluding NOTCH_EDGE_SIGMAS IRF widths around each notch edge."""
    mask = np.zeros(centers.shape, dtype=bool)
    for low, high in det.amp_ranges:
        mask |= (centers > low) & (centers < high)
    if det.notch:
        center, width, _ = det.notch
        for edge in (center - width / 2, center + width / 2):
            mask &= np.abs(centers - edge) > NOTCH_EDGE_SIGMAS * det.sigma
    return mask


def assert_detector_matches(label, det, centers, model, expected, tolerances=REFERENCE):
    """Spectrum (Linf, L1) and peak-position agreement between a code's detector spectrum and the oracle's."""
    linf, l1 = oracle.relative_errors(model, expected, compared_pixels(det, centers))
    linf_tol, l1_tol = tolerances.detector_linf[det.channel], tolerances.detector_l1[det.channel]
    assert linf < linf_tol and l1 < l1_tol, (
        f"{label}: detector spectrum departs from the analytic one in the fit ranges: "
        f"Linf={linf:.2e} (tol {linf_tol:.1e}), L1={l1:.2e} (tol {l1_tol:.1e})"
    )
    peak_tol = tolerances.peak_nm[det.channel]
    for window in det.peak_windows:
        shift = oracle.peak_wavelength(centers, model, window) - oracle.peak_wavelength(centers, expected, window)
        assert abs(shift) < peak_tol, (
            f"{label}: the detector moved the peak in {window} nm by {shift:+.4f} nm (tol {peak_tol:.0e} nm)"
        )
