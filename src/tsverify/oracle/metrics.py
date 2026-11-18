"""Comparison metrics between a code's spectrum and the oracle's."""

import numpy as np


def relative_errors(actual, expected, mask=None):
    """Peak-normalized L-infinity and relative L1 errors, optionally restricted to ``mask``."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if mask is not None:
        actual, expected = actual[mask], expected[mask]
    diff = np.abs(actual - expected)
    return float(diff.max() / np.abs(expected).max()), float(diff.sum() / np.abs(expected).sum())


def peak_wavelength(lam, values, window):
    """Three-point parabolic peak inside ``window = (lo, hi)``."""
    lam = np.asarray(lam)
    values = np.asarray(values)
    idx = np.flatnonzero((lam > window[0]) & (lam < window[1]))
    i = idx[np.argmax(values[idx])]
    if i in (idx[0], idx[-1]):
        raise ValueError(f"peak sits on the edge of window {window}")
    left, mid, right = values[i - 1 : i + 2]
    return float(lam[i] + 0.5 * (left - right) / (left - 2.0 * mid + right) * (lam[i + 1] - lam[i]))
