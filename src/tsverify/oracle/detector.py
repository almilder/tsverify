"""Analytic detector response: exact pixel integration of a Gaussian-blurred spectrum, notch and normalization."""

import numpy as np
from scipy.special import ndtr


def pixel_edges(centers):
    """Edges of a uniform pixel axis given its centers."""
    centers = np.asarray(centers, dtype=float)
    d = np.diff(centers)
    if not np.allclose(d, d[0], rtol=1e-9, atol=0):
        raise ValueError("pixel_edges expects uniformly spaced centers")
    return np.concatenate([centers - d[0] / 2.0, [centers[-1] + d[0] / 2.0]])


def detector_spectrum(spectrum, centers, sigma, *, transmission=None, samples_per_sigma=40, pad_sigmas=8.0,
                      band_sigmas=None):
    """Pixel means of the Gaussian-blurred spectrum (optionally filtered by ``transmission``), exact over each pixel.

    ``band_sigmas``: evaluate the Gaussian CDF only within that many sigma of each pixel edge and count the samples
    below the band as fully inside. With 10 the neglected terms are below 1e-20 relative and the result agrees with
    the full sum to summation roundoff (about 5e-14 relative), 3-5 times faster; the default (None) keeps the full
    sum."""
    edges = pixel_edges(centers)
    pixel = edges[1] - edges[0]
    step = min(sigma / samples_per_sigma, pixel / 400.0)
    lo, hi = edges[0] - pad_sigmas * sigma, edges[-1] + pad_sigmas * sigma
    fine = np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)
    source = spectrum(fine)
    if transmission is not None:
        source = source * transmission(fine)
    weights = np.full(fine.size, fine[1] - fine[0])
    weights[[0, -1]] *= 0.5
    weighted = source * weights
    if band_sigmas is None:
        cumulative = np.array([np.dot(weighted, ndtr((e - fine) / sigma)) for e in edges])
    else:
        below = np.concatenate([[0.0], np.cumsum(weighted)])
        first = np.searchsorted(fine, edges - band_sigmas * sigma)
        last = np.searchsorted(fine, edges + band_sigmas * sigma, side="right")
        cumulative = np.array([below[a] + np.dot(weighted[a:b], ndtr((e - fine[a:b]) / sigma))
                               for e, a, b in zip(edges, first, last)])
    return np.diff(cumulative) / np.diff(edges)


def rectangular_notch(center, width, optical_density):
    """Transmission ``10^-OD`` inside ``|lambda - center| < width/2``, one elsewhere."""

    def transmission(lam):
        lam = np.asarray(lam)
        return np.where(np.abs(lam - center) < width / 2.0, 10.0 ** (-optical_density), 1.0)

    return transmission


def normalize_in_ranges(centers, values, ranges, amplitude=100.0):
    """Scale so the maximum over the pixels inside ``ranges`` equals ``amplitude`` (tsadar's ``norm: 0`` rule)."""
    centers = np.asarray(centers)
    mask = np.zeros(centers.shape, dtype=bool)
    for low, high in ranges:
        mask |= (centers > low) & (centers < high)
    return amplitude * np.asarray(values) / np.max(np.asarray(values)[mask])
