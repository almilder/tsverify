"""Exact non-collective relativistic spectrum (Selden 1980)."""

import numpy as np

from .constants import ELECTRON_MASS_KEV


def selden_spectrum(eps, theta_deg, te_kev):
    """Exact non-collective relativistic spectrum shape, Selden, Phys. Lett. A 79, 405 (1980), eq. (1).

    ``eps = lambda_s / lambda_i - 1``; the normalizing constant is omitted.
    """
    eps = np.asarray(eps, dtype=float)
    alpha = ELECTRON_MASS_KEV / te_kev / 2.0
    one_minus_cos = 1.0 - np.cos(np.deg2rad(theta_deg))
    a = (1.0 + eps) ** 3 * np.sqrt(2.0 * one_minus_cos * (1.0 + eps) + eps**2)
    b = np.sqrt(1.0 + eps**2 / (2.0 * one_minus_cos * (1.0 + eps))) - 1.0
    return np.exp(-2.0 * alpha * b) / a
