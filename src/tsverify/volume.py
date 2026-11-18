"""The scattering volume: a plasma, optionally with a linear gradient across it and an angular aperture."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import oracle
from .oracle import Plasma


@dataclass(frozen=True)
class LinearGradient:
    """ne and Te vary linearly across the volume, end to end by ``ne_percent`` and ``Te_percent``.

    The volume is sampled at ``points`` equal-weight points with scales ``linspace(1 - p/200, 1 + p/200, points)``.
    """

    ne_percent: float = 0.0
    Te_percent: float = 0.0
    points: int = 1

    def plasmas(self, plasma: Plasma):
        scales_ne = np.linspace(1 - self.ne_percent / 200, 1 + self.ne_percent / 200, self.points)
        scales_te = np.linspace(1 - self.Te_percent / 200, 1 + self.Te_percent / 200, self.points)
        return [dataclasses.replace(plasma, ne=plasma.ne * a, Te=plasma.Te * b) for a, b in zip(scales_ne, scales_te)]


@dataclass(frozen=True)
class Aperture:
    """Scattering angles collected by the spectrometer and their weights."""

    angles_deg: tuple[float, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class Volume:
    plasma: Plasma
    gradient: LinearGradient | None = None
    aperture: Aperture | None = None


def _homogeneous(lam_nm, plasma, gradient, options):
    if gradient is None:
        return oracle.spectral_density(lam_nm, plasma, **options)
    return np.mean([oracle.spectral_density(lam_nm, p, **options) for p in gradient.plasmas(plasma)], axis=0)


def spectral_density(lam_nm, volume: Volume, **options):
    """The oracle's spectrum of ``volume``: the gradient's equal-weight mean, summed over the aperture with its weights.
    ``options`` go to :func:`tsverify.oracle.spectral_density`."""
    if volume.aperture is None:
        return _homogeneous(lam_nm, volume.plasma, volume.gradient, options)
    return sum(
        w * _homogeneous(lam_nm, dataclasses.replace(volume.plasma, angle_deg=a), volume.gradient, options)
        for a, w in zip(volume.aperture.angles_deg, volume.aperture.weights)
    )
