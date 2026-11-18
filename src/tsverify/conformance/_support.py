"""Calls into the adapter in the code's conventions, the oracle's matching spectrum, and skipping what the code
does not support."""

import dataclasses

import numpy as np
import pytest

from .. import oracle, volume as volume_oracle
from ..adapter import NotSupported, required_features
from ..volume import Volume


def _require(adapter, level, needed):
    if not adapter.supports(level):
        pytest.skip(f"{adapter.name}: {level} not supported")
    missing = set(needed) - set(adapter.features)
    if missing:
        pytest.skip(f"{adapter.name}: not supported: {', '.join(sorted(missing))}")


def spectral_density(adapter, plasma, wavelengths_nm):
    """The code's L2 spectrum of the canonical ``plasma``, as the code reports it."""
    _require(adapter, "spectral_density", required_features(plasma))
    try:
        return np.asarray(adapter.spectral_density(adapter.conventions.express(plasma), wavelengths_nm))
    except NotSupported as reason:
        pytest.skip(f"{adapter.name}: not supported: {reason}")


def expected_spectral_density(conventions, plasma, wavelengths_nm):
    """The oracle's spectrum of the canonical ``plasma``, reported the way ``conventions`` say."""
    return oracle.spectral_density(wavelengths_nm, plasma, **conventions.oracle_options())


def spectral_errors(conventions, model, expected, mask=None):
    """Peak-normalized (Linf, L1) errors; for a code without absolute units, after the best least-squares scale."""
    model = np.asarray(model, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if mask is not None:
        model, expected = model[mask], expected[mask]
    if not conventions.absolute:
        model = model * (np.dot(model, expected) / np.dot(model, model))
    return oracle.relative_errors(model, expected)


def detector_spectrum(adapter, volume, detector):
    """The code's L3 spectrum of the canonical ``volume`` on ``detector``."""
    if not isinstance(volume, Volume):
        volume = Volume(volume)
    _require(adapter, "detector_spectrum", required_features(volume.plasma, detector, volume))
    expressed = dataclasses.replace(volume, plasma=adapter.conventions.express(volume.plasma))
    try:
        centers, values = adapter.detector_spectrum(expressed, detector)
    except NotSupported as reason:
        pytest.skip(f"{adapter.name}: not supported: {reason}")
    return np.asarray(centers), np.asarray(values)


def expected_volume_spectrum(conventions, volume):
    """The spectrum behind the oracle's detector spectrum of the canonical ``volume``, as a callable of wavelength."""
    options = conventions.detector_options()
    return lambda lam: volume_oracle.spectral_density(lam, volume, **options)
