"""Calls into the adapter in the code's conventions, the oracle's matching spectrum, and skipping what the code
does not support."""

import dataclasses

import numpy as np
import pytest

from .. import volume as volume_oracle
from ..adapter import NotSupported, required_features
from ..compare import expected_spectral_density, spectral_errors  # noqa: F401
from ..volume import Volume


def _require(adapter, level, needed):
    if not adapter.supports(level):
        pytest.skip(f"{adapter.name}: {level} not supported")
    missing = set(needed) - set(adapter.features)
    if missing:
        pytest.skip(f"{adapter.name}: not supported: {', '.join(sorted(missing))}")


def zprime(adapter, zeta):
    """The code's Z' at real ``zeta``."""
    _require(adapter, "zprime", ())
    try:
        return np.asarray(adapter.zprime(zeta))
    except NotSupported as reason:
        pytest.skip(f"{adapter.name}: not supported: {reason}")


def zprime_hint(adapter):
    """Where to look next when a spectrum fails without a known explanation."""
    if adapter.supports("zprime"):
        return " The L1 Z' test shows whether the plasma dispersion function is behind it."
    return " An adapter zprime(zeta) would let the L1 Z' test check the plasma dispersion function."


def spectral_density(adapter, plasma, wavelengths_nm):
    """The code's L2 spectrum of the canonical ``plasma``, as the code reports it."""
    _require(adapter, "spectral_density", required_features(plasma))
    try:
        return np.asarray(adapter.spectral_density(adapter.conventions.express(plasma), wavelengths_nm))
    except NotSupported as reason:
        pytest.skip(f"{adapter.name}: not supported: {reason}")


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
