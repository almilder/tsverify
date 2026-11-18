"""Calls into the adapter that skip, rather than fail, what the code does not support."""

import numpy as np
import pytest

from ..adapter import NotSupported, required_features
from ..volume import Volume


def _require(adapter, level, needed):
    if not adapter.supports(level):
        pytest.skip(f"{adapter.name}: {level} not supported")
    missing = set(needed) - set(adapter.features)
    if missing:
        pytest.skip(f"{adapter.name}: not supported: {', '.join(sorted(missing))}")


def spectral_density(adapter, plasma, wavelengths_nm):
    _require(adapter, "spectral_density", required_features(plasma))
    try:
        return np.asarray(adapter.spectral_density(plasma, wavelengths_nm))
    except NotSupported as reason:
        pytest.skip(f"{adapter.name}: not supported: {reason}")


def detector_spectrum(adapter, volume, detector):
    if not isinstance(volume, Volume):
        volume = Volume(volume)
    _require(adapter, "detector_spectrum", required_features(volume.plasma, detector, volume))
    try:
        centers, values = adapter.detector_spectrum(volume, detector)
    except NotSupported as reason:
        pytest.skip(f"{adapter.name}: not supported: {reason}")
    return np.asarray(centers), np.asarray(values)
