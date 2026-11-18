"""The interface a code under test implements.

Each level is optional; a code overrides the ones it supports, lists its ``features`` and declares its
``conventions`` (:mod:`tsverify.conventions`; default canonical, which is tsadar's). A test that needs an
unsupported level or feature is skipped as *not supported*, which is reported apart from a failure.

- L2 ``spectral_density(plasma, wavelengths_nm) -> S``: the spectrum at the given wavelengths, in the oracle's
  units (``tsverify.oracle.spectral_density``) under the declared conventions.
- L3 ``detector_spectrum(volume, detector) -> (pixel_centers_nm, values)``: pixel values of ``volume`` on
  ``detector``, normalized so the maximum inside ``detector.amp_ranges`` is ``detector.amplitude``.

The plasma (and the volume's plasma) the adapter receives is already written in the code's conventions.
"""

from __future__ import annotations

from .conventions import CANONICAL
from .oracle import GaussianComponent
from .tolerances import REFERENCE
from .volume import Volume

MAXWELLIAN = (GaussianComponent(),)
FEATURES = frozenset(
    {
        "multiple_ions",
        "ion_flow",
        "electron_drift",
        "numeric_electron_edf",
        "numeric_ion_edf",
        "notch",
        "gradient",
        "aperture",
    }
)
LEVELS = ("spectral_density", "detector_spectrum")


class NotSupported(Exception):
    """Raised by an adapter for an input its code cannot represent."""


class Adapter:
    name = "unnamed"
    features: frozenset = frozenset()
    conventions = CANONICAL
    tolerances = REFERENCE

    def spectral_density(self, plasma, wavelengths_nm):
        raise NotSupported("spectral_density")

    def detector_spectrum(self, volume, detector):
        raise NotSupported("detector_spectrum")

    def supports(self, level):
        return getattr(type(self), level) is not getattr(Adapter, level)


def required_features(plasma, detector=None, volume: Volume | None = None):
    """The adapter features a plasma, detector and volume need."""
    needed = set()
    if len(plasma.ions) > 1:
        needed.add("multiple_ions")
    if any(ion.Va != 0.0 for ion in plasma.ions):
        needed.add("ion_flow")
    if plasma.ud != 0.0:
        needed.add("electron_drift")
    if plasma.electrons != MAXWELLIAN:
        needed.add("numeric_electron_edf")
    if any(ion.components != MAXWELLIAN for ion in plasma.ions):
        needed.add("numeric_ion_edf")
    if detector is not None and detector.notch:
        needed.add("notch")
    if volume is not None and volume.gradient is not None:
        needed.add("gradient")
    if volume is not None and volume.aperture is not None:
        needed.add("aperture")
    return needed
