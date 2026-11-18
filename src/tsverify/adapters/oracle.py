"""The oracle presented as a code under test. Every conformance test passes with zero model error; it checks the
tests themselves.

``OracleAdapter(conventions)`` reads its inputs and reports its outputs in ``conventions``.
``behaves_as`` makes it declare ``conventions`` but behave as other conventions, to check the convention probes.
"""

import dataclasses

import numpy as np

from .. import oracle
from ..adapter import FEATURES, Adapter
from ..compare import oracle_detector
from ..conventions import CANONICAL
from ..volume import spectral_density

UNKNOWN_SCALE = 3.7


class OracleAdapter(Adapter):
    name = "oracle"
    features = FEATURES

    def __init__(self, conventions=CANONICAL, behaves_as=None):
        self.conventions = conventions
        self.behaves_as = behaves_as or conventions

    def spectral_density(self, plasma, wavelengths_nm):
        actual = self.behaves_as
        spectrum = oracle.spectral_density(wavelengths_nm, actual.interpret(plasma), **actual.oracle_options())
        return spectrum if actual.absolute else UNKNOWN_SCALE * spectrum

    def detector_spectrum(self, volume, detector):
        actual = self.behaves_as
        modelled = dataclasses.replace(volume, plasma=actual.interpret(volume.plasma))
        options = actual.detector_options()
        centers = np.linspace(*detector.lam_range, detector.pixels)
        return centers, oracle_detector(detector, centers, lambda lam: spectral_density(lam, modelled, **options))
