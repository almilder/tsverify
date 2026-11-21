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
from ..fitting import least_squares_fit, sized_shot_fit
from ..volume import spectral_density

UNKNOWN_SCALE = 3.7
FIT_BAND = 10.0


class OracleAdapter(Adapter):
    name = "oracle"
    features = FEATURES

    def __init__(self, conventions=CANONICAL, behaves_as=None):
        self.conventions = conventions
        self.behaves_as = behaves_as or conventions

    def zprime(self, zeta):
        return oracle.plasma_dispersion_derivative(zeta)

    def spectral_density(self, plasma, wavelengths_nm):
        actual = self.behaves_as
        spectrum = oracle.spectral_density(wavelengths_nm, actual.interpret(plasma), **actual.oracle_options())
        return spectrum if actual.absolute else UNKNOWN_SCALE * spectrum

    def fit(self, problem):
        return least_squares_fit(lambda volume: self._detector(volume, problem.detector, FIT_BAND)[1], problem)

    def fit_shot(self, problem):
        det = problem.lineouts[0].detector
        return sized_shot_fit(lambda volume: self._detector(volume, det, FIT_BAND)[1], problem)

    def detector_spectrum(self, volume, detector):
        return self._detector(volume, detector)

    def _detector(self, volume, detector, band_sigmas=None):
        actual = self.behaves_as
        modelled = dataclasses.replace(volume, plasma=actual.interpret(volume.plasma))
        options = actual.detector_options()
        centers = np.linspace(*detector.lam_range, detector.pixels)
        spectrum = lambda lam: spectral_density(lam, modelled, **options)  # noqa: E731
        return centers, oracle_detector(detector, centers, spectrum, band_sigmas)
