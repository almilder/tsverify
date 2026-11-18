"""The oracle presented as a code under test. Every conformance test passes with zero model error; it checks the
tests themselves."""

import numpy as np

from .. import oracle
from ..adapter import FEATURES, Adapter
from ..compare import oracle_detector
from ..volume import spectral_density


class OracleAdapter(Adapter):
    name = "oracle"
    features = FEATURES

    def spectral_density(self, plasma, wavelengths_nm):
        return oracle.spectral_density(wavelengths_nm, plasma)

    def detector_spectrum(self, volume, detector):
        centers = np.linspace(*detector.lam_range, detector.pixels)
        return centers, oracle_detector(detector, centers, lambda lam: spectral_density(lam, volume))
