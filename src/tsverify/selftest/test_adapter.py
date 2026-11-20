"""The adapter interface: declared levels and features, and skipping what a code does not support."""

import numpy as np
import pytest

from tsverify.adapter import Adapter, NotSupported, required_features
from tsverify.cases import A1_CASES, SPECTRUM_CASES
from tsverify.execute import evaluate
from tsverify.tasks import Run
from tsverify.oracle import Ion, Plasma
from tsverify.volume import Aperture, LinearGradient, Volume


class SpectrumOnly(Adapter):
    name = "spectrum-only"

    def spectral_density(self, plasma, wavelengths_nm):
        if plasma.ions[0].Z > 1:
            raise NotSupported("Z > 1")
        return np.ones_like(wavelengths_nm)


def test_supports_reports_the_overridden_levels():
    assert SpectrumOnly().supports("spectral_density")
    assert not SpectrumOnly().supports("detector_spectrum")


def test_required_features():
    by_name = {case.name: case for case in SPECTRUM_CASES}
    assert required_features(by_name["A1-epw-hydrogen"].plasma) == set()
    assert required_features(by_name["A1-iaw-CH-relative-drift"].plasma) == {"multiple_ions", "ion_flow"}
    assert required_features(by_name["A1-epw-electron-drift"].plasma) == {"electron_drift"}
    assert required_features(by_name["A5-epw-kappa-3"].plasma) == {"numeric_electron_edf"}
    assert required_features(by_name["A3-iaw-ion-beam"].plasma) == {"numeric_ion_edf"}
    case = A1_CASES[0]
    volume = Volume(case.plasma, LinearGradient(10.0, 10.0, 3), Aperture((50.0, 60.0), (0.5, 0.5)))
    assert required_features(case.plasma, case.detector, volume) == {"notch", "gradient", "aperture"}


def test_unsupported_levels_features_and_inputs_skip():
    adapter = SpectrumOnly()
    lam = np.linspace(500.0, 550.0, 11)
    hydrogen = A1_CASES[0].plasma

    def spectrum(plasma):
        return evaluate(adapter, Run("spectral_density", plasma=plasma, wavelengths_nm=lam))

    assert np.all(spectrum(hydrogen) == 1.0)
    with pytest.raises(NotSupported, match="not supported: electron_drift"):
        spectrum(A1_CASES[1].plasma)
    with pytest.raises(NotSupported, match="detector_spectrum not supported"):
        evaluate(adapter, Run("detector_spectrum", volume=Volume(hydrogen), detector=A1_CASES[0].detector))
    carbon = Plasma(ne=0.2, Te=0.6, ions=(Ion(A=12.0, Z=6.0, Ti=0.2),))
    with pytest.raises(NotSupported, match="not supported: Z > 1"):
        spectrum(carbon)