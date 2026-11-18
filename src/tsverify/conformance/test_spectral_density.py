"""L2: the code's spectral density against the oracle's."""

import itertools

import numpy as np
import pytest

from .. import oracle
from ..cases import SPECTRUM_CASES
from ..oracle import ELECTRON_MASS, Ion, Plasma
from ._support import expected_spectral_density, spectral_density, spectral_errors


@pytest.mark.parametrize("npts", [256, 2048])
@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_spectral_density_matches_the_oracle(ts_adapter, case, npts):
    lam = np.linspace(*case.detector.lam_range, npts)
    model = spectral_density(ts_adapter, case.plasma, lam)
    expected = expected_spectral_density(ts_adapter.conventions, case.plasma, lam)
    linf, l1 = spectral_errors(ts_adapter.conventions, model, expected)
    linf_tol, l1_tol = ts_adapter.tolerances.spectral_density[case.problem]
    assert linf < linf_tol and l1 < l1_tol, (
        f"{case.problem}-L2/{case}: S(k, w) departs from the oracle: "
        f"peak-normalized Linf={linf:.2e} (tol {linf_tol:.1e}), L1={l1:.2e} (tol {l1_tol:.1e})"
    )


def test_high_z_ion_acoustic_resonance(ts_adapter):
    """Z Te/Ti = 160 puts the IAW resonance at zeta = 9, where tabulated Z' functions usually end."""
    plasma = Plasma(ne=0.5, Te=1.0, ions=(Ion(A=197.0, Z=40.0, Ti=0.25),))
    lam = np.linspace(525.0, 528.0, 8192)
    model = spectral_density(ts_adapter, plasma, lam)
    expected = expected_spectral_density(ts_adapter.conventions, plasma, lam)
    linf, _ = spectral_errors(ts_adapter.conventions, model, expected)
    tol = ts_adapter.tolerances
    assert linf < tol.high_z_linf, f"high-Z IAW departs from the oracle by {linf:.2e}"
    for window in ((525.0, 526.45), (526.55, 528.0)):
        shift = oracle.peak_wavelength(lam, model, window) - oracle.peak_wavelength(lam, expected, window)
        assert abs(shift) < tol.high_z_peak_nm, f"high-Z IAW peak in {window} nm moved by {shift:+.6f} nm"


SWEEP_MAX_ALPHA = 3.5
SWEEP_LAM_RANGE = (400.0, 700.0)


def _min_k_lambda_at_epw_peaks(plasma):
    lam = np.linspace(*SWEEP_LAM_RANGE, 30001)
    spectrum = oracle.spectral_density(lam, plasma)
    values = []
    for side in (lam < plasma.lam - 3.0, lam > plasma.lam + 3.0):
        k, _, _ = oracle.scattering_geometry(lam[np.argmax(np.where(side, spectrum, 0.0))], plasma)
        values.append(k * np.sqrt(plasma.Te / ELECTRON_MASS) / oracle.plasma_frequency(plasma.ne))
    return min(values)


def _sweep_plasmas():
    for ne, te, angle in itertools.product((0.05, 0.2, 0.5), (0.1, 0.6, 2.0), (30.0, 60.0, 120.0)):
        plasma = Plasma(ne=ne, Te=te, ions=(Ion(A=1.0, Z=1.0, Ti=te / 5),), angle_deg=angle)
        if 1.0 / _min_k_lambda_at_epw_peaks(plasma) <= SWEEP_MAX_ALPHA:
            yield pytest.param(plasma, id=f"ne{ne}-Te{te}-{angle:.0f}deg")


@pytest.mark.slow
@pytest.mark.parametrize("plasma", list(_sweep_plasmas()))
def test_spectral_density_matches_the_oracle_across_plasmas(ts_adapter, plasma):
    """A1 over a density, temperature and angle matrix (alpha <= 3.5), EPW wings only."""
    lam = np.linspace(*SWEEP_LAM_RANGE, 4096)
    model = spectral_density(ts_adapter, plasma, lam)
    wings = np.abs(lam - plasma.lam) > 3.0
    expected = expected_spectral_density(ts_adapter.conventions, plasma, lam[wings])
    linf, _ = spectral_errors(ts_adapter.conventions, model[wings], expected)
    tol = ts_adapter.tolerances.sweep_linf
    assert linf < tol, f"A1-sweep: EPW wings depart from the oracle by {linf:.2e} (tol {tol:.0e})"
