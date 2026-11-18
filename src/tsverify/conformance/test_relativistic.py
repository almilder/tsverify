"""A12: the code's non-collective spectrum against Selden's exact relativistic one."""

import numpy as np
import pytest

from .. import oracle
from ..oracle import ELECTRON_MASS_KEV, Ion, Plasma
from ._support import spectral_density


def _peak(x, y):
    i = int(np.argmax(y))
    left, mid, right = y[i - 1 : i + 2]
    return x[i] + 0.5 * (left - right) / (left - 2.0 * mid + right) * (x[1] - x[0])


@pytest.mark.parametrize("theta", [45.0, 90.0, 135.0])
@pytest.mark.parametrize("te", [0.5, 2.0, 5.0])
def test_first_order_relativistic_accuracy_against_the_exact_spectrum(ts_adapter, te, theta, record_property):
    plasma = Plasma(ne=1.0e-6, Te=te, ions=(Ion(A=1.0, Z=1.0, Ti=te),), angle_deg=theta)
    lam = np.linspace(0.45 * plasma.lam, 1.6 * plasma.lam, 8192)
    model = spectral_density(ts_adapter, plasma, lam)
    eps = lam / plasma.lam - 1.0
    exact = oracle.selden_spectrum(eps, theta, te)
    model, exact = model / model.max(), exact / exact.max()
    beta2 = te / ELECTRON_MASS_KEV
    s2 = np.sin(np.deg2rad(theta) / 2.0) ** 2
    second_order_shift = 14.0 * s2 * (1.0 + 7.0 * s2) * beta2**2

    shape = float(np.abs(model - exact).max())
    shift = _peak(eps, model) - _peak(eps, exact)
    record_property("shape error / (Te/mc^2)", float(shape / beta2))
    record_property("peak error / second-order shift", float(shift / second_order_shift))
    tol = ts_adapter.tolerances
    assert shape < tol.relativistic_shape * beta2, (
        f"A12: shape off the exact spectrum by {shape:.2e} = {shape / beta2:.2f} Te/mc^2"
    )
    assert abs(shift) < tol.relativistic_peak * second_order_shift, (
        f"A12: peak off by {shift:+.5f} = {shift / second_order_shift:.2f} of the second-order shift"
    )
