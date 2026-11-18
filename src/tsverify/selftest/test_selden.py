"""The oracle's exact relativistic spectrum against Selden's published tables."""

import pytest
from scipy.optimize import minimize_scalar

from tsverify import oracle as oracles

SELDEN_TABLE_1 = {
    0.6: 0.0452, 0.5: 0.0819, 0.4: 0.1458, 0.3: 0.2532, 0.2: 0.4238, 0.1: 0.6745, 0.0: 1.0,
    -0.1: 1.3433, -0.2: 1.5703, -0.3: 1.5049, -0.4: 1.0782, -0.5: 0.4959, -0.6: 0.1113, -0.7: 0.0069,
}
SELDEN_TABLE_2 = {3.77: -0.05, 7.80: -0.1, 16.75: -0.2, 27.10: -0.3, 39.21: -0.4, 53.56: -0.5, 71.00: -0.6, 93.27: -0.7}


def test_oracle_reproduces_seldens_tables():
    norm = oracles.selden_spectrum(0.0, 90.0, 20.0)
    for eps, value in SELDEN_TABLE_1.items():
        assert oracles.selden_spectrum(eps, 90.0, 20.0) / norm == pytest.approx(value, abs=1e-4), f"Table 1, eps={eps}"
    for te, eps_c in SELDEN_TABLE_2.items():
        peak = minimize_scalar(
            lambda e: -oracles.selden_spectrum(e, 90.0, te), bounds=(-0.95, 0.5), method="bounded",
            options={"xatol": 1e-10},
        ).x
        assert peak == pytest.approx(eps_c, abs=5e-4), f"Table 2, Te={te} keV"
