"""L1: the code's Z' against the Faddeeva function, on a grid that runs past where tables usually end."""

import numpy as np
import pytest

from .. import oracle
from ..signatures import zprime_signatures
from ._support import zprime

ZETA = np.linspace(-15.0, 15.0, 30001)


def test_zprime_matches_faddeeva(ts_adapter, record_property):
    error = zprime(ts_adapter, ZETA) - oracle.plasma_dispersion_derivative(ZETA)
    worst = int(np.argmax(np.abs(error)))
    size = float(np.abs(error[worst]))
    record_property("max |Z' error|", size)
    tol = ts_adapter.tolerances.zprime_abs
    if size < tol:
        return
    readings = zprime_signatures(ZETA, error, tol) or ["No known signature."]
    pytest.fail(
        f"L1: Z' departs from the Faddeeva function by up to {size:.1e} at zeta = {ZETA[worst]:+.3f} "
        f"(tol {tol:.1e}). " + " ".join(readings)
    )
