"""Convention probes: small asymmetric plasmas that tell a convention mismatch apart from a physics error.

Each probe compares the code's L2 spectrum with the oracle's under the declared conventions. If it fails, the probe
retries every single-convention change and every known defect (:mod:`tsverify.signatures`) and names the ones
under which the code agrees with the oracle.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from ..oracle import Ion, Plasma
from ..signatures import explain_spectrum, verdict
from ._support import expected_spectral_density, spectral_density, spectral_errors, zprime_hint


@dataclass(frozen=True)
class Probe:
    name: str
    plasma: Plasma
    lam_range: tuple[float, float]
    sensitive_to: str


HYDROGEN = Ion(A=1.0, Z=1.0, Ti=0.08)
PROBES = [
    Probe("flow", Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, Va=3.0),)), (524.5, 528.5), "flow sign"),
    Probe("drift", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), ud=60.0), (440.0, 630.0), "drift sign"),
    Probe(
        "drift-frame",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, Va=30.0),), ud=60.0),
        (440.0, 630.0),
        "electron drift frame",
    ),
    Probe(
        "wings",
        Plasma(ne=0.5, Te=1.0, ions=(Ion(A=1.0, Z=1.0, Ti=0.2),)),
        (420.0, 660.0),
        "relativistic factor, spectral density per wavelength or frequency, and dispersion in k",
    ),
]


@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.name)
def test_convention_probe(ts_adapter, probe, record_property):
    lam = np.linspace(*probe.lam_range, 2048)
    model = spectral_density(ts_adapter, probe.plasma, lam)
    declared = ts_adapter.conventions
    linf_tol, l1_tol = ts_adapter.tolerances.spectral_density["A1"]

    linf, l1 = spectral_errors(declared, model, expected_spectral_density(declared, probe.plasma, lam))
    record_property("Linf under the declared conventions", linf)
    if linf < linf_tol and l1 < l1_tol:
        return
    found = explain_spectrum(declared, probe.plasma, lam, model, lambda a, b: a < linf_tol and b < l1_tol)
    pytest.fail(
        f"probe {probe.name!r} (sensitive to the {probe.sensitive_to}): the spectrum departs from the oracle under "
        f"the declared conventions (Linf={linf:.1e}, L1={l1:.1e}; tol {linf_tol:.0e}, {l1_tol:.0e}). "
        + verdict(*found, otherwise=zprime_hint(ts_adapter))
    )
