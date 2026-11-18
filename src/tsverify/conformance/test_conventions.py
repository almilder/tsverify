"""Convention probes: small asymmetric plasmas that tell a convention mismatch apart from a physics error.

Each probe compares the code's L2 spectrum with the oracle's under the declared conventions. If it fails, the probe
retries every single-convention change and names the ones under which the code agrees with the oracle.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from ..oracle import Ion, Plasma
from ._support import expected_spectral_density, spectral_density, spectral_errors


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

    def errors(conventions):
        modelled = conventions.interpret(declared.express(probe.plasma))
        return spectral_errors(conventions, model, expected_spectral_density(conventions, modelled, lam))

    linf, l1 = errors(declared)
    record_property("Linf under the declared conventions", linf)
    if linf < linf_tol and l1 < l1_tol:
        return
    explained = []
    for description, alternative in declared.alternatives():
        alt_linf, alt_l1 = errors(alternative)
        if alt_linf < linf_tol and alt_l1 < l1_tol:
            explained.append(f"{description} [Linf={alt_linf:.1e}]")
    verdict = (
        "It agrees with the oracle if " + "; or if ".join(explained) + ", so check the declared conventions."
        if explained
        else "No single convention change explains it, so this looks like a physics error, not a convention mismatch."
    )
    pytest.fail(
        f"probe {probe.name!r} (sensitive to the {probe.sensitive_to}): the spectrum departs from the oracle under "
        f"the declared conventions (Linf={linf:.1e}, L1={l1:.1e}; tol {linf_tol:.0e}, {l1_tol:.0e}). {verdict}"
    )
