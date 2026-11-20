"""Known-defect signatures: each planted defect in the toy code is caught and named, and nothing else is."""

import re

import numpy as np
import pytest

from tsverify import oracle
from tsverify.adapters.toy import DEFECTS, ToyAdapter
from tsverify.cases import A1_CASES, SPECTRUM_CASES
from tsverify.conformance import test_convention_probe as convention_probe
from tsverify.conformance import test_high_z_ion_acoustic_resonance as high_z_conformance
from tsverify.conformance import test_spectral_density_matches_the_oracle as spectral_conformance
from tsverify.conformance import test_zprime_matches_faddeeva as zprime_conformance
from tsverify.conformance.test_conventions import PROBES
from tsverify.conformance.test_zprime import ZETA
from tsverify.conventions import CANONICAL
from tsverify.signatures import SPECTRUM_SIGNATURES, explain_spectrum, zprime_signatures

NAMED = {
    "flow-sign": "flow_sign=-1",
    "no-relativistic-factor": "relativistic='none'",
    "forward-difference-f'": "forward-difference-f'",
    "thermal-speed": "Te-doubled",
}
LABEL = re.compile(r"\(([\w'-]+(?:=[^)]*)?)\) \[")


def _record(*_):
    pass


def _failures(adapter):
    calls = [(f"L2/{c}@{n}", lambda c=c, n=n: spectral_conformance(adapter, c, n, _record)) for c in SPECTRUM_CASES for n in (256, 2048)]
    calls += [(f"probe/{p.name}", lambda p=p: convention_probe(adapter, p, _record)) for p in PROBES]
    calls += [("L1", lambda: zprime_conformance(adapter, _record)), ("high-Z", lambda: high_z_conformance(adapter, _record))]
    failures = {}
    for name, call in calls:
        try:
            call()
        except pytest.skip.Exception:
            pass
        except pytest.fail.Exception as failure:
            failures[name] = str(failure)
    return failures


def test_the_toy_code_without_defects_passes():
    assert _failures(ToyAdapter()) == {}


@pytest.mark.parametrize("defect", sorted(NAMED))
def test_each_spectrum_defect_is_caught_and_named_alone(defect):
    failures = _failures(ToyAdapter(defect))
    assert failures, f"no test caught {DEFECTS[defect]}"
    for name, message in failures.items():
        if name == "L1":
            continue
        assert LABEL.findall(message) == [NAMED[defect]], f"{name}: {message}"


def test_a_zprime_table_is_localized_by_the_l1_test():
    failures = _failures(ToyAdapter("zprime-table"))
    assert "table edge" in failures["L1"] and "|zeta| = 8.00" in failures["L1"]
    assert "vanishes every 0.1 in zeta" in failures["L1"]
    spectra = [m for name, m in failures.items() if name != "L1"]
    assert spectra and all("physics error" in m and "L1 Z' test" in m for m in spectra)


def test_the_finite_difference_signature_converges_at_first_order():
    plasma = A1_CASES[0].plasma
    signature = next(s for s in SPECTRUM_SIGNATURES if s.name == "forward-difference-f'")
    errors = []
    for npts in (1024, 8192):
        lam = np.linspace(440.0, 630.0, npts)
        wings = np.abs(lam - plasma.lam) > 3.0
        defect, exact = signature.spectrum(CANONICAL, plasma, lam), oracle.spectral_density(lam, plasma)
        errors.append(oracle.relative_errors(defect[wings], exact[wings])[0])
    assert 7.0 < errors[0] / errors[1] < 9.0


def test_explain_names_a_doubled_electron_temperature():
    plasma = A1_CASES[0].plasma
    lam = np.linspace(440.0, 630.0, 1024)
    model = oracle.spectral_density(lam, oracle.Plasma(ne=plasma.ne, Te=2 * plasma.Te, ions=plasma.ions))
    conventions, defects = explain_spectrum(CANONICAL, plasma, lam, model, lambda a, b: a < 1e-3 and b < 5e-4)
    assert conventions == [] and len(defects) == 1 and "(Te-doubled)" in defects[0]


def test_zprime_readings():
    assert zprime_signatures(ZETA, np.zeros_like(ZETA), 1e-4) == []
    bump = 1.63e-3 * np.exp(-((np.abs(ZETA) - 1.37) ** 2) / 0.3)
    assert any("rdWT" in r for r in zprime_signatures(ZETA, bump, 1e-4))
    step = np.where(np.abs(ZETA) > 9.0, 5e-4, 0.0)
    readings = zprime_signatures(ZETA, step, 1e-4)
    assert len(readings) == 1 and "|zeta| = 9.00" in readings[0]
