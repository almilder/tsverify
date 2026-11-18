"""Conventions: the oracle's options, expressing a plasma for a code, and the probes' diagnoses."""

import dataclasses

import numpy as np
import pytest

from tsverify import oracle
from tsverify.adapters.oracle import OracleAdapter
from tsverify.cases import A1_CASES, SPECTRUM_CASES
from tsverify.conformance import test_convention_probe as convention_probe
from tsverify.conformance import test_detector_spectrum_matches_the_oracle as detector_conformance
from tsverify.conformance import test_spectral_density_matches_the_oracle as spectral_conformance
from tsverify.conformance.test_conventions import PROBES
from tsverify.conventions import CANONICAL, Conventions

CH = A1_CASES[3].plasma
LAM = np.linspace(440.0, 630.0, 501)
EVERYTHING_DIFFERENT = Conventions(
    flow_sign=-1, electron_drift="lab", relativistic="none", spectrum_per="frequency", dispersion_in_k=False,
    absolute=False,
)


def _record(*_):
    pass


def test_spectral_density_per_frequency_times_d_omega_d_lambda_is_per_wavelength():
    per_lambda = oracle.spectral_density(LAM, CH)
    per_omega = oracle.spectral_density(LAM, CH, per="frequency")
    np.testing.assert_allclose(per_omega * 2 * np.pi * oracle.C_CM_PER_S / (LAM * 1e-7) ** 2, per_lambda, rtol=1e-14)


def test_the_relativistic_option_is_the_first_order_factor():
    _, omega, omega_l = oracle.scattering_geometry(LAM, CH)
    ratio = oracle.spectral_density(LAM, CH) / oracle.spectral_density(LAM, CH, relativistic="none")
    np.testing.assert_allclose(ratio, 1 + 2 * omega / omega_l, rtol=1e-13)


def test_without_dispersion_k_is_the_vacuum_bragg_wavenumber():
    k, omega, omega_l = oracle.scattering_geometry(LAM, CH, dispersion_in_k=False)
    k_s, k_l = (omega + omega_l) / oracle.C_CM_PER_S, omega_l / oracle.C_CM_PER_S
    theta = np.deg2rad(CH.angle_deg)
    np.testing.assert_allclose(k, np.sqrt(k_s**2 + k_l**2 - 2 * k_s * k_l * np.cos(theta)), rtol=1e-14)


def test_options_are_checked():
    with pytest.raises(ValueError, match="relativistic"):
        oracle.spectral_density(LAM, CH, relativistic="exact")
    with pytest.raises(ValueError, match="electron_drift"):
        Conventions(electron_drift="ions")


@pytest.mark.parametrize("flow_sign", [1, -1])
@pytest.mark.parametrize("frame", ["ion_frame", "lab"])
def test_express_and_interpret_are_inverse(flow_sign, frame):
    plasma = dataclasses.replace(CH, ud=12.0)
    conventions = Conventions(flow_sign=flow_sign, electron_drift=frame)
    back = conventions.interpret(conventions.express(plasma))
    assert back.ud == pytest.approx(plasma.ud, abs=1e-12)
    assert [ion.Va for ion in back.ions] == pytest.approx([ion.Va for ion in plasma.ions], abs=1e-12)


def test_lab_frame_drift_adds_the_charge_weighted_ion_flow():
    plasma = dataclasses.replace(CH, ud=12.0)
    expressed = Conventions(electron_drift="lab").express(plasma)
    z_flow = (6.0 * 0.5 * 1.0 + 1.0 * 0.5 * 3.0) / (6.0 * 0.5 + 1.0 * 0.5)
    assert expressed.ud == pytest.approx(12.0 + z_flow)
    assert Conventions(flow_sign=-1).express(plasma).ions[1].Va == -3.0


def test_canonical_conventions_leave_the_plasma_untouched():
    assert CANONICAL.express(CH) is CH and CANONICAL.interpret(CH) is CH


@pytest.mark.parametrize("case", [c for c in SPECTRUM_CASES if c.problem == "A1"], ids=str)
def test_a_code_with_other_declared_conventions_passes(case):
    adapter = OracleAdapter(EVERYTHING_DIFFERENT)
    spectral_conformance(adapter, case, 256)
    detector_conformance(adapter, case)


@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.name)
def test_probes_pass_when_the_declared_conventions_are_right(probe):
    convention_probe(OracleAdapter(), probe, _record)
    convention_probe(OracleAdapter(EVERYTHING_DIFFERENT), probe, _record)


@pytest.mark.parametrize("description, actual", CANONICAL.alternatives(), ids=lambda x: getattr(x, "__name__", None))
def test_some_probe_names_each_wrongly_declared_convention(description, actual):
    adapter = OracleAdapter(CANONICAL, behaves_as=actual)
    messages = []
    for probe in PROBES:
        try:
            convention_probe(adapter, probe, _record)
        except pytest.fail.Exception as failure:
            messages.append(str(failure))
    assert messages, f"no probe noticed that {description}"
    assert any(description in m for m in messages), f"no probe named {description!r}: {messages}"
