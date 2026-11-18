"""A10: gradient, aperture and species averaging against the oracle, and plasmas written two ways."""

import dataclasses

import numpy as np
import pytest

from ..cases import A1_CASES
from ..compare import assert_detector_matches, oracle_detector
from ..volume import Aperture, LinearGradient, Volume, spectral_density
from ._support import detector_spectrum

EPW_CASE, _, IAW_CASE, CH_CASE = A1_CASES
GRADIENT = LinearGradient(ne_percent=20.0, Te_percent=20.0, points=5)


@pytest.mark.parametrize("case", [EPW_CASE, IAW_CASE], ids=str)
def test_gradient_average_matches_the_averaged_oracle_spectrum(ts_adapter, case):
    det = case.detector
    volume = Volume(case.plasma, gradient=GRADIENT)
    centers, model = detector_spectrum(ts_adapter, volume, det)
    expected = oracle_detector(det, centers, lambda lam: spectral_density(lam, volume))
    assert_detector_matches(f"A10-gradient/{case}", det, centers, model, expected, ts_adapter.tolerances)


def test_zero_gradient_collapses_to_a_single_point(ts_adapter):
    det = EPW_CASE.detector
    _, single = detector_spectrum(ts_adapter, Volume(EPW_CASE.plasma), det)
    _, flat = detector_spectrum(ts_adapter, Volume(EPW_CASE.plasma, gradient=LinearGradient(points=5)), det)
    tol = ts_adapter.tolerances.zero_gradient
    np.testing.assert_allclose(flat, single, rtol=tol, atol=tol, err_msg="A10: zero gradient changed the spectrum")


def test_aperture_weighted_sum_matches_the_oracle(ts_adapter):
    det = EPW_CASE.detector
    volume = Volume(EPW_CASE.plasma, aperture=Aperture((54.0, 60.0, 66.0), (0.25, 0.5, 0.25)))
    centers, model = detector_spectrum(ts_adapter, volume, det)
    expected = oracle_detector(det, centers, lambda lam: spectral_density(lam, volume))
    assert_detector_matches("A10-aperture", det, centers, model, expected, ts_adapter.tolerances)


@pytest.mark.parametrize(
    "case, same_plasma",
    [
        (CH_CASE, dataclasses.replace(CH_CASE.plasma, ions=CH_CASE.plasma.ions[::-1])),
        (
            IAW_CASE,
            dataclasses.replace(
                IAW_CASE.plasma,
                ions=tuple(dataclasses.replace(IAW_CASE.plasma.ions[0], fract=f) for f in (0.3, 0.7)),
            ),
        ),
    ],
    ids=["species-order", "species-split"],
)
def test_the_same_plasma_written_differently_gives_the_same_spectrum(ts_adapter, case, same_plasma):
    _, reference = detector_spectrum(ts_adapter, Volume(case.plasma), case.detector)
    _, rewritten = detector_spectrum(ts_adapter, Volume(same_plasma), case.detector)
    tol = ts_adapter.tolerances.rewritten_plasma
    np.testing.assert_allclose(
        rewritten, reference, rtol=tol, atol=tol * reference.max(), err_msg=f"A10: {case} rewritten"
    )
