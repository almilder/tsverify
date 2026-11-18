"""L3: the code's detector spectrum against the oracle's."""

import numpy as np
import pytest

from ..cases import SPECTRUM_CASES
from ..compare import assert_detector_matches, oracle_detector
from ..volume import Volume
from ._support import detector_spectrum, expected_volume_spectrum


@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_detector_spectrum_matches_the_oracle(ts_adapter, case):
    det = case.detector
    volume = Volume(case.plasma)
    centers, model = detector_spectrum(ts_adapter, volume, det)
    np.testing.assert_allclose(
        centers,
        np.linspace(*det.lam_range, det.pixels),
        rtol=0,
        atol=1e-9,
        err_msg=f"{case}: detector pixel centres moved",
    )
    expected = oracle_detector(det, centers, expected_volume_spectrum(ts_adapter.conventions, volume))
    assert_detector_matches(f"{case.problem}-L3/{case}", det, centers, model, expected, ts_adapter.tolerances)
