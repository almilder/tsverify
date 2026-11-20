"""L3: the code's detector spectrum against the oracle's."""

import pytest

from .. import tasks
from ..cases import SPECTRUM_CASES
from ._support import run_task


@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_detector_spectrum_matches_the_oracle(ts_adapter, case, record_property):
    run_task(ts_adapter, tasks.detector_task(case), record_property)
