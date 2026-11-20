"""L4: fitting oracle detector spectra: recovery, uncertainty against the Cramér–Rao bound, and calibration."""

import pytest

from .. import tasks
from ._support import run_task


@pytest.mark.parametrize("name, case, start", tasks.FIT_PROBLEMS, ids=[p[0] for p in tasks.FIT_PROBLEMS])
def test_fit_recovers_the_truth_from_oracle_data(ts_adapter, name, case, start, record_property):
    run_task(ts_adapter, tasks.recovery_task(name, case, start), record_property)


@pytest.mark.parametrize("variant", tasks.CRAMER_RAO_VARIANTS)
@pytest.mark.parametrize("name, case, start", tasks.FIT_PROBLEMS, ids=[p[0] for p in tasks.FIT_PROBLEMS])
def test_fit_uncertainty_matches_the_cramer_rao_bound(ts_adapter, name, case, start, variant, record_property):
    run_task(ts_adapter, tasks.cramer_rao_task(name, case, start, variant), record_property)


@pytest.mark.slow
@pytest.mark.parametrize("index", range(len(tasks.FIT_PROBLEMS)), ids=[p[0] for p in tasks.FIT_PROBLEMS])
def test_fit_uncertainty_is_calibrated(ts_adapter, index, record_property):
    run_task(ts_adapter, tasks.calibration_task(*tasks.FIT_PROBLEMS[index], index), record_property)
