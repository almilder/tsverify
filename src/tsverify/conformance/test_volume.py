"""A10: gradient, aperture and species averaging against the oracle, and plasmas written two ways."""

import pytest

from .. import tasks
from ._support import run_task


@pytest.mark.parametrize("case", [tasks.EPW_CASE, tasks.IAW_CASE], ids=str)
def test_gradient_average_matches_the_averaged_oracle_spectrum(ts_adapter, case, record_property):
    run_task(ts_adapter, tasks.gradient_task(case), record_property)


def test_zero_gradient_collapses_to_a_single_point(ts_adapter, record_property):
    run_task(ts_adapter, tasks.zero_gradient_task(), record_property)


def test_aperture_weighted_sum_matches_the_oracle(ts_adapter, record_property):
    run_task(ts_adapter, tasks.aperture_task(), record_property)


@pytest.mark.parametrize("name, case, same_plasma", tasks.REWRITTEN, ids=[r[0] for r in tasks.REWRITTEN])
def test_the_same_plasma_written_differently_gives_the_same_spectrum(ts_adapter, name, case, same_plasma, record_property):
    run_task(ts_adapter, tasks.rewritten_task(name, case, same_plasma), record_property)
