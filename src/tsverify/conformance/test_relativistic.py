"""A12: the code's non-collective spectrum against Selden's exact relativistic one."""

import pytest

from .. import tasks
from ._support import run_task


@pytest.mark.parametrize("theta", [45.0, 90.0, 135.0])
@pytest.mark.parametrize("te", [0.5, 2.0, 5.0])
def test_first_order_relativistic_accuracy_against_the_exact_spectrum(ts_adapter, te, theta, record_property):
    run_task(ts_adapter, tasks.relativistic_task(te, theta), record_property)
