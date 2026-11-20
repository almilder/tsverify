"""Running a task as a pytest test: skip what the code does not support, record measurements, fail with the
diagnosis."""

import pytest

from .. import report
from ..execute import score


def run_task(adapter, task, record_property):
    result = score(adapter, task)
    report.collect(adapter, result)
    if result.status == "not supported":
        pytest.skip(result.message)
    for name, value in result.measurements.items():
        record_property(name, value)
    if result.status == "failed":
        pytest.fail(result.message)
