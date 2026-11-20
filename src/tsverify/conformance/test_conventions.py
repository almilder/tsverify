"""Convention probes: small asymmetric plasmas that tell a convention mismatch apart from a physics error.

Each probe compares the code's L2 spectrum with the oracle's under the declared conventions. If it fails, the probe
retries every single-convention change and every known defect (:mod:`tsverify.signatures`) and names the ones
under which the code agrees with the oracle.
"""

import pytest

from .. import tasks
from ..tasks import PROBES, Probe  # noqa: F401
from ._support import run_task


@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.name)
def test_convention_probe(ts_adapter, probe, record_property):
    run_task(ts_adapter, tasks.probe_task(probe), record_property)
