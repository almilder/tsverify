"""L1: the code's Z' against the Faddeeva function, on a grid that runs past where tables usually end."""

from .. import tasks
from ..tasks import ZETA  # noqa: F401
from ._support import run_task


def test_zprime_matches_faddeeva(ts_adapter, record_property):
    run_task(ts_adapter, tasks.zprime_task(), record_property)
