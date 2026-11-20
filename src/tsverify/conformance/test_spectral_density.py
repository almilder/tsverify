"""L2: the code's spectral density against the oracle's."""

import pytest

from .. import tasks
from ..cases import SPECTRUM_CASES
from ._support import run_task


@pytest.mark.parametrize("npts", [256, 2048])
@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_spectral_density_matches_the_oracle(ts_adapter, case, npts, record_property):
    run_task(ts_adapter, tasks.spectral_density_task(case, npts), record_property)


def test_high_z_ion_acoustic_resonance(ts_adapter, record_property):
    """Z Te/Ti = 160 puts the IAW resonance at zeta = 9, where tabulated Z' functions usually end."""
    run_task(ts_adapter, tasks.high_z_task(), record_property)


@pytest.mark.slow
@pytest.mark.parametrize("name, plasma", tasks.sweep_plasmas(), ids=[name for name, _ in tasks.sweep_plasmas()])
def test_spectral_density_matches_the_oracle_across_plasmas(ts_adapter, name, plasma, record_property):
    """A1 over a density, temperature and angle matrix (alpha <= 3.5), EPW wings only."""
    run_task(ts_adapter, tasks.sweep_task(name, plasma), record_property)
