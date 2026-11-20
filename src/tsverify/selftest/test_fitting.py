"""L4 plumbing: parameter names, the Cramér–Rao bound, the reference fitter, and how fit results are scored."""

import numpy as np
import pytest

from tsverify import fitting, tasks
from tsverify.adapters.oracle import OracleAdapter
from tsverify.cases import A1_CASES
from tsverify.conventions import CANONICAL, Conventions
from tsverify.execute import score
from tsverify.fitting import AMPLITUDE, FitResult
from tsverify.volume import Volume

CH = A1_CASES[3].plasma
NAME, CASE, START = tasks.FIT_PROBLEMS[1]


def test_parameter_names_read_and_write_the_plasma():
    plasma = fitting.replace(CH, ne=0.3, **{"ion2.Va": 7.0, "ion1.Ti": 0.5})
    assert fitting.get(plasma, "ne") == 0.3 and fitting.get(plasma, "ion2.Va") == 7.0
    assert fitting.get(plasma, "ion1.Ti") == 0.5 and fitting.get(plasma, "ion2.Ti") == CH.ions[1].Ti
    with pytest.raises(ValueError, match="unknown parameter"):
        fitting.get(CH, "ion1.A")


def test_the_bound_on_the_amplitude_alone_is_the_linear_one():
    det = tasks._counts_detector(CASE)
    model = tasks.oracle_model(CANONICAL, det)
    shape = model(Volume(CASE.plasma))
    mask = fitting.fit_mask(det, np.linspace(*det.lam_range, det.pixels))
    variance = fitting.poisson_variance(shape)
    (bound,) = fitting.cramer_rao(model, Volume(CASE.plasma), {AMPLITUDE: det.amplitude}, det, (AMPLITUDE,), variance, mask)
    unit = shape[mask] / det.amplitude
    np.testing.assert_allclose(bound, 1.0 / np.sqrt(np.sum(unit**2 / variance[mask])), rtol=1e-6)


class BoundFitter(OracleAdapter):
    """Reports the start values (the truth, in the Cramér–Rao tasks) with ``factor`` times the bound as sigmas."""

    def __init__(self, factor=1.0, sigmas=True, drop=None):
        super().__init__()
        self.factor, self.with_sigmas, self.drop = factor, sigmas, drop

    def fit(self, problem):
        def model(volume):
            return self.detector_spectrum(volume, problem.detector)[1]

        bound = fitting.cramer_rao(
            model, problem.start, problem.start_amplitudes, problem.detector, problem.free, problem.variance, problem.mask
        )
        estimates = {n: fitting.value_of(problem.start, problem.start_amplitudes, n) for n in problem.free if n != self.drop}
        sigmas = {n: self.factor * b for n, b in zip(problem.free, bound)} if self.with_sigmas else None
        return FitResult(estimates, sigmas)


def test_sigmas_at_the_bound_pass_and_sqrt2_too_large_is_named():
    task = tasks.cramer_rao_task(NAME, CASE, START, "oracle")
    assert score(BoundFitter(), task).status == "passed"
    inflated = score(BoundFitter(np.sqrt(2.0)), task)
    assert inflated.status == "failed" and "off by the same factor, sqrt(2)" in inflated.message


def test_a_fit_without_sigmas_or_with_a_missing_estimate():
    task = tasks.cramer_rao_task(NAME, CASE, START, "oracle")
    result = score(BoundFitter(sigmas=False), task)
    assert result.status == "not supported" and "no uncertainties" in result.message
    missing = score(BoundFitter(drop=AMPLITUDE), task)
    assert missing.status == "failed" and "no estimate for 'amplitude'" in missing.message


def test_estimates_return_to_canonical_conventions():
    conventions = Conventions(flow_sign=-1)
    problem = tasks._problem(CASE, START, tasks.COUNTS, tasks._oracle_data(CASE))(conventions, {})
    result = FitResult({"ion1.Va": -1.5, "ion1.Ti": 0.08, AMPLITUDE: 1.0e4}, None)
    estimates, _ = tasks.canonical_estimates(result, problem, conventions)
    assert estimates["ion1.Va"] == 1.5 and estimates["ion1.Ti"] == 0.08


def test_per_range_amplitudes_scale_their_own_pixels():
    det = tasks._counts_detector(tasks.EPW_CASE)
    names = Conventions(per_range_amplitudes=("EPW",)).amplitude_names(det)
    assert names == ("amplitude1", "amplitude2") and CANONICAL.amplitude_names(det) == (AMPLITUDE,)
    g = fitting.gains(det, {"amplitude1": 2.0 * det.amplitude, "amplitude2": 3.0 * det.amplitude})
    centers = np.linspace(*det.lam_range, det.pixels)
    (b_lo, b_hi), (r_lo, r_hi) = det.amp_ranges
    assert np.all(g[(centers > b_lo) & (centers < b_hi)] == 2.0) and np.all(g[(centers > r_lo) & (centers < r_hi)] == 3.0)


def test_per_range_sigmas_at_the_bound_pass():
    task = tasks.cramer_rao_task(*tasks.FIT_PROBLEMS[0], "oracle")
    adapter = BoundFitter()
    adapter.conventions = Conventions(per_range_amplitudes=("EPW",))
    result = score(adapter, task)
    assert result.status == "passed" and "amplitude2: sigma / cramer-rao" in result.measurements


def test_calibration_widths_are_read():
    assert "scatter more than the reported sigma" in tasks._width_reading([1.44, 1.0], 0.38)
    assert "overestimated" in tasks._width_reading([0.5, 1.0], 0.38)
    assert tasks._width_reading([1.1, 0.9], 0.38) == ""
