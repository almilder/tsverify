"""Camera noise, read-noise sizing, the covariance-weighted fitter, and two fitters converging with the counts."""

import dataclasses

import numpy as np

from tsverify import convergence, tasks
from tsverify.adapters.oracle import OracleAdapter
from tsverify.camera import Camera
from tsverify.conventions import CANONICAL
from tsverify.fitting import least_squares_fit, size_read_noise

CAMERA = Camera(gain=2.0, excess=1.15, read_noise=30.0, rows=3, psf_pixels=1.0)


def test_the_sampled_noise_has_the_stated_covariance():
    mean = np.linspace(50.0, 5000.0, 24)
    rng = np.random.default_rng(1)
    draws = np.array([CAMERA.sample(mean, rng) for _ in range(6000)])
    expected = CAMERA.covariance(mean)
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=4.0 * np.sqrt(np.diag(expected) / 6000).max())
    empirical = np.cov(draws.T)
    np.testing.assert_allclose(np.diag(empirical), np.diag(expected), rtol=0.08)
    variance = np.diag(expected)
    standard_error = np.sqrt((variance[:-1] * variance[1:] + np.diag(expected, 1) ** 2) / 6000)
    np.testing.assert_allclose(np.diag(empirical, 1), np.diag(expected, 1), atol=4.0 * standard_error.max())
    assert np.all(np.abs(np.diag(empirical, 1) - np.diag(expected, 1)) < 4.0 * standard_error)


def test_without_a_spread_function_the_noise_is_uncorrelated():
    plain = dataclasses.replace(CAMERA, psf_pixels=0.0)
    k = plain.covariance(np.full(5, 100.0))
    np.testing.assert_allclose(k, np.diag(np.full(5, 2.0 * 1.15 * 100.0 + 3 * 30.0**2)))


def test_the_read_noise_is_sized_from_pooled_residuals():
    rng = np.random.default_rng(2)
    means = [np.linspace(10.0, 8000.0, 128) for _ in range(8)]
    residuals = [CAMERA.sample(m, rng) - m for m in means]
    told = CAMERA.with_read_noise(5.0)
    assert abs(size_read_noise(told, means, residuals, 0) / 30.0 - 1.0) < 0.05


def test_a_diagonal_covariance_reproduces_the_weighted_fit():
    name, case, start = tasks.FIT_PROBLEMS[1]
    at_truth = {k: tasks._truth(case, k) for k in start}
    problem = tasks._problem(case, at_truth, tasks.COUNTS, tasks._oracle_data(case))(CANONICAL, {})
    model = tasks.oracle_model(CANONICAL, problem.detector)
    weighted = least_squares_fit(model, problem)
    full = least_squares_fit(model, problem, covariance=np.diag(problem.variance[problem.mask]))
    for p in problem.free:
        assert abs(weighted.estimates[p] - full.estimates[p]) <= 1e-9 * abs(weighted.estimates[p])
        assert abs(weighted.sigmas[p] / full.sigmas[p] - 1.0) < 1e-6


class ConstantWeights(OracleAdapter):
    """The oracle fitted with constant weights: consistent, but not efficient."""

    def fit(self, problem):
        return super().fit(dataclasses.replace(problem, variance=np.ones_like(problem.variance)))


def test_two_consistent_fitters_converge_like_the_noise():
    result = convergence.compare_fitters(OracleAdapter(), ConstantWeights(), problem_index=0, realizations=4)
    for p, slope in result.slopes.items():
        assert -0.7 < slope < -0.3, (p, slope)
    for p in result.slopes:
        scaled = [result.rms_difference_over_sigma[c][p] for c in result.counts]
        assert max(scaled) / min(scaled) < 3.0, (p, scaled)


def test_shot_problems_are_python_route_only():
    from tsverify import exchange

    shots = [t for t in tasks.all_tasks(slow=True) if t.id.startswith("L4/camera/")]
    assert len(shots) == len(tasks.FIT_PROBLEMS) * len(tasks.CAMERA_REGIMES)
    assert all(exchange.not_exported(t, CANONICAL) == exchange.SHOT_PYTHON_ONLY for t in shots)


def test_the_difference_estimator_sizes_the_read_noise():
    from tsverify.fitting import size_noise_from_differences

    rng = np.random.default_rng(3)
    means = [np.linspace(10.0, 8000.0, 128) for _ in range(8)]
    residuals = [CAMERA.sample(m, rng) - m for m in means]
    _, read = size_noise_from_differences(CAMERA.with_read_noise(5.0), means, residuals)
    assert abs(read / 30.0 - 1.0) < 0.05
