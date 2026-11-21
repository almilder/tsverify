"""Two fitters on the same data at several count levels.

In the small-noise limit every consistent fitter reaches the same answer, so the difference between two codes' (or
two configurations of one code's) estimates on the same noisy data shrinks like the noise, about counts^(-1/2),
and stays a roughly fixed fraction of sigma. A difference that stops shrinking as the counts rise is systematic:
the two fitters disagree beyond the noise, for example through different model errors.

    result = compare_fitters(adapter_a, adapter_b)
    result.slopes        # d log(rms difference) / d log(counts) per parameter, about -0.5 when both converge
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import fitting, tasks
from .execute import express_problem
from .fitting import FitProblem
from .volume import Volume

COUNT_LEVELS = (1.0e3, 1.0e4, 1.0e5)
SEED = 20261122


@dataclass
class Comparison:
    """Per count level and parameter: the rms difference of the two fitters' estimates, and its rms in units of
    fitter A's sigma; ``slopes``: the fitted log-log slope of the rms difference against the counts."""

    counts: tuple
    rms_difference: dict
    rms_difference_over_sigma: dict
    slopes: dict


def _problem(case, start, counts, conventions, data):
    det = dataclasses.replace(case.detector, amplitude=counts)
    amplitudes = conventions.amplitude_names(det)
    volume, _ = fitting.with_values(Volume(case.plasma), {}, start)
    mask = fitting.fit_mask(det, np.linspace(*det.lam_range, det.pixels))
    expected = tasks.oracle_model(conventions, det)(Volume(case.plasma))
    return FitProblem(volume, dict.fromkeys(amplitudes, counts), det, data, fitting.poisson_variance(expected), mask,
                      tuple(start) + amplitudes)


def _fit(adapter, problem):
    result = adapter.fit(express_problem(problem, adapter.conventions))
    return tasks.canonical_estimates(result, problem, adapter.conventions)


def compare_fitters(adapter_a, adapter_b, problem_index=0, counts=COUNT_LEVELS, realizations=8):
    """Fit the same noisy oracle data (Gaussian, variance = expected counts) with both adapters at each count level,
    from the truth, and compare their estimates of the plasma parameters (amplitudes excluded, since the two may
    model them differently). Both adapters must support ``fit`` and report sigmas for fitter A."""
    name, case, start = tasks.FIT_PROBLEMS[problem_index]
    at_truth = {k: tasks._truth(case, k) for k in start}
    rms, rms_sigma = {}, {}
    for level in counts:
        det = dataclasses.replace(case.detector, amplitude=level)
        expected = tasks.oracle_model(adapter_a.conventions, det)(Volume(case.plasma))
        rng = np.random.default_rng((SEED, problem_index, int(level)))
        differences, scaled = {p: [] for p in start}, {p: [] for p in start}
        for _ in range(realizations):
            data = expected + rng.standard_normal(expected.shape) * np.sqrt(fitting.poisson_variance(expected))
            est_a, sig_a = _fit(adapter_a, _problem(case, at_truth, level, adapter_a.conventions, data))
            est_b, _ = _fit(adapter_b, _problem(case, at_truth, level, adapter_b.conventions, data))
            for p in start:
                differences[p].append(est_a[p] - est_b[p])
                scaled[p].append((est_a[p] - est_b[p]) / sig_a[p])
        rms[level] = {p: float(np.sqrt(np.mean(np.square(v)))) for p, v in differences.items()}
        rms_sigma[level] = {p: float(np.sqrt(np.mean(np.square(v)))) for p, v in scaled.items()}
    logs = np.log(np.asarray(counts, dtype=float))
    slopes = {p: float(np.polyfit(logs, np.log([max(rms[c][p], 1e-300) for c in counts]), 1)[0]) for p in start}
    return Comparison(tuple(counts), rms, rms_sigma, slopes)
