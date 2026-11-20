"""L4: the fit problem a code solves, its answer, the Cramér–Rao bound, and a reference least-squares fitter.

A :class:`FitProblem` gives the code a detector spectrum to fit: ``data`` on the pixels ``linspace(*lam_range,
pixels)``, the noise ``variance`` of each pixel, the ``mask`` of pixels in the fit, the ``free`` parameters, and a
``start`` volume and ``start_amplitudes`` holding their start values (every other parameter at its true value). The
code minimizes ``sum over the mask of (data - model)^2 / variance`` and returns a :class:`FitResult`.

The model is the L3 detector spectrum, normalized as at L3 and scaled by free amplitudes: one ``amplitude`` (the
peak counts inside the fit ranges), or, for a code that declares it (``Conventions.amplitudes``), one amplitude per
fit range, ``amplitude1``, ``amplitude2``, ..., each scaling the pixels of its range.

Plasma parameter names: ``ne``, ``Te``, ``ud``, and ``ion<k>.Ti``, ``ion<k>.Va``, ``ion<k>.fract`` with ``k``
counted from 1, in the catalog's units.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

import numpy as np

from .instrument import Detector
from .volume import Volume

_ION = re.compile(r"ion(\d+)\.(Ti|Va|fract)$")
_RANGE_AMPLITUDE = re.compile(r"amplitude(\d+)$")
PLASMA_PARAMETERS = ("ne", "Te", "ud")
AMPLITUDE = "amplitude"


def get(plasma, name):
    """The value of the plasma parameter ``name``."""
    if name in PLASMA_PARAMETERS:
        return getattr(plasma, name)
    match = _ION.match(name)
    if not match:
        raise ValueError(f"unknown parameter {name!r}: use ne, Te, ud or ion<k>.Ti/Va/fract")
    return getattr(plasma.ions[int(match[1]) - 1], match[2])


def replace(plasma, **values):
    """``plasma`` with the named parameters set; dotted names go in as a dict: ``replace(p, **{"ion1.Va": 2})``."""
    plain = {k: v for k, v in values.items() if k in PLASMA_PARAMETERS}
    ions = list(plasma.ions)
    for name, value in values.items():
        if name in PLASMA_PARAMETERS:
            continue
        match = _ION.match(name)
        if not match:
            raise ValueError(f"unknown parameter {name!r}: use ne, Te, ud or ion<k>.Ti/Va/fract")
        k = int(match[1]) - 1
        ions[k] = dataclasses.replace(ions[k], **{match[2]: value})
    return dataclasses.replace(plasma, ions=tuple(ions), **plain)


def base(name):
    """``Va`` for ``ion1.Va``, ``amplitude`` for ``amplitude2``: the key of per-parameter tolerances."""
    return AMPLITUDE if is_amplitude(name) else name.split(".")[-1]


def is_amplitude(name):
    return name == AMPLITUDE or bool(_RANGE_AMPLITUDE.match(name))


def amplitude_names(det, per_range):
    """``("amplitude",)``, or one name per fit range of ``det``."""
    return tuple(f"{AMPLITUDE}{k + 1}" for k in range(len(det.amp_ranges))) if per_range else (AMPLITUDE,)


@dataclass(frozen=True, eq=False)
class FitProblem:
    start: Volume
    start_amplitudes: dict
    detector: Detector
    data: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    free: tuple

    @property
    def centers(self):
        return np.linspace(*self.detector.lam_range, self.detector.pixels)


@dataclass(frozen=True)
class FitResult:
    """``estimates`` and, if the code reports them, ``sigmas`` (one-standard-deviation uncertainties), keyed by
    parameter name."""

    estimates: dict
    sigmas: dict | None = None


def fit_mask(det, centers):
    """Pixels inside the detector's fit ranges."""
    mask = np.zeros(centers.shape, dtype=bool)
    for low, high in det.amp_ranges:
        mask |= (centers > low) & (centers < high)
    return mask


def poisson_variance(expected):
    """The counting-noise variance of an expected spectrum, floored at one count."""
    return np.maximum(np.abs(expected), 1.0)


def gains(det, amplitudes):
    """The factor on each pixel of the L3 spectrum (normalized to ``det.amplitude``) for these ``amplitudes``."""
    centers = np.linspace(*det.lam_range, det.pixels)
    if AMPLITUDE in amplitudes:
        return np.full(centers.shape, amplitudes[AMPLITUDE] / det.amplitude)
    out = np.ones(centers.shape)
    for k, (low, high) in enumerate(det.amp_ranges):
        out[(centers > low) & (centers < high)] = amplitudes[f"{AMPLITUDE}{k + 1}"] / det.amplitude
    return out


def with_values(volume, amplitudes, values):
    """``(volume, amplitudes)`` with the named parameters set."""
    plasma_values = {k: v for k, v in values.items() if not is_amplitude(k)}
    if plasma_values:
        volume = dataclasses.replace(volume, plasma=replace(volume.plasma, **plasma_values))
    return volume, {**amplitudes, **{k: v for k, v in values.items() if is_amplitude(k)}}


def value_of(volume, amplitudes, name):
    return amplitudes[name] if is_amplitude(name) else get(volume.plasma, name)


def predict(model, volume, amplitudes, det):
    """The model detector spectrum for these amplitudes; ``model(volume)`` is normalized to ``det.amplitude``."""
    return model(volume) * gains(det, amplitudes)


def jacobian(model, volume, amplitudes, det, free, step=1.0e-3):
    """``d prediction / d parameter`` by central differences, one column per free parameter."""
    columns = []
    for name in free:
        value = value_of(volume, amplitudes, name)
        h = step * abs(value) if value else step
        up, down = (with_values(volume, amplitudes, {name: value + s}) for s in (h, -h))
        columns.append((predict(model, *up, det) - predict(model, *down, det)) / (2.0 * h))
    return np.stack(columns, axis=1)


def cramer_rao(model, truth: Volume, amplitudes, det, free, variance, mask):
    """``sqrt(diag(F^-1))``, ``F = J^T diag(1/variance) J`` over the fit pixels: the smallest standard deviations an
    unbiased estimate of the free parameters can have, the other free parameters (amplitudes included)
    marginalized."""
    j = jacobian(model, truth, amplitudes, det, free)[mask]
    fisher = j.T @ (j / variance[mask][:, None])
    return np.sqrt(np.diag(np.linalg.inv(fisher)))


def least_squares_fit(model, problem: FitProblem):
    """A reference fitter for Python codes with an L3 model: weighted least squares from the start, with sigmas from
    ``(J^T W J)^-1`` at the solution. ``model(volume)`` returns the detector values normalized to
    ``problem.detector.amplitude``."""
    from scipy.optimize import least_squares

    start = np.array([value_of(problem.start, problem.start_amplitudes, n) for n in problem.free], dtype=float)
    scale = np.where(start != 0.0, np.abs(start), 1.0)
    weight = 1.0 / np.sqrt(problem.variance[problem.mask])
    shapes = {}

    def residuals(x):
        values = dict(zip(problem.free, x * scale))
        volume, amplitudes = with_values(problem.start, problem.start_amplitudes, values)
        key = tuple(v for k, v in values.items() if not is_amplitude(k))
        if key not in shapes:
            shapes[key] = model(volume)
        prediction = shapes[key] * gains(problem.detector, amplitudes)
        return (prediction[problem.mask] - problem.data[problem.mask]) * weight

    solution = least_squares(residuals, start / scale, jac="3-point", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    covariance = np.linalg.inv(solution.jac.T @ solution.jac) * np.outer(scale, scale)
    estimates = dict(zip(problem.free, (solution.x * scale).tolist()))
    sigmas = dict(zip(problem.free, np.sqrt(np.diag(covariance)).tolist()))
    return FitResult(estimates, sigmas)
