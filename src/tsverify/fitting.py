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

A :class:`ShotProblem` is several lineouts of one shot that share a camera (:mod:`tsverify.camera`). Its lineouts
carry no variance: the code is given the background and the camera's nominal parameters, and sizes the noise
itself. It returns a :class:`ShotResult`.
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


@dataclass(frozen=True, eq=False)
class ShotProblem:
    """``lineouts``: one :class:`FitProblem` per lineout, with ``variance`` None and ``data`` that include the
    background. ``backgrounds``: the background of each lineout, in counts, which the model adds to the spectrum.
    ``camera``: the nominal camera (:class:`tsverify.camera.Camera`); its gain, excess-noise factor, rows and
    spread function are right, its read noise may not be."""

    lineouts: tuple
    backgrounds: tuple
    camera: object


@dataclass(frozen=True)
class FitResult:
    """``estimates`` and, if the code reports them, ``sigmas`` (one-standard-deviation uncertainties), keyed by
    parameter name."""

    estimates: dict
    sigmas: dict | None = None


@dataclass(frozen=True)
class ShotResult:
    """One :class:`FitResult` per lineout, and optionally the noise the code sized, e.g. ``{"read_noise": ...}``."""

    fits: tuple
    noise: dict | None = None


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


def least_squares_fit(model, problem: FitProblem, covariance=None, offset=None):
    """A reference fitter for Python codes with an L3 model: weighted least squares from the start, with sigmas from
    ``(J^T W J)^-1`` at the solution. ``model(volume)`` returns the detector values normalized to
    ``problem.detector.amplitude``. ``covariance``: a full noise covariance on the fit pixels, used instead of
    ``problem.variance``; ``offset``: counts added to the prediction (a background)."""
    from scipy.linalg import cholesky, solve_triangular
    from scipy.optimize import least_squares

    start = np.array([value_of(problem.start, problem.start_amplitudes, n) for n in problem.free], dtype=float)
    scale = np.where(start != 0.0, np.abs(start), 1.0)
    if covariance is None:
        weight = 1.0 / np.sqrt(problem.variance[problem.mask])

        def whiten(r):
            return r * weight
    else:
        lower = cholesky(covariance, lower=True)

        def whiten(r):
            return solve_triangular(lower, r, lower=True)
    shapes = {}

    def residuals(x):
        values = dict(zip(problem.free, x * scale))
        volume, amplitudes = with_values(problem.start, problem.start_amplitudes, values)
        key = tuple(v for k, v in values.items() if not is_amplitude(k))
        if key not in shapes:
            shapes[key] = model(volume)
        prediction = shapes[key] * gains(problem.detector, amplitudes)
        if offset is not None:
            prediction = prediction + offset
        return whiten(prediction[problem.mask] - problem.data[problem.mask])

    solution = least_squares(residuals, start / scale, jac="3-point", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    covariance = np.linalg.inv(solution.jac.T @ solution.jac) * np.outer(scale, scale)
    estimates = dict(zip(problem.free, (solution.x * scale).tolist()))
    sigmas = dict(zip(problem.free, np.sqrt(np.diag(covariance)).tolist()))
    return FitResult(estimates, sigmas)


def size_read_noise(camera, means, residuals, parameters):
    """Maximum-likelihood read noise of a shot, the other camera parameters fixed: the residuals of all lineouts
    (on all pixels), with ``means`` the fitted model plus background, inflated by ``N/(N - parameters)`` for the
    fitted parameters."""
    from scipy.optimize import minimize_scalar

    decomposed = []
    for mean, residual in zip(means, residuals):
        n = len(residual)
        eigen, vectors = np.linalg.eigh(camera.shot_covariance(mean))
        decomposed.append((np.maximum(eigen, 0.0), (vectors.T @ residual) ** 2 * n / (n - parameters)))

    def negative_log_likelihood(log_variance):
        floor = camera.rows * np.exp(log_variance)
        return sum(np.sum(y2 / (eig + floor) + np.log(eig + floor)) for eig, y2 in decomposed)

    best = minimize_scalar(negative_log_likelihood, bounds=(np.log(1e-8), np.log(1e10)), method="bounded",
                           options={"xatol": 1e-6})
    return float(np.sqrt(np.exp(best.x)))


def sized_shot_fit(model, problem: ShotProblem):
    """The reference shot fitter: fit every lineout with the full covariance of the nominal camera, size the read
    noise from the pooled residuals (:func:`size_read_noise`), and fit again with it. ``model(volume)`` returns the
    detector values normalized to the detector amplitude."""
    camera = problem.camera
    current = [(lineout.start, lineout.start_amplitudes) for lineout in problem.lineouts]
    read = camera.read_noise
    fits = []
    for round_ in range(2):
        noise = camera.with_read_noise(read)
        fits = []
        for lineout, background, (volume, amplitudes) in zip(problem.lineouts, problem.backgrounds, current):
            mean = predict(model, volume, amplitudes, lineout.detector) + background
            covariance = noise.covariance(mean)[np.ix_(lineout.mask, lineout.mask)]
            start = dataclasses.replace(lineout, start=volume, start_amplitudes=amplitudes)
            fits.append(least_squares_fit(model, start, covariance=covariance, offset=background))
        current = [with_values(lineout.start, lineout.start_amplitudes, fit.estimates)
                   for lineout, fit in zip(problem.lineouts, fits)]
        if round_ == 0:
            means = [predict(model, volume, amplitudes, lineout.detector) + background
                     for lineout, background, (volume, amplitudes) in zip(problem.lineouts, problem.backgrounds, current)]
            residuals = [lineout.data - mean for lineout, mean in zip(problem.lineouts, means)]
            read = size_read_noise(camera, means, residuals, len(problem.lineouts[0].free))
    return ShotResult(tuple(fits), {"read_noise": read})
