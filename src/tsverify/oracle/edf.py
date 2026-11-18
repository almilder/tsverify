"""Velocity distributions and their susceptibility response: Gaussian mixtures through Faddeeva, other EDFs
through principal-value quadrature."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import wofz


@dataclass(frozen=True)
class GaussianComponent:
    """One drifting Maxwellian of an EDF, in units of the species' v_T: weight, shift and width."""

    weight: float = 1.0
    shift: float = 0.0
    width: float = 1.0


def plasma_dispersion(zeta):
    r"""Fried–Conte plasma dispersion function ``Z(zeta) = i sqrt(pi) w(zeta)``."""
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def maxwellian_response(x):
    """``1 + zeta Z(zeta)``, ``zeta = x / sqrt(2)``: ``(k lambda_D)^2 chi`` of a unit-variance Maxwellian."""
    zeta = np.asarray(x) / np.sqrt(2.0)
    return 1.0 + zeta * plasma_dispersion(zeta)


class QuadratureEDF:
    """An EDF with ``(k lambda_D)^2 chi(x) = -P.V. int f'(t)/(t - x) dt - i pi f'(x)`` by numerical quadrature.

    Subclasses define ``density`` and ``derivative`` on unit-variance velocity.
    """

    X_MAX = 25.0
    X_STEP = 0.02
    LIMIT = 200.0

    def density(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def response(self, x):
        x = np.asarray(x, dtype=float)
        spline, moments = _quadrature_response_table(self)

        def series(y):
            return -sum((n + 1.0) * m / y ** (n + 2) for n, m in enumerate(moments))

        def tail(edge):
            y = np.where(np.sign(x) == np.sign(edge), x, edge)
            y = np.where(np.abs(y) < self.X_MAX, edge, y)
            return series(y) + (spline(edge) - series(edge)) * (edge / y) ** 5

        real = np.where(x > self.X_MAX, tail(self.X_MAX), np.where(x < -self.X_MAX, tail(-self.X_MAX), 0.0))
        real = np.where(np.abs(x) <= self.X_MAX, spline(np.clip(x, -self.X_MAX, self.X_MAX)), real)
        return real - 1j * np.pi * self.derivative(x)


@lru_cache(maxsize=None)
def _quadrature_response_table(edf):
    from scipy.integrate import quad
    from scipy.interpolate import CubicSpline

    a, b = -edf.LIMIT, edf.LIMIT

    def principal_value(x):
        fx = edf.derivative(x)
        smooth = quad(lambda t: (edf.derivative(t) - fx) / (t - x), a, b, points=[x], limit=500, epsabs=1e-13)[0]
        return smooth + fx * np.log((b - x) / (x - a))

    grid = np.arange(-edf.X_MAX, edf.X_MAX + edf.X_STEP / 2, edf.X_STEP)
    moments = [quad(lambda t: t**n * edf.density(t), a, b, limit=500)[0] for n in range(4)]
    return CubicSpline(grid, -np.array([principal_value(x) for x in grid])), moments


@dataclass(frozen=True, eq=True)
class GaussianEDF(QuadratureEDF):
    """A unit-variance Maxwellian through the quadrature path: the check that the path reproduces Faddeeva."""

    def density(self, x):
        return np.exp(-0.5 * np.asarray(x) ** 2) / np.sqrt(2.0 * np.pi)

    def derivative(self, x):
        return -np.asarray(x) * self.density(x)


@dataclass(frozen=True, eq=True)
class KappaEDF(QuadratureEDF):
    """1-D kappa distribution: a Student-t with ``nu = 2 kappa - 1``, scaled to unit variance."""

    kappa: float

    @property
    def nu(self):
        return 2.0 * self.kappa - 1.0

    @property
    def scale(self):
        return np.sqrt((self.nu - 2.0) / self.nu)

    def density(self, x):
        from scipy.special import gammaln

        nu = self.nu
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
        u = np.asarray(x) / self.scale
        return np.exp(log_norm - (nu + 1.0) / 2.0 * np.log1p(u**2 / nu)) / self.scale

    def derivative(self, x):
        u = np.asarray(x) / self.scale
        return -(self.nu + 1.0) * u / (self.nu + u**2) * self.density(x) / self.scale


def mixture_susceptibility(x, k_lambda_d, components):
    """``chi`` at normalized phase velocity ``x`` for a Gaussian mixture (sum of shifted, scaled Faddeeva terms)
    or a :class:`QuadratureEDF`.
    """
    if isinstance(components, QuadratureEDF):
        return components.response(x) / k_lambda_d**2
    chi = 0.0
    for c in components:
        chi = chi + c.weight * maxwellian_response((x - c.shift) / c.width) / c.width**2
    return chi / k_lambda_d**2


def mixture_density(x, components):
    """The normalized EDF ``f(x)`` (units of v_T) of a Gaussian mixture or a :class:`QuadratureEDF`."""
    if isinstance(components, QuadratureEDF):
        return components.density(x)
    f = 0.0
    for c in components:
        f = f + c.weight * np.exp(-0.5 * ((x - c.shift) / c.width) ** 2) / (c.width * np.sqrt(2.0 * np.pi))
    return f
