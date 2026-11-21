"""Camera noise for L4 shot problems.

A lineout's counts are ``d = m + n``, with ``m = t + B`` the model spectrum ``t`` plus the background ``B`` (both
in counts), and the noise ``n``:
- shot noise: ``gain`` counts per photoelectron, photoelectron counts with mean ``m / gain`` and variance
  ``excess * m / gain`` (the excess-noise factor F^2 of an intensified or streak camera), spread over
  neighbouring pixels by the CCD spread function, a Gaussian of ``psf_pixels`` standard deviation;
- read noise: ``read_noise`` counts per detector row, summed over the ``rows`` rows of a lineout.

Its covariance is ``K = gain * excess * C diag(m) C^T + rows * read_noise^2 I``, with ``C`` the spread function as
a matrix, the form of tsadar's covar noise model (Swadling et al., Rev. Sci. Instrum. 93, 043503, 2022).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    gain: float
    excess: float
    read_noise: float
    rows: int = 1
    psf_pixels: float = 0.0

    def with_read_noise(self, read_noise):
        return dataclasses.replace(self, read_noise=float(read_noise))

    def spread(self, pixels):
        """``C``: the CCD spread function as a matrix on ``pixels`` pixels (a normalized Gaussian, truncated at the
        detector edges; the identity without a spread function)."""
        if self.psf_pixels <= 0.0:
            return np.eye(pixels)
        half = int(np.ceil(5.0 * self.psf_pixels))
        offsets = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (offsets / self.psf_pixels) ** 2)
        kernel /= kernel.sum()
        c = np.zeros((pixels, pixels))
        for o, k in zip(offsets, kernel):
            c += k * np.eye(pixels, k=o)
        return c

    def shot_covariance(self, mean):
        """The shot-noise part of the covariance of a lineout whose expected counts are ``mean``."""
        c = self.spread(len(mean))
        return self.gain * self.excess * (c * np.maximum(np.asarray(mean, dtype=float), 0.0)) @ c.T

    def covariance(self, mean):
        """``K`` for a lineout whose expected counts (model plus background) are ``mean``."""
        return self.shot_covariance(mean) + self.rows * self.read_noise**2 * np.eye(len(mean))

    def sample(self, mean, rng):
        """One noisy lineout with expected counts ``mean``."""
        mean = np.maximum(np.asarray(mean, dtype=float), 0.0)
        photo = mean / self.gain
        drawn = np.where(photo > 0.0, rng.gamma(np.maximum(photo, 1e-300) / self.excess, self.excess), 0.0)
        shot = self.spread(len(mean)) @ (self.gain * drawn - mean)
        read = rng.normal(0.0, np.sqrt(self.rows) * self.read_noise, mean.shape)
        return mean + shot + read
