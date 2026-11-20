"""The analytic oracle: NumPy and SciPy only, no dependency on any code under test."""

from .constants import (
    C_CM_PER_S,
    CLASSICAL_ELECTRON_RADIUS_CM,
    ELECTRON_CHARGE_SQUARED,
    ELECTRON_MASS,
    ELECTRON_MASS_KEV,
    PROTON_MASS,
)
from .detector import detector_spectrum, normalize_in_ranges, pixel_edges, rectangular_notch
from .edf import (
    GaussianComponent,
    GaussianEDF,
    KappaEDF,
    QuadratureEDF,
    maxwellian_response,
    mixture_density,
    mixture_susceptibility,
    plasma_dispersion,
    plasma_dispersion_derivative,
)
from .metrics import peak_wavelength, relative_errors
from .physics import Ion, Plasma, electron_phase_velocity, form_factor_terms, plasma_frequency, scattering_geometry, spectral_density
from .relativistic import selden_spectrum

__all__ = [
    "C_CM_PER_S",
    "CLASSICAL_ELECTRON_RADIUS_CM",
    "ELECTRON_CHARGE_SQUARED",
    "ELECTRON_MASS",
    "ELECTRON_MASS_KEV",
    "PROTON_MASS",
    "GaussianComponent",
    "GaussianEDF",
    "Ion",
    "KappaEDF",
    "Plasma",
    "QuadratureEDF",
    "detector_spectrum",
    "electron_phase_velocity",
    "form_factor_terms",
    "maxwellian_response",
    "mixture_density",
    "mixture_susceptibility",
    "normalize_in_ranges",
    "peak_wavelength",
    "pixel_edges",
    "plasma_dispersion",
    "plasma_dispersion_derivative",
    "plasma_frequency",
    "rectangular_notch",
    "relative_errors",
    "scattering_geometry",
    "selden_spectrum",
    "spectral_density",
]
