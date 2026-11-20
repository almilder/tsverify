"""The collisionless Thomson-scattering spectrum of a homogeneous plasma.

Units: ne in 1e20 cm^-3, temperatures in keV, flows in 1e6 cm/s along k, wavelengths in nm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_CM_PER_S, CLASSICAL_ELECTRON_RADIUS_CM, ELECTRON_CHARGE_SQUARED, ELECTRON_MASS, PROTON_MASS
from .edf import GaussianComponent, QuadratureEDF, mixture_density, mixture_susceptibility


@dataclass(frozen=True)
class Ion:
    """One ion species. ``Va`` is the flow projected on k, in 1e6 cm/s; ``components`` is its EDF as a
    Gaussian mixture in units of sqrt(Ti/mi) (default: Maxwellian)."""

    A: float
    Z: float
    Ti: float
    fract: float = 1.0
    Va: float = 0.0
    components: tuple[GaussianComponent, ...] = (GaussianComponent(),)


@dataclass(frozen=True)
class Plasma:
    """A homogeneous plasma at one scattering angle. ``ud`` is the electron drift relative to the charge-weighted
    ion fluid; ``electrons`` is a Gaussian mixture or a :class:`QuadratureEDF`.
    """

    ne: float
    Te: float
    ions: tuple[Ion, ...]
    angle_deg: float = 60.0
    lam: float = 526.5
    ud: float = 0.0
    electrons: tuple[GaussianComponent, ...] = field(default_factory=lambda: (GaussianComponent(),))

    def __post_init__(self):
        for label, components in [("electron", self.electrons)] + [(f"ion {i}", ion.components) for i, ion in enumerate(self.ions)]:
            if isinstance(components, QuadratureEDF):
                continue
            total = sum(c.weight for c in components)
            if not np.isclose(total, 1.0, rtol=0, atol=1e-12):
                raise ValueError(f"{label} mixture weights must sum to 1, got {total}")


def _angular_frequency(lam_nm):
    return 2.0 * np.pi * C_CM_PER_S / (np.asarray(lam_nm, dtype=float) * 1.0e-7)


def plasma_frequency(ne_1e20):
    """Electron plasma frequency in rad/s for a density in 1e20 cm^-3."""
    return np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ne_1e20 * 1.0e20 / ELECTRON_MASS)


def electron_phase_velocity(k, omega, plasma: Plasma):
    """``(x_e, k lambda_De, v_Te)``: the electrons' normalized phase velocity in the electron fluid frame."""
    z = np.array([ion.Z for ion in plasma.ions], dtype=float)
    fract = np.array([ion.fract for ion in plasma.ions], dtype=float)
    fract = fract / fract.sum()
    flows = np.array([ion.Va for ion in plasma.ions], dtype=float) * 1.0e6
    electron_flow = float(np.sum(z * fract * flows) / np.sum(z * fract)) + plasma.ud * 1.0e6
    v_te = np.sqrt(plasma.Te / ELECTRON_MASS)
    k_lambda_de = v_te * k / plasma_frequency(plasma.ne)
    x_e = (omega - k * electron_flow) / (k * v_te)
    return x_e, k_lambda_de, v_te


def form_factor_terms(k, omega, plasma: Plasma):
    """Electron and ion parts of the collisionless ``S(k, omega)`` (normalized so ``int S d omega`` is the static
    structure factor), with ``chi_e`` and ``chi_i``.
    """
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)
    ne = plasma.ne * 1.0e20

    z = np.array([ion.Z for ion in plasma.ions], dtype=float)
    fract = np.array([ion.fract for ion in plasma.ions], dtype=float)
    fract = fract / fract.sum()
    zbar = float(np.sum(z * fract))

    chi_i = 0.0
    ion_term = 0.0
    for ion, frac in zip(plasma.ions, fract):
        mass = ion.A * PROTON_MASS
        n_s = frac * ne / zbar
        omega_ps = np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ion.Z**2 * n_s / mass)
        v_t = np.sqrt(ion.Ti / mass)
        x = (omega - k * ion.Va * 1.0e6) / (k * v_t)
        chi_i = chi_i + mixture_susceptibility(x, v_t * k / omega_ps, ion.components)
        ion_term = ion_term + (frac * ion.Z**2 / zbar) * mixture_density(x, ion.components) / v_t

    x_e, k_lambda_de, v_te = electron_phase_velocity(k, omega, plasma)
    chi_e = mixture_susceptibility(x_e, k_lambda_de, plasma.electrons)
    f_e = mixture_density(x_e, plasma.electrons) / v_te

    eps2 = np.abs(1.0 + chi_e + chi_i) ** 2
    electron_part = np.abs(1.0 + chi_i) ** 2 * f_e / eps2 / k
    ion_part = np.abs(chi_e) ** 2 * ion_term / eps2 / k
    return electron_part, ion_part, chi_e, chi_i


def scattering_geometry(lam_nm, plasma: Plasma, *, dispersion_in_k=True):
    """``(k, omega, omega_l)`` along the spectrometer axis: the frequency shift and the Bragg wavenumber
    of light scattered at ``lam_nm`` through ``plasma.angle_deg``, with the plasma dispersion of both waves
    (``dispersion_in_k=False``: vacuum wavenumbers ``omega / c``)."""
    omega_pe = plasma_frequency(plasma.ne)
    omega_s = _angular_frequency(lam_nm)
    omega_l = _angular_frequency(plasma.lam)
    if dispersion_in_k:
        k_s = np.sqrt(omega_s**2 - omega_pe**2) / C_CM_PER_S
        k_l = np.sqrt(omega_l**2 - omega_pe**2) / C_CM_PER_S
    else:
        k_s = omega_s / C_CM_PER_S
        k_l = omega_l / C_CM_PER_S
    theta = np.deg2rad(plasma.angle_deg)
    k = np.sqrt(k_s**2 + k_l**2 - 2.0 * k_s * k_l * np.cos(theta))
    return k, omega_s - omega_l, omega_l


RELATIVISTIC = ("first_order", "none")
PER = ("wavelength", "frequency")


def spectral_density(lam_nm, plasma: Plasma, *, terms=False, relativistic="first_order", per="wavelength",
                     dispersion_in_k=True):
    """``S`` in tsadar's absolute units: :func:`form_factor_terms` times ``(1 + 2 omega/omega_L)``, ``r_e^2 n_e``
    and ``d omega / d lambda``. ``terms=True`` also returns the parts and susceptibilities.

    ``relativistic="none"`` drops ``(1 + 2 omega/omega_L)``; ``per="frequency"`` gives ``S`` per unit angular
    frequency (rad/s) instead of per wavelength (cm); ``dispersion_in_k`` is passed to :func:`scattering_geometry`.
    """
    if relativistic not in RELATIVISTIC:
        raise ValueError(f"relativistic must be one of {RELATIVISTIC}, got {relativistic!r}")
    if per not in PER:
        raise ValueError(f"per must be one of {PER}, got {per!r}")
    lam_nm = np.asarray(lam_nm, dtype=float)
    k, omega, omega_l = scattering_geometry(lam_nm, plasma, dispersion_in_k=dispersion_in_k)
    electron_part, ion_part, chi_e, chi_i = form_factor_terms(k, omega, plasma)

    factor = (1.0 + 2.0 * omega / omega_l) if relativistic == "first_order" else 1.0
    scale = factor * CLASSICAL_ELECTRON_RADIUS_CM**2 * plasma.ne * 1.0e20
    jacobian = 2.0 * np.pi * C_CM_PER_S / (lam_nm * 1.0e-7) ** 2 if per == "wavelength" else 1.0
    electron_part = electron_part * scale * jacobian
    ion_part = ion_part * scale * jacobian
    total = electron_part + ion_part
    if terms:
        return total, electron_part, ion_part, chi_e, chi_i
    return total
