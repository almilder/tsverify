"""A small Maxwellian Thomson-scattering code, written apart from the oracle, with switchable known defects.

``ToyAdapter()`` is correct and passes every test it supports. ``ToyAdapter(defect)`` plants one of
:data:`DEFECTS` while still declaring the canonical conventions; tsverify's self-tests check that each defect is
caught and named.
"""

import numpy as np
from scipy.special import wofz

from ..adapter import Adapter
from ..oracle.constants import (
    C_CM_PER_S,
    CLASSICAL_ELECTRON_RADIUS_CM,
    ELECTRON_CHARGE_SQUARED,
    ELECTRON_MASS,
    PROTON_MASS,
)

DEFECTS = {
    "flow-sign": "flows and drift taken along k_i - k_s",
    "no-relativistic-factor": "the (1 + 2 omega/omega_L) factor left out",
    "forward-difference-f'": "the electron f' taken as a forward difference along the wavelength samples",
    "thermal-speed": "the electron v_T = sqrt(2 Te/me) used where sqrt(Te/me) belongs",
    "zprime-table": "Z' interpolated linearly from a table with step 0.1 that ends at |zeta| = 8",
}
TABLE_STEP = 0.1
TABLE_EDGE = 8.0


def _zprime(zeta):
    return -2.0 * (1.0 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta))


class ToyAdapter(Adapter):
    features = frozenset({"multiple_ions", "ion_flow", "electron_drift"})

    def __init__(self, defect=None):
        if defect is not None and defect not in DEFECTS:
            raise ValueError(f"defect must be one of {sorted(DEFECTS)}, got {defect!r}")
        self.defect = defect
        self.name = "toy" if defect is None else f"toy[{defect}]"

    def zprime(self, zeta):
        zeta = np.asarray(zeta, dtype=float)
        if self.defect != "zprime-table":
            return _zprime(zeta)
        nodes = np.arange(-TABLE_EDGE, TABLE_EDGE + TABLE_STEP / 2, TABLE_STEP)
        table = _zprime(nodes)
        inside = np.abs(zeta) <= TABLE_EDGE
        outside = np.where(inside, 1.0, zeta)
        real = np.where(inside, np.interp(zeta, nodes, table.real), outside**-2.0)
        imag = np.where(inside, np.interp(zeta, nodes, table.imag), 0.0)
        return real + 1j * imag

    def spectral_density(self, plasma, wavelengths_nm):
        lam_cm = np.asarray(wavelengths_nm, dtype=float) * 1.0e-7
        omega_s = 2.0 * np.pi * C_CM_PER_S / lam_cm
        omega_l = 2.0 * np.pi * C_CM_PER_S / (plasma.lam * 1.0e-7)
        omega = omega_s - omega_l
        ne = plasma.ne * 1.0e20
        omega_pe = np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ne / ELECTRON_MASS)
        k_s = np.sqrt(omega_s**2 - omega_pe**2) / C_CM_PER_S
        k_l = np.sqrt(omega_l**2 - omega_pe**2) / C_CM_PER_S
        k = np.sqrt(k_s**2 + k_l**2 - 2.0 * k_s * k_l * np.cos(np.deg2rad(plasma.angle_deg)))
        sign = -1.0 if self.defect == "flow-sign" else 1.0

        charge = np.array([ion.Z for ion in plasma.ions])
        share = np.array([ion.fract for ion in plasma.ions])
        share = share / share.sum()
        zbar = np.sum(charge * share)
        ion_fluid = np.sum(charge * share * np.array([ion.Va for ion in plasma.ions])) / zbar

        chi_i = 0.0
        ions = 0.0
        for ion, part in zip(plasma.ions, share):
            mass = ion.A * PROTON_MASS
            v_t = np.sqrt(ion.Ti / mass)
            omega_p = np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ion.Z**2 * part * ne / zbar / mass)
            zeta = (omega - sign * k * ion.Va * 1.0e6) / (np.sqrt(2.0) * k * v_t)
            chi_i = chi_i - self.zprime(zeta) / (2.0 * (k * v_t / omega_p) ** 2)
            ions = ions + part * ion.Z**2 / zbar * np.exp(-(zeta**2)) / (np.sqrt(2.0 * np.pi) * v_t)

        v_te = np.sqrt((2.0 if self.defect == "thermal-speed" else 1.0) * plasma.Te / ELECTRON_MASS)
        k_lambda = k * v_te / omega_pe
        zeta_e = (omega - sign * k * (ion_fluid + plasma.ud) * 1.0e6) / (np.sqrt(2.0) * k * v_te)
        chi_e = -self.zprime(zeta_e) / (2.0 * k_lambda**2)
        if self.defect == "forward-difference-f'":
            x = np.sqrt(2.0) * zeta_e
            slope = np.diff(np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)) / np.diff(x)
            chi_e = chi_e.real - 1j * np.pi * np.append(slope, slope[-1]) / k_lambda**2
        electrons = np.exp(-(zeta_e**2)) / (np.sqrt(2.0 * np.pi) * v_te)

        s = (np.abs(1.0 + chi_i) ** 2 * electrons + np.abs(chi_e) ** 2 * ions) / np.abs(1.0 + chi_e + chi_i) ** 2 / k
        factor = 1.0 if self.defect == "no-relativistic-factor" else 1.0 + 2.0 * omega / omega_l
        return s * factor * CLASSICAL_ELECTRON_RADIUS_CM**2 * ne * 2.0 * np.pi * C_CM_PER_S / lam_cm**2
