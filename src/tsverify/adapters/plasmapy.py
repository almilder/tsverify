"""PlasmaPy's ``plasmapy.diagnostics.thomson.spectral_density`` as a code under test (``pip install plasmapy``).

PlasmaPy is a code under test here, never an oracle. It models each species as a sum of drifting Maxwellian
populations with lab-frame velocities, reports ``S(k, omega)`` in s/rad without the ``(1 + 2 omega/omega_L)``
factor, and includes plasma dispersion in k: :data:`PLASMAPY`. Its ``S`` integrates to ``2 pi`` times the
oracle's normalization, so the adapter divides by ``2 pi`` and multiplies by ``r_e^2 n_e`` to report absolute units.

A Gaussian-mixture EDF becomes one population per component. Velocities along k are placed on PlasmaPy's own
k direction, ``scatter_vec - probe_vec``. Kappa and other quadrature EDFs are not supported.
"""

import numpy as np

from ..adapter import Adapter, NotSupported
from ..conventions import Conventions
from ..oracle import ELECTRON_MASS, PROTON_MASS, QuadratureEDF
from ..oracle.constants import C_CM_PER_S, CLASSICAL_ELECTRON_RADIUS_CM

PLASMAPY = Conventions(electron_drift="lab", relativistic="none", spectrum_per="frequency")


def _fractions(weights):
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    weights[-1] = 1.0 - np.sum(weights[:-1])
    return weights


class PlasmaPyAdapter(Adapter):
    name = "plasmapy"
    features = frozenset({"multiple_ions", "ion_flow", "electron_drift", "numeric_electron_edf", "numeric_ion_edf"})
    conventions = PLASMAPY

    def __init__(self):
        import astropy.units as u
        from plasmapy.diagnostics import thomson
        from plasmapy.dispersion.dispersion_functions import plasma_dispersion_func_deriv
        from plasmapy.particles import CustomParticle, ParticleList

        self._u = u
        self._thomson = thomson
        self._zprime = plasma_dispersion_func_deriv
        self._custom_particle = CustomParticle
        self._particle_list = ParticleList

    def zprime(self, zeta):
        return np.asarray(self._zprime(np.asarray(zeta, dtype=float)))

    def spectral_density(self, plasma, wavelengths_nm):
        u = self._u
        if isinstance(plasma.electrons, QuadratureEDF) or any(isinstance(i.components, QuadratureEDF) for i in plasma.ions):
            raise NotSupported("PlasmaPy models Maxwellian populations only")
        theta = np.deg2rad(plasma.angle_deg)
        probe = np.array([1.0, 0.0, 0.0])
        scatter = np.array([np.cos(theta), np.sin(theta), 0.0])
        along_k = (scatter - probe) / np.linalg.norm(scatter - probe)

        def velocity(v_1e6_cm_per_s):
            return np.outer(np.asarray(v_1e6_cm_per_s, dtype=float) * 1.0e4, along_k) * u.m / u.s

        v_te = np.sqrt(plasma.Te / ELECTRON_MASS) * 1.0e-6
        electrons = plasma.electrons
        ions, ion_fract, ion_t, ion_v = [], [], [], []
        share = np.array([ion.fract for ion in plasma.ions], dtype=float)
        share = share / share.sum()
        for ion, part in zip(plasma.ions, share):
            mass = (ion.A * PROTON_MASS * C_CM_PER_S**2 * u.keV).to(u.kg, equivalencies=u.mass_energy())
            v_ti = np.sqrt(ion.Ti / (ion.A * PROTON_MASS)) * 1.0e-6
            for c in ion.components:
                ions.append(self._custom_particle(mass=mass, Z=ion.Z))
                ion_fract.append(part * c.weight)
                ion_t.append(ion.Ti * c.width**2)
                ion_v.append(ion.Va + c.shift * v_ti)

        _, skw = self._thomson.spectral_density(
            np.asarray(wavelengths_nm, dtype=float) * u.nm,
            plasma.lam * u.nm,
            plasma.ne * 1.0e26 * u.m**-3,
            T_e=self._kelvin([plasma.Te * c.width**2 for c in electrons]),
            T_i=self._kelvin(ion_t),
            efract=_fractions([c.weight for c in electrons]),
            ifract=_fractions(ion_fract),
            ions=self._particle_list(ions),
            electron_vel=velocity([plasma.ud + c.shift * v_te for c in electrons]),
            ion_vel=velocity(ion_v),
            probe_vec=probe,
            scatter_vec=scatter,
        )
        return skw.to(u.s / u.rad).value / (2.0 * np.pi) * CLASSICAL_ELECTRON_RADIUS_CM**2 * plasma.ne * 1.0e20

    def _kelvin(self, kev):
        u = self._u
        return (np.asarray(kev, dtype=float) * u.keV).to(u.K, equivalencies=u.temperature_energy())
