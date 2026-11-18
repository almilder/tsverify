"""How a code expresses its inputs and outputs, as opposed to the physics it computes.

tsverify's plasma description is canonical: ion flows ``Va`` and the electron drift ``ud`` are positive along
``k = k_s - k_i``, and ``ud`` is the electron drift relative to the charge-weighted ion fluid. An adapter
declares its code's :class:`Conventions`; tsverify then hands the code the plasma as the code reads it
(:meth:`Conventions.express`) and computes the oracle's spectrum the way the code reports it
(:meth:`Conventions.oracle_options`). The default, :data:`CANONICAL`, is also tsadar's (:data:`TSADAR`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .oracle.physics import PER, RELATIVISTIC, Plasma

ELECTRON_DRIFT = ("ion_frame", "lab")


@dataclass(frozen=True)
class Conventions:
    """``flow_sign``: +1 if the code's flows and drift are positive along ``k_s - k_i``, -1 if along ``k_i - k_s``.
    ``electron_drift``: ``"ion_frame"`` (drift relative to the charge-weighted ion fluid) or ``"lab"`` (the
    electron fluid velocity). ``relativistic``: ``"first_order"`` (the ``1 + 2 omega/omega_L`` factor) or
    ``"none"``. ``spectrum_per``: L2 output per unit ``"wavelength"`` or ``"frequency"``. ``dispersion_in_k``:
    plasma dispersion in the scattered and incident wavenumbers. ``absolute``: L2 output in the oracle's
    absolute units; if False, L2 is compared as a shape (best least-squares scale)."""

    flow_sign: int = 1
    electron_drift: str = "ion_frame"
    relativistic: str = "first_order"
    spectrum_per: str = "wavelength"
    dispersion_in_k: bool = True
    absolute: bool = True

    def __post_init__(self):
        if self.flow_sign not in (1, -1):
            raise ValueError(f"flow_sign must be +1 or -1, got {self.flow_sign!r}")
        for name, allowed in (
            ("electron_drift", ELECTRON_DRIFT),
            ("relativistic", RELATIVISTIC),
            ("spectrum_per", PER),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

    def express(self, plasma: Plasma) -> Plasma:
        """``plasma`` (canonical) written the way the code reads it."""
        if self.flow_sign == 1 and self.electron_drift == "ion_frame":
            return plasma
        ud = plasma.ud + _ion_fluid_velocity(plasma) if self.electron_drift == "lab" else plasma.ud
        return dataclasses.replace(
            plasma,
            ud=self.flow_sign * ud,
            ions=tuple(dataclasses.replace(ion, Va=self.flow_sign * ion.Va) for ion in plasma.ions),
        )

    def interpret(self, plasma: Plasma) -> Plasma:
        """The canonical plasma a code with these conventions models when it reads ``plasma``."""
        if self.flow_sign == 1 and self.electron_drift == "ion_frame":
            return plasma
        canonical = dataclasses.replace(
            plasma,
            ud=self.flow_sign * plasma.ud,
            ions=tuple(dataclasses.replace(ion, Va=self.flow_sign * ion.Va) for ion in plasma.ions),
        )
        if self.electron_drift == "lab":
            canonical = dataclasses.replace(canonical, ud=canonical.ud - _ion_fluid_velocity(canonical))
        return canonical

    def oracle_options(self):
        """Keyword options of :func:`tsverify.oracle.spectral_density` for the code's L2 output."""
        return {"relativistic": self.relativistic, "per": self.spectrum_per, "dispersion_in_k": self.dispersion_in_k}

    def detector_options(self):
        """The same for the spectrum behind an L3 detector spectrum, which is always integrated over wavelength."""
        return {"relativistic": self.relativistic, "dispersion_in_k": self.dispersion_in_k}

    def alternatives(self):
        """``(description, conventions)`` for every single-field change from these conventions."""
        out = []
        for name, allowed in (
            ("flow_sign", (1, -1)),
            ("electron_drift", ELECTRON_DRIFT),
            ("relativistic", RELATIVISTIC),
            ("spectrum_per", PER),
            ("dispersion_in_k", (True, False)),
            ("absolute", (True, False)),
        ):
            for value in allowed:
                if value != getattr(self, name):
                    out.append((f"{_PLAIN[name, value]} ({name}={value!r})", dataclasses.replace(self, **{name: value})))
        return out


_PLAIN = {
    ("flow_sign", 1): "the code's flows and drift are positive along k_s - k_i",
    ("flow_sign", -1): "the code's flows and drift are positive along k_i - k_s",
    ("electron_drift", "ion_frame"): "the code's electron drift is relative to the ion fluid",
    ("electron_drift", "lab"): "the code's electron drift is the lab-frame electron velocity",
    ("relativistic", "first_order"): "the code applies the first-order (1 + 2 omega/omega_L) factor",
    ("relativistic", "none"): "the code applies no (1 + 2 omega/omega_L) factor",
    ("spectrum_per", "wavelength"): "the code reports S per unit wavelength",
    ("spectrum_per", "frequency"): "the code reports S per unit angular frequency",
    ("dispersion_in_k", True): "the code includes plasma dispersion in k",
    ("dispersion_in_k", False): "the code uses vacuum wavenumbers",
    ("absolute", True): "the code's spectrum is in absolute units",
    ("absolute", False): "the code's spectrum is not in absolute units",
}


def _ion_fluid_velocity(plasma: Plasma):
    """The charge-weighted ion flow along k, in 1e6 cm/s (the frame of the canonical ``ud``)."""
    z = np.array([ion.Z for ion in plasma.ions], dtype=float)
    fract = np.array([ion.fract for ion in plasma.ions], dtype=float)
    flows = np.array([ion.Va for ion in plasma.ions], dtype=float)
    return float(np.sum(z * fract * flows) / np.sum(z * fract))


CANONICAL = Conventions()
TSADAR = CANONICAL
