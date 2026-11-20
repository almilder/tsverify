"""Known-defect signatures: what a code's spectrum or Z' looks like when a known mistake is behind a failure.

When a spectrum test fails, :func:`explain_spectrum` retries the oracle under every single-convention change
(:meth:`tsverify.conventions.Conventions.alternatives`) and with every defect in :data:`SPECTRUM_SIGNATURES`, and
names the ones under which the code agrees. :func:`zprime_signatures` reads the error of a code's Z' for table
edges, interpolation between table nodes and the error of tsadar's legacy table.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import oracle
from .compare import expected_spectral_density, spectral_errors


@dataclass(frozen=True)
class Signature:
    """A known defect: ``spectrum(conventions, plasma, wavelengths_nm)`` is the oracle's spectrum with it."""

    name: str
    description: str
    spectrum: Callable


def _temperatures_scaled(electrons=1.0, ions=1.0):
    def spectrum(conventions, plasma, wavelengths_nm):
        scaled = dataclasses.replace(
            plasma,
            Te=plasma.Te * electrons,
            ions=tuple(dataclasses.replace(ion, Ti=ion.Ti * ions) for ion in plasma.ions),
        )
        return expected_spectral_density(conventions, scaled, wavelengths_nm)

    return spectrum


def _electron_f_prime_differenced(step):
    def spectrum(conventions, plasma, wavelengths_nm):
        lam = np.asarray(wavelengths_nm, dtype=float)
        _, electron_part, ion_part, chi_e, chi_i = oracle.spectral_density(
            lam, plasma, terms=True, **conventions.oracle_options()
        )
        k, omega, _ = oracle.scattering_geometry(lam, plasma, dispersion_in_k=conventions.dispersion_in_k)
        x, k_lambda_de, _ = oracle.electron_phase_velocity(k, omega, plasma)
        slope = np.diff(oracle.mixture_density(x, plasma.electrons)) / np.diff(x)
        f_prime = np.append(slope, slope[-1]) if step == 1 else np.insert(slope, 0, slope[0])
        chi_defect = chi_e.real - 1j * np.pi * f_prime / k_lambda_de**2
        screening = np.abs(1.0 + chi_e + chi_i) ** 2 / np.abs(1.0 + chi_defect + chi_i) ** 2
        return screening * (electron_part + ion_part * np.abs(chi_defect) ** 2 / np.abs(chi_e) ** 2)

    return spectrum


_SQRT2 = "as when v_T = sqrt(2T/m) is used where sqrt(T/m) belongs"
_SQRT2_BACK = "as when v_T = sqrt(T/m) is used where sqrt(2T/m) belongs"

SPECTRUM_SIGNATURES = (
    Signature("Te-doubled", f"the electron temperature enters as 2 Te, {_SQRT2}", _temperatures_scaled(electrons=2.0)),
    Signature("Te-halved", f"the electron temperature enters as Te/2, {_SQRT2_BACK}", _temperatures_scaled(electrons=0.5)),
    Signature("Ti-doubled", f"the ion temperatures enter as 2 Ti, {_SQRT2}", _temperatures_scaled(ions=2.0)),
    Signature("Ti-halved", f"the ion temperatures enter as Ti/2, {_SQRT2_BACK}", _temperatures_scaled(ions=0.5)),
    Signature("T-doubled", f"every temperature enters doubled, {_SQRT2}", _temperatures_scaled(2.0, 2.0)),
    Signature("T-halved", f"every temperature enters halved, {_SQRT2_BACK}", _temperatures_scaled(0.5, 0.5)),
    Signature(
        "forward-difference-f'",
        "the electron Landau damping takes f' as a difference between each wavelength sample and the next, "
        "an O(h) error that shrinks only with finer sampling",
        _electron_f_prime_differenced(1),
    ),
    Signature(
        "backward-difference-f'",
        "the electron Landau damping takes f' as a difference between each wavelength sample and the previous "
        "one, an O(h) error that shrinks only with finer sampling",
        _electron_f_prime_differenced(-1),
    ),
)


def explain_spectrum(conventions, plasma, wavelengths_nm, model, agrees):
    """``(conventions, defects)``: the single-convention changes and known defects under which ``model`` agrees
    with the oracle, in plain words. ``agrees(linf, l1)`` is the failing test's tolerance."""
    lam = np.asarray(wavelengths_nm, dtype=float)
    found_conventions = []
    for description, alternative in conventions.alternatives():
        modelled = alternative.interpret(conventions.express(plasma))
        linf, l1 = spectral_errors(alternative, model, expected_spectral_density(alternative, modelled, lam))
        if agrees(linf, l1):
            found_conventions.append(f"{description} [Linf={linf:.1e}]")
    found_defects = []
    for signature in SPECTRUM_SIGNATURES:
        linf, l1 = spectral_errors(conventions, model, signature.spectrum(conventions, plasma, lam))
        if agrees(linf, l1):
            found_defects.append(f"{signature.description} ({signature.name}) [Linf={linf:.1e}]")
    return found_conventions, found_defects


def verdict(found_conventions, found_defects, otherwise=""):
    """One sentence per kind of explanation, or the physics-error verdict followed by ``otherwise``."""
    parts = []
    if found_conventions:
        parts.append(
            "It agrees with the oracle if " + "; or if ".join(found_conventions) + ", so check the declared conventions."
        )
    if found_defects:
        parts.append("It matches a known defect: " + "; or ".join(found_defects) + ".")
    if parts:
        return " ".join(parts)
    return (
        "No single convention change or known defect explains it, so this looks like a physics error, "
        "not a convention mismatch." + otherwise
    )


RDWT_PEAK_ZETA = 1.37
RDWT_PEAK_ERROR = 1.63e-3
JUMP_CONTRAST = 10.0
EDGE_MERGE = 0.05
NODE_REGION = 4.0
NODE_DEPTH = 0.05
MIN_NODES = 6


def zprime_signatures(zeta, error, tol):
    """Readings of ``error = Z'_code - Z'`` on a fine uniform ``zeta`` grid: table edges (jumps), linear
    interpolation between table nodes, and the error of tsadar's legacy rdWT table."""
    zeta = np.asarray(zeta, dtype=float)
    error = np.asarray(error)
    readings = []

    step = np.abs(np.diff(error))
    padded = np.pad(step, 3, mode="edge")
    neighbours = np.max([np.roll(padded, s)[3:-3] for s in (-3, -2, -1, 1, 2, 3)], axis=0)
    jumps = np.flatnonzero((step > tol / 4) & (step > JUMP_CONTRAST * neighbours))
    edges = []
    for where, size in sorted((abs(0.5 * (zeta[i] + zeta[i + 1])), step[i]) for i in jumps):
        if edges and where - edges[-1][2] <= EDGE_MERGE:
            edges[-1] = (edges[-1][0], max(edges[-1][1], size), where)
        else:
            edges.append((where, size, where))
    for where, size, _ in edges:
        readings.append(
            f"Z' jumps by {size:.1e} at |zeta| = {where:.2f}, the signature of a table edge (a tabulated Z' that "
            "switches to an asymptotic form or a clamped value there)."
        )

    inside = np.abs(zeta) <= NODE_REGION
    size = np.abs(error[inside])
    z = zeta[inside]
    if size.size > 2 and size.max() > 0:
        minima = (size[1:-1] <= size[:-2]) & (size[1:-1] <= size[2:]) & (size[1:-1] < NODE_DEPTH * size.max())
        nodes = z[1:-1][minima]
        spacing = np.diff(nodes)
        if nodes.size >= MIN_NODES and np.std(spacing) < 0.05 * np.mean(spacing):
            readings.append(
                f"The error vanishes every {np.mean(spacing):.3g} in zeta: linear interpolation between table "
                "nodes that far apart."
            )

    real = np.real(error)
    peak = np.argmax(np.abs(real))
    if abs(abs(zeta[peak]) - RDWT_PEAK_ZETA) < 0.05 and abs(real[peak] / RDWT_PEAK_ERROR - 1.0) < 0.15:
        readings.append(
            f"Re Z' is {real[peak]:.2e} too high at |zeta| = {abs(zeta[peak]):.2f}: the error of the rdWT table "
            "tsadar used before November 2026."
        )
    return readings
