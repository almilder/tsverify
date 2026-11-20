"""Every conformance check as a task: the runs a code computes, and how their outputs are scored.

A :class:`Run` is one evaluation by the code under test: Z' on a grid (L1), a spectrum on a wavelength grid (L2) or
a detector spectrum of a scattering volume (L3). A :class:`Task` lists its runs and scores their outputs into a
:class:`Result`. The pytest tests in :mod:`tsverify.conformance` and the file exchange in :mod:`tsverify.exchange`
both score through these tasks, so the two routes give the same results.

Runs hold canonical plasmas; a code receives each plasma written in its declared conventions.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np

from . import oracle
from .adapter import required_features
from .cases import A1_CASES, SPECTRUM_CASES
from .compare import detector_errors, expected_spectral_density, oracle_detector, spectral_errors
from .instrument import Detector
from .oracle import ELECTRON_MASS, ELECTRON_MASS_KEV, Ion, Plasma
from .signatures import explain_spectrum, verdict, zprime_signatures
from .volume import Aperture, LinearGradient, Volume
from .volume import spectral_density as volume_spectral_density

KINDS = ("zprime", "spectral_density", "detector_spectrum")


@dataclass(frozen=True, eq=False)
class Run:
    """One evaluation by the code: ``zprime`` at ``zeta``; ``spectral_density`` of ``plasma`` at ``wavelengths_nm``;
    or ``detector_spectrum`` of ``volume`` on ``detector``."""

    kind: str
    zeta: np.ndarray | None = None
    plasma: Plasma | None = None
    wavelengths_nm: np.ndarray | None = None
    volume: Volume | None = None
    detector: Detector | None = None

    def required_features(self):
        if self.kind == "spectral_density":
            return required_features(self.plasma)
        if self.kind == "detector_spectrum":
            return required_features(self.volume.plasma, self.detector, self.volume)
        return set()


@dataclass
class Result:
    """``status`` is ``"passed"``, ``"failed"`` or ``"not supported"``. ``measurements`` are the recorded values;
    ``checks`` are ``(what, measured, limit)`` rows for the report."""

    task: str
    test: str
    level: str
    status: str
    measurements: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class Context:
    """What scoring needs to know about the code: its declared conventions and tolerances, and whether it has an
    L1 Z' (for the hint after an unexplained failure)."""

    conventions: object
    tolerances: object
    has_zprime: bool = False


@dataclass(frozen=True, eq=False)
class Task:
    id: str
    test: str
    level: str
    runs: dict
    score: Callable
    slow: bool = False
    applies: Callable | None = None

    def not_applicable(self, conventions):
        """Why the task does not apply to a code with these conventions, or None."""
        return self.applies(conventions) if self.applies else None

    def result(self, context, outputs):
        status, measurements, checks, message = self.score(context, outputs)
        return Result(self.id, self.test, self.level, status, measurements, checks, message)

    def unsupported(self, reason):
        return Result(self.id, self.test, self.level, "not supported", message=reason)


def _verdict(passed, measurements, checks, message):
    return ("passed" if passed else "failed"), measurements, checks, ("" if passed else message)


def _hint(context):
    if context.has_zprime:
        return " The L1 Z' test shows whether the plasma dispersion function is behind it."
    return " An adapter zprime(zeta) would let the L1 Z' test check the plasma dispersion function."


# L1

ZETA = np.linspace(-15.0, 15.0, 30001)


def zprime_task():
    def score(context, out):
        error = out["zprime"] - oracle.plasma_dispersion_derivative(ZETA)
        worst = int(np.argmax(np.abs(error)))
        size = float(np.abs(error[worst]))
        tol = context.tolerances.zprime_abs
        readings = [] if size < tol else (zprime_signatures(ZETA, error, tol) or ["No known signature."])
        message = (
            f"L1: Z' departs from the Faddeeva function by up to {size:.1e} at zeta = {ZETA[worst]:+.3f} "
            f"(tol {tol:.1e}). " + " ".join(readings)
        )
        return _verdict(size < tol, {"max |Z' error|": size}, [("max |Z' error|", size, tol)], message)

    return Task("L1/zprime", "test_zprime_matches_faddeeva", "L1", {"zprime": Run("zprime", zeta=ZETA)}, score)


# L2


def _explained(context, plasma, lam, model, agrees):
    return verdict(*explain_spectrum(context.conventions, plasma, lam, model, agrees), otherwise=_hint(context))


def spectral_density_task(case, npts):
    lam = np.linspace(*case.detector.lam_range, npts)

    def score(context, out):
        model = out["spectrum"]
        expected = expected_spectral_density(context.conventions, case.plasma, lam)
        linf, l1 = spectral_errors(context.conventions, model, expected)
        linf_tol, l1_tol = context.tolerances.spectral_density[case.problem]
        passed = linf < linf_tol and l1 < l1_tol
        message = (
            f"{case.problem}-L2/{case}: S(k, w) departs from the oracle: "
            f"peak-normalized Linf={linf:.2e} (tol {linf_tol:.1e}), L1={l1:.2e} (tol {l1_tol:.1e}). "
        )
        if not passed:
            message += _explained(context, case.plasma, lam, model, lambda a, b: a < linf_tol and b < l1_tol)
        return _verdict(passed, {"Linf": linf, "L1": l1}, [("Linf", linf, linf_tol), ("L1", l1, l1_tol)], message)

    run = Run("spectral_density", plasma=case.plasma, wavelengths_nm=lam)
    return Task(f"L2/{case.id}/{npts}", "test_spectral_density_matches_the_oracle", "L2", {"spectrum": run}, score)


HIGH_Z = Plasma(ne=0.5, Te=1.0, ions=(Ion(A=197.0, Z=40.0, Ti=0.25),))
HIGH_Z_LAM = np.linspace(525.0, 528.0, 8192)


def high_z_task():
    """Z Te/Ti = 160 puts the IAW resonance at zeta = 9, where tabulated Z' functions usually end."""

    def score(context, out):
        model = out["spectrum"]
        expected = expected_spectral_density(context.conventions, HIGH_Z, HIGH_Z_LAM)
        linf, _ = spectral_errors(context.conventions, model, expected)
        tol = context.tolerances
        checks = [("Linf", linf, tol.high_z_linf)]
        if linf >= tol.high_z_linf:
            message = f"high-Z IAW departs from the oracle by {linf:.2e} (tol {tol.high_z_linf:.0e}). " + _explained(
                context, HIGH_Z, HIGH_Z_LAM, model, lambda a, _: a < tol.high_z_linf
            )
            return _verdict(False, {"Linf": linf}, checks, message)
        for window in ((525.0, 526.45), (526.55, 528.0)):
            shift = oracle.peak_wavelength(HIGH_Z_LAM, model, window) - oracle.peak_wavelength(
                HIGH_Z_LAM, expected, window
            )
            checks.append((f"peak shift in {window} nm", abs(shift), tol.high_z_peak_nm))
            if abs(shift) >= tol.high_z_peak_nm:
                return _verdict(False, {"Linf": linf}, checks, f"high-Z IAW peak in {window} nm moved by {shift:+.6f} nm")
        return _verdict(True, {"Linf": linf}, checks, "")

    run = Run("spectral_density", plasma=HIGH_Z, wavelengths_nm=HIGH_Z_LAM)
    return Task("L2/high-z", "test_high_z_ion_acoustic_resonance", "L2", {"spectrum": run}, score)


SWEEP_MAX_ALPHA = 3.5
SWEEP_LAM_RANGE = (400.0, 700.0)
SWEEP_LAM = np.linspace(*SWEEP_LAM_RANGE, 4096)


def _min_k_lambda_at_epw_peaks(plasma):
    lam = np.linspace(*SWEEP_LAM_RANGE, 30001)
    spectrum = oracle.spectral_density(lam, plasma)
    values = []
    for side in (lam < plasma.lam - 3.0, lam > plasma.lam + 3.0):
        k, _, _ = oracle.scattering_geometry(lam[np.argmax(np.where(side, spectrum, 0.0))], plasma)
        values.append(k * np.sqrt(plasma.Te / ELECTRON_MASS) / oracle.plasma_frequency(plasma.ne))
    return min(values)


@cache
def sweep_plasmas():
    """``(id, plasma)`` for the A1 sweep: density, temperature and angle with alpha <= 3.5."""
    out = []
    for ne, te, angle in itertools.product((0.05, 0.2, 0.5), (0.1, 0.6, 2.0), (30.0, 60.0, 120.0)):
        plasma = Plasma(ne=ne, Te=te, ions=(Ion(A=1.0, Z=1.0, Ti=te / 5),), angle_deg=angle)
        if 1.0 / _min_k_lambda_at_epw_peaks(plasma) <= SWEEP_MAX_ALPHA:
            out.append((f"ne{ne}-Te{te}-{angle:.0f}deg", plasma))
    return tuple(out)


def sweep_task(name, plasma):
    """A1 over a density, temperature and angle matrix (alpha <= 3.5), EPW wings only."""

    def score(context, out):
        model = out["spectrum"]
        wings = np.abs(SWEEP_LAM - plasma.lam) > 3.0
        expected = expected_spectral_density(context.conventions, plasma, SWEEP_LAM[wings])
        linf, _ = spectral_errors(context.conventions, model[wings], expected)
        tol = context.tolerances.sweep_linf
        message = f"A1-sweep: EPW wings depart from the oracle by {linf:.2e} (tol {tol:.0e})"
        return _verdict(linf < tol, {"Linf": linf}, [("Linf (EPW wings)", linf, tol)], message)

    run = Run("spectral_density", plasma=plasma, wavelengths_nm=SWEEP_LAM)
    return Task(
        f"L2-sweep/{name}", "test_spectral_density_matches_the_oracle_across_plasmas", "L2", {"spectrum": run}, score,
        slow=True,
    )


@dataclass(frozen=True)
class Probe:
    name: str
    plasma: Plasma
    lam_range: tuple[float, float]
    sensitive_to: str


HYDROGEN = Ion(A=1.0, Z=1.0, Ti=0.08)
PROBES = [
    Probe("flow", Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, Va=3.0),)), (524.5, 528.5), "flow sign"),
    Probe("drift", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), ud=60.0), (440.0, 630.0), "drift sign"),
    Probe(
        "drift-frame",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, Va=30.0),), ud=60.0),
        (440.0, 630.0),
        "electron drift frame",
    ),
    Probe(
        "wings",
        Plasma(ne=0.5, Te=1.0, ions=(Ion(A=1.0, Z=1.0, Ti=0.2),)),
        (420.0, 660.0),
        "relativistic factor, spectral density per wavelength or frequency, and dispersion in k",
    ),
]


def probe_task(probe):
    lam = np.linspace(*probe.lam_range, 2048)

    def score(context, out):
        model = out["spectrum"]
        declared = context.conventions
        linf_tol, l1_tol = context.tolerances.spectral_density["A1"]
        linf, l1 = spectral_errors(declared, model, expected_spectral_density(declared, probe.plasma, lam))
        passed = linf < linf_tol and l1 < l1_tol
        message = (
            f"probe {probe.name!r} (sensitive to the {probe.sensitive_to}): the spectrum departs from the oracle under "
            f"the declared conventions (Linf={linf:.1e}, L1={l1:.1e}; tol {linf_tol:.0e}, {l1_tol:.0e}). "
        )
        if not passed:
            message += _explained(context, probe.plasma, lam, model, lambda a, b: a < linf_tol and b < l1_tol)
        checks = [("Linf", linf, linf_tol), ("L1", l1, l1_tol)]
        return _verdict(passed, {"Linf under the declared conventions": linf}, checks, message)

    run = Run("spectral_density", plasma=probe.plasma, wavelengths_nm=lam)
    return Task(f"probe/{probe.name}", "test_convention_probe", "probe", {"spectrum": run}, score)


# A12


def _peak(x, y):
    i = int(np.argmax(y))
    left, mid, right = y[i - 1 : i + 2]
    return x[i] + 0.5 * (left - right) / (left - 2.0 * mid + right) * (x[1] - x[0])


def relativistic_task(te, theta):
    plasma = Plasma(ne=1.0e-6, Te=te, ions=(Ion(A=1.0, Z=1.0, Ti=te),), angle_deg=theta)
    lam = np.linspace(0.45 * plasma.lam, 1.6 * plasma.lam, 8192)

    def score(context, out):
        model = out["spectrum"]
        if context.conventions.spectrum_per == "frequency":
            model = model / lam**2
        eps = lam / plasma.lam - 1.0
        exact = oracle.selden_spectrum(eps, theta, te)
        model, exact = model / model.max(), exact / exact.max()
        beta2 = te / ELECTRON_MASS_KEV
        s2 = np.sin(np.deg2rad(theta) / 2.0) ** 2
        second_order_shift = 14.0 * s2 * (1.0 + 7.0 * s2) * beta2**2
        shape = float(np.abs(model - exact).max())
        shift = _peak(eps, model) - _peak(eps, exact)
        measurements = {
            "shape error / (Te/mc^2)": float(shape / beta2),
            "peak error / second-order shift": float(shift / second_order_shift),
        }
        tol = context.tolerances
        checks = [
            ("shape error / (Te/mc^2)", shape / beta2, tol.relativistic_shape),
            ("|peak error| / second-order shift", abs(shift) / second_order_shift, tol.relativistic_peak),
        ]
        if not shape < tol.relativistic_shape * beta2:
            return _verdict(False, measurements, checks, f"A12: shape off the exact spectrum by {shape:.2e} = {shape / beta2:.2f} Te/mc^2")
        if not abs(shift) < tol.relativistic_peak * second_order_shift:
            return _verdict(
                False, measurements, checks,
                f"A12: peak off by {shift:+.5f} = {shift / second_order_shift:.2f} of the second-order shift",
            )
        return _verdict(True, measurements, checks, "")

    def applies(conventions):
        return "declares no relativistic factor" if conventions.relativistic == "none" else None

    run = Run("spectral_density", plasma=plasma, wavelengths_nm=lam)
    return Task(
        f"A12/Te{te}-{theta:.0f}deg", "test_first_order_relativistic_accuracy_against_the_exact_spectrum", "A12",
        {"spectrum": run}, score, applies=applies,
    )


# L3 and A10


def _expected_detector(context, volume, det, centers):
    options = context.conventions.detector_options()
    return oracle_detector(det, centers, lambda lam: volume_spectral_density(lam, volume, **options))


def _detector_score(label, volume, det, check_centers=False):
    def score(context, out):
        centers, model = out["detector"]
        if check_centers:
            moved = float(np.max(np.abs(centers - np.linspace(*det.lam_range, det.pixels))))
            if not moved <= 1e-9:
                return _verdict(False, {}, [("pixel centre offset (nm)", moved, 1e-9)], f"{label}: detector pixel centres moved")
        expected = _expected_detector(context, volume, det, centers)
        passed, measurements, checks, message = detector_errors(label, det, centers, model, expected, context.tolerances)
        return _verdict(passed, measurements, checks, message)

    return score


def detector_task(case):
    det = case.detector
    volume = Volume(case.plasma)
    run = Run("detector_spectrum", volume=volume, detector=det)
    score = _detector_score(f"{case.problem}-L3/{case}", volume, det, check_centers=True)
    return Task(f"L3/{case.id}", "test_detector_spectrum_matches_the_oracle", "L3", {"detector": run}, score)


EPW_CASE, _, IAW_CASE, CH_CASE = A1_CASES
GRADIENT = LinearGradient(ne_percent=20.0, Te_percent=20.0, points=5)
APERTURE = Aperture((54.0, 60.0, 66.0), (0.25, 0.5, 0.25))


def gradient_task(case):
    det = case.detector
    volume = Volume(case.plasma, gradient=GRADIENT)
    run = Run("detector_spectrum", volume=volume, detector=det)
    score = _detector_score(f"A10-gradient/{case}", volume, det)
    return Task(
        f"A10/gradient/{case.id}", "test_gradient_average_matches_the_averaged_oracle_spectrum", "A10",
        {"detector": run}, score,
    )


def aperture_task():
    det = EPW_CASE.detector
    volume = Volume(EPW_CASE.plasma, aperture=APERTURE)
    run = Run("detector_spectrum", volume=volume, detector=det)
    return Task(
        "A10/aperture", "test_aperture_weighted_sum_matches_the_oracle", "A10", {"detector": run},
        _detector_score("A10-aperture", volume, det),
    )


def _close(a, b, rtol, atol):
    deviation = np.abs(a - b)
    return bool(np.all(deviation <= atol + rtol * np.abs(b))), float(deviation.max())


def zero_gradient_task():
    det = EPW_CASE.detector
    runs = {
        "single": Run("detector_spectrum", volume=Volume(EPW_CASE.plasma), detector=det),
        "flat": Run("detector_spectrum", volume=Volume(EPW_CASE.plasma, gradient=LinearGradient(points=5)), detector=det),
    }

    def score(context, out):
        single, flat = out["single"][1], out["flat"][1]
        tol = context.tolerances.zero_gradient
        passed, deviation = _close(flat, single, tol, tol)
        message = f"A10: zero gradient changed the spectrum by up to {deviation:.2e}"
        return _verdict(passed, {"max deviation": deviation}, [("max deviation", deviation, tol)], message)

    return Task("A10/zero-gradient", "test_zero_gradient_collapses_to_a_single_point", "A10", runs, score)


REWRITTEN = (
    ("species-order", CH_CASE, dataclasses.replace(CH_CASE.plasma, ions=CH_CASE.plasma.ions[::-1])),
    (
        "species-split",
        IAW_CASE,
        dataclasses.replace(
            IAW_CASE.plasma, ions=tuple(dataclasses.replace(IAW_CASE.plasma.ions[0], fract=f) for f in (0.3, 0.7))
        ),
    ),
)


def rewritten_task(name, case, same_plasma):
    runs = {
        "reference": Run("detector_spectrum", volume=Volume(case.plasma), detector=case.detector),
        "rewritten": Run("detector_spectrum", volume=Volume(same_plasma), detector=case.detector),
    }

    def score(context, out):
        reference, rewritten = out["reference"][1], out["rewritten"][1]
        tol = context.tolerances.rewritten_plasma
        passed, deviation = _close(rewritten, reference, tol, tol * reference.max())
        relative = deviation / float(reference.max())
        message = f"A10: {case} rewritten changes the spectrum by up to {relative:.2e} of its peak"
        return _verdict(passed, {"max deviation / peak": relative}, [("max deviation / peak", relative, tol)], message)

    return Task(
        f"A10/rewritten/{name}", "test_the_same_plasma_written_differently_gives_the_same_spectrum", "A10", runs, score
    )


def all_tasks(slow=True):
    """Every task, in the order of the conformance tests."""
    tasks = [zprime_task()]
    tasks += [spectral_density_task(case, npts) for npts in (256, 2048) for case in SPECTRUM_CASES]
    tasks.append(high_z_task())
    if slow:
        tasks += [sweep_task(name, plasma) for name, plasma in sweep_plasmas()]
    tasks += [probe_task(p) for p in PROBES]
    tasks += [relativistic_task(te, theta) for te in (0.5, 2.0, 5.0) for theta in (45.0, 90.0, 135.0)]
    tasks += [detector_task(case) for case in SPECTRUM_CASES]
    tasks += [gradient_task(case) for case in (EPW_CASE, IAW_CASE)]
    tasks += [zero_gradient_task(), aperture_task()]
    tasks += [rewritten_task(*r) for r in REWRITTEN]
    return tasks
