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
from . import fitting
from .compare import detector_errors, expected_spectral_density, oracle_detector, spectral_errors
from .camera import Camera
from .fitting import FitProblem, ShotProblem
from .instrument import Detector
from .oracle import ELECTRON_MASS, ELECTRON_MASS_KEV, Ion, Plasma
from .signatures import explain_spectrum, verdict, zprime_signatures
from .volume import Aperture, LinearGradient, Volume
from .volume import spectral_density as volume_spectral_density

KINDS = ("zprime", "spectral_density", "detector_spectrum", "fit", "fit_shot")


@dataclass(frozen=True, eq=False)
class Run:
    """One evaluation by the code: ``zprime`` at ``zeta``; ``spectral_density`` of ``plasma`` at ``wavelengths_nm``;
    ``detector_spectrum`` of ``volume`` on ``detector``; or a ``fit`` of the problem ``problem(conventions, outputs)``
    builds (canonical, with ``volume`` the truth), where ``outputs`` holds the outputs of the runs in ``needs``."""

    kind: str
    zeta: np.ndarray | None = None
    plasma: Plasma | None = None
    wavelengths_nm: np.ndarray | None = None
    volume: Volume | None = None
    detector: Detector | None = None
    problem: Callable | None = None
    needs: tuple = ()

    def required_features(self):
        if self.kind == "spectral_density":
            return required_features(self.plasma)
        if self.kind in ("detector_spectrum", "fit", "fit_shot"):
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


def _verdict(passed, measurements, checks, message, note=""):
    """A scoring outcome; ``note`` is kept as the message of a passing result (reported under Notes)."""
    return ("passed" if passed else "failed"), measurements, checks, (note if passed else message)


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


# L4

COUNTS = 1.0e4
START_AMPLITUDE = 0.8 * COUNTS
CALIBRATION_REALIZATIONS = 32
CALIBRATION_SEED = 20261120
FIT_PROBLEMS = (
    ("epw-ne-Te", EPW_CASE, {"ne": 0.16, "Te": 0.78}),
    ("iaw-Va-Ti", IAW_CASE, {"ion1.Va": -0.5, "ion1.Ti": 0.13}),
)
CRAMER_RAO_VARIANTS = ("self", "oracle")


def _counts_detector(case):
    return dataclasses.replace(case.detector, amplitude=COUNTS)


def oracle_model(conventions, det):
    """The oracle's detector spectrum on ``det`` as a function of the (canonical) volume."""
    centers = np.linspace(*det.lam_range, det.pixels)
    options = conventions.detector_options()
    return lambda volume: oracle_detector(det, centers, lambda lam: volume_spectral_density(lam, volume, **options))


def _oracle_data(case):
    det = _counts_detector(case)
    return lambda conventions, outputs: oracle_model(conventions, det)(Volume(case.plasma))


def _problem(case, start, start_amplitude, data_of, variance_of=None):
    """A problem builder: the truth volume of ``case`` with the free plasma parameters at ``start`` and the code's
    amplitudes at ``start_amplitude``."""
    det = _counts_detector(case)
    truth = Volume(case.plasma)

    def build(conventions, outputs):
        data = np.asarray(data_of(conventions, outputs), dtype=float)
        variance = variance_of(conventions) if variance_of else fitting.poisson_variance(data)
        amplitudes = conventions.amplitude_names(det)
        start_volume, _ = fitting.with_values(truth, {}, start)
        mask = fitting.fit_mask(det, np.linspace(*det.lam_range, det.pixels))
        free = tuple(start) + amplitudes
        return FitProblem(start_volume, dict.fromkeys(amplitudes, start_amplitude), det, data, variance, mask, free)

    return build


def _fit_run(case, build, needs=()):
    return Run("fit", volume=Volume(case.plasma), detector=_counts_detector(case), problem=build, needs=needs)


def canonical_estimates(result, problem, conventions):
    """``(estimates, sigmas)`` of a :class:`tsverify.fitting.FitResult` in the code's conventions, as canonical
    values. A missing parameter raises KeyError."""
    expressed = dataclasses.replace(problem.start, plasma=conventions.express(problem.start.plasma))
    values = {name: float(result.estimates[name]) for name in problem.free}
    volume, amplitudes = fitting.with_values(expressed, problem.start_amplitudes, values)
    canonical = dataclasses.replace(volume, plasma=conventions.interpret(volume.plasma))
    estimates = {name: float(fitting.value_of(canonical, amplitudes, name)) for name in problem.free}
    sigmas = None
    if result.sigmas is not None:
        sigmas = {name: float(result.sigmas[name]) for name in problem.free}
    return estimates, sigmas


def _truth(case, name):
    return COUNTS if fitting.is_amplitude(name) else fitting.get(case.plasma, name)


def _at_truth(case, start):
    return {k: _truth(case, k) for k in start}


def recovery_task(name, case, start):
    build = _problem(case, start, START_AMPLITUDE, _oracle_data(case))

    def score(context, out):
        estimates, _ = out["fit"]
        tol = context.tolerances
        measurements, checks, failed = {}, [], []
        for parameter, value in estimates.items():
            truth = _truth(case, parameter)
            error = abs(value - truth)
            allowed = tol.fit_rtol * abs(truth) + tol.fit_atol.get(fitting.base(parameter), 0.0)
            measurements[parameter] = value
            measurements[f"{parameter}: error / allowed"] = error / allowed
            checks.append((f"|{parameter} - truth|", error, allowed))
            if not error <= allowed:
                failed.append(f"{parameter} = {value:.6g} against {truth:.6g} (allowed {allowed:.1e})")
        message = f"L4-recovery/{name}: the fit of noiseless oracle data from a displaced start misses: " + "; ".join(failed)
        return _verdict(not failed, measurements, checks, message)

    return Task(f"L4/recovery/{name}", "test_fit_recovers_the_truth_from_oracle_data", "L4",
                {"fit": _fit_run(case, build)}, score)


_FACTORS = ((np.sqrt(2.0), "sqrt(2)"), (1.0 / np.sqrt(2.0), "1/sqrt(2)"), (2.0, "2"), (0.5, "1/2"))


def _constant_factor(ratios):
    for factor, label in _FACTORS:
        if np.all(np.abs(np.asarray(ratios) / factor - 1.0) < 0.03):
            return (
                f" Every sigma is off by the same factor, {label}: the signature of a loss normalization. The Hessian "
                "of chi^2 is twice that of -log L = chi^2/2, so using one where the other belongs scales sigma by "
                "sqrt(2); a variance taken as a standard deviation, or the reverse, does the same at the bound's scale."
            )
    return ""


def cramer_rao_task(name, case, start, variant):
    """``self``: the code's own noiseless detector spectrum as data (needs L3); ``oracle``: oracle data.

    One-sided: no unbiased estimator has a smaller standard deviation than the bound, so a sigma more than ``tol``
    below it is overconfident and fails. A sigma above the bound passes, with the efficiency (bound/sigma)^2
    reported: it is honest for an estimator that is inefficient by design, and an overestimate for an efficient
    one, which the calibration test tells apart."""
    det = _counts_detector(case)
    at_truth = _at_truth(case, start)
    if variant == "self":
        runs = {"spectrum": Run("detector_spectrum", volume=Volume(case.plasma), detector=det)}
        build = _problem(case, at_truth, COUNTS, lambda conventions, outputs: outputs["spectrum"][1])
        runs["fit"] = _fit_run(case, build, needs=("spectrum",))
    else:
        runs = {"fit": _fit_run(case, _problem(case, at_truth, COUNTS, _oracle_data(case)))}

    def score(context, out):
        _, sigmas = out["fit"]
        if sigmas is None:
            return "not supported", {}, [], "the code reports no uncertainties"
        problem = out["fit problem"]
        model = oracle_model(context.conventions, det)
        truth = dict.fromkeys(problem.start_amplitudes, COUNTS)
        bound = fitting.cramer_rao(model, Volume(case.plasma), truth, det, problem.free, problem.variance, problem.mask)
        tol = context.tolerances.cramer_rao_self if variant == "self" else context.tolerances.cramer_rao_oracle
        ratios = [sigmas[p] / b for p, b in zip(problem.free, bound)]
        measurements = {f"{p}: sigma / cramer-rao": float(r) for p, r in zip(problem.free, ratios)}
        measurements |= {f"{p}: efficiency": float(1.0 / r**2) for p, r in zip(problem.free, ratios)}
        checks = [(f"{p}: 1 - sigma / cramer-rao", 1.0 - r, tol) for p, r in zip(problem.free, ratios)]
        low = [f"{p} {r:.3f}" for p, r in zip(problem.free, ratios) if not r > 1.0 - tol]
        message = (
            f"L4-cramer-rao/{name}/{variant}: the reported sigmas are below the Cramér–Rao bound by more than "
            f"{tol:.0%}, which no unbiased fit can reach, so they are overconfident: sigma / bound = "
            + ", ".join(low) + "." + _constant_factor(ratios)
        )
        high = [f"{p} {r:.3f}" for p, r in zip(problem.free, ratios) if r > 1.0 + tol]
        note = ""
        if high:
            note = (
                "sigma above the bound (efficiency < 1): " + ", ".join(high) + ". Honest if the fit is inefficient "
                "by design; for an efficient fit it is an overestimate, which the calibration test shows as pulls "
                "narrower than 1." + _constant_factor(ratios)
            )
        return _verdict(not low, measurements, checks, message, note)

    def applies(conventions):
        if conventions.sigma_from == "residuals":
            return ("takes its sigma from the fit residuals, which vanish on the noiseless data of this test; the "
                    "calibration test checks it")
        return None

    return Task(f"L4/cramer-rao/{name}/{variant}", "test_fit_uncertainty_matches_the_cramer_rao_bound", "L4", runs, score,
                applies=applies)


def _width_reading(widths, tol):
    if max(widths) > 1.0 + tol:
        return (
            " The estimates scatter more than the reported sigma. If the Cramér–Rao tests pass, the sigma is right for "
            "an efficient estimator and the fit is not one: its objective is not the likelihood the sigma comes from, "
            "for example unweighted least squares on counting noise with sigma from a variance-weighted Hessian."
        )
    if min(widths) < 1.0 - tol:
        return (
            " The estimates scatter less than the reported sigma: the sigma is overestimated, or the fit stops before "
            "it converges and stays near its start."
        )
    return ""


def _calibration_data(case, index, k):
    """Realization ``k`` of counting noise on the oracle data: Gaussian with the Poisson variance, seeded."""

    def data(conventions, outputs):
        expected = _oracle_data(case)(conventions, outputs)
        rng = np.random.default_rng((CALIBRATION_SEED, index, k))
        return expected + rng.standard_normal(expected.shape) * np.sqrt(fitting.poisson_variance(expected))

    return data


def calibration_task(name, case, start, index):
    at_truth = _at_truth(case, start)

    def expected_variance(conventions):
        return fitting.poisson_variance(_oracle_data(case)(conventions, {}))

    n = CALIBRATION_REALIZATIONS
    runs = {"reference": _fit_run(case, _problem(case, at_truth, COUNTS, _oracle_data(case)))}
    for k in range(n):
        data = _calibration_data(case, index, k)
        runs[f"noisy{k:02d}"] = _fit_run(case, _problem(case, at_truth, COUNTS, data, expected_variance))
    mean_tol, width_tol = 3.0 / np.sqrt(n), 3.0 / np.sqrt(2.0 * (n - 1))

    def score(context, out):
        reference, _ = out["reference"]
        noisy = [out[f"noisy{k:02d}"] for k in range(n)]
        if any(sigmas is None for _, sigmas in noisy):
            return "not supported", {}, [], "the code reports no uncertainties"
        measurements, checks, off = {}, [], []
        for parameter in reference:
            pulls = np.array([(est[parameter] - reference[parameter]) / sig[parameter] for est, sig in noisy])
            mean, width = float(pulls.mean()), float(pulls.std(ddof=1))
            within1, within2 = float(np.mean(np.abs(pulls) < 1.0)), float(np.mean(np.abs(pulls) < 2.0))
            measurements |= {f"{parameter}: pull mean": mean, f"{parameter}: pull width": width,
                             f"{parameter}: 1-sigma coverage": within1, f"{parameter}: 2-sigma coverage": within2}
            checks += [(f"|{parameter} pull mean|", abs(mean), mean_tol),
                       (f"|{parameter} pull width - 1|", abs(width - 1.0), width_tol)]
            if not (abs(mean) < mean_tol and abs(width - 1.0) < width_tol):
                off.append(f"{parameter}: mean {mean:+.2f}, width {width:.2f} (1-sigma coverage {within1:.0%})")
        widths = [measurements[f"{p}: pull width"] for p in reference]
        message = (
            f"L4-calibration/{name}: over {n} noisy realizations, (estimate - noiseless estimate) / sigma should have "
            f"mean 0 (within {mean_tol:.2f}) and width 1 (within {width_tol:.2f}): " + "; ".join(off) + "."
            + (_constant_factor(widths) or _width_reading(widths, width_tol))
        )
        return _verdict(not off, measurements, checks, message)

    return Task(f"L4/calibration/{name}", "test_fit_uncertainty_is_calibrated", "L4", runs, score, slow=True)


# L4 with camera noise

NOMINAL_CAMERA = Camera(gain=2.0, excess=1.15, read_noise=17.0, rows=3, psf_pixels=1.0)
SHOTS = 12
LINEOUTS = 4
CAMERA_SEED = 20261121
READ_NOISE_TOL = 0.2
GAIN_TOL = 0.1
REFERENCE_SHOT = 10_000


@dataclass(frozen=True)
class CameraRegime:
    """``read_noise``: the true read noise at the centre of its shot-to-shot drift (x0.8 to x1.25);
    ``nominal_read_noise``: what the code is told; ``background``: flat background level in counts (drifting x0.7 to
    x1.4, with a 30 % ramp across the detector); ``gain_drift``: the range of the true gain times excess over the
    nominal one, drawn per shot; ``irf_scale``: per channel, the data's instrument function over the one the code
    is given, a misfit that grows with the level at the peaks (the values below give about 3.6x the noise at the
    peak on both channels, the 2-5x seen on real ARTS data); ``check_read_noise`` / ``check_gain``: score how well
    the code sized them here; ``check_pulls``: score the pulls. Under misfit they are a recorded diagnostic: a code
    that sizes its noise from the residuals weights the noiseless reference fit (pure misfit residuals) differently
    from the noisy ones, so their estimates differ systematically, and a sigma from the noise alone understates the
    scatter the misfit adds through data-dependent weights (the oracle reference: IAW Va width 1.35). There the test
    is whether the noise is still sized right."""

    name: str
    read_noise: float
    nominal_read_noise: float
    background: float
    check_read_noise: bool
    gain_drift: tuple = (1.0, 1.0)
    irf_scale: tuple = ()
    check_gain: bool = False
    check_pulls: bool = True


CAMERA_REGIMES = (
    CameraRegime("shot-noise", 2.0, 2.0, 0.0, False),
    CameraRegime("background", 10.0, 10.0, 2000.0, False),
    CameraRegime("read-noise", 65.0, 65.0, 50.0, True),
    CameraRegime("miscalibrated", 65.0, 17.0, 200.0, True),
    CameraRegime("gain-drift", 20.0, 20.0, 200.0, True, gain_drift=(1.0, 1.3)),
    CameraRegime("misfit", 20.0, 20.0, 200.0, True, irf_scale=(("EPW", 1.3), ("IAW", 1.05)), check_pulls=False),
)


def _shot_noise(index, regime_index, shot):
    """``(rng, true read noise, background level, true gain scale)`` of one shot."""
    regime = CAMERA_REGIMES[regime_index]
    rng = np.random.default_rng((CAMERA_SEED, index, regime_index, shot))
    read = regime.read_noise * rng.uniform(0.8, 1.25)
    level = regime.background * rng.uniform(0.7, 1.4)
    return rng, read, level, rng.uniform(*regime.gain_drift)


def _shot(case, start, index, regime_index, shot, conventions, noiseless=False):
    """A :class:`ShotProblem` of ``LINEOUTS`` lineouts of the truth plasma, at the truth start."""
    regime = CAMERA_REGIMES[regime_index]
    det = _counts_detector(case)
    centers = np.linspace(*det.lam_range, det.pixels)
    data_det = dataclasses.replace(det, sigma=det.sigma * dict(regime.irf_scale).get(det.channel, 1.0))
    expected = oracle_model(conventions, data_det)(Volume(case.plasma))
    rng, read, level, gain_scale = _shot_noise(index, regime_index, shot)
    if noiseless:
        level = regime.background
    ramp = (centers - centers.mean()) / (centers.max() - centers.mean())
    background = level * (1.0 + 0.3 * ramp)
    camera = NOMINAL_CAMERA.with_read_noise(read).with_gain_scale(gain_scale)
    amplitudes = conventions.amplitude_names(det)
    start_volume, _ = fitting.with_values(Volume(case.plasma), {}, start)
    mask = fitting.fit_mask(det, centers)
    lineouts = []
    for _ in range(LINEOUTS):
        data = expected + background if noiseless else camera.sample(expected + background, rng)
        lineouts.append(FitProblem(start_volume, dict.fromkeys(amplitudes, COUNTS), det, data, None, mask,
                                   tuple(start) + amplitudes))
    nominal = NOMINAL_CAMERA.with_read_noise(regime.nominal_read_noise)
    return ShotProblem(tuple(lineouts), (background,) * LINEOUTS, nominal)


def camera_task(name, case, start, index, regime_index):
    """Calibration with camera noise the code must size: ``SHOTS`` shots of ``LINEOUTS`` lineouts, the read noise
    and background drifting from shot to shot, the pulls (estimate - noiseless estimate)/sigma pooled over all
    lineouts, and, if the code reports it, the read noise it sized against the truth."""
    regime = CAMERA_REGIMES[regime_index]
    at_truth = _at_truth(case, start)
    det = _counts_detector(case)

    def shot_run(shot, noiseless=False):
        return Run("fit_shot", volume=Volume(case.plasma), detector=det,
                   problem=lambda conventions, outputs: _shot(case, at_truth, index, regime_index, shot, conventions,
                                                              noiseless))

    runs = {"reference": shot_run(REFERENCE_SHOT, noiseless=True)}
    for s in range(SHOTS):
        runs[f"shot{s:02d}"] = shot_run(s)
    n = SHOTS * LINEOUTS
    mean_tol, width_tol = 3.0 / np.sqrt(n), 3.0 / np.sqrt(2.0 * (n - 1))

    def score(context, out):
        reference = out["reference"]["fits"][0][0]
        shots = [out[f"shot{s:02d}"] for s in range(SHOTS)]
        if any(sig is None for shot in shots for _, sig in shot["fits"]):
            return "not supported", {}, [], "the code reports no uncertainties"
        measurements, checks, off, pulls_seen = {}, [], [], []
        for parameter in reference:
            pulls = np.array([(est[parameter] - reference[parameter]) / sig[parameter]
                              for shot in shots for est, sig in shot["fits"]])
            mean, width = float(pulls.mean()), float(pulls.std(ddof=1))
            measurements |= {f"{parameter}: pull mean": mean, f"{parameter}: pull width": width,
                             f"{parameter}: 1-sigma coverage": float(np.mean(np.abs(pulls) < 1.0))}
            if not regime.check_pulls:
                pulls_seen.append(f"{parameter} mean {mean:+.2f}, width {width:.2f}")
                continue
            checks += [(f"|{parameter} pull mean|", abs(mean), mean_tol),
                       (f"|{parameter} pull width - 1|", abs(width - 1.0), width_tol)]
            if not (abs(mean) < mean_tol and abs(width - 1.0) < width_tol):
                off.append(f"{parameter}: mean {mean:+.2f}, width {width:.2f}")
        truth = [_shot_noise(index, regime_index, s) for s in range(SHOTS)]
        true = {"read_noise": np.array([t[1] for t in truth]), "gain_scale": np.array([t[3] for t in truth])}
        true["read_noise_by_level_regression"] = true["read_noise"]
        labels = {"read_noise": "read noise", "gain_scale": "gain scale",
                  "read_noise_by_level_regression": "read noise by level regression"}
        limits = {"read_noise": (regime.check_read_noise, READ_NOISE_TOL), "gain_scale": (regime.check_gain, GAIN_TOL)}
        for key, label in labels.items():
            sized = [shot["noise"].get(key) if shot["noise"] else None for shot in shots]
            if not all(r is not None for r in sized):
                continue
            ratio = np.asarray(sized, dtype=float) / true[key]
            median = float(np.median(ratio))
            measurements |= {f"{label} sized / true: median": median,
                             f"{label} sized / true: min": float(ratio.min()),
                             f"{label} sized / true: max": float(ratio.max())}
            check, tol = limits.get(key, (False, None))
            if check:
                checks.append((f"|{label} sized / true - 1| (median)", abs(median - 1.0), tol))
                if not abs(median - 1.0) < tol:
                    off.append(f"{label} sized at {median:.2f} of the truth (median over shots)")
        message = (
            f"L4-camera/{name}/{regime.name}: over {SHOTS} shots of {LINEOUTS} lineouts with camera noise the code "
            f"sizes itself (nominal read noise {regime.nominal_read_noise:g}, true {regime.read_noise:g} x0.8-1.25; "
            f"gain x{regime.gain_drift[0]:g}-{regime.gain_drift[1]:g}; instrument function "
            f"x{dict(regime.irf_scale).get(det.channel, 1.0):g}), "
            + (f"the pulls should have mean 0 (within {mean_tol:.2f}) and width 1 (within {width_tol:.2f})"
               if regime.check_pulls else "the noise should be sized right")
            + ": " + "; ".join(off) + "."
        )
        note = ""
        if pulls_seen:
            note = ("pulls under misfit (diagnostic, not scored): " + "; ".join(pulls_seen)
                    + ". A sigma from the noise alone does not cover the scatter misfit adds.")
        return _verdict(not off, measurements, checks, message, note)

    return Task(f"L4/camera/{name}/{regime.name}", "test_fit_uncertainty_with_camera_noise", "L4", runs, score,
                slow=True)


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
    tasks += [recovery_task(*p) for p in FIT_PROBLEMS]
    tasks += [cramer_rao_task(*p, variant) for p in FIT_PROBLEMS for variant in CRAMER_RAO_VARIANTS]
    if slow:
        tasks += [calibration_task(*p, index) for index, p in enumerate(FIT_PROBLEMS)]
        tasks += [camera_task(*p, index, r) for index, p in enumerate(FIT_PROBLEMS) for r in range(len(CAMERA_REGIMES))]
    return tasks
