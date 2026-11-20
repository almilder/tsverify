"""Running tasks through a Python adapter."""

from __future__ import annotations

import dataclasses

import numpy as np

from .adapter import NotSupported
from .tasks import Context, canonical_estimates


def context(adapter):
    return Context(adapter.conventions, adapter.tolerances, adapter.supports("zprime"))


def express_problem(problem, conventions):
    """A fit problem with its start plasma written in the code's conventions."""
    return dataclasses.replace(problem, start=dataclasses.replace(problem.start, plasma=conventions.express(problem.start.plasma)))


def evaluate(adapter, run, outputs=None):
    """The code's output for ``run``, with the plasma written in the adapter's conventions: an array,
    ``(pixel_centers_nm, values)`` for a detector spectrum, or ``(problem, fit result)`` for a fit, ``problem`` being
    canonical and built from the earlier ``outputs`` the run needs. Raises :class:`NotSupported` with the reason."""
    if not adapter.supports(run.kind):
        raise NotSupported(f"{run.kind} not supported")
    missing = run.required_features() - set(adapter.features)
    if missing:
        raise NotSupported(f"not supported: {', '.join(sorted(missing))}")
    conventions = adapter.conventions
    try:
        if run.kind == "zprime":
            return np.asarray(adapter.zprime(run.zeta))
        if run.kind == "spectral_density":
            return np.asarray(adapter.spectral_density(conventions.express(run.plasma), run.wavelengths_nm))
        if run.kind == "fit":
            problem = run.problem(conventions, {n: (outputs or {})[n] for n in run.needs})
            return problem, adapter.fit(express_problem(problem, conventions))
        expressed = dataclasses.replace(run.volume, plasma=conventions.express(run.volume.plasma))
        centers, values = adapter.detector_spectrum(expressed, run.detector)
        return np.asarray(centers), np.asarray(values)
    except NotSupported as reason:
        raise NotSupported(f"not supported: {reason}") from None


def fit_outputs(name, problem, result, conventions):
    """The scoring inputs of a fit run: its canonical ``(estimates, sigmas)`` and, under ``<name> problem``, the
    problem. A result without an estimate for every free parameter raises KeyError."""
    return {name: canonical_estimates(result, problem, conventions), f"{name} problem": problem}


def score(adapter, task):
    """The :class:`tsverify.tasks.Result` of ``task`` for the code behind ``adapter``."""
    reason = task.not_applicable(adapter.conventions)
    if reason:
        return task.unsupported(f"{adapter.name}: {reason}")
    outputs = {}
    for name, run in task.runs.items():
        try:
            value = evaluate(adapter, run, outputs)
        except NotSupported as why:
            return task.unsupported(f"{adapter.name}: {why}")
        if run.kind == "fit":
            try:
                outputs |= fit_outputs(name, *value, adapter.conventions)
            except KeyError as key:
                return dataclasses.replace(task.unsupported(""), status="failed",
                                           message=f"{task.id}: the fit returned no estimate for {key}")
        else:
            outputs[name] = value
    return task.result(context(adapter), outputs)
