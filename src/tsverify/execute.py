"""Running tasks through a Python adapter."""

from __future__ import annotations

import dataclasses

import numpy as np

from .adapter import NotSupported
from .tasks import Context


def context(adapter):
    return Context(adapter.conventions, adapter.tolerances, adapter.supports("zprime"))


def evaluate(adapter, run):
    """The code's output for ``run``, with the plasma written in the adapter's conventions: an array, or
    ``(pixel_centers_nm, values)`` for a detector spectrum. Raises :class:`NotSupported` with the reason."""
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
        expressed = dataclasses.replace(run.volume, plasma=conventions.express(run.volume.plasma))
        centers, values = adapter.detector_spectrum(expressed, run.detector)
        return np.asarray(centers), np.asarray(values)
    except NotSupported as reason:
        raise NotSupported(f"not supported: {reason}") from None


def score(adapter, task):
    """The :class:`tsverify.tasks.Result` of ``task`` for the code behind ``adapter``."""
    reason = task.not_applicable(adapter.conventions)
    if reason:
        return task.unsupported(f"{adapter.name}: {reason}")
    outputs = {}
    for name, run in task.runs.items():
        try:
            outputs[name] = evaluate(adapter, run)
        except NotSupported as why:
            return task.unsupported(f"{adapter.name}: {why}")
    return task.result(context(adapter), outputs)
