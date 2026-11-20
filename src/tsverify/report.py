"""Conformance reports: Markdown for reading, JSON for scripts. Both routes (pytest and file exchange) write them."""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections import Counter, defaultdict

from . import __version__

LEVELS = (
    ("L1", "L1: Z' against the Faddeeva function"),
    ("L2", "L2: spectral density"),
    ("probe", "Convention probes"),
    ("A12", "A12: first-order relativistic spectrum against Selden's exact one"),
    ("L3", "L3: detector spectrum"),
    ("A10", "A10: gradient, aperture and rewritten plasmas"),
)
STATUS = {"passed": "pass", "failed": "**FAIL**", "not supported": "not supported"}

_collected = []


def collect(adapter, result):
    """Keep a result from the pytest route for the report the plugin writes at the end of the session."""
    _collected.append((adapter, result))


def collected():
    return list(_collected)


@dataclasses.dataclass
class Meta:
    code: str
    route: str
    conventions: object
    tolerances: str = "REFERENCE"
    version: str = __version__
    date: str = dataclasses.field(default_factory=lambda: datetime.datetime.now().astimezone().isoformat(timespec="minutes"))


def _number(x):
    return f"{x:.2e}" if isinstance(x, float) else str(x)


def _cell(text):
    return str(text).replace("|", r"\|")


def _checks(result):
    if not result.checks:
        return "", ""
    measured = "<br>".join(f"{_cell(what)} {_number(float(value))}" for what, value, _ in result.checks)
    limit = "<br>".join(f"< {_number(float(tol))}" for _, _, tol in result.checks)
    return measured, limit


def markdown(results, meta):
    """The report as Markdown."""
    counts = defaultdict(Counter)
    for r in results:
        counts[r.level][r.status] += 1
    total = Counter(r.status for r in results)
    lines = [
        f"# tsverify report: {meta.code}",
        "",
        f"tsverify {meta.version} · {meta.date} · {meta.route} · tolerances {meta.tolerances}",
        "",
        f"**{total['passed']} passed, {total['failed']} failed, {total['not supported']} not supported.**",
        "",
        "## Declared conventions",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    lines += [f"| `{f.name}` | `{getattr(meta.conventions, f.name)!r}` |" for f in dataclasses.fields(meta.conventions)]
    lines += ["", "## Summary", "", "| Level | Passed | Failed | Not supported |", "|---|---|---|---|"]
    for level, title in LEVELS:
        c = counts.get(level)
        if c:
            lines.append(f"| {title} | {c['passed']} | {c['failed']} | {c['not supported']} |")

    failed = [r for r in results if r.status == "failed"]
    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            lines += [f"### `{r.task}`", "", r.message, ""]

    for level, title in LEVELS:
        shown = [r for r in results if r.level == level and r.status != "not supported"]
        if not shown:
            continue
        lines += ["", f"## {title}", "", "| Test | Measured | Limit | Result |", "|---|---|---|---|"]
        for r in shown:
            measured, limit = _checks(r)
            lines.append(f"| `{r.task}` | {measured} | {limit} | {STATUS[r.status]} |")

    unsupported = defaultdict(list)
    for r in results:
        if r.status == "not supported":
            unsupported[r.message].append(r.task)
    if unsupported:
        lines += ["", "## Not supported", "", "| Reason | Tests |", "|---|---|"]
        for reason, tasks in unsupported.items():
            lines.append(f"| {_cell(reason)} | {len(tasks)}: " + ", ".join(f"`{t}`" for t in tasks) + " |")
    return "\n".join(lines) + "\n"


def as_json(results, meta):
    """The report as a JSON-ready dict."""
    return {
        "tsverify": meta.version,
        "code": meta.code,
        "route": meta.route,
        "date": meta.date,
        "tolerances": meta.tolerances,
        "conventions": dataclasses.asdict(meta.conventions),
        "results": [
            {
                "task": r.task,
                "test": r.test,
                "level": r.level,
                "status": r.status,
                "measurements": {k: float(v) for k, v in r.measurements.items()},
                "checks": [{"what": w, "measured": float(v), "limit": float(t)} for w, v, t in r.checks],
                "message": r.message,
            }
            for r in results
        ],
    }


def write(results, meta, markdown_path=None, json_path=None):
    if markdown_path:
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown(results, meta))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(as_json(results, meta), f, indent=1)
