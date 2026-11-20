"""File exchange: run the conformance suite against a code that tsverify cannot call.

:func:`export` writes a *bundle*: every run's inputs, written in the code's declared conventions, as CSV and/or JSON,
plus the oracle's answer for each run. The code reads each run's inputs and writes ``output.csv`` (or
``output.json``) into the run's folder. :func:`score` reads those outputs and scores them through the same tasks
as the pytest route (:mod:`tsverify.tasks`), so both routes give the same results. A run without an output counts
as not supported. :func:`respond` fills a bundle from a Python adapter, to check the round trip.

Values are written with ``repr``, so every float64 survives the round trip exactly.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from . import __version__, oracle
from .adapter import NotSupported
from .conventions import Conventions
from .instrument import Detector
from .oracle import GaussianComponent, Ion, KappaEDF, Plasma, QuadratureEDF
from .tasks import Context, Task, all_tasks
from .tolerances import REFERENCE
from .volume import Aperture, LinearGradient, Volume

FORMATS = ("csv", "json")
EDF_V = np.linspace(-25.0, 25.0, 5001)
OUTPUT_COLUMNS = {"zprime": ("re", "im"), "spectral_density": ("S",), "detector_spectrum": ("pixel_center_nm", "value")}
MAXWELLIAN = (GaussianComponent(),)


class BundleError(Exception):
    pass


def _slug(text):
    return re.sub(r"[^A-Za-z0-9.@-]+", "_", text).strip("_")


def run_ids(task: Task):
    """``{run name: run id}``; the run id is also the run's folder under ``runs/``."""
    if len(task.runs) == 1:
        return {name: _slug(task.id) for name in task.runs}
    return {name: _slug(f"{task.id}/{name}") for name in task.runs}


# Writing


def _fmt(x):
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return repr(float(x))


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(_fmt(x) for x in row) + "\n")


def _edf(components):
    if isinstance(components, KappaEDF):
        return "kappa", components.kappa
    if isinstance(components, QuadratureEDF):
        raise BundleError(f"no file form for the EDF {type(components).__name__}")
    return ("maxwellian" if tuple(components) == MAXWELLIAN else "mixture"), float("nan")


def plasma_record(plasma: Plasma):
    """A JSON-ready description of ``plasma``."""

    def edf(components):
        kind, kappa = _edf(components)
        out = {"type": kind}
        if kind == "kappa":
            out["kappa"] = kappa
        else:
            out["components"] = [dataclasses.asdict(c) for c in components]
        return out

    return {
        "ne": plasma.ne,
        "Te": plasma.Te,
        "angle_deg": plasma.angle_deg,
        "lam": plasma.lam,
        "ud": plasma.ud,
        "electrons": edf(plasma.electrons),
        "ions": [
            {"A": i.A, "Z": i.Z, "Ti": i.Ti, "fract": i.fract, "Va": i.Va, "edf": edf(i.components)} for i in plasma.ions
        ],
    }


def _write_plasma_csv(folder, plasma):
    kind, kappa = _edf(plasma.electrons)
    _write_csv(folder / "plasma.csv", ("ne", "Te", "angle_deg", "lam", "ud", "electron_edf", "electron_kappa"),
               [(plasma.ne, plasma.Te, plasma.angle_deg, plasma.lam, plasma.ud, kind, kappa)])
    rows = []
    for ion in plasma.ions:
        ion_kind, ion_kappa = _edf(ion.components)
        rows.append((ion.A, ion.Z, ion.Ti, ion.fract, ion.Va, ion_kind, ion_kappa))
    _write_csv(folder / "ions.csv", ("A", "Z", "Ti", "fract", "Va", "edf", "kappa"), rows)
    if kind == "mixture":
        _write_csv(folder / "electron_components.csv", ("weight", "shift", "width"),
                   [(c.weight, c.shift, c.width) for c in plasma.electrons])
    if kind != "maxwellian":
        _write_csv(folder / "electron_f.csv", ("v", "f"), zip(EDF_V, oracle.mixture_density(EDF_V, plasma.electrons)))
    mixtures = [(n, c) for n, ion in enumerate(plasma.ions, 1) if _edf(ion.components)[0] == "mixture" for c in ion.components]
    if mixtures:
        _write_csv(folder / "ion_components.csv", ("ion", "weight", "shift", "width"),
                   [(n, c.weight, c.shift, c.width) for n, c in mixtures])
    for n, ion in enumerate(plasma.ions, 1):
        if _edf(ion.components)[0] != "maxwellian":
            _write_csv(folder / f"ion{n}_f.csv", ("v", "f"), zip(EDF_V, oracle.mixture_density(EDF_V, ion.components)))


def _detector_record(det: Detector):
    return {
        "channel": det.channel,
        "lam_range": list(det.lam_range),
        "pixels": det.pixels,
        "points_per_pixel": det.points_per_pixel,
        "sigma": det.sigma,
        "amplitude": det.amplitude,
        "amp_ranges": [list(r) for r in det.amp_ranges],
        "notch": list(det.notch) if det.notch else None,
    }


def _write_detector_csv(folder, det: Detector):
    notch = det.notch or (float("nan"),) * 3
    _write_csv(folder / "detector.csv",
               ("channel", "lam_min", "lam_max", "pixels", "points_per_pixel", "sigma", "amplitude",
                "notch_center", "notch_width", "notch_od"),
               [(det.channel, *det.lam_range, det.pixels, det.points_per_pixel, det.sigma, det.amplitude, *notch)])
    _write_csv(folder / "amp_ranges.csv", ("low", "high"), det.amp_ranges)


def _input_record(run, plasma):
    record = {"kind": run.kind}
    if run.kind == "zprime":
        record["zeta"] = run.zeta.tolist()
    elif run.kind == "spectral_density":
        record["plasma"] = plasma_record(plasma)
        record["wavelengths_nm"] = run.wavelengths_nm.tolist()
    else:
        v = run.volume
        record["volume"] = {
            "plasma": plasma_record(plasma),
            "gradient": dataclasses.asdict(v.gradient) if v.gradient else None,
            "aperture": {"angles_deg": list(v.aperture.angles_deg), "weights": list(v.aperture.weights)} if v.aperture else None,
        }
        record["detector"] = _detector_record(run.detector)
    return record


def _expected(context, task, name, run):
    """The oracle's answer for ``run``, reported as the code reports it."""
    from .tasks import _expected_detector

    if run.kind == "zprime":
        z = oracle.plasma_dispersion_derivative(run.zeta)
        return ("zeta", "re", "im"), (run.zeta, z.real, z.imag)
    if run.kind == "spectral_density":
        options = context.conventions.oracle_options()
        return ("wavelength_nm", "S"), (run.wavelengths_nm, oracle.spectral_density(run.wavelengths_nm, run.plasma, **options))
    centers = np.linspace(*run.detector.lam_range, run.detector.pixels)
    return ("pixel_center_nm", "value"), (centers, _expected_detector(context, run.volume, run.detector, centers))


def export(bundle, conventions: Conventions = Conventions(), *, slow=False, formats=FORMATS, force=False):
    """Write the bundle for a code with ``conventions`` into the directory ``bundle``."""
    bundle = Path(bundle)
    formats = tuple(formats)
    if any(f not in FORMATS for f in formats):
        raise BundleError(f"formats must be among {FORMATS}, got {formats}")
    if (bundle / "manifest.json").exists() and not force:
        raise BundleError(f"{bundle} already holds a bundle; pass force=True (--force) to overwrite it")
    context = Context(conventions, REFERENCE)
    manifest = {"tsverify": __version__, "conventions": dataclasses.asdict(conventions), "slow": slow,
                "formats": list(formats), "tasks": [], "runs": []}
    for task in all_tasks(slow=slow):
        reason = task.not_applicable(conventions)
        ids = run_ids(task)
        manifest["tasks"].append({"task": task.id, "test": task.test, "level": task.level,
                                  "runs": [] if reason else list(ids.values()), "not_applicable": reason})
        if reason:
            continue
        for name, run in task.runs.items():
            folder = bundle / "runs" / ids[name]
            folder.mkdir(parents=True, exist_ok=True)
            plasma = run.plasma if run.kind == "spectral_density" else (run.volume.plasma if run.volume else None)
            expressed = conventions.express(plasma) if plasma is not None else None
            if "json" in formats:
                (folder / "input.json").write_text(json.dumps(_input_record(run, expressed)), encoding="utf-8")
            if "csv" in formats:
                if run.kind == "zprime":
                    _write_csv(folder / "zeta.csv", ("zeta",), ((z,) for z in run.zeta))
                elif run.kind == "spectral_density":
                    _write_plasma_csv(folder, expressed)
                    _write_csv(folder / "wavelengths.csv", ("wavelength_nm",), ((x,) for x in run.wavelengths_nm))
                else:
                    _write_plasma_csv(folder, expressed)
                    _write_detector_csv(folder, run.detector)
                    if run.volume.gradient:
                        g = run.volume.gradient
                        _write_csv(folder / "gradient.csv", ("ne_percent", "Te_percent", "points"),
                                   [(g.ne_percent, g.Te_percent, g.points)])
                    if run.volume.aperture:
                        a = run.volume.aperture
                        _write_csv(folder / "aperture.csv", ("angle_deg", "weight"), zip(a.angles_deg, a.weights))
            header, columns = _expected(context, task, name, run)
            if "csv" in formats:
                _write_csv(folder / "expected.csv", header, zip(*columns))
            if "json" in formats:
                (folder / "expected.json").write_text(
                    json.dumps({h: np.asarray(c).tolist() for h, c in zip(header, columns)}), encoding="utf-8"
                )
            manifest["runs"].append({"run": ids[name], "kind": run.kind, "task": task.id,
                                     "folder": f"runs/{ids[name]}", "features": sorted(run.required_features())})
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    _write_csv(bundle / "runs.csv", ("run", "kind", "folder", "task", "features"),
               [(r["run"], r["kind"], r["folder"], r["task"], " ".join(r["features"])) for r in manifest["runs"]])
    (bundle / "README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


# Reading


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    return header, [[x.strip() for x in row] for row in rows[1:] if row]


def _columns(path, numeric=True):
    header, rows = _read_csv(path)
    cols = list(zip(*rows)) if rows else [()] * len(header)
    return {h: (np.array([float(x) for x in c]) if numeric else list(c)) for h, c in zip(header, cols)}


def _records(path):
    header, rows = _read_csv(path)
    return [dict(zip(header, row)) for row in rows]


def _number(text):
    return float(text)


def _edf_from(kind, kappa, components):
    if kind == "maxwellian":
        return MAXWELLIAN
    if kind == "kappa":
        return KappaEDF(kappa)
    return tuple(GaussianComponent(*c) for c in components)


def _plasma_from_csv(folder):
    p = _records(folder / "plasma.csv")[0]
    electron_components = []
    if p["electron_edf"] == "mixture":
        c = _columns(folder / "electron_components.csv")
        electron_components = list(zip(c["weight"], c["shift"], c["width"]))
    ion_components = {}
    if (folder / "ion_components.csv").exists():
        c = _columns(folder / "ion_components.csv")
        for n, w, s, d in zip(c["ion"], c["weight"], c["shift"], c["width"]):
            ion_components.setdefault(int(n), []).append((w, s, d))
    ions = tuple(
        Ion(A=_number(r["A"]), Z=_number(r["Z"]), Ti=_number(r["Ti"]), fract=_number(r["fract"]), Va=_number(r["Va"]),
            components=_edf_from(r["edf"], _number(r["kappa"]), ion_components.get(n, [])))
        for n, r in enumerate(_records(folder / "ions.csv"), 1)
    )
    return Plasma(ne=_number(p["ne"]), Te=_number(p["Te"]), ions=ions, angle_deg=_number(p["angle_deg"]),
                  lam=_number(p["lam"]), ud=_number(p["ud"]),
                  electrons=_edf_from(p["electron_edf"], _number(p["electron_kappa"]), electron_components))


def _plasma_from_record(r):
    def edf(e):
        return _edf_from(e["type"], e.get("kappa"), [(c["weight"], c["shift"], c["width"]) for c in e.get("components", [])])

    ions = tuple(Ion(A=i["A"], Z=i["Z"], Ti=i["Ti"], fract=i["fract"], Va=i["Va"], components=edf(i["edf"])) for i in r["ions"])
    return Plasma(ne=r["ne"], Te=r["Te"], ions=ions, angle_deg=r["angle_deg"], lam=r["lam"], ud=r["ud"],
                  electrons=edf(r["electrons"]))


def _detector_from(record, amp_ranges):
    notch = record.get("notch")
    return Detector(channel=record["channel"], lam_range=tuple(record["lam_range"]), pixels=int(record["pixels"]),
                    points_per_pixel=int(record["points_per_pixel"]), sigma=record["sigma"],
                    amp_ranges=tuple(tuple(r) for r in amp_ranges), notch=tuple(notch) if notch else None,
                    amplitude=record["amplitude"])


def read_inputs(folder, kind, fmt="csv"):
    """A run's inputs as the code would read them: ``{"zeta"}``, ``{"plasma", "wavelengths_nm"}`` or
    ``{"volume", "detector"}`` (the plasma written in the code's conventions)."""
    folder = Path(folder)
    if fmt == "json":
        r = json.loads((folder / "input.json").read_text(encoding="utf-8"))
        if kind == "zprime":
            return {"zeta": np.array(r["zeta"])}
        if kind == "spectral_density":
            return {"plasma": _plasma_from_record(r["plasma"]), "wavelengths_nm": np.array(r["wavelengths_nm"])}
        v = r["volume"]
        volume = Volume(_plasma_from_record(v["plasma"]),
                        gradient=LinearGradient(**v["gradient"]) if v["gradient"] else None,
                        aperture=Aperture(tuple(v["aperture"]["angles_deg"]), tuple(v["aperture"]["weights"])) if v["aperture"] else None)
        return {"volume": volume, "detector": _detector_from(r["detector"], r["detector"]["amp_ranges"])}
    if kind == "zprime":
        return {"zeta": _columns(folder / "zeta.csv")["zeta"]}
    plasma = _plasma_from_csv(folder)
    if kind == "spectral_density":
        return {"plasma": plasma, "wavelengths_nm": _columns(folder / "wavelengths.csv")["wavelength_nm"]}
    d = _records(folder / "detector.csv")[0]
    notch = tuple(_number(d[k]) for k in ("notch_center", "notch_width", "notch_od"))
    record = {"channel": d["channel"], "lam_range": (_number(d["lam_min"]), _number(d["lam_max"])),
              "pixels": int(d["pixels"]), "points_per_pixel": int(d["points_per_pixel"]), "sigma": _number(d["sigma"]),
              "amplitude": _number(d["amplitude"]), "notch": None if np.isnan(notch[0]) else notch}
    a = _columns(folder / "amp_ranges.csv")
    gradient = aperture = None
    if (folder / "gradient.csv").exists():
        g = _records(folder / "gradient.csv")[0]
        gradient = LinearGradient(_number(g["ne_percent"]), _number(g["Te_percent"]), int(g["points"]))
    if (folder / "aperture.csv").exists():
        ap = _columns(folder / "aperture.csv")
        aperture = Aperture(tuple(ap["angle_deg"].tolist()), tuple(ap["weight"].tolist()))
    return {"volume": Volume(plasma, gradient, aperture), "detector": _detector_from(record, zip(a["low"], a["high"]))}


def read_output(folder, kind):
    """The code's output for a run, or None if it wrote none."""
    folder = Path(folder)
    csv_path, json_path = folder / "output.csv", folder / "output.json"
    if csv_path.exists() and json_path.exists():
        raise BundleError(f"{folder} has both output.csv and output.json; keep one")
    if csv_path.exists():
        data = _columns(csv_path)
    elif json_path.exists():
        data = {k: np.asarray(v, dtype=float) for k, v in json.loads(json_path.read_text(encoding="utf-8")).items()}
    else:
        return None
    missing = [c for c in OUTPUT_COLUMNS[kind] if c not in data]
    if missing:
        raise BundleError(f"{folder}: the output needs the columns {OUTPUT_COLUMNS[kind]}, missing {missing}")
    if kind == "zprime":
        return data["re"] + 1j * data["im"]
    if kind == "spectral_density":
        return data["S"]
    return data["pixel_center_nm"], data["value"]


def write_output(folder, kind, value, fmt="csv"):
    folder = Path(folder)
    if kind == "zprime":
        columns = {"re": np.real(value), "im": np.imag(value)}
    elif kind == "spectral_density":
        columns = {"S": np.asarray(value)}
    else:
        columns = {"pixel_center_nm": np.asarray(value[0]), "value": np.asarray(value[1])}
    if fmt == "json":
        (folder / "output.json").write_text(json.dumps({k: v.tolist() for k, v in columns.items()}), encoding="utf-8")
    else:
        _write_csv(folder / "output.csv", tuple(columns), zip(*columns.values()))


def _manifest(bundle):
    path = Path(bundle) / "manifest.json"
    if not path.exists():
        raise BundleError(f"{bundle} is not a tsverify bundle (no manifest.json); create one with `tsverify export`")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["tsverify"] != __version__:
        raise BundleError(
            f"{bundle} was exported by tsverify {manifest['tsverify']}, but this is {__version__}. Cases and tolerances "
            "can change between versions, so score with the version that exported it, or export a new bundle."
        )
    return manifest


def _tasks(manifest):
    tasks = {t.id: t for t in all_tasks(slow=manifest["slow"])}
    return [(tasks[entry["task"]], entry) for entry in manifest["tasks"]]


def respond(bundle, adapter, *, outputs=None, fmt="csv"):
    """Fill ``bundle`` with the outputs of a Python adapter's code, reading every input back from the files."""
    manifest = _manifest(bundle)
    if Conventions(**manifest["conventions"]) != adapter.conventions:
        raise BundleError(f"the bundle was exported for {manifest['conventions']}, not {adapter.name}'s conventions")
    out_root = Path(outputs or bundle)
    written = 0
    for task, entry in _tasks(manifest):
        for name, run in task.runs.items():
            if entry["not_applicable"] or not adapter.supports(run.kind) or run.required_features() - set(adapter.features):
                continue
            rid = run_ids(task)[name]
            inputs = read_inputs(Path(bundle) / "runs" / rid, run.kind, fmt)
            try:
                if run.kind == "zprime":
                    value = adapter.zprime(inputs["zeta"])
                elif run.kind == "spectral_density":
                    value = adapter.spectral_density(inputs["plasma"], inputs["wavelengths_nm"])
                else:
                    value = adapter.detector_spectrum(inputs["volume"], inputs["detector"])
            except NotSupported:
                continue
            folder = out_root / "runs" / rid
            folder.mkdir(parents=True, exist_ok=True)
            write_output(folder, run.kind, value, fmt)
            written += 1
    return written


def score(bundle, *, outputs=None, name="code"):
    """Score the outputs in a bundle: ``(results, meta)`` for :mod:`tsverify.report`."""
    from .report import Meta

    manifest = _manifest(bundle)
    conventions = Conventions(**manifest["conventions"])
    out_root = Path(outputs or bundle)
    loaded = {}
    for r in manifest["runs"]:
        loaded[r["run"]] = read_output(out_root / r["folder"], r["kind"])
    has_zprime = any(loaded.get(r["run"]) is not None for r in manifest["runs"] if r["kind"] == "zprime")
    context = Context(conventions, REFERENCE, has_zprime)
    results = []
    for task, entry in _tasks(manifest):
        if entry["not_applicable"]:
            results.append(task.unsupported(f"{name}: {entry['not_applicable']}"))
            continue
        ids = run_ids(task)
        outputs_ = {n: loaded[ids[n]] for n in task.runs}
        missing = [ids[n] for n, v in outputs_.items() if v is None]
        if missing:
            results.append(task.unsupported(f"{name}: no output ({task.runs[next(iter(task.runs))].kind})"))
            continue
        bad = _shape_problem(task, outputs_)
        if bad:
            results.append(dataclasses.replace(task.unsupported(""), status="failed", message=f"{task.id}: {bad}"))
            continue
        results.append(task.result(context, outputs_))
    meta = Meta(code=name, route=f"file exchange ({Path(bundle).name})", conventions=conventions)
    return results, meta


def _shape_problem(task, outputs):
    for n, run in task.runs.items():
        value = outputs[n]
        if run.kind == "zprime":
            expected, got = run.zeta.size, np.size(value)
        elif run.kind == "spectral_density":
            expected, got = run.wavelengths_nm.size, np.size(value)
        else:
            expected, got = run.detector.pixels, np.size(value[1])
            if np.size(value[0]) != got:
                return f"run {n}: pixel_center_nm and value differ in length"
        if got != expected:
            return f"run {n}: the output has {got} values, the run needs {expected}"
    return None


def _readme(manifest):
    conventions = ", ".join(f"{k}={v!r}" for k, v in manifest["conventions"].items())
    return f"""# tsverify bundle

Exported by tsverify {manifest['tsverify']} for a code with the conventions {conventions}.
It holds {len(manifest['runs'])} runs for {len(manifest['tasks'])} tests.

## What to do

1. Read `runs.csv`: one row per run, with its `kind`, its `folder` and the `features` it needs.
2. For each run your code supports, read the inputs in its folder and write `output.csv` (or `output.json`) there.
   Skip a run your code cannot do; it is reported as not supported.
3. Score: `tsverify score <this folder> --name mycode --report report.md`.

## Inputs (CSV)

Plasmas are written in your code's conventions: ne in 1e20 cm^-3, temperatures in keV, flows and drift in 1e6 cm/s
along k, angles in degrees, wavelengths in nm. EDF velocities are in units of sqrt(T/m).

| Kind | Files |
|---|---|
| `zprime` | `zeta.csv` (column `zeta`): real arguments of Z'(zeta) = -2 [1 + zeta Z(zeta)] |
| `spectral_density` | `plasma.csv`, `ions.csv`, `wavelengths.csv` (column `wavelength_nm`) |
| `detector_spectrum` | `plasma.csv`, `ions.csv`, `detector.csv`, `amp_ranges.csv`, optional `gradient.csv`, `aperture.csv` |

- `plasma.csv` (one row): `ne, Te, angle_deg, lam, ud, electron_edf, electron_kappa`. `electron_edf` is `maxwellian`,
  `mixture` (Gaussian components in `electron_components.csv`: `weight, shift, width`) or `kappa`
  (`electron_kappa`). Any non-Maxwellian EDF is also tabulated in `electron_f.csv` (`v, f`).
- `ions.csv` (one row per species): `A, Z, Ti, fract, Va, edf, kappa`; mixture components in `ion_components.csv`
  (`ion, weight, shift, width`, with `ion` counted from 1), tabulated EDFs in `ion<k>_f.csv`.
- `detector.csv` (one row): `channel, lam_min, lam_max, pixels, points_per_pixel, sigma, amplitude, notch_center,
  notch_width, notch_od` (NaN without a notch). Pixel centres are `linspace(lam_min, lam_max, pixels)`; the
  Gaussian IRF has standard deviation `sigma` nm; normalize so the maximum inside the `amp_ranges.csv` ranges
  (`low, high`) is `amplitude`.
- `gradient.csv`: `ne_percent, Te_percent, points`. ne and Te vary linearly end to end by those percentages,
  sampled at `points` equal-weight points with scales `linspace(1 - p/200, 1 + p/200, points)`.
- `aperture.csv`: `angle_deg, weight`. The spectrum is the weighted sum over these scattering angles.

`input.json` holds the same inputs in one file.

## Outputs

| Kind | `output.csv` columns |
|---|---|
| `zprime` | `re, im` (one row per zeta) |
| `spectral_density` | `S` (one row per wavelength; `wavelength_nm` may be included) |
| `detector_spectrum` | `pixel_center_nm, value` (one row per pixel) |

Or `output.json` with the same names as keys and arrays as values. Write full precision (17 significant digits).

`expected.csv` in each folder is the oracle's answer for that run, reported in your conventions.
"""
