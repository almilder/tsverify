"""``tsverify`` command line: export a bundle, score a code's outputs, or fill a bundle from a Python adapter."""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from . import __version__, exchange, report
from .conventions import ELECTRON_DRIFT, Conventions
from .oracle.physics import PER, RELATIVISTIC


def _conventions(args):
    return Conventions(
        flow_sign=args.flow_sign,
        electron_drift=args.electron_drift,
        relativistic=args.relativistic,
        spectrum_per=args.spectrum_per,
        dispersion_in_k=not args.vacuum_k,
        absolute=not args.shape_only,
        per_range_amplitudes=tuple(args.per_range_amplitudes),
        sigma_from=args.sigma_from,
    )


def _export(args):
    formats = exchange.FORMATS if args.format == "both" else (args.format,)
    manifest = exchange.export(args.bundle, _conventions(args), slow=args.slow, formats=formats, force=args.force)
    print(f"wrote {len(manifest['runs'])} runs for {len(manifest['tasks'])} tests to {args.bundle}; see its README.md")


def _score(args):
    results, meta = exchange.score(args.bundle, outputs=args.outputs, name=args.name)
    report.write(results, meta, args.report, args.json)
    counts = Counter(r.status for r in results)
    print(f"{meta.code}: {counts['passed']} passed, {counts['failed']} failed, {counts['not supported']} not supported")
    for r in results:
        if r.status == "failed":
            print(f"  FAILED {r.task}: {r.message}")
    for path in (args.report, args.json):
        if path:
            print(f"report: {path}")
    return 1 if counts["failed"] else 0


def _respond(args):
    from .pytest_plugin import load_adapter

    adapter = load_adapter(args.adapter)
    written = exchange.respond(args.bundle, adapter, outputs=args.outputs, fmt=args.format)
    print(f"{adapter.name}: wrote {written} outputs")


def parser():
    p = argparse.ArgumentParser(prog="tsverify", description="Analytic conformance tests for Thomson-scattering codes.")
    p.add_argument("--version", action="version", version=f"tsverify {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="write the inputs of every run, and the oracle's answers, to a bundle folder")
    e.add_argument("bundle")
    e.add_argument("--format", choices=("csv", "json", "both"), default="both")
    e.add_argument("--slow", action="store_true", help="include the A1 sweep (18 more spectra)")
    e.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    c = e.add_argument_group("your code's conventions (default: tsverify's canonical ones)")
    c.add_argument("--flow-sign", type=int, choices=(1, -1), default=1)
    c.add_argument("--electron-drift", choices=ELECTRON_DRIFT, default="ion_frame")
    c.add_argument("--relativistic", choices=RELATIVISTIC, default="first_order")
    c.add_argument("--spectrum-per", choices=PER, default="wavelength")
    c.add_argument("--vacuum-k", action="store_true", help="the code uses vacuum wavenumbers omega/c")
    c.add_argument("--shape-only", action="store_true", help="the code's spectra are not in absolute units")
    c.add_argument("--per-range-amplitudes", action="append", default=[], metavar="CHANNEL",
                   help="the code's fits on this channel (EPW or IAW) use one amplitude per fit range; repeatable")
    c.add_argument("--sigma-from", choices=("noise_model", "residuals"), default="noise_model",
                   help="where the code's fit uncertainties come from")
    e.set_defaults(func=_export)

    s = sub.add_parser("score", help="score a code's outputs in a bundle and write the report")
    s.add_argument("bundle")
    s.add_argument("--outputs", help="folder holding runs/<run>/output.* (default: the bundle itself)")
    s.add_argument("--name", default="code", help="the code's name for the report")
    s.add_argument("--report", help="Markdown report path")
    s.add_argument("--json", help="JSON report path")
    s.set_defaults(func=_score)

    r = sub.add_parser("respond", help="fill a bundle's outputs from a Python adapter (MODULE:CLASS)")
    r.add_argument("bundle")
    r.add_argument("adapter")
    r.add_argument("--outputs")
    r.add_argument("--format", choices=exchange.FORMATS, default="csv")
    r.set_defaults(func=_respond)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except exchange.BundleError as error:
        print(f"tsverify: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
