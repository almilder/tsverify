# Changelog

## 0.5.0 (2026-11-20)

File exchange, a command-line tool and reports (Phase 5).

- `tsverify.tasks`: every conformance check is now a task. A task lists the runs the code computes (Z′, a
  spectrum or a detector spectrum) and scores their outputs into a `Result`: status, measurements, the rows for
  the report, and the message with its diagnosis.
  - The pytest tests are thin wrappers over the tasks, with the same names, parameters and recorded values.
    Oracle, toy and PlasmaPy: 84 tests each, no state changes, every recorded value bit-identical.
  - L3 and A10 now record their measurements too.
- `tsverify.exchange` and the `tsverify` command (file route):
  - `tsverify export BUNDLE` writes every run's inputs in the code's declared conventions (`--flow-sign`,
    `--electron-drift`, `--relativistic`, `--spectrum-per`, `--vacuum-k`, `--shape-only`), as CSV and/or JSON,
    with the oracle's answer for each run and a README that describes every file. `--slow` adds the sweep.
  - The code writes `output.csv` or `output.json` into each run folder it supports.
  - `tsverify score BUNDLE --report report.md --json report.json` scores those outputs through the same tasks.
    A missing output counts as not supported; an output of the wrong length fails.
  - A bundle is scored only by the tsverify version that exported it.
  - `tsverify respond BUNDLE MODULE:CLASS` fills a bundle from a Python adapter.
  - Floats are written with `repr`, so every input reads back exactly.
- `tsverify.report`: a Markdown and JSON report, written by both routes (`tsverify score`, `pytest --ts-report`
  and `--ts-report-json`). It holds the conventions, a summary per level, the failures with their diagnoses, the
  measured errors against their limits, and the unsupported tests grouped by reason.
- `examples/matlab/`: a Maxwellian Thomson code in Matlab, with Weideman's Faddeeva expansion, that answers a
  bundle. Under Matlab R2023b it passes all 41 tests it supports: Z′ to 2.3e-15, spectra to 5.3e-12.
- Self-tests (67), including:
  - inputs read back exactly from CSV and JSON under all-different conventions;
  - the file route gives the same status, measurements and message as the Python route for every task, for the
    oracle under all-different conventions, the toy code, and two toy defects;
  - malformed and missing outputs, the version check, and the command line.
- CI runs the file route with the toy code. A new `bundle` workflow attaches the canonical CSV and JSON bundles
  to each tagged release.
- Gate against tsadar (branch `tsverify-phase5`, with tsadar on CODATA constants as in #171):
  - pytest route: 143 → 143 passed, no state changes. 113 values identical, plus 75 new detector measurements.
  - File route: `TsadarAdapter` answers all 87 runs, and all 84 tasks pass. All 177 measurements and every status
    are identical to the pytest route.
  - tsadar's legacy-Z′ test now scores through `tsverify.execute`, so its deliberate failure stays out of the
    report.

## 0.4.0 (2026-11-20)

Known-defect signatures, an L1 Z′ level, and two more codes under test: PlasmaPy and a toy code with planted
defects (Phase 4).

- `tsverify.signatures`: when a spectrum test fails (L2 cases, high-Z, probes), tsverify retries the oracle
  under every single-convention change and with every known defect, and names each one under which the code
  agrees.
  - The defects: a temperature off by 2 (electrons, ions or both, either way), as from mixing
    v_T = √(T/m) and √(2T/m); and the electron f′ taken as a forward or backward difference between
    wavelength samples.
  - If nothing explains the failure, it is called a likely physics error, with a pointer to the L1 test.
- L1 `Adapter.zprime(zeta)` and `tsverify.conformance.test_zprime_matches_faddeeva`: the code's Z′ on
  −15 ≤ ζ ≤ 15 against the Faddeeva function, to `REFERENCE.zprime_abs = 2.5e-4`. That is about the Z′ error
  that moves an IAW spectrum by the A1 L2 tolerance; rdWT's 1.6e-3 moved one by 0.65 %. A failure reads the
  error for table edges (jumps), linear interpolation between table nodes (regularly spaced zeros) and the
  error of tsadar's legacy rdWT table.
- `tsverify.adapters.toy:ToyAdapter(defect)`: a small Maxwellian code, written apart from the oracle, with
  five switchable defects: `flow-sign`, `no-relativistic-factor`, `forward-difference-f'`, `thermal-speed` and
  `zprime-table`. Without a defect it passes all 41 tests it supports. Self-test: each spectrum defect fails
  9–13 tests, and every one of those failures names that defect and nothing else. The Z′ table is caught at
  L1, with its node spacing (0.1) and its edge (|ζ| = 8).
- `tsverify.adapters.plasmapy:PlasmaPyAdapter`: PlasmaPy 2025.8.0 `thomson.spectral_density` at L1 and L2
  (`pip install "tsverify[plasmapy]"`). It declares lab-frame electron velocities, S per angular frequency and
  no relativistic factor, and is converted to absolute units (S/2π · r_e² n_e).
  - It passes all 46 tests it supports. All four probes confirm the declared conventions. Kappa EDFs and L3
    are not supported.
  - With the CODATA constants (below), its worst Linf is 5.0e-10 (A1 sweep) and 2.4e-11 on the catalog
    cases. Z′ matches the Faddeeva function exactly.
- **The oracle's constants are now the CODATA 2022 values from NIST**
  (https://physics.nist.gov/cuu/Constants/), each one cited in `oracle/constants.py`. Previously they were
  tsadar's rounded values.

  | Constant | Before | Now (CODATA 2022) | Relative change |
  |---|---|---|---|
  | m_e c² | 510.9896 keV | 510.99895069 keV | 1.8e-5 |
  | m_p/m_e | 1836.1 | 1836.152673426 | 2.9e-5 |
  | r_e | 2.8179e-13 cm | 2.8179403205e-13 cm | 1.4e-5 |
  | c | 2.99792458e10 cm/s | unchanged (exact) | 0 |

  - This deliberately ends bit-identity with 0.2.0. Spectra move by parts in 10⁵.
  - With it, PlasmaPy agrees with the oracle to 5e-10 or better everywhere, down from 3.0e-4, so its whole
    residual was the rounded constants.
  - Against tsadar `tsverify-phase4`, on the same 143 tests:
    - Oracle on CODATA, tsadar still on rounded constants: 2 tests fail. The A1 electron-drift case at 2048
      points reaches Linf 1.02e-3 and the drift probe 1.0e-3, against 1e-3.
    - With tsadar on CODATA as well (a separate tsadar PR): 143 pass. The errors return to tsadar's own
      numerical level (drift probe 6.16e-4; worst L2 1.80e-3, against 2.5e-3).
    - The inverse-problem ratios move by at most 1.4e-4 relative.
- The L2 tests record their measured Linf and L1 (`record_property`).
- `tsverify.compare` holds `expected_spectral_density` and `spectral_errors`, so the diagnosis code needs no
  pytest. `oracle.electron_phase_velocity` and `oracle.plasma_dispersion_derivative` are new. With the old
  constants, this refactor left the default oracle path bit-identical to 0.2.0 for every catalog case.
- CI also runs the conformance tests against the toy code, plus a separate PlasmaPy job.
- Gate against tsadar (branch `tsverify-phase4`):
  - `TsadarAdapter.zprime` exposes FormFactor's ion Z′ (its 0.01-step table, interpolated linearly, plus
    the tails). A new tsadar test checks that `other.zprime: legacy` is named as the rdWT table with an edge
    at |ζ| = 8.2.
  - 141 → 143 passed. The 2 new tests are the L1 test (max error 1.0e-4, which is exactly the linear-interpolation
    bound h²/8·|Re Z‴(0)|) and the legacy-table test. No state changes.
  - All 33 earlier recorded values are identical, plus 79 newly recorded L2 errors (worst 1.8e-3 on A3
    two-humped ions at 2048 points, against 2.5e-3).

## 0.3.0 (2026-11-20, tag only)

Tagged on the Phase 3 commit, whose version string reads 0.3.0.dev0; there was no separate release commit.

Code conventions as explicit options, plus convention probes (Phase 3).

- `tsverify.conventions.Conventions`: `flow_sign`, `electron_drift` (ion frame or lab), `relativistic`
  (first order or none), `spectrum_per` (wavelength or frequency), `dispersion_in_k` and `absolute`.
  `CANONICAL` (= `TSADAR`) is the default. An adapter declares its code's conventions; the conformance tests
  hand the code each plasma written its way (`express`) and compute the oracle's spectrum the way the code
  reports it (`oracle_options`). Codes without absolute units are compared as shapes.
- Oracle options: `spectral_density(..., relativistic=, per=, dispersion_in_k=)` and
  `scattering_geometry(..., dispersion_in_k=)`. The defaults are bit-identical to 0.2.0 for every catalog
  case.
- `tsverify.conformance.test_convention_probe`: four probes (flow, drift, drift-frame, wings). On a failure, a
  probe names every single-convention change under which the code agrees with the oracle, or reports a
  likely physics error.
  - Self-test: an oracle adapter that declares the canonical conventions but behaves as each of the 6
    single-field alternatives is caught, and the right convention is named every time.
  - An oracle adapter that declares and uses all-different conventions passes.
- `Detector.peak_windows`: the peak-comparison windows now belong to the detector (EPW and IAW in the
  catalog), not to a table keyed by channel name.
- A12 is skipped for codes that declare no relativistic factor; per-frequency output is converted before the
  shape comparison.
- Gate against tsadar (branch `tsverify-phase3`, `TsadarAdapter.conventions = TSADAR`): 137 → 141 passed, the
  4 new tests being the probes. Their Linf under the declared conventions is 2.1e-4 (drift-frame), 2.3e-4
  (wings), 3.9e-4 (flow) and 6.2e-4 (drift), against a tolerance of 1e-3. No state changes, and all 29
  recorded values are identical.

## 0.2.0 (2026-11-18)

Conformance tests that any code can run through an adapter (Phase 2).

- `tsverify.adapter`: the `Adapter` base class, with optional levels L2 `spectral_density(plasma,
  wavelengths_nm)` and L3 `detector_spectrum(volume, detector)`, and declared `features`. A test that needs
  an unsupported level, feature or input is skipped as *not supported*.
- `tsverify.volume`: `Volume`, a plasma with an optional `LinearGradient` and `Aperture`, and its oracle
  spectrum.
- `tsverify.conformance`: tsadar's L2, L3, A10 and A12 tests and the A1 sweep, rewritten against the adapter.
  It runs with `pytest --pyargs tsverify.conformance --ts-adapter MODULE:CLASS` (pytest plugin) or with
  `from tsverify.conformance import *` inside a code's own suite.
- `tsverify.tolerances`: the `REFERENCE` tier, which holds tsadar PR 164's values. `tsverify.compare`: the
  detector-depth comparison.
- `tsverify.adapters.oracle:OracleAdapter`: the oracle presented as a code, which CI uses to check the tests
  (79 pass).
- Gate against tsadar (branch `tsverify-phase2`): 137 passed both before and after, with the same tests and
  no state changes. 18 recorded values are identical. The 11 A12 values moved by at most 2.3e-10 relative,
  because the oracle is now evaluated on the requested wavelength grid instead of the code's returned grid,
  which differs from it by roundoff.

## 0.1.0 (2026-11-18)

Extracted from tsadar's physics test suite (tsadar PR 164, main 8d41c99c), with no change in behaviour.

- `tsverify.oracle`: the oracle from tsadar's `tests/oracles.py`, split into `constants`, `edf`, `physics`,
  `relativistic`, `detector` and `metrics`. Every function was copied verbatim. The outputs are bit-identical
  to the original for all 15 catalog cases: S, its electron and ion parts, χ_e, χ_i, the detector spectrum,
  the metrics and Selden's spectrum.
- `tsverify.cases`: the A1, A3 and A5 cases and the EPW and IAW channels, with IDs `<name>@1`. Tolerances
  are no longer attached to cases; codes under test set their own.
- `tsverify.instrument.Detector`: the spectrometer-channel description.
- `tsverify.selftest`: tsadar's `test_oracle.py` and the Selden table check (22 tests). They pass without
  tsadar or JAX installed.
- Gate against tsadar (branch `tsverify-phase1`): every tsadar test that uses the oracle was run before and
  after the move. Result: 137 passed, against 157 on main; the 20-test difference is exactly the tests that
  moved here. No test changed state, and all 29 recorded measured values are identical.
