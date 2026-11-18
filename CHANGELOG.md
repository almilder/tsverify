# Changelog

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
