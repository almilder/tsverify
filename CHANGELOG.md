# Changelog

## 0.1.0 (unreleased)

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
