# tsverify

An analytic conformance suite for Thomson-scattering codes.

tsverify checks a Thomson-scattering code against an independent analytic oracle at four depths: the
susceptibility χ, the spectral density S(k,ω), the full detector spectrum, and the inverse problem
(recovery, uncertainty and calibration). The oracle needs only NumPy and SciPy. It is validated against
physics it shares no code with: the Debye–Hückel sum rule, the kinetic Bohm–Gross and sound-speed limits,
Selden's relativistic tables, and the Faddeeva function.

It started as the physics test suite of [tsadar](https://github.com/ergodicio/tsadar) and is being
extracted into a code-independent package. **Status: pre-release (0.1).** Version 0.1 contains the oracle, its
self-tests and the case catalog. The adapter interface for other codes comes next.

## Install and check the oracle

```
pip install "tsverify @ git+https://github.com/almilder/tsverify"
pip install pytest
pytest --pyargs tsverify.selftest
```

## Contents

- `tsverify.oracle`: the collisionless spectral density S(k,ω) for Gaussian-mixture and kappa EDFs, the
  analytic detector response (pixel-exact IRF blur, notch, normalization), Selden's exact relativistic
  spectrum, and comparison metrics.
- `tsverify.cases`: the case catalog (Maxwellian, Gaussian-mixture and kappa plasmas on EPW and IAW
  channels), with versioned IDs such as `A1-epw-hydrogen@1`.
- `tsverify.selftest`: the oracle against the Debye–Hückel sum rule, the kinetic Bohm–Gross and
  screened sound-speed limits, quadrature against Faddeeva, and Selden's tables.

Units follow the case catalog: ne in 10^20 cm^-3, temperatures in keV, flows in 10^6 cm/s along k, and
wavelengths in nm.

## License

BSD 3-Clause, see [LICENSE](LICENSE). You may use it in any work, commercial or not.

If tsverify contributes to published work, please cite it. [CITATION.cff](CITATION.cff) has the
citation, and GitHub shows it under "Cite this repository". A DOI will be added with the first release.
