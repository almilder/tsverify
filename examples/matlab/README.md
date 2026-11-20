# Matlab example

A small collisionless Thomson-scattering code for Maxwellian electrons and ions, written in Matlab, that answers a
tsverify bundle through files. It covers L1 (Z′) and L2 (spectra). It is an example of the file route, not a
reference implementation.

```
tsverify export bundle --format csv
matlab -batch "addpath('path/to/tsverify/examples/matlab'); respond_maxwellian('bundle')"
tsverify score bundle --name matlab-example --report report.md
```

| File | What it does |
|---|---|
| `respond_maxwellian.m` | loops over `runs.csv`, reads each supported run's inputs, writes `output.csv` |
| `maxwellian_spectral_density.m` | S(k, ω) in tsverify's canonical conventions and units, with CODATA 2022 constants |
| `zprime.m`, `faddeeva_w.m` | Z′(ζ) from the Faddeeva function (Weideman's expansion; base Matlab has none) |

With tsverify 0.5 and Matlab R2023b it passes all 41 tests it supports: Z′ to 2.3e-15, spectra to 5.3e-12 and
the A12 relativistic checks. The other 43 tests (non-Maxwellian EDFs and the detector level) are reported as not
supported.

Write outputs with `%.17g`, as `respond_maxwellian.m` does. `writetable` rounds to 15 digits.
