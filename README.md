# tsverify

An analytic conformance suite for Thomson-scattering codes.

tsverify checks a Thomson-scattering code against an independent analytic oracle at four depths: the
susceptibility χ, the spectral density S(k,ω), the full detector spectrum, and the inverse problem
(recovery, uncertainty and calibration). The oracle needs only NumPy and SciPy. It is validated against
physics it shares no code with: the Debye–Hückel sum rule, the kinetic Bohm–Gross and sound-speed limits,
Selden's relativistic tables, and the Faddeeva function.

It started as the physics test suite of [tsadar](https://github.com/ergodicio/tsadar) and is being
extracted into a code-independent package. **Status: pre-release (0.2).** It contains the oracle, its
self-tests, the case catalog, and conformance tests at the spectrum (L2) and detector (L3) depths that any
Python-callable code can run through an adapter. Code conventions (flow sign, normalization, relativistic
factor) are tsadar's for now; making them explicit options is next.

## Install and check the oracle

```
pip install "tsverify @ git+https://github.com/almilder/tsverify"
pip install pytest
pytest --pyargs tsverify.selftest
```

## Test your code

Write an adapter: subclass `tsverify.adapter.Adapter`, implement the levels your code supports, and list
its features. A test that needs something the code does not support is skipped as *not supported*, which
is reported apart from a failure.

```python
from tsverify.adapter import Adapter


class MyCodeAdapter(Adapter):
    name = "mycode"
    features = frozenset({"multiple_ions", "ion_flow"})

    def spectral_density(self, plasma, wavelengths_nm):
        """L2: S per unit wavelength at ``wavelengths_nm``, in the units of tsverify.oracle.spectral_density."""
        ...

    def detector_spectrum(self, volume, detector):
        """L3: (pixel_centers_nm, values) of ``volume`` on ``detector``, normalized so the maximum inside
        detector.amp_ranges is detector.amplitude."""
        ...
```

Then run

```
pytest --pyargs tsverify.conformance --ts-adapter mypackage.module:MyCodeAdapter
```

The features are `multiple_ions`, `ion_flow`, `electron_drift`, `numeric_electron_edf`,
`numeric_ion_edf` (non-Maxwellian EDFs, given as `f(v)` through `tsverify.oracle.mixture_density` or a
`QuadratureEDF`), `notch`, `gradient` and `aperture`. Tolerances default to the `REFERENCE` tier in
`tsverify.tolerances`; an adapter can set its own `tolerances`. `tsverify.adapters.oracle:OracleAdapter`
runs the oracle itself as a code, which checks the tests. tsadar's adapter is in its
`tests/physics/adapter.py`.

To run the tests inside your own suite, put `from tsverify.conformance import *` in a test module and
provide a `ts_adapter` fixture.

## Contents

- `tsverify.oracle`: the collisionless spectral density S(k,ω) for Gaussian-mixture and kappa EDFs, the
  analytic detector response (pixel-exact IRF blur, notch, normalization), Selden's exact relativistic
  spectrum, and comparison metrics.
- `tsverify.cases`: the case catalog (Maxwellian, Gaussian-mixture and kappa plasmas on EPW and IAW
  channels), with versioned IDs such as `A1-epw-hydrogen@1`.
- `tsverify.volume`: the scattering volume, which is a plasma plus an optional linear gradient and angular
  aperture, and its oracle spectrum.
- `tsverify.conformance`: the code-under-test checks. They cover the L2 spectral density for every case at
  two resolutions, a high-Z IAW resonance, an A1 density/temperature/angle sweep (marked `slow`), and the
  L3 detector spectrum for every case. They also cover gradient and aperture averaging and invariance
  under rewriting a plasma, plus the first-order relativistic spectrum against Selden's exact one.
- `tsverify.selftest`: the oracle against the Debye–Hückel sum rule, the kinetic Bohm–Gross and
  screened sound-speed limits, quadrature against Faddeeva, and Selden's tables, plus the adapter
  plumbing.

Units follow the case catalog: ne in 10^20 cm^-3, temperatures in keV, flows in 10^6 cm/s along k, and
wavelengths in nm.

## License

BSD 3-Clause, see [LICENSE](LICENSE). You may use it in any work, commercial or not.

If tsverify contributes to published work, please cite it. [CITATION.cff](CITATION.cff) has the
citation, and GitHub shows it under "Cite this repository". A DOI will be added with the first release.
