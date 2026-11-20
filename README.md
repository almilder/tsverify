# tsverify

An analytic conformance suite for Thomson-scattering codes.

tsverify checks a Thomson-scattering code against an independent analytic oracle at four depths: the
susceptibility χ, the spectral density S(k,ω), the full detector spectrum, and the inverse problem
(recovery, uncertainty and calibration). The oracle needs only NumPy and SciPy. It is validated against
physics it shares no code with: the Debye–Hückel sum rule, the kinetic Bohm–Gross and sound-speed limits,
Selden's relativistic tables, and the Faddeeva function.

It started as the physics test suite of [tsadar](https://github.com/ergodicio/tsadar) and is being
extracted into a code-independent package. **Status: pre-release (0.4).** It contains the oracle, its
self-tests, the case catalog, and conformance tests at the Z′ (L1), spectrum (L2) and detector (L3) depths
that any Python-callable code can run through an adapter. Each code declares its own conventions. When a test
fails, tsverify names the convention mismatch or known defect that explains it, or says that none does.

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
from tsverify.conventions import Conventions


class MyCodeAdapter(Adapter):
    name = "mycode"
    features = frozenset({"multiple_ions", "ion_flow"})
    conventions = Conventions(relativistic="none", absolute=False)

    def zprime(self, zeta):
        """L1: the code's Z'(zeta) = -2 [1 + zeta Z(zeta)] for Maxwellian species, at real zeta."""
        ...

    def spectral_density(self, plasma, wavelengths_nm):
        """L2: S at ``wavelengths_nm`` as the code reports it (see the conventions)."""
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
runs the oracle itself as a code, which checks the tests. Two more adapters ship as examples:

- `tsverify.adapters.plasmapy:PlasmaPyAdapter`: PlasmaPy's `thomson.spectral_density` (`pip install
  "tsverify[plasmapy]"`). It supports L1 and L2 for Maxwellian and Gaussian-mixture plasmas, and passes all 46
  tests it supports. Its spectra agree with the oracle to 3.0e-4 or better, peak-normalized, and its Z′
  matches the Faddeeva function exactly.
- `tsverify.adapters.toy:ToyAdapter`: a small Maxwellian code with switchable defects (`ToyAdapter("flow-sign")`
  and so on), used to check that each defect is caught and named.

tsadar's adapter is in its `tests/physics/adapter.py`.

To run the tests inside your own suite, put `from tsverify.conformance import *` in a test module and
provide a `ts_adapter` fixture.

### Conventions

Codes differ in how they write inputs and report outputs. tsverify's plasma description is canonical:
flows `Va` and the electron drift `ud` are positive along k = k_s − k_i, and `ud` is relative to the
charge-weighted ion fluid. An adapter declares how its code differs (`tsverify.conventions.Conventions`).
tsverify then hands the code each plasma written its way, and computes the oracle's spectrum the way the
code reports it.

| Field | Values (canonical first) |
|---|---|
| `flow_sign` | `+1`: flows and drift positive along k_s − k_i; `−1`: along k_i − k_s |
| `electron_drift` | `"ion_frame"`: relative to the ion fluid; `"lab"`: the lab-frame electron velocity |
| `relativistic` | `"first_order"`: the (1 + 2ω/ω_L) factor; `"none"` (the A12 test is then skipped) |
| `spectrum_per` | `"wavelength"` or `"frequency"` (L2 output; L3 is always integrated over wavelength) |
| `dispersion_in_k` | `True`: k_s, k_i include the plasma dispersion; `False`: vacuum ω/c |
| `absolute` | `True`: L2 in the oracle's absolute units; `False`: L2 compared as a shape |

The **convention probes** are four small asymmetric plasmas: an ion flow, an electron drift, a drift with
flowing ions, and broad EPW wings. When a code fails a probe, the probe retries every single-convention
change and names the ones under which the code agrees with the oracle, for example "It agrees with the
oracle if the code's flows and drift are positive along k_i − k_s (flow_sign=−1)". If none does, the
failure is reported as a likely physics error.

### When a test fails

A failing spectrum test retries the oracle under every single-convention change and with every known defect
in `tsverify.signatures`, and names each one under which the code agrees:

| Signature | What the code does |
|---|---|
| `flow_sign=-1`, `relativistic='none'`, … | a convention other than the declared one (see above) |
| `Te-doubled`, `Ti-halved`, `T-doubled`, … | a temperature off by 2, as from mixing v_T = √(T/m) and √(2T/m) |
| `forward-difference-f'`, `backward-difference-f'` | the electron f′ taken between neighbouring wavelength samples (an O(h) error) |

If nothing explains the failure, it is reported as a likely physics error. The L1 test reads the error of the
code's Z′ for table edges (jumps), linear interpolation between table nodes (zeros of the error at regular
spacing), and the error of the rdWT table tsadar used before November 2026.

EDF velocities in `GaussianComponent` and `QuadratureEDF` are in units of √(T/m). A code that uses
√(2T/m) rescales them in its adapter.

## Contents

- `tsverify.oracle`: the collisionless spectral density S(k,ω) for Gaussian-mixture and kappa EDFs, the
  analytic detector response (pixel-exact IRF blur, notch, normalization), Selden's exact relativistic
  spectrum, and comparison metrics.
- `tsverify.cases`: the case catalog (Maxwellian, Gaussian-mixture and kappa plasmas on EPW and IAW
  channels), with versioned IDs such as `A1-epw-hydrogen@1`.
- `tsverify.volume`: the scattering volume, which is a plasma plus an optional linear gradient and angular
  aperture, and its oracle spectrum.
- `tsverify.conformance`: the code-under-test checks. They cover Z′ against the Faddeeva function (L1), the
  L2 spectral density for every case at
  two resolutions, a high-Z IAW resonance, an A1 density/temperature/angle sweep (marked `slow`), and the
  L3 detector spectrum for every case. They also cover gradient and aperture averaging and invariance
  under rewriting a plasma, plus the first-order relativistic spectrum against Selden's exact one and the
  four convention probes.
- `tsverify.conventions`: `Conventions`, `CANONICAL` and `TSADAR`.
- `tsverify.signatures`: the known-defect signatures and the explanations of a failure.
- `tsverify.adapters`: the oracle, PlasmaPy and toy adapters.
- `tsverify.instrument.Detector`: a spectrometer channel (pixel axis, Gaussian IRF, fit ranges, notch, peak
  windows).
- `tsverify.selftest`: the oracle against the Debye–Hückel sum rule, the kinetic Bohm–Gross and
  screened sound-speed limits, quadrature against Faddeeva, and Selden's tables. It also checks the
  adapter plumbing, the convention options, that each wrongly declared convention is named by a
  probe, and that each defect planted in the toy code is caught and named alone.

Units follow the case catalog: ne in 10^20 cm^-3, temperatures in keV, flows in 10^6 cm/s along k, and
wavelengths in nm.

## License

BSD 3-Clause, see [LICENSE](LICENSE). You may use it in any work, commercial or not.

If tsverify contributes to published work, please cite it. [CITATION.cff](CITATION.cff) has the
citation, and GitHub shows it under "Cite this repository". A DOI will be added with the first release.
