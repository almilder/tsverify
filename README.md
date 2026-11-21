# tsverify

An analytic conformance suite for Thomson-scattering codes.

tsverify checks a Thomson-scattering code against an independent analytic oracle at four depths: the
susceptibility χ, the spectral density S(k,ω), the full detector spectrum, and the inverse problem
(recovery, uncertainty and calibration). The oracle needs only NumPy and SciPy. It is validated against
physics it shares no code with: the Debye–Hückel sum rule, the kinetic Bohm–Gross and sound-speed limits,
Selden's relativistic tables, and the Faddeeva function.

It started as the physics test suite of [tsadar](https://github.com/ergodicio/tsadar) and is being
extracted into a code-independent package. **Status: pre-release (0.7).** It contains the oracle, its
self-tests, the case catalog, and conformance tests at the Z′ (L1), spectrum (L2), detector (L3) and fitting
(L4) depths that any code can run, through a Python adapter or by exchanging files. Each code declares its own conventions. When a test
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

    def fit(self, problem):
        """L4: fit problem.data (a tsverify.fitting.FitProblem) and return a tsverify.fitting.FitResult."""
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
  tests it supports. Its spectra agree with the oracle to 5e-10 or better, peak-normalized, and its Z′
  matches the Faddeeva function exactly.
- `tsverify.adapters.toy:ToyAdapter`: a small Maxwellian code with switchable defects (`ToyAdapter("flow-sign")`
  and so on), used to check that each defect is caught and named.

tsadar's adapter is in its `tests/physics/adapter.py`.

To run the tests inside your own suite, put `from tsverify.conformance import *` in a test module and
provide a `ts_adapter` fixture. Add `--ts-report report.md` (and `--ts-report-json report.json`) to either
command for the conformance report.

## Test a code that is not in Python

Codes in Matlab, Fortran, C++ or IDL exchange files with tsverify instead of writing an adapter:

```
tsverify export bundle                  # every run's inputs, in CSV and JSON, plus the oracle's answers
<your code reads bundle/runs/<run>/... and writes output.csv into each run folder>
tsverify score bundle --name mycode --report report.md --json report.json
```

- `bundle/README.md` describes every file. `bundle/runs.csv` lists the runs: their kind (`zprime`,
  `spectral_density` or `detector_spectrum`), their folder, and the features they need.
- A run your code skips is reported as not supported.
- Plasmas are written in your code's conventions: pass them to `export` (`--flow-sign -1`,
  `--electron-drift lab`, `--relativistic none`, `--spectrum-per frequency`, `--vacuum-k`, `--shape-only`,
  `--per-range-amplitudes EPW`).
- Fits are runs too: each fit folder has the data, its variance and the free parameters, and the code writes
  `parameter, estimate, sigma`. The `self` Cramér–Rao test needs the code's own spectrum fed back as data, so
  it runs on the Python route only.
- `--slow` adds the A1 sweep.
- `expected.csv` in each run folder is the oracle's answer, which is handy for plotting.

The file route scores through exactly the same code as the Python route, so both give the same numbers. The
self-tests check this for every test, with the toy code's defects included.
`tsverify respond bundle MODULE:CLASS` fills a bundle from a Python adapter.

`examples/matlab/` is a small Maxwellian Thomson code in Matlab that answers a bundle
(`respond_maxwellian('bundle')`). It passes all 41 tests it supports: Z′ to 2e-15 and spectra to 5e-12.

## The report

`tsverify score` and `pytest --ts-report` write the same report. It contains:

- the declared conventions;
- a summary for each level;
- every failure, with the convention or known defect that explains it;
- a table for each level of the measured errors against their limits;
- the tests the code does not support, grouped by reason.

The JSON form carries the same content for scripts.

### Fitting (L4)

A fit problem hands the code a detector spectrum in counts (peak 1e4), the noise variance of each pixel, the
pixels to fit, the free parameters and their start values. Every other parameter is at its true value. The
code minimizes Σ (data − model)²/variance however it likes, and returns its estimates and, if it has them,
their one-σ uncertainties. The model is the L3 spectrum times free amplitudes: one overall amplitude by
default, or one per fit range on the channels a code declares (`Conventions(per_range_amplitudes=("EPW",))`,
for codes that fit the blue and red wings with separate gains, as tsadar does).

| Test | Data | Check |
|---|---|---|
| recovery (EPW ne, Te; IAW Va, Ti) | noiseless oracle spectrum, start displaced (ne −20 %, Te +30 %, Va −2, Ti +63 %, amplitude −20 %) | every parameter within 1e-3 relative (Va 1e-3 absolute) |
| σ against the Cramér–Rao bound, `self` | the code's own noiseless spectrum (needs L3; Python route only) | σ not more than 1 % below the bound from the oracle's Jacobian; efficiency (bound/σ)² reported |
| σ against the Cramér–Rao bound, `oracle` | the noiseless oracle spectrum | σ not more than 5 % below the bound; efficiency reported |
| calibration (`slow`) | 32 seeded realizations of counting noise | the pulls (estimate − noiseless estimate)/σ have mean 0 and width 1 within their 3σ sampling bands; 1σ and 2σ coverage are reported |
| camera noise (`slow`, `heavy`, `fit_shot`) | 12 shots × 4 lineouts with camera noise the code sizes itself (below), in four regimes | pooled pulls as for calibration; the read noise the code sized, against the truth |

The Cramér–Rao test is one-sided: no unbiased fit beats the bound, so a σ below it is overconfident and fails.
A σ above it passes with its efficiency reported and a note in the report: it is honest for a fit that is
inefficient by design (for example unweighted least squares), and an overestimate for an efficient one, which
the calibration test then shows as pulls narrower than 1. A code whose σ comes from its fit residuals (for
example a sandwich covariance) declares `Conventions(sigma_from="residuals")`: residuals vanish on the
noiseless data of the Cramér–Rao test, so that test does not apply and calibration checks it instead.

A σ that is off the bound by √2 or 2 for every parameter is named as a loss-normalization slip (the Hessian
of χ² against that of −ln L = χ²/2). `tsverify.fitting.least_squares_fit` is a reference weighted
least-squares fitter for a Python code that has an L3 model but no fitter of its own.

**Camera noise (`Adapter.fit_shot`).** Real noise must be sized from the data: it drifts from day to day with the
camera and from shot to shot with the background. A `ShotProblem` gives the code several lineouts of one shot that
share a camera (`tsverify.camera.Camera`), each with its background, and the camera's *nominal* parameters, but
not the variance. The noise is Poisson photoelectrons with an excess-noise factor, spread over neighbouring pixels
by the CCD spread function, plus read noise summed over the lineout's rows: the form of tsadar's `covar` model.
The gain, excess-noise factor, rows and spread function given are right; the read noise may not be.

| Regime | true read noise (drifts ×0.8–1.25 per shot) | read noise the code is told | background |
|---|---|---|---|
| shot-noise | 2 | 2 | 0 |
| background | 10 | 10 | 2000 counts (drifts ×0.7–1.4, 30 % ramp) |
| read-noise | 65 | 65 | 50 |
| miscalibrated | 65 | 17 | 200 |

`tsverify.fitting.sized_shot_fit` is the reference: it fits each lineout with the full covariance of the nominal
camera, sizes the read noise by maximum likelihood from the residuals of all lineouts pooled, and fits again. Shot
problems run on the Python route only for now.

**Two fitters across count levels.** `tsverify.convergence.compare_fitters(a, b)` fits the same noisy data with two
adapters (two codes, or two configurations of one) at 1e3, 1e4 and 1e5 counts. Consistent fitters differ by an
amount that shrinks like the noise (log-log slope about −0.5) and stays a fixed fraction of σ; a difference that
stops shrinking is systematic.

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
| `per_range_amplitudes` | `()`: L4 fits use one amplitude; `("EPW",)`: one per fit range on the EPW channel |
| `sigma_from` | `"noise_model"`: σ from a likelihood with a noise model; `"residuals"`: σ from the fit residuals |

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

Physical constants are the CODATA 2022 values from NIST (https://physics.nist.gov/cuu/Constants/), cited in
`tsverify/oracle/constants.py`.

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
- `tsverify.tasks`: every check as runs plus a scoring function, shared by both routes.
- `tsverify.fitting`: the fit and shot problems and results, parameter names, the Cramér–Rao bound, the
  reference fitter and read-noise sizing.
- `tsverify.camera`: the camera noise model and generator. `tsverify.convergence`: two fitters across count
  levels.
- `tsverify.exchange`, `tsverify.report` and the `tsverify` command: the file route and the reports.
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
