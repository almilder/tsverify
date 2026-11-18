# tsverify

An analytic conformance suite for Thomson-scattering codes.

tsverify checks a Thomson-scattering code against an independent analytic oracle at four depths: the
susceptibility χ, the spectral density S(k,ω), the full detector spectrum, and the inverse problem
(recovery, uncertainty and calibration). The oracle needs only NumPy and SciPy. It is validated against
physics it shares no code with: the Debye–Hückel sum rule, the kinetic Bohm–Gross and sound-speed limits,
Selden's relativistic tables, and the Faddeeva function.

It started as the physics test suite of [tsadar](https://github.com/ergodicio/tsadar) and is being
extracted into a code-independent package. **Status: pre-release, nothing to install yet.**

## License

BSD 3-Clause, see [LICENSE](LICENSE). You may use it in any work, commercial or not.

If tsverify contributes to published work, please cite it. [CITATION.cff](CITATION.cff) has the
citation, and GitHub shows it under "Cite this repository". A DOI will be added with the first release.
