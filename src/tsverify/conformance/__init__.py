"""Conformance tests, run against a code through its adapter.

``pytest --pyargs tsverify.conformance --ts-adapter mypackage.module:MyAdapter``, or, inside a code's own test
suite, ``from tsverify.conformance import *`` in a test module with a ``ts_adapter`` fixture in scope.
"""

from .test_conventions import test_convention_probe
from .test_detector import test_detector_spectrum_matches_the_oracle
from .test_relativistic import test_first_order_relativistic_accuracy_against_the_exact_spectrum
from .test_spectral_density import (
    test_high_z_ion_acoustic_resonance,
    test_spectral_density_matches_the_oracle,
    test_spectral_density_matches_the_oracle_across_plasmas,
)
from .test_zprime import test_zprime_matches_faddeeva
from .test_volume import (
    test_aperture_weighted_sum_matches_the_oracle,
    test_gradient_average_matches_the_averaged_oracle_spectrum,
    test_the_same_plasma_written_differently_gives_the_same_spectrum,
    test_zero_gradient_collapses_to_a_single_point,
)

__all__ = [
    "test_convention_probe",
    "test_aperture_weighted_sum_matches_the_oracle",
    "test_detector_spectrum_matches_the_oracle",
    "test_first_order_relativistic_accuracy_against_the_exact_spectrum",
    "test_gradient_average_matches_the_averaged_oracle_spectrum",
    "test_high_z_ion_acoustic_resonance",
    "test_spectral_density_matches_the_oracle",
    "test_spectral_density_matches_the_oracle_across_plasmas",
    "test_the_same_plasma_written_differently_gives_the_same_spectrum",
    "test_zero_gradient_collapses_to_a_single_point",
    "test_zprime_matches_faddeeva",
]
