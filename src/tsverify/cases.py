"""The case catalog: analytic plasmas on a spectrometer channel, each checked at every depth.

Case IDs are ``<name>@<version>``. A released ID never changes meaning; a changed case gets a new version.
"""

from __future__ import annotations

from dataclasses import dataclass

from .instrument import Detector
from .oracle import GaussianComponent as G
from .oracle import Ion, KappaEDF, Plasma


@dataclass(frozen=True)
class Case:
    name: str
    plasma: Plasma
    detector: Detector
    problem: str
    version: int = 1

    @property
    def id(self):
        return f"{self.name}@{self.version}"

    def __str__(self):
        return self.name


EPW_DETECTOR = Detector(
    channel="EPW",
    lam_range=(440.0, 630.0),
    pixels=128,
    points_per_pixel=8,
    sigma=1.3,
    amp_ranges=((442.0, 522.0), (531.0, 628.0)),
    notch=(526.5, 24.0, 4.0),
)

IAW_DETECTOR = Detector(
    channel="IAW",
    lam_range=(524.5, 528.5),
    pixels=128,
    points_per_pixel=8,
    sigma=0.05,
    amp_ranges=((524.6, 526.4), (526.6, 528.4)),
)


HYDROGEN = Ion(A=1.0, Z=1.0, Ti=0.08)

A1_CASES = [
    Case("A1-epw-hydrogen", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,)), EPW_DETECTOR, "A1"),
    Case("A1-epw-electron-drift", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), ud=60.0), EPW_DETECTOR, "A1"),
    Case(
        "A1-iaw-hydrogen-flow",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, Va=1.5),)),
        IAW_DETECTOR,
        "A1",
    ),
    Case(
        "A1-iaw-CH-relative-drift",
        Plasma(
            ne=0.5,
            Te=1.0,
            ions=(Ion(A=12.0, Z=6.0, Ti=0.2, fract=0.5, Va=1.0), Ion(A=1.0, Z=1.0, Ti=0.2, fract=0.5, Va=3.0)),
        ),
        IAW_DETECTOR,
        "A1",
    ),
]

A3_CASES = [
    Case(
        "A3-epw-two-temperature",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.85, 0.0, 1.0), G(0.15, 0.0, 2.0))),
        EPW_DETECTOR,
        "A3",
    ),
    Case(
        "A3-epw-hot-tail-shoulder",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.9, 0.0, 1.0), G(0.1, 3.0, 0.8))),
        EPW_DETECTOR,
        "A3",
    ),
    *[
        Case(
            f"A3-{det.channel.lower()}-return-current",
            Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.8, 0.4, 1.0), G(0.2, -1.6, 0.8))),
            det,
            "A3",
        )
        for det in (EPW_DETECTOR, IAW_DETECTOR)
    ],
    Case(
        "A3-epw-flat-top",
        Plasma(
            ne=0.2,
            Te=0.6,
            ions=(HYDROGEN,),
            electrons=(G(1 / 3, -1.0, 0.8), G(1 / 3, 0.0, 0.8), G(1 / 3, 1.0, 0.8)),
        ),
        EPW_DETECTOR,
        "A3",
    ),
    Case(
        "A3-iaw-ion-beam",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, components=(G(0.9, 0.0, 1.0), G(0.1, 4.0, 0.7))),)),
        IAW_DETECTOR,
        "A3",
    ),
    Case(
        "A3-iaw-two-humped-ions",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, components=(G(0.5, -0.8, 0.7), G(0.5, 0.8, 0.7))),)),
        IAW_DETECTOR,
        "A3",
    ),
]

A5_CASES = [
    Case(
        f"A5-{det.channel.lower()}-kappa-{kappa}",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=KappaEDF(float(kappa))),
        det,
        "A5",
    )
    for kappa in (3, 5)
    for det in (EPW_DETECTOR, IAW_DETECTOR)
]

SPECTRUM_CASES = A1_CASES + A3_CASES + A5_CASES
