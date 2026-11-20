"""Physical constants in the CGS units the oracle uses (energies in keV).

Every value is the CODATA 2022 recommended value published by NIST, https://physics.nist.gov/cuu/Constants/
(each constant's own page is cited below). e^2 is derived as m_e c^2 r_e.
"""

# c = 299 792 458 m/s, exact by definition of the metre. https://physics.nist.gov/cgi-bin/cuu/Value?c
C_CM_PER_S = 2.99792458e10

# m_e c^2 = 0.510 998 950 69(16) MeV. https://physics.nist.gov/cgi-bin/cuu/Value?mec2mev
ELECTRON_MASS_KEV = 510.99895069
ELECTRON_MASS = ELECTRON_MASS_KEV / C_CM_PER_S**2

# m_p / m_e = 1836.152 673 426(32). https://physics.nist.gov/cgi-bin/cuu/Value?mpsme
PROTON_MASS = 1836.152673426 * ELECTRON_MASS

# r_e = 2.817 940 3205(13) x 10^-15 m. https://physics.nist.gov/cgi-bin/cuu/Value?re
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403205e-13

ELECTRON_CHARGE_SQUARED = ELECTRON_MASS * C_CM_PER_S**2 * CLASSICAL_ELECTRON_RADIUS_CM
