function S = maxwellian_spectral_density(plasma, ions, lam_nm)
%MAXWELLIAN_SPECTRAL_DENSITY Collisionless Thomson spectrum of Maxwellian electrons and ions.
%   S = MAXWELLIAN_SPECTRAL_DENSITY(PLASMA, IONS, LAM_NM) takes the one-row table from plasma.csv, the table
%   from ions.csv and a column of wavelengths in nm. It returns S in tsverify's canonical units: per unit
%   wavelength, with the (1 + 2 omega/omega_L) factor, times r_e^2 n_e. Flows and the electron drift are along
%   k = k_s - k_i, and the drift is relative to the charge-weighted ion fluid. Constants are CODATA 2022.
c = 2.99792458e10;
me = 510.99895069/c^2;
mp = 1836.152673426*me;
re = 2.8179403205e-13;
e2 = me*c^2*re;

lam = lam_nm(:)*1e-7;
ws = 2*pi*c./lam;
wl = 2*pi*c/(plasma.lam*1e-7);
w = ws - wl;
ne = plasma.ne*1e20;
wpe = sqrt(4*pi*e2*ne/me);
ks = sqrt(ws.^2 - wpe^2)/c;
kl = sqrt(wl^2 - wpe^2)/c;
k = sqrt(ks.^2 + kl^2 - 2*ks*kl*cosd(plasma.angle_deg));

share = ions.fract/sum(ions.fract);
zbar = sum(ions.Z.*share);
fluid = sum(ions.Z.*share.*ions.Va)/zbar;
chi_i = 0;
ion_part = 0;
for s = 1:height(ions)
    m = ions.A(s)*mp;
    vt = sqrt(ions.Ti(s)/m);
    wp = sqrt(4*pi*e2*ions.Z(s)^2*share(s)*ne/zbar/m);
    zeta = (w - k*ions.Va(s)*1e6)./(sqrt(2)*k*vt);
    chi_i = chi_i - zprime(zeta)./(2*(k*vt/wp).^2);
    ion_part = ion_part + share(s)*ions.Z(s)^2/zbar*exp(-zeta.^2)/(sqrt(2*pi)*vt);
end

vte = sqrt(plasma.Te/me);
zeta_e = (w - k*(fluid + plasma.ud)*1e6)./(sqrt(2)*k*vte);
chi_e = -zprime(zeta_e)./(2*(k*vte/wpe).^2);
fe = exp(-zeta_e.^2)/(sqrt(2*pi)*vte);

S = (abs(1 + chi_i).^2.*fe + abs(chi_e).^2.*ion_part)./abs(1 + chi_e + chi_i).^2./k;
S = S.*(1 + 2*w/wl)*re^2*ne.*(2*pi*c./lam.^2);
end
