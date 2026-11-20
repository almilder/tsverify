function zp = zprime(zeta)
%ZPRIME Z'(zeta) = -2 [1 + zeta Z(zeta)] at real zeta, with Z(zeta) = i sqrt(pi) w(zeta).
zp = -2*(1 + zeta.*(1i*sqrt(pi)*faddeeva_w(zeta)));
end
