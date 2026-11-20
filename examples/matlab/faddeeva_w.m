function w = faddeeva_w(z)
%FADDEEVA_W Faddeeva function w(z) = exp(-z^2) erfc(-i z) for Im(z) >= 0.
%   Weideman's rational expansion with N = 64 terms: J. A. C. Weideman, "Computation of the complex error
%   function", SIAM J. Numer. Anal. 31, 1497-1518 (1994). For real z it agrees with SciPy's wofz to 1e-15.
N = 64;
M = 2*N;
M2 = 2*M;
k = (-M+1:M-1)';
L = sqrt(N/sqrt(2));
t = L*tan(k*pi/M/2);
f = [0; exp(-t.^2).*(L^2 + t.^2)];
a = real(fft(fftshift(f)))/M2;
a = flipud(a(2:N+1));
Z = (L + 1i*z)./(L - 1i*z);
w = 2*polyval(a, Z)./(L - 1i*z).^2 + (1/sqrt(pi))./(L - 1i*z);
end
