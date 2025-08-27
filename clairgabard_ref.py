# Copyright (c) 2025 Tianyu Li.
#
# If you use or adapt this code in scholarly work, please cite
# T. Li, Y. Zha, and W. Jiang, "A Fourier Spectral Method for Acoustic Wave Scattering by Localized Flow Inhomogeneities,"
# Journal of Theoretical and Computational Acoustics, 33, 2550013, 2025.
# doi: 10.1142/S2591728525500136

import numpy as np
from scipy.special import hankel1, jv

try:
    from tqdm.auto import trange
except Exception:
    trange = range


GAMMA = 1.4

class ClairGabardRef:
    '''This is my personal re-implementation of the Clair and Gabard's semi-ode method, reference:
    V. Clair and G. Gabard, Spectral broadening of acoustic waves by convected vortices, Journal of Fluid Mechanics 841 (2018) 50-80. DOI: 10.1017/jfm.2018.94 .
    '''
    def __init__(self, wavelength, mach, radial_points, rmax=6.0, modes=50):
        self.k = 2 * np.pi / wavelength
        self.ma = -float(mach)
        self.h = float(rmax) / int(radial_points)
        self.n = int(radial_points)
        self._modes = np.arange(-int(modes), int(modes) + 1, dtype=int)
        self._am = None

    def taylor_vortex(self, r):
        exp_full = np.exp(1 - r**2)
        exp_half = np.exp((1 - r**2) / 2)
        Utheta = exp_half * r * self.ma
        dUtheta = self.ma * (1 - r**2) * exp_half
        C0sq = 1 - 0.5 * (GAMMA - 1) * self.ma**2 * exp_full
        Rho0 = C0sq ** (1 / (GAMMA - 1))
        dlogRho0 = self.ma**2 * exp_full * r / C0sq
        return Utheta, dUtheta, C0sq, Rho0, dlogRho0

    def _rhs(self, r, y, forced):
        modes = self._modes
        Utheta, dUtheta, C0sq, Rho0, dlogRho0 = self.taylor_vortex(r)
        denom = self.k * r - modes * Utheta
        shear = Utheta / r + dUtheta
        p_p = Utheta**2 / (C0sq * r) + 2 * Utheta * modes / (r * denom)
        p_vr = Rho0 * (1j * self.k - 1j * modes * Utheta / r - 2j * Utheta * shear / denom)
        vr_p = 1j / (Rho0 * r) * (denom / C0sq - modes**2 / denom)
        vr_vr = -(1 / r + dlogRho0) - modes * shear / denom

        out = np.empty_like(y)
        out[:, 0] = p_p * y[:, 0] + p_vr * y[:, 1]
        out[:, 1] = vr_p * y[:, 0] + vr_vr * y[:, 1]

        if forced:
            kr = self.k * r
            phase = np.power(1j, modes)
            pin = phase * jv(modes, kr)
            vin = -1j * phase * (jv(modes - 1, kr) - jv(modes + 1, kr)) / 2
            dpin = 1j * self.k * vin
            dvin = 1j * (self.k - modes**2 / (self.k * r**2)) * pin - vin / r
            out[:, 0] += p_p * pin + p_vr * vin - dpin
            out[:, 1] += vr_p * pin + vr_vr * vin - dvin

        return out

    def _rk4_step(self, r, y, forced):
        step = -self.h
        k1 = self._rhs(r, y, forced)
        k2 = self._rhs(r + step / 3, y + step * k1 / 3, forced)
        k3 = self._rhs(r + 2 * step / 3, y + step * (-k1 / 3 + k2), forced)
        k4 = self._rhs(r + step, y + step * (k1 - k2 + k3), forced)
        return y + step * (k1 + 3 * k2 + 3 * k3 + k4) / 8

    def solve_modes(self):
        modes = self._modes
        r_outer = (self.n - 0.5) * self.h
        y_mult = np.column_stack(
            (
                hankel1(modes, self.k * r_outer),
                0.5j * (hankel1(modes + 1, self.k * r_outer) - hankel1(modes - 1, self.k * r_outer)),
            )
        )
        y_add = np.zeros_like(y_mult)

        for i in trange(self.n - 1, -1, -1):
            if i == 0:
                y_mult_1 = y_mult.copy()
                y_add_1 = y_add.copy()
            r_start = (i + 0.5) * self.h
            y_mult = self._rk4_step(r_start, y_mult, False)
            y_add = self._rk4_step(r_start, y_add, True)

        even = modes % 2 == 0
        numerator = np.empty(modes.shape, dtype=np.complex128)
        denominator = np.empty(modes.shape, dtype=np.complex128)
        numerator[even] = y_add[even, 1] + y_add_1[even, 1]
        denominator[even] = y_mult[even, 1] + y_mult_1[even, 1]
        numerator[~even] = y_add[~even, 0] + y_add_1[~even, 0]
        denominator[~even] = y_mult[~even, 0] + y_mult_1[~even, 0]
        self._am = -numerator / denominator
        return self

    def predict_farfield(self, radius=8.0, ntheta=500):
        if self._am is None:
            self.solve_modes()

        ntheta = max(int(ntheta), self._modes.size)
        theta = np.linspace(-np.pi, np.pi, ntheta, endpoint=False)
        spec = np.zeros(ntheta, dtype=np.complex128)
        spec[self._modes % ntheta] = self._am * hankel1(self._modes, self.k * radius) * np.exp(1j * self._modes * theta[0])
        ptheta = ntheta * np.fft.ifft(spec)

        return theta, ptheta
