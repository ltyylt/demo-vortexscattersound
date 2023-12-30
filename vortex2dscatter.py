# Disclaimer:
# The alleged Lippmann-Schwinger form has nothing to do with these 2 guys.
# The whole formulation was proposed by this specific paper and the name serves as a short alias for this long story.

import numpy as np
from scipy import fft
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import hankel1, j0, j1, jv
from typing import override

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


GAMMA = 1.4


def helmholtz2d_truncated_greenf(L, k, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (1 + 0.5j * np.pi * L * s * hankel1(0, L * k) * j1(L * s) - 0.5j * np.pi * L * k * hankel1(1, L * k) * j0(L * s)) / (s**2 - k**2)
    out[np.isclose(s, k)] = 0.25j * np.pi * L**2 * (j0(L * k) * hankel1(0, L * k) + j1(L * k) * hankel1(1, L * k))
    return out


class IsenVortex:
    def __init__(self, strength, radius=1.0, center=(0.0, 0.0)):
        self.strength = float(strength)
        self.radius = float(radius)
        self.center = tuple(center)

    def _shifted(self, x, y):
        return x - self.center[0], y - self.center[1]

    def _profile(self, x, y):
        xs, ys = self._shifted(x, y)
        r2 = (xs**2 + ys**2) / self.radius**2
        return self.strength / self.radius * np.exp((1 - r2) / 2)

    def velocity(self, x, y):
        xs, ys = self._shifted(x, y)
        a = self._profile(x, y)
        return -a * ys, a * xs

    def dudx(self, x, y):
        xs, ys = self._shifted(x, y)
        return self._profile(x, y) * xs * ys / self.radius**2

    def dudy(self, x, y):
        _, ys = self._shifted(x, y)
        return self._profile(x, y) * (ys**2 - self.radius**2) / self.radius**2

    def dvdx(self, x, y):
        xs, _ = self._shifted(x, y)
        return self._profile(x, y) * (self.radius**2 - xs**2) / self.radius**2

    def dvdy(self, x, y):
        xs, ys = self._shifted(x, y)
        return -self._profile(x, y) * xs * ys / self.radius**2

    def c0sq(self, x, y):
        xs, ys = self._shifted(x, y)
        r2 = (xs**2 + ys**2) / self.radius**2
        base = 1 - 0.5 * (GAMMA - 1) * self.strength**2 * np.exp(1 - r2)
        if np.any(base <= 0):
            raise ValueError("isentropic vortex sound speed became non-positive")
        return base


class VortexScatter2D:
    def __init__(self, wavelength):
        self.wavelength = float(wavelength)
        self.k = 2 * np.pi / self.wavelength

    def init_mesh(self, domain_half_width, mesh_size):
        self.domain_half_width = float(domain_half_width)
        self.mesh_size = int(mesh_size)

        length = 2 * self.domain_half_width
        x = np.linspace(-length / 2, length / 2, self.mesh_size, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing="ij")
        self.h = x[1] - x[0]
        self.h2 = self.h**2

        k1 = fft.fftfreq(self.mesh_size, self.h) * 2 * np.pi
        self._kx, self._ky = np.meshgrid(k1, k1, indexing="ij")

        pad_n = 3 * self.mesh_size
        kp = fft.fftfreq(pad_n, self.h) * 2 * np.pi
        kpx, kpy = np.meshgrid(kp, kp, indexing="ij")
        green_hat = helmholtz2d_truncated_greenf(1.5 * length, self.k, np.sqrt(kpx**2 + kpy**2))
        self._green_toeplitz = self.toeplitzize_greenhat(green_hat, self.mesh_size)

        z = np.zeros_like(self.x, dtype=np.complex128)
        self.incident = np.exp(1j * self.k * self.x)
        self.U = np.zeros_like(self.x)
        self.V = np.zeros_like(self.x)
        self.DUDX = np.zeros_like(self.x)
        self.DUDY = np.zeros_like(self.x)
        self.DVDX = np.zeros_like(self.x)
        self.DVDY = np.zeros_like(self.x)
        self.C2inv = np.ones_like(self.x)
        self.sol_p = np.zeros_like(z)
        self.sol_vx = np.zeros_like(z)
        self.sol_vy = np.zeros_like(z)
        self._kernel_p = np.zeros_like(z)
        self.residual = []
        return self

    def set_incident(self, incident):
        self.incident = np.asarray(incident, dtype=np.complex128)
        return self

    def set_incident_from_kernel(self, kernel):
        self.incident = self._green_convolve(kernel)
        return self

    def set_velocity(self, uu, vv, dudx=None, dudy=None, dvdx=None, dvdy=None):
        def as_array(obj):
            if callable(obj):
                return np.asarray(obj(self.x, self.y))
            return np.asarray(obj)

        self.U = as_array(uu)
        self.V = as_array(vv)
        self.DUDX = as_array(dudx) if dudx is not None else self._dx(self.U).real
        self.DUDY = as_array(dudy) if dudy is not None else self._dy(self.U).real
        self.DVDX = as_array(dvdx) if dvdx is not None else self._dx(self.V).real
        self.DVDY = as_array(dvdy) if dvdy is not None else self._dy(self.V).real
        return self

    def set_c2inv(self, c2inv):
        self.C2inv = np.asarray(c2inv(self.x, self.y) if callable(c2inv) else c2inv)
        return self

    def add_taylor_vortex(self, mach, radius=1.0, center=(0.0, 0.0)):
        vort = IsenVortex(-mach, radius, center)
        U, V = vort.velocity(self.x, self.y)
        self.U += U
        self.V += V
        self.DUDX += vort.dudx(self.x, self.y)
        self.DUDY += vort.dudy(self.x, self.y)
        self.DVDX += vort.dvdx(self.x, self.y)
        self.DVDY += vort.dvdy(self.x, self.y)
        self.C2inv = 1 / vort.c0sq(self.x, self.y)
        return self

    def complete(self):
        z = np.zeros_like(self.incident)
        self.sol_p = z.copy()
        self.sol_vx = z.copy()
        self.sol_vy = z.copy()
        self._kernel_p = z.copy()
        self.residual = []
        return self

    def _fftd(self, a, multiplier):
        return fft.ifft2(multiplier * fft.fft2(a, workers=-1), workers=-1)

    def _dx(self, a):
        return self._fftd(a, 1j * self._kx)

    def _dy(self, a):
        return self._fftd(a, 1j * self._ky)

    def _dxx(self, a):
        return self._fftd(a, -(self._kx**2))

    def _dxy(self, a):
        return self._fftd(a, -self._kx * self._ky)

    def _dyy(self, a):
        return self._fftd(a, -(self._ky**2))

    def _dxxx(self, a):
        return self._fftd(a, -1j * self._kx**3)

    def _dxyy(self, a):
        return self._fftd(a, -1j * self._kx * self._ky**2)

    def _dyxx(self, a):
        return self._fftd(a, -1j * self._ky * self._kx**2)

    def _dyyy(self, a):
        return self._fftd(a, -1j * self._ky**3)

    @staticmethod
    def toeplitzize_greenhat(green_hat, n):
        green = fft.ifftshift(fft.ifft2(green_hat, workers=-1))
        c = (green_hat.shape[0] + 1) // 2
        green = green[c - n : c + n, c - n : c + n]
        return fft.fft2(green, workers=-1)

    def _green_convolve(self, source):
        n = self.mesh_size
        out = fft.ifft2(self._green_toeplitz * fft.fft2(source, s=(2 * n, 2 * n), workers=-1), workers=-1)
        return out[n:, n:]

    def _c1v(self, p, vx, vy):
        c1x = self._dx(2 * self.U * vx) + self._dy(self.V * vx + self.U * vy) + (-1j / self.k) * (self._dxx(2 * self.U * p) - self._dx(2 * self.DUDX * p) + self._dxy(self.V * p) - self._dy(self.DVDX * p) + self._dyy(self.U * p) - self._dy(self.DUDY * p))
        c1y = self._dy(2 * self.V * vy) + self._dx(self.V * vx + self.U * vy) + (-1j / self.k) * (self._dyy(2 * self.V * p) - self._dy(2 * self.DVDY * p) + self._dxx(self.V * p) - self._dx(self.DVDX * p) + self._dxy(self.U * p) - self._dx(self.DUDY * p))
        return c1x, c1y

    def _c2p(self, p):
        c2x = self._dx(p * self.C2inv * self.U * self.U) + self._dy(p * self.C2inv * self.U * self.V)
        c2y = self._dx(p * self.C2inv * self.U * self.V) + self._dy(p * self.C2inv * self.V * self.V)
        return c2x, c2y

    def _kernel(self, p, vx, vy):
        p1 = self._dxx(self.U * vx) + self._dxy(self.U * vy + self.V * vx) + self._dyy(self.V * vy) + (-1j / self.k) * (
            self._dxxx(self.U * p) - self._dxx(self.DUDX * p) + self._dxyy(self.U * p) - self._dxy(self.DUDY * p) + self._dyxx(self.V * p) - self._dxy(self.DVDX * p) + self._dyyy(self.V * p) - self._dyy(self.DVDY * p)
            )
        p1 *= 2
        p2 = self._dxx(self.U**2 * self.C2inv * p) + 2 * self._dxy(self.U * self.V * self.C2inv * p) + self._dyy(self.V**2 * self.C2inv * p)
        sos = self.k**2 * p * (self.C2inv - 1)
        return p1 + p2 + sos

    def born_step(self):
        p = self.p_total
        self._kernel_p = self._kernel(p, self.sol_vx, self.sol_vy)
        c1x, c1y = self._c1v(p, self.sol_vx, self.sol_vy)
        c2x, c2y = self._c2p(p)
        self.sol_vx = (c1x + c2x) / (1j * self.k) - self.U * p * self.C2inv
        self.sol_vy = (c1y + c2y) / (1j * self.k) - self.V * p * self.C2inv
        last = self.sol_p
        self.sol_p = self._green_convolve(self._kernel_p)
        self.residual.append(np.linalg.norm(self.sol_p - last) * self.h)
        return self

    def _ls_op(self, flat):
        n = self.mesh_size
        p, vx, vy = flat.reshape(3, n, n)
        kern = self._kernel(p, vx, vy)
        pterm = p - self._green_convolve(kern)
        c1x, c1y = self._c1v(p, vx, vy)
        c2x, c2y = self._c2p(p)
        vx_rhs = (c1x + c2x) / (1j * self.k) - self.U * p * self.C2inv
        vy_rhs = (c1y + c2y) / (1j * self.k) - self.V * p * self.C2inv
        return np.stack((pterm, vx - vx_rhs, vy - vy_rhs)).reshape(-1)

    def solve(self, maxiter=30, tol=1e-8):
        n = self.mesh_size
        zero = np.zeros_like(self.incident)
        rhs = np.stack((self.incident, zero, zero)).reshape(-1)
        x0 = rhs.copy()
        op = LinearOperator((rhs.size, rhs.size), matvec=self._ls_op, dtype=np.complex128)
        bar = tqdm(total=maxiter) if tqdm is not None else None
        last = [self.incident.copy()]

        def callback(x):
            p = x.reshape(3, n, n)[0]
            self.residual.append(np.linalg.norm(p - last[0]) * self.h)
            last[0] = p.copy()
            if bar is not None:
                bar.update()

        flat, info = gmres(op, rhs, x0=x0, maxiter=maxiter, rtol=tol, callback=callback, callback_type="x")
        if bar is not None:
            bar.close()
        p, vx, vy = flat.reshape(3, n, n)
        self.sol_p = p - self.incident
        self.sol_vx = vx
        self.sol_vy = vy
        self._kernel_p = self._kernel(p, vx, vy)
        return self

    def directivity(self, radius=8.0, ntheta=361, low=-np.pi, high=np.pi):
        theta = np.linspace(low, high, ntheta)
        rr = np.sqrt(self.x**2 + self.y**2)

        if radius > rr.max():
            phi = np.arctan2(self.y, self.x)
            mmax = int(np.ceil(self.k * rr.max() + 40))
            modes = np.arange(-mmax, mmax + 1)
            coeff = np.empty(modes.shape, dtype=np.complex128)
            for i, m in enumerate(modes):
                source_m = jv(m, self.k * rr) * np.exp(-1j * m * phi)
                coeff[i] = 0.25j * hankel1(m, self.k * radius) * np.sum(self._kernel_p * source_m) * self.h2
            return theta, np.exp(1j * theta[:, None] * modes[None, :]) @ coeff

        values = np.empty(theta.shape, dtype=np.complex128)
        z = self.k * self.h
        d0 = np.log(0.32478097 * z**3 - 0.6574281 * z**2 + 0.7298274 * z + 0.0197115)
        for i in range(0, ntheta, 32):
            th = theta[i : i + 32]
            tx = radius * np.cos(th)[:, None, None]
            ty = radius * np.sin(th)[:, None, None]
            r = np.sqrt((tx - self.x) ** 2 + (ty - self.y) ** 2)
            g = 0.25j * hankel1(0, self.k * r)
            g[np.isclose(r, 0)] = 0.25j * (1 + 1j * d0)
            values[i : i + th.size] = np.sum(g * self._kernel_p, axis=(1, 2)) * self.h2
        return theta, values

    @property
    def p_total(self):
        return self.incident + self.sol_p


class Mach1Scatter2D(VortexScatter2D):
    @override
    def _kernel(self, p, vx=None, vy=None):
        return (-2j / self.k) * (self._dxxx(self.U * p) - self._dxx(self.DUDX * p) + self._dxyy(self.U * p) - self._dxy(self.DUDY * p) + self._dyxx(self.V * p) - self._dxy(self.DVDX * p) + self._dyyy(self.V * p) - self._dyy(self.DVDY * p))

    @override
    def born_step(self):
        p = self.p_total
        self._kernel_p = self._kernel(p)
        last = self.sol_p
        self.sol_p = self._green_convolve(self._kernel_p)
        self.residual.append(np.linalg.norm(self.sol_p - last) * self.h)
        return self

    def rytov_step(self):
        p = self.p_total
        self._kernel_p = self._kernel(p)
        last = self.sol_p
        update = self._green_convolve(self._kernel_p)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.sol_p = p * np.exp(update / p) - self.incident
        self.sol_p[~np.isfinite(self.sol_p)] = 0
        self._kernel_p = self._kernel(self.p_total)
        self.residual.append(np.linalg.norm(self.sol_p - last) * self.h)
        return self

    @override
    def _ls_op(self, flat):
        n = self.mesh_size
        p = flat.reshape(n, n)
        return (p - self._green_convolve(self._kernel(p))).reshape(-1)

    @override
    def solve(self, maxiter=30, tol=1e-8):
        n = self.mesh_size
        rhs = self.incident.reshape(-1)
        x0 = rhs.copy()
        op = LinearOperator((rhs.size, rhs.size), matvec=self._ls_op, dtype=np.complex128)
        bar = tqdm(total=maxiter) if tqdm is not None else None
        last = [self.incident.copy()]

        def callback(x):
            p = x.reshape(n, n)
            self.residual.append(np.linalg.norm(p - last[0]) * self.h)
            last[0] = p.copy()
            if bar is not None:
                bar.update()

        flat, info = gmres(op, rhs, x0=x0, maxiter=maxiter, rtol=tol, callback=callback, callback_type="x")
        if bar is not None:
            bar.close()
        p = flat.reshape(n, n)
        self.sol_p = p - self.incident
        self._kernel_p = self._kernel(p)
        return self


class Mach1HighFreqApprox(Mach1Scatter2D):
    @override
    def init_mesh(self, domain_half_width, mesh_size):
        super().init_mesh(domain_half_width, mesh_size)
        pad_n = 3 * self.mesh_size
        kp = fft.fftfreq(pad_n, self.h) * 2 * np.pi
        kpx, kpy = np.meshgrid(kp, kp, indexing="ij")
        shifted = np.sqrt((kpx + self.k) ** 2 + kpy**2)
        green_hat = helmholtz2d_truncated_greenf(3 * self.domain_half_width, self.k, shifted)
        self._green_toeplitz = self.toeplitzize_greenhat(green_hat, self.mesh_size)
        self.incident = np.ones_like(self.x, dtype=np.complex128)
        return self

    @override
    def _kernel(self, p, vx=None, vy=None):
        return -2 * self.k**2 * self.U * p

    @property
    def envelope_scattered(self):
        return self.sol_p - 1

    @property
    @override
    def p_total(self):
        return self.sol_p

    @override
    def complete(self):
        super().complete()
        self.sol_p = np.ones_like(self.incident)
        return self

    @override
    def solve(self, maxiter=30, tol=1e-8):
        n = self.mesh_size
        rhs = np.ones_like(self.incident).reshape(-1)
        x0 = rhs.copy()
        op = LinearOperator((rhs.size, rhs.size), matvec=self._ls_op, dtype=np.complex128)
        bar = tqdm(total=maxiter) if tqdm is not None else None
        last = [np.ones_like(self.incident)]

        def callback(x):
            p = x.reshape(n, n)
            self.residual.append(np.linalg.norm(p - last[0]) * self.h)
            last[0] = p.copy()
            if bar is not None:
                bar.update()

        flat, info = gmres(op, rhs, x0=x0, maxiter=maxiter, rtol=tol, callback=callback, callback_type="x")
        if bar is not None:
            bar.close()
        self.sol_p = flat.reshape(n, n)
        self._kernel_p = self._kernel(self.sol_p)
        return self


class Mach1HighFreqApproxO1(Mach1HighFreqApprox):
    @override
    def _kernel(self, p, vx=None, vy=None):
        return -2 * self.k**2 * self.U * p + 2j * self.k * (self._dy(p * self.V) - p * self.DVDY + 3 * self._dx(p * self.U) - 2 * p * self.DUDX)
