import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar


# Interpolation methods for zero-coupon curves.
# Every interpolator works on continuously compounded zero rates r(T) and exposes :
# - y(T)       = r(T) * T = -ln P(0,T)        (integrated forward, i.e. minus the log discount factor)
# - zero(T)    = r(T)
# - forward(T) = f(0,T) = dy/dT               (instantaneous forward rate)
# Exact interpolators pass through every pillar. Outside the pillars :
# - before the first pillar, the zero rate is flat (constant forward r_1 on [0, t_1]),
# - after the last pillar, the instantaneous forward is flat (f(T) = f(t_n)).
# Reference : Hagan & West (2006), "Interpolation Methods for Curve Construction", Applied Mathematical Finance.


def _as_array(T):
    T = np.asarray(T, dtype=float)
    return T, T.ndim == 0


def _output(values, scalar):
    return float(values[0]) if scalar else values


class CurveInterpolator:
    """
    Base class of the exact interpolators.
    """
    name = "base"
    exact = True
    min_pillars = 2
    handles_origin = False      # True if the method defines the curve on [0, t_1] itself

    def __init__(self, times, zero_rates):
        t = np.asarray(times, dtype=float)
        r = np.asarray(zero_rates, dtype=float)
        # Sanity checks
        if t.ndim != 1 or r.ndim != 1 or len(t) != len(r):
            raise ValueError("times and zero_rates must be 1D arrays of the same length.")
        if len(t) < self.min_pillars:
            raise ValueError(f"{self.name} interpolation needs at least {self.min_pillars} pillars.")
        if np.any(t <= 0):
            raise ValueError("Pillar times must be positive.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Pillar times must be strictly increasing.")
        if not np.all(np.isfinite(r)):
            raise ValueError("Zero rates must be finite.")
        # Attributes
        self.t, self.r, self.yt = t, r, t * r
        self._fit()
        self.f_end = float(self._f_inner(np.array([t[-1]]))[0])

    def _fit(self):
        pass

    def _y_inner(self, T):
        raise NotImplementedError

    def _f_inner(self, T):
        raise NotImplementedError

    def _regions(self, T):
        start = 0.0 if self.handles_origin else self.t[0]
        return T < start, T > self.t[-1]

    def y(self, T):
        """ Integrated forward y(T) = r(T) T = -ln P(0,T). """
        T, scalar = _as_array(T)
        T = np.atleast_1d(T)
        if np.any(T < 0):
            raise ValueError("T must be non-negative.")
        lo, hi = self._regions(T)
        mid = ~lo & ~hi
        out = np.empty_like(T)
        out[lo] = self.r[0] * T[lo]
        out[hi] = self.yt[-1] + self.f_end * (T[hi] - self.t[-1])
        if mid.any():
            out[mid] = self._y_inner(T[mid])
        return _output(out, scalar)

    def forward(self, T):
        """ Instantaneous forward rate f(0,T). """
        T, scalar = _as_array(T)
        T = np.atleast_1d(T)
        lo, hi = self._regions(T)
        mid = ~lo & ~hi
        out = np.empty_like(T)
        out[lo] = self.r[0]
        out[hi] = self.f_end
        if mid.any():
            out[mid] = self._f_inner(T[mid])
        return _output(out, scalar)

    def zero(self, T):
        """ Continuously compounded zero rate r(T) (limit r(0) = f(0,0)). """
        T, scalar = _as_array(T)
        T = np.atleast_1d(T)
        out = np.empty_like(T)
        pos = T > 0
        out[pos] = np.atleast_1d(self.y(T[pos])) / T[pos]
        out[~pos] = np.atleast_1d(self.forward(np.zeros((~pos).sum())))
        return _output(out, scalar)

    def _segment(self, T):
        """ Index i of the segment [t_i, t_{i+1}] containing T (pillar grid without origin). """
        return np.clip(np.searchsorted(self.t, T, side='right') - 1, 0, len(self.t) - 2)


class LinearZero(CurveInterpolator):
    """
    Linear interpolation of the zero rates. Simple, local, but the forward curve is discontinuous
    at every pillar (saw-tooth shape) since f = r + T r'.
    """
    name = "linear_zero"

    def _y_inner(self, T):
        return np.interp(T, self.t, self.r) * T

    def _f_inner(self, T):
        i = self._segment(T)
        slope = (self.r[i + 1] - self.r[i]) / (self.t[i + 1] - self.t[i])
        return np.interp(T, self.t, self.r) + T * slope


class RawInterpolation(CurveInterpolator):
    """
    Linear interpolation of -ln P (log-linear discount factors) = piecewise constant forward rates.
    The market-standard "raw" method : perfectly local, forwards positive whenever the discrete forwards are,
    but the forward curve is a step function.
    """
    name = "raw"
    min_pillars = 1
    handles_origin = True

    def _fit(self):
        self._grid = np.concatenate([[0.0], self.t])
        self._ygrid = np.concatenate([[0.0], self.yt])
        self._fwd = np.diff(self._ygrid) / np.diff(self._grid)

    def _y_inner(self, T):
        return np.interp(T, self._grid, self._ygrid)

    def _f_inner(self, T):
        i = np.clip(np.searchsorted(self._grid, T, side='left') - 1, 0, len(self._fwd) - 1)
        return self._fwd[i]


class CubicSplineZero(CurveInterpolator):
    """
    Natural cubic spline on the zero rates. Smooth (C2) but non-local : one pillar moves the whole curve,
    and the forward curve may oscillate.
    """
    name = "cubic_zero"

    def _fit(self):
        self._spline = CubicSpline(self.t, self.r, bc_type='natural')

    def _y_inner(self, T):
        return self._spline(T) * T

    def _f_inner(self, T):
        return self._spline(T) + T * self._spline(T, 1)


class PchipZero(CurveInterpolator):
    """
    Monotone piecewise cubic Hermite interpolation (PCHIP) on the zero rates. C1, no overshoot of the zero rates,
    fairly local, but the forward curve is only C0 and not controlled.
    """
    name = "pchip_zero"

    def _fit(self):
        self._pchip = PchipInterpolator(self.t, self.r, extrapolate=True)

    def _y_inner(self, T):
        return self._pchip(T) * T

    def _f_inner(self, T):
        return self._pchip(T) + T * self._pchip(T, 1)


class MonotoneConvex(CurveInterpolator):
    """
    Monotone convex method of Hagan & West (2006).
    The instantaneous forward curve is continuous, reproduces the discrete forwards exactly on each segment
    (so all pillars are repriced), stays monotone/convex where the data are, and is local (a pillar only moves
    the forward curve over a few neighbouring segments).
    """
    name = "monotone_convex"
    min_pillars = 2
    handles_origin = True

    def _fit(self):
        tau = np.concatenate([[0.0], self.t])
        yt = np.concatenate([[0.0], self.yt])
        fd = np.diff(yt) / np.diff(tau)                         # discrete forwards f^d_i on (tau_{i-1}, tau_i], i = 1..n
        n = len(fd)
        f = np.empty(n + 1)                                     # instantaneous forwards at the nodes tau_0..tau_n
        for i in range(1, n):
            w = (tau[i] - tau[i - 1]) / (tau[i + 1] - tau[i - 1])
            f[i] = w * fd[i] + (1 - w) * fd[i - 1]
        f[0] = fd[0] - 0.5 * (f[1] - fd[0])
        f[n] = fd[n - 1] - 0.5 * (f[n - 1] - fd[n - 1])
        self._tau, self._ytau, self._fd, self._fnode = tau, yt, fd, f

    def _gG(self, T):
        """ Returns (i, g, G) : segment index, forward correction g(x) and its integral G(x) = int_0^x g. """
        tau, fd, f = self._tau, self._fd, self._fnode
        i = np.clip(np.searchsorted(tau, T, side='left') - 1, 0, len(fd) - 1)      # segment (tau_i, tau_{i+1}]
        h = tau[i + 1] - tau[i]
        x = (T - tau[i]) / h
        g0 = f[i] - fd[i]
        g1 = f[i + 1] - fd[i]
        # Round-off : a node forward equal to the discrete forward gives g = 0 exactly
        g0 = np.where(np.abs(g0) < 1e-12, 0.0, g0)
        g1 = np.where(np.abs(g1) < 1e-12, 0.0, g1)
        g = np.zeros_like(x)
        G = np.zeros_like(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            zero = (g0 == 0) & (g1 == 0)
            # Exactly one of g0, g1 is zero (e.g. two equal discrete forwards, frequent with rates rounded to 1bp) :
            # regions (iii) / (iv) degenerate into a jump, so the cubic of region (i) is used (continuous, exact).
            degenerate = (g0 == 0) ^ (g1 == 0)
            r1 = degenerate | ((g0 < 0) & (-0.5 * g0 <= g1) & (g1 <= -2 * g0)) | ((g0 > 0) & (-0.5 * g0 >= g1) & (g1 >= -2 * g0))
            r2 = ~r1 & (((g0 < 0) & (g1 > -2 * g0)) | ((g0 > 0) & (g1 < -2 * g0)))
            r3 = ~r1 & ~r2 & (((g0 > 0) & (g1 < 0) & (g1 > -0.5 * g0)) | ((g0 < 0) & (g1 > 0) & (g1 < -0.5 * g0)))
            r4 = ~(zero | r1 | r2 | r3)
            # Region (i) : cubic
            m = r1
            g[m] = g0[m] * (1 - 4 * x[m] + 3 * x[m]**2) + g1[m] * (-2 * x[m] + 3 * x[m]**2)
            G[m] = g0[m] * (x[m] - 2 * x[m]**2 + x[m]**3) + g1[m] * (-x[m]**2 + x[m]**3)
            # Region (ii) : flat then quadratic
            eta = (g1 + 2 * g0) / (g1 - g0)
            m = r2 & (x <= eta)
            g[m] = g0[m]; G[m] = g0[m] * x[m]
            m = r2 & (x > eta)
            g[m] = g0[m] + (g1[m] - g0[m]) * ((x[m] - eta[m]) / (1 - eta[m]))**2
            G[m] = g0[m] * x[m] + (g1[m] - g0[m]) * (x[m] - eta[m])**3 / (3 * (1 - eta[m])**2)
            # Region (iii) : quadratic then flat
            eta = 3 * g1 / (g1 - g0)
            m = r3 & (x < eta)
            g[m] = g1[m] + (g0[m] - g1[m]) * ((eta[m] - x[m]) / eta[m])**2
            G[m] = g1[m] * x[m] + (g0[m] - g1[m]) * (eta[m]**3 - (eta[m] - x[m])**3) / (3 * eta[m]**2)
            m = r3 & (x >= eta)
            g[m] = g1[m]; G[m] = g1[m] * x[m] + (g0[m] - g1[m]) * eta[m] / 3
            # Region (iv) : two quadratics
            eta = g1 / (g1 + g0)
            A = -g0 * g1 / (g0 + g1)
            m = r4 & (x <= eta)
            g[m] = A[m] + (g0[m] - A[m]) * ((eta[m] - x[m]) / eta[m])**2
            G[m] = A[m] * x[m] + (g0[m] - A[m]) * (eta[m]**3 - (eta[m] - x[m])**3) / (3 * eta[m]**2)
            m = r4 & (x > eta)
            g[m] = A[m] + (g1[m] - A[m]) * ((x[m] - eta[m]) / (1 - eta[m]))**2
            G[m] = (A[m] * x[m] + (g0[m] - A[m]) * eta[m] / 3
                    + (g1[m] - A[m]) * (x[m] - eta[m])**3 / (3 * (1 - eta[m])**2))
        return i, h, g, G

    def _y_inner(self, T):
        i, h, g, G = self._gG(T)
        return self._ytau[i] + (T - self._tau[i]) * self._fd[i] + h * G

    def _f_inner(self, T):
        i, h, g, G = self._gG(T)
        return self._fd[i] + g


class ParametricCurve(CurveInterpolator):
    """
    Base class of the parametric (least-squares fitted, non exact) curves. They are defined for every T >= 0.
    """
    exact = False
    min_pillars = 4

    def _regions(self, T):
        return np.zeros(T.shape, dtype=bool), np.zeros(T.shape, dtype=bool)

    @staticmethod
    def _loadings(T, lam):
        """ Nelson-Siegel loadings (1 - e^{-x}) / x and (1 - e^{-x}) / x - e^{-x}, with x = T / lam. """
        x = np.asarray(T, dtype=float) / lam
        with np.errstate(divide='ignore', invalid='ignore'):
            l1 = np.where(x > 1e-8, -np.expm1(-x) / x, 1.0 - 0.5 * x)
        return l1, l1 - np.exp(-x)

    def _design(self, T, lams):
        cols = [np.ones_like(np.asarray(T, dtype=float))]
        for k, lam in enumerate(lams):
            l1, l2 = self._loadings(T, lam)
            cols += [l1, l2] if k == 0 else [l2]
        return np.column_stack(cols)

    def _solve_betas(self, lams):
        X = self._design(self.t, lams)
        beta, *_ = np.linalg.lstsq(X, self.r, rcond=None)
        return beta, float(np.sum((X @ beta - self.r) ** 2))

    def _y_inner(self, T):
        return (self._design(T, self.lams) @ self.beta) * T

    def _f_inner(self, T):
        T = np.asarray(T, dtype=float)
        x = T / self.lams[0]
        f = self.beta[0] + self.beta[1] * np.exp(-x) + self.beta[2] * x * np.exp(-x)
        if len(self.lams) > 1:
            x2 = T / self.lams[1]
            f = f + self.beta[3] * x2 * np.exp(-x2)
        return f

    @property
    def params(self):
        names = ["beta0", "beta1", "beta2", "beta3"][:len(self.beta)]
        lam_names = ["lambda1", "lambda2"][:len(self.lams)]
        return {**dict(zip(names, map(float, self.beta))), **dict(zip(lam_names, map(float, self.lams)))}


class NelsonSiegel(ParametricCurve):
    """
    Nelson-Siegel (1987) : r(T) = b0 + b1 L1(T/l) + b2 L2(T/l). 4 parameters : level, slope, curvature, decay.
    The betas are linear given lambda, so lambda is found by a 1D search and the betas by least squares.
    """
    name = "nelson_siegel"

    def _fit(self):
        grid = np.linspace(0.05, 10.0, 200)
        best = min(grid, key=lambda l: self._solve_betas([l])[1])
        res = minimize_scalar(lambda l: self._solve_betas([l])[1], bounds=(max(0.02, best / 2), best * 2), method='bounded')
        self.lams = [float(res.x)]
        self.beta, self.sse = self._solve_betas(self.lams)


class Svensson(ParametricCurve):
    """
    Svensson (1994) : Nelson-Siegel plus a second curvature term b3 L2(T/l2). 6 parameters.
    """
    name = "svensson"
    min_pillars = 6

    def _fit(self):
        grid = np.linspace(0.1, 10.0, 25)
        candidates = [(l1, l2) for l1 in grid for l2 in grid if l2 > l1 * 1.2]
        start = min(candidates, key=lambda l: self._solve_betas(l)[1])
        res = minimize(lambda l: self._solve_betas(np.abs(l) + 1e-3)[1], start, method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-14, 'maxiter': 2000})
        self.lams = list(np.abs(res.x) + 1e-3)
        self.beta, self.sse = self._solve_betas(self.lams)


INTERPOLATORS = {cls.name: cls for cls in
                 (LinearZero, RawInterpolation, CubicSplineZero, PchipZero, MonotoneConvex, NelsonSiegel, Svensson)}
INTERPOLATORS["log_linear_discount"] = RawInterpolation


def make_interpolator(method, times, zero_rates):
    """
    Builds the interpolator called method on the pillars (times, continuously compounded zero rates).
    """
    if method not in INTERPOLATORS:
        raise ValueError(f"Unknown interpolation method '{method}'. Available : {sorted(INTERPOLATORS)}")
    return INTERPOLATORS[method](times, zero_rates)
