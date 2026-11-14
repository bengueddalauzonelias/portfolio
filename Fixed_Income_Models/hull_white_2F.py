import copy
import numpy as np
from scipy.optimize import least_squares

try:
    from .rate_pricing import bachelier
    from .cms import swap_schedule, forward_swap
except ImportError:
    from rate_pricing import bachelier
    from cms import swap_schedule, forward_swap


# Two-factor Gaussian short-rate model G2++ (Brigo & Mercurio 2006, Ch. 4), equivalent to the two-factor
# Hull-White model (Hull & White 1994), fitted exactly to the initial YieldCurve :
#     r(t) = x(t) + y(t) + phi(t),   dx = -a x dt + sigma dW1,   dy = -b y dt + eta dW2,   d<W1, W2> = rho dt,
# x(0) = y(0) = 0. With V(t,T) the variance of int_t^T (x + y) :
# - zero-coupon bonds P(t,T) = P(0,T)/P(0,t) exp(1/2 [V(t,T) - V(0,T) + V(0,t)] - B_a(t,T) x - B_b(t,T) y),
#   B_c(t,T) = (1 - e^{-c(T-t)}) / c, and int_0^t phi = -ln P(0,t) + V(0,t) / 2 ;
# - European swaptions with the frozen-gradient approximation : under the annuity measure the swap rate is a
#   driftless Gaussian with normal variance int_0^T0 |grad S . vol|^2, in closed form (same formula gives the
#   terminal correlation of two swap rates, which drives CMS spread options) ;
# - exact joint simulation of (x, y, int x, int y) ;
# - calibration of (a, b, sigma, eta, rho) to ATM normal swaption vols and to swap rate correlation targets.
# Same simulation interface as HullWhite : simulate(times, ...) -> (states, D), bond_price(t, T, state) with
# state = states[k] of shape (2, n_paths).


def _E(c, h):
    """ int_0^h e^{-c s} ds. """
    return (1 - np.exp(-c * h)) / c


class G2pp:
    """
    G2++ / two-factor Hull-White model on a YieldCurve.
    """
    PARAMS = {"a": (1e-3, 1.5), "b": (1e-3, 1.5), "sigma": (1e-4, 0.05), "eta": (1e-4, 0.05), "rho": (-0.999, 0.999)}
    NAMES = ("a", "b", "sigma", "eta", "rho")
    model = "g2pp"

    def __init__(self, curve, a=0.05, b=0.5, sigma=0.008, eta=0.006, rho=-0.5):
        """
        Init method.
        - curve      : YieldCurve, fitted exactly by the model
        - a, b       : mean reversion speeds of the two factors
        - sigma, eta : normal volatilities of the factors (rate units, 0.008 = 80bp per year)
        - rho        : correlation of the two Brownian motions
        """
        if a <= 0 or b <= 0: raise ValueError("a and b must be positive.")
        if sigma <= 0 or eta <= 0: raise ValueError("sigma and eta must be positive.")
        if not -1 < rho < 1: raise ValueError("rho must be in (-1, 1).")
        self.curve = curve
        self.a, self.b, self.sigma, self.eta, self.rho = map(float, (a, b, sigma, eta, rho))

    @property
    def params(self):
        return {n: getattr(self, n) for n in self.NAMES}

    def with_curve(self, curve):
        model = copy.copy(self)
        model.curve = curve
        return model

    def with_params(self, **params):
        model = copy.copy(self)
        for name, value in params.items():
            setattr(model, name, float(value))
        return model

    # ------------------------------------------------------------------
    # Zero-coupon bonds
    # ------------------------------------------------------------------
    @staticmethod
    def B(c, t, T):
        return (1.0 - np.exp(-c * (np.asarray(T) - np.asarray(t)))) / c

    def V(self, t, T):
        """ Variance of int_t^T (x(u) + y(u)) du given F_t. """
        a, b, s, e, rho = self.a, self.b, self.sigma, self.eta, self.rho
        tau = np.asarray(T, dtype=float) - np.asarray(t, dtype=float)
        term = lambda c: tau + 2 / c * np.exp(-c * tau) - 1 / (2 * c) * np.exp(-2 * c * tau) - 3 / (2 * c)
        cross = tau + np.expm1(-a * tau) / a + np.expm1(-b * tau) / b - np.expm1(-(a + b) * tau) / (a + b)
        return s**2 / a**2 * term(a) + e**2 / b**2 * term(b) + 2 * rho * s * e / (a * b) * cross

    def bond_price(self, t, T, state):
        """ P(t,T) given the state (x(t), y(t)) of shape (2, ...) (vectorised in T and the paths). """
        x, y = state[0], state[1]
        convexity = 0.5 * (self.V(t, T) - self.V(0.0, T) + self.V(0.0, t))
        return (self.curve.discount(T) / self.curve.discount(t)
                * np.exp(convexity - self.B(self.a, t, T) * x - self.B(self.b, t, T) * y))

    # ------------------------------------------------------------------
    # Swap rate dynamics (frozen gradient)
    # ------------------------------------------------------------------
    def _loadings(self, expiry, tenor, frequency=1):
        """
        Constants (K_a, K_b) such that, under the annuity measure and freezing the gradient of S at time 0,
            dS(t) = sigma K_a e^{-a(T0-t)} dW1 + eta K_b e^{-b(T0-t)} dW2,   t <= T0.
        With c_k = d S / d ln P(T_k) (k = 0 : start), K_c = (1/c) sum_k c_k e^{-c(T_k - T0)}.
        """
        times, accruals = swap_schedule(expiry, tenor, frequency)
        S0, A0 = forward_swap(self.curve, expiry, tenor, frequency)
        P = self.curve.discount(times)
        c = -accruals * P * S0 / A0
        c[-1] -= P[-1] / A0
        c = np.concatenate([[self.curve.discount(expiry) / A0], c])
        dates = np.concatenate([[expiry], times]) - expiry
        return (np.sum(c * np.exp(-self.a * dates)) / self.a, np.sum(c * np.exp(-self.b * dates)) / self.b)

    def _covariance(self, expiry, load1, load2):
        """ int_0^T0 of the instantaneous covariance of two swap rates with loadings load1, load2. """
        a, b, s, e, rho = self.a, self.b, self.sigma, self.eta, self.rho
        (ka1, kb1), (ka2, kb2) = load1, load2
        return (s**2 * ka1 * ka2 * _E(2 * a, expiry) + e**2 * kb1 * kb2 * _E(2 * b, expiry)
                + rho * s * e * (ka1 * kb2 + kb1 * ka2) * _E(a + b, expiry))

    def swaption_normal_vol(self, expiry, tenor, frequency=1):
        """ Normal volatility of the swap rate expiry x tenor (frozen-gradient approximation, flat smile). """
        load = self._loadings(expiry, tenor, frequency)
        return float(np.sqrt(self._covariance(expiry, load, load) / expiry))

    def swap_rate_correlation(self, expiry, tenor1, tenor2, frequency=1):
        """ Terminal correlation of the swap rates expiry x tenor1 and expiry x tenor2 (frozen gradients). """
        l1, l2 = self._loadings(expiry, tenor1, frequency), self._loadings(expiry, tenor2, frequency)
        return float(self._covariance(expiry, l1, l2)
                     / np.sqrt(self._covariance(expiry, l1, l1) * self._covariance(expiry, l2, l2)))

    def european_swaption_value(self, swaption):
        """ Bachelier price with the model's normal swap rate volatility (approximation, see _loadings). """
        if swaption.exercise != 'european':
            raise ValueError("Only European swaptions are priced in closed form : use the Monte Carlo.")
        S0, A0 = forward_swap(self.curve, swaption.expiry, swaption.tenor, swaption.fixed_frequency)
        vol = self.swaption_normal_vol(swaption.expiry, swaption.tenor, swaption.fixed_frequency)
        return float(swaption.notional * A0 * bachelier(S0, swaption.strike, vol, swaption.expiry, swaption.payer))

    def price(self, product):
        if product.kind == "swaption":
            return self.european_swaption_value(product)
        raise ValueError(f"Unknown product kind '{product.kind}'.")

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def _step_covariance(self, h):
        """ Covariance of the Gaussian increments of (x, y, int x, int y) over a step of length h. """
        a, b, s, e, rho = self.a, self.b, self.sigma, self.eta, self.rho
        factors = [(a, s, 0), (b, e, 1)]
        kernels = [(f, "level") for f in factors] + [(f, "integral") for f in factors]
        C = np.empty((4, 4))
        for i, ((c1, v1, w1), k1) in enumerate(kernels):
            for j, ((c2, v2, w2), k2) in enumerate(kernels):
                corr = 1.0 if w1 == w2 else rho
                if k1 == "level" and k2 == "level":
                    integral = _E(c1 + c2, h)
                elif k1 == "level":
                    integral = (_E(c1, h) - _E(c1 + c2, h)) / c2
                elif k2 == "level":
                    integral = (_E(c2, h) - _E(c1 + c2, h)) / c1
                else:
                    integral = (h - _E(c1, h) - _E(c2, h) + _E(c1 + c2, h)) / (c1 * c2)
                C[i, j] = corr * v1 * v2 * integral
        return C

    def simulate(self, times, n_paths=50_000, seed=None, antithetic=True):
        """
        Exact joint simulation of (x, y) and int_0^t (x + y) on the given dates.
        Returns (states, D) : states of shape (len(times), 2, n_paths) and the bank account discount factors
        D(0,t) = exp(-int_0^t r) of shape (len(times), n_paths).
        """
        times = np.asarray(times, dtype=float)
        if np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise ValueError("times must be positive and strictly increasing.")
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even with antithetic variates.")
        rng = np.random.default_rng(seed)
        a, b = self.a, self.b
        x, y, Ix, Iy = (np.zeros(n_paths) for _ in range(4))
        states = np.empty((len(times), 2, n_paths))
        integral = np.empty((len(times), n_paths))
        previous = 0.0
        for k, t in enumerate(times):
            h = t - previous
            L = np.linalg.cholesky(self._step_covariance(h) + 1e-20 * np.eye(4))
            m = n_paths // 2 if antithetic else n_paths
            Z = rng.standard_normal((4, m))
            if antithetic:
                Z = np.concatenate([Z, -Z], axis=1)
            eps = L @ Z
            Ix = Ix + x * _E(a, h) + eps[2]
            Iy = Iy + y * _E(b, h) + eps[3]
            x = x * np.exp(-a * h) + eps[0]
            y = y * np.exp(-b * h) + eps[1]
            states[k] = x, y
            integral[k] = Ix + Iy
            previous = t
        phi_integral = -np.log(self.curve.discount(times)) + 0.5 * self.V(0.0, times)
        return states, np.exp(-integral - phi_integral[:, None])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, vol_grid, expiries=None, tenors=None, correlation_targets=(), correlation_weight=1.0,
                  fixed=None, bounds=None):
        """
        Calibrates the parameters to the ATM normal vols of vol_grid (a SwaptionVolGrid) on expiries x tenors and,
        optionally, to correlation targets [(expiry, tenor1, tenor2, target correlation), ...].
        Residuals : vol errors in bp and correlation_weight * correlation errors in percentage points.
        `fixed` = {"a": value, ...} keeps parameters constant.
        Returns a dict with the parameters, the RMSE in bp of normal vol and the model correlations.
        """
        expiries = vol_grid.expiries if expiries is None else np.asarray(expiries, dtype=float)
        tenors = vol_grid.tenors if tenors is None else np.asarray(tenors, dtype=float)
        points = [(e, t) for e in expiries for t in tenors]
        targets = np.array([vol_grid.vol(e, t) for e, t in points])
        fixed = dict(fixed or {})
        bounds = {**self.PARAMS, **(bounds or {})}
        names = [n for n in self.NAMES if n not in fixed]

        def build(values):
            return self.with_params(**{**fixed, **dict(zip(names, values))})

        def residuals(values):
            model = build(values)
            vols = np.array([model.swaption_normal_vol(e, t) for e, t in points])
            out = list(1e4 * (vols - targets))
            for e, t1, t2, target in correlation_targets:
                out.append(correlation_weight * 100 * (model.swap_rate_correlation(e, t1, t2) - target))
            return np.array(out)

        x0 = [getattr(self, n) for n in names]
        res = least_squares(residuals, x0, bounds=([bounds[n][0] for n in names], [bounds[n][1] for n in names]),
                            x_scale='jac')
        for name, value in {**fixed, **dict(zip(names, res.x))}.items():
            setattr(self, name, float(value))
        vol_errors = res.fun[:len(points)]
        return {**self.params, "RMSE (bp of normal vol)": float(np.sqrt(np.mean(vol_errors**2))),
                "max error (bp)": float(np.max(np.abs(vol_errors))),
                **{f"corr {e:g}y {t1:g}y/{t2:g}y": self.swap_rate_correlation(e, t1, t2)
                   for e, t1, t2, _ in correlation_targets}}

    def __repr__(self):
        return (f"G2pp(a={self.a:.4f}, b={self.b:.4f}, sigma={1e4 * self.sigma:.1f}bp, eta={1e4 * self.eta:.1f}bp, "
                f"rho={self.rho:+.3f})")
