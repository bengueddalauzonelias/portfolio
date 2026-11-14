import copy
import numpy as np
from scipy.optimize import least_squares

try:
    from .cms import mc_swaption_vols
except ImportError:
    from cms import mc_swaption_vols


# Multi-factor quasi-Gaussian (Cheyette) model with local and stochastic volatility (Cheyette 1992 ; Andersen &
# Piterbarg 2010, Vol. II, Ch. 13-14). HJM model with separable volatility sigma_f(t,T)_i = sigma_r,i(t) e^{-kappa_i (T-t)} :
# with d factors, a diagonal mean reversion kappa and G_i(t,T) = (1 - e^{-kappa_i (T-t)}) / kappa_i,
#     r(t)   = f(0,t) + sum_i x_i(t),
#     dx     = (Y(t) 1 - kappa x) dt + sigma_r(t)^T dW,                 x(0) = 0,
#     dY     = (sigma_r^T sigma_r - kappa Y - Y kappa) dt,              Y(0) = 0   (d x d, path dependent),
#     P(t,T) = P(0,T)/P(0,t) exp(-G(t,T)^T x(t) - 1/2 G(t,T)^T Y(t) G(t,T)).
# The volatility is the Gaussian one (factor vols sigma_i, correlation corr) times a scalar multiplier :
#     m(t) = sqrt(z(t)) max(1 + beta (x_1 + ... + x_d), floor),
# - LOCAL volatility : beta (skew of the swaption smile, beta > 0 : vol increases with the rates) ;
# - STOCHASTIC volatility : dz = theta (1 - z) dt + epsilon sqrt(z) dZ, z(0) = 1, Z independent of W
#   (curvature of the smile).
# beta = epsilon = 0 is the d-factor Gaussian model : with d = 2 it is G2++ (QuasiGaussian.from_g2).
# Monte Carlo by Euler (full truncation for z, trapezoidal integral of the short rate) ; swaption smiles by
# Monte Carlo (cms.mc_swaption_vols), so the smile calibration uses common random numbers.
# Same simulation interface as HullWhite / G2pp : simulate(times, ...) -> (states, D), bond_price(t, T, state)
# with state = states[k] of shape (d + d^2, n_paths) = (x, Y flattened).


class QuasiGaussian:
    """
    d-factor quasi-Gaussian (Cheyette) model on a YieldCurve, with local and stochastic volatility.
    """
    model = "quasi_gaussian"

    def __init__(self, curve, kappa, sigma, corr=None, beta=0.0, epsilon=0.0, theta=1.0, floor=0.05, dt=1 / 24):
        """
        Init method.
        - curve   : YieldCurve, fitted exactly by the model
        - kappa   : mean reversions of the d factors (distinct, positive)
        - sigma   : normal volatilities of the d factors (rate units)
        - corr    : d x d correlation matrix of the factors (identity by default)
        - beta    : local volatility skew of the multiplier
        - epsilon : volatility of the variance z (0 : no stochastic volatility), theta its mean reversion
        - floor   : lower bound of the local volatility multiplier
        - dt      : maximal Euler step (years)
        """
        self.curve = curve
        self.kappa = np.asarray(kappa, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.d = len(self.kappa)
        self.corr = np.eye(self.d) if corr is None else np.asarray(corr, dtype=float)
        if self.sigma.shape != (self.d,) or self.corr.shape != (self.d, self.d):
            raise ValueError("kappa, sigma and corr must have consistent dimensions.")
        if np.any(self.kappa <= 0) or np.any(self.sigma <= 0):
            raise ValueError("kappa and sigma must be positive.")
        self.beta, self.epsilon, self.theta, self.floor, self.dt = map(float, (beta, epsilon, theta, floor, dt))

    @classmethod
    def from_g2(cls, g2, **kwargs):
        """ The two-factor quasi-Gaussian model with the Gaussian core of a calibrated G2pp. """
        corr = np.array([[1.0, g2.rho], [g2.rho, 1.0]])
        return cls(g2.curve, [g2.a, g2.b], [g2.sigma, g2.eta], corr, **kwargs)

    @property
    def params(self):
        return {"kappa": self.kappa.tolist(), "sigma": self.sigma.tolist(), "beta": self.beta,
                "epsilon": self.epsilon, "theta": self.theta}

    def with_curve(self, curve):
        model = copy.copy(self)
        model.curve = curve
        return model

    def with_params(self, vol_scale=1.0, **params):
        """ Copy with the factor vols multiplied by vol_scale and the scalar parameters replaced. """
        model = copy.copy(self)
        model.sigma = self.sigma * vol_scale
        for name, value in params.items():
            setattr(model, name, float(value))
        return model

    # ------------------------------------------------------------------
    # Zero-coupon bonds
    # ------------------------------------------------------------------
    def G(self, t, T):
        """ G_i(t,T) = (1 - e^{-kappa_i (T-t)}) / kappa_i, shape T.shape + (d,). """
        tau = np.asarray(T, dtype=float)[..., None] - t
        return (1.0 - np.exp(-self.kappa * tau)) / self.kappa

    def bond_price(self, t, T, state):
        """ P(t,T) given the state (x, Y) at t of shape (d + d^2, n_paths) ; T scalar, (m,) or (m, 1). """
        d = self.d
        T_arr = np.asarray(T, dtype=float)
        x, Y = state[:d], state[d:].reshape(d, d, -1)
        G = self.G(t, T_arr.reshape(-1))                                  # (m, d)
        exponent = G @ x + 0.5 * np.einsum('mi,ijn,mj->mn', G, Y, G)       # (m, n)
        P0 = (self.curve.discount(T_arr.reshape(-1)) / self.curve.discount(t))[:, None]
        out = P0 * np.exp(-exponent)
        return out[0] if T_arr.ndim == 0 else out.reshape(T_arr.shape[:1] + (-1,))

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def simulate(self, times, n_paths=50_000, seed=None, antithetic=True):
        """
        Euler simulation of (x, Y, z) and of int_0^t sum_i x_i on a grid of step <= dt containing `times`.
        Returns (states, D) : states of shape (len(times), d + d^2, n_paths) and D(0,t) = exp(-int_0^t r).
        """
        times = np.asarray(times, dtype=float)
        if np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise ValueError("times must be positive and strictly increasing.")
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even with antithetic variates.")
        rng = np.random.default_rng(seed)
        d, kappa = self.d, self.kappa
        vol_chol = np.diag(self.sigma) @ np.linalg.cholesky(self.corr)        # dx noise = m vol_chol dW
        cov = np.diag(self.sigma) @ self.corr @ np.diag(self.sigma)
        x = np.zeros((d, n_paths))
        Y = np.zeros((d, d, n_paths))
        z = np.ones(n_paths)
        integral = np.zeros(n_paths)
        states = np.empty((len(times), d + d * d, n_paths))
        I_out = np.empty((len(times), n_paths))
        m_half = n_paths // 2 if antithetic else n_paths
        previous = 0.0
        for k, t_end in enumerate(times):
            n_steps = max(1, int(np.ceil((t_end - previous) / self.dt - 1e-9)))
            h = (t_end - previous) / n_steps
            for _ in range(n_steps):
                Z = rng.standard_normal((d + 1, m_half))
                if antithetic:
                    Z = np.concatenate([Z, -Z], axis=1)
                zp = np.maximum(z, 0.0)
                m = np.sqrt(zp) * np.maximum(1.0 + self.beta * x.sum(axis=0), self.floor)
                drift = Y.sum(axis=1) - kappa[:, None] * x
                x_new = x + drift * h + m * (vol_chol @ Z[:d]) * np.sqrt(h)
                Y = Y + (m**2 * cov[:, :, None] - kappa[:, None, None] * Y - Y * kappa[None, :, None]) * h
                if self.epsilon > 0:
                    z = z + self.theta * (1.0 - zp) * h + self.epsilon * np.sqrt(zp * h) * Z[d]
                integral += 0.5 * (x.sum(axis=0) + x_new.sum(axis=0)) * h
                x = x_new
            states[k, :d], states[k, d:] = x, Y.reshape(d * d, -1)
            I_out[k] = integral
            previous = t_end
        return states, self.curve.discount(times)[:, None] * np.exp(-I_out)

    # ------------------------------------------------------------------
    # Smile calibration
    # ------------------------------------------------------------------
    def calibrate_smile(self, targets, names=("vol_scale", "beta", "epsilon"), x0=None, bounds=None,
                        n_paths=50_000, seed=7):
        """
        Calibrates the volatility scale, the local volatility skew beta and the vol of variance epsilon to swaption
        smiles by Monte Carlo with common random numbers.
        - targets : list of (expiry, tenor, strikes, market normal vols)
        Residuals in bp of normal vol. The model is updated in place (sigma scaled) ; returns a dict of results.
        """
        default_bounds = {"vol_scale": (0.3, 3.0), "beta": (-20.0, 20.0), "epsilon": (0.0, 1.5)}
        bounds = {**default_bounds, **(bounds or {})}
        x0 = [1.0 if n == "vol_scale" else getattr(self, n) for n in names] if x0 is None else x0
        market = np.concatenate([np.asarray(v, dtype=float) for *_, v in targets])

        def build(values):
            p = dict(zip(names, values))
            return self.with_params(p.pop("vol_scale", 1.0), **p)

        def residuals(values):
            model = build(values)
            vols = np.concatenate([mc_swaption_vols(model, e, t, K, n_paths, seed)[0] for e, t, K, _ in targets])
            return 1e4 * (vols - market)

        res = least_squares(residuals, x0, bounds=([bounds[n][0] for n in names], [bounds[n][1] for n in names]),
                            diff_step=1e-2)
        calibrated = build(res.x)
        self.sigma, self.beta, self.epsilon = calibrated.sigma, calibrated.beta, calibrated.epsilon
        return {**dict(zip(names, map(float, res.x))), "RMSE (bp of normal vol)": float(np.sqrt(np.mean(res.fun**2))),
                "max error (bp)": float(np.max(np.abs(res.fun)))}

    def __repr__(self):
        return (f"QuasiGaussian(d={self.d}, kappa={np.round(self.kappa, 4).tolist()}, "
                f"sigma(bp)={np.round(1e4 * self.sigma, 1).tolist()}, beta={self.beta:.3f}, epsilon={self.epsilon:.3f})")
