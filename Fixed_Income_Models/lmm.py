import copy
import numpy as np
from scipy.optimize import least_squares

try:
    from .cms import forward_swap
except ImportError:
    from cms import forward_swap


# LIBOR Market Model with SABR stochastic volatility (SABR-LMM), single curve, forward rates on a regular grid
# T_j = j tau, j = 0..N : F_j(t) = (P(t,T_j) / P(t,T_{j+1}) - 1) / tau fixes at T_j.
#     dF_j = sigma_j(t) V(t) phi(F_j) dW_j,    phi(F) = (F + shift)^beta,
#     dV   = nu V dZ,   V(0) = 1,                                  (common volatility factor, Mercurio & Morini 2009)
#     sigma_j(t) = k_j [(a + b (T_j - t)) e^{-c (T_j - t)} + d]    (Rebonato's abcd),
#     W_j = rho_sv Z + sqrt(1 - rho_sv^2) W~_j,   corr(W~_i, W~_j) = rho_inf + (1 - rho_inf) e^{-decay |T_i - T_j|}.
# nu = 0 is the plain (CEV / shifted / normal) LMM. beta = 0 (default) is the NORMAL SABR-LMM : rates may be
# negative (EUR 2015-2021) and each swap rate is approximately a normal SABR with the same nu.
# - Simulation under the spot LIBOR measure (numeraire B(T_k) = prod_{j<k} (1 + tau F_j(T_j)), V defined under it),
#   predictor-corrector drift
#       mu_j = sigma_j V phi(F_j) sum_{i = q(t)}^{j} rho_ij tau sigma_i V phi(F_i) / (1 + tau F_i),
#   q(t) = first forward not fixed at t, exact lognormal step for V ;
# - swaption normal vols and swap rate correlations by Rebonato's frozen weights S = sum_j w_j F_j,
#   w_j = tau P(0,T_{j+1}) / A(0), for the calibration of abcd and of the correlation.
# Same simulation interface as HullWhite : simulate(times, ...) -> (states, D), bond_price(t, T, state) with
# state = states[k] = the N forwards at t (fixed forwards keep their fixing) ; times must lie on the grid.


class SABRLMM:
    """
    SABR-LMM (or plain LMM with nu = 0) on a YieldCurve.
    """
    NAMES = ("a", "b", "c", "d", "rho_inf", "decay")
    PARAMS = {"a": (-0.05, 0.05), "b": (-0.05, 0.05), "c": (0.01, 3.0), "d": (1e-4, 0.05),
              "rho_inf": (0.0, 0.99), "decay": (1e-3, 2.0)}
    model = "sabr_lmm"

    def __init__(self, curve, n_forwards, tau=1.0, a=0.002, b=0.002, c=0.5, d=0.007, rho_inf=0.4, decay=0.1,
                 nu=0.0, rho_sv=0.0, beta=0.0, shift=0.0, k=None, steps_per_period=4):
        """
        Init method.
        - curve           : YieldCurve, gives the initial forwards
        - n_forwards      : number N of forwards (grid up to T_N = N tau)
        - tau             : forward tenor (1.0 : annual forwards, matching the annual fixed leg of EUR swaps)
        - a, b, c, d      : abcd volatility parameters (normal vols when beta = 0)
        - rho_inf, decay  : correlation parameters of the forwards
        - nu, rho_sv      : vol of vol and correlation forwards / volatility
        - beta, shift     : CEV exponent and shift of phi
        - k               : per-forward scaling of the abcd volatility (ones by default)
        """
        self.curve = curve
        self.N, self.tau = int(n_forwards), float(tau)
        self.grid = self.tau * np.arange(self.N + 1)
        self.F0 = (curve.discount(self.grid[:-1]) / curve.discount(self.grid[1:]) - 1) / self.tau
        self.a, self.b, self.c, self.d, self.rho_inf, self.decay = map(float, (a, b, c, d, rho_inf, decay))
        self.nu, self.rho_sv, self.beta, self.shift = map(float, (nu, rho_sv, beta, shift))
        self.k = np.ones(self.N) if k is None else np.asarray(k, dtype=float)
        self.steps_per_period = int(steps_per_period)
        if self.beta > 0 and np.any(self.F0 + self.shift <= 0):
            raise ValueError("F + shift must be positive when beta > 0.")

    @property
    def params(self):
        return {n: getattr(self, n) for n in self.NAMES + ("nu", "rho_sv", "beta", "shift")}

    def with_curve(self, curve):
        return SABRLMM(curve, self.N, self.tau, **{n: getattr(self, n) for n in self.NAMES}, nu=self.nu,
                       rho_sv=self.rho_sv, beta=self.beta, shift=self.shift, k=self.k,
                       steps_per_period=self.steps_per_period)

    def with_params(self, **params):
        model = copy.copy(self)
        for name, value in params.items():
            setattr(model, name, float(value))
        return model

    # ------------------------------------------------------------------
    # Volatility and correlation
    # ------------------------------------------------------------------
    def vol(self, t, j):
        """ sigma_j(t) (zero once F_j has fixed). Vectorised in t and j. """
        T = self.grid[np.asarray(j)]
        s = np.maximum(T - np.asarray(t, dtype=float), 0.0)
        return np.where(T > t, self.k[np.asarray(j)] * ((self.a + self.b * s) * np.exp(-self.c * s) + self.d), 0.0)

    def forward_correlation(self):
        """ Correlation matrix of the forward Brownian motions W_j (including the common vol factor). """
        T = self.grid[:-1]
        tilde = self.rho_inf + (1 - self.rho_inf) * np.exp(-self.decay * np.abs(T[:, None] - T[None, :]))
        return self.rho_sv**2 + (1 - self.rho_sv**2) * tilde

    def phi(self, F):
        return np.ones_like(F) if self.beta == 0 else np.maximum(F + self.shift, 1e-8)**self.beta

    # ------------------------------------------------------------------
    # Zero-coupon bonds
    # ------------------------------------------------------------------
    def _index(self, t):
        i = np.rint(np.asarray(t, dtype=float) / self.tau).astype(int)
        if np.any(np.abs(i * self.tau - t) > 1e-8) or np.any(i > self.N) or np.any(i < 0):
            raise ValueError("Times must lie on the forward grid.")
        return i

    def bond_price(self, t, T, state):
        """ P(t,T) = prod_{j=q}^{m-1} 1 / (1 + tau F_j(t)), for t = T_q and T = T_m on the grid. """
        q, m = int(self._index(t)), self._index(T)
        cumulative = np.concatenate([np.zeros((1,) + state.shape[1:]),
                                     np.cumsum(np.log1p(self.tau * state), axis=0)])
        return np.exp(-(cumulative[m] - cumulative[q])).reshape(np.shape(T)[:1] + (-1,)) if np.ndim(T) \
            else np.exp(-(cumulative[m] - cumulative[q]))

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def _drift(self, F, V, sig, corr, alive):
        """ Spot measure drift of the alive forwards. """
        vol = sig[:, None] * V * self.phi(F)                                   # (N, n)
        weighted = np.where(alive[:, None], self.tau * vol / (1 + self.tau * F), 0.0)
        # sum_{i alive, i <= j} rho_ij weighted_i : lower triangular part of the correlation
        return vol * (np.tril(corr) @ weighted)

    def simulate(self, times, n_paths=50_000, seed=None, antithetic=True):
        """
        Simulation of the forwards on the grid dates `times` under the spot LIBOR measure.
        Returns (states, D) : states of shape (len(times), N, n_paths) and D(0, t) = 1 / B(t).
        """
        times = np.asarray(times, dtype=float)
        idx = self._index(times)
        if np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise ValueError("times must be positive and strictly increasing.")
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even with antithetic variates.")
        rng = np.random.default_rng(seed)
        corr_tilde = (self.forward_correlation() - self.rho_sv**2) / (1 - self.rho_sv**2)
        chol = np.linalg.cholesky(corr_tilde + 1e-12 * np.eye(self.N))
        corr = self.forward_correlation()
        F = np.repeat(self.F0[:, None], n_paths, axis=1)
        V = np.ones(n_paths)
        m_half = n_paths // 2 if antithetic else n_paths
        h = self.tau / self.steps_per_period
        states = np.empty((len(times), self.N, n_paths))
        out = 0
        for period in range(int(idx[-1])):
            for step in range(self.steps_per_period):
                t = period * self.tau + step * h
                alive = self.grid[:-1] > t + 1e-12
                sig = self.vol(t, np.arange(self.N))
                G = rng.standard_normal((self.N + 1, m_half))
                if antithetic:
                    G = np.concatenate([G, -G], axis=1)
                dZ = G[0] * np.sqrt(h)
                dW = (self.rho_sv * G[0][None, :] + np.sqrt(1 - self.rho_sv**2) * (chol @ G[1:])) * np.sqrt(h)
                diffusion = sig[:, None] * V * self.phi(F) * dW
                mu0 = self._drift(F, V, sig, corr, alive)
                predictor = F + mu0 * h + diffusion
                mu1 = self._drift(predictor, V, sig, corr, alive)
                F = np.where(alive[:, None], F + 0.5 * (mu0 + mu1) * h + diffusion, F)
                V = V * np.exp(self.nu * dZ - 0.5 * self.nu**2 * h)
            while out < len(times) and idx[out] == period + 1:
                states[out] = F
                out += 1
        log_numeraire = np.cumsum(np.log1p(self.tau * states[-1]), axis=0)      # prod_{j<k} (1 + tau F_j(T_j))
        D = np.exp(-np.stack([log_numeraire[i - 1] for i in idx]))
        return states, D

    # ------------------------------------------------------------------
    # Swap rates : frozen weights
    # ------------------------------------------------------------------
    def _weights(self, expiry, tenor):
        q = int(self._index(expiry))
        n = int(round(tenor / self.tau))
        P = self.curve.discount(self.grid[q + 1:q + n + 1])
        w = self.tau * P / np.sum(self.tau * P)
        return np.arange(q, q + n), w * self.phi(self.F0[q:q + n])

    def _covariance(self, expiry, leg1, leg2, n_nodes=48):
        """ int_0^T0 sum_ij w1_i w2_j rho_ij sigma_i sigma_j dt (Gauss-Legendre in t). """
        (j1, w1), (j2, w2) = leg1, leg2
        x, wq = np.polynomial.legendre.leggauss(n_nodes)
        t = 0.5 * expiry * (x + 1)
        s1 = self.vol(t[:, None], j1[None, :]) * w1           # (nodes, n1)
        s2 = self.vol(t[:, None], j2[None, :]) * w2
        rho = self.forward_correlation()[np.ix_(j1, j2)]
        return float(0.5 * expiry * np.sum(wq * np.einsum('ti,ij,tj->t', s1, rho, s2)))

    def swaption_normal_vol(self, expiry, tenor):
        """
        ATM normal vol of the swap rate expiry x tenor : the swap rate is approximately a normal SABR with
        alpha_S^2 = covariance / T, nu and rho_S = rho_sv (sum w sigma) / sqrt(w' rho w sigma sigma).
        """
        leg = self._weights(expiry, tenor)
        alpha = np.sqrt(self._covariance(expiry, leg, leg) / expiry)
        return float(alpha * (1 + (2 - 3 * self.swap_rate_vol_correlation(expiry, tenor)**2) / 24 * self.nu**2 * expiry))

    def swap_rate_vol_correlation(self, expiry, tenor):
        """ Correlation between the swap rate and the volatility factor (frozen weights, time averaged). """
        j, w = self._weights(expiry, tenor)
        x, wq = np.polynomial.legendre.leggauss(32)
        t = 0.5 * expiry * (x + 1)
        s = self.vol(t[:, None], j[None, :]) * w
        num = np.sum(wq * s.sum(axis=1))
        den = np.sum(wq * np.sqrt(np.einsum('ti,ij,tj->t', s, self.forward_correlation()[np.ix_(j, j)], s)))
        return float(self.rho_sv * num / den)

    def swap_rate_correlation(self, expiry, tenor1, tenor2):
        """ Terminal correlation of the swap rates expiry x tenor1 and expiry x tenor2 (frozen weights). """
        l1, l2 = self._weights(expiry, tenor1), self._weights(expiry, tenor2)
        return self._covariance(expiry, l1, l2) / np.sqrt(self._covariance(expiry, l1, l1) * self._covariance(expiry, l2, l2))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, vol_grid, expiries=None, tenors=None, correlation_targets=(), correlation_weight=1.0,
                  fixed=None, bounds=None):
        """
        Calibrates abcd and the correlation (rho_inf, decay) to the ATM normal vols of vol_grid and to correlation
        targets [(expiry, tenor1, tenor2, target), ...], nu and rho_sv being fixed (taken from the SABR smile).
        Residuals : vol errors in bp and correlation_weight * correlation errors in percentage points.
        """
        expiries = vol_grid.expiries if expiries is None else np.asarray(expiries, dtype=float)
        tenors = vol_grid.tenors if tenors is None else np.asarray(tenors, dtype=float)
        points = [(e, t) for e in expiries for t in tenors if e + t <= self.grid[-1] + 1e-9]
        targets = np.array([vol_grid.vol(e, t) for e, t in points])
        fixed = dict(fixed or {})
        bounds = {**self.PARAMS, **(bounds or {})}
        names = [n for n in self.NAMES if n not in fixed]

        def residuals(values):
            model = self.with_params(**{**fixed, **dict(zip(names, values))})
            if np.any(model.vol(0.0, np.arange(1, model.N)) <= 0):               # positivity of the alive sigmas
                return np.full(len(points) + len(correlation_targets), 1e3)
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
        return {**{n: getattr(self, n) for n in self.NAMES},
                "RMSE (bp of normal vol)": float(np.sqrt(np.mean(vol_errors**2))),
                "max error (bp)": float(np.max(np.abs(vol_errors))),
                **{f"corr {e:g}y {t1:g}y/{t2:g}y": self.swap_rate_correlation(e, t1, t2)
                   for e, t1, t2, _ in correlation_targets}}

    def __repr__(self):
        return (f"SABRLMM(N={self.N}, tau={self.tau:g}, abcd=({self.a:.4f}, {self.b:.4f}, {self.c:.3f}, {self.d:.4f}), "
                f"rho_inf={self.rho_inf:.3f}, decay={self.decay:.3f}, nu={self.nu:.3f}, rho_sv={self.rho_sv:+.3f}, "
                f"beta={self.beta:g})")
