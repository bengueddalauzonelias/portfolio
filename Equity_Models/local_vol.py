import copy
import time
import numpy as np


# Local volatility (Dupire) and Local Stochastic Volatility (LSV) models.
#
# Everything is written in forward log-moneyness y = ln(S_t / F(0,t)), so that rates and dividends only enter
# through the forward curve : dy = -1/2 sigma^2 dt + sigma dW.
#
# 1. LocalVolSurface : Dupire local volatility from a smooth implied total variance w(y, T) = sigma_imp^2 T
#    (SSVI or SVI fit of ImpliedVolatilitySurface), Gatheral's formula
#        sigma_loc^2(T, y) = (dw/dT) / [1 - y/w dw/dy + 1/4 (-1/4 - 1/w + y^2/w^2) (dw/dy)^2 + 1/2 d2w/dy2].
# 2. LSVModel : Heston variance with a leverage function L(t, y),
#        dS/S = mu(t) dt + L(t, y) sqrt(v) dW^S,   dv = kappa (theta - v) dt + eta xi sqrt(v) dW^v,
#    calibrated so that L^2(t, y) E[v_t | y_t = y] = sigma_loc^2(t, y) : the model reprices every vanilla of the
#    local volatility surface (Gyongy's theorem). The mixing fraction eta in [0, 1] scales the vol-of-vol :
#    eta = 0 is the pure local volatility model, eta = 1 the full Heston dynamics.
#    The conditional expectation is computed with the particle method of Guyon & Henry-Labordere (2012) :
#    Nadaraya-Watson kernel regression of v on y at each time step of the simulation.
# The models expose generate_paths(T, M, N, seed, antithetic), S0, forward_curve, forward_price and zc_rate :
# they plug into MonteCarloPricer (monte_carlo.py), the Greeks (greeks.py) and every product.


class LocalVolSurface:
    """
    Dupire local volatility computed from an implied total variance function w(y, T), y = ln(K / F(0,T)).
    Derivatives are central finite differences ; the denominator is floored and the local volatility is clipped to
    [vol_floor, vol_cap] to stay finite where the implied surface has (or is close to) butterfly arbitrage.
    """

    def __init__(self, total_variance, name="local vol", vol_floor=0.03, vol_cap=1.5, dy=1e-3, dT=1e-4):
        """
        Init method.
        - total_variance : callable (y array, T float) -> w(y, T) = implied variance * T
        """
        self.total_variance = total_variance
        self.name = name
        self.vol_floor, self.vol_cap = vol_floor, vol_cap
        self.dy, self.dT = dy, dT

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_ssvi(cls, iv_surface, **kwargs):
        """ Local volatility of the SSVI fit of an ImpliedVolatilitySurface (calibrated if needed). """
        if getattr(iv_surface, 'ssvi_params', None) is None:
            iv_surface.fit_ssvi(verbose=False)
        params = dict(iv_surface.ssvi_params)
        w = lambda y, T: iv_surface.ssvi_total_variance(np.asarray(y, dtype=float), iv_surface.ssvi_theta(T), **params)
        return cls(w, name="Dupire (SSVI)", **kwargs)

    @classmethod
    def from_svi(cls, iv_surface, **kwargs):
        """
        Local volatility of the SVI slices of an ImpliedVolatilitySurface : total variance linear in T at fixed y
        between expiries, flat implied volatility outside.
        """
        if getattr(iv_surface, 'svi_params', None) is None:
            iv_surface.fit_svi(verbose=False)
        expiries = np.array(iv_surface.expiries, dtype=float)
        slices = [iv_surface.svi_params[T] for T in iv_surface.expiries]

        def w(y, T):
            y = np.asarray(y, dtype=float)
            slice_w = lambda k: np.maximum(iv_surface.svi_total_variance(y, **slices[k]), 1e-10)
            if T <= expiries[0]:
                return slice_w(0) * T / expiries[0]
            if T >= expiries[-1]:
                return slice_w(-1) * T / expiries[-1]
            k = int(np.searchsorted(expiries, T))
            T1, T2 = expiries[k - 1], expiries[k]
            return slice_w(k - 1) + (slice_w(k) - slice_w(k - 1)) * (T - T1) / (T2 - T1)
        return cls(w, name="Dupire (SVI)", **kwargs)

    @classmethod
    def flat(cls, sigma):
        """ Constant volatility (Black-Scholes) : sigma_loc = sigma. """
        return cls(lambda y, T: sigma**2 * T * np.ones_like(np.asarray(y, dtype=float)), name=f"flat {sigma:.0%}")

    def shifted(self, d):
        """ Surface with every implied volatility shifted by d (vega) : w_d = T (sqrt(w / T) + d)^2. """
        w = self.total_variance
        shifted = lambda y, T: T * (np.sqrt(np.maximum(w(y, T), 0.0) / T) + d)**2
        return LocalVolSurface(shifted, f"{self.name} {d:+.2%}", self.vol_floor, self.vol_cap, self.dy, self.dT)

    # ------------------------------------------------------------------
    # Volatilities
    # ------------------------------------------------------------------
    def implied_vol(self, y, T):
        return np.sqrt(np.maximum(self.total_variance(y, T), 0.0) / T)

    def local_variance(self, y, T):
        """ Dupire local variance at (T, y), vectorised in y. """
        y = np.asarray(y, dtype=float)
        T = max(float(T), 2 * self.dT)
        w = np.maximum(self.total_variance(y, T), 1e-12)
        w_up, w_down = self.total_variance(y + self.dy, T), self.total_variance(y - self.dy, T)
        w_y = (w_up - w_down) / (2 * self.dy)
        w_yy = (w_up - 2 * w + w_down) / self.dy**2
        w_T = (self.total_variance(y, T + self.dT) - self.total_variance(y, T - self.dT)) / (2 * self.dT)
        g = 1 - y / w * w_y + 0.25 * (-0.25 - 1 / w + y**2 / w**2) * w_y**2 + 0.5 * w_yy
        local = np.maximum(w_T, 0.0) / np.maximum(g, 1e-3)
        return np.clip(local, self.vol_floor**2, self.vol_cap**2)

    def local_vol(self, y, T):
        return np.sqrt(self.local_variance(y, T))

    def grid(self, times, ys):
        """ Local volatilities on the grid times x ys (rows : times). """
        return np.array([self.local_vol(ys, t) for t in times])


class LSVModel:
    """
    Local Stochastic Volatility model : Heston variance with a leverage function calibrated to a LocalVolSurface.
    With mixing = 0 the variance is deterministic and the model is the pure local volatility model (no particle
    noise : E[v | y] = v).
    """
    model = "lsv"

    def __init__(self, forward_curve, zc_rate, local_vol, v0, theta, kappa, xi, rho, mixing=1.0):
        """
        Init method.
        - forward_curve : ForwardCurve (drift of the spot through the forward)
        - zc_rate       : zero-coupon rate function (discounting)
        - local_vol     : LocalVolSurface to reprice
        - v0, theta, kappa, xi, rho : parameters of the stochastic variance (e.g. a calibrated Heston model)
        - mixing        : fraction eta of the vol-of-vol kept in the LSV model, in [0, 1]
        """
        if not 0 <= mixing <= 1: raise ValueError("mixing must be in [0, 1].")
        self.forward_curve = forward_curve
        self.forward_price = forward_curve.calc_forward
        self.zc_rate = zc_rate
        self.S0 = forward_curve.S0
        self.local_vol_surface = local_vol
        self.v0, self.theta, self.kappa, self.xi, self.rho = v0, theta, kappa, xi, rho
        self.mixing = float(mixing)
        self.leverage_grid = None

    @classmethod
    def from_heston(cls, heston, local_vol, mixing=1.0):
        return cls(heston.forward_curve, heston.zc_rate, local_vol, heston.v0, heston.theta, heston.kappa,
                   heston.xi, heston.rho, mixing)

    @property
    def params(self):
        return {"v0": self.v0, "theta": self.theta, "kappa": self.kappa, "xi": self.xi, "rho": self.rho,
                "mixing": self.mixing, "surface": self.local_vol_surface.name}

    @property
    def vol_of_vol(self):
        return self.mixing * self.xi

    # ------------------------------------------------------------------
    # Calibration of the leverage function (particle method)
    # ------------------------------------------------------------------
    def calibrate(self, t_max, n_particles=50_000, steps_per_year=52, seed=0, n_grid=60, bandwidth=1.5,
                  verbose=True):
        """
        Calibrates the leverage function L(t, y) on the time grid {i / steps_per_year} up to t_max.
        At each step, E[v_t | y_t = y] is estimated on a grid of n_grid points (quantiles 0.05% - 99.95% of the
        particles) by a Gaussian kernel regression of bandwidth h = bandwidth * std(y) * N^(-1/5), and
        L = sigma_loc / sqrt(E[v | y]) ; the particles are then moved one step with this leverage.
        """
        start = time.time()
        M = max(1, int(round(t_max * steps_per_year)))
        dt = t_max / M
        rng = np.random.default_rng(seed)
        N = n_particles
        y = np.zeros(N)
        v = np.full(N, float(self.v0))
        deterministic = self.vol_of_vol == 0
        times, grids, levels = [], [], []
        for i in range(M + 1):
            t = i * dt
            vp = np.maximum(v, 0.0)
            if i == 0 or deterministic:
                spread = max(4 * np.sqrt(self.v0 * max(t, dt)), 0.05) if i == 0 else None
                grid = (np.linspace(-spread, spread, n_grid) if i == 0
                        else np.linspace(*np.quantile(y, [5e-4, 1 - 5e-4]), n_grid))
                conditional = np.full(n_grid, vp.mean())
            else:
                grid = np.linspace(*np.quantile(y, [5e-4, 1 - 5e-4]), n_grid)
                h = bandwidth * np.std(y) * N**(-0.2)
                weights = np.exp(-0.5 * ((y[:, None] - grid[None, :]) / h)**2)
                conditional = (weights * vp[:, None]).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-300)
            leverage = self.local_vol_surface.local_vol(grid, max(t, dt)) / np.sqrt(np.maximum(conditional, 1e-8))
            times.append(t); grids.append(grid); levels.append(leverage)
            if i == M:
                break
            L = np.interp(y, grid, leverage)
            Z1, Z2 = rng.standard_normal(N), rng.standard_normal(N)
            sigma = L * np.sqrt(vp)
            y = y - 0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
            v = v + self.kappa * (self.theta - vp) * dt + self.vol_of_vol * np.sqrt(vp * dt) * Z1
        self.leverage_grid = (np.array(times), grids, levels)
        self.calibration_settings = dict(t_max=t_max, n_particles=n_particles, steps_per_year=steps_per_year, seed=seed,
                                         n_grid=n_grid, bandwidth=bandwidth)
        self.calibration_time = time.time() - start
        if verbose:
            print(f"LSV leverage calibrated (mixing {self.mixing:g}) : {M} steps, {N:,} particles, "
                  f"{self.calibration_time:.1f}s")
        return self

    def leverage(self, t, y):
        """ Leverage function at time t (calibration step just before t) and log-moneyness y. """
        if self.leverage_grid is None:
            raise RuntimeError("Call calibrate() first.")
        times, grids, levels = self.leverage_grid
        i = min(int(np.searchsorted(times, t + 1e-12, side='right')) - 1, len(times) - 1)
        return np.interp(y, grids[max(i, 0)], levels[max(i, 0)])

    # ------------------------------------------------------------------
    # Simulation (MonteCarloPricer interface)
    # ------------------------------------------------------------------
    def generate_paths(self, T, M, N, seed=None, antithetic=False):
        """
        Paths of the spot under the calibrated LSV model (Euler, full truncation of v).
        Returns (S, v_eff) of shape (M + 1, N) : v_eff = L^2 v is the instantaneous variance of the spot used over
        each step (needed by the Brownian bridge barrier correction).
        """
        if self.leverage_grid is None:
            raise RuntimeError("Call calibrate() first.")
        if antithetic and N % 2:
            raise ValueError("N must be even with antithetic variates.")
        rng = np.random.default_rng(seed)
        dt = T / M
        forwards = np.array([self.forward_price(i * dt) for i in range(M + 1)])
        y = np.zeros(N)
        v = np.full(N, float(self.v0))
        S = np.empty((M + 1, N))
        V = np.empty((M + 1, N))
        S[0] = self.S0
        n_draws = N // 2 if antithetic else N
        for i in range(M):
            vp = np.maximum(v, 0.0)
            L = self.leverage(i * dt, y)
            V[i] = L**2 * vp
            Z1, Z2 = rng.standard_normal(n_draws), rng.standard_normal(n_draws)
            if antithetic:
                Z1, Z2 = np.concatenate([Z1, -Z1]), np.concatenate([Z2, -Z2])
            sigma = np.sqrt(V[i])
            y = y - 0.5 * V[i] * dt + sigma * np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
            v = v + self.kappa * (self.theta - vp) * dt + self.vol_of_vol * np.sqrt(vp * dt) * Z1
            S[i + 1] = forwards[i + 1] * np.exp(y)
        V[M] = self.leverage(T, y)**2 * np.maximum(v, 0.0)
        return S, V

    # ------------------------------------------------------------------
    # Greeks support
    # ------------------------------------------------------------------
    def vol_shifted(self, d_sigma):
        """
        Copy of the model with every implied volatility shifted by d_sigma, the leverage being recalibrated
        (vega of the LSV model : a parallel move of the whole implied surface).
        """
        model = copy.copy(self)
        model.local_vol_surface = self.local_vol_surface.shifted(d_sigma)
        model.leverage_grid = None
        return model.calibrate(**self.calibration_settings, verbose=False)

    def __repr__(self):
        return (f"LSVModel(mixing={self.mixing:g}, v0={self.v0:.4f}, theta={self.theta:.4f}, kappa={self.kappa:.3f}, "
                f"xi={self.xi:.3f}, rho={self.rho:.3f}, surface={self.local_vol_surface.name})")


def reprice_surface(model, iv_surface, expiries=None, n_paths=100_000, steps_per_year=252, seed=7, target=None):
    """
    Implied volatilities of `model` at the quoted (expiry, strike) points of an ImpliedVolatilitySurface, from one
    simulation up to the last expiry (OTM options, Black inversion with the market forward and discounting).
    Daily steps by default : the shortest expiry (9 days) needs several steps for the smile to build up.
    target : optional LocalVolSurface, whose implied volatility is reported as well (the surface the model fits).
    Returns a DataFrame with one row per quote : expiry, strike, market vol, model vol (and target vol).
    """
    import pandas as pd
    expiries = list(iv_surface.expiries) if expiries is None else list(expiries)
    T_max = max(expiries)
    M = max(1, int(round(T_max * steps_per_year)))
    S, _ = model.generate_paths(T_max, M, n_paths, seed=seed, antithetic=True)
    rows = []
    for T in expiries:
        i = int(round(T / T_max * M))
        T_grid = i * T_max / M
        F = model.forward_price(T_grid)
        df = np.exp(-model.zc_rate(T_grid) * T_grid)
        for K, market_vol in zip(iv_surface.strikes[T], iv_surface.implied_vols[T]):
            is_call = bool(K > F)
            payoff = np.maximum(S[i] - K, 0.0) if is_call else np.maximum(K - S[i], 0.0)
            vol = iv_surface.get_implied_vol(T_grid, K, float(df * payoff.mean()), is_call, verbose=False)
            row = {"expiry": T, "strike": K, "market vol": market_vol, "model vol": np.nan if vol is None else vol}
            if target is not None:
                row["target vol"] = float(target.implied_vol(np.log(K / model.forward_price(T)), T))
            rows.append(row)
    return pd.DataFrame(rows)
