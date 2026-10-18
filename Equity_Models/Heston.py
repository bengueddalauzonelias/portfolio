import copy
import time
import numpy as np
import pandas as pd
import scipy.optimize as opt
from forward_curve import ForwardCurve


class HestonModel:
    # Calibrated parameters and their bounds (read by optimizer.Calibrator)
    PARAMS = {"v0": (1e-3, 1.0), "theta": (1e-3, 1.0), "kappa": (1e-3, 20.0), "xi": (1e-2, 5.0), "rho": (-0.999, 0.999)}

    def __init__(self, zc_rate, forward_curve,
                 v0: float = 0.04, theta: float = 0.04, kappa: float = 2.0,
                 xi: float = 0.3, rho: float = -0.7):
        """
        Heston Model variables initialization.

        Market Parameters :
        - zc_rate       : Callable T -> continuously compounded zero rate for maturity T
        - forward_curve : ForwardCurve object (market implied forwards, i.e. rates + borrow + dividends)

        Heston Parameters :
        - v0    : Instantaneous variance
        - theta : Long term mean variance
        - kappa : Mean reversion speed of the variance
        - xi    : Vol-of-vol
        - rho   : Correlation between spot and variance
        """
        # Sanity checks
        if zc_rate is None or not callable(zc_rate):
            raise ValueError("zc_rate must be a callable function.")
        if forward_curve is None or not isinstance(forward_curve, ForwardCurve):
            raise ValueError("forward_curve must be an object of the ForwardCurve class.")
        if xi <= 0:
            raise ValueError("xi (vol-of-vol) must be positive.")

        self.zc_rate = zc_rate
        self.forward_curve = forward_curve
        self.forward_price = forward_curve.calc_forward
        self.S0 = forward_curve.S0

        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.xi = xi
        self.rho = rho

    @property
    def params(self):
        return {"v0": self.v0, "theta": self.theta, "kappa": self.kappa, "xi": self.xi, "rho": self.rho}

    def vol_shifted(self, d_sigma):
        """
        Copy of the model with a parallel shift of the volatility level (used for vega / volga / vanna) :
        sqrt(v0) and sqrt(theta) are both shifted by d_sigma (e.g. 0.01 = 1 vol point).
        """
        model = copy.copy(self)
        model.v0 = (np.sqrt(self.v0) + d_sigma)**2
        model.theta = (np.sqrt(self.theta) + d_sigma)**2
        return model

    # ------------------------------------------------------------------
    # Characteristic function
    # ------------------------------------------------------------------
    def characteristic_function(self, u, T):
        """
        Characteristic function of X_T = ln(S_T / F(0,T)) : E[exp(i u X_T)].
        Gatheral's formulation with the Albrecher et al. stable branch
        (avoids the "Little Heston Trap" discontinuities of the log).
        """
        i = 1j
        beta = self.kappa - i * self.rho * self.xi * u
        d = np.sqrt(beta**2 + (self.xi**2) * (u**2 + i * u))
        g = (beta - d) / (beta + d)
        exp_minus_dT = np.exp(-d * T)

        C = (self.kappa * self.theta / self.xi**2) * (
            (beta - d) * T - 2.0 * np.log((1.0 - g * exp_minus_dT) / (1.0 - g))
        )
        D = ((beta - d) / self.xi**2) * ((1.0 - exp_minus_dT) / (1.0 - g * exp_minus_dT))
        return np.exp(C + D * self.v0)

    # ------------------------------------------------------------------
    # Semi-analytical pricing
    # ------------------------------------------------------------------
    def price_vanillas(self, strikes, maturity: float, isCall=True, u_max: float = 1000.0, n_u: int = 4000):
        """
        Vectorised Heston prices for several strikes at one maturity (Lewis 2001 single-integral formula) :
        C = DF(T) * [ F - sqrt(F K) / pi * int_0^inf Re( e^{i u k} phi(u - i/2) ) / (u^2 + 1/4) du ],  k = ln(F/K)
        Puts are obtained by Call / Put parity.
        """
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
        isCall = np.broadcast_to(np.asarray(isCall, dtype=bool), strikes.shape)
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if (strikes <= 0).any():
            raise ValueError("Strikes must be positive.")

        forward = self.forward_price(maturity)
        df = np.exp(-self.zc_rate(maturity) * maturity)

        u = np.linspace(1e-8, u_max, n_u)
        phi = self.characteristic_function(u - 0.5j, maturity)
        k = np.log(forward / strikes)[:, None]
        integrand = (np.exp(1j * u[None, :] * k) * phi[None, :]).real / (u**2 + 0.25)[None, :]
        integral = np.trapezoid(integrand, u, axis=1)

        call = df * (forward - np.sqrt(forward * strikes) / np.pi * integral)
        put = call - df * (forward - strikes)
        return np.where(isCall, call, put)

    def price_vanilla(self, strike: float, maturity: float, isCall=True) -> float:
        """ Heston price of a single European vanilla option. """
        return float(self.price_vanillas([strike], maturity, isCall)[0])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, implied_vol_surface, bs_model, bounds=None, maxiter=200):
        """
        Calibration on the market implied volatility surface.
        The objective is the vega-weighted price error (P_model - P_market) / Vega_market,
        which is a first-order approximation of the implied volatility error.
        Errors are averaged per expiry first, then across expiries (each expiry weighs the same).
        """
        # Market quotes (only the strikes where an implied vol was found)
        quotes = []
        for exp in implied_vol_surface.expiries:
            forward = self.forward_price(exp)
            strikes = np.array(implied_vol_surface.strikes[exp], dtype=float)
            vols = np.array(implied_vol_surface.implied_vols[exp], dtype=float)
            isCall = strikes > forward
            mkt_prices = np.array([bs_model.bs_price_vanilla(exp, K, v, bool(c)) for K, v, c in zip(strikes, vols, isCall)])
            vegas = np.array([bs_model.bs_vega(exp, K, v) for K, v in zip(strikes, vols)])
            quotes.append((exp, strikes, isCall, mkt_prices, np.maximum(vegas, 1e-8)))

        def objective_function(params):
            self.v0, self.theta, self.kappa, self.xi, self.rho = params
            mat_means = []
            for exp, strikes, isCall, mkt_prices, vegas in quotes:
                model_prices = self.price_vanillas(strikes, exp, isCall)
                errors = ((model_prices - mkt_prices) / vegas) ** 2
                mat_means.append(np.mean(errors) if np.all(np.isfinite(errors)) else 10.0)
            return np.mean(mat_means)

        initial_guess = [self.v0, self.theta, self.kappa, self.xi, self.rho]
        if bounds is None:
            bounds = ((1e-3, 1.0), (1e-3, 1.0), (1e-3, 20.0), (1e-2, 5.0), (-0.999, 0.999))

        result = opt.minimize(objective_function, initial_guess, method='L-BFGS-B',
                              bounds=bounds, options={'ftol': 1e-10, 'maxiter': maxiter})
        self.v0, self.theta, self.kappa, self.xi, self.rho = result.x

        status = "Calibration OK" if result.success else f"Calibration did not converge : {result.message}"
        print(status)
        print(f"RMSE (implied vol, approx.) : {100 * np.sqrt(result.fun):.3f}%")
        print(f"Calibrated parameters : v0={self.v0:.4f}, theta={self.theta:.4f}, "
              f"kappa={self.kappa:.4f}, xi={self.xi:.4f}, rho={self.rho:.4f}")
        feller = 2 * self.kappa * self.theta - self.xi**2
        print(f"Feller condition 2*kappa*theta - xi^2 = {feller:.4f} ({'satisfied' if feller > 0 else 'violated'})")
        return result

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def generate_paths(self, T, M, N, seed=None, antithetic=False):
        """
        Path generation under the Heston model (Euler, full truncation).
        The drift over each step is ln(F(t2) / F(t1)), so that E[S_T] = F(0,T) is matched by construction.
        If antithetic is True (N must be even), paths N/2..N-1 use the opposite Gaussian draws of paths 0..N/2-1.
        """
        if antithetic and N % 2 != 0:
            raise ValueError("N must be even with antithetic variates.")
        rng = np.random.default_rng(seed)
        dt = T / M
        S = np.zeros((M + 1, N))
        v = np.zeros((M + 1, N))
        S[0] = self.S0
        v[0] = self.v0
        sqrt_1_m_rho2 = np.sqrt(1.0 - self.rho**2)
        forwards = np.array([self.forward_price(t * dt) for t in range(M + 1)])
        n_draws = N // 2 if antithetic else N

        for t in range(M):
            fwd_drift = np.log(forwards[t + 1] / forwards[t])
            Z1 = rng.standard_normal(n_draws)
            Z2 = rng.standard_normal(n_draws)
            if antithetic:
                Z1, Z2 = np.concatenate([Z1, -Z1]), np.concatenate([Z2, -Z2])
            Z_S = self.rho * Z1 + sqrt_1_m_rho2 * Z2

            v_curr = np.maximum(v[t], 0.0)
            v[t + 1] = v[t] + self.kappa * (self.theta - v_curr) * dt + self.xi * np.sqrt(v_curr * dt) * Z1
            S[t + 1] = S[t] * np.exp(fwd_drift - 0.5 * v_curr * dt + np.sqrt(v_curr * dt) * Z_S)

        return S, v


class TimeDependentHestonModel(HestonModel):
    """
    Heston model with piecewise constant parameters (Mikhailov & Nogel 2003) :
        dS_t / S_t = mu(t) dt + sqrt(v_t) dW_t
        dv_t = kappa (theta(t) - v_t) dt + xi(t) sqrt(v_t) dZ_t,   d<W, Z>_t = rho(t) dt
    theta, xi and rho are constant on each interval (times[j-1], times[j]] (times[-1] = 0), and keep their last value
    beyond times[-1]. v0 and kappa are global. With one interval, the model is the constant-parameter Heston model.
    """
    SEGMENT_PARAMS = ("theta", "xi", "rho")

    def __init__(self, zc_rate, forward_curve, times, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7):
        """
        - times : increasing end times of the intervals (typically the listed expiries)
        - theta, xi, rho : scalars (same value on every interval) or arrays of len(times)
        """
        self.times = np.asarray(times, dtype=float)
        if self.times.ndim != 1 or len(self.times) == 0 or np.any(np.diff(self.times) <= 0) or self.times[0] <= 0:
            raise ValueError("times must be a non-empty increasing array of positive times.")
        n = len(self.times)
        theta, xi, rho = (np.broadcast_to(np.asarray(p, dtype=float), (n,)).copy() for p in (theta, xi, rho))
        if np.any(xi <= 0):
            raise ValueError("xi (vol-of-vol) must be positive on every interval.")
        super().__init__(zc_rate, forward_curve, v0=v0, theta=0.04, kappa=kappa, xi=0.3, rho=-0.7)
        self.theta, self.xi, self.rho = theta, xi, rho

    @classmethod
    def from_heston(cls, heston, times):
        """ Time-dependent model with every interval set to the parameters of a constant Heston model. """
        return cls(heston.zc_rate, heston.forward_curve, times, v0=heston.v0, kappa=heston.kappa,
                   theta=heston.theta, xi=heston.xi, rho=heston.rho)

    @property
    def params(self):
        return {"v0": self.v0, "kappa": self.kappa, "theta": self.theta.copy(), "xi": self.xi.copy(),
                "rho": self.rho.copy()}

    def segment_table(self):
        """ DataFrame of the piecewise constant parameters, one row per interval. """
        starts = np.concatenate([[0.0], self.times[:-1]])
        return pd.DataFrame({"start": starts, "end": self.times, "theta": self.theta, "xi": self.xi, "rho": self.rho,
                             "Feller ratio": 2 * self.kappa * self.theta / self.xi**2})

    def segment_index(self, t):
        """ Index of the interval (times[j-1], times[j]] containing t (the last one beyond times[-1]). """
        return np.minimum(np.searchsorted(self.times, t, side='left'), len(self.times) - 1)

    def _own_arrays(self):
        """ Copies the parameter arrays, so that in-place updates never reach a model this one was copied from. """
        for name in self.SEGMENT_PARAMS:
            setattr(self, name, np.array(getattr(self, name), dtype=float))

    def vol_shifted(self, d_sigma):
        """ Copy with sqrt(v0) and sqrt(theta(t)) shifted by d_sigma on every interval. """
        model = copy.copy(self)
        model.v0 = (np.sqrt(self.v0) + d_sigma)**2
        model.theta = (np.sqrt(self.theta) + d_sigma)**2
        return model

    # ------------------------------------------------------------------
    # Characteristic function
    # ------------------------------------------------------------------
    def characteristic_function(self, u, T):
        """
        Characteristic function of X_T = ln(S_T / F(0,T)) : E[exp(i u X_T)] = exp(C(u, T) + D(u, T) v0).
        The Riccati equations are solved backward from T to 0, interval by interval : on an interval of length tau
        with constant parameters and terminal values (C0, D0),
            g = (beta - d - xi^2 D0) / (beta + d - xi^2 D0)
            D = D0 + (beta - d - xi^2 D0) / xi^2 * (1 - e^{-d tau}) / (1 - g e^{-d tau})
            C = C0 + kappa theta / xi^2 * [ (beta - d) tau - 2 ln((1 - g e^{-d tau}) / (1 - g)) ]
        with beta = kappa - i rho xi u and d = sqrt(beta^2 + xi^2 (u^2 + i u)) (Albrecher et al. stable branch).
        """
        u = np.asarray(u)
        bounds = np.concatenate([[0.0], self.times[self.times < T], [T]])
        C = np.zeros(u.shape, dtype=complex)
        D = np.zeros(u.shape, dtype=complex)
        for k in range(len(bounds) - 2, -1, -1):
            tau = bounds[k + 1] - bounds[k]
            j = self.segment_index(bounds[k + 1])
            kappa, theta, xi, rho = self.kappa, self.theta[j], self.xi[j], self.rho[j]
            beta = kappa - 1j * rho * xi * u
            d = np.sqrt(beta**2 + xi**2 * (u**2 + 1j * u))
            a = beta - d - xi**2 * D
            g = a / (beta + d - xi**2 * D)
            e = np.exp(-d * tau)
            C = C + (kappa * theta / xi**2) * ((beta - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
            D = D + (a / xi**2) * (1.0 - e) / (1.0 - g * e)
        return np.exp(C + D * self.v0)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _expiry_objective(self, quote, names, j, bounds, n_u):
        """ Vega-weighted IV error of one expiry as a function of the parameters `names` (on interval j for theta,
        xi, rho). Parameters are mapped to [0, 1]. """
        lower, upper = np.array([bounds[n][0] for n in names]), np.array([bounds[n][1] for n in names])

        def to_params(x):
            return lower + np.clip(x, 0.0, 1.0) * (upper - lower)

        def objective(x):
            for name, value in zip(names, to_params(x)):
                if name in self.SEGMENT_PARAMS:
                    getattr(self, name)[j] = value
                else:
                    setattr(self, name, value)
            prices = self.price_vanillas(quote["strikes"], quote["expiry"], quote["isCall"], n_u=n_u)
            errors = ((prices - quote["prices"]) / quote["vegas"])**2
            if not np.all(np.isfinite(errors)):
                return 10.0
            return float(np.sum(quote["weights"] * errors) / np.sum(quote["weights"]))

        x0 = []
        for name in names:
            value = getattr(self, name)[j] if name in self.SEGMENT_PARAMS else getattr(self, name)
            x0.append((np.clip(value, lower[names.index(name)], upper[names.index(name)]) - lower[names.index(name)])
                      / (upper[names.index(name)] - lower[names.index(name)]))
        return objective, np.array(x0), to_params

    def calibrate_bootstrap(self, calibration_loss, free=("theta", "xi", "rho"), first_free=("v0",), bounds=None,
                            maxiter=200, n_u=4000, verbose=True):
        """
        Bootstrap calibration (the intervals must end at the expiries of calibration_loss) : for each expiry j in
        increasing order, the parameters `free` of interval j are fitted to the smile of expiry j alone (vega-weighted
        IV error), the earlier intervals being fixed. The global parameters `first_free` (e.g. v0) are fitted with the
        first expiry. kappa is kept. Returns a DataFrame with one row per expiry.
        """
        expiries = np.array([q["expiry"] for q in calibration_loss.quotes])
        if len(expiries) != len(self.times) or not np.allclose(expiries, self.times):
            raise ValueError("The intervals must end at the expiries of calibration_loss.")
        self._own_arrays()
        all_bounds = dict(self.PARAMS)
        all_bounds.update(bounds or {})
        rows = []
        for j, quote in enumerate(calibration_loss.quotes):
            names = list(free) + (list(first_free) if j == 0 else [])
            start = time.time()
            objective, x0, to_params = self._expiry_objective(quote, names, j, all_bounds, n_u)
            loss_before = objective(x0)
            res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=[(0.0, 1.0)] * len(names),
                               options={'ftol': 1e-12, 'maxiter': maxiter})
            objective(res.x)                                           # writes the optimum into the model
            if j + 1 < len(self.times):                                # next interval starts from this one
                for name in self.SEGMENT_PARAMS:
                    getattr(self, name)[j + 1] = getattr(self, name)[j]
            rows.append({"expiry": quote["expiry"], **{n: v for n, v in zip(names, to_params(res.x))},
                         "RMSE before (vol pts)": 100 * np.sqrt(loss_before),
                         "RMSE after (vol pts)": 100 * np.sqrt(res.fun), "evaluations": res.nfev,
                         "time (s)": time.time() - start})
        table = pd.DataFrame(rows).set_index("expiry")
        if verbose:
            print(f"Bootstrap of {len(rows)} expiries ({', '.join(free)}) : {table['evaluations'].sum()} evaluations, "
                  f"{table['time (s)'].sum():.1f}s")
        return table

    def calibrate_global(self, calibration_loss, free=("theta", "xi", "rho"), global_free=("v0", "kappa"),
                         bounds=None, smoothness=0.0, maxiter=300, verbose=True):
        """
        Joint calibration of all the parameters on all the expiries (CalibrationLoss 'vega_iv_mse'), started from
        the current parameters (e.g. after calibrate_bootstrap). smoothness adds
        smoothness * sum_j sum_p ((p_j - p_{j-1}) / range_p)^2 over the time-dependent parameters (0 : none).
        """
        self._own_arrays()
        all_bounds = dict(self.PARAMS)
        all_bounds.update(bounds or {})
        n = len(self.times)
        layout = [(name, None) for name in global_free] + [(name, j) for name in free for j in range(n)]
        lower = np.array([all_bounds[name][0] for name, _ in layout])
        upper = np.array([all_bounds[name][1] for name, _ in layout])

        def write(x):
            values = lower + np.clip(x, 0.0, 1.0) * (upper - lower)
            for (name, j), value in zip(layout, values):
                if j is None:
                    setattr(self, name, value)
                else:
                    getattr(self, name)[j] = value

        def objective(x):
            write(x)
            value = calibration_loss.loss(self, 'vega_iv_mse')
            if smoothness > 0:
                for name in free:
                    span = all_bounds[name][1] - all_bounds[name][0]
                    value += smoothness * float(np.sum((np.diff(getattr(self, name)) / span)**2))
            return value

        x0 = []
        for name, j in layout:
            value = getattr(self, name) if j is None else getattr(self, name)[j]
            x0.append((np.clip(value, all_bounds[name][0], all_bounds[name][1]) - all_bounds[name][0])
                      / (all_bounds[name][1] - all_bounds[name][0]))
        start = time.time()
        res = opt.minimize(objective, np.array(x0), method='L-BFGS-B', bounds=[(0.0, 1.0)] * len(layout),
                           options={'ftol': 1e-12, 'maxiter': maxiter})
        write(res.x)
        if verbose:
            fit = calibration_loss.loss(self, 'vega_iv_mse')
            print(f"Global calibration of {len(layout)} parameters : {res.nfev} evaluations, {time.time() - start:.1f}s,"
                  f" RMSE {100 * np.sqrt(fit):.3f} vol pts ({'converged' if res.success else res.message})")
        return res

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def generate_paths(self, T, M, N, seed=None, antithetic=False):
        """
        Path generation (Euler, full truncation), with the parameters of the interval containing each step.
        The drift over each step is ln(F(t2) / F(t1)), so that E[S_T] = F(0,T) is matched by construction.
        """
        if antithetic and N % 2 != 0:
            raise ValueError("N must be even with antithetic variates.")
        rng = np.random.default_rng(seed)
        dt = T / M
        S = np.zeros((M + 1, N))
        v = np.zeros((M + 1, N))
        S[0] = self.S0
        v[0] = self.v0
        forwards = np.array([self.forward_price(t * dt) for t in range(M + 1)])
        segments = self.segment_index((np.arange(M) + 0.5) * dt)          # interval of the middle of each step
        n_draws = N // 2 if antithetic else N

        for t in range(M):
            j = segments[t]
            theta, xi, rho = self.theta[j], self.xi[j], self.rho[j]
            fwd_drift = np.log(forwards[t + 1] / forwards[t])
            Z1 = rng.standard_normal(n_draws)
            Z2 = rng.standard_normal(n_draws)
            if antithetic:
                Z1, Z2 = np.concatenate([Z1, -Z1]), np.concatenate([Z2, -Z2])
            Z_S = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2

            v_curr = np.maximum(v[t], 0.0)
            v[t + 1] = v[t] + self.kappa * (theta - v_curr) * dt + xi * np.sqrt(v_curr * dt) * Z1
            S[t + 1] = S[t] * np.exp(fwd_drift - 0.5 * v_curr * dt + np.sqrt(v_curr * dt) * Z_S)

        return S, v
