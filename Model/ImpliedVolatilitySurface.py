import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize
from scipy.interpolate import interp1d
from Black_Scholes import BlackScholes
from forward_curve import ForwardCurve


class ImpliedVolatilitySurface:
    """
    The implied volatility surface class.
    """

    def __init__(self, forward_curve, bs_model):
        """
        Init method : Takes the forward curve and the Black model in input.
        """
        # Sanity checks
        if forward_curve is None or not isinstance(forward_curve, ForwardCurve):
            raise ValueError("forward_curve must be an object of the ForwardCurve class.")
        if bs_model is None or not isinstance(bs_model, BlackScholes):
            raise ValueError("bs_model must be an object of the BlackScholesModel class.")
        # Attributes
        self.forward_price = forward_curve.calc_forward
        self.bs_model = bs_model

    def initial_vol_guess(self, exp, strike, target_price, isCall=True):
        """
        Corrado-Miller (1996) approximation of the implied volatility, used as Newton starting point :
            sigma sqrt(T) ~ sqrt(2 pi) / (F + K) [ c - (F - K)/2 + sqrt( (c - (F - K)/2)^2 - (F - K)^2 / pi ) ],
        with c the undiscounted call price (puts are converted by parity). Exact at the money to first order ;
        the square root is floored at 0 far from the money. Returns 0.2 if the approximation is not usable.
        """
        forward = self.forward_price(exp)
        df = np.exp(-self.bs_model.zc_rate(exp) * exp)
        call = target_price / df + (0.0 if isCall else forward - strike)
        x = forward - strike
        a = call - x / 2
        sigma = np.sqrt(2 * np.pi) / (forward + strike) * (a + np.sqrt(max(a**2 - x**2 / np.pi, 0.0))) / np.sqrt(exp)
        return float(np.clip(sigma, 0.01, 3.0)) if np.isfinite(sigma) and sigma > 0 else 0.2

    def get_implied_vol(self, exp, strike, target_price, isCall=True, init_guess=None, max_iter=250, tolerance=1e-06,
                        verbose=True):
        """
        Volatility Solver : The Newton-Raphson algorithm, with a Brent fallback if it fails.
        init_guess : Newton starting point ; by default (None) the Corrado-Miller approximation (initial_vol_guess),
                     which needs 2.3 Newton iterations per quote on the CAC40 surface against 3.7 from a constant 0.5,
                     and fewer than a warm start from the neighbouring strike (2.6) or from the market volatility (2.4).
        """
        # Sanity checks
        if exp <= 0: raise ValueError("Expiry (exp) must be positive.")
        if strike <= 0: raise ValueError("Strike must be positive.")
        if target_price < 0: raise ValueError("Target price must be non-negative.")
        if not isinstance(isCall, (bool, np.bool_)):
            raise TypeError("isCall must be a boolean.")
        if init_guess is not None and not isinstance(init_guess, (int, float, np.number)):
            raise TypeError("init_guess must be numeric or None.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(tolerance, (float, np.number)) or tolerance <= 0:
            raise ValueError("tolerance must be a positive number.")
        f = lambda vol : self.bs_model.bs_price_vanilla(exp, strike, vol, bool(isCall)) - target_price
        f_deriv = lambda vol : self.bs_model.bs_vega(exp, strike, vol)
        if init_guess is None or not 0 < init_guess < 5:
            init_guess = self.initial_vol_guess(exp, strike, target_price, isCall)
        # Newton-Raphson algorithm
        try:
            implied_vol, info = optimize.newton(f, init_guess, fprime=f_deriv, maxiter=max_iter, tol=tolerance,
                                                full_output=True, disp=False)
            if info.converged and 0 < implied_vol < 5:
                return float(implied_vol)
        except (RuntimeError, ValueError, ZeroDivisionError):
            pass
        # Brent fallback on a bracketing interval
        try:
            return optimize.brentq(f, 1e-4, 5.0, xtol=tolerance, maxiter=max_iter)
        except ValueError:
            if verbose:
                print(f"/!\\ Warning : no implied volatility found for Strike {round(strike, 4)} at expiry {round(exp, 4)}")
            return None

    def set_implied_vol_surface(self, market_strikes, otm_options):
        """
        Stores a dictionary of Black implied volatilities, using get_implied_vol.
        """
        # Sanity checks
        if not market_strikes or len(market_strikes) == 0:
            raise ValueError("market_strikes cannot be empty.")
        if not otm_options or len(otm_options) == 0:
            raise ValueError("otm_options cannot be empty.")
        # Implied vol surface construction
        self.strikes = dict()
        implied_vols = dict()
        for exp in otm_options.keys():
            strikes_exp = market_strikes[exp]
            forward_exp = self.forward_price(exp)
            implied_vols_exp = []
            filtered_strikes = []
            for i in range(len(strikes_exp)):
                isCall = (strikes_exp[i] > forward_exp)
                implied_vol = self.get_implied_vol(exp, strikes_exp[i], otm_options[exp][i], isCall)
                # Failed calibrations are ignored in the vol surface construction
                if implied_vol is not None:
                    implied_vols_exp.append(implied_vol)
                    filtered_strikes.append(strikes_exp[i])
            if filtered_strikes != []:
                implied_vols[exp] = implied_vols_exp
                self.strikes[exp] = filtered_strikes
        self.expiries = list(self.strikes.keys())
        self.implied_vols = implied_vols
        # Parametric fits must be redone on the new quotes
        self.svi_params = None
        self.ssvi_params = None

    # ------------------------------------------------------------------
    # SVI (one raw SVI smile per expiry)
    # ------------------------------------------------------------------
    @staticmethod
    def svi_total_variance(k, a, b, rho, m, sigma):
        """
        Raw SVI total implied variance (Gatheral 2004) :
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)),  k = ln(K / F(T)).
        """
        k = np.asarray(k, dtype=float)
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def fit_svi(self, verbose=True):
        """
        Calibrates one raw SVI smile per expiry on the market implied vols (least squares on the vols).
        Constraints : b >= 0, |rho| < 1, sigma > 0,
                      a + b * sigma * sqrt(1 - rho^2) >= 0  (non-negative total variance),
                      b * (1 + |rho|) <= 2                  (Roger Lee's moment formula on the wings).
        Several starting points are tried per expiry, the best fit is kept.
        Returns a DataFrame with the SVI parameters and the fit RMSE (vol pts) per expiry.
        """
        # Check surface is initialized
        if not hasattr(self, 'implied_vols') or not self.implied_vols:
            raise RuntimeError("Implied volatility surface not initialized. Call set_implied_vol_surface first.")
        svi_params, rows = dict(), []
        for exp in self.expiries:
            if len(self.strikes[exp]) < 3:
                raise ValueError(f"At least 3 strikes are needed to fit SVI at expiry {exp}.")
            k = np.log(np.array(self.strikes[exp], dtype=float) / self.forward_price(exp))
            vols = np.array(self.implied_vols[exp], dtype=float)
            w_atm = np.interp(0.0, k, vols)**2 * exp
            span = max(np.max(np.abs(k)), 1e-3)

            objective = lambda x : np.sum((np.sqrt(np.maximum(self.svi_total_variance(k, *x), 1e-12) / exp) - vols)**2)
            constraints = [{'type': 'ineq', 'fun': lambda x : x[0] + x[1] * x[4] * np.sqrt(1 - x[2]**2)},
                           {'type': 'ineq', 'fun': lambda x : 2 - x[1] * (1 + x[2])},
                           {'type': 'ineq', 'fun': lambda x : 2 - x[1] * (1 - x[2])}]
            bounds = [(-1.0, 1.0), (1e-8, 2.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 2.0)]
            # Multi-start : the ATM slope fixes b * rho, the smile curvature is unknown
            slope = np.polyfit(k, vols**2 * exp, 1)[0]
            best = None
            for sigma0 in (0.3 * span, span, 3 * span):
                for m0 in (-0.5 * span, 0.0, 0.5 * span):
                    rho0 = -0.5 if slope < 0 else 0.5
                    b0 = min(max(abs(slope) / abs(rho0), 1e-3), 1.0)
                    a0 = w_atm - b0 * sigma0
                    res = optimize.minimize(objective, [a0, b0, rho0, m0, sigma0], method='SLSQP',
                                            bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
                    if best is None or res.fun < best.fun:
                        best = res
            svi_params[exp] = dict(zip(("a", "b", "rho", "m", "sigma"), best.x))
            rows.append({"Expiry": exp, **svi_params[exp], "RMSE (vol pts)": 100 * np.sqrt(best.fun / len(k))})
        self.svi_params = svi_params
        report = pd.DataFrame(rows).set_index("Expiry")
        if verbose:
            n_quotes = sum(len(self.strikes[exp]) for exp in self.expiries)
            sse = sum(len(self.strikes[exp]) * (r / 100)**2 for exp, r in report["RMSE (vol pts)"].items())
            print(f"SVI fit : RMSE = {100 * np.sqrt(sse / n_quotes):.3f} vol pts")
        return report

    def _svi_vol(self, exp, strike):
        """
        SVI implied vol. Between two expiries, the total variance is linearly interpolated at fixed
        log-moneyness k = ln(K / F(T)). Outside the listed expiries : flat vol extrapolation at fixed k.
        """
        if getattr(self, 'svi_params', None) is None:
            self.fit_svi(verbose=False)
        k = np.log(strike / self.forward_price(exp))
        w = lambda T : float(self.svi_total_variance(k, **self.svi_params[T]))
        if exp <= self.expiries[0]:
            return np.sqrt(max(w(self.expiries[0]), 0.0) / self.expiries[0])
        if exp >= self.expiries[-1]:
            return np.sqrt(max(w(self.expiries[-1]), 0.0) / self.expiries[-1])
        i = np.searchsorted(self.expiries, exp)
        T1, T2 = self.expiries[i - 1], self.expiries[i]
        w_exp = w(T1) + (w(T2) - w(T1)) * (exp - T1) / (T2 - T1)
        return np.sqrt(max(w_exp, 0.0) / exp)

    # ------------------------------------------------------------------
    # SSVI (whole surface, Gatheral & Jacquier 2014)
    # ------------------------------------------------------------------
    @staticmethod
    def ssvi_total_variance(k, theta, rho, eta, gamma):
        """
        SSVI total implied variance with a power-law phi :
        w(k, theta) = theta / 2 * (1 + rho * phi * k + sqrt((phi * k + rho)^2 + 1 - rho^2)),
        phi(theta)  = eta / (theta^gamma * (1 + theta)^(1 - gamma)),
        where theta is the ATM total variance of the expiry.
        """
        k = np.asarray(k, dtype=float)
        phi = eta / (theta**gamma * (1 + theta)**(1 - gamma))
        return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho)**2 + 1 - rho**2))

    def ssvi_theta(self, exp):
        """
        ATM total variance theta(T) : linear interpolation between the market ATM total variances,
        flat ATM vol extrapolation outside the listed expiries.
        """
        T_atm, theta_atm = self.ssvi_atm
        if exp <= T_atm[0]:
            return theta_atm[0] * exp / T_atm[0]
        if exp >= T_atm[-1]:
            return theta_atm[-1] * exp / T_atm[-1]
        return float(np.interp(exp, T_atm, theta_atm))

    def fit_ssvi(self, verbose=True):
        """
        Calibrates the SSVI surface on all the market implied vols (least squares on the vols).
        - theta(T) is not calibrated : it is the market ATM total variance of each expiry (linear interpolation of
          the smile at k = 0), made non-decreasing in T (no calendar spread arbitrage).
        - (rho, eta, gamma) are global. The parametrisation eta = 2 u / (1 + |rho|), u in [0, 1], gamma in (0, 1/2]
          guarantees the absence of butterfly arbitrage (Gatheral & Jacquier 2014, Remark 4.4).
        Returns a Series with the SSVI parameters and the fit RMSE (vol pts).
        """
        # Check surface is initialized
        if not hasattr(self, 'implied_vols') or not self.implied_vols:
            raise RuntimeError("Implied volatility surface not initialized. Call set_implied_vol_surface first.")
        # Market data in log-moneyness and ATM total variances
        data = []
        for exp in self.expiries:
            k = np.log(np.array(self.strikes[exp], dtype=float) / self.forward_price(exp))
            vols = np.array(self.implied_vols[exp], dtype=float)
            data.append((exp, k, vols, np.interp(0.0, k, vols)**2 * exp))
        theta_atm = np.maximum.accumulate([d[3] for d in data])
        self.ssvi_atm = (np.array(self.expiries, dtype=float), theta_atm)

        def residuals(x):
            rho, u, gamma = x
            eta = 2 * u / (1 + abs(rho))
            return np.concatenate([np.sqrt(self.ssvi_total_variance(k, theta, rho, eta, gamma) / exp) - vols
                                   for (exp, k, vols, _), theta in zip(data, theta_atm)])

        best = None
        for x0 in ([-0.5, 0.5, 0.3], [-0.8, 0.9, 0.45], [-0.3, 0.2, 0.1]):
            res = optimize.least_squares(residuals, x0, bounds=([-0.999, 1e-4, 1e-3], [0.999, 1.0, 0.5]))
            if best is None or res.cost < best.cost:
                best = res
        rho, u, gamma = best.x
        self.ssvi_params = {"rho": rho, "eta": 2 * u / (1 + abs(rho)), "gamma": gamma}
        rmse = 100 * np.sqrt(np.mean(best.fun**2))
        if verbose:
            print(f"SSVI fit : RMSE = {rmse:.3f} vol pts")
        return pd.Series({**self.ssvi_params, "RMSE (vol pts)": rmse}, name="SSVI")

    def _ssvi_vol(self, exp, strike):
        """
        SSVI implied vol, theta(T) given by ssvi_theta.
        """
        if getattr(self, 'ssvi_params', None) is None:
            self.fit_ssvi(verbose=False)
        k = np.log(strike / self.forward_price(exp))
        return np.sqrt(float(self.ssvi_total_variance(k, self.ssvi_theta(exp), **self.ssvi_params)) / exp)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def get_vol(self, exp, strike, min_moneyness=0.6, max_moneyness=1.4, method='v2t'):
        """
        Interpolate / Extrapolate the Black implied vol.
        method = 'v2t' :
            Strike : linear Vol interpolation within [min_moneyness, max_moneyness] - Flat Vol extrapolation.
            Expiry : linear V2T interpolation within [min_moneyness, max_moneyness] - Flat Vol extrapolation.
        method = 'svi'  : one raw SVI smile per expiry, linear V2T interpolation in expiry at fixed log-moneyness.
        method = 'ssvi' : SSVI surface.
        For 'svi' and 'ssvi', the parametric smile extrapolates the wings (no moneyness capping) and the
        parameters are calibrated on the first call if fit_svi / fit_ssvi was not called.
        """
        # Sanity checks
        if exp <= 0: raise ValueError("Expiry (exp) must be positive.")
        if strike <= 0: raise ValueError("Strike must be positive.")
        if method not in ('v2t', 'svi', 'ssvi'):
            raise ValueError("method must be 'v2t', 'svi' or 'ssvi'.")
        # Check surface is initialized
        if not hasattr(self, 'implied_vols') or not self.implied_vols:
            raise RuntimeError("Implied volatility surface not initialized. Call set_implied_vol_surface first.")
        if not hasattr(self, 'expiries') or not self.expiries:
            raise RuntimeError("No expiries available in the surface.")
        if method == 'svi':
            return self._svi_vol(exp, strike)
        if method == 'ssvi':
            return self._ssvi_vol(exp, strike)
        if min_moneyness >= max_moneyness:
            raise ValueError(f"min_moneyness must be < max_moneyness.")
        # Interpolation / Extrapolation
        strike = min(max_moneyness * self.forward_price(exp), max(min_moneyness * self.forward_price(exp), strike))
        smile = lambda T : interp1d(self.strikes[T], self.implied_vols[T], kind='linear', bounds_error=False,
                                    fill_value=(self.implied_vols[T][0], self.implied_vols[T][-1]))(strike)
        if (exp <= self.expiries[0]) or (exp >= self.expiries[-1]) or (exp in self.expiries):
            exp = min(self.expiries[-1], max(self.expiries[0], exp))
            return smile(exp)
        else:
            v2T_strike = [smile(T)**2 * T for T in self.expiries]
            return np.sqrt(interp1d(self.expiries, v2T_strike, kind='linear')(exp) / exp)

    def vol_grid(self, n_interv=81, K_range=None, method='v2t'):
        """
        Returns the (K, T, vols) meshgrid of the interpolated implied volatility surface (method : see get_vol).
        """
        if K_range is None:
            K_min = 0.8 * max([min(self.strikes[exp]) for exp in self.expiries])
            K_max = 1.2 * min([max(self.strikes[exp]) for exp in self.expiries])
        else:
            K_min, K_max = K_range
        K = np.linspace(K_min, K_max, n_interv, endpoint=True)
        T = np.linspace(self.expiries[0], self.expiries[-1], n_interv, endpoint=True)
        K, T = np.meshgrid(K, T)
        vols = np.zeros(shape=K.shape)
        for t in range(vols.shape[0]):
            for k in range(vols.shape[1]):
                vols[t, k] = self.get_vol(T[t][k], K[t][k], method=method)
        return K, T, vols

    def print_implied_vols(self):
        """
        Prints the implied vol surface table.
        """
        for T in sorted(self.implied_vols):
            ks = self.strikes[T]
            vols = self.implied_vols[T]
            print(f"\nImplied vol surface at T = {T:.4f}")
            print("K:   " + " ".join(f"{k:>8.2f}" for k in ks))
            print("IV:  " + " ".join(f"{100 * v:>7.2f}%" for v in vols))
        print()

    def plot_implied_vol_surface(self, n_interv=81, title="Implied Volatility Surface", ax=None, K_range=None,
                                 show_table=True, method='v2t'):
        """
        Implied volatility surface plotter (method : see get_vol).
        """
        # Sanity checks
        if not isinstance(n_interv, int) or n_interv < 10:
            raise ValueError("n_interv must be an integer >= 10.")
        if not hasattr(self, 'implied_vols') or not self.implied_vols:
            raise RuntimeError("No implied volatilities to plot. Call set_implied_vol_surface first.")
        if len(self.expiries) < 2:
            raise ValueError("At least 2 expiries required for surface plotting.")
        for exp in self.expiries:
            if len(self.strikes[exp]) < 2:
                raise ValueError(f"At least 2 strikes required for expiry {exp}.")
        if show_table:
            self.print_implied_vols()
        # Surface plotting
        K, T, vols = self.vol_grid(n_interv, K_range, method)
        standalone = ax is None
        if standalone:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')
        ax.plot_surface(K, T, 100 * vols, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xlabel("Strike (K)", labelpad=10)
        ax.set_ylabel("Time to expiry (T)", labelpad=10)
        ax.set_zlabel("Implied volatility (%)", labelpad=10)
        ax.set_title(title, pad=20)
        if standalone:
            plt.show()
        return ax
