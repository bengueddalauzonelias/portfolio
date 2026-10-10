import copy
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

try:
    from .rate_pricing import vega as market_vega
except ImportError:
    from rate_pricing import vega as market_vega


# One-factor Hull-White model, fitted exactly to the initial YieldCurve :
#     dr(t) = (theta(t) - a r(t)) dt + sigma dW(t),
# written r(t) = x(t) + alpha(t) with dx = -a x dt + sigma dW, x(0) = 0 and
#     alpha(t) = f(0,t) + sigma^2 / (2 a^2) (1 - e^{-a t})^2,
# so that the model reprices every discount factor P(0,T) of the curve (Brigo & Mercurio 2006, chapter 3).
# - zero-coupon bonds P(t,T) = P(0,T)/P(0,t) exp(-B(t,T) x(t) - V(t,T)), B(t,T) = (1 - e^{-a(T-t)}) / a ;
# - caplets / floorlets as options on zero-coupon bonds (closed form) ;
# - European swaptions with Jamshidian's decomposition (closed form) ;
# - exact simulation of (x, int x) for Monte Carlo, Bermudan swaptions by Longstaff-Schwartz.
# Products are read through their schedules (Produits/rate_products.py), as in rate_pricing.py.


class HullWhite:
    """
    One-factor Hull-White model on a YieldCurve.
    """
    PARAMS = {"a": (1e-3, 0.5), "sigma": (1e-4, 0.03)}
    model = "hull_white"

    def __init__(self, curve, a=0.03, sigma=0.008):
        """
        Init method.
        - curve : YieldCurve, fitted exactly by the model
        - a     : mean reversion speed (e.g. 0.03)
        - sigma : volatility of the short rate, in rate units (0.008 = 80bp per year, a normal volatility)
        """
        if a <= 0: raise ValueError("a must be positive.")
        if sigma <= 0: raise ValueError("sigma must be positive.")
        self.curve = curve
        self.a = float(a)
        self.sigma = float(sigma)

    @property
    def params(self):
        return {"a": self.a, "sigma": self.sigma}

    def with_curve(self, curve):
        model = copy.copy(self)
        model.curve = curve
        return model

    def with_vol_shift(self, d):
        """ Copy of the model with sigma shifted by d (vega of the short rate volatility). """
        model = copy.copy(self)
        model.sigma = self.sigma + d
        return model

    # ------------------------------------------------------------------
    # Zero-coupon bonds
    # ------------------------------------------------------------------
    def B(self, t, T):
        return (1.0 - np.exp(-self.a * (np.asarray(T) - np.asarray(t)))) / self.a

    def bond_price(self, t, T, x):
        """ P(t,T) given the state x(t) (vectorised in T and x). """
        a, s = self.a, self.sigma
        B = self.B(t, T)
        convexity = B * s**2 / (2 * a**2) * (1 - np.exp(-a * t))**2 + s**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B**2
        return self.curve.discount(T) / self.curve.discount(t) * np.exp(-B * x - convexity)

    def bond_option(self, T, S, X, is_call=True):
        """ Option of expiry T on the zero-coupon bond P(T,S), strike X (closed form, value at 0). """
        P_T, P_S = self.curve.discount(T), self.curve.discount(S)
        sigma_p = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * T)) / (2 * self.a)) * self.B(T, S)
        if np.any(sigma_p <= 0):
            return np.maximum(P_S - X * P_T, 0.0) if is_call else np.maximum(X * P_T - P_S, 0.0)
        h = np.log(P_S / (P_T * X)) / sigma_p + sigma_p / 2
        if is_call:
            return P_S * norm.cdf(h) - X * P_T * norm.cdf(h - sigma_p)
        return X * P_T * norm.cdf(-h + sigma_p) - P_S * norm.cdf(-h)

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------
    def swap_value(self, swap):
        """ Swaps only depend on the curve, which the model fits exactly. """
        _, ends, accruals, notionals = swap.fixed_leg()
        annuity = np.sum(notionals * accruals * self.curve.discount(ends))
        starts, ends, accruals, notionals = swap.float_leg()
        floating = np.sum(notionals * (self.curve.discount(starts) - self.curve.discount(ends)))
        value = floating - swap.fixed_rate * annuity
        return float(value if swap.payer else -value)

    def caplet_values(self, capfloor):
        """ Caplet = (1 + K tau) put on P(fixing, payment) of strike 1 / (1 + K tau) ; floorlet = call. """
        fixings, payments, accruals, notionals = capfloor.periods()
        factor = 1 + capfloor.strike * accruals
        return notionals * factor * self.bond_option(fixings, payments, 1 / factor, is_call=not capfloor.is_cap)

    def capfloor_value(self, capfloor):
        return float(np.sum(self.caplet_values(capfloor)))

    def _swaption_cashflows(self, swaption, start):
        """ Coupon bond of the swap entered at `start` : payment times and coupons c_i = K delta_i (+1 at maturity). """
        swap = swaption.underlying(start)
        if swap.amortization != 'bullet':
            raise ValueError("Only bullet swaps are supported.")
        _, ends, accruals, _ = swap.fixed_leg()
        coupons = swaption.strike * accruals
        coupons[-1] += 1.0
        return ends, coupons

    def european_swaption_value(self, swaption):
        """
        Jamshidian : the payer swaption is a put of strike 1 on the coupon bond sum_i c_i P(T0, t_i). With x* such
        that sum_i c_i P(T0, t_i; x*) = 1, it is the sum of puts on each zero-coupon bond with strikes P(T0, t_i; x*).
        """
        T0 = swaption.expiry
        times, coupons = self._swaption_cashflows(swaption, T0)
        bond = lambda x: float(np.sum(coupons * self.bond_price(T0, times, x))) - 1.0
        x_star = brentq(bond, -2.0, 2.0, xtol=1e-14)
        strikes = self.bond_price(T0, times, x_star)
        options = self.bond_option(T0, times, strikes, is_call=not swaption.payer)
        return float(swaption.notional * np.sum(coupons * options))

    def price(self, product, **mc_kwargs):
        if product.kind == "swap":
            return self.swap_value(product)
        if product.kind == "capfloor":
            return self.capfloor_value(product)
        if product.kind == "swaption":
            if product.exercise == "european":
                return self.european_swaption_value(product)
            return self.bermudan_swaption(product, **mc_kwargs)["price"]
        raise ValueError(f"Unknown product kind '{product.kind}'.")

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    def _integrated_alpha(self, t):
        """ int_0^t alpha(s) ds = -ln P(0,t) + sigma^2 / (2 a^2) int_0^t (1 - e^{-a s})^2 ds. """
        a = self.a
        t = np.asarray(t, dtype=float)
        integral = t - 2 * (1 - np.exp(-a * t)) / a + (1 - np.exp(-2 * a * t)) / (2 * a)
        return -np.log(self.curve.discount(t)) + self.sigma**2 / (2 * a**2) * integral

    def simulate(self, times, n_paths=50_000, seed=None, antithetic=True):
        """
        Exact joint simulation of x(t) and int_0^t x on the given dates (no discretisation bias).
        Returns (x, D) of shape (len(times), n_paths) : the state and the bank account discount factor
        D(0,t) = exp(-int_0^t r) along each path.
        """
        times = np.asarray(times, dtype=float)
        if np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise ValueError("times must be positive and strictly increasing.")
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even with antithetic variates.")
        rng = np.random.default_rng(seed)
        a, s = self.a, self.sigma
        x = np.zeros(n_paths)
        integral = np.zeros(n_paths)
        X, I = np.empty((len(times), n_paths)), np.empty((len(times), n_paths))
        previous = 0.0
        for k, t in enumerate(times):
            h = t - previous
            e1, e2 = np.exp(-a * h), np.exp(-2 * a * h)
            v11 = s**2 * (1 - e2) / (2 * a)
            v22 = s**2 / a**2 * (h - 2 * (1 - e1) / a + (1 - e2) / (2 * a))
            v12 = s**2 / (2 * a**2) * (1 - e1)**2
            L = np.linalg.cholesky(np.array([[v11, v12], [v12, v22]]) + 1e-18 * np.eye(2))
            m = n_paths // 2 if antithetic else n_paths
            Z = rng.standard_normal((2, m))
            if antithetic:
                Z = np.concatenate([Z, -Z], axis=1)
            eps = L @ Z
            integral = integral + x * (1 - e1) / a + eps[1]
            x = x * e1 + eps[0]
            X[k], I[k] = x, integral
            previous = t
        D = np.exp(-I - self._integrated_alpha(times)[:, None])
        return X, D

    def bermudan_swaption(self, swaption, n_paths=50_000, seed=42, antithetic=True, basis_degree=3):
        """
        Bermudan (or European) swaption by Longstaff-Schwartz on the exact Hull-White simulation.
        At each exercise date T_k, the exercise value is the value of the remaining co-terminal swap, known in
        closed form from the state x(T_k) ; the continuation value is regressed on (1, x, ..., x^d, exercise value).
        Returns a dict with the price, its standard error and the probability of exercise at each date.
        """
        dates = swaption.exercise_dates()
        X, D = self.simulate(dates, n_paths, seed, antithetic)
        sign = 1.0 if swaption.payer else -1.0
        exercise = np.empty_like(X)
        for k, T in enumerate(dates):
            times, coupons = self._swaption_cashflows(swaption, T)
            bonds = self.bond_price(T, times[:, None], X[k][None, :])
            exercise[k] = swaption.notional * sign * (1.0 - np.sum(coupons[:, None] * bonds, axis=0))
        cash = np.maximum(exercise[-1], 0.0) * D[-1]               # discounted to 0
        when = np.where(exercise[-1] > 0, len(dates) - 1, -1)
        for k in range(len(dates) - 2, -1, -1):
            itm = exercise[k] > 0
            if itm.sum() > 2 * (basis_degree + 2):
                basis = np.column_stack([X[k][itm]**d for d in range(basis_degree + 1)] + [exercise[k][itm]])
                coefs = np.linalg.lstsq(basis, cash[itm] / D[k][itm], rcond=None)[0]
                exercise_now = exercise[k][itm] > basis @ coefs
                idx = np.where(itm)[0][exercise_now]
                cash[idx] = exercise[k][idx] * D[k][idx]
                when[idx] = k
        samples = 0.5 * (cash[:n_paths // 2] + cash[n_paths // 2:]) if antithetic else cash
        return {"price": float(np.mean(samples)), "stderr": float(np.std(samples, ddof=1) / np.sqrt(len(samples))),
                "exercise probability": dict(zip(np.round(dates, 4), [float(np.mean(when == k)) for k in range(len(dates))]))}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, market_pricer, products, fixed=None, bounds=None):
        """
        Calibrates (a, sigma) to the market prices of `products` (caps / floors / European swaptions) given by
        market_pricer (a RatePricer). The residuals are (model - market) / market vega, i.e. errors in basis points
        of normal volatility at first order. `fixed` = {"a": value} keeps a parameter constant.
        Returns a dict with the parameters and the RMSE in bp of normal vol.
        """
        fixed = dict(fixed or {})
        bounds = {**self.PARAMS, **(bounds or {})}
        names = [n for n in ("a", "sigma") if n not in fixed]
        targets = np.array([market_pricer.price(p) for p in products])
        vegas = np.array([market_vega(market_pricer, p, 1e-4) for p in products])       # per bp of normal vol

        def residuals(values):
            model = copy.copy(self)
            for name, value in {**fixed, **dict(zip(names, values))}.items():
                setattr(model, name, value)
            return np.array([(model.price(p) - t) / v for p, t, v in zip(products, targets, vegas)])

        x0 = [getattr(self, n) for n in names]
        res = least_squares(residuals, x0, bounds=([bounds[n][0] for n in names], [bounds[n][1] for n in names]),
                            x_scale='jac')
        for name, value in {**fixed, **dict(zip(names, res.x))}.items():
            setattr(self, name, float(value))
        return {**self.params, "RMSE (bp of normal vol)": float(np.sqrt(np.mean(res.fun**2))),
                "max error (bp)": float(np.max(np.abs(res.fun)))}

    def __repr__(self):
        return f"HullWhite(a={self.a:.4f}, sigma={1e4 * self.sigma:.1f}bp)"
