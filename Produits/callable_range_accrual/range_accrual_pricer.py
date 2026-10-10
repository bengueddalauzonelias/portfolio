# This script prices the callable SOFR range accrual (range_accrual.py) with three methods :
# - static replication of the non-callable note (Section 8.1) : each day in range is a strip of Bachelier digitals
#   and calls struck at L and U, Eq. (4), with an optional linear smile (skew) entering through Eq. (5) ;
# - Monte Carlo and Longstaff-Schwartz (Sections 8.2-8.3), MODEL-AGNOSTIC : the model only has to provide
#       simulate(times, n_paths, seed, antithetic) -> (states, D)   states[k] = state at times[k], D = exp(-int r)
#       bond_price(t, T, state)                                     P(t, T) given the state at t
#   as HullWhite (Fixed_Income_Models/hull_white_1F.py) does ; a two-factor model with the same interface plugs in ;
# - exact backward induction on a grid for the ONE-FACTOR Hull-White model only (benchmark of the Monte Carlo) :
#   conditionally on x(T_{j-1}), the coupon of period j is known in closed form (each fixing is a Gaussian under the
#   T_j-forward measure), so the Bermudan call only needs the state at the coupon and notice dates.
# Values are per unit notional ; multiply by note.notional. Fixings dated on or before the valuation date are taken
# from the SOFR history.
import numpy as np
import pandas as pd
from scipy.stats import norm

from usd_market import year_fraction

try:
    from Fixed_Income_Models import bachelier
except ImportError:
    from rate_pricing import bachelier


# ======================================================================================================
# Schedule in model time
# ======================================================================================================
class NoteGrid:
    """
    The schedule of a note in model time (ACT/365 from the valuation date) : for each period, its fixing times,
    weights and overnight lengths (h in model time, delta in ACT/360), the known fixings and the call data.
    """

    def __init__(self, note, valuation_date, fixings=None):
        self.note = note
        self.valuation_date = pd.Timestamp(valuation_date)
        cal = note.calendar
        self.T0 = year_fraction(valuation_date, note.issue_date)
        self.pay_times = year_fraction(valuation_date, note.coupon_dates)
        self.periods = []
        for p in note.periods:
            dates = p["fixing_dates"]
            next_days = pd.DatetimeIndex([cal.add_business_days(d, 1) for d in dates])
            days = np.asarray((next_days - dates).days, dtype=float)
            known = dates <= self.valuation_date
            rates = np.full(len(dates), np.nan)
            if known.any():
                if fixings is None:
                    raise ValueError("Past fixings needed : pass the SOFR history.")
                rates[known] = fixings.reindex(dates[known]).values
                if np.isnan(rates[known]).any():
                    raise ValueError(f"Missing SOFR fixings in period {p['number']}.")
            self.periods.append({"t": year_fraction(valuation_date, dates), "w": p["weights"], "h": days / 365,
                                 "delta": days / 360, "known": known, "rates": rates})
        self.calls = {j: year_fraction(valuation_date, d) for j, d in note.notice_dates.items()}

    def future_fixing_times(self):
        t = np.concatenate([p["t"][~p["known"]] for p in self.periods])
        return np.unique(t)


# ======================================================================================================
# Static replication (non-callable note)
# ======================================================================================================
def replication_value(note, curve, normal_vol, valuation_date, fixings=None, skew=0.0, by_period=False):
    """
    Non-callable note by static replication, Eq. (4) : per fixing, (s+L) D(L) - (s+U) D(U) + c(L) - c(U) with
    Bachelier digitals and calls on the forward overnight rate F = (P(t)/P(t+h) - 1) / delta.
    - normal_vol : normal volatility (decimal, 0.008 = 80bp), a number or a function of the fixing time t
    - skew       : slope d sigma_N / dK of a linear smile around the forward ; the digitals then include the smile
                   term of Eq. (5) and the calls use sigma_N(K) = sigma_N + skew (K - F). skew = 0 : flat smile.
    The forward of each fixing is not adjusted to the payment measure : the difference with the exact Hull-White
    grid is below 0.5bp of notional here.
    """
    grid = NoteGrid(note, valuation_date, fixings)
    s, L, U = note.spread, note.lower, note.upper
    vol = normal_vol if callable(normal_vol) else (lambda t: normal_vol * np.ones_like(t))
    values = []
    for p, T in zip(grid.periods, grid.pay_times):
        t, h, delta = np.maximum(p["t"], 0.0), p["h"], p["delta"]
        F = (curve.discount(t) / curve.discount(t + h) - 1) / delta
        F = np.where(p["known"], p["rates"], F)
        sig_atm = np.where(p["known"], 0.0, vol(np.maximum(t, 1e-8)))
        accrual = 0.0
        for K, sign in ((L, 1.0), (U, -1.0)):
            sig = np.maximum(sig_atm + skew * (K - F), 1e-10) * (~p["known"])
            sd = sig * np.sqrt(t)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(sd > 0, (F - K) / sd, np.sign(F - K) * np.inf)
            digital = norm.cdf(d) - skew * np.sqrt(t) * norm.pdf(d) * (sd > 0)     # Eq. (5), undiscounted
            call = bachelier(F, K, sig, t)
            accrual = accrual + sign * ((s + K) * digital + call)
        values.append(curve.discount(T) * np.sum(p["w"] * accrual) / 360)
    values = np.array(values)
    redemption = curve.discount(grid.pay_times[-1])
    if by_period:
        return values, redemption
    return float(np.sum(values) + redemption)


def hull_white_normal_vol(hw):
    """ Normal volatility of the overnight rate fixing at t in the one-factor Hull-White model (B(t,t+h) ~ h = days/365, delta = days/360). """
    a, sigma = hw.a, hw.sigma
    return lambda t: sigma * np.sqrt((1 - np.exp(-2 * a * t)) / (2 * a * t)) * 360 / 365        # dR/dx = B / delta


# ======================================================================================================
# Monte Carlo and Longstaff-Schwartz (any model with simulate / bond_price)
# ======================================================================================================
def _state(states, k):
    return states[k] if not isinstance(states, (tuple, list)) else tuple(s[k] for s in states)


def _intrinsic_value(model, note, grid, j, tn, x):
    """
    Regressor of the Longstaff-Schwartz : value at the notice date tn of the flows after the call date T_j
    (coupon number j) when every remaining fixing equals its forward in the model state x (no optionality).
    Only uses model.bond_price, so it works for any model.
    """
    value = 0.0
    for p, T in zip(grid.periods[j:], grid.pay_times[j:]):
        total = 0.0
        for t, w, h, delta in zip(p["t"], p["w"], p["h"], p["delta"]):
            forward = (model.bond_price(tn, t, x) / model.bond_price(tn, t + h, x) - 1) / delta
            total = total + w * note.accrual(forward)
        value = value + model.bond_price(tn, T, x) * total / 360
    return value + model.bond_price(tn, grid.pay_times[-1], x)


def monte_carlo(model, note, valuation_date, fixings=None, n_paths=20_000, seed=1, antithetic=True, degree=3,
                policy=None, band=0.02):
    """
    Monte Carlo price of the note, non-callable and callable (Longstaff-Schwartz on the issuer's call).
    Every future SOFR fixing is simulated : R = (1 / P(t, t+h) - 1) / delta from the model state at t.
    At each notice date the issuer calls if the regressed value at the notice date of the flows after the call date
    is above the value of paying par at the call date, P(t_notice, T_call). Regressors, model-agnostic : the overnight
    rate and the zero rate to maturity at the notice date (polynomials of total degree `degree`) and the intrinsic
    value of the remaining flows on the model forwards (_intrinsic_value).
    - policy : regression coefficients from a previous run (out-of-sample estimate, removes the in-sample bias)
    - band   : if given, the regression and the call are restricted to the paths whose intrinsic value is within
               `band` of par (the analogue of the in-the-money paths of the original Longstaff-Schwartz)
    Returns a dict : prices, standard errors, call probabilities by date, regression coefficients.
    """
    grid = NoteGrid(note, valuation_date, fixings)
    T = grid.pay_times
    times = np.unique(np.concatenate([grid.future_fixing_times(), T, list(grid.calls.values())]))
    times = times[times > 0]
    states, D = model.simulate(times, n_paths=n_paths, seed=seed, antithetic=antithetic)
    index = {round(t, 12): k for k, t in enumerate(times)}
    k_of = lambda t: index[round(t, 12)]

    # coupons per path, discounted to 0
    coupons = np.zeros((len(T), n_paths))
    for j, p in enumerate(grid.periods):
        total = np.zeros(n_paths)
        for t, w, h, delta, known, rate in zip(p["t"], p["w"], p["h"], p["delta"], p["known"], p["rates"]):
            if not known:
                rate = (1 / model.bond_price(t, t + h, _state(states, k_of(t))) - 1) / delta
            total = total + w * note.accrual(rate)
        coupons[j] = total / 360 * D[k_of(T[j])]
    redemption = D[k_of(T[-1])]

    def mean_se(x):
        x = 0.5 * (x[:n_paths // 2] + x[n_paths // 2:]) if antithetic else x
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

    non_callable = coupons.sum(axis=0) + redemption
    # backward induction on the call dates ; `future` = discounted flows strictly after T_j
    future = redemption.copy()
    called_at = np.full(n_paths, -1)
    coefs = {}
    for j in range(len(T) - 1, 0, -1):            # j = coupon number of T_j (1-based), loop on j = n-1 ... 1
        future = future + coupons[j]              # coupon of period j+1, paid at T_{j+1}
        if j not in grid.calls:
            continue
        tn = grid.calls[j]
        x = _state(states, k_of(tn))
        overnight = (1 / model.bond_price(tn, tn + 1 / 365, x) - 1) * 360
        zero = -np.log(model.bond_price(tn, T[-1], x)) / (T[-1] - tn)
        intrinsic = _intrinsic_value(model, note, grid, j, tn, x)
        basis = np.column_stack([overnight**a * zero**b for a in range(degree + 1) for b in range(degree + 1 - a)]
                                + [intrinsic, intrinsic**2, intrinsic * overnight])
        Dn = D[k_of(tn)]
        par = model.bond_price(tn, T[j - 1], x)
        near = np.abs(intrinsic - par) < band if band is not None else np.ones(n_paths, dtype=bool)
        if policy is None:
            coefs[j] = (np.linalg.lstsq(basis[near], future[near] / Dn[near], rcond=None)[0]
                        if near.sum() > 2 * basis.shape[1] else None)
        else:
            coefs[j] = policy[j]
        call = near & (basis @ coefs[j] > par) if coefs[j] is not None else np.zeros(n_paths, dtype=bool)
        future = np.where(call, D[k_of(T[j - 1])], future)
        called_at = np.where(call, j, called_at)
    callable_value = future + coupons[0]
    nc, nc_se = mean_se(non_callable)
    cv, cv_se = mean_se(callable_value)
    probabilities = {note.coupon_dates[j - 1].date(): float(np.mean(called_at == j)) for j in sorted(grid.calls)}
    return {"non-callable": nc, "non-callable stderr": nc_se, "callable": cv, "callable stderr": cv_se,
            "issuer call": nc - cv, "call probabilities": probabilities, "policy": coefs}


# ======================================================================================================
# Exact backward induction for the one-factor Hull-White model
# ======================================================================================================
class HullWhiteGridPricer:
    """
    Benchmark for the ONE-FACTOR Hull-White model (x = r - alpha). Between two dates s < t, under the t-forward
    measure, x(t) | x(s) is Gaussian with mean x(s) e^{-a(t-s)} - M^t(s,t) and variance
    sigma^2 (1 - e^{-2a(t-s)}) / (2a) (Brigo & Mercurio 2006, Eq. 3.39), and
        E[D(s,t) f(x_t) | x_s] = P(s,t; x_s) E^t[f(x_t) | x_s]          (Gauss-Hermite quadrature).
    The overnight fixing R = (e^{B x + c} / P_0 ratio - 1) / delta is increasing in x, so {L <= R <= U} is an
    interval of x and the expected accrual of a fixing given x(s) is in closed form.
    """

    def __init__(self, hw, note, valuation_date, fixings=None, n_grid=301, width=7.0, n_quad=48):
        self.hw = hw
        self.note = note
        self.grid = NoteGrid(note, valuation_date, fixings)
        a, s = hw.a, hw.sigma
        T_end = self.grid.pay_times[-1]
        sd = s * np.sqrt((1 - np.exp(-2 * a * T_end)) / (2 * a))
        self.x = np.linspace(-width * sd, width * sd, n_grid)
        self.nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
        self.weights = weights / np.sqrt(2 * np.pi)

    # --- transition ---------------------------------------------------------------------------------
    def _moments(self, s, t, T, xs):
        """ Mean and variance of x(t) given x(s) = xs under the T-forward measure (s <= t <= T). """
        a, sig = self.hw.a, self.hw.sigma
        h = t - s
        M = sig**2 / a**2 * (1 - np.exp(-a * h)) - sig**2 / (2 * a**2) * (np.exp(-a * (T - t)) - np.exp(-a * (T + t - 2 * s)))
        return xs * np.exp(-a * h) - M, sig**2 * (1 - np.exp(-2 * a * h)) / (2 * a)

    def _bond(self, s, t, xs):
        return self.hw.bond_price(s, t, xs) if t > s else np.ones_like(xs)

    def _expect(self, values, s, t, xs):
        """ E^t[V(x_t) | x_s = xs] with V given on self.x (linear interpolation, flat extrapolation). """
        mean, var = self._moments(s, t, t, xs)
        points = mean[:, None] + np.sqrt(var) * self.nodes[None, :]
        return np.interp(points, self.x, values) @ self.weights

    # --- coupons ------------------------------------------------------------------------------------
    def expected_coupon(self, j, s, xs):
        """ E^{T_j}[Cpn_j | x(s) = xs] per unit notional, for s <= first unknown fixing of period j (0-based j). """
        hw, note, p, Tj = self.hw, self.note, self.grid.periods[j], self.grid.pay_times[j]
        xs = np.atleast_1d(xs)
        total = np.zeros_like(xs)
        for t, w, h, delta, known, rate in zip(p["t"], p["w"], p["h"], p["delta"], p["known"], p["rates"]):
            if known:
                total = total + w * note.accrual(rate)
                continue
            if t < s - 1e-12:
                raise ValueError("Fixing before the conditioning date.")
            B = float(hw.B(t, t + h))
            c = -np.log(hw.bond_price(t, t + h, 0.0))                     # 1 / P(t, t+h; x) = e^{B x + c}
            x_of = lambda K: (np.log1p(K * delta) - c) / B
            lo, hi = x_of(note.lower), x_of(note.upper)
            m, v = self._moments(s, t, Tj, xs)
            sd = np.sqrt(max(v, 1e-300))
            prob = norm.cdf((hi - m) / sd) - norm.cdf((lo - m) / sd)
            m2 = m + B * v
            exp_part = np.exp(c + B * m + 0.5 * B**2 * v) * (norm.cdf((hi - m2) / sd) - norm.cdf((lo - m2) / sd))
            total = total + w * (exp_part / delta + (note.spread - 1 / delta) * prob)
        return total / 360

    # --- pricing ------------------------------------------------------------------------------------
    def price(self, callable=True):
        """
        Backward induction on the coupon and notice dates. Returns the value per unit notional at the valuation
        date and, for each call date, the interval of overnight SOFR (at the notice date) where the issuer calls.
        """
        hw, x, g = self.hw, self.x, self.grid
        T = np.concatenate([[g.T0], g.pay_times])
        n = len(g.pay_times)
        cont = np.zeros_like(x)                  # value at T_j of the flows after T_j (given not called)
        boundaries = {}
        for j in range(n, 0, -1):                # period j : (T_{j-1}, T_j]
            s, Tj = max(T[j - 1], 0.0), T[j]
            redemption = 1.0 if j == n else 0.0
            if callable and j in g.calls:
                tn = g.calls[j]
                hold = self._bond(tn, Tj, x) * self._expect(cont, tn, Tj, x)
                par = self._bond(tn, Tj, x)
                at_notice = np.minimum(hold, par)
                calls = hold > par
                if calls.any():
                    r = (1 / hw.bond_price(tn, tn + 1 / 365, x) - 1) * 360
                    boundaries[self.note.coupon_dates[j - 1].date()] = (float(r[calls].min()), float(r[calls].max()))
                after = self._bond(s, tn, x) * self._expect(at_notice, s, tn, x)
            else:
                after = self._bond(s, Tj, x) * (self._expect(cont, s, Tj, x) + redemption)
            cont = after + self._bond(s, Tj, x) * self.expected_coupon(j - 1, s, x)
        # from the valuation date (x = 0) to the issue date
        if g.T0 > 0:
            return float(hw.curve.discount(g.T0) * self._expect(cont, 0.0, g.T0, np.zeros(1))[0]), boundaries
        return float(np.interp(0.0, x, cont)), boundaries
