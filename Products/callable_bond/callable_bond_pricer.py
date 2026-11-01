# Callable fixed-rate bonds with issuer default risk, under a one-factor Hull-White short rate and a deterministic
# default intensity (reduced-form credit model, Duffie & Singleton 1999, Lando 1998), for European, Bermudan and
# American calls.
# - CreditCurve         : piecewise constant hazard rate lambda(t), survival Q(0,t) = exp(-int_0^t lambda),
#                         bootstrapped from CDS par spreads ;
# - CallableBond        : term sheet in year fractions (coupons, call protection, call price schedule) ;
# - DatedCallableBond   : term sheet in calendar dates (ACT/365F model time, ACT/ACT ICMA accrual), seasoned bonds
#                         with accrued interest at the valuation date, clean and dirty prices ;
# - CallableBondPDE     : backward Crank-Nicolson solver of the pricing PDE in the Hull-White state x, written
#                         r(t) = x(t) + alpha(t) (see Fixed_Income_Models/hull_white_1F.py) :
#                             V_t - a x V_x + sigma^2 / 2 V_xx - (x + alpha(t) + lambda(t)) V + lambda(t) R N = 0,
#                         with coupons added on payment dates and V = min(V, call price + accrued) on exercise dates ;
# - CallableBondTree    : Hull-White trinomial tree (Hull & White 1994) on the same time grid as the PDE, the drift
#                         alpha fitted to the curve by forward induction of Arrow-Debreu prices ;
# - CallableBondLSM     : Monte Carlo on the exact Hull-White simulation with a Longstaff-Schwartz call policy,
#                         estimated on one set of paths and applied to an independent one ;
# - closed forms used as benchmarks : straight risky bond, European callable bond with zero recovery (Jamshidian) ;
# - OAS, effective risk, yield measures (to maturity, to each call date, to worst), key-rate durations, call
#   probabilities, expected life, and a one-call pricing report.
# The three pricers share one interface (price(bond, callable), with_model, with_credit), so the risk and spread
# functions apply to all of them. Times are year fractions from the valuation date ; amounts are in currency
# (notional N).
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", ".."))           # Projets : Fixed_Income_Models

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.optimize import brentq


EXERCISE_STYLES = ("european", "bermudan", "american")


# ======================================================================================================
# Credit curve
# ======================================================================================================
class CreditCurve:
    """
    Piecewise constant hazard rate : lambda(t) = hazards[i] on (times[i-1], times[i]], flat beyond the last pillar.
    The default time is independent of interest rates (deterministic intensity).
    """

    def __init__(self, times, hazards, name="credit curve", allow_negative=False):
        """
        Init method.
        - times          : ends of the hazard intervals in years (strictly increasing, positive)
        - hazards        : hazard rate on each interval (0.02 = 2% per year)
        - allow_negative : accept negative rates (a flat curve used as a pure discount spread, e.g. an OAS)
        """
        self.times = np.atleast_1d(np.asarray(times, dtype=float))
        self.hazards = np.atleast_1d(np.asarray(hazards, dtype=float))
        if len(self.times) != len(self.hazards): raise ValueError("times and hazards must have the same length.")
        if np.any(np.diff(self.times) <= 0) or self.times[0] <= 0:
            raise ValueError("times must be positive and strictly increasing.")
        if np.any(self.hazards < 0) and not allow_negative: raise ValueError("hazard rates must be non-negative.")
        self.name = name
        self._cum = np.concatenate([[0.0], np.cumsum(self.hazards * np.diff(np.concatenate([[0.0], self.times])))])

    @classmethod
    def flat(cls, hazard, name=None, allow_negative=False):
        return cls([1.0], [hazard], name or f"flat {1e4 * hazard:.0f}bp", allow_negative)

    def hazard(self, t):
        idx = np.minimum(np.searchsorted(self.times, np.asarray(t, dtype=float), side='left'), len(self.times) - 1)
        return self.hazards[idx]

    def integrated_hazard(self, t):
        """ Lambda(t) = int_0^t lambda(s) ds. """
        t = np.asarray(t, dtype=float)
        idx = np.minimum(np.searchsorted(self.times, t, side='left'), len(self.times) - 1)
        start = np.concatenate([[0.0], self.times])[idx]
        return self._cum[idx] + self.hazards[idx] * (t - start)

    def survival(self, t):
        """ Q(0,t) = P(default time > t). """
        out = np.exp(-self.integrated_hazard(t))
        return float(out) if np.ndim(out) == 0 else out

    def shifted(self, bp):
        """ Parallel shift of every hazard rate by bp basis points (CS01). """
        return CreditCurve(self.times, np.maximum(self.hazards + bp * 1e-4, 0.0), f"{self.name} {bp:+g}bp")

    # ------------------------------------------------------------------
    # CDS
    # ------------------------------------------------------------------
    def cds_legs(self, curve, maturity, recovery=0.4, frequency=4, steps_per_year=52):
        """
        Legs of a CDS starting today, per unit notional :
        - risky annuity  : sum_i delta_i P(t_i) Q(t_i) + accrual on default (half a period on average) ;
        - protection leg : (1 - R) int_0^T P(t) dQ(t), on a weekly grid with default at mid-step.
        """
        pay = YieldCurveSchedule.times(maturity, frequency)
        deltas = np.diff(np.concatenate([[0.0], pay]))
        Q = self.survival(pay)
        Q_prev = self.survival(pay - deltas)
        annuity = np.sum(deltas * curve.discount(pay) * (Q + 0.5 * (Q_prev - Q)))
        grid = np.linspace(0.0, maturity, max(int(np.ceil(maturity * steps_per_year)), 1) + 1)
        dQ = -np.diff(self.survival(grid))
        protection = (1 - recovery) * np.sum(curve.discount(0.5 * (grid[1:] + grid[:-1])) * dQ)
        return float(annuity), float(protection)

    def cds_par_spread(self, curve, maturity, recovery=0.4, frequency=4):
        annuity, protection = self.cds_legs(curve, maturity, recovery, frequency)
        return protection / annuity

    @classmethod
    def bootstrap_cds(cls, curve, maturities, spreads, recovery=0.4, frequency=4, name="CDS bootstrap"):
        """
        Sequential bootstrap : the hazard rate of (T_{i-1}, T_i] reprices the CDS of maturity T_i at par, the
        previous intervals being fixed.
        """
        maturities, spreads = np.asarray(maturities, dtype=float), np.asarray(spreads, dtype=float)
        hazards = []
        for k, (T, s) in enumerate(zip(maturities, spreads)):
            def error(h):
                trial = cls(maturities[:k + 1], hazards + [h])
                return trial.cds_par_spread(curve, T, recovery, frequency) - s
            hazards.append(brentq(error, 0.0, 5.0, xtol=1e-14))
        return cls(maturities, hazards, name)

    def __repr__(self):
        return f"CreditCurve({self.name!r}, {len(self.times)} intervals to {self.times[-1]:g}y)"


class YieldCurveSchedule:
    @staticmethod
    def times(maturity, frequency=1):
        """ Payment times every 1/frequency year until maturity, short first period if needed. """
        step = 1.0 / frequency
        n = int(np.ceil(maturity * frequency - 1e-9))
        times = maturity - step * np.arange(n)[::-1]
        return times[times > 1e-9]


# ======================================================================================================
# Product
# ======================================================================================================
class CallableBond:
    """
    Fixed-rate bullet bond that the issuer can redeem early at the call price (plus accrued interest).
    - european : one call date (the first call date) ;
    - bermudan : every coupon date from the first call date to the last but one coupon date (or every
                 1 / call_frequency year from the first call date) ;
    - american : any time from the first call date to maturity, at the call price plus accrued interest.
    The call price schedule is a step function {from time: price in % of notional}, e.g. {3: 102, 5: 101, 7: 100}.
    On a call date that is also a coupon date, the holder receives the coupon and the call price.
    """
    kind = "callable_bond"

    def __init__(self, maturity, coupon, notional=100.0, frequency=1, exercise='bermudan', first_call=None,
                 call_schedule=None, last_call=None, call_frequency=None):
        """
        Init method.
        - maturity       : final maturity in years
        - coupon         : annual coupon rate (0.045 = 4.5%), paid frequency times a year
        - notional       : redemption amount N
        - exercise       : 'european', 'bermudan' or 'american'
        - first_call     : end of the call protection (default : 1 year before maturity)
        - call_schedule  : {from time: call price in % of notional} (default : par from the first call date)
        - last_call      : last call date (default : maturity)
        - call_frequency : Bermudan call dates every 1 / call_frequency year (default : the coupon dates)
        """
        if exercise not in EXERCISE_STYLES: raise ValueError(f"exercise must be one of {EXERCISE_STYLES}.")
        if maturity <= 0: raise ValueError("maturity must be positive.")
        self.maturity = float(maturity)
        self.coupon = float(coupon)
        self.notional = float(notional)
        self.frequency = frequency
        self.exercise = exercise
        self.first_call = float(maturity - 1.0 if first_call is None else first_call)
        if not 0 < self.first_call <= self.maturity: raise ValueError("first_call must be in (0, maturity].")
        self.call_schedule = dict(sorted((call_schedule or {self.first_call: 100.0}).items()))
        self.last_call = float(maturity if last_call is None else last_call)
        self.call_frequency = call_frequency

    def with_exercise(self, exercise):
        bond = copy.copy(self)
        bond.exercise = exercise
        if exercise not in EXERCISE_STYLES: raise ValueError(f"exercise must be one of {EXERCISE_STYLES}.")
        return bond

    def with_call_frequency(self, call_frequency):
        bond = copy.copy(self)
        bond.call_frequency = call_frequency
        return bond

    def with_call_schedule(self, call_schedule):
        bond = copy.copy(self)
        bond.call_schedule = dict(sorted(call_schedule.items()))
        return bond

    # ------------------------------------------------------------------
    # Cash flows
    # ------------------------------------------------------------------
    def coupon_times(self):
        return YieldCurveSchedule.times(self.maturity, self.frequency)

    def cashflows(self):
        """ (payment times, amounts) of the straight bond : coupons, plus the notional at maturity. """
        times = self.coupon_times()
        accruals = np.diff(np.concatenate([[0.0], times]))
        amounts = self.notional * self.coupon * accruals
        amounts[-1] += self.notional
        return times, amounts

    def coupon_amounts(self):
        times = self.coupon_times()
        return times, self.notional * self.coupon * np.diff(np.concatenate([[0.0], times]))

    def accrued(self, t):
        """ Accrued interest at t (zero on coupon dates, after payment), linear in time. """
        times = self.coupon_times()
        t = np.asarray(t, dtype=float)
        idx = np.minimum(np.searchsorted(times, t, side='right'), len(times) - 1)
        start = np.concatenate([[0.0], times])[idx]
        end = times[idx]
        out = self.notional * self.coupon * (t - start)
        return np.where(np.isclose(t, end) | np.isclose(t, start), 0.0, out)

    def call_price(self, t):
        """ Clean call price in currency at t (np.inf outside the call period). """
        steps = np.array(list(self.call_schedule.keys()))
        prices = np.array(list(self.call_schedule.values()))
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(steps, t + 1e-10, side='right') - 1
        price = np.where(idx >= 0, prices[np.maximum(idx, 0)], prices[0]) * self.notional / 100
        return np.where((t >= self.first_call - 1e-10) & (t <= self.last_call + 1e-10), price, np.inf)

    def exercise_dates(self):
        """ Discrete call dates (European, Bermudan) or (start, end) of the call period (American). """
        if self.exercise == 'european':
            return np.array([self.first_call])
        if self.exercise == 'bermudan':
            if self.call_frequency is None:
                times = self.coupon_times()
            else:
                n = int(np.floor((self.maturity - self.first_call) * self.call_frequency + 1e-9))
                times = self.first_call + np.arange(n + 1) / self.call_frequency
            return times[(times >= self.first_call - 1e-10) & (times < self.maturity - 1e-10)
                         & (times <= self.last_call + 1e-10)]
        return np.array([self.first_call, self.last_call])

    def __repr__(self):
        calls = ", ".join(f"{t:g}y: {p:g}" for t, p in self.call_schedule.items())
        return (f"CallableBond({self.exercise}, {self.maturity:g}y, coupon {100 * self.coupon:.3g}%, "
                f"N = {self.notional:g}, call from {self.first_call:g}y [{calls}])")


# ======================================================================================================
# Closed forms (benchmarks)
# ======================================================================================================
def straight_bond_value(curve, bond, credit=None, recovery=0.4, steps_per_year=365):
    """
    Risky bullet bond with recovery of par : sum_i c_i P(t_i) Q(t_i) + R N int_0^T P(t) (-dQ(t)).
    It only depends on the discount curve and the survival curve (default independent of rates).
    """
    times, amounts = bond.cashflows()
    if credit is None:
        return float(np.sum(amounts * curve.discount(times)))
    value = np.sum(amounts * curve.discount(times) * credit.survival(times))
    grid = np.linspace(0.0, bond.maturity, int(np.ceil(bond.maturity * steps_per_year)) + 1)
    dQ = -np.diff(credit.survival(grid))
    value += recovery * bond.notional * np.sum(curve.discount(0.5 * (grid[1:] + grid[:-1])) * dQ)
    return float(value)


def european_callable_zero_recovery(model, bond, credit=None):
    """
    European callable bond with ZERO recovery, closed form (Jamshidian 1989) : the call is a call option of expiry
    T0 on the risky coupon bond sum_i c_i Q(t_i) / Q(T0) P(T0, t_i), struck at the call price + accrued, held only if
    the issuer survives to T0. With x* such that the bond equals the strike, it is a sum of zero-coupon bond calls.
    Returns (callable bond, straight bond, call option).
    """
    surv = (lambda t: np.ones_like(np.asarray(t, dtype=float))) if credit is None else credit.survival
    times, amounts = bond.cashflows()
    straight = float(np.sum(amounts * model.curve.discount(times) * surv(times)))
    T0 = bond.first_call
    strike = float(bond.call_price(T0) + bond.accrued(T0))
    after = times > T0 + 1e-10
    t_i, c_i = times[after], amounts[after] * surv(times[after]) / surv(T0)
    risky = lambda x: float(np.sum(c_i * model.bond_price(T0, t_i, x))) - strike
    x_star = brentq(risky, -1.0, 1.0, xtol=1e-15)
    strikes = model.bond_price(T0, t_i, x_star)
    option = float(surv(T0) * np.sum(c_i * model.bond_option(T0, t_i, strikes, is_call=True)))
    return straight - option, straight, option


# ======================================================================================================
# PDE pricer
# ======================================================================================================
class CallableBondPDE:
    """
    Finite difference pricer of callable bonds under Hull-White + deterministic default intensity.

    The PDE is solved backward in the state x on a uniform grid of +/- n_sd standard deviations of x(maturity).
    Since alpha(t) and lambda(t) do not depend on x, V = exp(-int (alpha + lambda)) U where U solves the
    time-homogeneous PDE U_t - a x U_x + sigma^2 / 2 U_xx - x U = 0 : its operator is constant, so each step is one
    tridiagonal solve (theta scheme, Crank-Nicolson with Rannacher implicit steps after each exercise date), and the
    deterministic discounting is exact. Over a step [t, t + dt], the recovery R N is paid at mid-step with
    probability 1 - Q(t + dt) / Q(t), discounted with the Hull-White zero-coupon bond P(t, t + dt / 2 ; x).
    """

    def __init__(self, model, credit=None, recovery=0.4, n_x=401, steps_per_year=100, n_sd=6.0, rannacher=2):
        """
        Init method.
        - model          : HullWhite (Fixed_Income_Models), fitted to the risk-free discount curve
        - credit         : CreditCurve of the issuer (None : default-free)
        - recovery       : recovery rate R, paid as R N at default (recovery of par)
        - n_x            : number of space nodes (odd : x = 0 is a node)
        - steps_per_year : maximal time step 1 / steps_per_year ; every cash flow and call date is a node
        - n_sd           : half-width of the grid in standard deviations of x(maturity)
        - rannacher      : number of fully implicit steps after each discrete exercise date
        """
        self.model = model
        self.credit = credit
        self.recovery = float(recovery)
        self.n_x = int(n_x) + (1 - int(n_x) % 2)
        self.steps_per_year = steps_per_year
        self.n_sd = n_sd
        self.rannacher = rannacher

    def with_model(self, model):
        pricer = copy.copy(self)
        pricer.model = model
        return pricer

    def with_credit(self, credit, recovery=None):
        pricer = copy.copy(self)
        pricer.credit = credit
        pricer.recovery = self.recovery if recovery is None else float(recovery)
        return pricer

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    def space_grid(self, T):
        a, s = self.model.a, self.model.sigma
        sd = s * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a))
        x_max = self.n_sd * sd
        return np.linspace(-x_max, x_max, self.n_x)

    def time_grid(self, bond):
        """ Union of the key dates, each interval cut in equal steps of at most 1 / steps_per_year. """
        key = [0.0, bond.maturity, *bond.coupon_times()]
        if bond.exercise != 'american':
            key += list(bond.exercise_dates())
        else:
            key += [bond.first_call, bond.last_call]
        key += [t for t in bond.call_schedule if 0 < t < bond.maturity]
        if self.credit is not None:
            key += [t for t in self.credit.times if t < bond.maturity]
        key = np.unique(np.round([t for t in key if 0 <= t <= bond.maturity], 12))
        grid = [0.0]
        for t0, t1 in zip(key[:-1], key[1:]):
            n = max(int(np.ceil((t1 - t0) * self.steps_per_year - 1e-9)), 1)
            grid += list(np.linspace(t0, t1, n + 1)[1:])
        return np.array(grid)

    def _operator(self, x):
        """ Tridiagonal operator of -a x U_x + sigma^2 / 2 U_xx - x U, upwind and one-sided at the boundaries. """
        a, s = self.model.a, self.model.sigma
        dx = x[1] - x[0]
        mu = -a * x
        lower = -mu / (2 * dx) + s**2 / (2 * dx**2)
        diag = -s**2 / dx**2 - x
        upper = mu / (2 * dx) + s**2 / (2 * dx**2)
        lower[0] = upper[-1] = 0.0
        diag[0], upper[0] = -mu[0] / dx - x[0], mu[0] / dx          # mu > 0 at x < 0 : forward difference
        diag[-1], lower[-1] = mu[-1] / dx - x[-1], -mu[-1] / dx      # mu < 0 at x > 0 : backward difference
        return lower, diag, upper

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def solve(self, bond, callable=True):
        """
        Backward induction. Returns a dict with the dirty price at x(0) = 0, the grid x, the values at t = 0 on the
        grid (as a function of the initial short rate r(0) = x + alpha(0)), and the exercise boundary : on each
        exercise date, the largest x where the issuer calls (it calls when rates are low).
        """
        model, credit, R, N = self.model, self.credit, self.recovery, bond.notional
        x = self.space_grid(bond.maturity)
        times = self.time_grid(bond)
        lower, diag, upper = self._operator(x)

        coupon_t, coupon_c = bond.coupon_amounts()
        coupons = {round(t, 10): c for t, c in zip(coupon_t, coupon_c)}
        if not callable:
            exercise_times = set()
        elif bond.exercise == 'american':
            exercise_times = {round(t, 10) for t in times
                              if bond.first_call - 1e-10 <= t <= min(bond.last_call, bond.maturity - 1e-10) + 1e-10}
        else:
            exercise_times = {round(t, 10) for t in bond.exercise_dates()}

        int_alpha = model._integrated_alpha(np.maximum(times, 1e-14))
        int_alpha[times == 0] = 0.0
        survival = np.ones_like(times) if credit is None else credit.survival(times)

        cache = {}

        def step_matrices(dt, theta):
            key = (round(dt, 12), theta)
            if key not in cache:
                ab = np.vstack([np.concatenate([[0.0], -theta * dt * upper[:-1]]),
                                1 - theta * dt * diag,
                                np.concatenate([-theta * dt * lower[1:], [0.0]])])
                cache[key] = ab
            return cache[key]

        def explicit(U, dt, theta):
            w = (1 - theta) * dt
            out = U + w * diag * U
            out[1:] += w * lower[1:] * U[:-1]
            out[:-1] += w * upper[:-1] * U[1:]
            return out

        V = np.full_like(x, N)                     # redemption at maturity
        boundary = {}
        implicit_left = 0
        for k in range(len(times) - 1, -1, -1):
            t = times[k]
            key = round(t, 10)
            if key in exercise_times and t > 0:
                strike = float(bond.call_price(t) + bond.accrued(t))
                called = V >= strike
                boundary[t] = x[called].max() if called.any() else np.nan
                V = np.minimum(V, strike)
                if bond.exercise != 'american':
                    implicit_left = self.rannacher
            if key in coupons:
                V = V + coupons[key]
            if k == 0:
                break
            dt = t - times[k - 1]
            theta = 1.0 if implicit_left > 0 else 0.5
            implicit_left = max(implicit_left - 1, 0)
            U = solve_banded((1, 1), step_matrices(dt, theta), explicit(V, dt, theta))
            discount = np.exp(-(int_alpha[k] - int_alpha[k - 1]))
            surv = survival[k] / survival[k - 1]
            V = discount * surv * U
            if credit is not None and R > 0:
                V = V + R * N * (1 - surv) * model.bond_price(times[k - 1], times[k - 1] + 0.5 * dt, x)

        mid = self.n_x // 2
        alpha0 = float(model.curve.instantaneous_forward(1e-8))
        return {"price": float(V[mid]), "x": x, "short rate": x + alpha0, "values": V,
                "boundary": dict(sorted(boundary.items())), "time steps": len(times) - 1}

    def price(self, bond, callable=True):
        return self.solve(bond, callable)["price"]

    def short_rate_boundary(self, bond):
        """ Exercise boundary as a critical short rate r*(t) = x*(t) + alpha(t) : the issuer calls when r < r*. """
        b = self.solve(bond)["boundary"]
        t = np.array(list(b.keys()))
        a, s = self.model.a, self.model.sigma
        alpha = self.model.curve.instantaneous_forward(t) + s**2 / (2 * a**2) * (1 - np.exp(-a * t))**2
        return t, np.array(list(b.values())) + alpha


# ======================================================================================================
# Spreads and risk
# ======================================================================================================
def option_adjusted_spread(model, bond, market_price, callable=True, **pde_kwargs):
    """
    OAS : constant spread s over the risk-free Hull-White short rate such that the model reprices the dirty market
    price (discounting at r + s, no explicit default : a flat intensity s with zero recovery). With callable=False,
    it is the Z-spread of the straight bond.
    """
    def error(s):
        pricer = CallableBondPDE(model, CreditCurve.flat(s, allow_negative=True), recovery=0.0, **pde_kwargs)
        return pricer.price(bond, callable) - market_price
    return brentq(error, -0.05, 0.30, xtol=1e-10)


def effective_risk(pricer, bond, bp=10.0, callable=True):
    """
    Effective duration and convexity under parallel shifts of the risk-free curve (Hull-White refitted to each
    shifted curve, sigma and a unchanged), vega per bp of sigma, and CS01 for a 1bp parallel shift of the hazard.
    """
    P0 = pricer.price(bond, callable)
    up = pricer.with_model(pricer.model.with_curve(pricer.model.curve.shifted(bp))).price(bond, callable)
    down = pricer.with_model(pricer.model.with_curve(pricer.model.curve.shifted(-bp))).price(bond, callable)
    h = bp * 1e-4
    vega = pricer.with_model(pricer.model.with_vol_shift(1e-4)).price(bond, callable) - P0
    cs01 = (pricer.with_credit(pricer.credit.shifted(1.0)).price(bond, callable) - P0) if pricer.credit else np.nan
    return {"price": P0, "effective duration": (down - up) / (2 * h * P0),
            "effective convexity": (up + down - 2 * P0) / (h**2 * P0),
            "DV01": (down - up) / (2 * bp), "vega (per bp of sigma)": vega, "CS01": cs01}


# ======================================================================================================
# Term sheet in dates
# ======================================================================================================
def year_fraction(valuation_date, dates):
    """ ACT/365F model time from the valuation date. """
    days = (pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates))) - pd.Timestamp(valuation_date)).days
    return np.asarray(days, dtype=float) / 365.0


class DatedCallableBond(CallableBond):
    """
    Callable bullet bond described by its dates. Coupon dates roll back from the maturity date every 12 / frequency
    months (unadjusted) ; every regular coupon pays N c / frequency (ACT/ACT ICMA), and the accrued interest is
    pro rata of the period in model time. A bond issued before the valuation date pays a full first coupon : its
    dirty price includes the accrued interest AI(0), and clean = dirty - AI(0).
    The call schedule is {date: price in % of notional} ; the call protection ends at first_call_date.
    """

    def __init__(self, valuation_date, maturity_date, coupon, first_call_date, notional=100.0, frequency=1,
                 exercise='bermudan', call_schedule=None, last_call_date=None, call_frequency=None, issue_date=None,
                 name="callable bond"):
        """
        Init method.
        - valuation_date  : pricing date (model time 0)
        - maturity_date   : final redemption date
        - coupon          : annual coupon rate (0.0425 = 4.25%), paid frequency times a year
        - first_call_date : end of the call protection
        - call_schedule   : {date: clean call price in % of notional} (default : par from the first call date)
        - last_call_date  : last call date (default : maturity)
        - call_frequency  : Bermudan call dates every 1 / call_frequency year (default : the coupon dates)
        - issue_date      : if after the regular start of the first period, the first coupon is a short stub
        """
        self.valuation_date = pd.Timestamp(valuation_date)
        self.maturity_date = pd.Timestamp(maturity_date)
        self.first_call_date = pd.Timestamp(first_call_date)
        self.issue_date = None if issue_date is None else pd.Timestamp(issue_date)
        self.name = name
        if self.maturity_date <= self.valuation_date: raise ValueError("The bond has matured.")
        t = lambda d: float(year_fraction(self.valuation_date, d)[0])

        # Coupon schedule, rolled back from the maturity date
        months = 12 // frequency
        dates = [self.maturity_date]
        while dates[-1] > self.valuation_date:
            dates.append(self.maturity_date - pd.DateOffset(months=months * len(dates)))
        self.coupon_dates = pd.DatetimeIndex(dates[-2::-1])                 # after the valuation date
        self.previous_coupon_date = dates[-1]                                 # start of the current period
        start = self.previous_coupon_date if self.issue_date is None else max(self.previous_coupon_date,
                                                                               self.issue_date)
        self._times = year_fraction(self.valuation_date, self.coupon_dates)
        self._starts = np.concatenate([[t(start)], self._times[:-1]])
        self._regular_starts = np.concatenate([[t(self.previous_coupon_date)], self._times[:-1]])

        if call_schedule is None:
            schedule = {t(self.first_call_date): 100.0}
        else:
            schedule = {t(d): float(p) for d, p in sorted(call_schedule.items(), key=lambda kv: pd.Timestamp(kv[0]))}
        if exercise == 'european' and t(self.first_call_date) <= 0:
            raise ValueError("The European call date has passed.")
        first_call = max(t(self.first_call_date), 1e-6)                      # already callable : from today
        last_call = None if last_call_date is None else t(last_call_date)
        super().__init__(maturity=self._times[-1], coupon=coupon, notional=notional, frequency=frequency,
                         exercise=exercise, first_call=first_call, call_schedule=schedule, last_call=last_call,
                         call_frequency=call_frequency)

    # ------------------------------------------------------------------
    # Cash flows
    # ------------------------------------------------------------------
    def coupon_times(self):
        return self._times.copy()

    def coupon_amounts(self):
        """ N c / f per regular period, pro rata for a short first stub. """
        regular = self.notional * self.coupon / self.frequency
        fraction = (self._times - self._starts) / (self._times - self._regular_starts)
        return self._times.copy(), regular * fraction

    def cashflows(self):
        times, amounts = self.coupon_amounts()
        amounts = amounts.copy()
        amounts[-1] += self.notional
        return times, amounts

    def accrued(self, t):
        """ Accrued interest at t : the running coupon pro rata of its period (zero right after a payment). """
        t = np.asarray(t, dtype=float)
        _, amounts = self.coupon_amounts()
        idx = np.minimum(np.searchsorted(self._times, t, side='right'), len(self._times) - 1)
        start, end = self._starts[idx], self._times[idx]
        out = amounts[idx] * np.clip((t - start) / (end - start), 0.0, 1.0)
        out = np.where(np.isclose(t, end) | np.isclose(t, start), 0.0, out)
        return float(out) if out.ndim == 0 else out

    @property
    def accrued_today(self):
        """ Accrued interest at the valuation date (dirty - clean). """
        return float(self.accrued(0.0)) if self._starts[0] < 0 else 0.0

    def date_of(self, t):
        return self.valuation_date + pd.to_timedelta(np.round(np.asarray(t) * 365), unit="D")

    def schedule_frame(self):
        """ Coupon schedule with the call rights of each style. """
        times, amounts = self.coupon_amounts()
        rows = []
        for d, t, c in zip(self.coupon_dates, times, amounts):
            rights = []
            for style in EXERCISE_STYLES:
                b = self.with_exercise(style)
                if style == 'american':
                    ok = b.first_call - 1e-10 <= t <= min(b.last_call, b.maturity) + 1e-10
                else:
                    ok = np.any(np.isclose(b.exercise_dates(), t))
                rights.append(style[0].upper() if ok else "")
            price = self.call_price(t)
            rows.append({"date": d.date(), "t (ACT/365)": t, "coupon": c,
                         "call price": np.nan if np.isinf(price) else float(price), "callable": "".join(rights)})
        return pd.DataFrame(rows).set_index("date")

    def __repr__(self):
        calls = ", ".join(f"{self.date_of(t).date()}: {p:g}" for t, p in self.call_schedule.items())
        return (f"DatedCallableBond({self.exercise}, maturity {self.maturity_date.date()}, coupon "
                f"{100 * self.coupon:.3g}% x{self.frequency}, N = {self.notional:g}, call [{calls}])")


# ======================================================================================================
# Exercise times shared by the backward pricers
# ======================================================================================================
def exercise_times(bond, times, callable=True):
    """ Grid times where the issuer may call : the discrete dates, or every node of the call period (American). """
    if not callable:
        return set()
    if bond.exercise == 'american':
        return {round(t, 10) for t in times
                if bond.first_call - 1e-10 <= t <= min(bond.last_call, bond.maturity - 1e-10) + 1e-10}
    return {round(t, 10) for t in bond.exercise_dates()}


# ======================================================================================================
# Trinomial tree
# ======================================================================================================
class CallableBondTree(CallableBondPDE):
    """
    Hull-White trinomial tree on the state x (dx = -a x dt + sigma dW), r = x + alpha_k on each step.

    The time grid is the PDE's (every coupon, call and credit date is a node, steps of at most 1 / steps_per_year),
    so the steps are not equal : level k+1 has its own spacing dx_{k+1} = sqrt(3 V_k), V_k the variance of x over
    step k. A node x branches to the nearest node of E[x'] = x e^{-a h} and its two neighbours, with probabilities
    matching the mean and the variance ; the levels are truncated at +/- n_sd standard deviations of x(t).
    alpha_k is solved level by level so that the Arrow-Debreu prices reprice P(0, t_{k+1}) exactly.
    Default : survival Q(t_{k+1}) / Q(t_k) on each step, recovery R N at mid-step.
    """

    def __init__(self, model, credit=None, recovery=0.4, steps_per_year=100, n_sd=7.0):
        super().__init__(model, credit, recovery, n_x=3, steps_per_year=steps_per_year, n_sd=n_sd, rannacher=0)

    def build(self, times):
        """ Levels x_k, transitions (centre index, probabilities) and the fitted alpha_k, discount factors. """
        a, s, curve = self.model.a, self.model.sigma, self.model.curve
        h = np.diff(times)
        var = s**2 * (1 - np.exp(-2 * a * h)) / (2 * a)
        sd = s * np.sqrt((1 - np.exp(-2 * a * times)) / (2 * a))
        levels, transitions, discounts = [np.zeros(1)], [], []
        Q = np.ones(1)                                         # Arrow-Debreu prices of level k
        for k in range(len(h)):
            x = levels[k]
            dx = np.sqrt(3 * var[k])
            J = max(int(np.ceil(self.n_sd * sd[k + 1] / dx)), 1)
            mean = x * np.exp(-a * h[k])
            centre = np.clip(np.rint(mean / dx).astype(int), -J + 1, J - 1)
            eta = (mean - centre * dx) / dx
            m2 = var[k] / dx**2 + eta**2
            p = np.vstack([(m2 - eta) / 2, 1 - m2, (m2 + eta) / 2])          # down, middle, up
            p = np.maximum(p, 0.0)                                          # only binds at the truncation
            p /= p.sum(axis=0)
            # alpha_k : sum_j Q_j exp(-(alpha + x_j) h) = P(0, t_{k+1})
            alpha = (np.log(np.sum(Q * np.exp(-x * h[k]))) - np.log(curve.discount(times[k + 1]))) / h[k]
            d = np.exp(-(alpha + x) * h[k])
            idx = centre + J
            Q = sum(np.bincount(idx + o, weights=Q * d * p[o + 1], minlength=2 * J + 1) for o in (-1, 0, 1))
            levels.append(np.arange(-J, J + 1) * dx)
            transitions.append((idx, p))
            discounts.append(d)
        return levels, transitions, discounts

    def solve(self, bond, callable=True):
        """ Backward induction on the tree ; same outputs as CallableBondPDE.solve (the grid at t = 0 is one node). """
        credit, R, N = self.credit, self.recovery, bond.notional
        times = self.time_grid(bond)
        levels, transitions, discounts = self.build(times)
        coupon_t, coupon_c = bond.coupon_amounts()
        coupons = {round(t, 10): c for t, c in zip(coupon_t, coupon_c)}
        calls = exercise_times(bond, times, callable)
        survival = np.ones_like(times) if credit is None else credit.survival(times)

        V = np.full(len(levels[-1]), N)
        boundary = {}
        for k in range(len(times) - 1, -1, -1):
            t, key = times[k], round(times[k], 10)
            if key in calls and t > 0:
                strike = float(bond.call_price(t) + bond.accrued(t))
                called = V >= strike
                boundary[t] = levels[k][called].max() if called.any() else np.nan
                V = np.minimum(V, strike)
            if key in coupons:
                V = V + coupons[key]
            if k == 0:
                break
            idx, p = transitions[k - 1]
            d = discounts[k - 1]
            q = survival[k] / survival[k - 1]
            V = d * q * (p[0] * V[idx - 1] + p[1] * V[idx] + p[2] * V[idx + 1])
            if credit is not None and R > 0:
                V = V + R * N * (1 - q) * np.sqrt(d)                       # recovery at mid-step
        return {"price": float(V[0]), "boundary": dict(sorted(boundary.items())), "time steps": len(times) - 1,
                "nodes (last level)": len(levels[-1])}


# ======================================================================================================
# Monte Carlo + Longstaff-Schwartz
# ======================================================================================================
class CallableBondLSM:
    """
    Callable bond by Monte Carlo on the exact Hull-White simulation (HullWhite.simulate) and Longstaff-Schwartz.

    On each call date, the issuer calls when the regressed continuation value exceeds the call price + accrued.
    Regressors : 1, x, x^2, ..., x^d and the closed-form value of the remaining straight flows (an upper bound of
    the continuation value : only the paths where it exceeds the strike can be called, and only they are regressed).
    The policy is estimated on a first set of paths and applied to an independent one (out-of-sample) : a
    sub-optimal policy for the issuer can only raise the bond price, so the out-of-sample price is an upper bound
    of the true one, up to Monte Carlo noise. The American call is approximated by call dates every
    1 / american_steps_per_year year. Default : the flows of date t_k are weighted by Q(t_k), and the recovery R N
    of (t_{k-1}, t_k] is discounted with the average of D(t_{k-1}) and D(t_k) on a grid of credit_steps_per_year.
    """

    def __init__(self, model, credit=None, recovery=0.4, n_paths=50_000, seed=2026, american_steps_per_year=12,
                 basis_degree=3, out_of_sample=True, credit_steps_per_year=12):
        self.model = model
        self.credit = credit
        self.recovery = float(recovery)
        self.n_paths = int(n_paths)
        self.seed = seed
        self.american_steps_per_year = american_steps_per_year
        self.basis_degree = basis_degree
        self.out_of_sample = out_of_sample
        self.credit_steps_per_year = credit_steps_per_year

    def with_model(self, model):
        pricer = copy.copy(self)
        pricer.model = model
        return pricer

    def with_credit(self, credit, recovery=None):
        pricer = copy.copy(self)
        pricer.credit = credit
        pricer.recovery = self.recovery if recovery is None else float(recovery)
        return pricer

    def call_dates(self, bond, callable=True):
        if not callable:
            return np.array([])
        if bond.exercise != 'american':
            return bond.exercise_dates()
        end = min(bond.last_call, bond.maturity - 1e-10)
        n = int(np.floor((end - bond.first_call) * self.american_steps_per_year + 1e-9))
        dates = bond.first_call + np.arange(n + 1) / self.american_steps_per_year
        coupons = bond.coupon_times()
        return np.unique(np.concatenate([dates, coupons[(coupons >= bond.first_call) & (coupons <= end)]]))

    def _straight_after(self, bond, t, x):
        """ Value at t, per survivor, of the straight flows after t : coupons, redemption and recovery at default. """
        times, amounts = bond.cashflows()
        after = times > t + 1e-10
        surv = 1.0 if self.credit is None else self.credit.survival(times[after]) / self.credit.survival(t)
        bonds = self.model.bond_price(t, times[after][:, None], x[None, :])
        value = np.sum((amounts[after] * surv)[:, None] * bonds, axis=0)
        if self.credit is not None and self.recovery > 0:
            grid = np.linspace(t, bond.maturity, max(int(np.ceil((bond.maturity - t) * self.credit_steps_per_year)), 1) + 1)
            dQ = -np.diff(self.credit.survival(grid)) / self.credit.survival(t)
            mid = 0.5 * (grid[1:] + grid[:-1])
            value += self.recovery * bond.notional * (dQ @ self.model.bond_price(t, mid[:, None], x[None, :]))
        return value

    def _run(self, bond, times, flows, strikes, seed, policy=None):
        model, credit, R, N = self.model, self.credit, self.recovery, bond.notional
        X, D = model.simulate(times, self.n_paths, seed, antithetic=True)
        Q = np.ones_like(times) if credit is None else credit.survival(times)
        Q_prev = np.concatenate([[1.0], Q[:-1]])
        D_prev = np.vstack([np.ones(self.n_paths), D[:-1]])
        sd = model.sigma * np.sqrt((1 - np.exp(-2 * model.a * times)) / (2 * model.a))
        fitted = {}
        value = np.zeros(self.n_paths)                     # PV at 0 of the flows after t_k under the policy
        when = np.full(self.n_paths, -1)
        for k in range(len(times) - 1, -1, -1):
            if np.isfinite(strikes[k]):
                straight = self._straight_after(bond, times[k], X[k])
                candidate = straight > strikes[k]
                basis = np.column_stack([(X[k] / sd[k])**d for d in range(self.basis_degree + 1)] + [straight])
                if policy is None:
                    coefs = None
                    if candidate.sum() > 4 * basis.shape[1]:
                        continuation = value[candidate] / (D[k][candidate] * Q[k])
                        coefs = np.linalg.lstsq(basis[candidate], continuation, rcond=None)[0]
                    fitted[k] = coefs
                else:
                    coefs = policy.get(k)
                if coefs is not None:
                    call = candidate & (basis @ coefs > strikes[k])
                    value[call] = strikes[k] * D[k][call] * Q[k]
                    when[call] = k
            value += flows[k] * D[k] * Q[k] + R * N * (Q_prev[k] - Q[k]) * 0.5 * (D_prev[k] + D[k])
        return value, when, fitted

    def solve(self, bond, callable=True):
        """
        Returns the price, its standard error, the in-sample price, the probability of a call on each date given
        survival, and the expected life (call or maturity date given survival).
        """
        coupon_t, coupon_c = bond.coupon_amounts()
        calls = self.call_dates(bond, callable)
        grid = [coupon_t, calls]
        if self.credit is not None and self.recovery > 0:
            n = int(np.ceil(bond.maturity * self.credit_steps_per_year))
            grid.append(np.linspace(0, bond.maturity, n + 1)[1:])
        times = np.unique(np.round(np.concatenate(grid), 10))
        times = times[times > 1e-10]
        flows = np.zeros(len(times))
        flows[np.searchsorted(times, np.round(coupon_t, 10))] = coupon_c
        flows[-1] += bond.notional
        strikes = np.full(len(times), np.inf)
        idx = np.searchsorted(times, np.round(calls, 10))
        strikes[idx] = bond.call_price(times[idx]) + bond.accrued(times[idx])

        value, when, policy = self._run(bond, times, flows, strikes, self.seed)
        in_sample = float(np.mean(value))
        if self.out_of_sample:
            value, when, _ = self._run(bond, times, flows, strikes, self.seed + 1, policy)
        half = self.n_paths // 2
        pairs = 0.5 * (value[:half] + value[half:])
        life = np.where(when >= 0, times[np.maximum(when, 0)], bond.maturity)
        probability = {float(t): float(np.mean(when == k)) for k, t in enumerate(times) if np.isfinite(strikes[k])}
        return {"price": float(np.mean(pairs)), "stderr": float(np.std(pairs, ddof=1) / np.sqrt(half)),
                "in-sample price": in_sample, "call probability": probability,
                "probability called": float(np.mean(when >= 0)), "expected life": float(np.mean(life)),
                "life": life}

    def price(self, bond, callable=True):
        return self.solve(bond, callable)["price"]


# ======================================================================================================
# Yields and risk
# ======================================================================================================
def bond_yield(times, flows, dirty_price, frequency=1):
    """ Yield y, compounded `frequency` times a year, such that sum_i F_i (1 + y / f)^(-f t_i) = dirty price. """
    times, flows = np.asarray(times, dtype=float), np.asarray(flows, dtype=float)
    f = lambda y: float(np.sum(flows * (1 + y / frequency)**(-frequency * times))) - dirty_price
    return brentq(f, -0.5 * frequency + 1e-6, 2.0, xtol=1e-12)


def redemption_dates(bond):
    """ Possible redemption dates for the yield to worst : the call dates (coupon dates for an American) and maturity. """
    if bond.exercise == 'american':
        coupons = bond.coupon_times()
        dates = np.concatenate([[bond.first_call],
                                coupons[(coupons > bond.first_call) & (coupons <= bond.last_call + 1e-10)]])
    else:
        dates = bond.exercise_dates()
    return np.unique(np.round(np.concatenate([dates[dates < bond.maturity - 1e-10], [bond.maturity]]), 10))


def yield_table(bond, dirty_price):
    """ Yield to each possible redemption date (call price + accrued, coupons until then) ; the minimum is the YTW. """
    times, amounts = bond.cashflows()
    rows = []
    for T in redemption_dates(bond):
        keep = times <= T + 1e-10
        t, f = times[keep], amounts[keep].copy()
        if T < bond.maturity - 1e-10:
            redemption = float(bond.call_price(T) + bond.accrued(T))
            if np.isclose(t[-1], T):
                f[-1] += redemption
            else:
                t, f = np.append(t, T), np.append(f, redemption)
        else:
            redemption = bond.notional
        rows.append({"t": T, "redemption": redemption, "yield (%)": 100 * bond_yield(t, f, dirty_price,
                                                                                     bond.frequency)})
    out = pd.DataFrame(rows)
    out["worst"] = np.where(out["yield (%)"] == out["yield (%)"].min(), "<-", "")
    return out


def key_rate_bump(curve, pillar, bp, step=1 / 52):
    """
    Curve with a triangular key-rate shift (Ho 1992) : the zero rates move by bp at the pillar, linearly less until
    the neighbouring pillars, flat beyond the first and last ones. The base zero curve is sampled every `step` year
    and interpolated linearly, so the shifts of all pillars add up to a parallel shift exactly (bumping one pillar of
    a monotone convex curve does not : that scheme is not linear in its inputs).
    """
    from Fixed_Income_Models import YieldCurve
    times = np.unique(np.concatenate([np.arange(step, curve.times[-1] + 1e-9, step), curve.times]))
    weight = np.interp(times, curve.times, np.eye(len(curve.times))[pillar])
    return YieldCurve(times, curve.zero_rate(times) + bp * 1e-4 * weight, 'linear_zero',
                      f"{curve.name} key rate {curve.times[pillar]:g}y {bp:+g}bp")


def key_rate_durations(pricer, bond, bp=1.0, callable=True, max_pillar=None):
    """
    Key-rate durations : -dV/V per unit triangular shift of each pillar (Hull-White refitted, a and sigma fixed).
    Pillars beyond max_pillar (default : the first pillar after maturity) have no effect and are skipped.
    """
    curve = pricer.model.curve
    last = np.searchsorted(curve.times, bond.maturity) if max_pillar is None else np.searchsorted(curve.times,
                                                                                                   max_pillar)
    P0 = pricer.price(bond, callable)
    out = {}
    for i in range(min(last + 1, len(curve.times))):
        up = pricer.with_model(pricer.model.with_curve(key_rate_bump(curve, i, bp))).price(bond, callable)
        down = pricer.with_model(pricer.model.with_curve(key_rate_bump(curve, i, -bp))).price(bond, callable)
        out[curve.times[i]] = (down - up) / (2 * bp * 1e-4 * P0)
    return pd.Series(out, name="key-rate duration")


def pricing_report(pricer, bond, market_price=None):
    """
    Full report of one bond : model dirty / clean prices, straight bond and call option, spreads (Z-spread and OAS
    of the market price if given, else of the model price), yields, effective duration and convexity, DV01, vega
    and CS01. All quantities are per bond (notional N) ; prices are dirty unless stated.
    """
    dirty = pricer.price(bond)
    straight = pricer.price(bond, callable=False)
    ai = getattr(bond, "accrued_today", 0.0)
    quoted = dirty if market_price is None else market_price + ai          # market prices are quoted clean
    yields = yield_table(bond, quoted)
    risk = effective_risk(pricer, bond)
    model = pricer.model
    return pd.Series({
        "dirty price": dirty, "accrued interest": ai, "clean price": dirty - ai,
        "straight bond (dirty)": straight, "call option (issuer)": straight - dirty,
        "Z-spread (bp)": 1e4 * option_adjusted_spread(model, bond, quoted, callable=False),
        "OAS (bp)": 1e4 * option_adjusted_spread(model, bond, quoted),
        "yield to maturity (%)": float(yields["yield (%)"].iloc[-1]),
        "yield to worst (%)": float(yields["yield (%)"].min()),
        "effective duration": risk["effective duration"], "effective convexity": risk["effective convexity"],
        "DV01": risk["DV01"], "vega (per bp of sigma)": risk["vega (per bp of sigma)"], "CS01": risk["CS01"]})
