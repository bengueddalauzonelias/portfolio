# Callable fixed-rate bonds with issuer default risk, under a one-factor Hull-White short rate and a deterministic
# default intensity (reduced-form credit model, Duffie & Singleton 1999, Lando 1998).
# - CreditCurve       : piecewise constant hazard rate lambda(t), survival Q(0,t) = exp(-int_0^t lambda),
#                       bootstrapped from CDS par spreads ;
# - CallableBond      : term sheet (coupons, call protection, call price schedule) with European, Bermudan or
#                       American issuer call ;
# - CallableBondPDE   : backward Crank-Nicolson solver of the pricing PDE in the Hull-White state x, written
#                       r(t) = x(t) + alpha(t) (see Fixed_Income_Models/hull_white_1F.py) :
#                           V_t - a x V_x + sigma^2 / 2 V_xx - (x + alpha(t) + lambda(t)) V + lambda(t) R N = 0,
#                       with coupons added on payment dates and V = min(V, call price + accrued) on exercise dates ;
# - closed forms used as benchmarks : straight risky bond, European callable bond with zero recovery (Jamshidian).
# Times are year fractions from the valuation date ; amounts are in currency (notional N).
import copy
import numpy as np
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
