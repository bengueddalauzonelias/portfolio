import copy
from types import SimpleNamespace
import numpy as np


# This script defines the option products priced by the Monte Carlo engine (Model/monte_carlo.py).
# A product only describes its contract : payoff, barrier and exercise rights.
# It knows nothing about the simulation ; it only tells which European payoff with a known price can be used as
# control variate (control_variate), using the semi-analytical prices of the model when it has some.
# Products : VanillaOption, BarrierOption, DigitalOption, DigitalBarrierOption, Athena, Phoenix.
# Path-dependent products with several cash flows (Athena, Phoenix) value their own paths in path_payoff.


class Exercise:
    """
    Exercise rights of an option : European, Bermudan (on a list of dates) or American (at any time).
    """
    STYLES = ('european', 'bermudan', 'american')

    def __init__(self, style='european', dates=None):
        """
        Init method.
        - style : 'european', 'bermudan' or 'american'
        - dates : exercise dates in years (Bermudan only). The maturity is always an exercise date.
        """
        # Sanity checks
        if style not in self.STYLES:
            raise ValueError(f"style must be one of {self.STYLES}.")
        if style == 'bermudan':
            if dates is None or len(dates) == 0:
                raise ValueError("A Bermudan exercise needs a non-empty list of exercise dates.")
            if any(d <= 0 for d in dates):
                raise ValueError("Exercise dates must be positive.")
        # Attributes
        self.style = style
        self.dates = None if dates is None else sorted(float(d) for d in dates)

    @classmethod
    def european(cls):
        return cls('european')

    @classmethod
    def american(cls):
        return cls('american')

    @classmethod
    def bermudan(cls, dates):
        return cls('bermudan', dates)

    @classmethod
    def bermudan_every(cls, maturity, period):
        """
        Bermudan exercise every period years until maturity (e.g. period = 1/12 for monthly exercise).
        """
        if period <= 0: raise ValueError("period must be positive.")
        n = int(np.floor(maturity / period + 1e-9))
        return cls('bermudan', [period * (i + 1) for i in range(n)] + [maturity])

    @classmethod
    def from_style(cls, style, maturity=None, dates=None, period=None):
        """
        Builds an exercise from its style name. A Bermudan exercise needs either dates, or a period
        (exercise every period years until maturity).
        """
        if style == 'bermudan' and dates is None:
            if period is None or maturity is None:
                raise ValueError("A Bermudan exercise needs dates, or a period and the maturity.")
            return cls.bermudan_every(maturity, period)
        return cls(style, dates if style == 'bermudan' else None)

    def indices(self, maturity, n_steps):
        """
        Exercise dates as indices of the time grid {i * maturity / n_steps, i = 0..n_steps}.
        Bermudan dates are rounded to the closest grid date. The maturity (index n_steps) is always included.
        Exercise at t = 0 is not allowed.
        """
        if self.style == 'european':
            return np.array([n_steps])
        if self.style == 'american':
            return np.arange(1, n_steps + 1)
        idx = np.rint(np.array(self.dates) / maturity * n_steps).astype(int)
        idx = idx[(idx >= 1) & (idx <= n_steps)]
        return np.unique(np.append(idx, n_steps))

    def shifted(self, dt):
        """
        Exercise rights dt years later (used for theta) : Bermudan dates are moved by -dt.
        """
        if self.style != 'bermudan':
            return self
        dates = [d - dt for d in self.dates if d - dt > 0]
        return Exercise('bermudan', dates) if dates else Exercise('european')

    def __repr__(self):
        if self.style == 'bermudan':
            return f"Exercise('bermudan', {len(self.dates)} dates)"
        return f"Exercise('{self.style}')"


class VanillaOption:
    """
    Call or Put on the underlying, with European, Bermudan or American exercise.
    """

    def __init__(self, strike, maturity, isCall=True, exercise=None):
        """
        Init method.
        - strike   : strike K
        - maturity : maturity T in years
        - isCall   : True for a Call (payoff (S - K)+), False for a Put (payoff (K - S)+)
        - exercise : Exercise object (European by default)
        """
        # Sanity checks
        if strike <= 0: raise ValueError("Strike must be positive.")
        if maturity <= 0: raise ValueError("Maturity must be positive.")
        if exercise is not None and not isinstance(exercise, Exercise):
            raise ValueError("exercise must be an object of the Exercise class.")
        # Attributes
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.isCall = bool(isCall)
        self.exercise = exercise if exercise is not None else Exercise.european()

    def payoff(self, S):
        """
        Exercise value at spot S (vectorised).
        """
        return np.maximum(S - self.strike, 0.0) if self.isCall else np.maximum(self.strike - S, 0.0)

    def shifted(self, dt):
        """
        Same contract seen dt years later (used for theta) : maturity and exercise dates are moved by -dt.
        """
        if dt >= self.maturity: raise ValueError("dt must be lower than the maturity.")
        product = copy.copy(self)
        product.maturity = self.maturity - dt
        product.exercise = self.exercise.shifted(dt)
        return product

    def control_variate(self, model):
        """
        European vanilla with the same strike and type : (payoff function of S_T, exact price under model).
        None if the model has no semi-analytical vanilla price.
        """
        if not hasattr(model, 'price_vanillas'):
            return None
        vanilla = VanillaOption(self.strike, self.maturity, self.isCall)
        return vanilla.payoff, float(model.price_vanillas([self.strike], self.maturity, self.isCall)[0])

    @property
    def code(self):
        return "Call" if self.isCall else "Put"

    def __repr__(self):
        return f"{self.code}(K={self.strike:g}, T={self.maturity:.4f}, {self.exercise})"


class BarrierOption(VanillaOption):
    """
    Single barrier Call / Put : knock-in or knock-out, up or down.
    - knock-out : the option dies as soon as the barrier is touched.
    - knock-in  : the option only exists once the barrier has been touched.
    No rebate is paid. Codes : PDI, PDO, PUI, PUO, CDI, CDO, CUI, CUO.
    """
    CODES = ('PDI', 'PDO', 'PUI', 'PUO', 'CDI', 'CDO', 'CUI', 'CUO')

    def __init__(self, strike, maturity, barrier, isCall=True, direction='down', knock='out', exercise=None,
                 monitoring='continuous'):
        """
        Init method.
        - barrier    : barrier level H
        - direction  : 'down' (barrier below the spot) or 'up' (barrier above the spot)
        - knock      : 'in' or 'out'
        - monitoring : 'continuous' (barrier crossings between two simulation dates are accounted for with a
                       Brownian bridge) or 'discrete' (the barrier is only checked on the simulation dates)
        """
        super().__init__(strike, maturity, isCall, exercise)
        # Sanity checks
        if barrier <= 0: raise ValueError("Barrier must be positive.")
        if direction not in ('down', 'up'): raise ValueError("direction must be 'down' or 'up'.")
        if knock not in ('in', 'out'): raise ValueError("knock must be 'in' or 'out'.")
        if monitoring not in ('continuous', 'discrete'): raise ValueError("monitoring must be 'continuous' or 'discrete'.")
        # Attributes
        self.barrier = float(barrier)
        self.direction = direction
        self.knock = knock
        self.monitoring = monitoring

    @classmethod
    def from_code(cls, code, strike, barrier, maturity, exercise=None, monitoring='continuous'):
        """
        Builds a barrier option from its code, e.g. 'PDO' = Put Down-and-Out, 'CUI' = Call Up-and-In.
        """
        code = code.upper()
        if code not in cls.CODES:
            raise ValueError(f"code must be one of {cls.CODES}.")
        return cls(strike, maturity, barrier, isCall=(code[0] == 'C'), direction={'D': 'down', 'U': 'up'}[code[1]],
                   knock={'I': 'in', 'O': 'out'}[code[2]], exercise=exercise, monitoring=monitoring)

    def breached(self, S):
        """
        True where the spot S is on the knocked side of the barrier (vectorised).
        """
        return S <= self.barrier if self.direction == 'down' else S >= self.barrier

    @property
    def code(self):
        return ("C" if self.isCall else "P") + self.direction[0].upper() + self.knock[0].upper()

    def vanilla(self):
        """
        The underlying vanilla option (same strike, maturity and exercise), for In / Out parity checks.
        """
        return VanillaOption(self.strike, self.maturity, self.isCall, self.exercise)

    def __repr__(self):
        return f"{self.code}(K={self.strike:g}, H={self.barrier:g}, T={self.maturity:.4f}, {self.exercise}, {self.monitoring})"


class DigitalOption(VanillaOption):
    """
    Digital (binary) Call / Put with European, Bermudan or American exercise :
    - cash-or-nothing  : pays `cash` if S > K (call) or S < K (put) at exercise ;
    - asset-or-nothing : pays S under the same condition (asset=True).
    Early exercise is priced by Longstaff-Schwartz. With non-negative rates, a cash digital is exercised as soon as
    it is in the money : the American cash digital is a one-touch paid at hit, the Bermudan one is paid on the
    first exercise date in the money.
    """

    def __init__(self, strike, maturity, isCall=True, cash=1.0, asset=False, exercise=None):
        """
        Init method.
        - cash  : amount paid by a cash-or-nothing digital
        - asset : True for an asset-or-nothing digital (cash is then ignored)
        """
        super().__init__(strike, maturity, isCall, exercise)
        # Sanity checks
        if cash <= 0: raise ValueError("cash must be positive.")
        # Attributes
        self.cash = float(cash)
        self.asset = bool(asset)

    def payoff(self, S):
        """
        Exercise value at spot S (vectorised).
        """
        in_the_money = S > self.strike if self.isCall else S < self.strike
        return np.where(in_the_money, S if self.asset else self.cash, 0.0)

    def exact_european_price(self, model, rel_step=1e-3):
        """
        Price of the European digital (without barrier) from the model vanilla prices :
        cash call = -cash dC/dK, cash put = cash dP/dK, asset call = C - K dC/dK, asset put = K dP/dK - P.
        dC/dK and dP/dK are central finite differences of model.price_vanillas. None without vanilla prices.
        """
        if not hasattr(model, 'price_vanillas'):
            return None
        K, h = self.strike, rel_step * self.strike
        prices = model.price_vanillas([K - h, K, K + h], self.maturity, self.isCall)
        slope = (prices[2] - prices[0]) / (2 * h)
        if self.asset:
            return float(prices[1] - K * slope) if self.isCall else float(K * slope - prices[1])
        return float(-self.cash * slope) if self.isCall else float(self.cash * slope)

    def control_variate(self, model):
        """
        The European digital itself (same strike, type and payment) : much more correlated with a digital barrier
        or an early exercise digital than the vanilla. None if the model has no semi-analytical vanilla price.
        """
        exact = self.exact_european_price(model)
        if exact is None:
            return None
        digital = DigitalOption(self.strike, self.maturity, self.isCall, self.cash, self.asset)
        return digital.payoff, exact

    @property
    def code(self):
        return ("Asset" if self.asset else "Cash") + ("DigitalCall" if self.isCall else "DigitalPut")

    def __repr__(self):
        paid = "S_T" if self.asset else f"{self.cash:g}"
        return f"{self.code}(K={self.strike:g}, T={self.maturity:.4f}, pays {paid}, {self.exercise})"


class DigitalBarrierOption(DigitalOption, BarrierOption):
    """
    European digital with a single knock-in / knock-out barrier, e.g. a down-and-out cash digital call.
    The payoff comes from DigitalOption, the barrier (breached, knock, monitoring) from BarrierOption.
    In / Out parity : DigitalBarrier(in) + DigitalBarrier(out) = Digital.
    """

    def __init__(self, strike, maturity, barrier, isCall=True, direction='down', knock='out', cash=1.0,
                 asset=False, exercise=None, monitoring='continuous'):
        BarrierOption.__init__(self, strike, maturity, barrier, isCall, direction, knock, exercise, monitoring)
        # Sanity checks
        if cash <= 0: raise ValueError("cash must be positive.")
        if self.exercise.style != 'european':
            raise ValueError("Digital barrier options are European only.")
        # Attributes
        self.cash = float(cash)
        self.asset = bool(asset)

    def digital(self):
        """
        The digital without barrier, for In / Out parity checks.
        """
        return DigitalOption(self.strike, self.maturity, self.isCall, self.cash, self.asset)

    @property
    def code(self):
        return BarrierOption.code.fget(self) + ("-AssetDigital" if self.asset else "-CashDigital")

    def __repr__(self):
        paid = "S_T" if self.asset else f"{self.cash:g}"
        return (f"{self.code}(K={self.strike:g}, H={self.barrier:g}, T={self.maturity:.4f}, pays {paid}, "
                f"{self.monitoring})")


class AutocallProduct:
    """
    Base class of the autocallable notes (Athena and Phoenix), on a single underlying.
    All the levels (autocall, coupon and protection barriers, put strike) are fractions of the initial fixing,
    which is an absolute level fixed at inception (so that the Greeks bump the spot, not the reference).

    Life of the note, at each observation date t_k (k = 1..n, t_n = maturity), with performance P_k = S_k / S_ref :
    - Phoenix coupon : paid if P_k >= coupon barrier ; with memory, the coupons missed since the last payment are
      paid as well ;
    - autocall : if k is callable and P_k >= autocall barrier, the note is redeemed at nominal (+ Athena coupon).
    At maturity, if the note was never called :
    - not knocked-in : redemption at nominal ;
    - knocked-in     : redemption at nominal * max(floor, min(1, P_T / put_strike)).
    The knock-in event is P_T < protection barrier (European protection) or the underlying touching the
    protection barrier at any time before maturity (American protection, continuous or on the simulation grid).
    Standard note : put_strike = 1 (loss = performance). Airbag : put_strike = protection barrier (no loss jump
    at the barrier). Capital guarantee : floor > 0.
    """
    PROTECTIONS = ('european', 'american', 'american_discrete')

    def __init__(self, maturity, initial_fixing, coupon_rate, observation_dates=None, frequency=1.0,
                 autocall_barrier=1.0, step_down=0.0, first_call_date=None, protection_barrier=0.6,
                 protection_type='european', put_strike=1.0, floor=0.0, nominal=100.0):
        """
        Init method.
        - maturity           : maturity T in years (last observation date)
        - initial_fixing     : reference level S_ref of the underlying (absolute)
        - coupon_rate        : coupon per annum, as a fraction of the nominal (e.g. 0.08)
        - observation_dates  : observation dates in years ; by default every `frequency` years until maturity
        - autocall_barrier   : fraction of S_ref, scalar or one value per observation
        - step_down          : decrease of the autocall barrier at each observation (scalar barrier only)
        - first_call_date    : observations before this date are not callable (non-call period)
        - protection_barrier : knock-in barrier of the capital protection, fraction of S_ref
        - protection_type    : 'european' (at maturity), 'american' (continuous, Brownian bridge) or
                               'american_discrete' (on the simulation dates)
        - put_strike         : strike of the short put, fraction of S_ref (1 : standard, protection barrier : airbag)
        - floor              : minimum redemption, fraction of the nominal (capital guarantee)
        - nominal            : nominal of the note
        """
        if observation_dates is None:
            if frequency <= 0: raise ValueError("frequency must be positive.")
            n = int(np.floor(maturity / frequency + 1e-9))
            observation_dates = [frequency * (k + 1) for k in range(n)]
            if abs(observation_dates[-1] - maturity) > 1e-9:
                observation_dates.append(maturity)
        dates = np.array(sorted(float(d) for d in observation_dates))
        # Sanity checks
        if maturity <= 0: raise ValueError("Maturity must be positive.")
        if initial_fixing <= 0: raise ValueError("initial_fixing must be positive.")
        if np.any(dates <= 0) or abs(dates[-1] - maturity) > 1e-9:
            raise ValueError("Observation dates must be positive and end at the maturity.")
        if protection_type not in self.PROTECTIONS:
            raise ValueError(f"protection_type must be one of {self.PROTECTIONS}.")
        if not 0 <= floor <= 1: raise ValueError("floor must be in [0, 1].")
        if put_strike <= 0: raise ValueError("put_strike must be positive.")
        # Attributes
        self.maturity = float(maturity)
        self.initial_fixing = float(initial_fixing)
        self.coupon_rate = float(coupon_rate)
        self.observation_dates = dates
        self.autocall_barriers = self._levels(autocall_barrier, step_down)
        self.first_call_date = 0.0 if first_call_date is None else float(first_call_date)
        self.protection_barrier = float(protection_barrier)
        self.protection_type = protection_type
        self.put_strike = float(put_strike)
        self.floor = float(floor)
        self.nominal = float(nominal)
        self.exercise = None

    def _levels(self, level, step=0.0):
        """ One level per observation date, from a scalar (with an optional step-down) or a list. """
        n = len(self.observation_dates)
        if np.ndim(level) == 0:
            return float(level) - step * np.arange(n)
        level = np.asarray(level, dtype=float)
        if len(level) != n:
            raise ValueError("Give one level per observation date.")
        return level

    # ------------------------------------------------------------------
    # Contract
    # ------------------------------------------------------------------
    def observation_indices(self, n_steps):
        """ Observation dates as indices of the simulation grid {i * T / n_steps}. """
        idx = np.rint(self.observation_dates / self.maturity * n_steps).astype(int)
        if np.any(np.diff(idx) <= 0) or idx[0] < 1:
            raise ValueError("The simulation grid is too coarse for the observation dates : increase steps_per_year.")
        return idx

    def accrual_periods(self):
        """ Length of each observation period (t_k - t_k-1), the first one starting at 0. """
        return np.diff(np.concatenate([[0.0], self.observation_dates]))

    def autocall_coupons(self):
        """ Coupon paid with the redemption when the note is called at t_k (fraction of the nominal). """
        return np.zeros(len(self.observation_dates))

    def phoenix_coupons(self):
        """ (coupon barriers, coupon per period, memory) of the periodic conditional coupons, or None. """
        return None

    def redemption_if_knocked(self, performance):
        """ Redemption at maturity, as a fraction of the nominal, once the protection is knocked-in. """
        return np.maximum(self.floor, np.minimum(1.0, performance / self.put_strike))

    def shifted(self, dt):
        """ Same note seen dt years later (used for theta) : all the dates are moved by -dt. """
        if dt >= self.observation_dates[0]:
            raise ValueError("dt must be lower than the first observation date.")
        note = copy.copy(self)
        note.maturity = self.maturity - dt
        note.observation_dates = self.observation_dates - dt
        note.first_call_date = max(self.first_call_date - dt, 0.0)
        note._accrual_offset = getattr(self, "_accrual_offset", 0.0) + dt
        return note

    def with_coupon(self, coupon_rate):
        """ Copy of the note with another coupon per annum. """
        note = copy.copy(self)
        note.coupon_rate = float(coupon_rate)
        return note

    # ------------------------------------------------------------------
    # Pricing (called by MonteCarloPricer.price)
    # ------------------------------------------------------------------
    def knock_in_probability(self, pricer, S, v, dt):
        """ Probability, path by path, that the capital protection has been knocked-in at maturity. """
        level = self.protection_barrier * self.initial_fixing
        if self.protection_type == 'european':
            return (S[-1] < level).astype(float)
        proxy = SimpleNamespace(barrier=level, breached=lambda x: x <= level,
                                monitoring='continuous' if self.protection_type == 'american' else 'discrete')
        return 1.0 - pricer.survival_probability(proxy, S, v, dt)

    def path_payoff(self, pricer, S, v, dt, df, details=False):
        """
        Discounted value of all the cash flows of each path. With details=True, also returns the call index of each
        path (-1 if never called), the knock-in probability and the number of coupons paid.
        """
        M, N = S.shape[0] - 1, S.shape[1]
        idx = self.observation_indices(M)
        performance = S[idx] / self.initial_fixing
        callable_ = self.observation_dates >= self.first_call_date - 1e-9
        callable_[-1] = True
        call_coupons = self.autocall_coupons()
        phoenix = self.phoenix_coupons()

        alive = np.ones(N, dtype=bool)
        value = np.zeros(N)
        call_index = np.full(N, -1)
        coupons_paid = np.zeros(N)
        unpaid = np.zeros(N)                      # coupons accrued and not paid yet (memory)
        for k, i in enumerate(idx):
            if phoenix is not None:
                barriers, amounts, memory = phoenix
                pay = alive & (performance[k] >= barriers[k])
                due = amounts[k] + (unpaid if memory else 0.0)
                value += np.where(pay, self.nominal * due * df[i], 0.0)
                coupons_paid += pay
                if memory:
                    unpaid = np.where(pay, 0.0, np.where(alive, unpaid + amounts[k], unpaid))
            if callable_[k]:
                call = alive & (performance[k] >= self.autocall_barriers[k])
                value += np.where(call, self.nominal * (1.0 + call_coupons[k]) * df[i], 0.0)
                call_index[call] = k
                alive &= ~call

        knocked = self.knock_in_probability(pricer, S, v, dt)
        final = self.nominal * ((1.0 - knocked) + knocked * self.redemption_if_knocked(S[-1] / self.initial_fixing))
        value += np.where(alive, final * df[M], 0.0)
        if details:
            return value, {"call_index": call_index, "knock_in": np.where(alive, knocked, 0.0),
                           "loss": np.where(alive, self.nominal - final, 0.0), "coupons_paid": coupons_paid}
        return value

    def control_variate(self, model):
        """ European put struck at the protection barrier (exact price from the model vanilla prices). """
        put = VanillaOption(self.protection_barrier * self.initial_fixing, self.maturity, isCall=False)
        return put.control_variate(model)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def fair_coupon(self, pricer, model, target=None):
        """
        Coupon per annum that prices the note at `target` (the nominal by default, i.e. issued at par without
        margin). The price is linear in the coupon on a fixed set of paths, so two valuations are enough.
        """
        target = self.nominal if target is None else target
        p0 = pricer.price(model, self.with_coupon(0.0)).price
        p1 = pricer.price(model, self.with_coupon(1.0)).price
        return (target - p0) / (p1 - p0)

    def statistics(self, pricer, model):
        """
        Scenario statistics on the simulated paths : probability of a call at each observation date, probability
        to reach maturity, probability of a capital loss, expected loss, expected life and expected number of coupons.
        """
        S, v = pricer.paths(model, self.maturity)
        M = S.shape[0] - 1
        df = pricer.discount_factors(model, self.maturity, M)
        _, d = self.path_payoff(pricer, S, v, self.maturity / M, df, details=True)
        call_probabilities = np.array([np.mean(d["call_index"] == k) for k in range(len(self.observation_dates))])
        life = np.where(d["call_index"] >= 0, self.observation_dates[np.maximum(d["call_index"], 0)], self.maturity)
        return {"call probability by date": dict(zip(np.round(self.observation_dates, 4), call_probabilities)),
                "probability to reach maturity": float(np.mean(d["call_index"] < 0)),
                "probability of a capital loss": float(np.mean(d["knock_in"] * (d["loss"] > 0))),
                "expected loss at maturity (% nominal)": float(100 * np.mean(d["loss"]) / self.nominal),
                "expected life (years)": float(np.mean(life)),
                "expected number of coupons": float(np.mean(d["coupons_paid"]))}

    def __repr__(self):
        return (f"{self.code}(T={self.maturity:g}, {len(self.observation_dates)} obs, coupon {100 * self.coupon_rate:.2f}% p.a., "
                f"autocall {100 * self.autocall_barriers[0]:g}%, protection {100 * self.protection_barrier:g}% "
                f"{self.protection_type})")


class Athena(AutocallProduct):
    """
    Athena autocall : the coupon is only paid when the note is called (including at maturity, if the final
    performance is above the last autocall barrier).
    - coupon_type 'snowball' : coupon_rate * t_k at the call date t_k (the coupons accumulate) ;
    - coupon_type 'fixed'    : coupon_rate * (t_k - t_k-1), whatever the call date.
    """

    def __init__(self, maturity, initial_fixing, coupon_rate, coupon_type='snowball', **kwargs):
        super().__init__(maturity, initial_fixing, coupon_rate, **kwargs)
        if coupon_type not in ('snowball', 'fixed'):
            raise ValueError("coupon_type must be 'snowball' or 'fixed'.")
        self.coupon_type = coupon_type

    def autocall_coupons(self):
        if self.coupon_type == 'snowball':
            elapsed = self.observation_dates + getattr(self, "_accrual_offset", 0.0)
            return self.coupon_rate * elapsed
        return self.coupon_rate * self.accrual_periods()

    @property
    def code(self):
        return "Athena"


class Phoenix(AutocallProduct):
    """
    Phoenix : a coupon coupon_rate * (t_k - t_k-1) is paid at each observation date where the performance is above
    the coupon barrier (memory : the coupons missed since the last payment are paid as well). The note is called
    at nominal when the performance is above the autocall barrier. coupon_barrier = 0 gives an unconditional
    (fixed) coupon ; coupon_barrier = autocall_barrier with memory gives back a snowball Athena.
    """

    def __init__(self, maturity, initial_fixing, coupon_rate, coupon_barrier=0.7, memory=True, **kwargs):
        super().__init__(maturity, initial_fixing, coupon_rate, **kwargs)
        self.coupon_barriers = self._levels(coupon_barrier)
        self.memory = bool(memory)

    def phoenix_coupons(self):
        periods = self.accrual_periods()
        periods[0] += getattr(self, "_accrual_offset", 0.0)
        return self.coupon_barriers, self.coupon_rate * periods, self.memory

    @property
    def code(self):
        return "Phoenix"
