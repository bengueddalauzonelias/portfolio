# Phoenix autocallable note on a single underlying, self-contained : contract, pricing on the Monte Carlo paths
# (path_payoff, called by MonteCarloPricer.price), fair coupon, scenario statistics and reporting.
# It only needs a pricer exposing paths, discount_factors and survival_probability (Equity_Models/monte_carlo.py).
import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd


class Phoenix:
    """
    Phoenix autocall. All the levels (coupon, autocall and protection barriers, put strike) are fractions of the
    initial fixing S_ref, an absolute level fixed at inception (so that the Greeks bump the spot, not the reference).

    Life of the note, at each observation date t_k (k = 1..n, t_n = maturity), with performance P_k = S_k / S_ref :
    - coupon : coupon_rate * (t_k - t_k-1) is paid if P_k >= coupon barrier ; with memory, the coupons missed since
      the last payment are paid as well ;
    - autocall : if t_k is callable and P_k >= autocall barrier, the note is redeemed at nominal.
    At maturity, if the note was never called :
    - not knocked-in : redemption at nominal ;
    - knocked-in     : redemption at nominal * max(floor, min(1, P_T / put_strike + buffer)).
    The knock-in event is P_T < protection barrier (European protection) or the underlying touching the
    protection barrier at any time before maturity (American protection, continuous or on the simulation grid).
    Standard note : put_strike = 1 (loss = performance). Airbag : put_strike = protection barrier (no loss jump
    at the barrier). Capital guarantee : floor > 0. Buffer : buffer > 0 with protection_barrier = 1 - buffer, the
    investor is protected against the first `buffer` of decline and bears the rest one for one (no loss jump).
    coupon_barrier = 0 gives an unconditional (fixed) coupon ; coupon_barrier = autocall_barrier with memory gives
    back a snowball Athena.

    Optional features :
    - range accrual : the coupon of a period only accrues on the simulation dates where lower <= P_t <= upper
                      (it is still paid only if P_k >= coupon barrier : use coupon_barrier=0 for a pure range accrual) ;
    - duo           : if P_k is below the autocall barrier but above the duo barrier, the note is still called,
                      with the coupon paid at that date multiplied by duo_coupon_ratio ;
    - look back     : S_ref is the minimum (or the average) of the underlying over [0, lookback], path by path,
                      instead of the initial fixing (the levels are then fractions of this reference).
    """
    PROTECTIONS = ('european', 'american', 'american_discrete')
    LOOKBACK_TYPES = ('min', 'average')

    def __init__(self, maturity, initial_fixing, coupon_rate, coupon_barrier=0.7, memory=True, observation_dates=None,
                 frequency=1.0, autocall_barrier=1.0, step_down=0.0, first_call_date=None, protection_barrier=0.6,
                 protection_type='european', put_strike=1.0, floor=0.0, buffer=0.0, accrual_range=None,
                 duo_barrier=None, duo_coupon_ratio=0.5, lookback=None, lookback_type='min', nominal=100.0):
        """
        Init method.
        - maturity           : maturity T in years (last observation date)
        - initial_fixing     : reference level S_ref of the underlying (absolute)
        - coupon_rate        : coupon per annum, as a fraction of the nominal (e.g. 0.08)
        - coupon_barrier     : fraction of S_ref, scalar or one value per observation
        - memory             : True if the missed coupons are paid at the next coupon payment
        - observation_dates  : observation dates in years ; by default every `frequency` years until maturity
        - autocall_barrier   : fraction of S_ref, scalar or one value per observation
        - step_down          : decrease of the autocall barrier at each observation (scalar barrier only)
        - first_call_date    : observations before this date are not callable (non-call period)
        - protection_barrier : knock-in barrier of the capital protection, fraction of S_ref
        - protection_type    : 'european' (at maturity), 'american' (continuous, Brownian bridge) or
                               'american_discrete' (on the simulation dates)
        - put_strike         : strike of the short put, fraction of S_ref (1 : standard, protection barrier : airbag)
        - floor              : minimum redemption, fraction of the nominal (capital guarantee)
        - buffer             : first decline absorbed by the issuer, fraction of S_ref (e.g. 0.1 : the investor only
                               loses below 90%) ; use it with protection_barrier = 1 - buffer and put_strike = 1
        - accrual_range      : (lower, upper) fractions of S_ref : range accrual coupon (None : standard coupon)
        - duo_barrier        : second call level, fraction of S_ref, scalar or one value per observation (None : off)
        - duo_coupon_ratio   : fraction of the coupon paid when the note is called between the two levels
        - lookback           : length in years of the look back period at inception (None : no look back)
        - lookback_type      : 'min' (best entry point for the investor) or 'average' (Asian fixing)
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
        if not 0 <= buffer < 1: raise ValueError("buffer must be in [0, 1).")
        if buffer > 0 and abs(put_strike - 1.0) > 1e-9:
            raise ValueError("A buffer cannot be combined with an airbag (put_strike must be 1).")
        if accrual_range is not None and not accrual_range[0] < accrual_range[1]:
            raise ValueError("accrual_range must be (lower, upper) with lower < upper.")
        if not 0 <= duo_coupon_ratio <= 1: raise ValueError("duo_coupon_ratio must be in [0, 1].")
        if lookback is not None and not 0 < lookback < dates[0]:
            raise ValueError("lookback must be positive and end before the first observation date.")
        if lookback_type not in self.LOOKBACK_TYPES:
            raise ValueError(f"lookback_type must be one of {self.LOOKBACK_TYPES}.")
        # Attributes
        self.maturity = float(maturity)
        self.initial_fixing = float(initial_fixing)
        self.coupon_rate = float(coupon_rate)
        self.observation_dates = dates
        self.coupon_barriers = self._levels(coupon_barrier)
        self.memory = bool(memory)
        self.autocall_barriers = self._levels(autocall_barrier, step_down)
        self.first_call_date = 0.0 if first_call_date is None else float(first_call_date)
        self.protection_barrier = float(protection_barrier)
        self.protection_type = protection_type
        self.put_strike = float(put_strike)
        self.floor = float(floor)
        self.buffer = float(buffer)
        self.accrual_range = None if accrual_range is None else (float(accrual_range[0]), float(accrual_range[1]))
        self.duo_barriers = None if duo_barrier is None else self._levels(duo_barrier)
        if self.duo_barriers is not None and np.any(self.duo_barriers > self.autocall_barriers + 1e-12):
            raise ValueError("The duo barrier must be below the autocall barrier.")
        self.duo_coupon_ratio = float(duo_coupon_ratio)
        self.lookback = None if lookback is None else float(lookback)
        self.lookback_type = lookback_type
        self.nominal = float(nominal)
        self.exercise = None
        self._accrual_offset = 0.0                # time already elapsed since inception (set by shifted, for theta)

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

    def coupon_amounts(self):
        """ Coupon paid at t_k if P_k >= coupon barrier (fraction of the nominal), without the memory. """
        periods = self.accrual_periods()
        periods[0] += self._accrual_offset
        return self.coupon_rate * periods

    def reference_level(self, S):
        """ S_ref path by path : initial fixing, or min / average of the underlying over the look back period. """
        if self.lookback is None:
            return self.initial_fixing
        M = S.shape[0] - 1
        end = int(np.rint(self.lookback / self.maturity * M))
        window = S[:end + 1]
        return window.min(axis=0) if self.lookback_type == 'min' else window.mean(axis=0)

    def path_coupon_amounts(self, S, reference, dt, idx):
        """
        Coupon of each period, path by path (shape : observations x paths), without the memory.
        Range accrual : the coupon only accrues on the grid dates of the period where the performance is in the range.
        """
        if self.accrual_range is None:
            return self.coupon_amounts()[:, None]
        lower, upper = self.accrual_range
        performance = S[1:] / reference
        in_range = (performance >= lower) & (performance <= upper)
        accrued = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(in_range, axis=0) * dt])[idx]   # time in range at t_k
        periods = np.diff(np.vstack([np.zeros((1, S.shape[1])), accrued]), axis=0)
        periods[0] += self._accrual_offset
        return self.coupon_rate * periods

    def redemption_if_knocked(self, performance):
        """ Redemption at maturity, as a fraction of the nominal, once the protection is knocked-in. """
        return np.maximum(self.floor, np.minimum(1.0, performance / self.put_strike + self.buffer))

    def shifted(self, dt):
        """ Same note seen dt years later (used for theta) : all the dates are moved by -dt. """
        if dt >= self.observation_dates[0]:
            raise ValueError("dt must be lower than the first observation date.")
        note = copy.copy(self)
        note.maturity = self.maturity - dt
        note.observation_dates = self.observation_dates - dt
        note.first_call_date = max(self.first_call_date - dt, 0.0)
        note._accrual_offset = self._accrual_offset + dt
        if self.lookback is not None:                 # approximation : the elapsed part of the window is ignored
            note.lookback = max(self.lookback - dt, 0.0)
        return note

    def with_coupon(self, coupon_rate):
        """ Copy of the note with another coupon per annum. """
        note = copy.copy(self)
        note.coupon_rate = float(coupon_rate)
        return note

    # ------------------------------------------------------------------
    # Pricing (called by MonteCarloPricer.price)
    # ------------------------------------------------------------------
    def knock_in_probability(self, pricer, S, v, dt, reference=None):
        """
        Probability, path by path, that the capital protection has been knocked-in at maturity. With a look back,
        the barrier is path dependent and an American protection is only monitored after the look back period.
        """
        reference = self.initial_fixing if reference is None else reference
        level = self.protection_barrier * reference
        if self.protection_type == 'european':
            return (S[-1] < level).astype(float)
        if self.lookback is not None:
            start = int(np.rint(self.lookback / self.maturity * (S.shape[0] - 1)))
            S, v = S[start:], (None if v is None else v[start:])
        proxy = SimpleNamespace(barrier=level, breached=lambda x: x <= level,
                                monitoring='continuous' if self.protection_type == 'american' else 'discrete')
        return 1.0 - pricer.survival_probability(proxy, S, v, dt)

    def path_payoff(self, pricer, S, v, dt, df, details=False):
        """
        Discounted value of all the cash flows of each path. With details=True, also returns the call index of each
        path (-1 if never called), the knock-in probability, the loss at maturity, the number of coupons paid and the
        total coupon paid (fraction of the nominal, not discounted).
        """
        M, N = S.shape[0] - 1, S.shape[1]
        idx = self.observation_indices(M)
        reference = self.reference_level(S)
        performance = S[idx] / reference
        callable_ = self.observation_dates >= self.first_call_date - 1e-9
        callable_[-1] = True
        amounts = np.broadcast_to(self.path_coupon_amounts(S, reference, dt, idx), (len(idx), N))

        alive = np.ones(N, dtype=bool)
        value = np.zeros(N)
        call_index = np.full(N, -1)
        coupons_paid = np.zeros(N)
        coupon_total = np.zeros(N)
        unpaid = np.zeros(N)                      # coupons accrued and not paid yet (memory)
        for k, i in enumerate(idx):
            # calls of the date : full call above the autocall barrier, duo call between the duo and autocall barriers
            full_call = alive & (performance[k] >= self.autocall_barriers[k]) if callable_[k] else np.zeros(N, bool)
            duo_call = np.zeros(N, dtype=bool)
            if callable_[k] and self.duo_barriers is not None:
                duo_call = alive & ~full_call & (performance[k] >= self.duo_barriers[k])
            # coupon of the date
            pay = alive & (performance[k] >= self.coupon_barriers[k])
            due = (amounts[k] + (unpaid if self.memory else 0.0)) * np.where(duo_call, self.duo_coupon_ratio, 1.0)
            value += np.where(pay, self.nominal * due * df[i], 0.0)
            coupons_paid += pay
            coupon_total += np.where(pay, due, 0.0)
            if self.memory:
                unpaid = np.where(pay, 0.0, np.where(alive, unpaid + amounts[k], unpaid))
            # redemption at par of the called paths
            call = full_call | duo_call
            value += np.where(call, self.nominal * df[i], 0.0)
            call_index[call] = k
            alive &= ~call

        knocked = self.knock_in_probability(pricer, S, v, dt, reference)
        final = self.nominal * ((1.0 - knocked) + knocked * self.redemption_if_knocked(S[-1] / reference))
        value += np.where(alive, final * df[M], 0.0)
        if details:
            return value, {"call_index": call_index, "knock_in": np.where(alive, knocked, 0.0),
                           "loss": np.where(alive, self.nominal - final, 0.0), "coupons_paid": coupons_paid,
                           "coupon_paid": coupon_total}
        return value

    def control_variate(self, model):
        """
        European put struck at the protection barrier : (payoff function of S_T, exact price under model).
        None if the model has no semi-analytical vanilla price.
        """
        if not hasattr(model, 'price_vanillas'):
            return None
        K = self.protection_barrier * self.initial_fixing
        return (lambda S_T: np.maximum(K - S_T, 0.0)), float(model.price_vanillas([K], self.maturity, False)[0])

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
        to reach maturity, probability of a capital loss, expected loss, expected life, expected number of coupons
        and expected coupon paid (sum of the coupons, not discounted).
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
                "expected number of coupons": float(np.mean(d["coupons_paid"])),
                "expected coupon paid (% nominal)": float(100 * np.mean(d["coupon_paid"]))}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def features(self):
        """ Main terms of the contract, in a readable form (levels in % of S_ref, coupon in % p.a.). """
        periods = self.accrual_periods()
        barriers = self.autocall_barriers
        steps = -np.diff(barriers)
        if len(steps) == 0 or np.allclose(steps, 0.0):
            step_down = 0.0
        elif np.allclose(steps, steps[0]):
            step_down = 100 * steps[0]
        else:
            step_down = "custom schedule"
        callable_dates = self.observation_dates[self.observation_dates >= self.first_call_date - 1e-9]
        if self.buffer > 0:
            redemption = f"buffer {100 * self.buffer:g}%"
        elif abs(self.put_strike - self.protection_barrier) < 1e-9:
            redemption = "airbag"
        elif abs(self.put_strike - 1.0) < 1e-9:
            redemption = "standard"
        else:
            redemption = f"put struck at {100 * self.put_strike:g}%"
        return {"maturity (years)": self.maturity,
                "number of observations": len(self.observation_dates),
                "observation frequency (years)": float(periods[0]) if np.allclose(periods, periods[0]) else "irregular",
                "coupon (% p.a.)": 100 * self.coupon_rate,
                "first coupon barrier (%)": 100 * self.coupon_barriers[0],
                "last coupon barrier (%)": 100 * self.coupon_barriers[-1],
                "memory": self.memory,
                "first autocall barrier (%)": 100 * barriers[0],
                "last autocall barrier (%)": 100 * barriers[-1],
                "step-down per observation (%)": step_down,
                "first call date (years)": float(callable_dates[0]),
                "protection barrier (%)": 100 * self.protection_barrier,
                "protection type": self.protection_type,
                "redemption if knocked-in": redemption,
                "capital guaranteed (%)": 100 * self.floor,
                "buffer (%)": 100 * self.buffer,
                "range accrual (%)": "none" if self.accrual_range is None else
                                     f"[{100 * self.accrual_range[0]:g}, {100 * self.accrual_range[1]:g}]",
                "duo barrier (%)": "none" if self.duo_barriers is None else 100 * self.duo_barriers[0],
                "duo coupon ratio": "none" if self.duo_barriers is None else self.duo_coupon_ratio,
                "look back": "none" if self.lookback is None else f"{self.lookback_type} over {self.lookback:g}y",
                "nominal": self.nominal}

    def report(self, pricer, model, name=None, by_date=False):
        """
        One table with the contract terms, the pricing (price, 95% confidence interval, fair coupon) and the
        scenario statistics. With by_date=True, also returns the call probability at each observation date.
        """
        res = pricer.price(model, self)
        stats = self.statistics(pricer, model)
        by_date_probabilities = stats.pop("call probability by date")
        pricing = {"price": res.price,
                   "95% confidence (+/-)": 1.96 * res.stderr,
                   "price (% nominal)": 100 * res.price / self.nominal,
                   "fair coupon (% p.a.)": 100 * self.fair_coupon(pricer, model)}
        rows = {}
        for section, items in (("Contract", self.features()), ("Pricing", pricing), ("Scenarios", stats)):
            rows.update({(section, key): value for key, value in items.items()})
        table = pd.Series(rows, name=name or self.code, dtype=object).to_frame()
        if not by_date:
            return table
        calls = pd.Series(by_date_probabilities, name="call probability").rename_axis("observation date").to_frame()
        calls["cumulative"] = calls["call probability"].cumsum()
        return table, calls

    @property
    def code(self):
        return "Phoenix"

    def __repr__(self):
        return (f"Phoenix(T={self.maturity:g}, {len(self.observation_dates)} obs, coupon {100 * self.coupon_rate:.2f}% p.a., "
                f"coupon barrier {100 * self.coupon_barriers[0]:g}%{' memory' if self.memory else ''}, "
                f"autocall {100 * self.autocall_barriers[0]:g}%, protection {100 * self.protection_barrier:g}% "
                f"{self.protection_type})")
