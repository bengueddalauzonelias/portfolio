import copy
import numpy as np


# This script defines the interest rate products : Swap, CapFloor and Swaption.
# A product only describes its contract : schedules, accruals, notionals, strike and exercise rights.
# It knows nothing about the curve or the model ; the pricers of Fixed_Income_Models (rate_pricing.py, hull_white.py)
# read these schedules. Times are year fractions (no day-count convention, calendar or settlement lag).
# Single-curve framework : the floating leg pays the forward of the curve used for discounting (EURIBOR 6M).


def schedule(start, end, frequency):
    """
    Accrual periods of a leg paying `frequency` times a year from start to end, built backwards from the end
    (short first period if needed). Returns (period starts, period ends).
    """
    if end <= start: raise ValueError("Need start < end.")
    if frequency <= 0: raise ValueError("frequency must be positive.")
    step = 1.0 / frequency
    n = int(np.ceil((end - start) * frequency - 1e-9))
    ends = end - step * np.arange(n)[::-1]
    starts = np.concatenate([[start], ends[:-1]])
    return starts, ends


class Swap:
    """
    Fixed / floating interest rate swap.
    - payer swap    : pays the fixed rate, receives the floating rate (value increases with rates) ;
    - receiver swap : the opposite.
    Forward start (start > 0) and linearly amortizing notional are supported.
    """
    kind = "swap"

    def __init__(self, maturity, fixed_rate, payer=True, start=0.0, notional=1_000_000.0, fixed_frequency=1,
                 float_frequency=2, amortization='bullet'):
        """
        Init method.
        - maturity        : end of the swap in years
        - fixed_rate      : fixed rate K (e.g. 0.025)
        - payer           : True to pay fixed, False to receive fixed
        - start           : start of the swap in years (0 = spot, > 0 = forward start)
        - notional        : initial notional
        - fixed_frequency : fixed leg payments per year (1 = annual, EUR convention)
        - float_frequency : floating leg payments per year (2 = EURIBOR 6M)
        - amortization    : 'bullet' (constant notional) or 'linear' (notional decreasing linearly to 0 at maturity,
                            constant over each period, equal to its value at the period start)
        """
        # Sanity checks
        if not 0 <= start < maturity: raise ValueError("Need 0 <= start < maturity.")
        if notional <= 0: raise ValueError("notional must be positive.")
        if amortization not in ('bullet', 'linear'): raise ValueError("amortization must be 'bullet' or 'linear'.")
        # Attributes
        self.maturity = float(maturity)
        self.fixed_rate = float(fixed_rate)
        self.payer = bool(payer)
        self.start = float(start)
        self.notional = float(notional)
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.amortization = amortization

    def notional_at(self, t):
        """ Notional of a period starting at t. """
        t = np.asarray(t, dtype=float)
        if self.amortization == 'bullet':
            return self.notional * np.ones_like(t)
        return self.notional * (1.0 - (t - self.start) / (self.maturity - self.start))

    def fixed_leg(self):
        """ (period starts, period ends, accruals, notionals) of the fixed leg. """
        starts, ends = schedule(self.start, self.maturity, self.fixed_frequency)
        return starts, ends, ends - starts, self.notional_at(starts)

    def float_leg(self):
        """ (period starts, period ends, accruals, notionals) of the floating leg. """
        starts, ends = schedule(self.start, self.maturity, self.float_frequency)
        return starts, ends, ends - starts, self.notional_at(starts)

    def with_rate(self, fixed_rate):
        """ Same swap with another fixed rate. """
        swap = copy.copy(self)
        swap.fixed_rate = float(fixed_rate)
        return swap

    def reversed(self):
        """ The opposite swap (payer <-> receiver). """
        swap = copy.copy(self)
        swap.payer = not self.payer
        return swap

    @property
    def code(self):
        side = "Payer" if self.payer else "Receiver"
        start = f"{self.start:g}y x " if self.start > 0 else ""
        return f"{side} swap {start}{self.maturity - self.start:g}y"

    def __repr__(self):
        return (f"{self.code}(K={100 * self.fixed_rate:.4f}%, N={self.notional:,.0f}, fixed {self.fixed_frequency}/y, "
                f"float {self.float_frequency}/y, {self.amortization})")


class CapFloor:
    """
    Cap (strip of caplets) or floor (strip of floorlets) on the floating rate.
    A caplet on [t_i-1, t_i] fixes the forward L_i at t_i-1 and pays N tau_i (L_i - K)+ at t_i
    (floorlet : N tau_i (K - L_i)+). For a spot-starting cap, the first period is excluded since its rate is
    already fixed (market convention).
    """
    kind = "capfloor"

    def __init__(self, maturity, strike, is_cap=True, start=0.0, frequency=2, notional=1_000_000.0, skip_first=None):
        """
        Init method.
        - maturity   : end of the last period in years
        - strike     : strike K (e.g. 0.025)
        - is_cap     : True for a cap, False for a floor
        - start      : start of the first period (0 = spot start)
        - frequency  : number of periods per year (2 = EURIBOR 6M)
        - skip_first : exclude the first period (default : True for a spot start, False for a forward start)
        """
        # Sanity checks
        if not 0 <= start < maturity: raise ValueError("Need 0 <= start < maturity.")
        if notional <= 0: raise ValueError("notional must be positive.")
        # Attributes
        self.maturity = float(maturity)
        self.strike = float(strike)
        self.is_cap = bool(is_cap)
        self.start = float(start)
        self.frequency = frequency
        self.notional = float(notional)
        self.skip_first = (start == 0) if skip_first is None else bool(skip_first)
        if self.skip_first and len(schedule(start, maturity, frequency)[0]) < 2:
            raise ValueError("The cap needs at least one period after the first one.")

    def periods(self):
        """ (fixing times, payment times, accruals, notionals) of the caplets / floorlets. """
        starts, ends = schedule(self.start, self.maturity, self.frequency)
        if self.skip_first:
            starts, ends = starts[1:], ends[1:]
        return starts, ends, ends - starts, self.notional * np.ones_like(starts)

    def with_strike(self, strike):
        option = copy.copy(self)
        option.strike = float(strike)
        return option

    def swap(self):
        """ The swap with the same schedule as the cap (payer for a cap) : Cap - Floor = this payer swap. """
        start = self.periods()[0][0]
        return Swap(self.maturity, self.strike, payer=True, start=start, notional=self.notional,
                    fixed_frequency=self.frequency, float_frequency=self.frequency)

    @property
    def code(self):
        return f"{'Cap' if self.is_cap else 'Floor'} {self.maturity:g}y"

    def __repr__(self):
        return f"{self.code}(K={100 * self.strike:.4f}%, {len(self.periods()[0])} periods, N={self.notional:,.0f})"


class Swaption:
    """
    Option to enter at the exercise date into a swap ending at expiry + tenor, with fixed rate = strike.
    - payer swaption    : right to pay fixed (a call on the swap rate) ;
    - receiver swaption : right to receive fixed (a put on the swap rate).
    European : one exercise date (expiry). Bermudan : the holder can exercise at every fixed leg date of the
    underlying from expiry to the last but one, into the remaining (co-terminal) swap.
    """
    kind = "swaption"

    def __init__(self, expiry, tenor, strike, payer=True, notional=1_000_000.0, fixed_frequency=1, float_frequency=2,
                 exercise='european'):
        # Sanity checks
        if expiry <= 0 or tenor <= 0: raise ValueError("expiry and tenor must be positive.")
        if exercise not in ('european', 'bermudan'): raise ValueError("exercise must be 'european' or 'bermudan'.")
        # Attributes
        self.expiry = float(expiry)
        self.tenor = float(tenor)
        self.strike = float(strike)
        self.payer = bool(payer)
        self.notional = float(notional)
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.exercise = exercise

    @property
    def maturity(self):
        return self.expiry + self.tenor

    def underlying(self, start=None):
        """ The swap entered at exercise (start = expiry by default, or a later exercise date). """
        start = self.expiry if start is None else start
        return Swap(self.maturity, self.strike, payer=self.payer, start=start, notional=self.notional,
                    fixed_frequency=self.fixed_frequency, float_frequency=self.float_frequency)

    def exercise_dates(self):
        """ Exercise dates : the expiry (European), or every fixed leg date before the maturity (Bermudan). """
        if self.exercise == 'european':
            return np.array([self.expiry])
        return self.underlying().fixed_leg()[0]

    def european(self, expiry=None):
        """ The European swaption exercisable at `expiry` into the co-terminal swap. """
        expiry = self.expiry if expiry is None else expiry
        return Swaption(expiry, self.maturity - expiry, self.strike, self.payer, self.notional,
                        self.fixed_frequency, self.float_frequency, 'european')

    def with_strike(self, strike):
        option = copy.copy(self)
        option.strike = float(strike)
        return option

    @property
    def code(self):
        side = "Payer" if self.payer else "Receiver"
        style = "" if self.exercise == 'european' else "Bermudan "
        return f"{style}{side} {self.expiry:g}y x {self.tenor:g}y"

    def __repr__(self):
        return f"{self.code}(K={100 * self.strike:.4f}%, N={self.notional:,.0f})"
