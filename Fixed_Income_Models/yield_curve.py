import numpy as np

try:
    from .interpolation import make_interpolator, INTERPOLATORS
except ImportError:
    from interpolation import make_interpolator, INTERPOLATORS


def to_continuous(rates, times, compounding):
    """
    Converts zero rates quoted with the given compounding into continuously compounded rates.
    compounding : 'continuous', 'simple' or an integer frequency m (1 = annual, 2 = semi-annual, ...).
    """
    rates, times = np.asarray(rates, dtype=float), np.asarray(times, dtype=float)
    if compounding == 'continuous':
        return rates
    if compounding == 'simple':
        return np.log1p(rates * times) / times
    m = float(compounding)
    return m * np.log1p(rates / m)


def from_continuous(rates, times, compounding):
    """
    Converts continuously compounded zero rates into the given compounding.
    """
    rates, times = np.asarray(rates, dtype=float), np.asarray(times, dtype=float)
    if compounding == 'continuous':
        return rates
    if compounding == 'simple':
        return np.expm1(rates * times) / times
    m = float(compounding)
    return m * np.expm1(rates / m)


class YieldCurve:
    """
    Zero-coupon yield curve.

    The curve is defined by pillars (times in years, continuously compounded zero rates) and an interpolation
    method (see interpolation.py). Every quantity is derived from y(T) = -ln P(0,T) :
    discount factors, zero rates in any compounding, discrete and instantaneous forwards, par swap rates.

    A YieldCurve is callable : curve(T) returns the continuously compounded zero rate r(T), so it can be used
    wherever a zc_rate(T) function is expected (ForwardCurve, BlackScholes, HestonModel).
    """

    def __init__(self, times, zero_rates, method='monotone_convex', name=None):
        """
        Init method.
        - times      : pillar times in years (strictly increasing, positive)
        - zero_rates : continuously compounded zero rates at the pillars
        - method     : interpolation method, one of interpolation.INTERPOLATORS
        """
        self.times = np.asarray(times, dtype=float)
        self.zero_rates = np.asarray(zero_rates, dtype=float)
        self.method = method
        self.name = name or method
        self.interpolator = make_interpolator(method, self.times, self.zero_rates)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_zero_rates(cls, times, rates, compounding='continuous', method='monotone_convex', name=None):
        """ Curve from zero rates quoted with any compounding convention. """
        return cls(times, to_continuous(rates, times, compounding), method, name)

    @classmethod
    def from_discount_factors(cls, times, discount_factors, method='monotone_convex', name=None):
        """ Curve from discount factors P(0, T_i). """
        times = np.asarray(times, dtype=float)
        return cls(times, -np.log(np.asarray(discount_factors, dtype=float)) / times, method, name)

    def with_method(self, method, name=None):
        """ Same pillars, other interpolation method. """
        return YieldCurve(self.times, self.zero_rates, method, name)

    # ------------------------------------------------------------------
    # Core quantities
    # ------------------------------------------------------------------
    def __call__(self, T):
        return self.zero_rate(T)

    def zero_rate(self, T, compounding='continuous'):
        """ Zero rate r(T) with the given compounding ('continuous', 'simple' or a frequency). """
        r = self.interpolator.zero(T)
        if compounding == 'continuous':
            return r
        out = from_continuous(r, np.maximum(T, 1e-12), compounding)
        return float(out) if np.ndim(out) == 0 else out

    def discount(self, T):
        """ Discount factor P(0,T). """
        y = self.interpolator.y(T)
        return float(np.exp(-y)) if np.ndim(y) == 0 else np.exp(-y)

    def forward_rate(self, T1, T2, compounding='continuous'):
        """
        Forward rate between T1 and T2 seen today :
        - continuous : ln(P(T1) / P(T2)) / (T2 - T1)
        - simple     : (P(T1) / P(T2) - 1) / (T2 - T1)       (FRA / Libor-type forward)
        """
        T1, T2 = np.asarray(T1, dtype=float), np.asarray(T2, dtype=float)
        if np.any(T1 < 0) or np.any(T2 <= T1):
            raise ValueError("Need 0 <= T1 < T2.")
        dy = np.asarray(self.interpolator.y(T2)) - np.asarray(self.interpolator.y(T1))
        tau = T2 - T1
        out = dy / tau if compounding == 'continuous' else np.expm1(dy) / tau
        return float(out) if np.ndim(out) == 0 else out

    def instantaneous_forward(self, T):
        """ Instantaneous forward rate f(0,T) = -d ln P(0,T) / dT. """
        return self.interpolator.forward(T)

    def integrated_short_rate(self, t1, t2):
        """
        int_{t1}^{t2} f(0,s) ds = ln(P(0,t1) / P(0,t2)) : deterministic drift of the spot over [t1, t2]
        in Monte Carlo simulations.
        """
        return self.interpolator.y(t2) - self.interpolator.y(t1)

    # ------------------------------------------------------------------
    # Swaps
    # ------------------------------------------------------------------
    @staticmethod
    def schedule(maturity, frequency=1):
        """ Payment times of a leg paying every 1/frequency year until maturity (short first period if needed). """
        step = 1.0 / frequency
        n = int(np.ceil(maturity * frequency - 1e-9))
        times = maturity - step * np.arange(n)[::-1]
        return times[times > 1e-9]

    def annuity(self, maturity, frequency=1):
        """ Annuity (PV01 of a unit notional fixed leg) : sum_i delta_i P(0, t_i). """
        times = self.schedule(maturity, frequency)
        deltas = np.diff(np.concatenate([[0.0], times]))
        return float(np.sum(deltas * self.discount(times)))

    def par_rate(self, maturity, frequency=1):
        """ Par swap rate (single-curve) : (1 - P(0,T)) / annuity. Also the par yield of a bond. """
        return (1.0 - self.discount(maturity)) / self.annuity(maturity, frequency)

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------
    def shifted(self, bp):
        """ Parallel shift of the pillar zero rates by bp basis points. """
        return YieldCurve(self.times, self.zero_rates + bp * 1e-4, self.method, f"{self.name} {bp:+g}bp")

    def bumped(self, pillar, bp=1.0):
        """ Key-rate bump : only the zero rate of pillar index `pillar` moves by bp basis points. """
        rates = self.zero_rates.copy()
        rates[pillar] += bp * 1e-4
        return YieldCurve(self.times, rates, self.method, f"{self.name} bump {self.times[pillar]:g}y")

    # ------------------------------------------------------------------
    # Backward compatible names
    # ------------------------------------------------------------------
    rate = zero_rate
    discount_factor = discount

    def forward(self, T1, T2):
        return self.forward_rate(T1, T2)

    def __repr__(self):
        return (f"YieldCurve(name={self.name!r}, method={self.method!r}, "
                f"pillars={len(self.times)} from {self.times[0]:g}y to {self.times[-1]:g}y)")


METHODS = sorted(set(INTERPOLATORS) - {"log_linear_discount"})
