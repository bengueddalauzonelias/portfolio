import numpy as np


# Market instruments used to build (bootstrap) a yield curve.
# Single-curve framework : the same curve discounts cash flows and projects floating rates.
# Times are year fractions (no day-count convention, calendar or settlement lag).
# Each instrument knows its market quote and the quote implied by a curve ; the bootstrap solves
# implied_quote(curve) = quote for every instrument.


class Instrument:
    """
    Base class of the curve instruments.
    """
    kind = "instrument"

    def __init__(self, maturity, quote):
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")
        self.maturity = float(maturity)
        self.quote = float(quote)

    def implied_quote(self, curve):
        raise NotImplementedError

    def residual(self, curve):
        """ Implied quote minus market quote (in rate units : 1e-4 = 1bp). """
        return self.implied_quote(curve) - self.quote

    def __repr__(self):
        return f"{type(self).__name__}(maturity={self.maturity:g}, quote={100 * self.quote:.4f}%)"


class ZeroCouponRate(Instrument):
    """
    Zero-coupon rate quoted with the given compounding ('continuous', 'simple' or a frequency).
    """
    kind = "zero-coupon"

    def __init__(self, maturity, rate, compounding='continuous'):
        super().__init__(maturity, rate)
        self.compounding = compounding

    def implied_quote(self, curve):
        return curve.zero_rate(self.maturity, self.compounding)


class Deposit(Instrument):
    """
    Money-market deposit starting today, simple interest : 1 + R T = 1 / P(0,T).
    """
    kind = "deposit"

    def implied_quote(self, curve):
        return (1.0 / curve.discount(self.maturity) - 1.0) / self.maturity


class FRA(Instrument):
    """
    Forward Rate Agreement on [start, end], simple compounding : 1 + R (end - start) = P(0,start) / P(0,end).
    """
    kind = "FRA"

    def __init__(self, start, end, rate):
        if not 0 <= start < end:
            raise ValueError("Need 0 <= start < end.")
        super().__init__(end, rate)
        self.start = float(start)

    def implied_quote(self, curve):
        return curve.forward_rate(self.start, self.maturity, compounding='simple')

    def __repr__(self):
        return f"FRA(start={self.start:g}, end={self.maturity:g}, quote={100 * self.quote:.4f}%)"


class Swap(Instrument):
    """
    Par interest rate swap, fixed leg paid `frequency` times a year (single-curve : the floating leg is worth 1 - P(0,T)).
    Par condition : R * sum_i delta_i P(0,t_i) = 1 - P(0,T).
    """
    kind = "swap"

    def __init__(self, maturity, rate, frequency=1):
        super().__init__(maturity, rate)
        self.frequency = frequency

    def implied_quote(self, curve):
        return curve.par_rate(self.maturity, self.frequency)


class ParBond(Swap):
    """
    Bond priced at par with coupon = quote paid `frequency` times a year (e.g. US Treasury par yields, semi-annual).
    Same pricing equation as the par swap : 1 = c sum_i delta_i P(0,t_i) + P(0,T).
    """
    kind = "par bond"

    def __init__(self, maturity, rate, frequency=2):
        super().__init__(maturity, rate, frequency)
