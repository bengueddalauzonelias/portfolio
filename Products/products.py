import copy
import numpy as np


# This script defines the option products priced by the Monte Carlo engine (Equity_Models/monte_carlo.py).
# A product only describes its contract : payoff, barrier and exercise rights.
# It knows nothing about the simulation ; it only tells which European payoff with a known price can be used as
# control variate (control_variate), using the semi-analytical prices of the model when it has some.
# Products : VanillaOption, BarrierOption, DigitalOption, DigitalBarrierOption.
# The autocallable notes are self-contained, in Products/Athena/athena.py and Products/Phoenix/phoenix.py.


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
