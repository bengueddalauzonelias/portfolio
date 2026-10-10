import copy
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# This script prices linear and vanilla interest rate products on a YieldCurve (single-curve framework) :
# - swaps : legs, PV, par rate, annuity ;
# - caps / floors and European swaptions with the market models : Bachelier (normal vols, the market standard
#   since negative rates), Black (lognormal vols) and shifted Black ;
# - curve risk (DV01, convexity, key-rate sensitivities) and vega, for any pricer exposing price(product)
#   and with_curve(curve) (also used by the Hull-White model).
# Products are read through their schedules (Produits/rate_products.py) : swap.fixed_leg(), swap.float_leg(),
# capfloor.periods(), swaption.underlying() ; they are not imported.


# ======================================================================================================
# Option formulas on a forward rate (undiscounted : multiply by the discount factor / annuity)
# ======================================================================================================
def bachelier(F, K, sigma, T, is_call=True):
    """ Normal (Bachelier) model : E[(F_T - K)+] or E[(K - F_T)+] with dF = sigma dW. """
    F, K, sigma, T = np.broadcast_arrays(*map(lambda a: np.asarray(a, dtype=float), (F, K, sigma, T)))
    s = sigma * np.sqrt(T)
    phi = np.where(is_call, 1.0, -1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(s > 0, (F - K) / s, 0.0)
        value = phi * (F - K) * norm.cdf(phi * d) + s * norm.pdf(d)
    return np.where(s > 0, value, np.maximum(phi * (F - K), 0.0))


def black(F, K, sigma, T, is_call=True, shift=0.0):
    """ Lognormal (Black 76) model on F + shift : shift = 0 is the plain Black model. """
    F, K, sigma, T = np.broadcast_arrays(*map(lambda a: np.asarray(a, dtype=float), (F, K, sigma, T)))
    Fs, Ks = F + shift, K + shift
    if np.any(Fs <= 0) or np.any(Ks <= 0):
        raise ValueError("Black model needs F + shift > 0 and K + shift > 0 : use a shift or the Bachelier model.")
    s = sigma * np.sqrt(T)
    phi = np.where(is_call, 1.0, -1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = np.where(s > 0, (np.log(Fs / Ks) + 0.5 * s**2) / s, 0.0)
        value = phi * (Fs * norm.cdf(phi * d1) - Ks * norm.cdf(phi * (d1 - s)))
    return np.where(s > 0, value, np.maximum(phi * (F - K), 0.0))


MODELS = ('bachelier', 'black', 'shifted_black')


def option_value(model, F, K, sigma, T, is_call=True, shift=0.0):
    """ Undiscounted option value on a forward rate in the given model. """
    if model == 'bachelier':
        return bachelier(F, K, sigma, T, is_call)
    if model == 'black':
        return black(F, K, sigma, T, is_call, 0.0)
    if model == 'shifted_black':
        return black(F, K, sigma, T, is_call, shift)
    raise ValueError(f"model must be one of {MODELS}.")


def implied_vol(value, F, K, T, is_call=True, model='bachelier', shift=0.0):
    """ Volatility of `model` matching the undiscounted option value (Brent search). """
    intrinsic = max((F - K) if is_call else (K - F), 0.0)
    if value <= intrinsic + 1e-14:
        return 0.0
    upper = 0.1 if model == 'bachelier' else 5.0
    f = lambda s: float(option_value(model, F, K, s, T, is_call, shift)) - value
    return brentq(f, 1e-10, upper, xtol=1e-12, maxiter=500)


# ======================================================================================================
# Volatility inputs
# ======================================================================================================
class CapVolCurve:
    """
    Flat cap volatilities by cap maturity (one volatility for all the caplets of a cap), linearly interpolated
    in maturity and flat outside. The smile is ignored : the same volatility is used for every strike.
    """

    def __init__(self, maturities, vols, model='bachelier', name="cap vols"):
        self.maturities = np.asarray(maturities, dtype=float)
        self.vols = np.asarray(vols, dtype=float)
        self.model = model
        self.name = name

    def flat_vol(self, maturity):
        return float(np.interp(maturity, self.maturities, self.vols))

    def shifted(self, d):
        return CapVolCurve(self.maturities, self.vols + d, self.model, self.name)


class SwaptionVolGrid:
    """
    Swaption volatilities on an (expiry x tenor) grid, bilinearly interpolated and flat outside.
    The smile is ignored : the same volatility is used for every strike.
    """

    def __init__(self, expiries, tenors, vols, model='bachelier', name="swaption vols"):
        self.expiries = np.asarray(expiries, dtype=float)
        self.tenors = np.asarray(tenors, dtype=float)
        self.vols = np.asarray(vols, dtype=float)
        if self.vols.shape != (len(self.expiries), len(self.tenors)):
            raise ValueError("vols must have shape (len(expiries), len(tenors)).")
        self.model = model
        self.name = name

    def vol(self, expiry, tenor):
        e = np.clip(expiry, self.expiries[0], self.expiries[-1])
        t = np.clip(tenor, self.tenors[0], self.tenors[-1])
        by_expiry = [np.interp(t, self.tenors, row) for row in self.vols]
        return float(np.interp(e, self.expiries, by_expiry))

    def shifted(self, d):
        return SwaptionVolGrid(self.expiries, self.tenors, self.vols + d, self.model, self.name)

    def to_frame(self):
        return pd.DataFrame(self.vols, index=pd.Index(self.expiries, name="expiry"),
                            columns=pd.Index(self.tenors, name="tenor"))


# ======================================================================================================
# Pricer
# ======================================================================================================
class RatePricer:
    """
    Market pricer on a YieldCurve :
    - swaps from the curve alone (single-curve : floating leg = forwards of the curve) ;
    - caps / floors from cap volatilities (flat vols, or caplet spot vols stripped from them) ;
    - European swaptions from a swaption volatility grid.
    The volatilities (floats, CapVolCurve, SwaptionVolGrid) must be quoted in the convention of `model`.
    """

    def __init__(self, curve, cap_vols=None, swaption_vols=None, model='bachelier', shift=0.0, cap_vol_type='flat'):
        """
        Init method.
        - curve         : YieldCurve (discounting and projection)
        - cap_vols      : float or CapVolCurve
        - swaption_vols : float or SwaptionVolGrid
        - model         : 'bachelier', 'black' or 'shifted_black'
        - shift         : shift of the shifted Black model (e.g. 0.03 = 3%)
        - cap_vol_type  : 'flat' (one vol per cap) or 'spot' (caplet vols stripped from the flat vols)
        """
        if model not in MODELS: raise ValueError(f"model must be one of {MODELS}.")
        if cap_vol_type not in ('flat', 'spot'): raise ValueError("cap_vol_type must be 'flat' or 'spot'.")
        self.curve = curve
        self.cap_vols = cap_vols
        self.swaption_vols = swaption_vols
        self.model = model
        self.shift = shift
        self.cap_vol_type = cap_vol_type
        self._spot_vols = None

    def with_curve(self, curve):
        pricer = copy.copy(self)
        pricer.curve, pricer._spot_vols = curve, None
        return pricer

    def with_vol_shift(self, d):
        """ Copy of the pricer with every volatility shifted by d (vega). """
        pricer = copy.copy(self)
        shift = lambda v: None if v is None else (v + d if np.isscalar(v) else v.shifted(d))
        pricer.cap_vols, pricer.swaption_vols, pricer._spot_vols = shift(self.cap_vols), shift(self.swaption_vols), None
        return pricer

    # ------------------------------------------------------------------
    # Swaps
    # ------------------------------------------------------------------
    def swap_legs(self, swap):
        """ PV of the fixed leg (per unit of fixed rate : the annuity) and of the floating leg. """
        _, ends, accruals, notionals = swap.fixed_leg()
        annuity = float(np.sum(notionals * accruals * self.curve.discount(ends)))
        starts, ends, accruals, notionals = swap.float_leg()
        forwards = self.curve.forward_rate(starts, ends, compounding='simple')
        floating = float(np.sum(notionals * accruals * forwards * self.curve.discount(ends)))
        return annuity, floating

    def par_rate(self, swap):
        """ Fixed rate that gives the swap a zero value. """
        annuity, floating = self.swap_legs(swap)
        return floating / annuity

    def swap_value(self, swap):
        annuity, floating = self.swap_legs(swap)
        value = floating - swap.fixed_rate * annuity
        return value if swap.payer else -value

    def swap_cashflows(self, swap):
        """ One row per payment : leg, period, notional, rate, amount, discount factor, PV (sign of the holder). """
        sign = 1.0 if swap.payer else -1.0
        rows = []
        starts, ends, accruals, notionals = swap.fixed_leg()
        for s, e, a, n in zip(starts, ends, accruals, notionals):
            df = self.curve.discount(e)
            rows.append({"leg": "fixed", "start": s, "end": e, "notional": n, "rate (%)": 100 * swap.fixed_rate,
                         "amount": -sign * n * a * swap.fixed_rate, "discount": df, "PV": -sign * n * a * swap.fixed_rate * df})
        starts, ends, accruals, notionals = swap.float_leg()
        forwards = self.curve.forward_rate(starts, ends, compounding='simple')
        for s, e, a, n, f in zip(starts, ends, accruals, notionals, forwards):
            df = self.curve.discount(e)
            rows.append({"leg": "float", "start": s, "end": e, "notional": n, "rate (%)": 100 * f,
                         "amount": sign * n * a * f, "discount": df, "PV": sign * n * a * f * df})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Caps / floors
    # ------------------------------------------------------------------
    def _caplet_inputs(self, capfloor):
        fixings, payments, accruals, notionals = capfloor.periods()
        forwards = self.curve.forward_rate(fixings, payments, compounding='simple')
        weights = notionals * accruals * self.curve.discount(payments)
        return fixings, payments, forwards, weights

    def caplet_vols(self, capfloor):
        """ Volatility of each caplet : flat vol of the cap, or stripped spot vols. """
        fixings = capfloor.periods()[0]
        if np.isscalar(self.cap_vols):
            return np.full(len(fixings), float(self.cap_vols))
        if self.cap_vols is None:
            raise ValueError("No cap volatilities.")
        if self.cap_vol_type == 'flat':
            return np.full(len(fixings), self.cap_vols.flat_vol(capfloor.maturity))
        spot = self.spot_vols(capfloor.frequency)
        return np.array([spot(t) for t in fixings])

    def _atm_caplets(self, maturity, frequency):
        """ Caplets of a spot-starting cap (first period excluded) and its at-the-money strike (par rate). """
        step = 1.0 / frequency
        n = int(np.ceil(maturity * frequency - 1e-9))
        ends = maturity - step * np.arange(n)[::-1]
        starts = np.concatenate([[0.0], ends[:-1]])
        fixings, payments = starts[1:], ends[1:]
        forwards = self.curve.forward_rate(fixings, payments, compounding='simple')
        weights = (payments - fixings) * self.curve.discount(payments)
        return fixings, forwards, weights, float(np.sum(weights * forwards) / np.sum(weights))

    def spot_vols(self, frequency=2):
        """
        Caplet (spot) volatilities stripped from the flat cap volatilities, piecewise constant between the cap
        maturities : for each cap maturity T_m, the caplets fixing in (T_m-1, T_m] get the volatility that reprices
        the at-the-money cap of maturity T_m (strike = its par rate) quoted with its flat volatility.
        Returns a function fixing time -> spot vol.
        """
        if self._spot_vols is not None and self._spot_vols[0] == frequency:
            return self._spot_vols[1]
        bounds, levels = [], []
        for T in self.cap_vols.maturities:
            if T * frequency < 2 - 1e-9:
                continue                                   # no caplet once the first period is excluded
            fixings, forwards, weights, atm = self._atm_caplets(T, frequency)
            target = float(np.sum(weights * option_value(self.model, forwards, atm, self.cap_vols.flat_vol(T),
                                                         fixings, True, self.shift)))
            vols = np.full(len(fixings), np.nan)
            for b, l in zip(bounds[::-1], levels[::-1]):
                vols[fixings <= b + 1e-12] = l
            new = np.isnan(vols)
            if not new.any():
                continue
            f = lambda s: float(np.sum(weights * option_value(self.model, forwards, atm, np.where(new, s, vols),
                                                              fixings, True, self.shift))) - target
            levels.append(brentq(f, 1e-8, 0.1 if self.model == 'bachelier' else 5.0, xtol=1e-14))
            bounds.append(fixings[-1])
        spot = lambda t: levels[min(int(np.searchsorted(bounds, t - 1e-12)), len(levels) - 1)]
        self._spot_vols = (frequency, spot)
        self.spot_vol_table = pd.DataFrame({"last fixing (y)": bounds, "spot vol": levels})
        return spot

    def caplet_table(self, capfloor):
        """ One row per caplet / floorlet : fixing, payment, forward, volatility, value, delta, vega. """
        fixings, payments, forwards, weights = self._caplet_inputs(capfloor)
        vols = self.caplet_vols(capfloor)
        values = weights * option_value(self.model, forwards, capfloor.strike, vols, fixings, capfloor.is_cap, self.shift)
        h, dv = 1e-6, (1e-6 if self.model == 'bachelier' else 1e-4)
        delta = weights * (option_value(self.model, forwards + h, capfloor.strike, vols, fixings, capfloor.is_cap, self.shift)
                           - option_value(self.model, forwards - h, capfloor.strike, vols, fixings, capfloor.is_cap, self.shift)) / (2 * h)
        vega = weights * (option_value(self.model, forwards, capfloor.strike, vols + dv, fixings, capfloor.is_cap, self.shift)
                          - option_value(self.model, forwards, capfloor.strike, vols - dv, fixings, capfloor.is_cap, self.shift)) / (2 * dv)
        unit = 1e-4 if self.model == 'bachelier' else 1e-2
        return pd.DataFrame({"fixing": fixings, "payment": payments, "forward (%)": 100 * forwards,
                             "vol": vols, "value": values, "delta (per bp of forward)": delta * 1e-4,
                             f"vega (per {'bp' if self.model == 'bachelier' else '%'} of vol)": vega * unit})

    def capfloor_value(self, capfloor):
        return float(self.caplet_table(capfloor)["value"].sum())

    def implied_cap_vol(self, capfloor, value, model=None):
        """ Flat volatility of `model` (default : the pricer's) that gives the cap / floor the value `value`. """
        model = model or self.model
        fixings, _, forwards, weights = self._caplet_inputs(capfloor)
        f = lambda s: float(np.sum(weights * option_value(model, forwards, capfloor.strike, s, fixings,
                                                          capfloor.is_cap, self.shift))) - value
        return brentq(f, 1e-10, 0.1 if model == 'bachelier' else 5.0, xtol=1e-14)

    # ------------------------------------------------------------------
    # Swaptions
    # ------------------------------------------------------------------
    def swaption_vol(self, swaption):
        if np.isscalar(self.swaption_vols):
            return float(self.swaption_vols)
        if self.swaption_vols is None:
            raise ValueError("No swaption volatilities.")
        return self.swaption_vols.vol(swaption.expiry, swaption.tenor)

    def swaption_value(self, swaption, vol=None):
        """ European swaption : annuity * option on the forward swap rate (payer = call). """
        if swaption.exercise != 'european':
            raise ValueError("The market model prices European swaptions only : use the Hull-White model for Bermudans.")
        underlying = swaption.underlying()
        annuity, floating = self.swap_legs(underlying)
        forward = floating / annuity
        vol = self.swaption_vol(swaption) if vol is None else vol
        return float(annuity * option_value(self.model, forward, swaption.strike, vol, swaption.expiry, swaption.payer, self.shift))

    def implied_swaption_vol(self, swaption, value, model=None):
        """ Volatility of `model` (default : the pricer's) matching a swaption value. """
        annuity, floating = self.swap_legs(swaption.underlying())
        return implied_vol(value / annuity, floating / annuity, swaption.strike, swaption.expiry, swaption.payer,
                           model or self.model, self.shift)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def price(self, product):
        if product.kind == "swap":
            return self.swap_value(product)
        if product.kind == "capfloor":
            return self.capfloor_value(product)
        if product.kind == "swaption":
            return self.swaption_value(product)
        raise ValueError(f"Unknown product kind '{product.kind}'.")


# ======================================================================================================
# Risk
# ======================================================================================================
def curve_risk(pricer, product, bp=1.0):
    """
    Parallel curve risk of any pricer with price(product) and with_curve(curve) :
    DV01 (value change for +1bp, central difference) and convexity (second difference, per bp squared).
    """
    up = pricer.with_curve(pricer.curve.shifted(bp)).price(product)
    down = pricer.with_curve(pricer.curve.shifted(-bp)).price(product)
    base = pricer.price(product)
    return {"value": base, "DV01 (per +1bp)": (up - down) / (2 * bp), "convexity (per bp^2)": (up - 2 * base + down) / bp**2}


def key_rate_risk(pricer, product, bp=1.0):
    """ Value change for a +1bp bump of each pillar zero rate (central difference), one row per pillar. """
    curve = pricer.curve
    rows = []
    for i, t in enumerate(curve.times):
        up = pricer.with_curve(curve.bumped(i, bp)).price(product)
        down = pricer.with_curve(curve.bumped(i, -bp)).price(product)
        rows.append({"pillar (y)": t, "sensitivity (per +1bp)": (up - down) / (2 * bp)})
    table = pd.DataFrame(rows).set_index("pillar (y)")
    return table


def vega(pricer, product, d=None):
    """
    Value change for +1bp of normal vol (Bachelier, Hull-White sigma) or +1% of lognormal vol (Black),
    central difference. The pricer must expose with_vol_shift(d).
    """
    d = d if d is not None else (1e-2 if getattr(pricer, 'model', 'bachelier') in ('black', 'shifted_black') else 1e-4)
    return (pricer.with_vol_shift(d).price(product) - pricer.with_vol_shift(-d).price(product)) / 2
