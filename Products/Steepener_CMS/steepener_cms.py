# This script defines and prices a CMS steepener note : each period pays
#     coupon_k = tau_k min(max(L (S_long(T_{k-1}) - S_short(T_{k-1})) + m, floor), cap),
# the CMS rates being fixed IN ADVANCE (at the start of the period) and paid at its end, and the notional is
# redeemed at maturity (capital protected note). The first coupon is fixed today.
# Two pricing methods :
# - price_replication_copula : market approach. Each CMS rate has its convexity-adjusted forward E^Tp[S] from the
#   swaption smile (Fixed_Income_Models.CMSReplication, timing adjustment included since Tp > T) ; the spread is
#   taken Gaussian with the ATM normal vols and a correlation rho (Bachelier spread), so the floor / cap are
#   Bachelier options on the spread ;
# - price_mc : Monte Carlo in ANY term-structure model exposing simulate(times, ...) -> (states, D) and
#   bond_price(t, T, state) : HullWhite, G2pp, QuasiGaussian, SABRLMM. The swap rates are recomputed from the
#   simulated bond prices, so convexity, timing and correlation are those of the model.
# Values are in currency for the notional of the note (default 100 : values read in % of notional).
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", ".."))          # Projets : Fixed_Income_Models package

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from Fixed_Income_Models import bachelier, forward_swap, swap_rates_on_paths


class CMSSteepener:
    """
    CMS steepener note (in model time, year fractions from the valuation date).
    """
    kind = "cms_steepener"

    def __init__(self, maturity=10, long_tenor=10, short_tenor=2, leverage=4.0, margin=0.0, floor=0.0, cap=None,
                 frequency=1, notional=100.0, redemption=True):
        """
        Init method.
        - maturity              : note maturity (years), coupons every 1/frequency year
        - long_tenor, short_tenor : CMS tenors (years), e.g. 10 and 2 for a CMS 10y - 2y steepener
        - leverage, margin      : coupon rate L (S_long - S_short) + m before floor and cap
        - floor, cap            : bounds of the coupon rate (None : no bound)
        - redemption            : include the redemption of the notional at maturity
        """
        if maturity <= 0 or frequency <= 0: raise ValueError("maturity and frequency must be positive.")
        if floor is not None and cap is not None and cap < floor: raise ValueError("cap must be above floor.")
        self.maturity, self.frequency = float(maturity), frequency
        self.long_tenor, self.short_tenor = float(long_tenor), float(short_tenor)
        self.leverage, self.margin = float(leverage), float(margin)
        self.floor, self.cap = floor, cap
        self.notional = float(notional)
        self.redemption = bool(redemption)

    @property
    def payment_times(self):
        n = int(round(self.maturity * self.frequency))
        return np.arange(1, n + 1) / self.frequency

    @property
    def fixing_times(self):
        return self.payment_times - 1.0 / self.frequency

    @property
    def accruals(self):
        return np.full(len(self.payment_times), 1.0 / self.frequency)

    def coupon_rate(self, s_long, s_short):
        return self.bounded(self.leverage * (np.asarray(s_long) - np.asarray(s_short)) + self.margin)

    def bounded(self, rate):
        """ min(max(rate, floor), cap). """
        if self.floor is not None:
            rate = np.maximum(rate, self.floor)
        if self.cap is not None:
            rate = np.minimum(rate, self.cap)
        return rate

    def with_margin(self, margin):
        note = CMSSteepener.__new__(CMSSteepener)
        note.__dict__.update(self.__dict__)
        note.margin = float(margin)
        return note

    @property
    def code(self):
        bounds = f", floor {100 * self.floor:g}%" if self.floor is not None else ""
        bounds += f", cap {100 * self.cap:g}%" if self.cap is not None else ""
        return (f"{self.maturity:g}y steepener {self.leverage:g} x (CMS{self.long_tenor:g}y - CMS{self.short_tenor:g}y)"
                f" + {1e4 * self.margin:.0f}bp{bounds}")

    def __repr__(self):
        return f"CMSSteepener({self.code}, N={self.notional:g})"


# ======================================================================================================
# Market approach : replicated CMS forwards + Gaussian spread
# ======================================================================================================
def expected_coupon_rate(note, mean, std):
    """ E[min(max(Y, floor), cap)] for Y ~ N(mean, std^2) : floor + call(floor) - call(cap) (Bachelier, T = 1). """
    if std <= 0:
        return float(note.bounded(mean))
    value = mean if note.floor is None else note.floor + float(bachelier(mean, note.floor, std, 1.0))
    if note.cap is not None:
        value -= float(bachelier(mean, note.cap, std, 1.0))
    return value


def price_replication_copula(note, replication, vol_grid, rho):
    """
    Note value with the CMS forwards of `replication` (a CMSReplication) and a Gaussian spread :
        Y_k = L (S_long - S_short) + m ~ N(L (E[S_long] - E[S_short]) + m, L^2 (s_l^2 + s_s^2 - 2 rho s_l s_s) T_k),
    s = ATM normal vols of vol_grid. rho : a number or a function of the fixing time.
    Returns a dict with the value and the coupon table.
    """
    curve = replication.curve
    rho_fn = rho if callable(rho) else (lambda T: rho)
    rows = []
    for T, Tp, tau in zip(note.fixing_times, note.payment_times, note.accruals):
        long_ = replication.cms_rate(T, note.long_tenor, Tp)
        short = replication.cms_rate(T, note.short_tenor, Tp)
        s_l, s_s = vol_grid.vol(T, note.long_tenor), vol_grid.vol(T, note.short_tenor)
        r = rho_fn(T)
        std = note.leverage * np.sqrt(max(s_l**2 + s_s**2 - 2 * r * s_l * s_s, 0.0) * T)
        mean = note.leverage * (long_ - short) + note.margin
        rate = expected_coupon_rate(note, mean, std)
        fwd_long = forward_swap(curve, T, note.long_tenor)[0]
        fwd_short = forward_swap(curve, T, note.short_tenor)[0]
        rows.append({"fixing": T, "payment": Tp, "forward long": fwd_long, "forward short": fwd_short,
                     "CMS long": long_, "CMS short": short, "CA long (bp)": 1e4 * (long_ - fwd_long),
                     "CA short (bp)": 1e4 * (short - fwd_short), "spread vol (bp)": 1e4 * std / (note.leverage * np.sqrt(T)) if T > 0 else 0.0,
                     "expected coupon rate": rate, "discount": curve.discount(Tp),
                     "PV": note.notional * tau * rate * curve.discount(Tp)})
    table = pd.DataFrame(rows).set_index("payment")
    coupons = float(table["PV"].sum())
    redemption = note.notional * curve.discount(note.maturity) if note.redemption else 0.0
    return {"price": coupons + redemption, "coupons": coupons, "redemption": redemption, "table": table}


# ======================================================================================================
# Monte Carlo in a term-structure model
# ======================================================================================================
def simulate_cms(model, note, n_paths=50_000, seed=42, antithetic=True):
    """
    Simulated CMS fixings and discount factors. Returns (S_long, S_short, D_pay) of shape (n_coupons, n_paths) ;
    the fixings at time 0 are today's par rates.
    """
    fix, pay = note.fixing_times, note.payment_times
    grid = np.unique(np.concatenate([fix[fix > 1e-12], pay]))
    states, D = model.simulate(grid, n_paths, seed, antithetic)
    where = {round(t, 10): k for k, t in enumerate(grid)}
    S_long, S_short = np.empty((len(fix), n_paths)), np.empty((len(fix), n_paths))
    for i, T in enumerate(fix):
        for tenor, out in ((note.long_tenor, S_long), (note.short_tenor, S_short)):
            if T <= 1e-12:
                out[i] = forward_swap(model.curve, 0.0, tenor, note.frequency)[0]
            else:
                out[i] = swap_rates_on_paths(model, T, tenor, states[where[round(T, 10)]], note.frequency)[0]
    D_pay = np.stack([D[where[round(T, 10)]] for T in pay])
    return S_long, S_short, D_pay


def price_mc(model, note, n_paths=50_000, seed=42, antithetic=True, paths=None):
    """
    Note value by Monte Carlo in `model`. `paths` = simulate_cms(...) output can be reused (e.g. to price several
    margins on the same paths). Returns a dict with the price, its standard error, the coupon table (model CMS
    forwards E^Tp[S] = E[D S] / P(0,Tp), expected coupons, terminal correlation of the two CMS rates) and the
    Monte Carlo martingale error on the discount factors.
    """
    S_long, S_short, D_pay = simulate_cms(model, note, n_paths, seed, antithetic) if paths is None else paths
    curve = model.curve
    rates = note.coupon_rate(S_long, S_short)
    cash = note.notional * note.accruals[:, None] * rates * D_pay
    total = cash.sum(axis=0)
    if note.redemption:
        total = total + note.notional * D_pay[-1]
    samples = 0.5 * (total[:len(total) // 2] + total[len(total) // 2:]) if antithetic else total
    P = curve.discount(note.payment_times)
    rows = []
    for i, (T, Tp) in enumerate(zip(note.fixing_times, note.payment_times)):
        weight = D_pay[i] / P[i]                              # Radon-Nikodym density of Q^Tp
        corr = np.corrcoef(S_long[i], S_short[i])[0, 1] if T > 0 else np.nan
        fwd_long = forward_swap(curve, T, note.long_tenor)[0]
        fwd_short = forward_swap(curve, T, note.short_tenor)[0]
        rows.append({"fixing": T, "payment": Tp,
                     "CMS long": float(np.mean(weight * S_long[i])), "CMS short": float(np.mean(weight * S_short[i])),
                     "CA long (bp)": 1e4 * (float(np.mean(weight * S_long[i])) - fwd_long),
                     "CA short (bp)": 1e4 * (float(np.mean(weight * S_short[i])) - fwd_short),
                     "corr long/short": corr, "expected coupon rate": float(np.mean(weight * rates[i])),
                     "PV": float(np.mean(cash[i])), "martingale error (bp)": 1e4 * (float(np.mean(D_pay[i])) - P[i])})
    return {"price": float(np.mean(samples)), "stderr": float(np.std(samples, ddof=1) / np.sqrt(len(samples))),
            "table": pd.DataFrame(rows).set_index("payment")}


def fair_margin(note, pricer, target=None, bracket=(-0.05, 0.05)):
    """
    Margin m such that pricer(note.with_margin(m)) = target (default : par, the notional).
    pricer : function note -> value (e.g. lambda n: price_mc(model, n, paths=paths)["price"]).
    """
    target = note.notional if target is None else target
    return brentq(lambda m: pricer(note.with_margin(m)) - target, *bracket, xtol=1e-8)
