# This script prices digital (binary) options : cash-or-nothing or asset-or-nothing Call / Put, with European,
# Bermudan or American exercise, and European digitals with a knock-in / knock-out barrier.
# It relies on the reusable scripts of the Model and Produits folders :
# products.py (contracts), monte_carlo.py (Monte Carlo / Longstaff-Schwartz pricer), greeks.py (Greeks).
# Command line (python digital.py --help) : prices any digital on the CAC40 (calibrated Heston) or on a flat
# Black-Scholes market, with strikes and barriers in % of spot, and gives the digital coupon paid for a budget.
# Examples :
#   python digital.py --strike 100 --maturity 1 --cash 5
#   python digital.py --put --strike 90 --exercise bermudan --period 0.25
#   python digital.py --strike 105 --barrier 120 --direction up --knock out --budget 3
#   python digital.py --market flat --spot 100 --rate 0.03 --div 0.01 --vol 0.2 --strike 110 --exercise american
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "Model"))   # models, Monte Carlo engine, Greeks
sys.path.append(os.path.join(HERE, ".."))                  # Produits/products.py, market.py

from products import DigitalOption, DigitalBarrierOption, Exercise
from monte_carlo import MonteCarloPricer
from greeks import Greeks


def digital_product(strike, maturity, isCall=True, cash=1.0, asset=False, exercise='european', exercise_dates=None,
                    exercise_period=None, barrier=None, direction='down', knock='out', monitoring='continuous'):
    """
    Builds a digital option.
    - strike, maturity : strike K and maturity T in years
    - isCall           : True for a digital Call (pays if S > K), False for a digital Put (pays if S < K)
    - cash, asset      : amount paid by a cash-or-nothing digital, or asset=True for an asset-or-nothing (pays S)
    - exercise         : 'european', 'bermudan' or 'american'
    - exercise_dates   : Bermudan exercise dates (years), or exercise_period (e.g. 1/12 for monthly)
    - barrier          : barrier level H (None for a digital without barrier). Barrier digitals are European only.
    - direction, knock : 'down' / 'up' and 'in' / 'out' (barrier digitals only)
    - monitoring       : 'continuous' (Brownian bridge correction) or 'discrete' (simulation dates only)
    """
    exercise = Exercise.from_style(exercise, maturity, exercise_dates, exercise_period)
    if barrier is None:
        return DigitalOption(strike, maturity, isCall, cash, asset, exercise)
    return DigitalBarrierOption(strike, maturity, barrier, isCall, direction, knock, cash, asset, exercise, monitoring)


def digital_pricer(model, strike, maturity, isCall=True, cash=1.0, asset=False, exercise='european',
                   exercise_dates=None, exercise_period=None, barrier=None, direction='down', knock='out',
                   monitoring='continuous', pricer=None, greeks=False, greeks_list=None):
    """
    Prices a digital option (arguments of digital_product).
    - model       : model with generate_paths (e.g. HestonModel, calibrated)
    - pricer      : MonteCarloPricer (a default one is built if None)
    - greeks      : if True, also returns the first and second order Greeks
    - greeks_list : subset of Greeks (default : delta, gamma, vega, vanna, volga, rho, theta)
    Returns a dict : product, price, stderr, european (semi-analytical price of the European digital without
    barrier, None if the model has no vanilla prices) and greeks (DataFrame, or None).
    """
    product = digital_product(strike, maturity, isCall, cash, asset, exercise, exercise_dates, exercise_period,
                              barrier, direction, knock, monitoring)
    pricer = pricer if pricer is not None else MonteCarloPricer()
    result = pricer.price(model, product)
    european = DigitalOption(strike, maturity, isCall, cash, asset).exact_european_price(model)
    greeks_table = None
    if greeks:
        greeks_table = Greeks(pricer).compute(model, product, greeks_list or Greeks.FIRST_ORDER + Greeks.SECOND_ORDER)
    return {"product": product, "price": result.price, "stderr": result.stderr, "european": european,
            "greeks": greeks_table}



def digital_coupon(model, budget, strike, maturity, **kwargs):
    """
    Structuring : digital coupon (cash) paid for a given budget (premium, same unit as the result).
    A cash digital price is linear in its cash : coupon = budget / price of the digital paying 1.
    kwargs : other arguments of digital_pricer (isCall, exercise, barrier...). Returns (coupon, result for cash = 1).
    """
    if kwargs.get('asset'):
        raise ValueError("The coupon is only defined for a cash-or-nothing digital.")
    kwargs.pop('cash', None)
    unit = digital_pricer(model, strike, maturity, cash=1.0, **kwargs)
    return budget / unit["price"], unit


def build_market(name='cac40', spot=100.0, rate=0.03, div=0.01, vol=0.20, calibrate=True):
    """
    Market of the command line : 'cac40' (CAC40 of 12/02/2025, calibrated Heston) or 'flat' (flat Black-Scholes,
    constant rate, dividend yield and volatility). Returns (model, spot).
    """
    from market import cac40_heston, flat_market
    if name == 'cac40':
        mkt = cac40_heston(calibrate=calibrate)
    elif name == 'flat':
        mkt = flat_market(S0=spot, r=rate, q=div, sigma=vol)
    else:
        raise ValueError("market must be 'cac40' or 'flat'.")
    return mkt.heston, mkt.S0


def main(argv=None):
    p = argparse.ArgumentParser(description="Digital option pricer (strikes and barriers in % of spot, "
                                            "cash and prices in % of notional).")
    m = p.add_argument_group("market")
    m.add_argument("--market", choices=["cac40", "flat"], default="cac40")
    m.add_argument("--spot", type=float, default=100.0, help="flat market spot")
    m.add_argument("--rate", type=float, default=0.03, help="flat market rate")
    m.add_argument("--div", type=float, default=0.01, help="flat market dividend yield")
    m.add_argument("--vol", type=float, default=0.20, help="flat market volatility")
    m.add_argument("--no-calibrate", action="store_true", help="CAC40 : known Heston parameters, no recalibration")
    d = p.add_argument_group("product")
    d.add_argument("--strike", type=float, default=100.0, help="strike in %% of spot")
    d.add_argument("--maturity", type=float, default=1.0, help="maturity in years")
    d.add_argument("--put", action="store_true", help="digital put (default : call)")
    d.add_argument("--cash", type=float, default=100.0, help="cash paid, in %% of notional")
    d.add_argument("--asset", action="store_true", help="asset-or-nothing (pays S / S0 in %% of notional)")
    d.add_argument("--exercise", choices=["european", "bermudan", "american"], default="european")
    d.add_argument("--period", type=float, help="Bermudan exercise period in years (e.g. 0.25)")
    d.add_argument("--dates", type=float, nargs="+", help="Bermudan exercise dates in years")
    d.add_argument("--barrier", type=float, help="barrier in %% of spot (European digitals only)")
    d.add_argument("--direction", choices=["down", "up"], default="down")
    d.add_argument("--knock", choices=["in", "out"], default="out")
    d.add_argument("--monitoring", choices=["continuous", "discrete"], default="continuous")
    s = p.add_argument_group("structuring and numerics")
    s.add_argument("--budget", type=float, help="budget in %% of notional : prints the digital coupon it pays")
    s.add_argument("--paths", type=int, default=50_000)
    s.add_argument("--steps", type=int, default=252, help="time steps per year")
    s.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    if a.exercise == "bermudan" and a.period is None and a.dates is None:
        p.error("--exercise bermudan needs --period or --dates.")
    model, S0 = build_market(a.market, a.spot, a.rate, a.div, a.vol, not a.no_calibrate)
    pricer = MonteCarloPricer(n_paths=a.paths, steps_per_year=a.steps, seed=a.seed)
    # Prices are computed on a notional of 100 % : cash in % of notional, asset digitals pay 100 * S / S0
    scale = 100.0 / S0 if a.asset else 1.0
    kwargs = dict(isCall=not a.put, asset=a.asset, exercise=a.exercise, exercise_dates=a.dates,
                  exercise_period=a.period, barrier=None if a.barrier is None else a.barrier / 100 * S0,
                  direction=a.direction, knock=a.knock, monitoring=a.monitoring, pricer=pricer)
    res = digital_pricer(model, a.strike / 100 * S0, a.maturity, cash=a.cash, **kwargs)

    print(f"\n{res['product']}   (S0 = {S0:g}, {a.market} market)")
    print(f"Price             : {scale * res['price']:.4f} % of notional  (std error {scale * res['stderr']:.4f})")
    if res["european"] is not None and (a.exercise != "european" or a.barrier is not None):
        european = scale * res["european"]
        label = "Early exercise premium" if a.exercise != "european" else "Barrier effect"
        print(f"European digital  : {european:.4f} %   ->  {label} : {scale * res['price'] - european:+.4f} %")
    if a.budget is not None:
        coupon, unit = digital_coupon(model, a.budget, a.strike / 100 * S0, a.maturity, **kwargs)
        print(f"Budget {a.budget:g} %       : digital coupon = {coupon:.4f} % of notional "
              f"(price of the digital paying 1 : {unit['price']:.6f})")
    return res


if __name__ == "__main__":
    main()
