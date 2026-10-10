# This script prices knock-in barrier options : Put Down-and-In (PDI), Put Up-and-In (PUI),
# Call Down-and-In (CDI) and Call Up-and-In (CUI), with European, Bermudan or American exercise.
# It relies on the reusable scripts of the Model and Produits folders :
# products.py (contracts), monte_carlo.py (Monte Carlo / Longstaff-Schwartz pricer), greeks.py (Greeks).
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "Model"))   # models, Monte Carlo engine, Greeks
sys.path.append(os.path.join(HERE, ".."))                  # Produits/products.py

from products import BarrierOption, Exercise
from monte_carlo import MonteCarloPricer
from greeks import Greeks

KI_CODES = ('PDI', 'PUI', 'CDI', 'CUI')


def KI_pricer(model, code, strike, barrier, maturity, exercise='european', exercise_dates=None, exercise_period=None,
              monitoring='continuous', pricer=None, greeks=False, greeks_list=None):
    """
    Prices a knock-in barrier option : the option only exists once the barrier has been touched (no rebate).
    With early exercise, the holder can only exercise after the knock-in date.
    - model           : model with generate_paths (e.g. HestonModel, calibrated)
    - code            : 'PDI', 'PUI', 'CDI' or 'CUI'
    - strike, barrier : strike K and barrier H
    - maturity        : maturity T in years
    - exercise        : 'european', 'bermudan' or 'american'
    - exercise_dates  : Bermudan exercise dates (years), or exercise_period (e.g. 1/12 for monthly)
    - monitoring      : 'continuous' (Brownian bridge correction) or 'discrete' (simulation dates only)
    - pricer          : MonteCarloPricer (a default one is built if None)
    - greeks          : if True, also returns the first and second order Greeks
    - greeks_list     : subset of Greeks (default : delta, gamma, vega, vanna, volga, rho, theta)
    Returns a dict : product, price, stderr and greeks (DataFrame, or None).
    """
    code = code.upper()
    if code not in KI_CODES:
        raise ValueError(f"code must be one of {KI_CODES} for a knock-in option.")
    pricer = pricer if pricer is not None else MonteCarloPricer()
    product = BarrierOption.from_code(code, strike, barrier, maturity,
                                      Exercise.from_style(exercise, maturity, exercise_dates, exercise_period),
                                      monitoring)
    result = pricer.price(model, product)
    greeks_table = None
    if greeks:
        greeks_table = Greeks(pricer).compute(model, product, greeks_list or Greeks.FIRST_ORDER + Greeks.SECOND_ORDER)
    return {"product": product, "price": result.price, "stderr": result.stderr, "greeks": greeks_table}
