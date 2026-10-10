# This script builds the interest rate market shared by the rate product notebooks (Produits/Swap, CapFloor, Swaption) :
# - eur_curve()          : EURIBOR 6M zero-coupon curve of 12/02/2025 (same data as the CAC40 notebooks), monotone
#                          convex interpolation, flat forward beyond the last pillar (5 years) ;
# - illustrative_cap_vols() and illustrative_swaption_vols() : ILLUSTRATIVE normal (Bachelier) volatilities.
#   No volatility quotes are available in Data/ : these levels are assumptions of the order of magnitude of EUR
#   volatilities in early 2025, not market data. Replace them by quotes when available.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))          # Projets : Fixed_Income_Models package

import numpy as np
import pandas as pd

from Fixed_Income_Models import YieldCurve, CapVolCurve, SwaptionVolGrid

DATA = os.path.join(HERE, "..", "Data")
VALUATION_DATE = "2025-02-12"


def eur_curve(method='monotone_convex'):
    """ EURIBOR 6M zero-coupon curve (continuously compounded zero rates, 15 pillars from 1 week to 5 years). """
    df = pd.read_csv(os.path.join(DATA, "EURIBOR6M_ZCRates_12022025.csv"))
    return YieldCurve(df["Expiry"].values, df["ZCRate"].values, method, name=f"EURIBOR 6M {VALUATION_DATE}")


def illustrative_cap_vols():
    """ ILLUSTRATIVE flat normal cap volatilities (rate units : 0.0080 = 80bp), EURIBOR 6M caps of 1 to 10 years. """
    maturities = [1, 2, 3, 4, 5, 7, 10]
    vols_bp = [72, 82, 86, 87, 86, 84, 80]
    return CapVolCurve(maturities, np.array(vols_bp) * 1e-4, model='bachelier', name="illustrative cap normal vols")


def illustrative_swaption_vols():
    """ ILLUSTRATIVE normal swaption volatilities (rate units), expiries 0.5 to 5 years x tenors 1 to 10 years. """
    expiries = [0.5, 1, 2, 3, 5]
    tenors = [1, 2, 3, 5, 10]
    vols_bp = [[70, 74, 76, 77, 76],
               [76, 79, 80, 80, 78],
               [81, 82, 82, 81, 78],
               [82, 82, 81, 80, 77],
               [80, 79, 78, 77, 74]]
    return SwaptionVolGrid(expiries, tenors, np.array(vols_bp) * 1e-4, model='bachelier',
                           name="illustrative swaption normal vols")
