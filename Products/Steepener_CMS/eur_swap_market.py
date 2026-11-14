# This script builds the EUR market of the CMS steepener from Data/EUR/EUR_SWAP_RATES.csv (daily EUR par swap rates,
# in percent, tenors 1y to 50y, 2000-01-01 to 2022-08-18) :
# - swap_rates()            : the history on TARGET business days (the file repeats the last fixing on weekends) ;
# - eur_curve(date)         : single-curve zero-coupon curve bootstrapped from the par swaps of that date (annual fixed
#                             leg, Fixed_Income_Models.Bootstrapper, monotone convex) ;
# - realised_normal_vols()  : realised normal volatilities of the swap rates (bp per year), from daily changes ;
# - change_correlation()    : correlation of the daily changes of two swap rates (rolling or on a window) ;
# - pca()                   : principal components of the daily curve changes (level, slope, curvature) ;
# - historical_swaption_vols(date) : PROXY ATM normal swaption vols = realised vol of the spot swap rate of the same
#                             tenor over the year before `date`, flat in the expiry ;
# - sabr_smile(curve, vols) : normal SABR smiles (beta = 0) on top of these ATM vols, with ILLUSTRATIVE rho and nu.
# THE DATASET CONTAINS NO OPTION PRICE : every volatility input is either a historical estimate or an assumption,
# and the prices of this folder illustrate the models, they are not market prices.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", ".."))          # Projets : Fixed_Income_Models package

import numpy as np
import pandas as pd

from Fixed_Income_Models import Bootstrapper, Swap, SwaptionVolGrid, SABR

DATA = os.path.join(HERE, "..", "..", "Data", "EUR")
VALUATION_DATE = "2022-08-18"                            # last date of the file
ILLUSTRATIVE_SABR = {"rho": 0.10, "nu": 0.30}           # assumptions, not market data


def tenor_years(column):
    return float(column.rstrip("Y"))


def swap_rates():
    """ Par swap rates in decimals, business days only (weekends dropped), columns = tenors in years. """
    df = pd.read_csv(os.path.join(DATA, "EUR_SWAP_RATES.csv"), parse_dates=["DATE"]).set_index("DATE")
    df = df[df.index.dayofweek < 5]
    df.columns = [tenor_years(c) for c in df.columns]
    return df / 100


def eur_curve(date=VALUATION_DATE, rates=None, method='monotone_convex'):
    """ Zero-coupon curve that reprices the par swaps of `date` (single-curve, annual fixed leg). """
    rates = swap_rates() if rates is None else rates
    row = rates.loc[pd.Timestamp(date)].dropna()
    instruments = [Swap(T, r, frequency=1) for T, r in row.items()]
    bootstrapper = Bootstrapper(instruments, method)
    curve = bootstrapper.run()
    curve.name = f"EUR swaps {pd.Timestamp(date).date()}"
    return curve, bootstrapper


def _window(rates, end, window):
    end = pd.Timestamp(end)
    return rates.loc[:end].iloc[-(window + 1):]


def realised_normal_vols(rates, end=VALUATION_DATE, window=252, tenors=None):
    """ Realised normal vols (rate units per sqrt(year)) of the swap rates over the `window` business days to `end`. """
    sample = _window(rates, end, window)
    tenors = sample.columns if tenors is None else tenors
    return sample[tenors].diff().dropna().std() * np.sqrt(252)


def change_correlation(rates, tenor1, tenor2, end=VALUATION_DATE, window=252, rolling=False):
    """ Correlation of the daily changes of two swap rates : on the window to `end`, or rolling (a Series). """
    changes = rates[[tenor1, tenor2]].diff().dropna()
    if rolling:
        return changes[tenor1].rolling(window).corr(changes[tenor2]).dropna()
    changes = changes.loc[:pd.Timestamp(end)].iloc[-window:]
    return float(changes.corr().iloc[0, 1])


def pca(rates, end=VALUATION_DATE, window=252 * 5, tenors=(1, 2, 3, 5, 7, 10, 15, 20, 30)):
    """ Principal components of the daily changes : (explained variance ratios, loadings DataFrame). """
    changes = _window(rates, end, window)[list(tenors)].diff().dropna()
    eigval, eigvec = np.linalg.eigh(np.cov(changes.values.T))
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    eigvec = eigvec * np.sign(eigvec.sum(axis=0))                                    # level loading positive
    loadings = pd.DataFrame(eigvec[:, :3], index=pd.Index(tenors, name="tenor"), columns=["PC1", "PC2", "PC3"])
    return eigval / eigval.sum(), loadings


def historical_swaption_vols(rates, date=VALUATION_DATE, window=252, expiries=(1, 2, 3, 5, 7, 10),
                             tenors=(1, 2, 5, 10, 15, 20)):
    """
    PROXY ATM normal swaption vol grid : vol(expiry, tenor) = realised normal vol of the tenor's spot swap rate over
    the `window` business days to `date`, the same for every expiry (no term structure is observable in the data).
    """
    vols = realised_normal_vols(rates, date, window, list(tenors))
    grid = np.tile(vols.values, (len(expiries), 1))
    return SwaptionVolGrid(expiries, tenors, grid, model='bachelier',
                           name=f"historical proxy ({window}d realised vols to {pd.Timestamp(date).date()})")


def sabr_smile(vol_grid, rho=ILLUSTRATIVE_SABR["rho"], nu=ILLUSTRATIVE_SABR["nu"]):
    """
    Normal SABR smile factory (expiry, tenor, forward) -> SABR, alpha fitted to the ATM vol of vol_grid,
    rho and nu ILLUSTRATIVE. Used by CMSReplication.
    """
    def smile(expiry, tenor, forward):
        return SABR.from_atm_normal_vol(forward, expiry, vol_grid.vol(expiry, tenor), beta=0.0, rho=rho, nu=nu)
    return smile
