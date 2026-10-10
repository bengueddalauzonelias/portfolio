# This script builds the market objects shared by the product test notebooks (Produits/<product>/*.ipynb) :
# - cac40_heston() : CAC40 data of 12/02/2025, forward curve, Black model, implied volatility surface and the
#                    Heston model calibrated in Model/heston_test.ipynb (local run from its global minimum) ;
# - flat_market()  : synthetic flat Black-Scholes market (constant rate, dividend yield and volatility), where
#                    closed-form prices are exact, with the equivalent Heston model (no vol-of-vol).
import os
import sys
import io
import datetime
import contextlib
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "Model"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from forward_curve import ForwardCurve
from Black_Scholes import BlackScholes
from ImpliedVolatilitySurface import ImpliedVolatilitySurface
from Heston import HestonModel
from error import CalibrationLoss
from optimizer import Calibrator

DATA = os.path.join(HERE, "..", "Data")
CAC40_SPOT = 8042.19
CAC40_DATE = datetime.date(2025, 2, 12)
# Global minimum of the calibration (Latin Hypercube multi-start of Model/heston_test.ipynb)
CAC40_HESTON_START = {"v0": 0.0197, "theta": 0.0438, "kappa": 1.308, "xi": 0.482, "rho": -0.605}


@contextlib.contextmanager
def _quiet():
    """ Silences the prints and closes the plots of ForwardCurve.__init__. """
    interactive = plt.isinteractive()
    plt.ioff()
    with contextlib.redirect_stdout(io.StringIO()):
        yield
    plt.close("all")
    if interactive:
        plt.ion()


def _year_fraction(label, as_of_date):
    """ 'Month-Year' expiry label -> ACT/365 time to the third Friday of that month. """
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    month, year = label.split("-")
    third_friday = datetime.date(int(year), months.index(month) + 1, 15)
    third_friday = third_friday.replace(day=15 + (4 - third_friday.weekday()) % 7)
    return (third_friday - as_of_date).days / 365


def cac40_heston(calibrate=True, verbose=True):
    """
    CAC40 market of 12/02/2025 and calibrated Heston model.
    Returns a SimpleNamespace with S0, zc_rate, forward_curve, bs_model, iv_surface, heston, calibration_rmse
    (implied vol RMSE in vol points) and the raw quotes (market_strikes, call_prices, put_prices).
    If calibrate is False, the Heston parameters are set to the known global minimum without recalibration.
    """
    df_rates = pd.read_csv(os.path.join(DATA, "EURIBOR6M_ZCRates_12022025.csv"))
    df_options = pd.read_csv(os.path.join(DATA, "CAC40_MarketOptions_12022025.csv"))
    zc_rate = PchipInterpolator(df_rates['Expiry'], df_rates['ZCRate'], extrapolate=True)
    df_options['Expiry'] = df_options['Expiry'].apply(lambda label: _year_fraction(label, CAC40_DATE))

    market_strikes, call_prices, put_prices = dict(), dict(), dict()
    for exp in np.sort(df_options['Expiry'].unique()):
        df = df_options[df_options['Expiry'] == exp].sort_values(by='Strike')
        market_strikes[exp], call_prices[exp], put_prices[exp] = list(df['Strike']), list(df['Call']), list(df['Put'])
    expiries = list(market_strikes)

    with _quiet():
        forward_curve = ForwardCurve(CAC40_SPOT, zc_rate, expiries, market_strikes, call_prices, put_prices)
    bs_model = BlackScholes(zc_rate, forward_curve)
    otm = {exp: [p if K <= forward_curve.implied_forwards[exp] else c
                 for K, c, p in zip(market_strikes[exp], call_prices[exp], put_prices[exp])] for exp in expiries}
    iv_surface = ImpliedVolatilitySurface(forward_curve, bs_model)
    iv_surface.set_implied_vol_surface(market_strikes, otm)

    heston = HestonModel(zc_rate, forward_curve, **CAC40_HESTON_START)
    loss = CalibrationLoss(iv_surface, bs_model)
    if calibrate:
        Calibrator(heston, loss, loss_name='vega_iv_mse').run('local', x0=CAC40_HESTON_START)
    rmse = 100 * np.sqrt(loss.loss(heston, 'vega_iv_mse'))
    if verbose:
        params = ", ".join(f"{k}={v:.4f}" for k, v in heston.params.items())
        print(f"CAC40 {CAC40_DATE} : S0 = {CAC40_SPOT}, {len(df_options)} quotes on {len(expiries)} expiries")
        print(f"Heston : {params} | implied vol RMSE = {rmse:.3f} vol pts")
    return SimpleNamespace(S0=CAC40_SPOT, zc_rate=zc_rate, forward_curve=forward_curve, bs_model=bs_model,
                           iv_surface=iv_surface, heston=heston, calibration_rmse=rmse,
                           market_strikes=market_strikes, call_prices=call_prices, put_prices=put_prices)


def flat_market(S0=100.0, r=0.03, q=0.01, sigma=0.20, expiries=(0.5, 1.0, 2.0)):
    """
    Flat Black-Scholes market : constant rate r, dividend yield q and volatility sigma.
    The forward curve is bootstrapped from synthetic call / put prices satisfying the parity with
    F(T) = S0 exp((r - q) T), so that the borrow + dividend yield is exactly q.
    Returns a SimpleNamespace with S0, r, q, sigma, zc_rate, forward_curve, bs_model and heston, a Heston model with
    v0 = theta = sigma^2 and a negligible vol-of-vol, i.e. a Black-Scholes model with volatility sigma.
    """
    zc_rate = lambda t: r + 0.0 * np.asarray(t, dtype=float)
    strikes = {T: list(np.arange(0.6 * S0, 1.45 * S0, 0.05 * S0)) for T in expiries}
    forwards = {T: S0 * np.exp((r - q) * T) for T in expiries}
    calls = {T: [max(forwards[T] - K, 0.0) + 0.05 * S0 for K in strikes[T]] for T in expiries}
    puts = {T: [c - np.exp(-r * T) * (forwards[T] - K) for c, K in zip(calls[T], strikes[T])] for T in expiries}
    with _quiet():
        forward_curve = ForwardCurve(S0, zc_rate, list(expiries), strikes, calls, puts)
    heston = HestonModel(zc_rate, forward_curve, v0=sigma**2, theta=sigma**2, kappa=1.0, xi=1e-4, rho=0.0)
    return SimpleNamespace(S0=S0, r=r, q=q, sigma=sigma, zc_rate=zc_rate, forward_curve=forward_curve,
                           bs_model=BlackScholes(zc_rate, forward_curve), heston=heston)


def cac40_lsv(mkt=None, mixing=1.0, surface='ssvi', n_particles=50_000, steps_per_year=52, seed=0, verbose=True):
    """
    LSV model on the CAC40 : Dupire local volatility of the SSVI (or SVI) fit of the market surface, stochastic
    variance of the calibrated Heston model with its vol-of-vol scaled by `mixing` (0 = pure local volatility),
    leverage calibrated up to the last listed expiry by the particle method (Model/local_vol.py).
    """
    from local_vol import LocalVolSurface, LSVModel
    mkt = cac40_heston(verbose=False) if mkt is None else mkt
    local_vol = (LocalVolSurface.from_ssvi(mkt.iv_surface) if surface == 'ssvi'
                 else LocalVolSurface.from_svi(mkt.iv_surface))
    model = LSVModel.from_heston(mkt.heston, local_vol, mixing)
    return model.calibrate(t_max=max(mkt.iv_surface.expiries), n_particles=n_particles,
                           steps_per_year=steps_per_year, seed=seed, verbose=verbose)
