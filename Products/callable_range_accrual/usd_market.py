# This script builds the USD rates market of the callable SOFR range accrual from the public data of Data/USD
# (downloaded by download_data.py) :
# - loaders for SOFR fixings, the Fed funds target range, Treasury par yields, FOMC meetings and holidays ;
# - the US government securities calendar (SIFMA) : business days, modified following, business day shifts ;
# - the Treasury zero curve, bootstrapped from the constant maturity par yields of FRED ;
# - a PROXY SOFR curve : Treasury zero curve + a constant spread matching the overnight SOFR. SOFR futures and OIS
#   quotes are not available for free (Data/USD/README.md), so this curve is an assumption, not a market curve.
# Model times are ACT/365 year fractions from the valuation date ; accruals are ACT/360 (range_accrual.py).
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", ".."))        # Projets : Fixed_Income_Models package

import numpy as np
import pandas as pd

from Fixed_Income_Models import YieldCurve, Bootstrapper, ZeroCouponRate, ParBond

DATA = os.path.join(HERE, "..", "..", "Data", "USD")
TREASURY_TENORS = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0,
                   "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0}


# ======================================================================================================
# Loaders (rates in decimal : 0.0389 = 3.89%)
# ======================================================================================================
def _read(name, **kwargs):
    return pd.read_csv(os.path.join(DATA, name), parse_dates=["Date"], index_col="Date", **kwargs)


def load_sofr():
    """ SOFR fixings (decimal) with percentiles and volumes, indexed by the business day the rate applies to. """
    df = _read("SOFR_Fixings.csv")
    df[["SOFR", "P1", "P25", "P75", "P99"]] /= 100
    return df


def load_target_range():
    """ Fed funds target range (lower, upper) and EFFR, decimal, calendar days. """
    return _read("FedFunds_TargetRange.csv").rename(columns={"DFEDTARL": "lower", "DFEDTARU": "upper"}) / 100


def load_treasury():
    """ Treasury constant maturity par yields (decimal), columns = tenors in years. """
    df = _read("UST_ParYields.csv").dropna(how="all") / 100
    return df.rename(columns=TREASURY_TENORS)


def load_fomc():
    df = pd.read_csv(os.path.join(DATA, "FOMC_Meetings.csv"), parse_dates=["Start", "Decision"])
    return df[df["Type"] == "scheduled"].reset_index(drop=True)


def load_holidays():
    return pd.to_datetime(pd.read_csv(os.path.join(DATA, "US_GovSecurities_Holidays.csv"))["Date"])


# ======================================================================================================
# Calendar
# ======================================================================================================
class USGovCalendar:
    """ US government securities calendar (SIFMA) : weekends and the holidays of Data/USD. """

    def __init__(self, holidays=None):
        holidays = load_holidays() if holidays is None else pd.to_datetime(holidays)
        self.holidays = holidays.values.astype("datetime64[D]")

    def is_business_day(self, d):
        return bool(np.is_busday(np.datetime64(pd.Timestamp(d).date()), holidays=self.holidays))

    def _offset(self, d, n, roll):
        return pd.Timestamp(np.busday_offset(np.datetime64(pd.Timestamp(d).date()), n, roll=roll,
                                             holidays=self.holidays))

    def add_business_days(self, d, n):
        """ d shifted by n business days (n < 0 : backwards) ; d is first rolled in the direction of the shift. """
        return self._offset(d, n, "forward" if n >= 0 else "backward")

    def previous_business_day(self, d):
        """ Last business day <= d. """
        return self._offset(d, 0, "backward")

    def modified_following(self, d):
        following = self._offset(d, 0, "forward")
        return following if following.month == pd.Timestamp(d).month else self.previous_business_day(d)

    def business_days(self, start, end):
        """ Business days in [start, end]. """
        days = pd.date_range(start, end, freq="D")
        return days[np.is_busday(days.values.astype("datetime64[D]"), holidays=self.holidays)]


# ======================================================================================================
# Curves
# ======================================================================================================
def year_fraction(valuation_date, dates):
    """ Model time : ACT/365 year fraction from the valuation date. """
    delta = pd.to_datetime(dates) - pd.Timestamp(valuation_date)
    return np.asarray(delta / pd.Timedelta(days=365), dtype=float) if np.ndim(dates) else delta.days / 365.0


def treasury_instruments(par_yields):
    """
    Curve instruments from one row of Treasury par yields (tenor in years -> decimal yield).
    Bills (up to 1 year) are quoted as bond-equivalent yields : zero-coupon rates with semi-annual compounding.
    Notes and bonds (2 years and more) are par bonds with semi-annual coupons.
    """
    out = []
    for tenor, y in par_yields.dropna().items():
        out.append(ZeroCouponRate(tenor, y, compounding=2) if tenor <= 1 else ParBond(tenor, y, frequency=2))
    return out


def treasury_curve(date, treasury=None, method="monotone_convex"):
    """ Treasury zero curve of `date`, bootstrapped from the par yields. Returns (curve, bootstrapper). """
    treasury = load_treasury() if treasury is None else treasury
    row = treasury.loc[pd.Timestamp(date)]
    boot = Bootstrapper(treasury_instruments(row), method=method)
    curve = boot.run()
    curve.name = f"UST {pd.Timestamp(date).date()}"
    return curve, boot


def sofr_proxy_curve(date, treasury=None, sofr=None, spread_bp=None, method="monotone_convex"):
    """
    PROXY SOFR curve of `date` : Treasury zero curve shifted by a constant spread.
    By default the spread makes the shortest pillar (1 month) equal to the last SOFR fixing (continuously
    compounded), i.e. it assumes the SOFR - Treasury spread observed at 1 month holds at every maturity.
    In the market, SOFR swap spreads to Treasuries vary with maturity : this is the main limitation of the curve.
    Returns (curve, spread in bp).
    """
    tsy, _ = treasury_curve(date, treasury, method)
    if spread_bp is None:
        sofr = load_sofr() if sofr is None else sofr
        fixings = sofr["SOFR"] if isinstance(sofr, pd.DataFrame) else sofr          # history or fixings Series
        overnight = fixings.loc[:pd.Timestamp(date)].iloc[-1]
        spread_bp = 1e4 * (np.log1p(overnight / 360) * 365 - tsy.zero_rate(tsy.times[0]))
    curve = YieldCurve(tsy.times, tsy.zero_rates + spread_bp * 1e-4, method,
                       name=f"SOFR proxy {pd.Timestamp(date).date()} (UST {spread_bp:+.1f}bp)")
    return curve, float(spread_bp)
