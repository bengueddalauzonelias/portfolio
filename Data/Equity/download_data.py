# This script downloads the public equity data used by the correlation study (Models/correlation.ipynb) and stores
# it in Data/Equity (one command : python Data/Equity/download_data.py ; see Data/Equity/README.md) :
# - daily close of LVMH (MC.PA) and Hermès (RMS.PA) on Euronext Paris, raw and adjusted for dividends and splits
#                                                                              (Yahoo Finance through yfinance)
# - cash dividends paid by both stocks                                         (Yahoo Finance)
# Yahoo Finance data is free but unofficial : the CSV files are frozen so that the notebook is reproducible, and the
# download date is written in LAST_DOWNLOAD.txt.
import os
from datetime import date

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
START = "2005-01-01"
TICKERS = {"MC.PA": "LVMH", "RMS.PA": "HERMES"}


def prices(start=START, end=None):
    """ Raw close (Close) and dividend and split adjusted close (AdjClose), one column per stock. """
    raw = yf.download(list(TICKERS), start=start, end=end, auto_adjust=False, actions=False, progress=False)
    out = {}
    for field, name in [("Close", "Close"), ("Adj Close", "AdjClose")]:
        df = raw[field].rename(columns=TICKERS)[list(TICKERS.values())]
        out[name] = df.dropna(how="all")
    return out


def dividends():
    rows = []
    for ticker, name in TICKERS.items():
        d = yf.Ticker(ticker).dividends
        d.index = d.index.tz_localize(None).normalize()
        rows.append(d.rename(name))
    return pd.concat(rows, axis=1).rename_axis("Date").sort_index()


def main():
    px = prices()
    close = px["Close"].join(px["AdjClose"], lsuffix="_Close", rsuffix="_AdjClose").rename_axis("Date")
    close.to_csv(os.path.join(HERE, "LVMH_HERMES_Daily.csv"), float_format="%.6f")
    dividends().to_csv(os.path.join(HERE, "LVMH_HERMES_Dividends.csv"), float_format="%.6f")
    with open(os.path.join(HERE, "LAST_DOWNLOAD.txt"), "w") as f:
        f.write(date.today().isoformat() + "\n")
    print(f"{len(close)} days, {close.index[0].date()} -> {close.index[-1].date()}")


if __name__ == "__main__":
    main()
