# This script downloads the daily adjusted close of a universe of large caps listed on Euronext Paris, used by the
# test of constant correlation on baskets of two or three stocks (Models/Correlation/constant_correlation.ipynb) :
# one command, python Data/Equity/download_universe.py (see Data/Equity/README.md). All the stocks close at the same
# time on the same exchange, so daily returns are synchronous. Yahoo Finance data is free but unofficial : the CSV
# file is frozen so that the notebook is reproducible, and the download date is written in LAST_DOWNLOAD_UNIVERSE.txt.
# AXA (CS.PA) and Engie (ENGI.PA) are left out : stale Yahoo prices for AXA in 2005-2007 and a pre-merger (Gaz de
# France) series for Engie before 2008.
import os
from datetime import date

import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
START = "2005-01-01"
TICKERS = {
    "MC.PA": "LVMH", "RMS.PA": "HERMES", "KER.PA": "KERING", "OR.PA": "LOREAL", "BN.PA": "DANONE",
    "RI.PA": "PERNOD", "BNP.PA": "BNP", "GLE.PA": "SOCGEN", "ACA.PA": "CREDITAGRICOLE",
    "AIR.PA": "AIRBUS", "SAF.PA": "SAFRAN", "HO.PA": "THALES", "TTE.PA": "TOTAL",
    "SAN.PA": "SANOFI", "DG.PA": "VINCI", "EN.PA": "BOUYGUES", "SGO.PA": "SAINTGOBAIN", "CAP.PA": "CAPGEMINI",
    "DSY.PA": "DASSAULTSYS", "SU.PA": "SCHNEIDER", "LR.PA": "LEGRAND", "ORA.PA": "ORANGE", "AI.PA": "AIRLIQUIDE",
}


def main():
    raw = yf.download(list(TICKERS), start=START, auto_adjust=False, actions=False, progress=False)
    px = raw["Adj Close"].rename(columns=TICKERS)[list(TICKERS.values())].dropna(how="all").rename_axis("Date")
    px.to_csv(os.path.join(HERE, "Universe_Daily_AdjClose.csv"), float_format="%.6f")
    with open(os.path.join(HERE, "LAST_DOWNLOAD_UNIVERSE.txt"), "w") as f:
        f.write(date.today().isoformat() + "\n")
    print(f"{len(px)} days, {px.index[0].date()} -> {px.index[-1].date()}")
    print(px.apply(lambda s: s.first_valid_index().date()).to_string())


if __name__ == "__main__":
    main()
