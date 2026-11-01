# Equity data — LVMH / Hermès

Downloaded by `python Data/Equity/download_data.py` (re-run to refresh; download date in `LAST_DOWNLOAD.txt`).
Prices are in EUR, dates are Euronext Paris trading days (ISO).

| File | Content | Source | History |
|---|---|---|---|
| `LVMH_HERMES_Daily.csv` | daily close of LVMH (`MC.PA`) and Hermès (`RMS.PA`): raw close (`*_Close`) and close adjusted for dividends and splits (`*_AdjClose`) | Yahoo Finance (`yfinance`) | 2005-01-03 → |
| `LVMH_HERMES_Dividends.csv` | cash dividends per share, by ex-date | Yahoo Finance | 2005 → |

## Notes

- **Which price for what.** Statistics on returns (volatility, correlation) use the adjusted close, so that an
  ex-dividend date is not read as a price drop. Payoffs of worst-of / best-of products are on the raw price
  (price return), dividends entering the pricing through the forward.
- **Same exchange.** Both stocks close at the same time on Euronext Paris: daily returns are synchronous and there is
  no need for the multi-day returns used for pairs listed on different time zones.
- **Yahoo Finance** is free but unofficial and its history can be revised; the files are frozen in the repository so
  that the results of the notebook are reproducible.
