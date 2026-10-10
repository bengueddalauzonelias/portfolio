# EUR rates data — €STR curve

Downloaded by `python Data/EUR/download_data.py` (re-run to refresh; download date in `LAST_DOWNLOAD.txt`).
Rates are in percent; dates are ISO. For €STR, `Date` is the TARGET business day the rate applies to (published at
08:00 CET the next business day).

| File | Content | Source | History |
|---|---|---|---|
| `ESTR_Fixings.csv` | €STR (`ESTR`), 25th / 75th volume percentiles, total volume (EUR mn), number of transactions and of banks, share of the 5 largest banks, publication type and calculation method (0 = normal) | ECB Data Portal API, dataset `EST` | 2019-10-01 → |
| `ESTR_Compounded_Index.csv` | €STR index (1 Oct 2019 = 100) and compounded averages over 1W, 1M, 3M, 6M, 12M | ECB API, `EST` | 2019-10-01 → |
| `ECB_KeyRates.csv` | deposit facility (DFR), main refinancing (MRO), marginal lending (MLF), calendar days | ECB API, `FM` | 2019-01-01 → |
| `ECB_Meetings.csv` | upcoming monetary policy meetings (decision day) and **effective date** of the decision; past DFR changes with their effective dates | ecb.europa.eu calendar + `FM` | 2019 → 2028 |
| `ECB_AAA_YieldCurve.csv` | euro area AAA government curve: spot rates 3M to 30Y (continuously compounded) and the Svensson parameters `BETA0..3`, `TAU1..2` | ECB API, `YC` | 2019-10-01 → |
| `Euribor_Monthly.csv` | Euribor 1M, 3M, 6M, 12M, monthly averages of daily fixings | ECB API, `FM` | 2015-01 → |
| `TARGET_Holidays.csv` | weekday TARGET closing days (no €STR fixing) | QuantLib `TARGET` | 2019 → 2060 |
| `ESTR_OIS_Quotes_TEMPLATE.csv` | template for €STR OIS par rates, to fill by hand (see below) | — | — |

## Notes

- **Effective dates.** ECB rate decisions apply from the Wednesday after the meeting (start of the reserve maintenance
  period); checked on every DFR change since 2023. These are the dates where the overnight forward of the curve jumps.
- **The calendar page** lists the upcoming meetings only (to end-2028). Past meetings without a rate change are not
  stored; past changes are taken from the DFR series.
- **Daily Euribor** is published by EMMI under licence and is not on the ECB API: only monthly averages are free.

## Not available for free: €STR OIS quotes

No free API or CSV provides €STR OIS swap rates (they come from brokers, Bloomberg / Refinitiv, or the Eurex / ICE
€STR futures, which are licensed). To build the curve from real market quotes:

1. copy `ESTR_OIS_Quotes_TEMPLATE.csv` to `ESTR_OIS_Quotes_YYYYMMDD.csv` (quote date);
2. fill the `Rate` column (percent, annual ACT/360 fixed vs compounded €STR, spot start T+2) with the quotes you can
   read for that day, e.g. on a public market page; leave unknown tenors empty;
3. the notebook `Fixed_Income_Models/estr_curve.ipynb` picks the most recent filled file automatically.

Without such a file, the notebook builds **proxy** quotes from the AAA government curve shifted to the €STR level.
