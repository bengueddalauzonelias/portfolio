# USD rates data — callable SOFR range accrual

Downloaded by `Produits/callable_range_accrual/download_data.py` (re-run to refresh; download date in `LAST_DOWNLOAD.txt`).
All rates are in percent. Dates are ISO (YYYY-MM-DD); for SOFR, `Date` is the business day the rate applies to
(published the next business day).

| File | Content | Source | History | Paper |
|---|---|---|---|---|
| `SOFR_Fixings.csv` | SOFR, 1/25/75/99th percentiles of the transaction distribution, volume (USD bn), revision flag | New York Fed API | 2018-04-02 → | §3, 5, 7 |
| `SOFR_Averages_Index.csv` | 30/90/180-day compounded averages, SOFR Index (= compounding in arrears, Eq. 7, between any two dates) | New York Fed API | 2020-03-02 → | §3, 5 |
| `FedFunds_TargetRange.csv` | Fed funds target range lower / upper bound (`DFEDTARL`, `DFEDTARU`), effective fed funds rate (`EFFR`), calendar days | FRED | 2018-04-02 → | §3 (Fig. 6) |
| `UST_ParYields.csv` | Treasury constant maturity par yields, 1M 3M 6M 1Y 2Y 3Y 5Y 7Y 10Y 20Y 30Y (`DGS*`) | FRED | 2018-04-02 → | §4 |
| `FOMC_Meetings.csv` | Meeting start and decision dates, SEP flag, scheduled vs notation vote, past / upcoming | federalreserve.gov | 2021 → 2027 | §5 (curve nodes) |
| `US_GovSecurities_Holidays.csv` | Weekday holidays of the US government securities market (SIFMA) = days without a SOFR fixing | QuantLib `UnitedStates.GovernmentBond` | 2018 → 2040 | §3.3, payoff |

## Notes

- **Missing values** are empty cells (FRED holidays, EFFR not yet published for the last day).
- **Good Friday**: SIFMA sometimes recommends an early close instead of a full close (2021, 2023, 2026) and QuantLib
  then treats it as a business day, but no SOFR is published on those days: every Good Friday is added to the
  holiday file (checked: every business day of the file since 2018 has a SOFR fixing).
- **FOMC**: the Fed calendar page covers the last five years and the next year only. Meetings after 2027 (the note
  runs to 2029) are not yet scheduled; assume the usual 8 meetings per year at the usual weeks if needed.
- **SOFR on 2018-04-02** is the first published rate; the series is official from 2018-04-03.

## Not included: CME SOFR futures (SR1, SR3) and options on SR3

The pipeline of the paper uses CME settlement prices for the curve (§5), the smile (§6) and the hedge (§9).
They are **not** downloaded here: CME blocks automated requests and its Data Terms of Use explicitly prohibit
scripts or scrapers. Options to obtain them legitimately:

1. **Manual daily download** from the CME settlements pages, saved as
   `Data/USD/CME/SR3_Futures_YYYYMMDD.csv` and `Data/USD/CME/SR3_Options_YYYYMMDD.csv` (check the terms of use for
   personal / academic use first);
2. **School access**: Bloomberg or Refinitiv terminals, if available, which also give
   SOFR OIS swap rates and swaption / cap volatilities;
3. **CME DataMine** (paid, historical end-of-day settlements).

Until then, the SOFR curve can be built from SOFR fixings + Treasury bills as a proxy, and volatilities must be
illustrative and labelled as such (as in `Produits/rates_market.py`).
