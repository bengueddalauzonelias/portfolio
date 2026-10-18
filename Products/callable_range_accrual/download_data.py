# This script downloads the public market data used to price the callable SOFR range accrual and stores it in
# Data/USD (one command, re-run to refresh ; see Data/USD/README.md) :
# - SOFR fixings, percentiles and volumes since 2018-04-02    (New York Fed API)
# - SOFR 30/90/180-day averages and SOFR Index                (New York Fed API)
# - Fed funds target range and EFFR                           (FRED : DFEDTARL, DFEDTARU, EFFR)
# - Treasury constant maturity par yields, 1M to 30Y          (FRED : DGS1MO ... DGS30)
# - FOMC meeting dates, past and scheduled                    (federalreserve.gov)
# - US government securities holidays (SOFR fixing calendar)  (QuantLib, UnitedStates.GovernmentBond)
# SR1 / SR3 futures and options on SR3 are NOT downloaded : CME's terms of use forbid automated collection.
import gzip
import io
import json
import os
import re
import urllib.request
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "Data", "USD")
START = "2018-04-02"                                   # first SOFR publication (rate of 2018-04-02)
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}

TREASURY_SERIES = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                                      "Dec"], start=1)}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":                              # FRED may answer gzip-compressed
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def fred(series, start=START):
    """ FRED series as a DataFrame indexed by date ('.' = missing value). """
    frames = []
    for s in series:                                      # one request per series : FRED rejects some multi-id requests
        df = pd.read_csv(io.StringIO(fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}")),
                         parse_dates=["observation_date"], na_values=".")
        frames.append(df.rename(columns={"observation_date": "Date"}).set_index("Date"))
    df = pd.concat(frames, axis=1).sort_index()
    return df[df.index >= start]


def nyfed(kind, end):
    url = f"https://markets.newyorkfed.org/api/rates/secured/{kind}/search.json?startDate={START}&endDate={end}"
    return pd.DataFrame(json.loads(fetch(url))["refRates"])


def sofr_fixings(end):
    df = nyfed("sofr", end)
    df = df.rename(columns={"effectiveDate": "Date", "percentRate": "SOFR", "percentPercentile1": "P1",
                            "percentPercentile25": "P25", "percentPercentile75": "P75",
                            "percentPercentile99": "P99", "volumeInBillions": "VolumeBn",
                            "revisionIndicator": "Revised"})
    return df[["Date", "SOFR", "P1", "P25", "P75", "P99", "VolumeBn", "Revised"]].sort_values("Date")


def sofr_averages(end):
    df = nyfed("sofrai", end)
    df = df.rename(columns={"effectiveDate": "Date", "average30day": "Avg30d", "average90day": "Avg90d",
                            "average180day": "Avg180d", "index": "SOFRIndex", "revisionIndicator": "Revised"})
    return df[["Date", "Avg30d", "Avg90d", "Avg180d", "SOFRIndex", "Revised"]].sort_values("Date")


def fomc_meetings():
    """
    Scheduled FOMC meetings from the Fed calendar page (current year, next year and the last five years).
    Decision date = last day of the meeting. SEP = Summary of Economic Projections released (* on the page).
    """
    html = fetch("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    rows = []
    for panel in re.split(r"(?=<h4><a[^>]*>\d{4} FOMC Meetings)", html)[1:]:
        year = int(re.search(r"(\d{4}) FOMC Meetings", panel).group(1))
        pattern = r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<'
        for month_txt, day_txt in re.findall(pattern, panel, flags=re.S):
            months = [MONTHS[m[:3]] for m in re.findall(r"[A-Z][a-z]+", month_txt) if m[:3] in MONTHS]   # "Jan/Feb"
            days = [int(d) for d in re.findall(r"\d+", day_txt)]
            if not months or not days:
                continue
            notation = "notation" in day_txt.lower() or "unscheduled" in day_txt.lower()
            first = date(year, months[0], days[0])
            last = date(year, months[-1], days[-1])
            rows.append({"Start": first, "Decision": last, "SEP": "*" in day_txt,
                         "Type": "unscheduled/notation vote" if notation else "scheduled"})
    df = pd.DataFrame(rows).drop_duplicates().sort_values("Decision")
    df["Status"] = ["past" if d < date.today() else "upcoming" for d in df["Decision"]]
    return df


def government_bond_holidays(first_year=2018, last_year=2040):
    """
    Weekday holidays of the US government securities market (SIFMA), i.e. days without a SOFR fixing.
    QuantLib treats Good Friday as a business day in the years SIFMA only recommends an early close (2021, 2023,
    2026), but the New York Fed publishes no SOFR on those days : every Good Friday is added.
    """
    import QuantLib as ql
    from dateutil.easter import easter
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    days = {date(d.year(), d.month(), d.dayOfMonth())
            for d in cal.holidayList(ql.Date(1, 1, first_year), ql.Date(31, 12, last_year), False)}
    days |= {easter(y) - pd.Timedelta(days=2) for y in range(first_year, last_year + 1)}
    return pd.DataFrame({"Date": sorted(days)})


def main():
    os.makedirs(OUT, exist_ok=True)
    end = date.today().isoformat()
    files = {
        "SOFR_Fixings.csv": sofr_fixings(end),
        "SOFR_Averages_Index.csv": sofr_averages(end),
        "FedFunds_TargetRange.csv": fred(["DFEDTARL", "DFEDTARU", "EFFR"]).reset_index(),
        "UST_ParYields.csv": fred(TREASURY_SERIES).reset_index(),
        "FOMC_Meetings.csv": fomc_meetings(),
        "US_GovSecurities_Holidays.csv": government_bond_holidays(),
    }
    for name, df in files.items():
        df.to_csv(os.path.join(OUT, name), index=False)
        print(f"{name:32s} {len(df):6d} rows")
    with open(os.path.join(OUT, "LAST_DOWNLOAD.txt"), "w") as f:
        f.write(f"{end}\n")


if __name__ == "__main__":
    main()
