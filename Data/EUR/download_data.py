# This script downloads the public data used to build the EUR (€STR) curve and stores it in Data/EUR
# (one command : python Data/EUR/download_data.py ; see Data/EUR/README.md) :
# - €STR fixings with percentiles, volume, number of transactions and banks   (ECB Data Portal API, dataset EST)
# - compounded €STR averages (1W to 12M) and the €STR index                    (ECB API, EST)
# - ECB key rates : deposit facility, main refinancing, marginal lending        (ECB API, FM)
# - ECB Governing Council monetary policy meetings (upcoming) and the effective dates of past rate changes
#                                                                              (ecb.europa.eu calendar + FM)
# - euro area AAA government yield curve : spot rates and Svensson parameters  (ECB API, YC)
# - Euribor 1M 3M 6M 12M, monthly averages (daily Euribor is not on the ECB API) (ECB API, FM)
# - TARGET holidays (calendar of €STR fixings)                                 (QuantLib TARGET)
# €STR OIS swap quotes are not available for free : ESTR_OIS_Quotes_TEMPLATE.csv is a template to fill by hand.
import io
import os
import re
import urllib.request
from datetime import date, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
START = "2019-10-01"                                    # first €STR publication
API = "https://data-api.ecb.europa.eu/service/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)", "Accept": "text/csv"}

AAA_TENORS = ["3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y",
              "25Y", "30Y"]
SVENSSON = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
OIS_TENORS = ["1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y",
              "8Y", "9Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"]


def fetch(url, accept_csv=True):
    headers = HEADERS if accept_csv else {"User-Agent": HEADERS["User-Agent"]}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read().decode("utf-8", errors="ignore")


def ecb(key, start=START, column="KEY"):
    """ ECB Data Portal series (several keys with '+'), one column per series, indexed by date. """
    df = pd.read_csv(io.StringIO(fetch(f"{API}/{key}?startPeriod={start}&format=csvdata")))
    return df.pivot(index="TIME_PERIOD", columns=column, values="OBS_VALUE").rename_axis(index="Date", columns=None)


def estr_fixings():
    names = {"WT": "ESTR", "R25": "R25", "R75": "R75", "TT": "VolumeMn", "NT": "Transactions", "NB": "Banks",
             "VL": "Top5Share", "RP": "PublicationType", "CM": "CalculationMethod"}
    df = ecb(f"EST/B.EU000A2X2A25.{'+'.join(names)}", column="DATA_TYPE_EST")
    return df[list(names)].rename(columns=names)


def estr_compounded():
    names = {"EU000A2QQF08": "Index", "EU000A2QQF16": "Avg1W", "EU000A2QQF24": "Avg1M", "EU000A2QQF32": "Avg3M",
             "EU000A2QQF40": "Avg6M", "EU000A2QQF57": "Avg12M"}
    df = ecb(f"EST/B.{'+'.join(names)}.CI+CR", column="BENCHMARK_ITEM")
    return df[list(names)].rename(columns=names)


def key_rates(start="2019-01-01"):
    names = {"DFR": "DFR", "MRR_FR": "MRO", "MLFR": "MLF"}
    df = ecb(f"FM/D.U2.EUR.4F.KR.{'+'.join(names)}.LEV", start, column="PROVIDER_FM_ID")
    return df[list(names)].rename(columns=names)


def aaa_curve():
    keys = [f"SR_{t}" for t in AAA_TENORS] + SVENSSON
    df = ecb(f"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{'+'.join(keys)}", column="DATA_TYPE_FM")
    return df[keys]


def euribor_monthly(start="2015-01"):
    names = {"EURIBOR1MD_": "Euribor1M", "EURIBOR3MD_": "Euribor3M", "EURIBOR6MD_": "Euribor6M",
             "EURIBOR1YD_": "Euribor12M"}
    df = ecb(f"FM/M.U2.EUR.RT.MM.{'+'.join(names)}.HSTA", start, column="PROVIDER_FM_ID")
    return df[list(names)].rename(columns=names).rename_axis(index="Month")


def ecb_meetings(rates):
    """
    Upcoming monetary policy meetings (decision = day 2, followed by the press conference) from the ECB calendar page,
    and the past changes of the deposit facility rate. Rate decisions take effect on the Wednesday after the meeting
    (start of the next reserve maintenance period) : Effective = first Wednesday after the decision (checked on the
    rate changes since 2023).
    """
    page = fetch("https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html", accept_csv=False)
    text = re.sub(r"<[^>]+>", "\n", page)
    rows = []
    for d, label in re.findall(r"(\d{2}/\d{2}/\d{4})\s*\n+\s*([^\n]+)", text):
        if "monetary policy meeting" in label and "non-monetary" not in label and "Day 1" not in label:
            decision = pd.to_datetime(d, dayfirst=True).date()
            effective = decision + timedelta(days=(2 - decision.weekday()) % 7 or 7)
            rows.append({"Decision": decision, "Effective": effective, "Status": "upcoming",
                         "DFR before": None, "DFR after": None})
    dfr = rates["DFR"].astype(float)
    changes = dfr[dfr.diff() != 0].iloc[1:]
    for eff, value in changes.items():
        eff = pd.Timestamp(eff).date()
        rows.append({"Decision": None, "Effective": eff, "Status": "past change",
                     "DFR before": float(dfr.loc[:pd.Timestamp(eff).strftime("%Y-%m-%d")].iloc[-2]),
                     "DFR after": float(value)})
    df = pd.DataFrame(rows).drop_duplicates(subset=["Effective"]).sort_values("Effective").reset_index(drop=True)
    return df


def target_holidays(first_year=2019, last_year=2060):
    """ TARGET closing days on weekdays (no €STR fixing) : New Year, Good Friday, Easter Monday, 1 May, 25-26 Dec. """
    import QuantLib as ql
    days = ql.TARGET().holidayList(ql.Date(1, 1, first_year), ql.Date(31, 12, last_year), False)
    return pd.DataFrame({"Date": [date(d.year(), d.month(), d.dayOfMonth()) for d in days]})


def ois_template(path):
    """
    Template for manually entered €STR OIS par rates (percent) : copy it to ESTR_OIS_Quotes_YYYYMMDD.csv (quote date)
    and fill the Rate column ; empty rates are ignored. Never overwritten.
    """
    if not os.path.exists(path):
        pd.DataFrame({"Tenor": OIS_TENORS, "Rate": [None] * len(OIS_TENORS)}).to_csv(path, index=False)


def main():
    files = {}
    files["ESTR_Fixings.csv"] = estr_fixings()
    files["ESTR_Compounded_Index.csv"] = estr_compounded()
    files["ECB_KeyRates.csv"] = rates = key_rates()
    files["ECB_Meetings.csv"] = ecb_meetings(rates)
    files["ECB_AAA_YieldCurve.csv"] = aaa_curve()
    files["Euribor_Monthly.csv"] = euribor_monthly()
    files["TARGET_Holidays.csv"] = target_holidays()
    for name, df in files.items():
        df.to_csv(os.path.join(HERE, name), index=not isinstance(df.index, pd.RangeIndex))
        print(f"{name:28s} {len(df):6d} rows")
    ois_template(os.path.join(HERE, "ESTR_OIS_Quotes_TEMPLATE.csv"))
    with open(os.path.join(HERE, "LAST_DOWNLOAD.txt"), "w") as f:
        f.write(f"{date.today().isoformat()}\n")


if __name__ == "__main__":
    main()
