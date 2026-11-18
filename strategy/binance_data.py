# Download and cache of Binance public data (data.binance.vision) for the spot / perpetual cash and carry :
# spot klines, USD-M perpetual klines, premium index klines and funding rates, on the 8-hour funding grid.
# Monthly archives are used for the complete months and daily archives for the current month. Every file is
# downloaded once and cached in Data/Crypto/raw ; the merged panel is cached in Data/Crypto/<symbol>_8h.csv.
import io
import os
import time
import zipfile
import numpy as np
import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data", "Crypto")
RAW = os.path.join(DATA, "raw")
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                 "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
_SESSION = requests.Session()

# kind -> (path of the monthly / daily archives, file name pattern)
SOURCES = {
    "spot":     ("spot/{period}/klines/{symbol}/8h", "{symbol}-8h-{date}.zip"),
    "perp":     ("futures/um/{period}/klines/{symbol}/8h", "{symbol}-8h-{date}.zip"),
    "premium":  ("futures/um/{period}/premiumIndexKlines/{symbol}/8h", "{symbol}-8h-{date}.zip"),
    "funding":  ("futures/um/{period}/fundingRate/{symbol}", "{symbol}-fundingRate-{date}.zip"),
}


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def _download(kind, symbol, period, date, refresh=False):
    """
    Raw csv of one archive (period 'monthly' with date 'YYYY-MM', or 'daily' with date 'YYYY-MM-DD'), cached in
    Data/Crypto/raw. Returns None if the archive does not exist (e.g. a day not published yet).
    """
    folder, name = SOURCES[kind]
    folder = folder.format(period=period, symbol=symbol)
    name = name.format(symbol=symbol, date=date)
    cache = os.path.join(RAW, kind, symbol, name.replace(".zip", ".csv"))
    if os.path.exists(cache) and not refresh:
        return cache
    for attempt in range(5):                                   # data.binance.vision times out from time to time
        try:
            response = _SESSION.get(f"{BASE_URL}/{folder}/{name}", timeout=30)
            break
        except requests.exceptions.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with open(cache, "wb") as out:
            out.write(archive.read(archive.namelist()[0]))
    return cache


def _archives(start, end):
    """ (period, date) of the archives covering [start, end] : complete months, then the days of the last month. """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    first_day_of_month = end.replace(day=1)
    months = pd.period_range(start.to_period("M"), (first_day_of_month - pd.Timedelta(days=1)).to_period("M"), freq="M")
    days = pd.date_range(max(first_day_of_month, start), end, freq="D")
    return [("monthly", str(m)) for m in months] + [("daily", d.strftime("%Y-%m-%d")) for d in days]


def _to_timestamp(values):
    """ Binance times in ms, or in microseconds for the spot archives since 2025. """
    values = pd.to_numeric(values).astype("int64")
    values = np.where(values > 10**14, values // 1000, values)
    return pd.to_datetime(values, unit="ms", utc=True)


def _read(path, kind):
    first = open(path, encoding="utf-8").readline()
    has_header = not first.split(",")[0].strip().lstrip("-").isdigit()
    if kind == "funding":
        df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None,
                                                              names=["calc_time", "funding_interval_hours", "last_funding_rate"])
        df["time"] = _to_timestamp(df["calc_time"]).round("h")          # calc_time is a few ms after the hour
        return df[["time", "funding_interval_hours", "last_funding_rate"]]
    df = pd.read_csv(path, header=0 if has_header else None)
    df.columns = KLINE_COLUMNS
    df["time"] = _to_timestamp(df["open_time"])
    return df[["time", "open", "close"]]


def load_raw(kind, symbol, start="2020-01-01", end=None, refresh=False, verbose=True):
    """ One kind of data ('spot', 'perp', 'premium', 'funding') for one symbol, concatenated over [start, end]. """
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames, missing = [], []
    for period, date in _archives(start, end):
        path = _download(kind, symbol, period, date, refresh=refresh)
        if path is None:
            missing.append(date)
            continue
        frames.append(_read(path, kind))
    if verbose and missing:
        print(f"{kind} {symbol} : {len(missing)} archive(s) not published ({missing[0]} ... {missing[-1]})")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    return df.set_index("time")


# ----------------------------------------------------------------------
# Panel on the funding grid
# ----------------------------------------------------------------------
def load_panel(symbol, start="2020-01-01", end=None, refresh=False, verbose=True):
    """
    Panel of one symbol on the 8-hour funding grid (00:00, 08:00, 16:00 UTC), indexed by the funding time t_k :
    - spot, perp      : open price of the 8h kline starting at t_k, i.e. the price at t_k
    - premium         : premium index (perp / index - 1) at t_k
    - funding         : funding rate paid at t_k by the longs to the shorts (fraction of the notional) ; when the
                        funding interval is shorter than 8 hours, the payments of the 8-hour window ending at t_k are
                        summed
    - funding_interval: hours between two payments (8 for BTC and ETH, except some periods on some symbols)
    The panel is cached in Data/Crypto/<symbol>_8h.csv ; refresh=True downloads the archives again.
    """
    cache = os.path.join(DATA, f"{symbol}_8h.csv")
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if os.path.exists(cache) and not refresh:                # refresh=True to extend the panel with new archives
        panel = pd.read_csv(cache, index_col=0, parse_dates=True)
        if panel.index.tz is None:
            panel.index = panel.index.tz_localize("UTC")
        return panel.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    spot = load_raw("spot", symbol, start, end, refresh, verbose)["open"].rename("spot")
    perp = load_raw("perp", symbol, start, end, refresh, verbose)["open"].rename("perp")
    premium = load_raw("premium", symbol, start, end, refresh, verbose)["open"].rename("premium")
    funding = load_raw("funding", symbol, start, end, refresh, verbose)
    # sum the payments of each 8-hour window (t_k - 8h, t_k] on the funding grid
    window_end = funding.index.ceil("8h")
    funding_8h = funding["last_funding_rate"].groupby(window_end).sum().rename("funding")
    interval = funding["funding_interval_hours"].groupby(window_end).min().rename("funding_interval")
    panel = pd.concat([spot, perp, premium, funding_8h, interval], axis=1, join="inner").astype(float)
    panel = panel[panel.index.hour.isin([0, 8, 16])]
    panel.index.name = "time"
    os.makedirs(DATA, exist_ok=True)
    panel.to_csv(cache)
    return panel.loc[pd.Timestamp(start, tz="UTC"):]


def load_panels(symbols=("BTCUSDT", "ETHUSDT"), start="2020-01-01", end=None, refresh=False, verbose=True):
    """ {symbol: panel} for several symbols, restricted to their common dates. """
    panels = {s: load_panel(s, start, end, refresh, verbose) for s in symbols}
    common = panels[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(panels[s].index)
    return {s: p.loc[common] for s, p in panels.items()}


if __name__ == "__main__":
    for s, p in load_panels().items():
        print(s, p.shape, p.index[0], p.index[-1])
        print(p.describe().T)
