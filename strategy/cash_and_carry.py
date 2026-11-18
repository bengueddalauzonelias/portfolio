# Spot / perpetual cash and carry on Binance : long spot, short USD-M perpetual, same quantity. The position earns the
# funding paid by the longs to the shorts every 8 hours, plus the change of the basis (perp - spot) while it is held.
# Strategies : always invested (benchmark), funding threshold with hysteresis, dynamic allocation between symbols.
# Margin : cross / portfolio margin, the spot being the collateral of the short perpetual (no rebalancing).
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 3 * 365          # funding every 8 hours, crypto markets open every day
INTEREST_8H = 0.0001                # Binance interest rate component of the funding (0.01% per 8 hours)
CLAMP_8H = 0.0005                   # clamp of (interest - premium) in the Binance funding formula


# ----------------------------------------------------------------------
# Returns of the position
# ----------------------------------------------------------------------
def period_returns(panel):
    """
    Returns over (t_k, t_k+1] of one unit of capital invested in the cash and carry at t_k : buy 1/S_k unit of spot,
    sell the same quantity of perpetual. Indexed by t_k+1 :
    - funding : f_k+1 * F_k+1 / S_k, the funding received by the short at t_k+1 on its notional
    - basis   : [(S_k+1 - S_k) - (F_k+1 - F_k)] / S_k, i.e. minus the change of the basis F - S, per unit of capital
    - total   : funding + basis
    Missing dates of the archives are skipped : the return then covers several periods but only the funding paid at
    t_k+1 is counted.
    """
    S, F, f = panel["spot"], panel["perp"], panel["funding"]
    funding = (f * F / S.shift(1)).iloc[1:]
    basis = (((S - S.shift(1)) - (F - F.shift(1))) / S.shift(1)).iloc[1:]
    return pd.DataFrame({"funding": funding, "basis": basis, "total": funding + basis})


# ----------------------------------------------------------------------
# Signals : expected funding of the next period, known at t_k
# ----------------------------------------------------------------------
def funding_signal(panel, method="last", window=21):
    """
    Expected funding of the next payment, per 8 hours, from the information available at t_k :
    - 'last'    : the funding just paid at t_k (the funding is very persistent)
    - 'mean'    : mean of the last `window` payments (21 = one week)
    - 'ewm'     : exponentially weighted mean with a half-life of `window` payments
    - 'premium' : Binance formula applied to the premium index at t_k, P + clamp(I - P, -0.05%, 0.05%)
    """
    f = panel["funding"]
    if method == "last":
        return f
    if method == "mean":
        return f.rolling(window, min_periods=1).mean()
    if method == "ewm":
        return f.ewm(halflife=window, min_periods=1).mean()
    if method == "premium":
        p = panel["premium"]
        return p + np.clip(INTEREST_8H - p, -CLAMP_8H, CLAMP_8H)
    raise ValueError("method must be 'last', 'mean', 'ewm' or 'premium'.")


# ----------------------------------------------------------------------
# Positions : weights held over (t_k, t_k+1], decided at t_k
# ----------------------------------------------------------------------
def always_invested(signals):
    """ Benchmark : fully invested in every symbol, the capital being split equally. """
    return pd.DataFrame(1.0 / signals.shape[1], index=signals.index, columns=signals.columns)


def threshold_positions(signals, entry, exit=None):
    """
    One position per symbol (the capital being split equally) with hysteresis : enter when the signal is above
    `entry`, stay while it is above `exit` (default : entry). Thresholds are per annum (e.g. 0.05 = 5% a year).
    """
    entry_8h = entry / PERIODS_PER_YEAR
    exit_8h = (entry if exit is None else exit) / PERIODS_PER_YEAR
    held = np.zeros(signals.shape, dtype=bool)
    values = signals.to_numpy()
    state = np.zeros(signals.shape[1], dtype=bool)
    for k in range(len(values)):
        state = np.where(state, values[k] > exit_8h, values[k] > entry_8h)
        held[k] = state
    return pd.DataFrame(held / signals.shape[1], index=signals.index, columns=signals.columns)


def dynamic_allocation(signals, entry, exit=None, mode="best", switch=0.0):
    """
    Dynamic allocation between the symbols. A symbol is eligible when its signal is above `entry`, or above `exit`
    if it is already held. Among the eligible symbols :
    - mode 'best'  : all the capital goes to the highest signal ; the current symbol is only replaced if the new one
                     is better by more than `switch` (per annum), to avoid switching back and forth
    - mode 'equal' : the capital is split equally between the eligible symbols
    The capital stays in cash when no symbol is eligible.
    """
    entry_8h = entry / PERIODS_PER_YEAR
    exit_8h = (entry if exit is None else exit) / PERIODS_PER_YEAR
    switch_8h = switch / PERIODS_PER_YEAR
    values = signals.to_numpy()
    weights = np.zeros(values.shape)
    current = np.zeros(values.shape[1], dtype=bool)
    for k in range(len(values)):
        eligible = np.where(current, values[k] > exit_8h, values[k] > entry_8h)
        if mode == "equal":
            current = eligible
            if eligible.any():
                weights[k] = eligible / eligible.sum()
            continue
        if not eligible.any():
            current[:] = False
            continue
        best = int(np.argmax(np.where(eligible, values[k], -np.inf)))
        held = np.flatnonzero(current)
        if held.size and eligible[held[0]] and values[k, best] <= values[k, held[0]] + switch_8h:
            best = held[0]
        current[:] = False
        current[best] = True
        weights[k, best] = 1.0
    return pd.DataFrame(weights, index=signals.index, columns=signals.columns)


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
def backtest(panels, weights, cash_rate=0.0, fees=(0.0, 0.0)):
    """
    Portfolio of cash and carry positions. weights : DataFrame (index t_k, one column per symbol) of the fraction of
    the capital invested in each symbol over (t_k, t_k+1] ; the rest earns cash_rate (per annum).
    fees = (spot, perp) proportional fees paid on each leg when the weight of a symbol changes (0 by default).
    Returns a DataFrame indexed by t_k+1 with the portfolio return, its funding and basis parts, the cash part, the
    fees and the equity curve (starting at 1).
    """
    symbols = list(weights.columns)
    returns = {s: period_returns(panels[s]) for s in symbols}
    index = returns[symbols[0]].index
    w = weights.reindex(columns=symbols).shift(1).loc[index].fillna(0.0)        # weight decided at t_k, applied at t_k+1
    funding = sum(w[s] * returns[s]["funding"] for s in symbols)
    basis = sum(w[s] * returns[s]["basis"] for s in symbols)
    gaps = index.to_series().diff().dt.total_seconds().fillna(8 * 3600) / (8 * 3600)
    cash = (1.0 - w.sum(axis=1)) * cash_rate / PERIODS_PER_YEAR * gaps
    turnover = weights.reindex(columns=symbols).diff().abs().fillna(weights.abs()).shift(1).loc[index].fillna(0.0)
    cost = turnover.sum(axis=1) * (fees[0] + fees[1])
    total = funding + basis + cash - cost
    out = pd.DataFrame({"return": total, "funding": funding, "basis": basis, "cash": cash, "fees": cost,
                        "exposure": w.sum(axis=1)})
    out["equity"] = (1.0 + out["return"]).cumprod()
    return out


def margin_ratio(panel, weights, haircut=0.95, maintenance_margin=0.004):
    """
    Cross / portfolio margin of each holding episode, the spot being the collateral of the short perpetual and the
    quantity being fixed at the entry : q = 1 / S_entry per unit of capital.
    equity(t)      = haircut * q * S_t + q * (F_entry - F_t) + funding received since the entry
    maintenance(t) = maintenance_margin * q * F_t
    Returns the series of equity / maintenance while the position is held (liquidation below 1), and the rise of the
    index from the entry that would trigger the liquidation, F/F_entry ~ 1 / (1 - haircut + maintenance_margin).
    """
    S, F, f = panel["spot"], panel["perp"], panel["funding"]
    held = weights.reindex(S.index).fillna(0.0) > 0
    episode = (held & ~held.shift(1, fill_value=False)).cumsum().where(held)
    ratio = pd.Series(np.nan, index=S.index)
    for _, dates in S.index.to_series().groupby(episode):
        k0 = dates.index
        S_e, F_e = S.loc[k0[0]], F.loc[k0[0]]
        q = 1.0 / S_e
        # the position decided at t_k0 is held until the next date after the last held date
        span = S.index[S.index.get_loc(k0[0]):S.index.get_loc(k0[-1]) + 2]
        accrued = (f.loc[span] * q * F.loc[span]).cumsum() - f.loc[span[0]] * q * F.loc[span[0]]
        equity = haircut * q * S.loc[span] + q * (F_e - F.loc[span]) + accrued
        ratio.loc[span] = np.fmin(ratio.loc[span].fillna(np.inf), equity / (maintenance_margin * q * F.loc[span]))
    return ratio.replace(np.inf, np.nan), 1.0 / (1.0 - haircut + maintenance_margin)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def performance(result, weights=None, cash_rate=0.0):
    """ Annualised statistics of a backtest (output of `backtest`). """
    r = result["return"]
    years = (r.index[-1] - r.index[0]).total_seconds() / (365 * 24 * 3600)
    equity = result["equity"]
    drawdown = equity / equity.cummax() - 1.0
    excess = r - cash_rate / PERIODS_PER_YEAR
    stats = {
        "annual return (%)": 100 * (equity.iloc[-1] ** (1 / years) - 1),
        "annual volatility (%)": 100 * r.std() * np.sqrt(PERIODS_PER_YEAR),
        "Sharpe ratio": excess.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR) if r.std() > 0 else np.nan,
        "max drawdown (%)": 100 * drawdown.min(),
        "funding (% p.a.)": 100 * result["funding"].sum() / years,
        "basis (% p.a.)": 100 * result["basis"].sum() / years,
        "time invested (%)": 100 * (result["exposure"] > 0).mean(),
        "positive periods (%)": 100 * (r[result["exposure"] > 0] > 0).mean(),
        "worst period (%)": 100 * r.min(),
    }
    if weights is not None:
        entries = (weights.diff().clip(lower=0).fillna(weights) > 0).sum().sum()
        stats["number of entries"] = int(entries)
        stats["average holding (days)"] = (stats["time invested (%)"] / 100 * years * 365 / entries) if entries else np.nan
    return stats


def yearly_returns(result):
    """ Calendar-year returns of a backtest (%), with the funding and basis parts. """
    grouped = result.groupby(result.index.year)
    return pd.DataFrame({"return (%)": 100 * (grouped["return"].apply(lambda r: (1 + r).prod()) - 1),
                         "funding (%)": 100 * grouped["funding"].sum(),
                         "basis (%)": 100 * grouped["basis"].sum(),
                         "time invested (%)": 100 * grouped["exposure"].apply(lambda e: (e > 0).mean())})
