# Is the realised correlation of a basket of two or three stocks well approximated by a constant ?
# (Models/Correlation/constant_correlation.ipynb)
# For each basket and each horizon H (1 month to 5 years), the realised correlation of every pair over windows of H
# trading days is compared with two constants :
# - in-sample   : the full-sample correlation (the best a constant can do, with hindsight)
# - out-of-sample : the correlation of all the returns known before the window starts (expanding window), i.e. the
#                 constant a pricer could have used
# The RMSE between the two is not enough by itself : over H days the realised correlation of a pair with a truly
# constant correlation is still noisy, all the more so as returns have fat tails and volatility clustering. The
# statistics are therefore compared with their distribution under H0 "constant conditional correlation", simulated by
# a filtered bootstrap of a CCC-GARCH(1,1) model (GARCH per stock, standardised residual vectors resampled i.i.d. and
# rescaled by the observed GARCH volatilities, so that the dependence between the stocks does not change over time
# while the marginals keep their fat tails and their actual volatility history). Tests :
# - Monte Carlo test on the RMSE (in- and out-of-sample) : p = P_H0(RMSE >= observed RMSE)
# - Fisher z homogeneity test on non-overlapping windows, asymptotic chi2 p-value (i.i.d. normal returns) and Monte
#   Carlo p-value
# - multivariate Engle-Sheppard (2001) test of constant conditional correlation on the GARCH residuals
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

HORIZONS = {"1m": 21, "2m": 42, "3m": 63, "6m": 126, "1y": 252, "2y": 504, "3y": 756, "5y": 1260}

BASKETS = {
    # same sector
    "LVMH / Hermès": ["LVMH", "HERMES"],
    "LVMH / Kering": ["LVMH", "KERING"],
    "LVMH / Hermès / Kering": ["LVMH", "HERMES", "KERING"],
    "BNP / SocGen": ["BNP", "SOCGEN"],
    "BNP / SocGen / Crédit Agricole": ["BNP", "SOCGEN", "CREDITAGRICOLE"],
    "Airbus / Safran": ["AIRBUS", "SAFRAN"],
    "Airbus / Safran / Thales": ["AIRBUS", "SAFRAN", "THALES"],
    "L'Oréal / Danone": ["LOREAL", "DANONE"],
    "L'Oréal / Pernod": ["LOREAL", "PERNOD"],
    "Vinci / Bouygues / Saint-Gobain": ["VINCI", "BOUYGUES", "SAINTGOBAIN"],
    "Capgemini / Dassault Systèmes": ["CAPGEMINI", "DASSAULTSYS"],
    "Schneider / Legrand": ["SCHNEIDER", "LEGRAND"],
    "Total / Air Liquide": ["TOTAL", "AIRLIQUIDE"],
    # different sectors
    "LVMH / Total": ["LVMH", "TOTAL"],
    "Sanofi / BNP": ["SANOFI", "BNP"],
    "L'Oréal / Airbus": ["LOREAL", "AIRBUS"],
    "Orange / Sanofi": ["ORANGE", "SANOFI"],
    "Total / Sanofi / L'Oréal": ["TOTAL", "SANOFI", "LOREAL"],
    "BNP / Airbus / LVMH": ["BNP", "AIRBUS", "LVMH"],
    "Schneider / Air Liquide / Capgemini": ["SCHNEIDER", "AIRLIQUIDE", "CAPGEMINI"],
}


def basket_returns(px, basket):
    """ Daily log returns of the basket on the days where all its stocks are quoted. """
    return np.log(px[basket].dropna()).diff().dropna()


def pair_labels(columns):
    i, j = np.triu_indices(len(columns), 1)
    return [f"{columns[a]}/{columns[b]}" for a, b in zip(i, j)]


# ----------------------------------------------------------------------------------------------------------------------
# Realised correlations (vectorised over simulations)
# ----------------------------------------------------------------------------------------------------------------------

def _cumsum0(x):
    """ Cumulative sum on the time axis (-2) with a leading zero row : sum over [a, b) = c[b] - c[a]. """
    c = np.cumsum(x, axis=-2)
    return np.concatenate([np.zeros(c.shape[:-2] + (1,) + c.shape[-1:]), c], axis=-2)


class _Sums:
    """ Cumulative first and second moments of returns X (..., T, N), to get the correlation of each pair on any
    segment of days in O(1). """

    def __init__(self, X):
        N = X.shape[-1]
        self.i, self.j = np.triu_indices(N, 1)
        self.c1, self.c2 = _cumsum0(X), _cumsum0(X ** 2)
        self.c12 = _cumsum0(X[..., self.i] * X[..., self.j])

    def corr(self, a, b):
        """ Correlation of each pair over the days [a_k, b_k), shape (..., K, P). """
        a, b = np.asarray(a), np.asarray(b)
        n = (b - a)[:, None]
        m = (self.c1[..., b, :] - self.c1[..., a, :]) / n
        v = (self.c2[..., b, :] - self.c2[..., a, :]) / n - m ** 2
        c = (self.c12[..., b, :] - self.c12[..., a, :]) / n - m[..., self.i] * m[..., self.j]
        return c / np.sqrt(v[..., self.i] * v[..., self.j])


def constancy_statistics(X, horizons, step=21, min_history=504):
    """ Statistics of constancy of the correlation for returns X (..., T, N), per horizon H and per pair :
    - rmse_is  : RMSE between the correlation realised over [s, s + H) (s every `step` days) and the full-sample one
    - rmse_oos : same against the correlation over [0, s) (windows starting after `min_history` days)
    - Q        : Fisher z homogeneity statistic on the floor(T / H) non-overlapping windows ending at the last day,
                 sum_k (H - 3) (z_k - zbar)^2, chi2(K - 1) for i.i.d. normal returns with constant correlation
    Returns {H: {"rmse_is", "rmse_oos", "Q": arrays (..., P), "K": number of non-overlapping windows,
    "n_windows", "n_windows_oos"}}. """
    T = X.shape[-2]
    s = _Sums(X)
    rho_full = s.corr([0], [T])                                           # (..., 1, P)
    out = {}
    for H in horizons:
        start = np.arange(0, T - H + 1, step)
        rw = s.corr(start, start + H)
        rmse_is = np.sqrt(np.mean((rw - rho_full) ** 2, axis=-2))
        oos = start >= min_history
        expanding = s.corr(np.zeros(oos.sum(), dtype=int), start[oos])
        rmse_oos = np.sqrt(np.mean((rw[..., oos, :] - expanding) ** 2, axis=-2))
        K = T // H
        a = T - H * np.arange(K, 0, -1)
        z = np.arctanh(s.corr(a, a + H))
        Q = (H - 3) * np.sum((z - z.mean(axis=-2, keepdims=True)) ** 2, axis=-2)
        out[H] = {"rmse_is": rmse_is, "rmse_oos": rmse_oos, "Q": Q, "K": K, "n_windows": len(start),
                  "n_windows_oos": int(oos.sum())}
    return out


def realised_correlation(r, H):
    """ Realised correlation of each pair over the H days ending at t (pandas, for the plots). """
    s = _Sums(r.values)
    end = np.arange(H, len(r) + 1)
    return pd.DataFrame(s.corr(end - H, end), index=r.index[end - 1], columns=pair_labels(list(r.columns)))


# ----------------------------------------------------------------------------------------------------------------------
# H0 : constant conditional correlation (CCC-GARCH filtered bootstrap)
# ----------------------------------------------------------------------------------------------------------------------

def fit_garch(r):
    """ GARCH(1,1) with constant mean and Student-t innovations per stock, on returns in % : parameters, standardised
    residuals and conditional volatilities. """
    resid, vol, rows = {}, {}, {}
    for c in r:
        fit = arch_model(100 * r[c], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        p = fit.params
        rows[c] = {"mu": p["mu"], "omega": p["omega"], "alpha": p["alpha[1]"], "beta": p["beta[1]"], "nu": p["nu"]}
        resid[c], vol[c] = fit.std_resid, fit.conditional_volatility
    return pd.DataFrame(rows).T, pd.DataFrame(resid), pd.DataFrame(vol)


def simulate_h0(params, resid, vol, n_sims, rng):
    """ n_sims paths of returns (in %) under H0 : r*_t = mu + sigma_t z*_t, with sigma_t the observed GARCH
    volatility of each stock and the vectors z*_t drawn i.i.d. among the observed standardised residual vectors.
    The dependence between the stocks is the same every day, while the marginals keep their fat tails and their
    actual volatility history (the test is conditional on it). Simulating the volatility with the GARCH recursion
    instead would feed the extreme residuals back into the variance and produce explosive paths (daily returns of
    50 % and more) that make the H0 distribution far too wide. Shape (n_sims, T, N). """
    z = resid.values[rng.integers(0, len(resid), (n_sims, len(resid)))]
    return params["mu"].values + vol.values * z


def engle_sheppard_test(e, lags=(1, 5, 10)):
    """ Multivariate Engle-Sheppard (2001) test of constant conditional correlation : whiten the standardised
    residuals with the constant correlation, stack the off-diagonal cross products Y_t,ij = u_t,i u_t,j and regress
    them on a constant and s of their own lags (same coefficients for all the pairs). Under H0,
    delta' X'X delta / sigma^2 ~ chi2(s + 1). """
    e = np.asarray(e)
    L = np.linalg.cholesky(np.corrcoef(e.T))
    u = e @ np.linalg.inv(L).T
    i, j = np.triu_indices(e.shape[1], 1)
    Y = u[:, i] * u[:, j]                                                 # (n, P)
    n, rows = len(Y), []
    for s in lags:
        X = np.concatenate([np.column_stack([np.ones(n - s)] + [Y[s - k - 1:n - k - 1, p] for k in range(s)])
                            for p in range(Y.shape[1])])
        y = np.concatenate([Y[s:, p] for p in range(Y.shape[1])])
        delta = np.linalg.lstsq(X, y, rcond=None)[0]
        stat = delta @ X.T @ X @ delta / (y - X @ delta).var()
        rows.append({"lags": s, "statistic": stat, "df": s + 1, "p-value": stats.chi2.sf(stat, s + 1)})
    return pd.DataFrame(rows).set_index("lags")


# ----------------------------------------------------------------------------------------------------------------------
# One basket
# ----------------------------------------------------------------------------------------------------------------------

def _aggregate(stat, name):
    """ Basket-level statistic from the per-pair ones : RMSE pooled over the pairs, Q summed. """
    return np.sum(stat, axis=-1) if name == "Q" else np.sqrt(np.mean(stat ** 2, axis=-1))


def analyse_basket(r, horizons=HORIZONS, n_sims=1000, step=21, min_history=504, seed=0, chunk=100):
    """ Observed statistics, their distribution under H0 (n_sims CCC-GARCH bootstrap paths of the same length) and
    the tests for one basket. Returns a dict with "pairs" (one row per pair and horizon), "basket" (one row per
    horizon), the GARCH parameters and the Engle-Sheppard test. """
    Hs = list(horizons.values())
    labels = pair_labels(list(r.columns))
    obs = constancy_statistics(r.values, Hs, step, min_history)
    params, resid, vol = fit_garch(r)
    rng = np.random.default_rng(seed)
    sims = {H: {k: [] for k in ("rmse_is", "rmse_oos", "Q")} for H in Hs}
    for k in range(0, n_sims, chunk):
        X = simulate_h0(params, resid, vol, min(chunk, n_sims - k), rng)
        for H, st in constancy_statistics(X, Hs, step, min_history).items():
            for name in sims[H]:
                sims[H][name].append(st[name])
    sims = {H: {name: np.concatenate(v) for name, v in d.items()} for H, d in sims.items()}

    pair_rows, basket_rows = [], []
    for label, H in horizons.items():
        o, s, K = obs[H], sims[H], obs[H]["K"]
        for p, pair in enumerate(labels):
            row = {"horizon": label, "H": H, "pair": pair, "K": K}
            for name in ("rmse_is", "rmse_oos", "Q"):
                row[name] = o[name][p]
                row[f"{name} H0 median"] = np.median(s[name][:, p])
                row[f"{name} H0 95%"] = np.quantile(s[name][:, p], 0.95)
                row[f"{name} p MC"] = (1 + np.sum(s[name][:, p] >= o[name][p])) / (len(s[name]) + 1)
            row["Q p asymptotic"] = stats.chi2.sf(o["Q"][p], K - 1)
            row["Q asymptotic size under H0"] = np.mean(stats.chi2.sf(s["Q"][:, p], K - 1) < 0.05)
            pair_rows.append(row)
        row = {"horizon": label, "H": H, "K": K, "n_windows": o["n_windows"], "n_windows_oos": o["n_windows_oos"]}
        for name in ("rmse_is", "rmse_oos", "Q"):
            ob, sb = _aggregate(o[name], name), _aggregate(s[name], name)
            row[name] = ob
            row[f"{name} H0 median"] = np.median(sb)
            row[f"{name} H0 95%"] = np.quantile(sb, 0.95)
            row[f"{name} p MC"] = (1 + np.sum(sb >= ob)) / (len(sb) + 1)
            if name != "Q":                                               # part of the RMSE not explained by noise
                row[f"{name} excess"] = np.sqrt(max(ob ** 2 - np.mean(sb ** 2), 0.0))
        basket_rows.append(row)
    return {"pairs": pd.DataFrame(pair_rows), "basket": pd.DataFrame(basket_rows), "garch": params,
            "engle_sheppard": engle_sheppard_test(resid.values), "correlation": r.corr(),
            "start": r.index[0], "end": r.index[-1], "n_days": len(r)}


def analyse_all(px, baskets=BASKETS, horizons=HORIZONS, n_sims=1000, verbose=True, **kw):
    out = {}
    for k, (name, basket) in enumerate(baskets.items()):
        out[name] = analyse_basket(basket_returns(px, basket), horizons, n_sims, seed=k, **kw)
        if verbose:
            print(f"{k + 1:2d}/{len(baskets)} {name}")
    return out


def stack(results, part):
    """ Concatenate the "pairs" or "basket" tables of all the baskets. """
    return pd.concat({name: res[part] for name, res in results.items()}, names=["basket"]).reset_index(level=0)


def benjamini_hochberg(p):
    """ Benjamini-Hochberg adjusted p-values (false discovery rate). """
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out
