# Is the realised correlation of a basket of two or three stocks well approximated by an EWMA correlation ?
# (Models/Correlation/ewma_correlation.ipynb ; same baskets, horizons and windows as constant_correlation.py)
# Model : EWMA correlation of the GARCH(1,1)-t standardised residuals,
#     Q_t = (1 - lambda) e_{t-1} e_{t-1}' + lambda Q_{t-1},   R_t = corr(Q_t),
# whose forecast of the correlation over the window [t, t + H) is R_t (flat : the EWMA is a martingale). lambda is
# estimated by maximum likelihood (correlation part of the Gaussian likelihood) on a grid :
# - in-sample     : lambda fitted on the whole history
# - out-of-sample : lambda re-fitted at every window start on the days before it
# Unlike the constant correlation, the EWMA cannot serve as the null model of a Monte Carlo test : as a data-generating
# process its correlation is a bounded martingale that is absorbed at -1 or +1 (ewma_dynamics_test). It is therefore
# evaluated as a forecast of the correlation realised over [t, t + H) (windows every 21 days after 2 years) :
# - RMSE, in- and out-of-sample lambda, against the same constant correlations as constant_correlation.py
# - noise floor : RMSE of the true correlation itself under H0 "constant conditional correlation" (filtered bootstrap
#   of constant_correlation.py), i.e. the sampling noise of the realised correlation that no forecast can remove ;
#   excess RMSE = sqrt(RMSE^2 - floor^2) is the part of the error due to the correlation moving and not forecast
# - Diebold-Mariano test of equal squared error, EWMA against the expanding-window constant (Newey-West variance)
# - Mincer-Zarnowitz regression realised = a + b forecast, Wald test of (a, b) = (0, 1) with Newey-West variance
# - RMSE as a function of lambda for each horizon (with hindsight) : the lambda the horizon calls for
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

import constant_correlation as cc

HORIZONS, BASKETS = cc.HORIZONS, cc.BASKETS
GRID = 1 - np.geomspace(0.15, 0.0005, 40)                                 # lambda from 0.85 to 0.9995
BURN_IN = 63


# ----------------------------------------------------------------------------------------------------------------------
# EWMA correlation and its likelihood (vectorised over simulations and over the grid of lambda)
# ----------------------------------------------------------------------------------------------------------------------

def _entries(N):
    """ Indices of the Q entries kept : the N variances then the P covariances (i < j). """
    i, j = np.triu_indices(N, 1)
    return np.r_[np.arange(N), i], np.r_[np.arange(N), j], i, j


def ewma_correlation_path(E, lam, burn_in=BURN_IN):
    """ Off-diagonal EWMA correlations rho_t (..., T, P) of residuals E (..., T, N) : Q_0 = second moment of the first
    `burn_in` days, Q_t uses the residuals up to t - 1. """
    N = E.shape[-1]
    a, b, i, j = _entries(N)
    prods = E[..., a] * E[..., b]
    u = np.empty_like(prods)
    u[..., 0, :] = prods[..., :burn_in, :].mean(axis=-2)
    u[..., 1:, :] = (1 - lam) * prods[..., :-1, :]
    Q = lfilter([1.0], [1.0, -lam], u, axis=-2)
    return Q[..., N:] / np.sqrt(Q[..., i] * Q[..., j])


def correlation_matrix(rho, N):
    """ Off-diagonal correlations (..., P) -> matrices (..., N, N). """
    i, j = np.triu_indices(N, 1)
    R = np.zeros(rho.shape[:-1] + (N, N)) + np.eye(N)
    R[..., i, j] = rho
    R[..., j, i] = rho
    return R


def daily_correlation_loglik(E, rho):
    """ Correlation part of the Gaussian log-likelihood of each day, -(log det R_t + e_t' R_t^-1 e_t - e_t' e_t) / 2,
    closed form for N = 2 and 3. """
    N = E.shape[-1]
    if N == 2:
        a, x, y = rho[..., 0], E[..., 0], E[..., 1]
        det = 1 - a ** 2
        q = (x * x + y * y - 2 * a * x * y) / det
    elif N == 3:
        a, b, c = rho[..., 0], rho[..., 1], rho[..., 2]                   # pairs (0,1), (0,2), (1,2)
        x, y, z = E[..., 0], E[..., 1], E[..., 2]
        det = 1 - a ** 2 - b ** 2 - c ** 2 + 2 * a * b * c
        q = ((1 - c ** 2) * x * x + (1 - b ** 2) * y * y + (1 - a ** 2) * z * z + 2 * (b * c - a) * x * y
             + 2 * (a * c - b) * x * z + 2 * (a * b - c) * y * z) / det
    else:
        R = correlation_matrix(rho, N)
        det = np.linalg.det(R)
        q = np.einsum("...i,...i->...", E, np.linalg.solve(R, E[..., None])[..., 0])
    return -0.5 * (np.log(det) + q - (E ** 2).sum(axis=-1))


# ----------------------------------------------------------------------------------------------------------------------
# Forecast errors
# ----------------------------------------------------------------------------------------------------------------------

def ewma_forecasts(X, E, horizons, step=21, min_history=504, grid=GRID, burn_in=BURN_IN):
    """ EWMA forecasts and realised correlations of one history, returns X and residuals E (T, N). Per horizon H, with
    s = min_history, + step, ... (windows [s, s + H) inside the sample) :
    "realised" (K, P), "ewma_is", "ewma_oos" (lambda fitted on everything / on [0, s)), "const_is" (full-sample
    correlation), "const_oos" (correlation over [0, s)), "rmse_grid" (G, P) the RMSE for each lambda of the grid ;
    plus "lambda_is" and the out-of-sample "lambda_oos" (one per window start). """
    T, N = X.shape
    sums = cc._Sums(X)
    starts = np.arange(min_history, T, step)
    rho_at, ll_at, ll_T = [], [], []
    for lam in grid:
        rho = ewma_correlation_path(E, lam, burn_in)
        ll = daily_correlation_loglik(E, rho)
        ll[:burn_in] = 0
        c = np.r_[0.0, np.cumsum(ll)]
        rho_at.append(rho[starts]); ll_at.append(c[starts]); ll_T.append(c[T])
    rho_at, ll_at = np.array(rho_at), np.array(ll_at)                      # (G, M, P), (G, M)
    g_is, g_oos = int(np.argmax(ll_T)), ll_at.argmax(axis=0)
    fc_oos = rho_at[g_oos, np.arange(len(starts))]
    out = {"lambda_is": grid[g_is], "lambda_oos": grid[g_oos], "starts": starts}
    for H in horizons:
        k = starts + H <= T
        s = starts[k]
        real = sums.corr(s, s + H)
        out[H] = {"starts": s, "realised": real, "ewma_is": rho_at[g_is][k], "ewma_oos": fc_oos[k],
                  "const_is": np.broadcast_to(sums.corr([0], [T]), real.shape),
                  "const_oos": sums.corr(np.zeros(len(s), dtype=int), s),
                  "rmse_grid": np.sqrt(np.mean((rho_at[:, k] - real) ** 2, axis=1))}
    return out


def noise_floor(r, horizons, n_sims=300, step=21, min_history=504, seed=0):
    """ Per horizon, simulated distribution (n_sims,) of the RMSE, pooled over the pairs, between the correlation
    realised over the same windows as ewma_forecasts and the full-sample correlation of the path, under H0
    "constant conditional correlation" (cc.simulate_h0) : the sampling noise of the realised correlation. """
    params, resid, vol = cc.fit_garch(r)
    X = cc.simulate_h0(params, resid, vol, n_sims, np.random.default_rng(seed))
    T = X.shape[1]
    sums = cc._Sums(X)
    full = sums.corr([0], [T])
    out = {}
    for H in horizons:
        s = np.arange(min_history, T - H + 1, step)
        out[H] = np.sqrt(np.mean((sums.corr(s, s + H) - full) ** 2, axis=(1, 2)))
    return out


def _newey_west(g, lags):
    """ Bartlett long-run covariance of the rows of g (n, k), autocovariances divided by n. """
    g = g - g.mean(axis=0)
    n = len(g)
    S = g.T @ g / n
    for l in range(1, lags + 1):
        G = g[l:].T @ g[:-l] / n
        S += (1 - l / (lags + 1)) * (G + G.T)
    return S


def diebold_mariano(d, lags):
    """ DM statistic and p-value of a loss differential series (negative : the first forecast is better). """
    stat = d.mean() / np.sqrt(_newey_west(d[:, None], lags)[0, 0] / len(d))
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def mincer_zarnowitz(realised, forecast, lags):
    """ realised_{s,p} = a + b forecast_{s,p} + u, pooled over the pairs p ; Newey-West variance on the scores summed
    over the pairs of each date (dependence across pairs and overlapping windows). Returns a, b, their standard
    errors and the Wald test of (a, b) = (0, 1), chi2(2). """
    K, P = realised.shape
    x, y = forecast.reshape(-1), realised.reshape(-1)
    Z = np.column_stack([np.ones_like(x), x])
    ZZ_inv = np.linalg.inv(Z.T @ Z)
    beta = ZZ_inv @ Z.T @ y
    u = y - Z @ beta
    scores = (Z * u[:, None]).reshape(K, P, 2).sum(axis=1)                 # (K, 2)
    V = ZZ_inv @ (K * _newey_west(scores, lags)) @ ZZ_inv
    diff = beta - np.array([0.0, 1.0])
    W = float(diff @ np.linalg.solve(V, diff))
    se = np.sqrt(np.diag(V))
    return {"a": beta[0], "b": beta[1], "se a": se[0], "se b": se[1], "Wald": W, "p-value": stats.chi2.sf(W, 2)}


# ----------------------------------------------------------------------------------------------------------------------
# EWMA as a data-generating process
# ----------------------------------------------------------------------------------------------------------------------

def whitened_innovations(E, lam, burn_in=BURN_IN):
    """ u_t = L_t^-1 e_t with L_t the Cholesky factor of the fitted EWMA correlation (t >= burn_in), re-whitened by
    their own covariance so that E[u u'] = I exactly. """
    N = E.shape[-1]
    R = correlation_matrix(ewma_correlation_path(E, lam, burn_in), N)[burn_in:]
    u = np.linalg.solve(np.linalg.cholesky(R), E[burn_in:, :, None])[..., 0]
    return u @ np.linalg.inv(np.linalg.cholesky(np.cov(u.T, bias=True))).T


def simulate_ewma(lam, Q0, u, n_sims, T, rng):
    """ Residuals generated by the EWMA model itself : e_t = L_t u*_t with L_t the Cholesky factor of
    R_t = corr(Q_t), Q_t = (1 - lambda) e_{t-1} e_{t-1}' + lambda Q_{t-1}, u*_t drawn i.i.d. among the whitened
    innovations. Returns the residuals (n_sims, T, N) and the conditional correlations (n_sims, T, P). """
    N = len(Q0)
    i, j = np.triu_indices(N, 1)
    z = u[rng.integers(0, len(u), (n_sims, T))]
    Q = np.broadcast_to(Q0, (n_sims, N, N)).copy()
    e, rho = np.empty((n_sims, T, N)), np.empty((n_sims, T, len(i)))
    for t in range(T):
        d = np.sqrt(np.einsum("bii->bi", Q))
        R = Q / d[:, :, None] / d[:, None, :]
        rho[:, t] = R[:, i, j]
        e[:, t] = np.einsum("bij,bj->bi", np.linalg.cholesky(R), z[:, t])
        Q = (1 - lam) * e[:, t, :, None] * e[:, t, None, :] + lam * Q
    return e, rho


def ewma_dynamics_test(E, lam, n_sims=200, threshold=0.95, burn_in=BURN_IN, seed=0):
    """ H0 : the residuals are generated by the EWMA recursion with the fitted lambda. Statistic : fraction of the days
    where a pairwise conditional correlation exceeds `threshold` in absolute value (the EWMA correlation drifts to
    -1 or +1), observed (on the fitted EWMA path) against simulated ; one-sided p-value P_H0(fraction <= observed).
    Also returns the quantiles of the simulated correlation at the end of the sample. """
    Q0 = E[:burn_in].T @ E[:burn_in] / burn_in
    _, rho = simulate_ewma(lam, Q0, whitened_innovations(E, lam, burn_in), n_sims, len(E), np.random.default_rng(seed))
    obs_rho = ewma_correlation_path(E, lam, burn_in)
    frac_sim = np.mean(np.abs(rho) > threshold, axis=(1, 2))
    frac_obs = float(np.mean(np.abs(obs_rho) > threshold))
    return {"lambda": lam, f"days |rho| > {threshold} observed": frac_obs,
            f"days |rho| > {threshold} H0 median": float(np.median(frac_sim)),
            "p-value": (1 + np.sum(frac_sim <= frac_obs)) / (n_sims + 1),
            "observed rho min": float(obs_rho.min()), "observed rho max": float(obs_rho.max()),
            "H0 final rho 5%": float(np.quantile(rho[:, -1], 0.05)), "H0 final rho 95%": float(np.quantile(rho[:, -1], 0.95)),
            "H0 final rho |.| > 0.9": float(np.mean(np.abs(rho[:, -1]) > 0.9))}


# ----------------------------------------------------------------------------------------------------------------------
# One basket
# ----------------------------------------------------------------------------------------------------------------------

FORECASTS = {"ewma_is": "EWMA, in-sample lambda", "ewma_oos": "EWMA, out-of-sample lambda",
             "const_is": "constant, in-sample", "const_oos": "constant, out-of-sample"}


def analyse_basket(r, horizons=HORIZONS, n_floor=300, n_dynamics=200, step=21, min_history=504, seed=0):
    """ Errors of the EWMA and constant forecasts, noise floor, Diebold-Mariano and Mincer-Zarnowitz tests per horizon
    ("basket" : one row per horizon), RMSE by lambda ("rmse_lambda" : horizons x grid), the out-of-sample lambda path
    and the test of the EWMA dynamics. """
    Hs = list(horizons.values())
    params, resid, vol = cc.fit_garch(r)
    X, E = 100 * r.values, resid.values
    fc = ewma_forecasts(X, E, Hs, step, min_history)
    floor = noise_floor(r, Hs, n_floor, step, min_history, seed)
    rows, rmse_lambda = [], {}
    for label, H in horizons.items():
        f = fc[H]
        lags = int(np.ceil(H / step)) - 1
        row = {"horizon": label, "H": H, "n_windows": len(f["starts"])}
        err = {k: f[k] - f["realised"] for k in FORECASTS}
        for k in FORECASTS:
            row[f"rmse {k}"] = np.sqrt(np.mean(err[k] ** 2))
        fl2 = np.mean(floor[H] ** 2)
        row["noise floor"] = np.sqrt(fl2)
        for k in FORECASTS:
            row[f"excess {k}"] = np.sqrt(max(row[f"rmse {k}"] ** 2 - fl2, 0.0))
        row["ratio EWMA / constant (oos)"] = row["rmse ewma_oos"] / row["rmse const_oos"]
        d = np.mean(err["ewma_oos"] ** 2 - err["const_oos"] ** 2, axis=1)
        row["DM stat"], row["DM p-value"] = diebold_mariano(d, lags)
        mz = mincer_zarnowitz(f["realised"], f["ewma_oos"], lags)
        row.update({f"MZ {k}": v for k, v in mz.items()})
        rows.append(row)
        rmse_lambda[label] = np.sqrt(np.mean(f["rmse_grid"] ** 2, axis=1))
    return {"basket": pd.DataFrame(rows), "lambda": fc["lambda_is"],
            "lambda_oos": pd.Series(fc["lambda_oos"], index=r.index[fc["starts"]]),
            "rmse_lambda": pd.DataFrame(rmse_lambda, index=GRID),
            "dynamics": ewma_dynamics_test(E, fc["lambda_is"], n_dynamics, seed=seed), "garch": params}


def analyse_all(px, baskets=BASKETS, horizons=HORIZONS, verbose=True, **kw):
    out = {}
    for k, (name, basket) in enumerate(baskets.items()):
        out[name] = analyse_basket(cc.basket_returns(px, basket), horizons, seed=k, **kw)
        if verbose:
            print(f"{k + 1:2d}/{len(baskets)} {name}")
    return out


def ewma_forecast(r, lam):
    """ EWMA correlation forecast R_t of each pair (pandas, for the plots) on the GARCH residuals of r. """
    _, resid, _ = cc.fit_garch(r)
    rho = ewma_correlation_path(resid.values, lam)
    return pd.DataFrame(rho, index=r.index, columns=cc.pair_labels(list(r.columns)))
