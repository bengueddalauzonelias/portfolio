# Is the realised correlation of a basket of two or three stocks well approximated by a DCC-type model ?
# (Models/Correlation/dcc_correlation.ipynb ; same baskets, horizons and windows as constant_correlation.py and
# ewma_correlation.py)
# Models, two-step (marginals, then dependence) :
# - DCC-GARCH                 : GARCH(1,1)-t marginals, DCC(1,1) (Engle 2002) on the standardised residuals
# - DCC-GJR                   : GJR-GARCH(1,1,1)-t marginals (leverage effect), DCC(1,1)
# - ADCC-GJR                  : GJR marginals, asymmetric DCC (Cappiello, Engle and Sheppard 2006) : the correlation
#                               also reacts to joint negative shocks
# - ADCC-GJR, Gaussian copula : GJR marginals, empirical distribution of the residuals (probability integral transform),
#                               ADCC on the normal scores, Gaussian copula likelihood
# - ADCC-GJR, t copula        : same with a Student-t copula (nu estimated jointly with the ADCC parameters)
# Scalar (A)DCC : Q_t = (1 - a - b) Qbar - g Nbar + a z_{t-1} z_{t-1}' + g n_{t-1} n_{t-1}' + b Q_{t-1}, R_t = corr(Q_t),
# with n = min(z, 0), Qbar = E[z z'], Nbar = E[n n'] (correlation targeting) ; DCC is g = 0.
# Forecast of the correlation over [t, t + H) (Engle-Sheppard 2001) :
#     E_t[R_{t+h}] = Rbar + phi^(h-1) (R_t - Rbar),  phi = a + b + g / 2,  Rbar = corr(Qbar),
# averaged over h = 1..H : Rbar + w(H) (R_t - Rbar), w(H) = (1 - phi^H) / (H (1 - phi)).
# In-sample : every parameter estimated on the whole history. Out-of-sample : marginals, empirical distributions and
# (A)DCC re-estimated every year on the past only, the filter running on until the next re-estimation.
# Evaluation, as for the EWMA (ewma_correlation.py) : RMSE against the realised correlation, noise floor, excess RMSE,
# Diebold-Mariano and Mincer-Zarnowitz tests ; plus likelihood ratio tests between nested models, and, since the
# (A)DCC is stationary, the Monte Carlo test of the constant study : statistics compared with their distribution when
# the fitted model is the data-generating process.
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import optimize, stats
from scipy.signal import lfilter
from scipy.special import gammaln

import constant_correlation as cc
import ewma_correlation as ew

HORIZONS, BASKETS = cc.HORIZONS, cc.BASKETS
MODELS = {
    "DCC-GARCH": {"vol": "garch", "asym": False, "copula": None},
    "DCC-GJR": {"vol": "gjr", "asym": False, "copula": None},
    "ADCC-GJR": {"vol": "gjr", "asym": True, "copula": None},
    "ADCC-GJR, Gaussian copula": {"vol": "gjr", "asym": True, "copula": "gaussian"},
    "ADCC-GJR, t copula": {"vol": "gjr", "asym": True, "copula": "t"},
}
REFIT = 252                                                                # out-of-sample re-estimation, in days


# ----------------------------------------------------------------------------------------------------------------------
# Marginals
# ----------------------------------------------------------------------------------------------------------------------

def fit_marginals(r, vol="garch", last_obs=None):
    """ GARCH(1,1)-t (vol="garch") or GJR-GARCH(1,1,1)-t (vol="gjr") per stock on returns in %, estimated on the first
    `last_obs` days (all if None) and filtered over the whole sample with these parameters. Returns the parameters,
    the standardised residuals (T, N), the conditional volatilities (T, N) and the log-likelihood per stock. """
    params, resid, sigma, ll = {}, {}, {}, {}
    for c in r:
        am = arch_model(100 * r[c], mean="Constant", vol="GARCH", p=1, o=int(vol == "gjr"), q=1, dist="t")
        res = am.fit(disp="off", last_obs=last_obs)
        fx = am.fix(res.params)
        params[c], ll[c] = res.params, res.loglikelihood
        sigma[c] = fx.conditional_volatility
        resid[c] = fx.resid / fx.conditional_volatility
    return pd.DataFrame(params).T, pd.DataFrame(resid).values, pd.DataFrame(sigma).values, pd.Series(ll)


def pit(E, n_ref):
    """ Probability integral transform with the empirical distribution of the first n_ref residuals of each stock,
    in (0, 1) (Genest-Ghoudi-Rivest scaling n / (n + 1)). """
    U = np.empty_like(E)
    for k in range(E.shape[1]):
        ref = np.sort(E[:n_ref, k])
        U[:, k] = np.searchsorted(ref, E[:, k], side="right") / (n_ref + 1)
    return np.clip(U, 0.5 / (n_ref + 1), 1 - 0.5 / (n_ref + 1))


def copula_shocks(U, copula, nu=None):
    """ Shocks driving the (A)DCC : normal scores, or Student-t quantiles rescaled to unit variance. Also returns the
    unscaled t quantiles needed by the t copula density. """
    if copula == "gaussian":
        Z = stats.norm.ppf(U)
        return Z, Z
    X = stats.t.ppf(U, nu)
    return X * np.sqrt((nu - 2) / nu), X


# ----------------------------------------------------------------------------------------------------------------------
# (A)DCC filter and likelihoods
# ----------------------------------------------------------------------------------------------------------------------

def _det_quad(rho, X):
    """ det R_t and X_t' R_t^-1 X_t from the off-diagonal correlations rho (..., P), closed form for N = 2, 3. """
    N = X.shape[-1]
    if N == 2:
        a, x, y = rho[..., 0], X[..., 0], X[..., 1]
        det = 1 - a ** 2
        return det, (x * x + y * y - 2 * a * x * y) / det
    if N == 3:
        a, b, c = rho[..., 0], rho[..., 1], rho[..., 2]
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        det = 1 - a ** 2 - b ** 2 - c ** 2 + 2 * a * b * c
        q = ((1 - c ** 2) * x * x + (1 - b ** 2) * y * y + (1 - a ** 2) * z * z + 2 * (b * c - a) * x * y
             + 2 * (a * c - b) * x * z + 2 * (a * b - c) * y * z) / det
        return det, q
    R = ew.correlation_matrix(rho, N)
    return np.linalg.det(R), np.einsum("...i,...i->...", X, np.linalg.solve(R, X[..., None])[..., 0])


def targets(Z):
    """ Qbar = E[z z'] and Nbar = E[n n'] as vectors of the N variances then the P covariances. """
    a, b, _, _ = ew._entries(Z.shape[-1])
    Nn = np.minimum(Z, 0)
    return (Z[..., a] * Z[..., b]).mean(axis=-2), (Nn[..., a] * Nn[..., b]).mean(axis=-2)


def adcc_path(Z, a, g, b, Qbar, Nbar):
    """ Off-diagonal correlations rho_t (..., T + 1, P) of the (A)DCC filter, Q_0 = Qbar ; rho_t uses z up to t - 1
    (the last row is the forecast after the last day). Qbar, Nbar : vectors of entries (see targets). """
    N = Z.shape[-1]
    ia, ib, i, j = ew._entries(N)
    Nn = np.minimum(Z, 0)
    shape = Z.shape[:-2] + (Z.shape[-2] + 1, len(ia))
    u = np.empty(shape)
    u[..., 0, :] = Qbar
    u[..., 1:, :] = (1 - a - b) * Qbar - g * Nbar + a * Z[..., ia] * Z[..., ib] + g * Nn[..., ia] * Nn[..., ib]
    Q = lfilter([1.0], [1.0, -b], u, axis=-2)
    return Q[..., N:] / np.sqrt(Q[..., i] * Q[..., j])


def _stationary(a, g, b, Qbar, Nbar, N):
    """ a + b + delta g < 1, delta = largest eigenvalue of Qbar^-1/2 Nbar Qbar^-1/2 (Cappiello et al. 2006). """
    if g == 0:
        return a + b < 1
    Qm, Nm = _matrix(Qbar, N), _matrix(Nbar, N)
    w, v = np.linalg.eigh(Qm)
    S = (v / np.sqrt(w)) @ v.T
    return a + b + g * np.linalg.eigvalsh(S @ Nm @ S).max() < 1


def _matrix(entries, N):
    """ Vector of the N variances then the P covariances -> symmetric N x N matrix. """
    ia, ib, _, _ = ew._entries(N)
    M = np.zeros((N, N))
    M[ia, ib] = entries
    M[ib, ia] = entries
    return M


def t_copula_logdensity(X, rho, nu):
    """ Log-density of the N-dimensional t copula with correlations rho_t at the t quantiles X (unscaled). """
    N = X.shape[-1]
    det, q = _det_quad(rho, X)
    return (gammaln((nu + N) / 2) + (N - 1) * gammaln(nu / 2) - N * gammaln((nu + 1) / 2) - 0.5 * np.log(det)
            - (nu + N) / 2 * np.log1p(q / nu) + (nu + 1) / 2 * np.log1p(X ** 2 / nu).sum(axis=-1))


def gaussian_logdensity(Z, rho):
    """ Correlation part of the Gaussian log-likelihood (= Gaussian copula density at the normal scores). """
    det, q = _det_quad(rho, Z)
    return -0.5 * (np.log(det) + q - (Z ** 2).sum(axis=-1))


def fit_adcc(E, asym=False, copula=None, n_obs=None, x0=None):
    """ (A)DCC on the residuals E[:n_obs] : Gaussian (quasi-)likelihood on the residuals (copula=None) or copula
    likelihood on the empirical PIT (copula="gaussian" or "t", nu estimated jointly). Returns a dict of parameters
    (a, g, b, nu, phi, Qbar, Nbar), the log-likelihood and the number of observations. """
    n_obs = len(E) if n_obs is None else n_obs
    N = E.shape[1]
    U = pit(E, n_obs) if copula else None

    def shocks(nu):
        if copula is None:
            return E[:n_obs], None
        Z, X = copula_shocks(U[:n_obs], copula, nu)
        return Z, X

    def nll(p):
        a, g, b = p[0], (p[1] if asym else 0.0), p[-1 if copula != "t" else -2]
        nu = p[-1] if copula == "t" else None
        if min(a, g, b) < 0 or a + b >= 1 or (nu is not None and not 2.1 < nu < 200):
            return 1e10
        Z, X = shocks(nu)
        Qbar, Nbar = targets(Z)
        if asym and not _stationary(a, g, b, Qbar, Nbar, N):
            return 1e10
        rho = adcc_path(Z, a, g, b, Qbar, Nbar)[:-1]
        ll = t_copula_logdensity(X, rho, nu) if copula == "t" else gaussian_logdensity(Z, rho)
        return -float(np.sum(ll)) if np.all(np.isfinite(ll)) else 1e10

    if x0 is None:
        x0 = [0.02] + ([0.01] if asym else []) + [0.95] + ([8.0] if copula == "t" else [])
    res = optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 3000})
    p = res.x
    a, g, b = p[0], (p[1] if asym else 0.0), p[-1 if copula != "t" else -2]
    nu = float(p[-1]) if copula == "t" else None
    Z, _ = shocks(nu)
    Qbar, Nbar = targets(Z)
    return {"a": a, "g": g, "b": b, "nu": nu, "phi": a + b + g / 2, "Qbar": Qbar, "Nbar": Nbar, "x": p,
            "loglik": -res.fun, "n_obs": n_obs}


def model_shocks(E, fit, copula, n_ref):
    """ Shocks of the whole sample with the PIT of the estimation sample (copula models) or the residuals. """
    if copula is None:
        return E
    return copula_shocks(pit(E, n_ref), copula, fit["nu"])[0]


def forecast_correlation(rho_s, fit, H):
    """ Average forecast correlation over [s, s + H) given R_s (rows of rho_s) : Rbar + w(H) (R_s - Rbar). """
    N = int(round((1 + np.sqrt(1 + 8 * rho_s.shape[-1])) / 2))
    _, _, i, j = ew._entries(N)
    Qbar = fit["Qbar"]
    rbar = Qbar[N:] / np.sqrt(Qbar[i] * Qbar[j])
    phi = min(fit["phi"], 0.999999)
    w = (1 - phi ** H) / (H * (1 - phi))
    return rbar + w * (rho_s - rbar)


# ----------------------------------------------------------------------------------------------------------------------
# Forecasts of one model on one basket
# ----------------------------------------------------------------------------------------------------------------------

def model_forecasts(r, spec, horizons, step=21, min_history=504, refit=REFIT):
    """ In- and out-of-sample forecasts of one model at the window starts s = min_history, + step, ... :
    {"is": {H: (K, P)}, "oos": {H: (K, P)}, "fit": in-sample fit, "marginals": in-sample marginal parameters and
    log-likelihoods, "oos_params": parameters at each re-estimation date}. """
    T = len(r)
    starts = np.arange(min_history, T, step)
    copula = spec["copula"]
    # in-sample
    mp, E, _, mll = fit_marginals(r, spec["vol"])
    fit = fit_adcc(E, spec["asym"], copula)
    rho = adcc_path(model_shocks(E, fit, copula, T), fit["a"], fit["g"], fit["b"], fit["Qbar"], fit["Nbar"])
    out = {"is": {}, "oos": {}, "fit": fit, "marginals": mp, "marginal_loglik": mll, "residuals": E}
    for H in horizons:
        s = starts[starts + H <= T]
        out["is"][H] = forecast_correlation(rho[s], fit, H)
    # out-of-sample : re-estimation every `refit` days, forecasts at the starts until the next one
    rho_oos, fits, x0 = {}, [], None
    for tau in range(min_history, T, refit):
        _, E_t, _, _ = fit_marginals(r, spec["vol"], last_obs=tau)
        f = fit_adcc(E_t, spec["asym"], copula, n_obs=tau, x0=x0)
        x0 = f["x"]
        rho_t = adcc_path(model_shocks(E_t, f, copula, tau), f["a"], f["g"], f["b"], f["Qbar"], f["Nbar"])
        block = starts[(starts >= tau) & (starts < tau + refit)]
        for s in block:
            rho_oos[s] = (rho_t[s], f)
        fits.append({"date": r.index[tau], "a": f["a"], "g": f["g"], "b": f["b"], "nu": f["nu"], "phi": f["phi"]})
    for H in horizons:
        s = starts[starts + H <= T]
        out["oos"][H] = np.array([forecast_correlation(rho_oos[k][0], rho_oos[k][1], H) for k in s])
    out["oos_params"] = pd.DataFrame(fits).set_index("date")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Monte Carlo test : the fitted model as data-generating process
# ----------------------------------------------------------------------------------------------------------------------

def simulate_model(fit, spec, E, sigma, mu, n_sims, rng):
    """ n_sims return paths (in %) generated by the fitted model : shocks z_t = L_t w_t with L_t the Cholesky factor of
    R_t from the (A)DCC recursion of the simulated shocks ; w_t = whitened observed innovations resampled i.i.d.
    (copula=None), N(0, I) (Gaussian copula) or multivariate t (t copula), the copula shocks being mapped to the
    empirical distribution of the residuals. Returns are mu + sigma_t e_t with the observed conditional volatilities.
    Returns (returns, rho) with rho (n_sims, T, P) the true conditional correlations. """
    T, N = E.shape
    ia, ib, i, j = ew._entries(N)
    copula, a, g, b = spec["copula"], fit["a"], fit["g"], fit["b"]
    Qbar, Nbar = fit["Qbar"], fit["Nbar"]
    if copula is None:                                                     # whitened innovations
        rho_obs = adcc_path(E, a, g, b, Qbar, Nbar)[:-1]
        L = np.linalg.cholesky(ew.correlation_matrix(rho_obs, N))
        u = np.linalg.solve(L, E[:, :, None])[..., 0]
        u = u @ np.linalg.inv(np.linalg.cholesky(np.cov(u.T, bias=True))).T
        W = u[rng.integers(0, T, (n_sims, T))]
    else:
        W = rng.standard_normal((n_sims, T, N))
        if copula == "t":
            W = W / np.sqrt(rng.chisquare(fit["nu"], (n_sims, T, 1)) / fit["nu"])
    Qm, Nm = _matrix(Qbar, N), _matrix(Nbar, N)
    const = (1 - a - b) * Qm - g * Nm
    Q = np.broadcast_to(Qm, (n_sims, N, N)).copy()
    X, rho = np.empty((n_sims, T, N)), np.empty((n_sims, T, len(i)))
    for t in range(T):
        d = np.sqrt(np.einsum("bii->bi", Q))
        R = Q / d[:, :, None] / d[:, None, :]
        rho[:, t] = R[:, i, j]
        x = np.einsum("bij,bj->bi", np.linalg.cholesky(R), W[:, t])
        X[:, t] = x
        z = x * np.sqrt((fit["nu"] - 2) / fit["nu"]) if copula == "t" else x
        n = np.minimum(z, 0)
        Q = const + a * z[:, :, None] * z[:, None, :] + g * n[:, :, None] * n[:, None, :] + b * Q
    if copula is None:
        e = X
    else:
        U = stats.norm.cdf(X) if copula == "gaussian" else stats.t.cdf(X, fit["nu"])
        e = np.stack([np.quantile(E[:, k], U[..., k]) for k in range(N)], axis=-1)
    return mu + sigma * e, rho


def mc_test(r, spec, fc, horizons, n_sims=200, step=21, min_history=504, seed=0):
    """ Two-sided Monte Carlo p-values of the in-sample RMSE (pooled over the pairs) and of the Fisher statistic
    sum_k (H - 3) (z_realised - z_forecast)^2 on non-overlapping windows, against simulations of the fitted model
    (forecasts from the true conditional correlations and the fitted parameters : estimation error is ignored, a
    slightly anti-conservative approximation). """
    T = len(r)
    fit = fc["fit"]
    mp, E, sigma, _ = fit_marginals(r, spec["vol"])
    X, rho = simulate_model(fit, spec, E, sigma, mp["mu"].values, n_sims, np.random.default_rng(seed))
    sums_o, sums_s = cc._Sums(100 * r.values), cc._Sums(X)
    starts = np.arange(min_history, T, step)
    rho_obs = adcc_path(model_shocks(E, fit, spec["copula"], T), fit["a"], fit["g"], fit["b"], fit["Qbar"], fit["Nbar"])
    rows = []
    for label, H in horizons.items():
        s = starts[starts + H <= T]
        a = T - H * np.arange((T - min_history) // H, 0, -1)
        stat = {}
        for name, sums, rh in [("obs", sums_o, rho_obs), ("sim", sums_s, rho)]:
            fs, fa = forecast_correlation(rh[..., s, :], fit, H), forecast_correlation(rh[..., a, :], fit, H)
            rmse = np.sqrt(np.mean((sums.corr(s, s + H) - fs) ** 2, axis=(-2, -1)))
            zr = np.arctanh(np.clip(sums.corr(a, a + H), -0.999999, 0.999999))
            zf = np.arctanh(np.clip(fa, -0.999999, 0.999999))
            stat[name] = (rmse, (H - 3) * np.sum((zr - zf) ** 2, axis=(-2, -1)))
        rows.append({"horizon": label, "H": H,
                     "rmse": float(stat["obs"][0]), "rmse H0 median": float(np.median(stat["sim"][0])),
                     "rmse p MC": _p_two_sided(stat["sim"][0], stat["obs"][0]),
                     "Q": float(stat["obs"][1]), "Q H0 median": float(np.median(stat["sim"][1])),
                     "Q p MC": _p_two_sided(stat["sim"][1], stat["obs"][1])})
    return pd.DataFrame(rows)


def _p_two_sided(sim, obs):
    hi = (1 + np.sum(sim >= obs)) / (len(sim) + 1)
    lo = (1 + np.sum(sim <= obs)) / (len(sim) + 1)
    return float(min(1.0, 2 * min(hi, lo)))


# ----------------------------------------------------------------------------------------------------------------------
# One basket
# ----------------------------------------------------------------------------------------------------------------------

def analyse_basket(r, models=MODELS, horizons=HORIZONS, step=21, min_history=504, n_sims=200, n_floor=300, seed=0):
    """ For one basket : forecasts of every model, of the EWMA and of the constants on the same windows ; RMSE, noise
    floor, excess RMSE, Diebold-Mariano (against the constant and against the previous model of the list),
    Mincer-Zarnowitz ; likelihood ratio tests between nested models ; Monte Carlo test of each fitted model. """
    Hs = list(horizons.values())
    X = 100 * r.values
    bench = ew.ewma_forecasts(X, cc.fit_garch(r)[1].values, Hs, step, min_history)
    floor = ew.noise_floor(r, Hs, n_floor, step, min_history, seed)
    fcs = {name: model_forecasts(r, spec, Hs, step, min_history) for name, spec in models.items()}
    names = list(models)

    rows = []
    for label, H in horizons.items():
        lags = int(np.ceil(H / step)) - 1
        real = bench[H]["realised"]
        fl2 = np.mean(floor[H] ** 2)
        forecasts = {"constant": bench[H]["const_oos"], "EWMA": bench[H]["ewma_oos"]}
        forecasts.update({n: fcs[n]["oos"][H] for n in names})
        forecasts_is = {n: fcs[n]["is"][H] for n in names}
        for k, (name, f) in enumerate(forecasts.items()):
            err = f - real
            row = {"horizon": label, "H": H, "model": name, "rmse oos": np.sqrt(np.mean(err ** 2)),
                   "noise floor": np.sqrt(fl2)}
            row["excess oos"] = np.sqrt(max(row["rmse oos"] ** 2 - fl2, 0.0))
            if name in forecasts_is:
                row["rmse is"] = np.sqrt(np.mean((forecasts_is[name] - real) ** 2))
            if name != "constant":
                d = np.mean(err ** 2 - (forecasts["constant"] - real) ** 2, axis=1)
                row["DM vs constant"], row["DM vs constant p"] = ew.diebold_mariano(d, lags)
                prev = list(forecasts)[k - 1]
                if prev != "constant":
                    d = np.mean(err ** 2 - (forecasts[prev] - real) ** 2, axis=1)
                    row["DM vs previous"], row["DM vs previous p"] = ew.diebold_mariano(d, lags)
                    row["previous"] = prev
                mz = ew.mincer_zarnowitz(real, f, lags)
                row.update({f"MZ {k2}": v for k2, v in mz.items()})
            rows.append(row)

    # likelihood ratio tests (in-sample)
    lr = {}
    stat = 2 * (fcs["DCC-GJR"]["marginal_loglik"] - fcs["DCC-GARCH"]["marginal_loglik"])
    lr["GJR vs GARCH (marginals, sum)"] = (float(stat.sum()), len(stat), stats.chi2.sf(stat.sum(), len(stat)))
    stat = 2 * (fcs["ADCC-GJR"]["fit"]["loglik"] - fcs["DCC-GJR"]["fit"]["loglik"])
    lr["ADCC vs DCC (g = 0)"] = (stat, 1, 0.5 * stats.chi2.sf(max(stat, 0), 1))
    stat = 2 * (fcs["ADCC-GJR, t copula"]["fit"]["loglik"] - fcs["ADCC-GJR, Gaussian copula"]["fit"]["loglik"])
    lr["t vs Gaussian copula (1/nu = 0)"] = (stat, 1, 0.5 * stats.chi2.sf(max(stat, 0), 1))
    lr = pd.DataFrame(lr, index=["LR", "df", "p-value"]).T

    params = pd.DataFrame({n: {k: fcs[n]["fit"][k] for k in ("a", "g", "b", "nu", "phi", "loglik")} for n in names}).T
    params["half-life (days)"] = np.log(0.5) / np.log(params["phi"].astype(float))
    gjr = fcs["DCC-GJR"]["marginals"]
    mc = pd.concat({n: mc_test(r, models[n], fcs[n], horizons, n_sims, step, min_history, seed) for n in names},
                   names=["model"]).reset_index(level=0)
    return {"table": pd.DataFrame(rows), "lr": lr, "params": params, "gjr": gjr, "mc": mc,
            "oos_params": {n: fcs[n]["oos_params"] for n in names}}


def analyse_named(args):
    """ Wrapper for a process pool : (name, px, kwargs) -> (name, result). """
    name, px, kw = args
    return name, analyse_basket(cc.basket_returns(px, BASKETS[name]), seed=list(BASKETS).index(name), **kw)
