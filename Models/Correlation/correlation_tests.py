# Statistical tests on the dependence between two stocks, run before choosing a correlation model for worst-of /
# best-of pricing (Models/correlation.ipynb) :
# - marginals : fat tails, autocorrelation, volatility clustering                        (Jarque-Bera, Ljung-Box, ARCH-LM)
# - correlation and return horizon                                                       (Epps effect)
# - constancy of the correlation : equality across sub-periods (Fisher z) and of the conditional correlation
#   (Engle-Sheppard 2001), on raw returns and on GARCH-standardised residuals
# - constant (CCC), integrated (EWMA) and mean-reverting (DCC) conditional correlation : likelihoods and LR tests
# - mean reversion of the correlation (ADF, AR(1), DCC half-life)
# - asymmetry and tails : exceedance correlations (Longin-Solnik 2001), tail dependence, volatility regimes,
#   each against the value implied by a Gaussian dependence with the same correlation
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import optimize, stats
from scipy.signal import lfilter
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def log_returns(prices, horizon=1):
    """ Non-overlapping log returns over `horizon` trading days. """
    lp = np.log(prices)
    return lp.iloc[::horizon].diff().dropna()


def fisher_z(rho):
    return np.arctanh(rho)


# ----------------------------------------------------------------------------------------------------------------------
# Marginals
# ----------------------------------------------------------------------------------------------------------------------

def marginal_tests(r, lags=10, arch_lags=5):
    """ Moments and p-values of the normality, no-autocorrelation and no-ARCH tests, one row per asset. """
    rows = {}
    for c in r:
        x = r[c]
        rows[c] = {"vol (ann.)": x.std() * np.sqrt(252), "skew": stats.skew(x), "excess kurt.": stats.kurtosis(x),
                   "Jarque-Bera p": stats.jarque_bera(x).pvalue,
                   f"Ljung-Box r ({lags}) p": acorr_ljungbox(x, [lags]).lb_pvalue.iloc[0],
                   f"Ljung-Box r² ({lags}) p": acorr_ljungbox(x ** 2, [lags]).lb_pvalue.iloc[0],
                   f"ARCH-LM ({arch_lags}) p": het_arch(x, nlags=arch_lags)[1]}
    return pd.DataFrame(rows).T


def garch_standardise(r, dist="t"):
    """ GARCH(1,1) per asset (returns in %), standardised residuals and parameters. """
    eps, params = pd.DataFrame(index=r.index), {}
    for c in r:
        fit = arch_model(100 * r[c], mean="Constant", vol="GARCH", p=1, q=1, dist=dist).fit(disp="off")
        eps[c] = fit.std_resid
        p = fit.params
        params[c] = {"omega": p["omega"], "alpha": p["alpha[1]"], "beta": p["beta[1]"],
                     "alpha+beta": p["alpha[1]"] + p["beta[1]"], "nu": p.get("nu", np.nan)}
    return eps, pd.DataFrame(params).T


# ----------------------------------------------------------------------------------------------------------------------
# Is the correlation constant ?
# ----------------------------------------------------------------------------------------------------------------------

def horizon_correlation(prices, horizons=(1, 2, 5, 10, 20, 60)):
    """ Correlation of non-overlapping h-day log returns, with the Fisher 95% interval. """
    rows = []
    for h in horizons:
        rh = log_returns(prices, h)
        rho, n = rh.corr().iloc[0, 1], len(rh)
        half = 1.96 / np.sqrt(n - 3)
        rows.append({"horizon (days)": h, "n": n, "rho": rho,
                     "95% low": np.tanh(fisher_z(rho) - half), "95% high": np.tanh(fisher_z(rho) + half)})
    return pd.DataFrame(rows).set_index("horizon (days)")


def subperiod_equality_test(x, freq="YE", min_obs=50):
    """ H0 : same correlation on every sub-period. Q = sum (n_k - 3)(z_k - zbar)² ~ chi2(K-1) under iid normality. """
    rows = [(k, len(d), d.corr().iloc[0, 1]) for k, d in x.groupby(pd.Grouper(freq=freq)) if len(d) >= min_obs]
    table = pd.DataFrame(rows, columns=["period", "n", "rho"]).set_index("period")
    z, w = fisher_z(table.rho), table.n - 3
    zbar = (w * z).sum() / w.sum()
    Q = float((w * (z - zbar) ** 2).sum())
    df = len(table) - 1
    return table, {"Q": Q, "df": df, "p-value": stats.chi2.sf(Q, df), "pooled rho": np.tanh(zbar)}


def engle_sheppard_test(eps, lags=(1, 5, 10)):
    """ Engle-Sheppard (2001) : whiten the standardised residuals with the constant correlation, regress the cross
    product on a constant and its own lags. Under constant correlation, delta' X'X delta / s² ~ chi2(s + 1). """
    e = np.asarray(eps)
    L = np.linalg.cholesky(np.corrcoef(e.T))
    u = e @ np.linalg.inv(L).T
    Y, n, rows = u[:, 0] * u[:, 1], len(u), []
    for s in lags:
        X = np.column_stack([np.ones(n - s)] + [Y[s - k - 1:n - k - 1] for k in range(s)])
        y = Y[s:]
        delta = np.linalg.lstsq(X, y, rcond=None)[0]
        stat = delta @ X.T @ X @ delta / (y - X @ delta).var()
        rows.append({"lags": s, "statistic": stat, "df": s + 1, "p-value": stats.chi2.sf(stat, s + 1)})
    return pd.DataFrame(rows).set_index("lags")


# ----------------------------------------------------------------------------------------------------------------------
# Constant, integrated and mean-reverting conditional correlation
# ----------------------------------------------------------------------------------------------------------------------

def q_entries(e, a, b, target, extend=False):
    """ Q_t = target (1 - a - b) + a e_{t-1} e_{t-1}' + b Q_{t-1}, Q_0 = sample covariance, for the 3 entries
    (Q11, Q22, Q12) of the 2x2 matrix (linear filter). target=None gives the integrated recursion (EWMA,
    a = 1 - lambda, b = lambda). Q_t only uses e up to t-1 ; extend=True adds Q_n, the forecast made after the last
    observation. """
    Q0 = np.cov(e.T, bias=True)
    n = len(e) + int(extend)
    entries = []
    for i, j in [(0, 0), (1, 1), (0, 1)]:
        c = 0.0 if target is None else (1 - a - b) * target[i, j]
        u = np.empty(n)
        u[0] = Q0[i, j]
        u[1:] = c + a * e[:n - 1, i] * e[:n - 1, j]
        entries.append(lfilter([1.0], [1.0, -b], u))
    return entries


def _q_path(e, a, b, target, extend=False):
    q11, q22, q12 = q_entries(e, a, b, target, extend)
    return q12 / np.sqrt(q11 * q22)


def correlation_loglik(e, rho):
    """ Correlation part of the Gaussian log-likelihood of standardised residuals (Engle 2002, eq. 26). """
    x, y = e[:, 0], e[:, 1]
    one = 1 - rho ** 2
    return -0.5 * np.sum(np.log(one) + (x * x + y * y - 2 * rho * x * y) / one - (x * x + y * y))


def ccc_fit(e):
    e = np.asarray(e)
    rho = np.full(len(e), np.corrcoef(e.T)[0, 1])
    return {"model": "CCC", "params": {"rho": rho[0]}, "loglik": correlation_loglik(e, rho), "k": 1, "rho": rho}


def ewma_fit(e, bounds=(0.80, 0.9995)):
    e = np.asarray(e)
    f = lambda lam: -correlation_loglik(e, _q_path(e, 1 - lam, lam, None))
    lam = optimize.minimize_scalar(f, bounds=bounds, method="bounded").x
    return {"model": "EWMA", "params": {"lambda": lam}, "loglik": -f(lam), "k": 1, "rho": _q_path(e, 1 - lam, lam, None)}


def dcc_fit(e, x0=(0.03, 0.95)):
    """ DCC(1,1) with correlation targeting (Engle 2002), two-step : marginals already standardised. """
    e = np.asarray(e)
    Qbar = np.cov(e.T, bias=True)

    def f(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        return -correlation_loglik(e, _q_path(e, a, b, Qbar))

    a, b = optimize.minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6}).x
    return {"model": "DCC", "params": {"a": a, "b": b, "a+b": a + b, "half-life (days)": np.log(0.5) / np.log(a + b)},
            "loglik": -f((a, b)), "k": 3, "rho": _q_path(e, a, b, Qbar)}


def compare_conditional_models(eps):
    """ Fits CCC, EWMA and DCC on the standardised residuals ; likelihoods, information criteria and LR tests.
    CCC is nested in DCC (a = 0, b free : 2 restrictions with b unidentified, chi2(2) is conservative) ; EWMA is
    DCC on the boundary a + b = 1 (1 restriction, the chi2(1) p-value is approximate). """
    e = np.asarray(eps)
    n = len(e)
    fits = [ccc_fit(e), ewma_fit(e), dcc_fit(e)]
    table = pd.DataFrame([{"model": f["model"], "loglik": f["loglik"], "k": f["k"],
                           "AIC": 2 * f["k"] - 2 * f["loglik"], "BIC": f["k"] * np.log(n) - 2 * f["loglik"],
                           "params": {k: round(v, 4) for k, v in f["params"].items()}} for f in fits]).set_index("model")
    ll = table.loglik
    lr = pd.DataFrame([
        {"H0": "CCC (constant)", "H1": "DCC", "LR": 2 * (ll.DCC - ll.CCC), "df": 2},
        {"H0": "EWMA (integrated, a+b=1)", "H1": "DCC", "LR": 2 * (ll.DCC - ll.EWMA), "df": 1},
        {"H0": "CCC (constant)", "H1": "EWMA", "LR": 2 * (ll.EWMA - ll.CCC), "df": 1}])
    lr["p-value"] = stats.chi2.sf(lr.LR, lr.df)
    paths = pd.DataFrame({f["model"]: f["rho"] for f in fits}, index=eps.index)
    return table, lr, paths, {f["model"]: f["params"] for f in fits}


# ----------------------------------------------------------------------------------------------------------------------
# Mean reversion
# ----------------------------------------------------------------------------------------------------------------------

def mean_reversion_table(r, windows=(63, 126, 252)):
    """ Rolling correlation over several windows : level, dispersion, ADF on its Fisher transform (H0 : unit root)
    and AR(1) coefficient of non-overlapping windows. """
    rows = []
    for w in windows:
        rc = r.iloc[:, 0].rolling(w).corr(r.iloc[:, 1]).dropna()
        z = fisher_z(rc)
        zn = z.iloc[::w]
        rows.append({"window (days)": w, "mean": rc.mean(), "std": rc.std(), "min": rc.min(), "max": rc.max(),
                     "ADF p": adfuller(z, maxlag=w // 5, autolag=None)[1],
                     "AR(1) non-overlapping": np.corrcoef(zn[:-1], zn[1:])[0, 1]})
    return pd.DataFrame(rows).set_index("window (days)")


# ----------------------------------------------------------------------------------------------------------------------
# Asymmetry and tails, against a Gaussian dependence with the same correlation
# ----------------------------------------------------------------------------------------------------------------------

def exceedance_correlation(x, thresholds=(0.0, 0.5, 1.0, 1.5), n_sim=2_000_000, seed=0):
    """ Longin-Solnik (2001) : correlation conditional on both standardised returns below -q (down) or above q (up).
    The Gaussian column is the same statistic under a bivariate normal with the sample correlation (simulated) : the
    exceedance correlation of a Gaussian decreases with q, only the gap to it is informative. """
    s = ((x - x.mean()) / x.std()).values
    rho = np.corrcoef(s.T)[0, 1]
    g = np.random.default_rng(seed).multivariate_normal([0, 0], [[1, rho], [rho, 1]], n_sim)
    rows = []
    for q in thresholds:
        for side, sign in [("down", -1), ("up", 1)]:
            m = (sign * s[:, 0] > q) & (sign * s[:, 1] > q)
            mg = (sign * g[:, 0] > q) & (sign * g[:, 1] > q)
            rows.append({"q (sd)": q, "side": side, "n": m.sum(), "rho data": np.corrcoef(s[m].T)[0, 1],
                         "rho gaussian": np.corrcoef(g[mg].T)[0, 1]})
    return pd.DataFrame(rows).set_index(["q (sd)", "side"])


def tail_dependence(x, quantiles=(0.10, 0.05, 0.01)):
    """ Empirical lambda_L(q) = P(U < q, V < q) / q and lambda_U(q) on ranks, against the Gaussian copula with the
    sample correlation (whose tail dependence tends to 0 as q -> 0). """
    n = len(x)
    U = x.rank().values / (n + 1)
    rho = x.corr().iloc[0, 1]
    mvn = stats.multivariate_normal(cov=[[1, rho], [rho, 1]])
    rows = []
    for q in quantiles:
        gauss = mvn.cdf([stats.norm.ppf(q)] * 2) / q
        rows.append({"q": q, "lower data": ((U[:, 0] < q) & (U[:, 1] < q)).mean() / q,
                     "upper data": ((U[:, 0] > 1 - q) & (U[:, 1] > 1 - q)).mean() / q, "gaussian": gauss,
                     "joint obs. lower": int(((U[:, 0] < q) & (U[:, 1] < q)).sum())})
    return pd.DataFrame(rows).set_index("q")


def regime_correlation(r, window=21, quantile=0.2):
    """ Rolling correlation in high / low volatility regimes and in down / up markets of the equally weighted basket.
    Caveat (Forbes-Rigobon 2002) : conditioning on high volatility mechanically raises the sample correlation even
    when the underlying correlation is constant ; the gap is compared to a constant-correlation simulation. """
    rc = r.iloc[:, 0].rolling(window).corr(r.iloc[:, 1])
    vol = r.rolling(window).std().mean(axis=1)          # average of the individual vols : the basket vol would
                                                        # itself depend on the correlation
    ret = r.mean(axis=1).rolling(window).sum()
    hi, lo = vol > vol.quantile(1 - quantile), vol < vol.quantile(quantile)
    return pd.Series({f"high vol (top {quantile:.0%})": rc[hi].mean(), f"low vol (bottom {quantile:.0%})": rc[lo].mean(),
                      f"down market ({window}d basket < 0)": rc[ret < 0].mean(),
                      f"up market ({window}d basket > 0)": rc[ret > 0].mean(),
                      "all": rc.mean()}, name="mean rolling correlation")


def regime_correlation_benchmark(r, window=21, quantile=0.2, n_paths=50, seed=0):
    """ Same statistics on paths with the sample volatilities and a constant correlation (resampled GARCH-free
    Gaussian returns scaled by the realised volatility path of the data), to separate the Forbes-Rigobon bias from
    a genuine dependence of the correlation on the regime. """
    rng = np.random.default_rng(seed)
    rho = r.corr().iloc[0, 1]
    sig = r.rolling(window, min_periods=2).std().bfill().values
    L = np.linalg.cholesky([[1, rho], [rho, 1]])
    out = []
    for _ in range(n_paths):
        z = rng.standard_normal((len(r), 2)) @ L.T
        out.append(regime_correlation(pd.DataFrame(z * sig, index=r.index, columns=r.columns), window, quantile))
    return pd.concat(out, axis=1).mean(axis=1).rename("constant-corr. benchmark")
