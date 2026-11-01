# Non-stochastic correlation models for worst-of / best-of pricing (Models/correlation.ipynb). Each model is fitted on
# the daily log returns known at the pricing date and gives the pricing correlation as a deterministic function of
# the maturity : rho(H) = average forecast correlation over the next H trading days. In a multi-asset Black-Scholes
# model with constant volatilities only the integrated covariance matters for a European payoff, so rho(H) is the
# input of the pricer ; the daily path E_t[rho_{t+h}] is kept for path-dependent products.
# - HistoricalCorrelation      : sample correlation over a window (full sample, last n days, since a date, or the
#                                maturity of the product) ; flat forecast
# - EWMACorrelation            : exponentially weighted correlation (RiskMetrics) on returns or on GARCH residuals,
#                                lambda by likelihood or by the forecast error over the product horizon ; flat forecast
# - DCCCorrelation             : DCC(1,1) on GARCH residuals (Engle 2002) ; forecast reverting to the long-run
#                                correlation (Engle-Sheppard approximation)
# - BlendCorrelation           : short- and long-window correlations, the weight of the short one decaying with the
#                                maturity at a speed fitted on the realised forward correlations
# - RegimeSwitchingCorrelation : correlation of the GARCH residuals switching between two levels with a Markov chain
#                                (Hamilton filter) ; forecast = expected correlation given the filtered probabilities
# - StudentTCopula             : Student-t copula on the ranks of the GARCH residuals, correlation from Kendall's tau
#                                and degrees of freedom by profile likelihood ; flat forecast, tail dependence
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

import correlation_tests as ct

TRADING_DAYS = 252


def days(maturity):
    """ Maturity in years -> number of trading days. """
    return int(round(TRADING_DAYS * maturity))


def _corr(x):
    return float(np.corrcoef(np.asarray(x).T)[0, 1])


def realised_forward_correlation(r, H):
    """ Correlation of the daily returns over [t, t + H - 1], indexed by t (NaN when the window ends after the sample). """
    return r.iloc[:, 0].rolling(H).corr(r.iloc[:, 1]).shift(-(H - 1))


class CorrelationModel:
    """ Common interface : fit(r) on the returns up to the pricing date, forecast_path(H) = E_t[rho_{t+h}] for
    h = 1..H, pricing_correlation(H) = its average. """
    name = "model"
    copula = "gaussian"

    def fit(self, r):
        raise NotImplementedError

    def forecast_path(self, H):
        raise NotImplementedError

    def pricing_correlation(self, H):
        return float(np.mean(self.forecast_path(int(H))))

    def term_structure(self, horizons):
        return pd.Series({H: self.pricing_correlation(H) for H in horizons}, name=self.name)

    @property
    def params(self):
        return {}

    def __repr__(self):
        p = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({p})"


# ----------------------------------------------------------------------------------------------------------------------
# 1. Historical
# ----------------------------------------------------------------------------------------------------------------------

class HistoricalCorrelation(CorrelationModel):
    """ Sample correlation of daily returns over the last `window` days, since `start`, over the whole sample (both
    None) or over the last H days for a product of H days (match_maturity=True). """

    def __init__(self, window=None, start=None, match_maturity=False, name=None):
        self.window, self.start, self.match_maturity = window, start, match_maturity
        self.name = name or ("historical, window = maturity" if match_maturity else
                             f"historical {window}d" if window else
                             f"historical since {start}" if start else "historical, full sample")

    def fit(self, r):
        self.r = r
        return self

    def pricing_correlation(self, H):
        if self.match_maturity:
            return _corr(self.r.iloc[-int(H):])
        if self.window:
            return _corr(self.r.iloc[-self.window:])
        return _corr(self.r.loc[self.start:] if self.start else self.r)

    def forecast_path(self, H):
        return np.full(int(H), self.pricing_correlation(H))

    @property
    def params(self):
        return {"rho": self.pricing_correlation(TRADING_DAYS)} if not self.match_maturity else {}


# ----------------------------------------------------------------------------------------------------------------------
# 2. EWMA
# ----------------------------------------------------------------------------------------------------------------------

class EWMACorrelation(CorrelationModel):
    """ Q_t = (1 - lambda) x_{t-1} x_{t-1}' + lambda Q_{t-1} on raw returns (on="returns", RiskMetrics, zero mean) or on
    GARCH residuals (on="residuals"). lambda is given, or fitted by maximum likelihood (criterion="likelihood" : full
    Gaussian likelihood on returns, correlation part on residuals) or by the mean squared error between rho_t and the
    correlation realised over [t, t + horizon - 1] (criterion="forecast"). The forecast is flat. """

    def __init__(self, lam=None, on="returns", criterion="likelihood", horizon=TRADING_DAYS, bounds=(0.85, 0.9995),
                 burn_in=63, name=None):
        self.lam, self.on, self.criterion, self.horizon = lam, on, criterion, horizon
        self.bounds, self.burn_in = bounds, burn_in
        self.name = name or f"EWMA {on}, {'fixed' if lam else criterion}"

    def _path(self, x, lam, extend=False):
        return ct.q_entries(x, 1 - lam, lam, None, extend)

    def _neg_loglik(self, x, lam):
        q11, q22, q12 = (q[self.burn_in:] for q in self._path(x, lam))
        x1, x2 = x[self.burn_in:, 0], x[self.burn_in:, 1]
        if self.on == "residuals":
            return -ct.correlation_loglik(x[self.burn_in:], q12 / np.sqrt(q11 * q22))
        det = q11 * q22 - q12 ** 2
        return 0.5 * np.sum(np.log(det) + (q22 * x1 ** 2 + q11 * x2 ** 2 - 2 * q12 * x1 * x2) / det)

    def _forecast_error(self, x, lam, target):
        q11, q22, q12 = self._path(x, lam)
        err = (q12 / np.sqrt(q11 * q22) - target)[self.burn_in:]
        return np.nanmean(err ** 2)

    def fit(self, r):
        if self.on == "residuals":
            eps, _ = ct.garch_standardise(r)
            x = eps.values
        else:
            x = r.values
        if self.lam is None:
            if self.criterion == "likelihood":
                f = lambda lam: self._neg_loglik(x, lam)
            else:
                target = realised_forward_correlation(r, self.horizon).values
                f = lambda lam: self._forecast_error(x, lam, target)
            self.lam = optimize.minimize_scalar(f, bounds=self.bounds, method="bounded").x
        q11, q22, q12 = self._path(x, self.lam, extend=True)
        rho = q12 / np.sqrt(q11 * q22)
        self.rho_path = pd.Series(rho[:-1], index=r.index)
        self.rho_next = float(rho[-1])
        return self

    def forecast_path(self, H):
        return np.full(int(H), self.rho_next)

    @property
    def params(self):
        return {"lambda": self.lam, "half-life (days)": np.log(0.5) / np.log(self.lam), "rho_next": self.rho_next}


# ----------------------------------------------------------------------------------------------------------------------
# 3. DCC
# ----------------------------------------------------------------------------------------------------------------------

class DCCCorrelation(CorrelationModel):
    """ DCC(1,1) with correlation targeting on GARCH(1,1)-t residuals. Forecast (Engle-Sheppard 2001) :
    E_t[rho_{t+h}] = rho_bar + (a + b)^(h-1) (rho_{t+1} - rho_bar). """
    name = "DCC"

    def fit(self, r):
        eps, self.garch = ct.garch_standardise(r)
        e = eps.values
        fit = ct.dcc_fit(e)
        self.a, self.b = fit["params"]["a"], fit["params"]["b"]
        Qbar = np.cov(e.T, bias=True)
        self.rho_bar = float(Qbar[0, 1] / np.sqrt(Qbar[0, 0] * Qbar[1, 1]))
        rho = ct._q_path(e, self.a, self.b, Qbar, extend=True)
        self.rho_path = pd.Series(rho[:-1], index=r.index)
        self.rho_next = float(rho[-1])
        return self

    def forecast_path(self, H):
        h = np.arange(1, int(H) + 1)
        return self.rho_bar + (self.a + self.b) ** (h - 1) * (self.rho_next - self.rho_bar)

    @property
    def params(self):
        return {"a": self.a, "b": self.b, "a+b": self.a + self.b,
                "half-life (days)": np.log(0.5) / np.log(self.a + self.b), "rho_bar": self.rho_bar,
                "rho_next": self.rho_next}


# ----------------------------------------------------------------------------------------------------------------------
# 4. Blend of a short and a long window
# ----------------------------------------------------------------------------------------------------------------------

class BlendCorrelation(CorrelationModel):
    """ E_t[rho_{t+h}] = rho_long + phi^(h-1) (rho_short - rho_long), with rho_short and rho_long the historical
    correlations over the last `short` and `long` days : the average over H days puts the weight
    w(H) = (1 - phi^H) / (H (1 - phi)) on the short window. phi = 0.5^(1 / half-life) ; the half-life is given or
    fitted by least squares on the correlations realised over [t, t + H - 1] for H in `horizons`. """

    def __init__(self, short=63, long=756, half_life=None, horizons=(63, 126, 252, 504), name=None):
        self.short, self.long, self.half_life, self.horizons = short, long, half_life, horizons
        self.name = name or f"blend {short}d / {long}d"

    @staticmethod
    def weight(H, phi):
        return (1 - phi ** H) / (H * (1 - phi))

    def fit(self, r):
        x, y = r.iloc[:, 0], r.iloc[:, 1]
        self.rho_short, self.rho_long = _corr(r.iloc[-self.short:]), _corr(r.iloc[-self.long:])
        if self.half_life is None:
            s = x.rolling(self.short).corr(y).shift(1)                 # known at t-1, forecast for [t, t + H - 1]
            l = x.rolling(self.long).corr(y).shift(1)
            targets = {H: realised_forward_correlation(r, H) for H in self.horizons}

            def sse(log_hl):
                phi = 0.5 ** (1 / np.exp(log_hl))
                return sum(np.nanmean((l + self.weight(H, phi) * (s - l) - targets[H]) ** 2) for H in self.horizons)

            self.half_life = float(np.exp(optimize.minimize_scalar(sse, bounds=(np.log(2), np.log(5000)),
                                                                   method="bounded").x))
        self.phi = 0.5 ** (1 / self.half_life)
        return self

    def forecast_path(self, H):
        h = np.arange(1, int(H) + 1)
        return self.rho_long + self.phi ** (h - 1) * (self.rho_short - self.rho_long)

    @property
    def params(self):
        return {"rho_short": self.rho_short, "rho_long": self.rho_long, "half-life (days)": self.half_life,
                "w(252)": self.weight(TRADING_DAYS, self.phi)}


# ----------------------------------------------------------------------------------------------------------------------
# 5. Markov regime switching
# ----------------------------------------------------------------------------------------------------------------------

def _bivariate_normal_density(e, rho):
    """ Density of standardised residuals with unit variances and correlation rho, one column per value of rho. """
    x, y = e[:, [0]], e[:, [1]]
    one = 1 - rho ** 2
    return np.exp(-0.5 * (x * x + y * y - 2 * rho * x * y) / one) / (2 * np.pi * np.sqrt(one))


def hamilton_filter(dens, P):
    """ Filtered regime probabilities P(s_t = k | e_1..e_t) and log-likelihood, starting from the stationary law. """
    n, k = dens.shape
    w, v = np.linalg.eig(P.T)
    pred = np.real(v[:, np.argmin(np.abs(w - 1))])
    pred = pred / pred.sum()
    filt, ll = np.empty((n, k)), 0.0
    for t in range(n):
        joint = pred * dens[t]
        s = joint.sum()
        ll += np.log(s)
        filt[t] = joint / s
        pred = filt[t] @ P
    return filt, ll


class RegimeSwitchingCorrelation(CorrelationModel):
    """ Two regimes s_t in {low, high} following a Markov chain with transition matrix P ; given s_t = k the GARCH
    residuals are bivariate normal with correlation rho_k. Maximum likelihood with the Hamilton filter ; forecast
    E_t[rho_{t+h}] = pi_t' P^h rho, with pi_t the filtered probabilities at the pricing date. """
    name = "regime switching"

    def __init__(self, x0=(0.4, 0.8, 0.98, 0.98)):
        self.x0 = x0                                                     # (rho_low, rho_high, p_low,low, p_high,high)

    def fit(self, r):
        x0 = self.x0
        eps, _ = ct.garch_standardise(r)
        e = eps.values

        def unpack(p):
            rho = np.tanh(p[:2])
            stay = 1 / (1 + np.exp(-p[2:]))
            return rho, np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])

        def nll(p):
            rho, P = unpack(p)
            return -hamilton_filter(_bivariate_normal_density(e, rho), P)[1]

        start = np.r_[np.arctanh(x0[:2]), np.log(np.array(x0[2:]) / (1 - np.array(x0[2:])))]
        res = optimize.minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6,
                                                                            "maxiter": 4000})
        rho, P = unpack(res.x)
        order = np.argsort(rho)                                          # regime 0 = low correlation
        self.rho, self.P = rho[order], P[np.ix_(order, order)]
        self.warm_start = (*self.rho, *np.diag(self.P))                  # starting point of the next refit
        self.loglik = -res.fun
        # correlation part, comparable to the DCC likelihood : remove the log-density of independent N(0, 1) marginals
        self.loglik_correlation = self.loglik - np.sum(-np.log(2 * np.pi) - 0.5 * (e ** 2).sum(axis=1))
        filt, _ = hamilton_filter(_bivariate_normal_density(e, self.rho), self.P)
        self.probabilities = pd.DataFrame(filt, index=r.index, columns=["low", "high"])
        self.pi = filt[-1]
        return self

    def forecast_path(self, H):
        out, p = np.empty(int(H)), self.pi
        for h in range(int(H)):
            p = p @ self.P
            out[h] = p @ self.rho
        return out

    @property
    def params(self):
        stay = np.diag(self.P)
        return {"rho_low": self.rho[0], "rho_high": self.rho[1], "p_low,low": stay[0], "p_high,high": stay[1],
                "duration low (days)": 1 / (1 - stay[0]), "duration high (days)": 1 / (1 - stay[1]),
                "P(high) today": self.pi[1]}


# ----------------------------------------------------------------------------------------------------------------------
# 6. Student-t copula
# ----------------------------------------------------------------------------------------------------------------------

def t_copula_loglik(U, rho, nu):
    """ Log-likelihood of the bivariate Student-t copula on pseudo-observations U in (0, 1)^2. """
    x = stats.t.ppf(U, nu)
    one = 1 - rho ** 2
    q = (x[:, 0] ** 2 + x[:, 1] ** 2 - 2 * rho * x[:, 0] * x[:, 1]) / one
    joint = gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - 0.5 * np.log(one) \
        - (nu + 2) / 2 * np.log1p(q / nu)
    return float(np.sum(joint - stats.t.logpdf(x, nu).sum(axis=1)))


def gaussian_copula_loglik(U, rho):
    x = stats.norm.ppf(U)
    one = 1 - rho ** 2
    return float(np.sum(-0.5 * np.log(one) - (rho ** 2 * (x[:, 0] ** 2 + x[:, 1] ** 2) - 2 * rho * x[:, 0] * x[:, 1])
                        / (2 * one)))


def t_tail_dependence(rho, nu):
    """ Tail dependence coefficient of the t copula (same on both tails), 0 for the Gaussian copula. """
    return float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))


class StudentTCopula(CorrelationModel):
    """ Student-t copula fitted on the ranks of the GARCH residuals (on="residuals") or of the returns. The correlation
    is obtained from Kendall's tau, rho = sin(pi tau / 2) (valid for every elliptical copula), and nu maximises the
    copula likelihood given rho. The Gaussian copula with the normal-scores correlation is the nu -> infinity limit. """
    name = "Student-t copula"
    copula = "t"

    def __init__(self, on="residuals", nu_bounds=(2.05, 200.0)):
        self.on, self.nu_bounds = on, nu_bounds

    def fit(self, r):
        x = ct.garch_standardise(r)[0] if self.on == "residuals" else r
        U = x.rank().values / (len(x) + 1)
        self.tau = stats.kendalltau(U[:, 0], U[:, 1])[0]
        self.rho = float(np.sin(np.pi * self.tau / 2))
        res = optimize.minimize_scalar(lambda nu: -t_copula_loglik(U, self.rho, nu), bounds=self.nu_bounds,
                                       method="bounded")
        self.nu, self.loglik = float(res.x), -res.fun
        self.rho_gaussian = _corr(stats.norm.ppf(U))
        self.loglik_gaussian = gaussian_copula_loglik(U, self.rho_gaussian)
        return self

    def forecast_path(self, H):
        return np.full(int(H), self.rho)

    @property
    def tail_dependence(self):
        return t_tail_dependence(self.rho, self.nu)

    @property
    def params(self):
        return {"tau": self.tau, "rho": self.rho, "nu": self.nu, "tail dependence": self.tail_dependence,
                "LR vs gaussian": 2 * (self.loglik - self.loglik_gaussian)}


# ----------------------------------------------------------------------------------------------------------------------
# Out-of-sample evaluation
# ----------------------------------------------------------------------------------------------------------------------

def realised_correlation_after(r, H):
    """ Correlation of the daily returns over [t + 1, t + H], indexed by t : the target of a forecast made at the close
    of t. """
    return realised_forward_correlation(r, H).shift(-1)


def backtest(r, factories, start, step=21, horizons=(63, 126, 252, 504), verbose=False):
    """ Every `step` trading days from `start`, refits each model on the returns up to t (expanding window, nothing
    after t is used) and stores its pricing correlation rho(H). `factories` maps a name to a function of the model
    fitted at the previous date (None at the first one), so that a model can be warm-started. """
    dates = r.index[r.index >= pd.Timestamp(start)][::step]
    previous, rows = {}, []
    for k, t in enumerate(dates):
        rt = r.loc[:t]
        for name, factory in factories.items():
            m = factory(previous.get(name)).fit(rt)
            previous[name] = m
            rows += [(t, name, H, m.pricing_correlation(H)) for H in horizons]
        if verbose and k % 12 == 0:
            print(f"{t.date()} ({k + 1}/{len(dates)})")
    return pd.DataFrame(rows, columns=["date", "model", "H", "forecast"])


def _newey_west_variance(d, lags):
    """ Bartlett-weighted long-run variance, autocovariances divided by n (non-negative by construction). """
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (d[l:] @ d[:-l]) / n
    return v


def evaluate(forecasts, r, start=None, end=None, step=21):
    """ Forecast errors against the realised correlation over [t + 1, t + H] : number of forecasts, bias (forecast
    minus realised), MAE and RMSE per model and horizon, and the Diebold-Mariano test of equal squared error against
    the model with the lowest RMSE at that horizon. Realised windows overlap over ceil(H / step) - 1 evaluation dates,
    so the variance of the loss differential is Newey-West with that many lags. """
    f = forecasts.copy()
    if start is not None:
        f = f[f.date >= pd.Timestamp(start)]
    if end is not None:
        f = f[f.date <= pd.Timestamp(end)]
    realised = {H: realised_correlation_after(r, H) for H in f.H.unique()}
    f["realised"] = [realised[H].get(t, np.nan) for t, H in zip(f.date, f.H)]
    f = f.dropna(subset=["realised"])
    f["error"] = f.forecast - f.realised
    table = f.groupby(["H", "model"]).error.agg(n="size", bias="mean", MAE=lambda e: e.abs().mean(),
                                                 RMSE=lambda e: np.sqrt((e ** 2).mean()))
    dm = {}
    for H, g in f.groupby("H"):
        errors = g.pivot(index="date", columns="model", values="error")
        best = table.loc[H].RMSE.idxmin()
        lags = int(np.ceil(H / step)) - 1
        for model in errors:
            d = (errors[model] ** 2 - errors[best] ** 2).values
            if model == best:
                dm[(H, model)] = (np.nan, np.nan)
                continue
            if np.allclose(d, 0):                                        # same forecasts as the best model
                dm[(H, model)] = (0.0, 1.0)
                continue
            stat = d.mean() / np.sqrt(_newey_west_variance(d, lags) / len(d))
            dm[(H, model)] = (stat, 2 * stats.norm.sf(abs(stat)))
    table[["DM stat vs best", "DM p-value"]] = pd.DataFrame(dm, index=["s", "p"]).T.reindex(table.index).values
    return table, f


def backtest_factories():
    """ The models compared out of sample. The historical correlation since 2015 is left out : its start date is
    chosen with hindsight (and it does not exist before 2015). """
    return {
        "historical, full sample": lambda prev: HistoricalCorrelation(),
        "historical 252d": lambda prev: HistoricalCorrelation(window=252),
        "historical 756d": lambda prev: HistoricalCorrelation(window=756),
        "historical, window = maturity": lambda prev: HistoricalCorrelation(match_maturity=True),
        "EWMA returns, likelihood": lambda prev: EWMACorrelation(on="returns"),
        "EWMA residuals, likelihood": lambda prev: EWMACorrelation(on="residuals"),
        "EWMA returns, forecast": lambda prev: EWMACorrelation(on="returns", criterion="forecast"),
        "EWMA residuals, forecast": lambda prev: EWMACorrelation(on="residuals", criterion="forecast"),
        "DCC": lambda prev: DCCCorrelation(),
        "blend 63d / 756d": lambda prev: BlendCorrelation(),
        "regime switching": lambda prev: RegimeSwitchingCorrelation(*([prev.warm_start] if prev else [])),
        "Student-t copula": lambda prev: StudentTCopula(),
    }
