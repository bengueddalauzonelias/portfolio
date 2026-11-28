# Non-stochastic multivariate correlation models for worst-of / best-of pricing (Models/correlation.ipynb). Each model
# is fitted on the daily log returns of N assets known at the pricing date and gives the pricing correlation matrix as
# a deterministic function of the maturity : R(H) = average forecast correlation matrix over the next H trading days.
# In a multi-asset Black-Scholes model with constant volatilities only the integrated covariance matters for a
# European payoff, so R(H) is the input of the pricer ; the daily path E_t[R_{t+h}] is kept for path-dependent
# products. Every R(H) is a valid correlation matrix (unit diagonal, positive semi-definite).
# - HistoricalCorrelation      : sample correlation matrix over a window (full sample, last n days, since a date, or
#                                the maturity of the product) ; flat forecast
# - EWMACorrelation            : exponentially weighted correlation (RiskMetrics) on returns or on GARCH residuals,
#                                lambda by likelihood or by the forecast error over the product horizon ; flat forecast
# - DCCCorrelation             : scalar DCC(1,1) on GARCH residuals (Engle 2002) ; forecast reverting to the long-run
#                                correlation matrix (Engle-Sheppard approximation)
# - BlendCorrelation           : short- and long-window correlation matrices, the weight of the short one decaying with
#                                the maturity at a speed fitted on the realised forward correlations of all the pairs
# - RegimeSwitchingCorrelation : correlation matrix of the GARCH residuals switching between two regimes with a Markov
#                                chain (EM with the Hamilton filter and the Kim smoother) ; forecast = expected
#                                correlation matrix given the filtered probabilities
# - StudentTCopula             : multivariate Student-t copula on the ranks of the GARCH residuals, correlation matrix
#                                from Kendall's tau and degrees of freedom by profile likelihood ; flat forecast, tail
#                                dependence
# Pairs are labelled "A/B" (i < j in the column order of the returns) ; forecasts and errors are stored per pair.
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import lfilter
from scipy.special import gammaln

import correlation_tests as ct

TRADING_DAYS = 252


def days(maturity):
    """ Maturity in years -> number of trading days. """
    return int(round(TRADING_DAYS * maturity))


# ----------------------------------------------------------------------------------------------------------------------
# Matrix utilities
# ----------------------------------------------------------------------------------------------------------------------

def pair_index(columns):
    """ Upper-triangle indices (i < j) and the labels "A/B" of the pairs. """
    columns = list(columns)
    i, j = np.triu_indices(len(columns), 1)
    return i, j, [f"{columns[a]}/{columns[b]}" for a, b in zip(i, j)]


def corr_matrix(x):
    return np.corrcoef(np.asarray(x).T)


def cov_to_corr(Q):
    """ Covariance (or Q) matrix -> correlation matrix, on the last two axes. """
    d = np.sqrt(np.einsum("...ii->...i", Q))
    return Q / d[..., :, None] / d[..., None, :]


def nearest_correlation(C, floor=1e-8, n_iter=100):
    """ Nearest correlation matrix in Frobenius norm (Higham 2002, alternating projections with Dykstra's
    correction). Returns C unchanged when it is already positive semi-definite. """
    C = (C + C.T) / 2
    if np.linalg.eigvalsh(C).min() >= floor:
        return C
    Y, dS = C.copy(), np.zeros_like(C)
    for _ in range(n_iter):
        Rk = Y - dS
        w, v = np.linalg.eigh(Rk)
        X = (v * np.maximum(w, floor)) @ v.T
        dS = X - Rk
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.linalg.eigvalsh(Y).min() >= 0:
            break
    return cov_to_corr((Y + Y.T) / 2)


def average_offdiagonal(R):
    """ Average pairwise correlation of a correlation matrix (or of a stack of them on the last two axes). """
    N = R.shape[-1]
    i, j = np.triu_indices(N, 1)
    return R[..., i, j].mean(axis=-1)


def gaussian_loglik(x, S):
    """ Zero-mean Gaussian log-likelihood without the 2 pi constant ; S is one matrix or one per observation. """
    logdet = np.linalg.slogdet(S)[1]
    q = np.einsum("ti,ti->t", x, np.linalg.solve(S, x[..., None])[..., 0])
    return -0.5 * float(np.sum(logdet + q))


def correlation_loglik(e, R):
    """ Correlation part of the Gaussian log-likelihood of standardised residuals (Engle 2002, eq. 26). """
    return gaussian_loglik(e, R) + 0.5 * float(np.sum(e ** 2))


def q_path(e, a, b, target, extend=False):
    """ Q_t = target (1 - a - b) + a e_{t-1} e_{t-1}' + b Q_{t-1}, Q_0 = sample covariance, as an (n, N, N) array
    (linear filter on every entry). target=None gives the integrated recursion (EWMA, a = 1 - lambda, b = lambda).
    Q_t only uses e up to t-1 ; extend=True adds Q_n, the forecast made after the last observation. """
    n = len(e) + int(extend)
    u = np.empty((n, e.shape[1], e.shape[1]))
    u[0] = np.cov(e.T, bias=True)
    u[1:] = a * e[:n - 1, :, None] * e[:n - 1, None, :]
    if target is not None:
        u[1:] += (1 - a - b) * target
    return lfilter([1.0], [1.0, -b], u, axis=0)


def rolling_correlation(r, window):
    """ Correlation of each pair over the last `window` days, one column per pair. """
    i, j, labels = pair_index(r.columns)
    return pd.DataFrame({l: r.iloc[:, a].rolling(window).corr(r.iloc[:, b]) for a, b, l in zip(i, j, labels)})


def realised_forward_correlation(r, H):
    """ Correlation of the daily returns over [t, t + H - 1] for each pair, indexed by t (NaN when the window ends
    after the sample). """
    return rolling_correlation(r, H).shift(-(H - 1))


# ----------------------------------------------------------------------------------------------------------------------
# Common interface
# ----------------------------------------------------------------------------------------------------------------------

class CorrelationModel:
    """ fit(r) on the (n, N) returns up to the pricing date ; forecast_path(H) = E_t[R_{t+h}] for h = 1..H as an
    (H, N, N) array ; pricing_correlation(H) = its average as an N x N DataFrame ; pair_correlations(H) the same as a
    Series of the N(N-1)/2 pairs. """
    name = "model"
    copula = "gaussian"

    def _set_assets(self, r):
        self.assets = list(r.columns)
        self._i, self._j, self.pairs = pair_index(self.assets)

    def fit(self, r):
        raise NotImplementedError

    def forecast_path(self, H):
        raise NotImplementedError

    def _matrix(self, H):
        return self.forecast_path(int(H)).mean(axis=0)

    def pricing_correlation(self, H):
        return pd.DataFrame(self._matrix(H), index=self.assets, columns=self.assets)

    def pair_correlations(self, H):
        return pd.Series(self._matrix(H)[self._i, self._j], index=self.pairs, name=self.name)

    def average_correlation(self, H):
        return float(average_offdiagonal(self._matrix(H)))

    def term_structure(self, horizons, pair=None):
        """ DataFrame H x pairs, or the Series of one pair ("A/B") over the horizons. """
        table = pd.DataFrame({H: self.pair_correlations(H) for H in horizons}).T
        return table if pair is None else table[pair].rename(self.name)

    def _path_frame(self, R, index):
        return pd.DataFrame(R[:, self._i, self._j], index=index, columns=self.pairs)

    @property
    def params(self):
        return {}

    def __repr__(self):
        p = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({p})"


def _flat(R, H):
    return np.broadcast_to(R, (int(H),) + R.shape)


# ----------------------------------------------------------------------------------------------------------------------
# 1. Historical
# ----------------------------------------------------------------------------------------------------------------------

class HistoricalCorrelation(CorrelationModel):
    """ Sample correlation matrix of daily returns over the last `window` days, since `start`, over the whole sample
    (both None) or over the last H days for a product of H days (match_maturity=True). """

    def __init__(self, window=None, start=None, match_maturity=False, name=None):
        self.window, self.start, self.match_maturity = window, start, match_maturity
        self.name = name or ("historical, window = maturity" if match_maturity else
                             f"historical {window}d" if window else
                             f"historical since {start}" if start else "historical, full sample")

    def fit(self, r):
        self._set_assets(r)
        self.r = r
        return self

    def _matrix(self, H):
        if self.match_maturity:
            return corr_matrix(self.r.iloc[-int(H):])
        if self.window:
            return corr_matrix(self.r.iloc[-self.window:])
        return corr_matrix(self.r.loc[self.start:] if self.start else self.r)

    def forecast_path(self, H):
        return _flat(self._matrix(H), H)

    @property
    def params(self):
        return {} if self.match_maturity else {"average rho": self.average_correlation(TRADING_DAYS)}


# ----------------------------------------------------------------------------------------------------------------------
# 2. EWMA
# ----------------------------------------------------------------------------------------------------------------------

class EWMACorrelation(CorrelationModel):
    """ Q_t = (1 - lambda) x_{t-1} x_{t-1}' + lambda Q_{t-1} on raw returns (on="returns", RiskMetrics, zero mean) or on
    GARCH residuals (on="residuals"), one lambda for the whole matrix. lambda is given, or fitted by maximum
    likelihood (criterion="likelihood" : full Gaussian likelihood on returns, correlation part on residuals) or by
    the mean squared error, over all the pairs, between rho_t and the correlation realised over [t, t + horizon - 1]
    (criterion="forecast"). The forecast is flat. """

    def __init__(self, lam=None, on="returns", criterion="likelihood", horizon=TRADING_DAYS, bounds=(0.85, 0.9995),
                 burn_in=63, name=None):
        self.lam, self.on, self.criterion, self.horizon = lam, on, criterion, horizon
        self.bounds, self.burn_in = bounds, burn_in
        self.name = name or f"EWMA {on}, {'fixed' if lam else criterion}"

    def _neg_loglik(self, x, lam):
        Q = q_path(x, 1 - lam, lam, None)[self.burn_in:]
        xb = x[self.burn_in:]
        if self.on == "residuals":
            return -correlation_loglik(xb, cov_to_corr(Q))
        return -gaussian_loglik(xb, Q)

    def _forecast_error(self, x, lam, target):
        R = cov_to_corr(q_path(x, 1 - lam, lam, None))
        err = (R[:, self._i, self._j] - target)[self.burn_in:]
        return np.nanmean(err ** 2)

    def fit(self, r):
        self._set_assets(r)
        x = ct.garch_standardise(r)[0].values if self.on == "residuals" else r.values
        if self.lam is None:
            if self.criterion == "likelihood":
                f = lambda lam: self._neg_loglik(x, lam)
            else:
                target = realised_forward_correlation(r, self.horizon).values
                f = lambda lam: self._forecast_error(x, lam, target)
            self.lam = optimize.minimize_scalar(f, bounds=self.bounds, method="bounded").x
        R = cov_to_corr(q_path(x, 1 - self.lam, self.lam, None, extend=True))
        self.rho_path = self._path_frame(R[:-1], r.index)
        self.R_next = R[-1]
        return self

    def forecast_path(self, H):
        return _flat(self.R_next, H)

    @property
    def params(self):
        return {"lambda": self.lam, "half-life (days)": np.log(0.5) / np.log(self.lam),
                "average rho_next": float(average_offdiagonal(self.R_next))}


# ----------------------------------------------------------------------------------------------------------------------
# 3. DCC
# ----------------------------------------------------------------------------------------------------------------------

def dcc_fit(e, x0=(0.03, 0.95)):
    """ Scalar DCC(1,1) with correlation targeting (Engle 2002), two-step : marginals already standardised. """
    Qbar = np.cov(e.T, bias=True)

    def f(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        return -correlation_loglik(e, cov_to_corr(q_path(e, a, b, Qbar)))

    res = optimize.minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6})
    return res.x[0], res.x[1], -res.fun, Qbar


class DCCCorrelation(CorrelationModel):
    """ Scalar DCC(1,1) with correlation targeting on GARCH(1,1)-t residuals. Forecast (Engle-Sheppard 2001) :
    E_t[R_{t+h}] = R_bar + (a + b)^(h-1) (R_{t+1} - R_bar), a convex combination of correlation matrices. """
    name = "DCC"

    def fit(self, r):
        self._set_assets(r)
        eps, self.garch = ct.garch_standardise(r)
        e = eps.values
        self.a, self.b, self.loglik, Qbar = dcc_fit(e)
        self.R_bar = cov_to_corr(Qbar)
        R = cov_to_corr(q_path(e, self.a, self.b, Qbar, extend=True))
        self.rho_path = self._path_frame(R[:-1], r.index)
        self.R_next = R[-1]
        return self

    def forecast_path(self, H):
        w = (self.a + self.b) ** np.arange(int(H))
        return self.R_bar + w[:, None, None] * (self.R_next - self.R_bar)

    @property
    def params(self):
        return {"a": self.a, "b": self.b, "a+b": self.a + self.b,
                "half-life (days)": np.log(0.5) / np.log(self.a + self.b),
                "average rho_bar": float(average_offdiagonal(self.R_bar)),
                "average rho_next": float(average_offdiagonal(self.R_next))}


# ----------------------------------------------------------------------------------------------------------------------
# 4. Blend of a short and a long window
# ----------------------------------------------------------------------------------------------------------------------

class BlendCorrelation(CorrelationModel):
    """ E_t[R_{t+h}] = R_long + phi^(h-1) (R_short - R_long), with R_short and R_long the historical correlation
    matrices over the last `short` and `long` days : the average over H days puts the weight
    w(H) = (1 - phi^H) / (H (1 - phi)) on the short window. phi = 0.5^(1 / half-life) ; the half-life is given or
    fitted by least squares, over all the pairs, on the correlations realised over [t, t + H - 1] for H in
    `horizons`. """

    def __init__(self, short=63, long=756, half_life=None, horizons=(63, 126, 252, 504), name=None):
        self.short, self.long, self.half_life, self.horizons = short, long, half_life, horizons
        self.name = name or f"blend {short}d / {long}d"

    @staticmethod
    def weight(H, phi):
        return (1 - phi ** H) / (H * (1 - phi))

    def fit(self, r):
        self._set_assets(r)
        self.R_short, self.R_long = corr_matrix(r.iloc[-self.short:]), corr_matrix(r.iloc[-self.long:])
        if self.half_life is None:
            s = rolling_correlation(r, self.short).shift(1).values       # known at t-1, forecast for [t, t + H - 1]
            l = rolling_correlation(r, self.long).shift(1).values
            targets = {H: realised_forward_correlation(r, H).values for H in self.horizons}

            def sse(log_hl):
                phi = 0.5 ** (1 / np.exp(log_hl))
                return sum(np.nanmean((l + self.weight(H, phi) * (s - l) - targets[H]) ** 2) for H in self.horizons)

            self.half_life = float(np.exp(optimize.minimize_scalar(sse, bounds=(np.log(2), np.log(5000)),
                                                                   method="bounded").x))
        self.phi = 0.5 ** (1 / self.half_life)
        return self

    def forecast_path(self, H):
        w = self.phi ** np.arange(int(H))
        return self.R_long + w[:, None, None] * (self.R_short - self.R_long)

    @property
    def params(self):
        return {"average rho_short": float(average_offdiagonal(self.R_short)),
                "average rho_long": float(average_offdiagonal(self.R_long)), "half-life (days)": self.half_life,
                "w(252)": self.weight(TRADING_DAYS, self.phi)}


# ----------------------------------------------------------------------------------------------------------------------
# 5. Markov regime switching
# ----------------------------------------------------------------------------------------------------------------------

def normal_log_density(e, R):
    """ Log-density of standardised residuals with unit variances and correlation matrix R_k, one column per regime
    (R is (K, N, N)). """
    N = e.shape[1]
    out = np.empty((len(e), len(R)))
    for k, Rk in enumerate(R):
        L = np.linalg.cholesky(Rk)
        z = np.linalg.solve(L, e.T)
        out[:, k] = -0.5 * (N * np.log(2 * np.pi) + (z * z).sum(axis=0)) - np.log(np.diag(L)).sum()
    return out


def stationary_distribution(P):
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def hamilton_filter(log_dens, P):
    """ Filtered P(s_t = k | e_1..e_t) and predicted P(s_t = k | e_1..e_{t-1}) regime probabilities and
    log-likelihood, starting from the stationary law. Works on log-densities (no underflow for large N). """
    n, k = log_dens.shape
    m = log_dens.max(axis=1)
    dens = np.exp(log_dens - m[:, None])
    pred = stationary_distribution(P)
    filt, preds, ll = np.empty((n, k)), np.empty((n, k)), float(m.sum())
    for t in range(n):
        preds[t] = pred
        joint = pred * dens[t]
        s = joint.sum()
        ll += np.log(s)
        filt[t] = joint / s
        pred = filt[t] @ P
    return filt, preds, ll


def kim_smoother(filt, preds, P):
    """ Smoothed P(s_t = k | e_1..e_n) and sum over t of the smoothed transition probabilities P(s_t = i, s_{t+1} = j). """
    n, k = filt.shape
    smooth, transitions = np.empty((n, k)), np.zeros((k, k))
    smooth[-1] = filt[-1]
    for t in range(n - 2, -1, -1):
        ratio = smooth[t + 1] / preds[t + 1]
        smooth[t] = filt[t] * (P @ ratio)
        transitions += filt[t][:, None] * P * ratio[None, :]
    return smooth, transitions


def weighted_correlation_mle(S, W, R0):
    """ argmax over correlation matrices R of -(W log det R + tr(R^-1 S)) / 2, the weighted Gaussian likelihood of
    unit-variance residuals with S = sum w_t e_t e_t' and W = sum w_t. R = corr(A A') with A lower triangular, which
    covers every correlation matrix ; started from R0. """
    N = len(S)
    il = np.tril_indices(N)

    def build(p):
        A = np.zeros((N, N))
        A[il] = p
        return cov_to_corr(A @ A.T)

    def f(p):
        R = build(p)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e20
        return W * logdet + np.trace(np.linalg.solve(R, S))

    p = optimize.minimize(f, np.linalg.cholesky(R0)[il], method="L-BFGS-B").x
    return build(p)


class RegimeSwitchingCorrelation(CorrelationModel):
    """ Two regimes s_t in {low, high} following a Markov chain with transition matrix P ; given s_t = k the GARCH
    residuals are multivariate normal with correlation matrix R_k. Fitted by EM : Hamilton filter and Kim smoother
    (E step), R_k = maximum likelihood correlation matrix of the residuals weighted by the smoothed probabilities
    and P = smoothed transition frequencies (M step), until the log-likelihood gains less than `tol`. The low regime has the lower average correlation. Forecast
    E_t[R_{t+h}] = sum_k (pi_t' P^h)_k R_k, with pi_t the filtered probabilities at the pricing date. """
    name = "regime switching"

    def __init__(self, x0=None, max_iter=2000, tol=1e-4):
        self.x0, self.max_iter, self.tol = x0, max_iter, tol                # x0 = (R (2, N, N), P (2, 2))

    @staticmethod
    def _initial(e, window=63):
        """ Split the days by the rolling average cross-product of the residuals around its median. """
        i, j = np.triu_indices(e.shape[1], 1)
        cross = pd.Series((e[:, i] * e[:, j]).mean(axis=1)).rolling(window, min_periods=1).mean().values
        high = cross > np.median(cross)
        R = np.stack([corr_matrix(e[~high]), corr_matrix(e[high])])
        return R, np.array([[0.98, 0.02], [0.02, 0.98]])

    def fit(self, r):
        self._set_assets(r)
        e = ct.garch_standardise(r)[0].values
        R, P = self.x0 if self.x0 is not None and self.x0[0].shape[-1] == e.shape[1] else self._initial(e)
        ll_old = -np.inf
        for self.n_iter in range(1, self.max_iter + 1):
            filt, preds, ll = hamilton_filter(normal_log_density(e, R), P)
            smooth, transitions = kim_smoother(filt, preds, P)
            P = transitions / transitions.sum(axis=1, keepdims=True)
            R = np.stack([weighted_correlation_mle((w[:, None] * e).T @ e, w.sum(), Rk) for w, Rk in zip(smooth.T, R)])
            if ll - ll_old < self.tol:
                break
            ll_old = ll
        order = np.argsort(average_offdiagonal(R))                          # regime 0 = low correlation
        self.R, self.P = R[order], P[np.ix_(order, order)]
        self.warm_start = (self.R, self.P)                                  # starting point of the next refit
        filt, _, self.loglik = hamilton_filter(normal_log_density(e, self.R), self.P)
        # correlation part, comparable to the DCC likelihood : remove the log-density of independent N(0, 1) marginals
        self.loglik_correlation = self.loglik - np.sum(-0.5 * e.shape[1] * np.log(2 * np.pi) - 0.5 * (e ** 2).sum(axis=1))
        self.n_params = len(self._i) * 2 + 2
        self.probabilities = pd.DataFrame(filt, index=r.index, columns=["low", "high"])
        self.pi = filt[-1]
        return self

    @property
    def R_low(self):
        return pd.DataFrame(self.R[0], index=self.assets, columns=self.assets)

    @property
    def R_high(self):
        return pd.DataFrame(self.R[1], index=self.assets, columns=self.assets)

    def forecast_path(self, H):
        probs, p = np.empty((int(H), 2)), self.pi
        for h in range(int(H)):
            p = p @ self.P
            probs[h] = p
        return np.einsum("hk,kij->hij", probs, self.R)

    @property
    def params(self):
        stay = np.diag(self.P)
        rho = average_offdiagonal(self.R)
        return {"average rho_low": float(rho[0]), "average rho_high": float(rho[1]), "p_low,low": stay[0],
                "p_high,high": stay[1], "duration low (days)": 1 / (1 - stay[0]),
                "duration high (days)": 1 / (1 - stay[1]), "P(high) today": self.pi[1]}


# ----------------------------------------------------------------------------------------------------------------------
# 6. Student-t copula
# ----------------------------------------------------------------------------------------------------------------------

def t_copula_loglik(U, R, nu):
    """ Log-likelihood of the N-dimensional Student-t copula with correlation matrix R on pseudo-observations U. """
    x = stats.t.ppf(U, nu)
    N = x.shape[1]
    L = np.linalg.cholesky(R)
    q = (np.linalg.solve(L, x.T) ** 2).sum(axis=0)
    joint = gammaln((nu + N) / 2) - gammaln(nu / 2) - N / 2 * np.log(nu * np.pi) - np.log(np.diag(L)).sum() \
        - (nu + N) / 2 * np.log1p(q / nu)
    return float(np.sum(joint - stats.t.logpdf(x, nu).sum(axis=1)))


def gaussian_copula_loglik(U, R):
    x = stats.norm.ppf(U)
    L = np.linalg.cholesky(R)
    q = (np.linalg.solve(L, x.T) ** 2).sum(axis=0)
    return float(np.sum(-np.log(np.diag(L)).sum() - 0.5 * (q - (x ** 2).sum(axis=1))))


def t_tail_dependence(rho, nu):
    """ Tail dependence coefficient of a pair of the t copula (same on both tails), 0 for the Gaussian copula ;
    element-wise on a correlation matrix. """
    rho = np.asarray(rho, dtype=float)
    lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    return float(lam) if lam.ndim == 0 else lam


class StudentTCopula(CorrelationModel):
    """ N-dimensional Student-t copula fitted on the ranks of the GARCH residuals (on="residuals") or of the returns.
    The correlation matrix is obtained from the matrix of Kendall's tau, rho_ij = sin(pi tau_ij / 2) (valid for every
    elliptical copula), projected on the nearest correlation matrix if needed ; nu (one for all the assets) maximises
    the copula likelihood given R. The Gaussian copula with the normal-scores correlation is the nu -> infinity
    limit. """
    name = "Student-t copula"
    copula = "t"

    def __init__(self, on="residuals", nu_bounds=(2.05, 200.0)):
        self.on, self.nu_bounds = on, nu_bounds

    def fit(self, r):
        self._set_assets(r)
        x = ct.garch_standardise(r)[0] if self.on == "residuals" else r
        U = x.rank().values / (len(x) + 1)
        self.tau = pd.DataFrame(U).corr(method="kendall").values
        self.R = nearest_correlation(np.sin(np.pi * self.tau / 2))
        res = optimize.minimize_scalar(lambda nu: -t_copula_loglik(U, self.R, nu), bounds=self.nu_bounds,
                                       method="bounded")
        self.nu, self.loglik = float(res.x), -res.fun
        self.R_gaussian = corr_matrix(stats.norm.ppf(U))
        self.loglik_gaussian = gaussian_copula_loglik(U, self.R_gaussian)
        return self

    def forecast_path(self, H):
        return _flat(self.R, H)

    @property
    def tail_dependence(self):
        """ N x N matrix of the pairwise tail dependence coefficients. """
        lam = t_tail_dependence(np.clip(self.R, -1, 1 - 1e-12), self.nu)
        np.fill_diagonal(lam, 1.0)
        return pd.DataFrame(lam, index=self.assets, columns=self.assets)

    @property
    def params(self):
        return {"average tau": float(average_offdiagonal(self.tau)), "average rho": float(average_offdiagonal(self.R)),
                "nu": self.nu, "average tail dependence": float(average_offdiagonal(self.tail_dependence.values)),
                "LR vs gaussian": 2 * (self.loglik - self.loglik_gaussian)}


# ----------------------------------------------------------------------------------------------------------------------
# Out-of-sample evaluation
# ----------------------------------------------------------------------------------------------------------------------

def realised_correlation_after(r, H):
    """ Correlation of the daily returns over [t + 1, t + H] for each pair, indexed by t : the target of a forecast
    made at the close of t. """
    return realised_forward_correlation(r, H).shift(-1)


def backtest(r, factories, start, step=21, horizons=(63, 126, 252, 504), verbose=False):
    """ Every `step` trading days from `start`, refits each model on the returns up to t (expanding window, nothing
    after t is used) and stores its pricing correlation rho_ij(H) for every pair. `factories` maps a name to a
    function of the model fitted at the previous date (None at the first one), so that a model can be
    warm-started. """
    dates = r.index[r.index >= pd.Timestamp(start)][::step]
    previous, rows = {}, []
    for k, t in enumerate(dates):
        rt = r.loc[:t]
        for name, factory in factories.items():
            m = factory(previous.get(name)).fit(rt)
            previous[name] = m
            for H in horizons:
                rows += [(t, name, H, p, v) for p, v in m.pair_correlations(H).items()]
        if verbose and k % 12 == 0:
            print(f"{t.date()} ({k + 1}/{len(dates)})")
    return pd.DataFrame(rows, columns=["date", "model", "H", "pair", "forecast"])


def _newey_west_variance(d, lags):
    """ Bartlett-weighted long-run variance, autocovariances divided by n (non-negative by construction). """
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (d[l:] @ d[:-l]) / n
    return v


def evaluate(forecasts, r, start=None, end=None, step=21):
    """ Forecast errors against the realised correlation over [t + 1, t + H], for every pair : number of forecast
    dates and of pairs, bias (forecast minus realised), MAE and RMSE over all the (date, pair) forecasts per model
    and horizon, and the Diebold-Mariano test of equal loss against the model with the lowest RMSE at that horizon,
    the loss at a date being the mean squared error over the pairs. Realised windows overlap over ceil(H / step) - 1
    evaluation dates, so the variance of the loss differential is Newey-West with that many lags. A backtest without
    a "pair" column (two assets) is read as the single pair of r. """
    f = forecasts.copy()
    if "pair" not in f:
        f["pair"] = pair_index(r.columns)[2][0]
    if start is not None:
        f = f[f.date >= pd.Timestamp(start)]
    if end is not None:
        f = f[f.date <= pd.Timestamp(end)]
    realised = pd.concat({H: realised_correlation_after(r, H).stack() for H in f.H.unique()},
                         names=["H", "date", "pair"]).rename("realised").reset_index()
    f = f.merge(realised, on=["H", "date", "pair"], how="left").dropna(subset=["realised"])
    f["error"] = f.forecast - f.realised
    table = f.groupby(["H", "model"]).agg(n=("date", "nunique"), pairs=("pair", "nunique"),
                                          bias=("error", "mean"), MAE=("error", lambda e: e.abs().mean()),
                                          RMSE=("error", lambda e: np.sqrt((e ** 2).mean())))
    dm = {}
    for H, g in f.groupby("H"):
        loss = (g.error ** 2).groupby([g.date, g.model]).mean().unstack("model").dropna()
        best = table.loc[H].RMSE.idxmin()
        lags = int(np.ceil(H / step)) - 1
        for model in loss:
            d = (loss[model] - loss[best]).values
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
        "regime switching": lambda prev: RegimeSwitchingCorrelation(prev.warm_start if prev else None),
        "Student-t copula": lambda prev: StudentTCopula(),
    }
