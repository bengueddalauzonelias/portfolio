import numpy as np
from scipy.stats import norm


# This script is a model-agnostic Monte Carlo pricer for the products of Produits/products.py.
# It does not import them : any object with the same attributes can be priced.
# The product must expose :
# - maturity, strike, isCall, exercise (with .style and .indices(maturity, n_steps)), payoff(S) (vectorised)
# - barrier products : breached(S), barrier, knock ('in' / 'out'), monitoring ('continuous' / 'discrete')
# - optionally control_variate(model) -> (payoff function of S_T, exact price) or None
# - or, for products with several cash flows (Athena, Phoenix...) : maturity and
#   path_payoff(pricer, S, v, dt, discount_factors) -> discounted value of all the cash flows of each path
# The model must expose :
# - generate_paths(T, M, N, seed=..., antithetic=...) -> (S, v) arrays of shape (M + 1, N), or S only
#   (v is the instantaneous variance used over each step ; it is needed for the continuous barrier correction)
# - zc_rate(T) -> continuously compounded zero rate (discounting)
# Exercise :
# - European          : discounted payoff at maturity.
# - Bermudan/American : Longstaff-Schwartz (2001) regression of the continuation value.
# Barriers :
# - discrete monitoring   : the barrier is checked on the simulation dates only.
# - continuous monitoring : between two dates, the barrier crossing probability of the Brownian bridge
#                           p = exp(-2 ln(S_i / H) ln(S_i+1 / H) / (v_i dt)) is used (Beaglehole, Dybvig & Zhou 1997).
#                           European : the payoff is weighted by the survival probability (smooth, low variance).
#                           Early exercise : the crossing is sampled (U < p), since the knock date matters.


class PricingResult:
    """
    Monte Carlo price, its standard error and the independent samples it is the mean of
    (the samples are kept so that Greeks computed by finite differences have a standard error too).
    """

    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=float)
        self.price = float(np.mean(self.samples))
        self.stderr = float(np.std(self.samples, ddof=1) / np.sqrt(len(self.samples)))

    def __repr__(self):
        return f"PricingResult(price={self.price:.4f}, stderr={self.stderr:.4f})"


class MonteCarloPricer:
    """
    Monte Carlo pricer of vanilla and barrier options with European, Bermudan or American exercise.
    """

    def __init__(self, n_paths=50_000, steps_per_year=252, seed=42, antithetic=True, control_variate=True,
                 basis_degree=3):
        """
        Init method.
        - n_paths         : number of simulated paths
        - steps_per_year  : time steps per year (the number of steps of a maturity T is round(T * steps_per_year))
        - seed            : random seed. Keeping it fixed gives common random numbers across bumped models (Greeks)
        - antithetic      : antithetic variates (requires the model generate_paths to accept antithetic=True)
        - control_variate : uses the European vanilla with the same strike as control variate when the model has
                            a semi-analytical price_vanillas
        - basis_degree    : degree of the polynomial in S / K of the Longstaff-Schwartz regression
                            (the variance terms v, v * S / K, v^2 are added when the model simulates a variance)
        """
        # Sanity checks
        if n_paths < 2: raise ValueError("n_paths must be >= 2.")
        if antithetic and n_paths % 2 != 0: raise ValueError("n_paths must be even with antithetic variates.")
        if steps_per_year <= 0: raise ValueError("steps_per_year must be positive.")
        if basis_degree < 1: raise ValueError("basis_degree must be >= 1.")
        # Attributes
        self.n_paths = int(n_paths)
        self.steps_per_year = steps_per_year
        self.seed = seed
        self.antithetic = antithetic
        self.control_variate = control_variate
        self.basis_degree = basis_degree
        self._cache = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def n_steps(self, maturity):
        return max(1, int(round(maturity * self.steps_per_year)))

    def paths(self, model, maturity, n_steps=None):
        """
        Returns (S, v) simulated paths. The last simulation is cached, so that several products with the same
        maturity are priced on the same paths.
        """
        M = self.n_steps(maturity) if n_steps is None else int(n_steps)
        key = (id(model), tuple(sorted(getattr(model, 'params', {}).items())), getattr(model, 'S0', None),
               float(maturity), M, self.n_paths, self.seed, self.antithetic)
        if self._cache is not None and self._cache[0] == key and self._cache[1] is model:
            return self._cache[2]
        kwargs = {'antithetic': True} if self.antithetic else {}
        out = model.generate_paths(maturity, M, self.n_paths, seed=self.seed, **kwargs)
        S, v = out if isinstance(out, tuple) else (out, None)
        self._cache = (key, model, (S, v))
        return S, v

    # ------------------------------------------------------------------
    # Barrier
    # ------------------------------------------------------------------
    @staticmethod
    def crossing_probability(S1, S2, var, H):
        """
        Probability that the Brownian bridge of ln S between S1 and S2, with total variance var over the step,
        touches the barrier H (both ends on the same, non knocked, side of H).
        """
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            p = np.exp(-2.0 * np.log(S1 / H) * np.log(S2 / H) / var)
        return np.where(var > 0, np.nan_to_num(p, nan=0.0), 0.0)

    def survival_probability(self, product, S, v, dt):
        """
        European pricing : probability, path by path, that the barrier has NOT been touched before maturity.
        """
        breached = product.breached(S[0])
        log_survival = np.zeros(S.shape[1])
        continuous = product.monitoring == 'continuous' and v is not None
        for i in range(S.shape[0] - 1):
            breached |= product.breached(S[i + 1])
            if continuous:
                p = self.crossing_probability(S[i], S[i + 1], np.maximum(v[i], 0.0) * dt, product.barrier)
                log_survival += np.log1p(-np.minimum(p, 1.0 - 1e-16))
        return np.where(breached, 0.0, np.exp(log_survival))

    def hitting_index(self, product, S, v, dt):
        """
        Early exercise pricing : first grid index at which the barrier is touched (S.shape[0] if never).
        A crossing inside the step [i, i+1] (sampled with the Brownian bridge probability) counts at index i + 1.
        """
        M, N = S.shape[0] - 1, S.shape[1]
        rng = np.random.default_rng(None if self.seed is None else self.seed + 1)
        hit = np.where(product.breached(S[0]), 0, M + 1)
        continuous = product.monitoring == 'continuous' and v is not None
        for i in range(M):
            newly = product.breached(S[i + 1])
            if continuous:
                p = self.crossing_probability(S[i], S[i + 1], np.maximum(v[i], 0.0) * dt, product.barrier)
                newly |= rng.random(N) < p
            hit = np.where((hit > M) & newly, i + 1, hit)
        return hit

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    def discount_factors(self, model, maturity, M):
        grid = np.linspace(0.0, maturity, M + 1)
        return np.exp(-np.array([float(model.zc_rate(t)) * t for t in grid]))

    def price(self, model, product, n_steps=None):
        """
        Prices product under model. Returns a PricingResult.
        n_steps forces the number of time steps (used for theta, to keep the same random numbers).
        """
        path_dependent = hasattr(product, 'path_payoff')
        required = ('maturity',) if path_dependent else ('maturity', 'strike', 'isCall', 'exercise', 'payoff')
        missing = [a for a in required if not hasattr(product, a)]
        if missing:
            raise ValueError(f"product must expose {missing} (see Produits/products.py).")
        T = product.maturity
        S, v = self.paths(model, T, n_steps)
        M = S.shape[0] - 1
        dt = T / M
        df = self.discount_factors(model, T, M)
        is_barrier = hasattr(product, 'breached')

        if path_dependent:
            # Products with several cash flows (autocallables...) discount their own cash flows on the paths
            samples = product.path_payoff(self, S, v, dt, df)
        elif product.exercise.style == 'european':
            weight = 1.0
            if is_barrier:
                survival = self.survival_probability(product, S, v, dt)
                weight = survival if product.knock == 'out' else 1.0 - survival
            samples = df[-1] * product.payoff(S[-1]) * weight
        else:
            samples = self._longstaff_schwartz(product, S, v, dt, df, is_barrier)

        if self.control_variate and hasattr(product, 'control_variate'):
            control = product.control_variate(model)
            if control is not None:
                samples = self._apply_control_variate(samples, control, S[-1], df[-1])
        if self.antithetic:
            half = len(samples) // 2
            samples = 0.5 * (samples[:half] + samples[half:])
        return PricingResult(samples)

    def _basis(self, S, v, product, tau, fwd_ratio, df_ratio, sigma_default):
        """
        Regression basis : polynomial in x = S / K, variance terms (u = v / mean(v)) and the Black price of the
        European vanilla on the remaining time to maturity tau (vol sqrt(v), or sigma_default without variance).
        The European price is a lower bound of the continuation value of a vanilla : adding it to the basis
        prevents the polynomial regression from triggering sub-optimal early exercises.
        """
        K = product.strike
        x = S / K
        columns = [x**d for d in range(self.basis_degree + 1)]
        if v is not None:
            u = v / max(np.mean(v), 1e-12)
            columns += [u, u * x, u**2]
        sigma = np.sqrt(np.maximum(v, 1e-8)) if v is not None else sigma_default
        sT = np.maximum(sigma * np.sqrt(tau), 1e-8)
        fwd = S * fwd_ratio
        d1 = (np.log(fwd / K) + 0.5 * sT**2) / sT
        phi = 1.0 if product.isCall else -1.0
        columns.append(df_ratio * phi * (fwd * norm.cdf(phi * d1) - K * norm.cdf(phi * (d1 - sT))) / K)
        return np.column_stack(columns)

    def _longstaff_schwartz(self, product, S, v, dt, df, is_barrier):
        """
        Longstaff-Schwartz backward induction. Returns the discounted cash flow of each path.
        Knock-out : exercise only while the barrier has not been touched, the cash flow is 0 once knocked out.
        Knock-in  : exercise only once the barrier has been touched.
        The regression is done on the in-the-money paths allowed to exercise.
        """
        M, N = S.shape[0] - 1, S.shape[1]
        if is_barrier:
            hit = self.hitting_index(product, S, v, dt)
            allowed = (lambda i: hit > i) if product.knock == 'out' else (lambda i: hit <= i)
        else:
            allowed = lambda i: np.ones(N, dtype=bool)

        # Forward ratios F(0,T) / F(0,t_i) from the simulated mean (exact drift of the paths)
        mean_S = np.mean(S, axis=1)
        # Without simulated variance : constant Black vol estimated on the terminal log-returns
        sigma_default = None if v is not None else float(np.std(np.log(S[-1] / S[0])) / np.sqrt(product.maturity))

        cash_flow = product.payoff(S[-1]) * allowed(M)
        cf_index = np.full(N, M)
        for i in product.exercise.indices(product.maturity, M)[::-1]:
            if i == M:
                continue
            exercise_value = product.payoff(S[i])
            candidates = np.where(allowed(i) & (exercise_value > 0))[0]
            n_basis = self.basis_degree + 2 + (3 if v is not None else 0)
            if len(candidates) <= 2 * n_basis:
                continue
            X = self._basis(S[i, candidates], None if v is None else v[i, candidates], product, (M - i) * dt,
                            mean_S[-1] / mean_S[i], df[-1] / df[i], sigma_default)
            Y = cash_flow[candidates] * df[cf_index[candidates]] / df[i]
            coefs = np.linalg.lstsq(X, Y, rcond=None)[0]
            exercise = exercise_value[candidates] > X @ coefs
            idx = candidates[exercise]
            cash_flow[idx] = exercise_value[idx]
            cf_index[idx] = i
        return cash_flow * df[cf_index]

    @staticmethod
    def _apply_control_variate(samples, control, S_T, df_T):
        """
        Control variate : X = discounted European payoff given by the product (e.g. the vanilla with the same
        strike), whose exact price E[X] is known. Y_cv = Y - beta * (X - E[X]), beta = Cov(Y, X) / Var(X).
        """
        payoff, exact = control
        X = df_T * payoff(S_T)
        var_X = np.var(X)
        if var_X <= 0:
            return samples
        beta = np.cov(samples, X)[0, 1] / var_X
        return samples - beta * (X - exact)
