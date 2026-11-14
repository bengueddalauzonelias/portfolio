import numpy as np
from scipy.stats import norm

try:
    from .rate_pricing import bachelier, implied_vol
except ImportError:
    from rate_pricing import bachelier, implied_vol


# Constant Maturity Swap (CMS) tools, model independent :
# - forward_swap(curve, T, tenor)       : forward par swap rate S(0) and annuity A(0) of the swap starting at T ;
# - CMSReplication                      : value of CMS rates, caplets and floorlets paid at Tp from the swaption
#                                         smile (static replication, Hagan 2003), with a LINEAR TERMINAL SWAP RATE
#                                         (TSR) annuity mapping P(T,Tp) / A(T) ~ G(S) = a S + b (Andersen &
#                                         Piterbarg 2010, Ch. 16) ; the slope a comes from a one-factor Gaussian
#                                         model of mean reversion kappa, b makes G(S(0)) = P(0,Tp) / A(0) exact ;
# - swap_rates_on_paths, mc_swaption_vols : swap rates and swaption smiles from ANY model exposing
#       simulate(times, n_paths, seed, antithetic) -> (states, D)    and    bond_price(t, T, state)
#   (HullWhite, G2pp, QuasiGaussian, SABRLMM), used to test calibrations and convexity adjustments.
# Single-curve framework, annual fixed leg (EUR convention), times in year fractions.


def swap_schedule(T, tenor, frequency=1):
    """ Fixed leg payment times and accruals of the swap starting at T of the given tenor. """
    n = int(round(tenor * frequency))
    times = T + np.arange(1, n + 1) / frequency
    return times, np.full(n, 1.0 / frequency)


def forward_swap(curve, T, tenor, frequency=1):
    """ Forward par swap rate S(0) and annuity A(0) of the swap starting at T (single-curve). """
    times, accruals = swap_schedule(T, tenor, frequency)
    annuity = float(np.sum(accruals * curve.discount(times)))
    return (curve.discount(T) - curve.discount(times[-1])) / annuity, annuity


class CMSReplication:
    """
    CMS rates and options by static replication on a swaption smile.

    Under the annuity measure Q^A the swap rate S(T) is a martingale and swaptions give its distribution.
    A payoff f(S(T)) paid at Tp is worth A(0) E^A[f(S) P(T,Tp) / A(T)] ~ A(0) E^A[f(S) G(S)] with G(S) = a S + b.
    With undiscounted payer / receiver prices C(k) = E^A[(S-k)+], P(k) = E^A[(k-S)+] (Carr-Madan) :
        E^Tp[S]       = (a E^A[S^2] + b S0) / G(S0),   E^A[S^2] = S0^2 + 2 int_{S0}^U C + 2 int_L^{S0} P,
        E^Tp[(S-K)+]  = (G(K) C(K) + 2 a int_K^U C) / G(S0),
        E^Tp[(K-S)+]  = (G(K) P(K) - 2 a int_L^K P) / G(S0).
    The strike integrals are truncated at S0 -+ n_std sigma_ATM sqrt(T) : the SABR expansion is not reliable far
    in the wings and its call prices decay too slowly for E^A[S^2] to converge without a cut-off.
    """

    def __init__(self, curve, smile, mean_reversion=0.03, n_std=5.0, n_nodes=200, frequency=1):
        """
        Init method.
        - curve          : YieldCurve
        - smile          : function (expiry, tenor, forward) -> smile object with option(K, is_call) and
                           normal_vol(K) (e.g. a SABR), the swaption smile of the swap starting at expiry
        - mean_reversion : kappa of the Gaussian model giving the slope of the TSR mapping
        - n_std          : truncation of the strike integrals in ATM standard deviations
        """
        self.curve = curve
        self.smile = smile
        self.kappa = float(mean_reversion)
        self.n_std = float(n_std)
        self.frequency = frequency
        self._nodes, self._weights = np.polynomial.legendre.leggauss(n_nodes)

    # ------------------------------------------------------------------
    # Annuity mapping
    # ------------------------------------------------------------------
    def mapping(self, T, tenor, Tp):
        """
        Linear TSR coefficients (a, b) of G(S) = a S + b ~ P(T,Tp) / A(T), with S0 and A0.
        The slope is dG/dS at the forward in the one-factor Gaussian model P(T,t) = P(0,t)/P(0,T) exp(-B(T,t) x).
        """
        times, accruals = swap_schedule(T, tenor, self.frequency)
        P0 = self.curve.discount(np.concatenate([[T], times, [Tp]]))
        k = self.kappa
        B = lambda t: (1 - np.exp(-k * (t - T))) / k

        def state(x):
            bonds = P0 / P0[0] * np.exp(-B(np.concatenate([[T], times, [Tp]])) * x)
            annuity = np.sum(accruals * bonds[1:-1])
            return (1 - bonds[-2]) / annuity, bonds[-1] / annuity

        h = 1e-4
        (s_up, g_up), (s_dn, g_dn) = state(h), state(-h)
        a = (g_up - g_dn) / (s_up - s_dn)
        S0, A0 = forward_swap(self.curve, T, tenor, self.frequency)
        b = self.curve.discount(Tp) / A0 - a * S0
        return a, b, S0, A0

    def _integral(self, f, lower, upper):
        """ Gauss-Legendre integral of f on [lower, upper] (f vectorised). """
        if upper <= lower:
            return 0.0
        mid, half = 0.5 * (upper + lower), 0.5 * (upper - lower)
        return float(half * np.sum(self._weights * f(mid + half * self._nodes)))

    def _setup(self, T, tenor, Tp):
        a, b, S0, A0 = self.mapping(T, tenor, Tp)
        smile = self.smile(T, tenor, S0)
        width = self.n_std * float(smile.normal_vol(S0)) * np.sqrt(T)
        return a, b, S0, smile, S0 - width, S0 + width

    # ------------------------------------------------------------------
    # CMS rates and options (expectations under the Tp-forward measure)
    # ------------------------------------------------------------------
    def cms_rate(self, T, tenor, Tp=None):
        """ E^Tp[S(T)] : the convexity-adjusted forward of the CMS rate fixing at T, paid at Tp (default T). """
        Tp = T if Tp is None else Tp
        if T <= 1e-10:
            return forward_swap(self.curve, 0.0, tenor, self.frequency)[0]
        a, b, S0, smile, L, U = self._setup(T, tenor, Tp)
        second_moment = S0**2 + 2 * self._integral(lambda k: smile.option(k, True), S0, U) \
                        + 2 * self._integral(lambda k: smile.option(k, False), L, S0)
        return (a * second_moment + b * S0) / (a * S0 + b)

    def convexity_adjustment(self, T, tenor, Tp=None):
        """ E^Tp[S(T)] - S(0) (in rate units). """
        return self.cms_rate(T, tenor, Tp) - forward_swap(self.curve, T, tenor, self.frequency)[0]

    def cms_option(self, T, tenor, K, Tp=None, is_call=True):
        """ Undiscounted CMS caplet E^Tp[(S(T)-K)+] or floorlet E^Tp[(K-S(T))+]. """
        Tp = T if Tp is None else Tp
        if T <= 1e-10:
            S0 = forward_swap(self.curve, 0.0, tenor, self.frequency)[0]
            return max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)
        a, b, S0, smile, L, U = self._setup(T, tenor, Tp)
        G = lambda s: a * s + b
        if is_call:
            value = G(K) * float(smile.option(K, True)) + 2 * a * self._integral(lambda k: smile.option(k, True), max(K, L), U)
        else:
            value = G(K) * float(smile.option(K, False)) - 2 * a * self._integral(lambda k: smile.option(k, False), L, min(K, U))
        return value / G(S0)

    def cms_rate_bachelier(self, T, tenor, normal_vol, Tp=None):
        """ Closed form with a flat normal volatility : E^Tp[S] = S0 + a sigma^2 T / G(S0) (linear TSR). """
        Tp = T if Tp is None else Tp
        a, b, S0, _ = self.mapping(T, tenor, Tp)
        return S0 + a * normal_vol**2 * T / (a * S0 + b)


# ======================================================================================================
# Swap rates and swaption smiles from a simulation model
# ======================================================================================================
def swap_rates_on_paths(model, t, tenor, state, frequency=1):
    """ Par swap rate S(t) and annuity A(t) of the spot-starting swap of the given tenor, on every path. """
    times, accruals = swap_schedule(t, tenor, frequency)
    bonds = model.bond_price(t, times[:, None], state)
    annuity = np.sum(accruals[:, None] * bonds, axis=0)
    return (1.0 - bonds[-1]) / annuity, annuity


def mc_swaption_vols(model, expiry, tenor, strikes, n_paths=50_000, seed=1, antithetic=True, frequency=1):
    """
    European payer swaptions of expiry x tenor by Monte Carlo in `model`, returned as implied normal vols
    (with the forward swap rate of the model's curve) together with the Monte Carlo forward swap rate.
    Out-of-the-money options (receivers below the forward) are used for the implied volatilities.
    """
    states, D = model.simulate(np.array([expiry]), n_paths, seed, antithetic)
    S, A = swap_rates_on_paths(model, expiry, tenor, states[0], frequency)
    S0, A0 = forward_swap(model.curve, expiry, tenor, frequency)
    deflated = D[0] * A
    vols = []
    for K in np.atleast_1d(strikes):
        call = K >= S0
        payoff = np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)
        vols.append(implied_vol(float(np.mean(deflated * payoff)) / A0, S0, float(K), expiry, call, 'bachelier'))
    return np.array(vols), float(np.mean(deflated * S) / A0)
