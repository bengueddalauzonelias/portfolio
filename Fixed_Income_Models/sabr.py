import numpy as np
from scipy.optimize import brentq

try:
    from .rate_pricing import bachelier, black, implied_vol
except ImportError:
    from rate_pricing import bachelier, black, implied_vol


# SABR smile of a single forward rate (Hagan, Kumar, Lesniewski & Woodward 2002) :
#     dF = alpha (F + shift)^beta dW,    d alpha = nu alpha dZ,    d<W, Z> = rho dt.
# - beta = 0 : normal SABR, the implied NORMAL (Bachelier) volatility in closed form (Hagan's expansion, Eq. A.69) :
#       sigma_N(K) = alpha zeta / x(zeta) [1 + (2 - 3 rho^2) / 24 nu^2 T],   zeta = nu (F - K) / alpha,
#       x(zeta) = ln((sqrt(1 - 2 rho zeta + zeta^2) + zeta - rho) / (1 - rho)) ;
#   rates may be negative, which is why it is used for EUR since 2015 ;
# - beta > 0 : shifted SABR, Hagan's implied (shifted) Black volatility on F + shift, converted to a normal vol.
# The expansion is accurate for nu^2 T of order one at most ; far in the wings it is not arbitrage free, so the
# CMS replication (cms.py) truncates the strike integrals.


class SABR:
    """
    SABR smile of one forward rate F of expiry T.
    """

    def __init__(self, F, T, alpha, beta=0.0, rho=0.0, nu=0.3, shift=0.0):
        if alpha <= 0: raise ValueError("alpha must be positive.")
        if not -1 < rho < 1: raise ValueError("rho must be in (-1, 1).")
        if nu < 0: raise ValueError("nu must be non-negative.")
        if not 0 <= beta <= 1: raise ValueError("beta must be in [0, 1].")
        if beta > 0 and F + shift <= 0: raise ValueError("F + shift must be positive when beta > 0.")
        self.F, self.T = float(F), float(T)
        self.alpha, self.beta, self.rho, self.nu, self.shift = float(alpha), float(beta), float(rho), float(nu), float(shift)

    @classmethod
    def from_atm_normal_vol(cls, F, T, atm_vol, beta=0.0, rho=0.0, nu=0.3, shift=0.0):
        """ SABR whose ATM normal volatility is atm_vol : alpha solved in closed form (beta = 0) or numerically. """
        if beta == 0:
            alpha = atm_vol / (1 + (2 - 3 * rho**2) / 24 * nu**2 * T)
            return cls(F, T, alpha, beta, rho, nu, shift)
        f = lambda a: cls(F, T, a, beta, rho, nu, shift).normal_vol(F) - atm_vol
        alpha = brentq(f, 1e-8, 10.0, xtol=1e-14)
        return cls(F, T, alpha, beta, rho, nu, shift)

    # ------------------------------------------------------------------
    # Implied volatilities
    # ------------------------------------------------------------------
    def _x_ratio(self, zeta):
        """ zeta / x(zeta), equal to 1 at zeta = 0 (series expansion near 0). """
        zeta = np.asarray(zeta, dtype=float)
        rho = self.rho
        small = np.abs(zeta) < 1e-6
        z = np.where(small, 1.0, zeta)
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        return np.where(small, 1 - rho * zeta / 2, z / x)

    def black_vol(self, K):
        """ Hagan's implied Black volatility of F + shift (beta > 0). """
        F, K = self.F + self.shift, np.asarray(K, dtype=float) + self.shift
        a, b, rho, nu, T = self.alpha, self.beta, self.rho, self.nu, self.T
        if np.any(K <= 0):
            raise ValueError("K + shift must be positive when beta > 0.")
        FK = F * K
        log_fk = np.log(F / K)
        omb = 1 - b
        zeta = nu / a * FK**(omb / 2) * log_fk
        denominator = FK**(omb / 2) * (1 + omb**2 / 24 * log_fk**2 + omb**4 / 1920 * log_fk**4)
        correction = 1 + (omb**2 / 24 * a**2 / FK**omb + rho * b * nu * a / (4 * FK**(omb / 2))
                          + (2 - 3 * rho**2) / 24 * nu**2) * T
        return a / denominator * self._x_ratio(zeta) * correction

    def normal_vol(self, K):
        """ Implied normal (Bachelier) volatility at strike K (vectorised in K). """
        K = np.asarray(K, dtype=float)
        if self.beta == 0:
            zeta = self.nu * (self.F - K) / self.alpha
            return self.alpha * self._x_ratio(zeta) * (1 + (2 - 3 * self.rho**2) / 24 * self.nu**2 * self.T)
        prices = black(self.F, K, self.black_vol(K), self.T, True, self.shift)
        out = np.array([implied_vol(float(p), self.F, float(k), self.T, True, 'bachelier')
                        for p, k in zip(np.ravel(prices), np.ravel(K))]).reshape(np.shape(K))
        return float(out) if out.ndim == 0 else out

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    def option(self, K, is_call=True):
        """ Undiscounted call (or put) E[(F_T - K)+] (or E[(K - F_T)+]) in the SABR smile. """
        if self.beta == 0:
            return bachelier(self.F, K, self.normal_vol(K), self.T, is_call)
        return black(self.F, K, self.black_vol(K), self.T, is_call, self.shift)

    @property
    def params(self):
        return {"alpha": self.alpha, "beta": self.beta, "rho": self.rho, "nu": self.nu, "shift": self.shift}

    def __repr__(self):
        return (f"SABR(F={100 * self.F:.4f}%, T={self.T:g}, alpha={self.alpha:.6g}, beta={self.beta:g}, "
                f"rho={self.rho:+.3f}, nu={self.nu:.3f}, shift={self.shift:g})")
