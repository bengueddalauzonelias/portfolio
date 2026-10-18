import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from forward_curve import ForwardCurve

class BlackScholes:
    """
    The Black-Scholes model class.
    """
    def __init__(self, zc_rate, forward_curve):
        # Sanity checks
        if zc_rate is None or not callable(zc_rate):
            raise ValueError("zc_rate must be a callable function.")
        if forward_curve is None or not isinstance(forward_curve, ForwardCurve):
            raise ValueError("forward_curve must be an object of the ForwardCurve class.")
        # Attributes
        self.zc_rate = zc_rate
        self.forward_price = forward_curve.calc_forward

    def bs_price_vanilla(self, exp, strike, vol, isCall=True) :
        """
        The European vanilla option Black price.
        """
        # Sanity checks
        if exp < 0: raise ValueError("Expiry (exp) must be non-negative.")
        if strike <= 0: raise ValueError("Strike must be positive.")
        if vol < 0: raise ValueError("Volatility must be non-negative.")
        if not isinstance(isCall, (bool, np.bool_)):
            raise TypeError("isCall must be a boolean.")
        # Black-Scholes formula
        r = self.zc_rate(exp)
        df = np.exp(-r * exp)
        forward = self.forward_price(exp)
        v2T = vol**2 * exp
        phi = 1 if isCall else  -1
        if v2T == 0:
            # Intrinsic value
            return df * max(0, phi * (forward - strike))
        d1 = (np.log(forward / strike) + v2T / 2) / v2T**0.5
        d2 = d1 - v2T**0.5
        return df * phi * (forward * norm.cdf(phi * d1) - strike * norm.cdf(phi * d2))
 
    def bs_vega(self, exp, strike, vol) :
        """
        The Black vega greek. 
        The derivative of the option value with respect to the volatility of the underlying asset.
        """
        # Sanity checks
        if exp < 0: raise ValueError("Expiry (exp) must be non-negative.")
        if strike <= 0: raise ValueError("Strike must be positive.")
        if vol < 0: raise ValueError("Volatility must be non-negative.")
        # Black-Scholes vega formula
        r = self.zc_rate(exp)
        df = np.exp(-r * exp)
        forward = self.forward_price(exp)
        v2T = vol**2 * exp
        if v2T == 0: return 0.
        d2 = (np.log(forward / strike) - v2T / 2) / v2T**0.5
        return df * strike * norm.pdf(d2) * exp**0.5

    def bs_price_digital(self, exp, strike, vol, isCall=True, cash=1.0, asset=False):
        """
        Closed-form price of a European digital :
        cash-or-nothing  : cash * DF * N(phi d2),  asset-or-nothing : DF * F * N(phi d1),  phi = +1 call / -1 put.
        """
        # Sanity checks
        if exp <= 0: raise ValueError("Expiry (exp) must be positive.")
        if strike <= 0: raise ValueError("Strike must be positive.")
        if vol <= 0: raise ValueError("Volatility must be positive.")
        # Black formula
        df = np.exp(-self.zc_rate(exp) * exp)
        forward = self.forward_price(exp)
        sT = vol * np.sqrt(exp)
        d1 = (np.log(forward / strike) + sT**2 / 2) / sT
        phi = 1 if isCall else -1
        if asset:
            return float(df * forward * norm.cdf(phi * d1))
        return float(cash * df * norm.cdf(phi * (d1 - sT)))

    def bs_price_barrier(self, exp, strike, barrier, vol, isCall=True, direction='down', knock='out'):
        """
        Closed-form price of a continuously monitored single barrier option, without rebate
        (Merton 1973, Reiner & Rubinstein 1991, as in Haug's "Complete Guide to Option Pricing Formulas").
        The rate r = zc_rate(exp) and the cost of carry b = ln(F(0,T) / S0) / T are taken constant up to exp :
        the formula is exact for flat curves only, and an approximation otherwise.
        Knock-in prices are obtained by In / Out parity : In = Vanilla - Out.
        """
        # Sanity checks
        if exp <= 0: raise ValueError("Expiry (exp) must be positive.")
        if strike <= 0 or barrier <= 0: raise ValueError("Strike and barrier must be positive.")
        if vol <= 0: raise ValueError("Volatility must be positive.")
        if direction not in ('down', 'up'): raise ValueError("direction must be 'down' or 'up'.")
        if knock not in ('in', 'out'): raise ValueError("knock must be 'in' or 'out'.")
        # Market data
        S = self.forward_price(0)
        r = self.zc_rate(exp)
        b = np.log(self.forward_price(exp) / S) / exp
        vanilla = self.bs_price_vanilla(exp, strike, vol, bool(isCall))
        breached = S <= barrier if direction == 'down' else S >= barrier
        if breached:
            return 0.0 if knock == 'out' else vanilla
        # Haug's building blocks
        phi = 1 if isCall else -1
        eta = 1 if direction == 'down' else -1
        sT = vol * np.sqrt(exp)
        mu = (b - vol**2 / 2) / vol**2
        x1 = np.log(S / strike) / sT + (1 + mu) * sT
        x2 = np.log(S / barrier) / sT + (1 + mu) * sT
        y1 = np.log(barrier**2 / (S * strike)) / sT + (1 + mu) * sT
        y2 = np.log(barrier / S) / sT + (1 + mu) * sT
        carry, disc = S * np.exp((b - r) * exp), strike * np.exp(-r * exp)
        A = phi * carry * norm.cdf(phi * x1) - phi * disc * norm.cdf(phi * x1 - phi * sT)
        B = phi * carry * norm.cdf(phi * x2) - phi * disc * norm.cdf(phi * x2 - phi * sT)
        C = (phi * carry * (barrier / S)**(2 * (mu + 1)) * norm.cdf(eta * y1)
             - phi * disc * (barrier / S)**(2 * mu) * norm.cdf(eta * y1 - eta * sT))
        D = (phi * carry * (barrier / S)**(2 * (mu + 1)) * norm.cdf(eta * y2)
             - phi * disc * (barrier / S)**(2 * mu) * norm.cdf(eta * y2 - eta * sT))
        above = strike > barrier
        if isCall and direction == 'down':
            out = A - C if above else B - D
        elif isCall and direction == 'up':
            out = 0.0 if above else A - B + C - D
        elif direction == 'down':
            out = A - B + C - D if above else 0.0
        else:
            out = B - D if above else A - C
        out = max(float(out), 0.0)
        return out if knock == 'out' else vanilla - out


    def generate_paths(self, T, M, N, sigma, seed=None):
        """
        Generation of paths under the Black-Scholes Model (Geometric Brownian Motion).
        The drift over each step is ln(F(t2) / F(t1)), so that E[S_T] = F(0,T).
        """
        rng = np.random.default_rng(seed)
        dt = T / M
        S = np.zeros((M + 1, N))
        S[0] = self.forward_price(0)

        for t in range(M):
            fwd_drift = np.log(self.forward_price((t + 1) * dt) / self.forward_price(t * dt))
            Z = rng.standard_normal(N)
            # Exact log-normal discretisation
            S[t + 1] = S[t] * np.exp(fwd_drift - 0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z)

        return S

