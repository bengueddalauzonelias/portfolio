import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq, root

try:
    from .yield_curve import YieldCurve
    from .instruments import Instrument
    from .interpolation import INTERPOLATORS
except ImportError:
    from yield_curve import YieldCurve
    from instruments import Instrument
    from interpolation import INTERPOLATORS


class Bootstrapper:
    """
    Builds the yield curve that exactly reprices a set of market instruments.

    One pillar per instrument, at its maturity ; the unknowns are the continuously compounded zero rates at the pillars.
    1. Sequential bootstrap with raw interpolation (piecewise constant forwards) : instruments are solved one by one,
       in maturity order, with a 1D root search. Raw interpolation being local, this is exact.
    2. For the other exact methods (splines, monotone convex), a pillar also moves the curve before the previous
       pillars, so the whole system implied_quote_i(curve) = quote_i is solved simultaneously (scipy root, hybr),
       starting from the raw solution.
    Parametric curves (Nelson-Siegel, Svensson) do not reprice instruments exactly : fit them on the bootstrapped
    zero rates instead (YieldCurve(times, zero_rates, 'nelson_siegel')).
    """

    def __init__(self, instruments, method='monotone_convex', tol=1e-10):
        """
        Init method.
        - instruments : list of Instrument objects with distinct maturities
        - method      : interpolation method of the resulting curve (exact methods only)
        - tol         : tolerance on the quote residuals (rate units, 1e-10 = 1e-6 bp)
        """
        # Sanity checks
        if not instruments:
            raise ValueError("instruments cannot be empty.")
        if not all(isinstance(inst, Instrument) for inst in instruments):
            raise ValueError("instruments must be Instrument objects.")
        if method not in INTERPOLATORS:
            raise ValueError(f"Unknown interpolation method '{method}'.")
        if not INTERPOLATORS[method].exact:
            raise ValueError(f"'{method}' is a parametric fit : bootstrap with an exact method, then fit it on the zero rates.")
        self.instruments = sorted(instruments, key=lambda inst: inst.maturity)
        self.times = np.array([inst.maturity for inst in self.instruments])
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Instrument maturities must be distinct.")
        # Attributes
        self.method = method
        self.tol = tol
        self.curve = None

    def _residuals(self, zero_rates, method=None, n=None):
        n = len(zero_rates) if n is None else n
        curve = YieldCurve(self.times[:n], zero_rates[:n], method or self.method)
        return np.array([inst.residual(curve) for inst in self.instruments[:n]])

    def _sequential(self):
        """ Pillar-by-pillar bootstrap with raw interpolation. """
        z = np.zeros(len(self.instruments))
        for i, inst in enumerate(self.instruments):
            def f(x):
                z[i] = x
                return inst.residual(YieldCurve(self.times[:i + 1], z[:i + 1], 'raw'))
            z[i] = brentq(f, -0.5, 1.0, xtol=1e-15, rtol=1e-15, maxiter=500)
        return z

    def run(self):
        """
        Bootstraps the curve, returns it and stores it in self.curve.
        """
        start = time.time()
        z = self._sequential()
        self.n_evaluations = 0
        if INTERPOLATORS[self.method] is INTERPOLATORS['raw']:
            self.solver = "sequential"
        else:
            def counted(x):
                self.n_evaluations += 1
                return self._residuals(x)
            res = root(counted, z, method='hybr', tol=1e-15)
            z = res.x
            self.solver = "simultaneous (hybr)"
        self.curve = YieldCurve(self.times, z, self.method)
        self.elapsed = time.time() - start
        self.max_residual_bp = 1e4 * float(np.max(np.abs(self._residuals(z))))
        if self.max_residual_bp > 1e4 * self.tol:
            print(f"/!\\ Warning : max bootstrap residual {self.max_residual_bp:.2e}bp above tolerance.")
        return self.curve

    def report(self):
        """ One row per instrument : market quote, quote implied by the curve, residual (bp), zero rate. """
        if self.curve is None:
            raise RuntimeError("Call run() first.")
        rows = [{"instrument": inst.kind, "maturity": inst.maturity, "quote (%)": 100 * inst.quote,
                 "implied (%)": 100 * inst.implied_quote(self.curve),
                 "residual (bp)": 1e4 * inst.residual(self.curve),
                 "zero rate (%)": 100 * self.curve.zero_rate(inst.maturity)} for inst in self.instruments]
        return pd.DataFrame(rows)
