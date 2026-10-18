import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This script computes the Greeks of any product by finite differences ("bump and revalue").
# It works with any pricer exposing price(model, product, n_steps=None) -> PricingResult (monte_carlo.py),
# and any model exposing :
# - S0, forward_curve (ForwardCurve), forward_price, zc_rate   -> spot and rate bumps
# - vol_shifted(d_sigma)                                        -> volatility bumps (e.g. HestonModel)
# With a Monte Carlo pricer, the seed is fixed, so that every bumped price uses the same random numbers
# (common random numbers) : the Monte Carlo noise mostly cancels in the differences.


def bump_spot(model, dS):
    """
    Copy of the model with the spot shifted by dS. The borrow + dividend yields are kept, so every forward
    F(0,T) moves proportionally to the spot.
    """
    forward_curve = copy.copy(model.forward_curve)
    forward_curve.S0 = model.S0 + dS
    bumped = copy.copy(model)
    bumped.forward_curve = forward_curve
    bumped.forward_price = forward_curve.calc_forward
    bumped.S0 = forward_curve.S0
    return bumped


def bump_rate(model, dr):
    """
    Copy of the model with a parallel shift dr of the zero-coupon curve. The borrow + dividend yields are kept,
    so every forward F(0,T) moves by a factor exp(dr * T).
    """
    zc_rate = model.zc_rate
    bumped_zc = lambda T: zc_rate(T) + dr
    forward_curve = copy.copy(model.forward_curve)
    forward_curve.zc_rate = bumped_zc
    bumped = copy.copy(model)
    bumped.forward_curve = forward_curve
    bumped.forward_price = forward_curve.calc_forward
    bumped.zc_rate = bumped_zc
    return bumped


def bump_carry(model, dq):
    """
    Copy of the model with a parallel shift dq of the borrow + dividend yield q(t). Rates are kept, so every
    forward F(0,T) moves by a factor exp(-dq T) and the discount factors are unchanged.
    """
    forward_curve = copy.copy(model.forward_curve)
    carry = model.forward_curve.borrow_div_func
    forward_curve.borrow_div_func = lambda t: carry(t) + dq
    bumped = copy.copy(model)
    bumped.forward_curve = forward_curve
    bumped.forward_price = forward_curve.calc_forward
    return bumped


def bump_vol(model, d_sigma):
    """
    Copy of the model with its volatility level shifted by d_sigma (delegated to model.vol_shifted).
    """
    if not hasattr(model, 'vol_shifted'):
        raise TypeError("The model must expose vol_shifted(d_sigma) to compute volatility Greeks.")
    return model.vol_shifted(d_sigma)


class Greeks:
    """
    First and second order Greeks by central finite differences.
    Units :
    - Delta : per 1 unit of spot          - Gamma : per 1 unit of spot, squared
    - Vega  : per 1 vol point (+0.01)     - Volga : per 1 vol point, squared
    - Vanna : change of Delta per 1 vol point
    - Rho   : per +1% parallel shift of the zero-coupon curve
    - Theta : value change over 1 calendar day (curves and model kept unchanged)
    - Dividend (optional, not computed by default) : per +1% parallel shift of the borrow + dividend yield
    """
    FIRST_ORDER = ('delta', 'vega', 'rho', 'theta')
    SECOND_ORDER = ('gamma', 'vanna', 'volga')
    OPTIONAL = ('dividend',)
    ORDER = ('delta', 'gamma', 'vega', 'vanna', 'volga', 'rho', 'dividend', 'theta')

    def __init__(self, pricer, spot_bump=0.01, vol_bump=0.01, rate_bump=1e-3, time_bump=1 / 365):
        """
        Init method.
        - pricer    : object with price(model, product, n_steps=None) -> PricingResult
        - spot_bump : relative spot bump (0.01 = 1% of the spot)
        - vol_bump  : absolute volatility bump (0.01 = 1 vol point)
        - rate_bump : absolute rate bump (1e-3 = 10 bp)
        - time_bump : time step of theta in years
        """
        if min(spot_bump, vol_bump, rate_bump, time_bump) <= 0:
            raise ValueError("Bump sizes must be positive.")
        self.pricer = pricer
        self.spot_bump = spot_bump
        self.vol_bump = vol_bump
        self.rate_bump = rate_bump
        self.time_bump = time_bump

    @staticmethod
    def _estimate(samples):
        """ Mean and standard error of a finite difference estimator given path by path. """
        return float(np.mean(samples)), float(np.std(samples, ddof=1) / np.sqrt(len(samples)))

    def compute(self, model, product, greeks=('delta', 'gamma', 'vega', 'vanna', 'volga', 'rho', 'theta')):
        """
        Returns a DataFrame (index = price + requested Greeks) with the value and its Monte Carlo standard error.
        """
        unknown = set(greeks) - set(self.ORDER)
        if unknown:
            raise ValueError(f"Unknown Greeks {unknown}. Available : {self.ORDER}")
        n_steps = self.pricer.n_steps(product.maturity) if hasattr(self.pricer, 'n_steps') else None
        price = lambda m, p=product: self.pricer.price(m, p, n_steps=n_steps).samples
        base = price(model)
        rows = {"price": self._estimate(base)}
        h, d = self.spot_bump * model.S0, self.vol_bump

        if {'delta', 'gamma', 'vanna'} & set(greeks):
            up, down = price(bump_spot(model, h)), price(bump_spot(model, -h))
            if 'delta' in greeks:
                rows["delta"] = self._estimate((up - down) / (2 * h))
            if 'gamma' in greeks:
                rows["gamma"] = self._estimate((up - 2 * base + down) / h**2)
        if {'vega', 'volga'} & set(greeks):
            up, down = price(bump_vol(model, d)), price(bump_vol(model, -d))
            if 'vega' in greeks:
                rows["vega"] = self._estimate((up - down) / (2 * d) * 0.01)
            if 'volga' in greeks:
                rows["volga"] = self._estimate((up - 2 * base + down) / d**2 * 0.01**2)
        if 'vanna' in greeks:
            pp = price(bump_vol(bump_spot(model, h), d))
            pm = price(bump_vol(bump_spot(model, h), -d))
            mp = price(bump_vol(bump_spot(model, -h), d))
            mm = price(bump_vol(bump_spot(model, -h), -d))
            rows["vanna"] = self._estimate((pp - pm - mp + mm) / (4 * h * d) * 0.01)
        if 'rho' in greeks:
            up, down = price(bump_rate(model, self.rate_bump)), price(bump_rate(model, -self.rate_bump))
            rows["rho"] = self._estimate((up - down) / (2 * self.rate_bump) * 0.01)
        if 'dividend' in greeks:
            up, down = price(bump_carry(model, self.rate_bump)), price(bump_carry(model, -self.rate_bump))
            rows["dividend"] = self._estimate((up - down) / (2 * self.rate_bump) * 0.01)
        if 'theta' in greeks:
            later = price(model, product.shifted(self.time_bump))
            rows["theta"] = self._estimate((later - base) * (1 / 365) / self.time_bump)

        order = ["price"] + [g for g in self.ORDER if g in rows]
        return pd.DataFrame({g: dict(zip(("value", "stderr"), rows[g])) for g in order}).T

    def parameter_sensitivities(self, model, product, rel_bump=0.01):
        """
        dPrice / dParameter for every calibrated parameter of the model (model.PARAMS), by central differences
        with a relative bump rel_bump. Values are per unit of parameter.
        """
        if not hasattr(model, 'PARAMS'):
            raise TypeError("The model must expose a PARAMS dict {name: (lower, upper)}.")
        n_steps = self.pricer.n_steps(product.maturity) if hasattr(self.pricer, 'n_steps') else None
        rows = {}
        for name in model.PARAMS:
            value = getattr(model, name)
            h = rel_bump * max(abs(value), 1e-4)
            up, down = copy.copy(model), copy.copy(model)
            setattr(up, name, value + h)
            setattr(down, name, value - h)
            diff = (self.pricer.price(up, product, n_steps).samples - self.pricer.price(down, product, n_steps).samples) / (2 * h)
            rows[f"d/d{name}"] = self._estimate(diff)
        return pd.DataFrame({k: dict(zip(("value", "stderr"), v)) for k, v in rows.items()}).T

    def table(self, model, products, greeks=('delta', 'gamma', 'vega', 'vanna', 'volga', 'rho', 'theta'),
              with_stderr=False, labels=None):
        """
        Greeks of several products : one row per product (index = labels, or the product codes),
        one column per Greek.
        """
        if labels is None:
            labels = [getattr(p, 'code', repr(p)) for p in products]
        if len(set(labels)) != len(labels):
            raise ValueError("Labels must be unique : pass labels=[...] to name the products.")
        rows = {}
        for product, label in zip(products, labels):
            res = self.compute(model, product, greeks)
            rows[label] = res["value"].to_dict()
            if with_stderr:
                rows[label].update({f"{g} stderr": s for g, s in res["stderr"].items()})
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------
    def profiles(self, model, products, spots, greeks=('delta', 'gamma', 'vega', 'vanna', 'volga', 'rho', 'theta'),
                 labels=None):
        """
        Price and Greeks of several products on a uniform grid of spots.
        Every scenario (spot x vol bump x rate bump) is simulated once and all the products are priced on the same
        paths. Delta, Gamma and Vanna are central differences along the spot grid (step = grid step), Vega, Volga,
        Rho and Theta use the bumps of the class. End points of the grid are NaN for spot derivatives.
        Returns a dict {label: DataFrame (index = spots, columns = price + greeks)}.
        """
        spots = np.asarray(spots, dtype=float)
        steps = np.diff(spots)
        if len(spots) < 3 or not np.allclose(steps, steps[0]) or steps[0] <= 0:
            raise ValueError("spots must be an increasing uniform grid of at least 3 points.")
        if labels is None:
            labels = [getattr(p, 'code', repr(p)) for p in products]
        if len(set(labels)) != len(labels):
            raise ValueError("Labels must be unique : pass labels=[...] to name the products.")
        h, d, dr = steps[0], self.vol_bump, self.rate_bump
        n_steps = [self.pricer.n_steps(p.maturity) if hasattr(self.pricer, 'n_steps') else None for p in products]

        def grid(bump=lambda m: m, shift=0.0):
            """ Prices (n_products x n_spots) under a bumped model, maturities shifted by shift. """
            out = np.zeros((len(products), len(spots)))
            for j, s in enumerate(spots):
                m = bump(bump_spot(model, s - model.S0))
                for i, (p, n) in enumerate(zip(products, n_steps)):
                    out[i, j] = self.pricer.price(m, p if shift == 0 else p.shifted(shift), n_steps=n).price
            return out

        def spot_diff(P, order):
            D = np.full_like(P, np.nan)
            if order == 1:
                D[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * h)
            else:
                D[:, 1:-1] = (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) / h**2
            return D

        base = grid()
        results = {"price": base}
        if 'delta' in greeks: results["delta"] = spot_diff(base, 1)
        if 'gamma' in greeks: results["gamma"] = spot_diff(base, 2)
        if {'vega', 'vanna', 'volga'} & set(greeks):
            up, down = grid(lambda m: bump_vol(m, d)), grid(lambda m: bump_vol(m, -d))
            if 'vega' in greeks: results["vega"] = (up - down) / (2 * d) * 0.01
            if 'volga' in greeks: results["volga"] = (up - 2 * base + down) / d**2 * 0.01**2
            if 'vanna' in greeks: results["vanna"] = (spot_diff(up, 1) - spot_diff(down, 1)) / (2 * d) * 0.01
        if 'rho' in greeks:
            up, down = grid(lambda m: bump_rate(m, dr)), grid(lambda m: bump_rate(m, -dr))
            results["rho"] = (up - down) / (2 * dr) * 0.01
        if 'theta' in greeks:
            results["theta"] = (grid(shift=self.time_bump) - base) * (1 / 365) / self.time_bump

        return {label: pd.DataFrame({g: values[i] for g, values in results.items()}, index=pd.Index(spots, name="spot"))
                for i, label in enumerate(labels)}


def plot_greek_profiles(profiles, greeks=('price', 'delta', 'gamma', 'vega', 'vanna', 'volga', 'rho', 'theta'),
                        vlines=None, n_cols=4, title=None):
    """
    Plots the output of Greeks.profiles : one subplot per Greek, one line per product.
    - vlines : optional {label: x} vertical reference lines (e.g. {"Spot": S0, "Barrier": H})
    """
    greeks = [g for g in greeks if g in next(iter(profiles.values())).columns]
    n_rows = int(np.ceil(len(greeks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows), squeeze=False)
    styles = [':', '--', '-.', (0, (1, 3))]
    for ax, g in zip(axes.flat, greeks):
        for label, df in profiles.items():
            ax.plot(df.index, df[g], label=label)
        for k, (name, x) in enumerate((vlines or {}).items()):
            ax.axvline(x, color="grey", linestyle=styles[k % len(styles)], linewidth=1, label=name)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(g.capitalize())
        ax.set_xlabel("Spot")
        ax.grid(True, linestyle="--", alpha=0.5)
    for ax in axes.flat[len(greeks):]:
        ax.axis("off")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 8), bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()
    return fig


def plot_greeks_bars(table, greeks=('price', 'delta', 'gamma', 'vega', 'vanna', 'volga', 'rho', 'theta'), n_cols=4,
                     title=None):
    """
    Bar charts of the output of Greeks.table : one subplot per Greek, one bar per product.
    If the table has '<greek> stderr' columns (with_stderr=True), 95% confidence intervals are drawn.
    """
    greeks = [g for g in greeks if g in table.columns]
    n_rows = int(np.ceil(len(greeks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows), squeeze=False)
    x = np.arange(len(table))
    for ax, g in zip(axes.flat, greeks):
        values = table[g].astype(float)
        errors = 1.96 * table[f"{g} stderr"].astype(float) if f"{g} stderr" in table.columns else None
        ax.bar(x, values, yerr=errors, capsize=3, color=np.where(values >= 0, "tab:blue", "tab:red"), alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(table.index, rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(g.capitalize())
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for ax in axes.flat[len(greeks):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()
    return fig
