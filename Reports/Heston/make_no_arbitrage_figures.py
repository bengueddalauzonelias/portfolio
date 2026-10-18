# This script builds the figures of the section "No-arbitrage conditions" of the Heston report
# (heston_rapport.tex, section 8.3) and prints the numbers of its table. It reproduces section 8 of Model/heston_test.ipynb :
# three static no-arbitrage conditions checked on six surfaces, on a grid of 71 log-moneynesses x 61 maturities.
#   figures/no_arbitrage_map.png          : where each condition is violated, for each surface
#   figures/monotonicity_quadrature.png   : Heston prices near zero, default vs fine Lewis quadrature
#   figures/butterfly_densities.png       : risk-neutral densities (Breeden-Litzenberger)
#   figures/calendar_svi.png              : normalised call prices in the right wing as functions of T, SVI vs SSVI
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "Model"), os.path.join(HERE, "..", "..", "Produits")]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from market import cac40_heston
from ImpliedVolatilitySurface import ImpliedVolatilitySurface

FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})

# ------------------------------------------------------------------
# Market, Heston model and Heston implied surface (as in heston_test.ipynb)
# ------------------------------------------------------------------
mkt = cac40_heston(verbose=False)
iv, forward_curve, zc_rate, heston = mkt.iv_surface, mkt.forward_curve, mkt.zc_rate, mkt.heston
heston_otm = {}
for exp in iv.expiries:
    strikes = np.array(iv.strikes[exp])
    heston_otm[exp] = list(heston.price_vanillas(strikes, exp, isCall=strikes > forward_curve.calc_forward(exp)))
heston_iv = ImpliedVolatilitySurface(forward_curve, mkt.bs_model)
heston_iv.set_implied_vol_surface(iv.strikes, heston_otm)

k_grid = np.round(np.arange(-0.40, 0.3001, 0.01), 4)
T_check = np.unique(np.concatenate([iv.expiries, np.linspace(0.05, 4.85, 48)]))
kappa = np.exp(k_grid)
tolerance = 1e-7


def normalised_call(k, w):
    s = np.sqrt(np.maximum(w, 1e-16))
    d1 = -k / s + s / 2
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - s)


def vol_surface_prices(surface, method, k=k_grid, T=T_check):
    out = np.empty((len(T), len(k)))
    for i, t in enumerate(T):
        F = forward_curve.calc_forward(t)
        vols = np.array([float(surface.get_vol(t, F * np.exp(x), method=method)) for x in k])
        out[i] = normalised_call(k, vols**2 * t)
    return out


def heston_prices(k=k_grid, T=T_check, **quadrature):
    out = np.empty((len(T), len(k)))
    for i, t in enumerate(T):
        F, D = forward_curve.calc_forward(t), np.exp(-zc_rate(t) * t)
        out[i] = heston.price_vanillas(F * np.exp(k), t, isCall=True, **quadrature) / (D * F)
    return out


def violations(c):
    slopes = np.diff(c, axis=1) / np.diff(kappa)
    return {"monotonicity": np.maximum(slopes, -1.0 - slopes), "butterfly": -np.diff(slopes, axis=1),
            "calendar": -np.diff(c, axis=0)}


start = time.time()
surfaces = {"Market, linear + V2T": vol_surface_prices(iv, "v2t"),
            "Market, SVI": vol_surface_prices(iv, "svi"),
            "Market, SSVI": vol_surface_prices(iv, "ssvi"),
            "Heston, default quadrature": heston_prices(),
            "Heston, fine quadrature": heston_prices(u_max=3000.0, n_u=60_000),
            "Heston implied, linear + V2T": vol_surface_prices(heston_iv, "v2t")}
checks = {name: violations(c) for name, c in surfaces.items()}
print(f"surfaces computed in {time.time() - start:.1f}s")

rows = {}
for name, check in checks.items():
    rows[name] = {}
    for condition, values in check.items():
        rows[name][f"{condition} violations"] = f"{int((values > tolerance).sum())} / {values.size}"
        rows[name][f"{condition} worst"] = float(max(values.max(), 0.0))
print(pd.DataFrame(rows).T.to_string())

# ------------------------------------------------------------------
# Figure 1 : map of the violations
# ------------------------------------------------------------------
style = {"monotonicity": ("tab:blue", "s"), "butterfly": ("tab:red", "o"), "calendar": ("tab:green", "^")}
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
for ax, (name, check) in zip(axes.flat, checks.items()):
    for condition, values in check.items():
        if condition == "calendar":
            T_points, k_points = (T_check[:-1] + T_check[1:]) / 2, k_grid
        elif condition == "monotonicity":
            T_points, k_points = T_check, (k_grid[:-1] + k_grid[1:]) / 2
        else:
            T_points, k_points = T_check, k_grid[1:-1]
        ii, jj = np.nonzero(values > tolerance)
        color, marker = style[condition]
        ax.scatter(100 * k_points[jj], T_points[ii], s=14, color=color, marker=marker, label=f"{condition} ({len(ii)})")
    for T in iv.expiries:
        ax.axhline(T, color="grey", linewidth=0.3)
    ax.set_xlim(-41, 31); ax.set_ylim(0, 5)
    ax.set_title(name)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
for ax in axes[-1]:
    ax.set_xlabel("log-moneyness $k$ (%)")
for ax in axes[:, 0]:
    ax.set_ylabel("maturity $T$ (years)")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "no_arbitrage_map.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------
# Figure 2 : monotonicity, Heston prices near zero (default vs fine quadrature)
# ------------------------------------------------------------------
k_left = np.round(np.arange(-0.40, -0.0999, 0.005), 4)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, T in zip(axes, [0.0247, 0.1014]):
    for label, kw, ls in [("default quadrature (4,000 nodes, $u \\leq 1000$)", {}, "-"),
                          ("fine quadrature (60,000 nodes, $u \\leq 3000$)", dict(u_max=3000.0, n_u=60_000), "--")]:
        F, D = forward_curve.calc_forward(T), np.exp(-zc_rate(T) * T)
        puts = heston.price_vanillas(F * np.exp(k_left), T, isCall=False, **kw) / (D * F)
        ax.plot(100 * k_left, 1e6 * puts, ls, linewidth=1.5, label=label)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Heston put prices, $T$ = {T:.3f} years")
    ax.set_xlabel("log-moneyness $k$ (%)"); ax.set_ylabel("put price / $D(T)F(T)$ ($\\times 10^{-6}$)")
    ax.set_ylim(-12, 40); ax.grid(True, linestyle="--", alpha=0.5); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "monotonicity_quadrature.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------
# Figure 3 : butterfly, risk-neutral densities
# ------------------------------------------------------------------
shown = ["Market, linear + V2T", "Market, SVI", "Market, SSVI", "Heston, fine quadrature", "Heston implied, linear + V2T"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, T_target in zip(axes, [0.1014, 1.0986, 3.8411]):
    i = int(np.argmin(np.abs(T_check - T_target)))
    for name in shown:
        slopes = np.diff(surfaces[name][i]) / np.diff(kappa)
        density = np.diff(slopes) / ((kappa[2:] - kappa[:-2]) / 2)
        ax.plot(100 * k_grid[1:-1], density, linewidth=1.3, label=name)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"$T$ = {T_check[i]:.2f} years")
    ax.set_xlabel("log-moneyness $k$ (%)"); ax.grid(True, linestyle="--", alpha=0.5)
axes[0].set_ylabel("density of $S_T / F(T)$"); axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "butterfly_densities.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------
# Figure 4 : calendar, normalised call prices of the right wing as functions of T (SVI vs SSVI)
# ------------------------------------------------------------------
T_fine = np.linspace(1.35, 2.1, 151)
k_wing = np.array([0.24, 0.27, 0.30])
svi = vol_surface_prices(iv, "svi", k=k_wing, T=T_fine)
ssvi = vol_surface_prices(iv, "ssvi", k=k_wing, T=T_fine)
fig, ax = plt.subplots(figsize=(8, 4.2))
for j, k in enumerate(k_wing):
    line, = ax.plot(T_fine, 1e3 * svi[:, j], linewidth=1.6, label=f"SVI, $k$ = {100 * k:.0f}%")
    ax.plot(T_fine, 1e3 * ssvi[:, j], "--", color=line.get_color(), linewidth=1.2, label=f"SSVI, $k$ = {100 * k:.0f}%")
for T in iv.expiries:
    if T_fine[0] <= T <= T_fine[-1]:
        ax.axvline(T, color="grey", linestyle=":", linewidth=1)
ax.set_xlabel("maturity $T$ (years) (dotted: listed expiries 1.60 and 1.85)")
ax.set_ylabel("call price / $D(T)F(T)$ ($\\times 10^{-3}$)")
ax.set_title("Calendar condition in the right wing: price must not decrease with $T$")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "calendar_svi.png"), dpi=200)
plt.close(fig)
print("figures written to", FIGURES)
