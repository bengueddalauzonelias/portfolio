# This script produces the figures and the numbers of the report hull_white_report.tex with the implementation
# Fixed_Income_Models/hull_white_1F.py, on the EURIBOR 6M curve of 12/02/2025 and the ILLUSTRATIVE normal
# volatilities of Products/rates_market.py. Figures are written in figures/, numbers are printed.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "Products"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rates_market import eur_curve, illustrative_cap_vols, illustrative_swaption_vols
from rate_products import CapFloor, Swaption
from Fixed_Income_Models import HullWhite, RatePricer

FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5})

curve = eur_curve()
pricer = RatePricer(curve, illustrative_cap_vols(), illustrative_swaption_vols())
A, SIGMA = 0.03, 0.0085
hw = HullWhite(curve, A, SIGMA)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name), dpi=200)
    plt.close(fig)
    print(f"saved {name}")


def alpha(model, t):
    return curve.instantaneous_forward(t) + model.sigma**2 / (2 * model.a**2) * (1 - np.exp(-model.a * t))**2


def theta(model, t, h=1e-4):
    """ theta(t) = f_T(0,t) + a f(0,t) + sigma^2 / (2a) (1 - e^{-2at}), derivative by central differences. """
    f = curve.instantaneous_forward
    df = (f(t + h) - f(np.maximum(t - h, 1e-6))) / (t + h - np.maximum(t - h, 1e-6))
    return df + model.a * f(t) + model.sigma**2 / (2 * model.a) * (1 - np.exp(-2 * model.a * t))


# ----------------------------------------------------------------------
# 1. Curve fit by Monte Carlo
# ----------------------------------------------------------------------
print("\n=== 1. Martingale test (400,000 paths, antithetic) ===")
times = np.array([0.5, 1, 2, 3, 5, 7, 10])
X, D = hw.simulate(times, 400_000, seed=1)
for k, t in enumerate(times):
    err = 1e4 * (D[k].mean() - curve.discount(t))
    se = 1e4 * D[k].std() / np.sqrt(D.shape[1])
    mart = 1e4 * (np.mean(D[k] * hw.bond_price(t, 10.0, X[k])) - curve.discount(10.0))
    print(f"t={t:>4}: P(0,t)={curve.discount(t):.6f}  E[D]={D[k].mean():.6f}  err={err:+.3f}bp  se={se:.3f}bp  "
          f"E[D P(t,10)]-P(0,10)={mart:+.3f}bp")

# ----------------------------------------------------------------------
# 2. Short-rate paths and drift
# ----------------------------------------------------------------------
t = np.linspace(0.05, 10, 200)
X_paths, _ = hw.simulate(t, 20_000, seed=2)
r_paths = X_paths + alpha(hw, t)[:, None]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, 100 * r_paths[:, :25], linewidth=0.6, alpha=0.7)
ax.fill_between(t, 100 * np.quantile(r_paths, 0.025, axis=1), 100 * np.quantile(r_paths, 0.975, axis=1),
                color="grey", alpha=0.2, label="95% band (20,000 paths)")
ax.plot(t, 100 * curve.instantaneous_forward(t), color="black", linewidth=2, label="Forward curve $f(0,t)$")
ax.plot(t, 100 * alpha(hw, t), color="black", linestyle="--", linewidth=1.5, label=r"$\mathbb{E}[r(t)]=\alpha(t)$")
ax.set_xlabel("t (years)"); ax.set_ylabel("Short rate (%)")
ax.set_title(f"Simulated short rates, a = {A:g}, sigma = {1e4 * SIGMA:g}bp")
ax.legend(loc="lower left")
save(fig, "short_rate_paths.png")
p_neg = [np.mean(r_paths[np.argmin(abs(t - s))] < 0) for s in (1, 5, 10)]
print("\n=== 2. P(r(t) < 0) at t = 1, 5, 10 :", [f"{100 * p:.1f}%" for p in p_neg])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(t, 100 * curve.instantaneous_forward(t), color="black", label="$f(0,t)$")
for a in [0.01, 0.03, 0.10]:
    axes[0].plot(t, 100 * alpha(HullWhite(curve, a, SIGMA), t), label=fr"$\alpha(t)$, a = {a:g}")
    axes[1].plot(t, 100 * theta(HullWhite(curve, a, SIGMA), t), label=f"a = {a:g}")
axes[0].set_title(r"Deterministic shift $\alpha(t)$ (sigma = 85bp)"); axes[0].set_ylabel("%")
axes[1].set_title(r"Drift function $\theta(t)$ (sigma = 85bp)"); axes[1].set_ylabel("% per year")
for ax in axes:
    ax.set_xlabel("t (years)"); ax.legend()
save(fig, "alpha_theta.png")

# ----------------------------------------------------------------------
# 3. Volatility structure : caplets and parameters
# ----------------------------------------------------------------------
fixings = np.arange(0.5, 9.6, 0.5)


def caplet_vol(model, t0, strike=None):
    c = CapFloor(t0 + 0.5, 0.0, start=t0, skip_first=False)
    c = c.with_strike(pricer.par_rate(c.swap()) if strike is None else strike)
    return 1e4 * pricer.implied_cap_vol(c, model.price(c))


fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for a in [0.001, 0.03, 0.10, 0.20]:
    axes[0].plot(fixings, [caplet_vol(HullWhite(curve, a, SIGMA), s) for s in fixings], label=f"a = {a:g}")
for s in [0.006, 0.0085, 0.011]:
    axes[1].plot(fixings, [caplet_vol(HullWhite(curve, A, s), f) for f in fixings], label=f"sigma = {1e4 * s:g}bp")
for ax, title in zip(axes, ["Mean reversion (sigma = 85bp)", "Volatility (a = 3%)"]):
    ax.set_xlabel("Caplet fixing (years)"); ax.set_ylabel("ATM caplet normal vol (bp)"); ax.set_title(title)
    ax.legend()
save(fig, "caplet_vols.png")

# Correlation of zero-coupon yields implied by the model : terminal correlation of R(T,T+1) and R(T,T+10)
print("\n=== 3. Terminal correlation of the 1y and 10y rates at T = 5 ===")
x5 = X_paths[np.argmin(abs(t - 5))]
for a in [0.001, 0.03, 0.10]:
    m = HullWhite(curve, a, SIGMA)
    y1 = -np.log(m.bond_price(5.0, 6.0, x5))
    y10 = -np.log(m.bond_price(5.0, 15.0, x5)) / 10
    print(f"a={a:g}: corr = {np.corrcoef(y1, y10)[0, 1]:.6f}, std(1y)/std(10y) = {y1.std() / y10.std():.3f}")

# Smile : Hull-White normal vol across strikes (5y fixing caplet)
f5 = pricer.par_rate(CapFloor(5.5, 0.0, start=5.0, skip_first=False).swap())
strikes = f5 + np.linspace(-0.02, 0.02, 17)
smile = [caplet_vol(hw, 5.0, k) for k in strikes]
print(f"\n=== 4. Caplet 5y x 6m smile : forward {100 * f5:.3f}%, normal vols from {min(smile):.2f} to {max(smile):.2f}bp")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(1e4 * (strikes - f5), smile, marker="o", label="Hull-White (normal vol)")
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_ylim(min(smile) - 5, max(smile) + 5)
ax.set_xlabel("Strike - forward (bp)"); ax.set_ylabel("Implied normal vol (bp)")
ax.set_title("Implied smile of a 5y-fixing 6M caplet: almost flat in normal vol")
ax.legend()
save(fig, "smile.png")

# ----------------------------------------------------------------------
# 4. Calibration to the swaption grid
# ----------------------------------------------------------------------
grid = illustrative_swaption_vols()
expiries, tenors = grid.expiries, grid.tenors


def atm_swaption(e, n):
    s = Swaption(e, n, 0.0)
    return s.with_strike(pricer.par_rate(s.underlying()))


products = [atm_swaption(e, n) for e in expiries for n in tenors]
model = HullWhite(curve, 0.03, 0.008)
res = model.calibrate(pricer, products)
print("\n=== 5. Calibration to the 25 ATM swaptions :", res)
model_vols = np.array([1e4 * pricer.implied_swaption_vol(p, model.price(p)) for p in products]).reshape(len(expiries), len(tenors))
errors = model_vols - 1e4 * grid.vols
print("errors (bp):\n", np.round(errors, 1))

# co-terminal 10y calibration (for a 10y Bermudan)
coterm = [atm_swaption(e, 10 - e) for e in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
model_ct = HullWhite(curve, 0.03, 0.008)
res_ct = model_ct.calibrate(pricer, coterm)
print("=== Co-terminal (maturity 10y) calibration :", res_ct)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for i, e in enumerate(expiries):
    line, = axes[0].plot(tenors, 1e4 * grid.vols[i], marker="o", linestyle="-", label=f"expiry {e:g}y")
    axes[0].plot(tenors, model_vols[i], marker="x", linestyle="--", color=line.get_color())
axes[0].set_xlabel("Tenor (years)"); axes[0].set_ylabel("ATM normal vol (bp)")
axes[0].set_title("Market (o, solid) vs calibrated Hull-White (x, dashed)"); axes[0].legend(fontsize=8)
im = axes[1].imshow(errors, cmap="RdBu_r", vmin=-np.abs(errors).max(), vmax=np.abs(errors).max(), aspect="auto")
axes[1].set_xticks(range(len(tenors)), [f"{n:g}y" for n in tenors]); axes[1].set_yticks(range(len(expiries)), [f"{e:g}y" for e in expiries])
axes[1].set_xlabel("Tenor"); axes[1].set_ylabel("Expiry"); axes[1].set_title("Model - market (bp of normal vol)")
axes[1].grid(False)
for i in range(len(expiries)):
    for j in range(len(tenors)):
        axes[1].text(j, i, f"{errors[i, j]:+.1f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=axes[1])
save(fig, "calibration.png")

# ----------------------------------------------------------------------
# 5. Bermudan swaption
# ----------------------------------------------------------------------
print("\n=== 6. 2y x 8y payer swaption (10y co-terminal), co-terminal calibration ===")
s0 = Swaption(2.0, 8.0, 0.0)
K = pricer.par_rate(s0.underlying())
berm = Swaption(2.0, 8.0, K, exercise="bermudan")
out = model_ct.bermudan_swaption(berm, n_paths=200_000, seed=7)
europeans = {T: model_ct.price(berm.european(T)) for T in berm.exercise_dates()}
print(f"K = {100 * K:.4f}%  Bermudan = {out['price']:,.0f} +/- {out['stderr']:,.0f}")
print("Europeans:", {k: round(v) for k, v in europeans.items()})
print("European 2y closed form vs MC :", round(europeans[2.0]),
      round(model_ct.bermudan_swaption(berm.european(2.0), n_paths=200_000, seed=7)["price"]))
print("Exercise probabilities:", out["exercise probability"])
for a in [0.001, 0.03, 0.10]:
    m = HullWhite(curve, a, 0.008)
    rmse = m.calibrate(pricer, coterm, fixed={"a": a})["RMSE (bp of normal vol)"]
    b = m.bermudan_swaption(berm, n_paths=200_000, seed=7)["price"]
    e = max(m.price(berm.european(T)) for T in berm.exercise_dates())
    print(f"a fixed = {a:g}: sigma = {1e4 * m.sigma:.1f}bp, Bermudan = {b:,.0f}, max European = {e:,.0f}, "
          f"switch option = {b - e:,.0f}, RMSE = {rmse:.2f}bp")
