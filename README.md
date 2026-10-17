# Quantitative Finance Portfolio

Derivatives pricing, model calibration and yield curve construction in Python, applied to real market data
(CAC 40 options, €STR, EURIBOR, SOFR, US Treasuries).

Each project comes with a research paper that derives the methods, documents the implementation and discusses the
results. Papers and project pages: **[portfolio-eliasbenguedda.fr](https://portfolio-eliasbenguedda.fr)**

## Research papers

| Paper | Topics | Key results |
|---|---|---|
| [Heston calibration](Rapports/Heston) | Affine characteristic function, Fourier pricing, inverse problem, identifiability, no-arbitrage checks | 142 CAC 40 options repriced with an implied vol RMSE of 0.42 vol pt, 0.12 with time-dependent parameters; out-of-sample and noise tests |
| [Local stochastic volatility](Rapports/LSV/lsv_report.pdf) | SSVI fit, Dupire local vol, leverage function by the particle method (Guyon–Henry-Labordère) | Calibration in 13 s with 50,000 particles, IV RMSE 0.25; model risk on barriers and autocalls at identical vanilla prices |
| [Autocallable notes](Rapports/Autocalls/autocall_report.pdf) | Athena and Phoenix notes under calibrated Heston, Monte Carlo with variance reduction, Brownian bridge | 5 closed-form limit cases recovered; fair coupons, scenario statistics and Greeks of 35 variants |
| [Yield curve construction](Rapports/YieldCurve/yield_curve_report.pdf) | 7 interpolation schemes, sequential and global bootstrap, PCA, key-rate risk | PCA of 181 Treasury curves: level / slope / curvature = 79.6% / 13.2% / 2.7% of variance |
| [€STR curve](Rapports/€str/estr_curve_report.pdf) | OIS conventions, bootstrap on ECB meeting dates, implied policy path | 25 quotes repriced to machine precision, QuantLib match within 2e-8 bp |
| [EURIBOR 6M curve](Rapports/Euribor/euribor_curve_report.pdf) | Dual-curve framework, tenor basis, single vs dual curve | *In progress* |
| [Digital options](Rapports/Digital) | Closed forms, one-touch and Bermudan digitals, Greeks, call-spread hedging | *In progress* |

## Repository structure

```
Model/                  Equity models
  Black_Scholes.py        Black–Scholes pricing and vega
  Heston.py               Heston characteristic function and Fourier pricing
  optimizer.py            Calibration routines
  local_vol.py            Dupire local volatility and LSV model (particle method)
  ImpliedVolatilitySurface.py, forward_curve.py
  monte_carlo.py          Model-agnostic Monte Carlo pricer
  greeks.py, error.py, calibration_validation.py
  heston_test.ipynb, lsv_test.ipynb

Fixed_Income_Models/    Rates models (importable package)
  instruments.py          Deposits, FRAs, swaps, par bonds
  interpolation.py        Linear, cubic, PCHIP, monotone convex, Nelson–Siegel, Svensson
  yield_curve.py, bootstrap.py
  hull_white_1F.py        Hull–White one-factor model
  rate_pricing.py         Bachelier, Black and shifted Black formulas
  yield_curve.ipynb, estr_curve.ipynb, euribor_curve.ipynb, hull_white_test.ipynb

Produits/               Product pricers, one notebook per product
  products.py             Equity product definitions
  rate_products.py        Swap, cap/floor, swaption
  market.py, rates_market.py   Shared market objects (CAC 40 + Heston, EUR curve)
  Vanille/  Barriere/  Digitale/  Athena/  Phoenix/
  Swap/  CapFloor/  Swaption/
  callable_range_accrual/  Callable SOFR range accrual under Hull–White:
                           static replication, Monte Carlo + Longstaff–Schwartz,
                           exact backward induction

Data/                   Market data
  CAC40_MarketOptions_12022025.csv, EURIBOR6M_ZCRates_12022025.csv
  UST_DailyParYields.csv   Daily US Treasury par yields (PCA of the yield curve paper)
  EUR/   €STR, ECB key rates and meetings, AAA curve, EURIBOR (see Data/EUR/README.md)
  USD/   SOFR, Fed funds, Treasury par yields, FOMC meetings (see Data/USD/README.md)

Rapports/               LaTeX sources, figures and PDFs of the papers
```

## Getting started

```bash
git clone https://github.com/bengueddalauzonelias/portfolio.git
cd portfolio
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib jupyter
pip install QuantLib   # optional, only used to cross-check the curves
```

Then open any notebook, for example `Model/heston_test.ipynb` or `Produits/Phoenix/phoenix_pricer.ipynb`.

The EUR and USD datasets come from public sources (ECB, New York Fed, FRED) and can be refreshed with:

```bash
python Data/EUR/download_data.py
python Produits/callable_range_accrual/download_data.py
```

## Author

Elias Benguedda Lauzon — engineering student in applied mathematics (Mines Saint-Étienne)  
[portfolio-eliasbenguedda.fr](https://portfolio-eliasbenguedda.fr) ·
[LinkedIn](https://linkedin.com/in/elias-benguedda)
