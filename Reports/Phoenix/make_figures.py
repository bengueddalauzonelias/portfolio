# This script extracts the figures of the report phoenix.tex from the executed notebook
# Products/Phoenix/phoenix_synthetic_pricing.ipynb into the figures/ folder.
# The redemption profiles of Section 2 are drawn in LaTeX (pgfplots / TikZ) ; every other figure is an image.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = os.path.join(HERE, "..", "..", "Products")
FIGURES = os.path.join(HERE, "figures")
NOTEBOOK = "Phoenix/phoenix_synthetic_pricing.ipynb"

# (fragment of the cell source, index of the image in the cell outputs, file name)
TARGETS = [
    ("ax.bar(calls.index", -1, "proba_call.png"),
    ("plot_scenarios(pricer, heston, base)", -1, "simulated_paths.png"),
    ("protection_sweep = sweep(", -1, "protection_move.png"),
    ("autocall_sweep = sweep(", -1, "autocall_move.png"),
    ("coupon_sweep = sweep(", -1, "coupon_move.png"),
    ("vol_sweep = market_sweep(", -1, "vol_move.png"),
    ("dividend_sweep = market_sweep(", -1, "dividend_move.png"),
    ("rate_sweep = market_sweep(", -1, "rate_move.png"),
    ("spread_sweep = market_sweep(", -1, "spread_move.png"),
    ("maturity_sweep = market_sweep(", -1, "maturity_move.png"),
    ("skew_sweep = market_sweep(", -1, "skew_move.png"),
    ("base_profiles = Greeks(", -1, "hedging_profiles.png"),
    ('fair = summary["fair coupon (% p.a.)"]', -1, "fair_coupons.png"),
    ("plot_redemption_profiles({", -1, "redemption_profiles.png"),
    ("plot_call_probabilities(pricer, heston, {", -1, "call_probabilities_features.png"),
    ("plot_greeks_bars(greeks", -1, "phoenix_greeks.png"),
]


def extract():
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(PRODUCTS, NOTEBOOK), encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for fragment, position, name in TARGETS:
        images = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code" and fragment in "".join(c["source"])
                  for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        if not images:
            print(f"/!\\ no image found for {name} : execute {NOTEBOOK} first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[position]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
