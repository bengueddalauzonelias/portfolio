# This script extracts the figures of the report estr_curve_report.tex from the executed notebook
# Fixed_Income_Models/estr_curve.ipynb into the figures/ folder.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "..", "Fixed_Income_Models", "estr_curve.ipynb")
FIGURES = os.path.join(HERE, "figures")

# (fragment of the cell source, file name)
TARGETS = [
    ("ESTR and the ECB key rates since October 2019", "estr_fixings.png"),                  # Section 3.1
    ("Euro area AAA government curve on", "ois_quotes.png"),                               # Section 3.2 (proxy source)
    ("AAA 3M rate minus ESTR compounded over the next 3 months", "aaa_ois_basis.png"),     # Section 4.2
    ("OIS maturities (pillars)", "interpolation_comparison.png"),                          # Section 5
    ("Overnight ESTR forward: smooth bootstrap vs ECB meeting steps", "ecb_steps.png"),    # Section 6
    ("Par-point DV01 of an 18m x 4y receiver OIS", "key_rates.png"),                       # Section 7
]


def extract():
    os.makedirs(FIGURES, exist_ok=True)
    if not os.path.exists(NOTEBOOK):
        print(f"/!\\ {NOTEBOOK} not found : write and execute the notebook first")
        return
    with open(NOTEBOOK, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for fragment, name in TARGETS:
        images = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code" and fragment in "".join(c["source"])
                  for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        if not images:
            print(f"/!\\ no image found for {name} : execute Fixed_Income_Models/estr_curve.ipynb first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[-1]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
