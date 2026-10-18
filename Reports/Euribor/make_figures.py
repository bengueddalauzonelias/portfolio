# This script extracts the figures of the report euribor_curve_report.tex from the executed notebook
# Fixed_Income_Models/euribor_curve.ipynb into the figures/ folder.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "..", "Fixed_Income_Models", "euribor_curve.ipynb")
FIGURES = os.path.join(HERE, "figures")

# (fragment of the cell source, file name)
TARGETS = [
    ("EURIBOR 1M, 3M, 6M, 12M (monthly averages) and ESTR", "euribor_fixings.png"),               # Section 3.1
    ("EURIBOR - ESTR basis: fixing minus ESTR compounded", "euribor_estr_basis.png"),             # Section 4.1
    ("Forward EURIBOR 6M rates L(t, t+6M) and forward compounded ESTR", "projection_forwards.png"),  # Section 6.2
    ("Single-curve EURIBOR curve vs dual-curve projection", "single_vs_dual.png"),                # Section 7
    ("EURIBOR 3M / 6M forward curves and tenor basis", "tenor_basis.png"),                        # Section 8
    ("Par-point DV01 of an 18m x 4y EURIBOR 6M payer swap", "key_rates.png"),                     # Section 9
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
            print(f"/!\\ no image found for {name} : execute Fixed_Income_Models/euribor_curve.ipynb first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[-1]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
