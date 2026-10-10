# This script extracts the figures of the report yield_curve_report.tex from the executed notebook
# Fixed_Income_Models/yield_curve.ipynb into the figures/ folder.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "..", "Fixed_Income_Models", "yield_curve.ipynb")
FIGURES = os.path.join(HERE, "figures")

# (fragment of the cell source, file name)
TARGETS = [
    ('ax.set_title("EURIBOR 6M zero-coupon curve', "euribor_data.png"),
    ('axes[0].set_title("Zero curves")', "zero_curves.png"),
    ('label="Instantaneous forward")', "forward_curves.png"),
    ("Forward curve response to a +1bp bump", "bump_locality.png"),
    ('label="Bootstrapped (9 quotes)")', "bootstrap_roundtrip.png"),
    ('"Quoted yields and bootstrapped zero curve"', "treasury_curves.png"),
    ("Principal components of daily zero-rate changes", "pca.png"),
]


def extract():
    os.makedirs(FIGURES, exist_ok=True)
    with open(NOTEBOOK, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for fragment, name in TARGETS:
        images = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code" and fragment in "".join(c["source"])
                  for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        if not images:
            print(f"/!\\ no image found for {name} : execute Fixed_Income_Models/yield_curve.ipynb first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[-1]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
