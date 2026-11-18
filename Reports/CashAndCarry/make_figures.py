# This script extracts the figures of the report cash_and_carry.tex from the executed notebook
# strategy/cash_and_carry.ipynb into the figures/ folder.
# Every figure of the report is an image of the notebook.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = os.path.join(HERE, "..", "..")
FIGURES = os.path.join(HERE, "figures")
NOTEBOOK = "strategy/cash_and_carry.ipynb"

# (fragment of the cell source, index of the image in the cell outputs, file name)
TARGETS = [
    ("ax.axhline(10.95", -1, "funding.png"),
    ("ax.hist(basis[s]", -1, "basis.png"),
    ("axes[1].set_title(\"Drawdown (%)\")", -1, "benchmark.png"),
    ("sweep = pd.DataFrame({e:", -1, "threshold_sweep.png"),
    ("allocation = w.resample", 0, "dynamic_equity.png"),
    ("allocation = w.resample", 1, "allocation.png"),
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
