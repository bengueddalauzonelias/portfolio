# This script extracts the figures of the report correlation_modelling_report.tex from the executed notebook
# Models/correlation.ipynb into the figures/ folder.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "..", "Models", "correlation.ipynb")
FIGURES = os.path.join(HERE, "figures")

# (fragment of the cell source, file name)
TARGETS = [
    ("Total return, base 1", "prices_returns.png"),
    ("correlation of daily returns", "rolling_correlation.png"),
    ("Conditional correlation of the GARCH residuals", "conditional_correlation.png"),
    ("Exceedance correlation", "exceedance_correlation.png"),
]


def extract():
    os.makedirs(FIGURES, exist_ok=True)
    with open(NOTEBOOK, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for fragment, name in TARGETS:
        images = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code" and fragment in "".join(c["source"])
                  for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        if not images:
            print(f"/!\\ no image found for {name} : execute Models/correlation.ipynb first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[-1]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
