# This script extracts the figures of the report lsv_report.tex from the executed notebook Model/lsv_test.ipynb
# into the figures/ folder. The model risk curves are drawn in LaTeX (pgfplots) from the numbers of the report.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "..", "Model", "lsv_test.ipynb")
FIGURES = os.path.join(HERE, "figures")

# (fragment of the cell source, file name)
TARGETS = [
    ("local_ssvi.grid(times, ys)", "local_vol.png"),
    ("Leverage surface, mixing 1", "leverage.png"),
    ("sqrt E[L^2 v | y]", "gyongy.png"),
    ("for ax, k in zip(axes, [0, 3, 6, 12])", "smiles.png"),
]


def extract():
    os.makedirs(FIGURES, exist_ok=True)
    with open(NOTEBOOK, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for fragment, name in TARGETS:
        images = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code" and fragment in "".join(c["source"])
                  for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        if not images:
            print(f"/!\\ no image found for {name} : execute Model/lsv_test.ipynb first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[-1]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
