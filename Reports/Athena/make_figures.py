# This script extracts the figures of the report athena.tex from the executed notebook
# Products/Athena/athena_pricer.ipynb into the figures/ folder.
# The payoffs, the call probabilities and the fair coupon curves are drawn in LaTeX
# (pgfplots / TikZ) from the numbers of the report : only the simulated scenarios and the Greeks are images.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = os.path.join(HERE, "..", "..", "Products")
FIGURES = os.path.join(HERE, "figures")

# (notebook, fragment of the cell source, index of the image in the cell outputs, file name)
TARGETS = [
    ("Athena/athena_pricer.ipynb", "plot_scenarios(pricer, heston, base)", -1, "athena_scenarios.png"),
    ("Athena/athena_pricer.ipynb", "plot_greeks_bars(", -1, "athena_greeks.png"),
    ("Athena/athena_pricer.ipynb", "plot_greek_profiles(", -1, "athena_profiles.png"),
]


def extract():
    os.makedirs(FIGURES, exist_ok=True)
    for notebook, fragment, position, name in TARGETS:
        with open(os.path.join(PRODUCTS, notebook), encoding="utf-8") as f:
            cells = json.load(f)["cells"]
        images = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code" and fragment in "".join(c["source"])
                  for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        if not images:
            print(f"/!\\ no image found for {name} : execute {notebook} first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[position]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
