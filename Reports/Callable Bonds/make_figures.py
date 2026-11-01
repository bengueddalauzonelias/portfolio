# This script extracts the figures of the report callable_bond_paper.tex from the executed notebook
# Products/callable_bond/callable_bond_pricer.ipynb into the figures/ folder.
# Re-run the notebook, then this script, to update the figures.
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "..", "Products", "callable_bond", "callable_bond_pricer.ipynb")
FIGURES = os.path.join(HERE, "figures")

# (fragment of the cell source, file name)
TARGETS = [
    ("Convergence of the PDE and the tree", "convergence.png"),
    ("Yield to each redemption date", "yields.png"),
    ('set_title("Key-rate durations")', "key_rate_durations.png"),
    ("Price vs initial short rate", "price_profile.png"),
    ("The issuer calls when the short rate is below", "boundary.png"),
    ("Probability of a call on each date", "call_probability.png"),
    ("Issuer option vs volatility", "vol_credit.png"),
]


def extract():
    os.makedirs(FIGURES, exist_ok=True)
    with open(NOTEBOOK, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for fragment, name in TARGETS:
        images = [o["data"]["image/png"] for c in cells if c["cell_type"] == "code" and fragment in "".join(c["source"])
                  for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
        if not images:
            print(f"/!\\ no image found for {name} : execute Products/callable_bond/callable_bond_pricer.ipynb first")
            continue
        with open(os.path.join(FIGURES, name), "wb") as out:
            out.write(base64.b64decode(images[-1]))
        print(f"extracted {name}")


if __name__ == "__main__":
    extract()
