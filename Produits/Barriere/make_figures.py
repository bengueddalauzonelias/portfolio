# This script builds the figures of the report barrier_pricing.tex into the figures/ folder :
# - paths.png        : illustrative paths, knocked or not, for a down and an up barrier ;
# - the price / Greeks figures, extracted from the outputs of the executed notebook pricer.ipynb.
import os
import json
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

S0, K, T = 8042.19, 8000.0, 1.0
BARRIERS = {"D": 7200.0, "U": 8800.0}


def plot_paths(n_paths=12, n_steps=252, sigma=0.18, seed=3):
    """
    Illustrative lognormal paths (constant volatility, zero drift), coloured by knock status.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    increments = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_steps, n_paths))
    paths = S0 * np.exp(np.vstack([np.zeros(n_paths), np.cumsum(increments, axis=0)]))
    t = np.linspace(0, T, n_steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, (d, H) in zip(axes, BARRIERS.items()):
        hit = (paths.min(axis=0) <= H) if d == "D" else (paths.max(axis=0) >= H)
        for j in range(n_paths):
            ax.plot(t, paths[:, j], color="tab:red" if hit[j] else "tab:blue", linewidth=0.9, alpha=0.85)
        ax.axhline(H, color="black", linestyle="-.", linewidth=1.2, label=f"barrier H = {H:g}")
        ax.axhline(K, color="black", linestyle=":", linewidth=1, label=f"strike K = {K:g}")
        ax.plot([], [], color="tab:red", label=f"touched ({hit.sum()} paths)")
        ax.plot([], [], color="tab:blue", label=f"never touched ({(~hit).sum()} paths)")
        ax.set_title(f"{'Down' if d == 'D' else 'Up'} barrier")
        ax.set_xlabel("t (years)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("$S_t$")
    fig.suptitle(f"Illustrative paths ({sigma:.0%} volatility, daily steps)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "paths.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def extract_notebook_figures(notebook=os.path.join(HERE, "pricer.ipynb")):
    """
    Saves the PNG outputs of the executed notebook, identified by a fragment of their cell source.
    """
    targets = {"prices.plot.bar": "exercise_prices.png",
               "n_list = [": "convergence.png",
               "plot_greeks_bars(": "greeks_bars.png",
               "Knock-out options : price and Greeks": "profiles_knock_out.png",
               "Knock-in options : price and Greeks": "profiles_knock_in.png"}
    with open(notebook, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for cell in cells:
        source = "".join(cell.get("source", []))
        for fragment, name in targets.items():
            if cell["cell_type"] == "code" and fragment in source:
                for output in cell.get("outputs", []):
                    png = output.get("data", {}).get("image/png")
                    if png:
                        with open(os.path.join(FIGURES, name), "wb") as out:
                            out.write(base64.b64decode(png))
                        print(f"extracted {name}")
                        break


if __name__ == "__main__":
    plot_paths()
    extract_notebook_figures()
    print(f"figures written to {FIGURES}")
