# This script gathers the analysis tools shared by the autocallable notebooks (Produits/Athena, Produits/Phoenix) :
# comparison of note variants (price, fair coupon, scenario statistics), redemption profiles, call probabilities and
# illustrative scenarios.
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare_notes(pricer, model, notes):
    """
    One row per note of the dict {name: note} : price at its own coupon, standard error, fair coupon (price at par),
    probability to be called at the first date, to reach maturity, of a capital loss, expected life and expected
    number of coupons. A note can also be given as (note, pricer) to use a specific pricer (e.g. a daily grid).
    """
    rows = {}
    for name, item in notes.items():
        note, note_pricer = item if isinstance(item, tuple) else (item, pricer)
        start = time.time()
        result = note_pricer.price(model, note)
        stats = note.statistics(note_pricer, model)
        rows[name] = {"Coupon p.a. (%)": 100 * note.coupon_rate,
                      "Price": result.price, "Std error": result.stderr,
                      "Fair coupon p.a. (%)": 100 * note.fair_coupon(note_pricer, model),
                      "P(call at 1st date) (%)": 100 * list(stats["call probability by date"].values())[0],
                      "P(reach maturity) (%)": 100 * stats["probability to reach maturity"],
                      "P(capital loss) (%)": 100 * stats["probability of a capital loss"],
                      "Expected loss (%)": stats["expected loss at maturity (% nominal)"],
                      "Expected life (y)": stats["expected life (years)"],
                      "Expected coupons": stats["expected number of coupons"],
                      "Time (s)": time.time() - start}
    return pd.DataFrame(rows).T


def plot_redemption_profiles(notes, performance=np.linspace(0.2, 1.4, 601), ax=None):
    """
    Redemption at maturity of notes that were never called, as a function of the final performance S_T / S_ref,
    when the protection is knocked-in (solid) and when it is not (the nominal, dotted).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, note in notes.items():
        redemption = 100 * np.where(performance < note.protection_barrier, note.redemption_if_knocked(performance), 1.0)
        line, = ax.plot(100 * performance, redemption, linewidth=2, label=name)
        ax.axvline(100 * note.protection_barrier, color=line.get_color(), linestyle=":", linewidth=1)
    ax.set_xlabel("Final performance $S_T / S_{ref}$ (%)")
    ax.set_ylabel("Redemption (% of nominal)")
    ax.set_title("Redemption at maturity if never called (European protection)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    if standalone:
        plt.show()
    return ax


def plot_call_probabilities(pricer, model, notes):
    """
    Probability of being called at each observation date, for each note of the dict {name: note}.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, note in notes.items():
        stats = note.statistics(pricer, model)
        dates = list(stats["call probability by date"].keys())
        probabilities = 100 * np.array(list(stats["call probability by date"].values()))
        ax.plot(dates, probabilities, "o-", label=f"{name} (maturity reached : {100 * stats['probability to reach maturity']:.1f}%)")
    ax.set_xlabel("Observation date (years)")
    ax.set_ylabel("Probability of a call (%)")
    ax.set_title("Autocall probability by observation date")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.show()


def plot_scenarios(pricer, model, note, n_examples=2):
    """
    Simulated paths of the performance S_t / S_ref, grouped by outcome : called early, redeemed at par at maturity,
    capital loss. Barriers and observation dates are drawn.
    """
    S, v = pricer.paths(model, note.maturity)
    M = S.shape[0] - 1
    df = pricer.discount_factors(model, note.maturity, M)
    _, d = note.path_payoff(pricer, S, v, note.maturity / M, df, details=True)
    t = np.linspace(0, note.maturity, M + 1)
    outcomes = {"called early": np.where(d["call_index"] >= 0)[0],
                "at par at maturity": np.where((d["call_index"] < 0) & (d["loss"] <= 1e-12))[0],
                "capital loss": np.where(d["loss"] > 1e-12)[0]}
    colors = {"called early": "tab:green", "at par at maturity": "tab:blue", "capital loss": "tab:red"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, paths in outcomes.items():
        for j, n in enumerate(paths[:n_examples]):
            end = M if d["call_index"][n] < 0 else note.observation_indices(M)[d["call_index"][n]]
            ax.plot(t[:end + 1], 100 * S[:end + 1, n] / note.initial_fixing, color=colors[name], linewidth=1,
                    label=f"{name} ({100 * len(paths) / S.shape[1]:.1f}% of paths)" if j == 0 else None)
    ax.step(np.concatenate([[0], note.observation_dates]), 100 * np.concatenate([[note.autocall_barriers[0]], note.autocall_barriers]),
            where="pre", color="black", linewidth=1.2, label="autocall barrier")
    if hasattr(note, "coupon_barriers"):
        ax.step(np.concatenate([[0], note.observation_dates]), 100 * np.concatenate([[note.coupon_barriers[0]], note.coupon_barriers]),
                where="pre", color="tab:orange", linewidth=1.2, linestyle="--", label="coupon barrier")
    ax.axhline(100 * note.protection_barrier, color="tab:red", linestyle="-.", linewidth=1.2, label="protection barrier")
    for date in note.observation_dates:
        ax.axvline(date, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("t (years)")
    ax.set_ylabel("Performance $S_t / S_{ref}$ (%)")
    ax.set_title(f"{note.code} : simulated scenarios")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)
    plt.show()
