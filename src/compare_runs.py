"""Stage 3 summary across runs: one table, the real-fraction plot, and the section 9 difference rule.

Reads only experiments.csv and the saved test predictions CSVs, never the test images.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
UPPER_BOUND_RUN = "real_lrb1e-4"
CEILING = 0.97  # above this, differences come down to a handful of images, so report error counts


def arm(run_id):
    """Which setting a run belongs to: synth_wide, synth_narrow, real_5, mixed_wide_5, ..."""
    if run_id == UPPER_BOUND_RUN:
        return "real_full"
    return run_id.rsplit("_s", 1)[0]


def summary_table(runs):
    rows = []
    upper = runs.loc[runs["run_id"] == UPPER_BOUND_RUN, "test_acc"].item()
    for _, run in runs.iterrows():
        preds = pd.read_csv(RESULTS / f"{run['run_id']}_test_predictions.csv")
        correct = preds["true"] == preds["pred"]
        row = {"run_id": run["run_id"], "arm": arm(run["run_id"]), "real_fraction": run["real_fraction"],
               "seed": run["seed"], "test_acc": run["test_acc"], "errors": int((~correct).sum()),
               "gap": round(upper - run["test_acc"], 4)}
        row.update({f"recall_{g}": round(v, 4) for g, v in correct.groupby(preds["true"]).mean().items()})
        row.update({f"acc_{c}": round(v, 4) for c, v in correct.groupby(preds["source_class"]).mean().items()})
        rows.append(row)
    return pd.DataFrame(rows)


def seed_range(table, arm_name):
    acc = table.loc[table["arm"] == arm_name, "test_acc"]
    return acc.max() - acc.min()


def compare(table, arm_a, arm_b, noise):
    """Section 9 rule: a real difference beats 2x the noise range and 2 points."""
    a, b = (table[table["arm"] == name] for name in (arm_a, arm_b))
    diff = a["test_acc"].mean() - b["test_acc"].mean()
    verdict = "real difference" if abs(diff) > max(2 * noise, 0.02) else "no clear difference"
    print(f"{arm_a} mean {a['test_acc'].mean():.4f} vs {arm_b} mean {b['test_acc'].mean():.4f}: "
          f"diff {diff * 100:+.1f} pts, bar {max(2 * noise, 0.02) * 100:.1f} pts -> {verdict}")
    if min(a["test_acc"].mean(), b["test_acc"].mean()) > CEILING:
        print(f"  near ceiling, errors out of 1,350: {arm_a} {a['errors'].tolist()}, {arm_b} {b['errors'].tolist()}")


def plot_real_fraction(table, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for source, label in (("mixed_wide", "synthetic wide + real"), ("real", "real only")):
        stats = [table.loc[table["arm"] == f"{source}_{pct}", "test_acc"].agg(["mean", "min", "max"])
                 for pct in (5, 10, 25)]
        means = [s["mean"] for s in stats]
        # Error bars show the seed range (only the 5% points have 3 seeds)
        err = [[s["mean"] - s["min"] for s in stats], [s["max"] - s["mean"] for s in stats]]
        ax.errorbar([5, 10, 25], means, yerr=err, marker="o", capsize=4, label=label)
    synth = table.loc[table["arm"] == "synth_wide", "test_acc"]
    ax.errorbar([0], [synth.mean()], yerr=[[synth.mean() - synth.min()], [synth.max() - synth.mean()]],
                marker="s", color="grey", capsize=4, linestyle="none", label="synthetic wide only (3 seeds)")
    upper = table.loc[table["arm"] == "real_full", "test_acc"].item()
    ax.axhline(upper, color="black", linestyle="--", label="all real train (upper bound)")
    ax.set_xlabel("real train used (%)")
    ax.set_ylabel("real test accuracy")
    ax.set_xticks([0, 5, 10, 25])
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    runs = pd.read_csv(RESULTS / "experiments.csv")
    runs = runs[runs["run_id"].str.match(r"(synth|real|mixed)_")]
    table = summary_table(runs)
    table.to_csv(RESULTS / "stage3_summary.csv", index=False)
    print(table[["run_id", "test_acc", "errors", "gap"]].to_string(index=False))

    print("\nStep 3.2, narrow vs wide (noise = larger seed range):")
    compare(table, "synth_narrow", "synth_wide", max(seed_range(table, "synth_narrow"), seed_range(table, "synth_wide")))

    # Amended section 9 rule: the larger 5% seed range is the noise estimate at every fraction
    noise = max(seed_range(table, "mixed_wide_5"), seed_range(table, "real_5"))
    print(f"\nStep 3.3, mixed vs real-only (noise = larger 5% seed range, {noise * 100:.1f} pts):")
    for pct in (5, 10, 25):
        compare(table, f"mixed_wide_{pct}", f"real_{pct}", noise)

    plot_real_fraction(table, RESULTS / "real_fraction.png")
    print(f"\nSaved {RESULTS / 'stage3_summary.csv'} and {RESULTS / 'real_fraction.png'}")


if __name__ == "__main__":
    main()
