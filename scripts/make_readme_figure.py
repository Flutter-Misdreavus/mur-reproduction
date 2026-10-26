"""Render the README figure from reviewed numeric evidence; requires matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    stats = json.loads((root / "experiments/mur-20261024-test100-r2/supervisor-statistics.json").read_text())
    auto = stats["primary_automated_scored_subset"]
    reviewed = stats["secondary_reviewed_same_scored_subset"]
    methods = ["cot", "per_step", "mur"]
    labels = ["CoT", "Per-step search", "MUR"]
    y = np.arange(3)
    costs = [auto["methods"][m]["mean_output_tokens"] for m in methods]
    automatic = [100 * auto["methods"][m]["accuracy"] for m in methods]
    audited = [100 * reviewed["methods"][m]["accuracy"] for m in methods]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.hashsalt": "mur-reviewed-partial50", "axes.titleweight": "bold"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#f5f8fc")
    fig.subplots_adjust(left=.13, right=.96, top=.66, bottom=.22, wspace=.27)
    fig.text(.045, .93, "Lower token use. Accuracy target not met.", size=23, weight="bold", color="#142b45")
    fig.text(.045, .865, "First 50 graded questions  /  one seed  /  fixed100 task incomplete", size=11, color="#566d84")
    for ax in (left, right):
        ax.set_facecolor("#f5f8fc")
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#e0e7ef", linewidth=.8)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="both", length=0, colors="#4b6279", pad=8)
        ax.set_ylim(-.6, 2.6)
        ax.invert_yaxis()
    left.barh(y, costs, height=.48, color=["#9bb1c7", "#4770a1", "#168b7b"], zorder=3)
    left.set_yticks(y, labels, color="#243d56")
    left.set_xlim(0, 6200)
    left.set_xticks([0, 2000, 4000, 6000], ["0", "2,000", "4,000", "6,000"])
    left.set_xlabel("Mean policy + critic output tokens", labelpad=12, color="#4b6279")
    left.set_title("Generation cost ↓", loc="left", pad=20, color="#243d56", size=13)
    for pos, value in zip(y, costs):
        left.text(value + 100, pos, f"{value:,.1f}", va="center", size=10, color="#243d56")
    right.barh(y - .15, automatic, height=.25, color="#9bb1c7", label="Automated", zorder=3)
    right.barh(y + .15, audited, height=.25, color="#168b7b", label="GPT review", zorder=3)
    right.set_yticks(y, labels, color="#243d56")
    right.set_xlim(0, 100)
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_xlabel("Accuracy (%)", labelpad=12, color="#4b6279")
    right.set_title("Answer accuracy ↑", loc="left", pad=20, color="#243d56", size=13)
    for offset, values in [(-.15, automatic), (.15, audited)]:
        for pos, value in zip(y, values):
            right.text(value + 1.8, pos + offset, f"{value:.0f}%", va="center", size=9, color="#243d56")
    right.legend(loc="lower left", bbox_to_anchor=(0, 1.28), ncol=2, frameon=False,
                 fontsize=9, handlelength=1.3, borderaxespad=0)
    fig.text(.045, .048, "API-returned tokens are a compute proxy, not FLOPs. GPT review is not human annotation.",
             size=10, color="#566d84")
    out = root / "assets"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "results.svg", metadata={"Date": None,
                "Title": "Partial50 MUR reproduction: generation cost and answer accuracy",
                "Description": "Cost bars and automatic/GPT-reviewed accuracy for 50 questions; the 100-question task is incomplete."})
    svg = out / "results.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(out / "results.png", dpi=160, metadata={"Description": "Partial50 results; generated from supervisor-statistics.json"})
    plt.close(fig)


if __name__ == "__main__":
    main()
