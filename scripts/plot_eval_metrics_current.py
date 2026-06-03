#!/usr/bin/env python3
"""Create Section 4 plots from eval_metrics_current CSV summaries."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PASTEL_BLUE = "#9ecae1"
PASTEL_ORANGE = "#fdd0a2"
PASTEL_GREEN = "#b7e4c7"
PASTEL_PINK = "#f4a6b8"
PASTEL_PURPLE = "#cdb4db"
PASTEL_YELLOW = "#fff2a8"
PASTEL_TEAL = "#bde0d6"
PASTEL_CORAL = "#f6b8a8"
PASTEL_LAVENDER = "#d7c4f0"
TEXT = "#334155"


def label_row(row):
    dataset = str(row.get("dataset", "")).replace("cultureatlas", "CultureAtlas").replace("normad", "NormAD")
    family = str(row.get("model_family", "")).title()
    stage = str(row.get("model_stage", "")).replace("unknown", "?")
    prompt = str(row.get("prompt_condition", ""))
    return f"{dataset} {family} {stage} {prompt}"


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.titlecolor": TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfdff",
            "grid.color": "#e2e8f0",
        }
    )


def save(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overall(metrics_dir, outdir):
    df = pd.read_csv(metrics_dir / "overall_metrics.csv")
    df = df.sort_values(["dataset", "model_family", "model_stage", "prompt_condition", "file"])
    df["label"] = df.apply(label_row, axis=1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = range(len(df))
    ax.barh(y, df["culture_rate_valid"], color=PASTEL_BLUE, label="FOLLOW_CULTURE")
    ax.barh(
        y,
        df["allow_rate_valid"],
        left=df["culture_rate_valid"],
        color=PASTEL_ORANGE,
        label="ALLOW_PREFERENCE",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate among valid decisions")
    ax.set_title("Overall Model Tendencies")
    ax.grid(axis="x", linewidth=0.7)
    ax.legend(loc="lower right")
    save(fig, outdir, "overall_model_tendencies")


def plot_by_pref(metrics_dir, outdir):
    df = pd.read_csv(metrics_dir / "by_pref_config.csv")
    df["label"] = df.apply(label_row, axis=1)
    for dataset, sub in df.groupby("dataset"):
        pivot = sub.pivot_table(
            index="label",
            columns="eval_pref_config",
            values="allow_rate_valid",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=["actor_pref_only", "swapped", "both_prefs"])
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(9.5, max(3.5, 0.38 * len(pivot))))
        pivot.plot(kind="barh", ax=ax, color=[PASTEL_BLUE, PASTEL_GREEN, PASTEL_PINK])
        ax.set_xlim(0, 1)
        ax.set_xlabel("ALLOW_PREFERENCE rate")
        ax.set_ylabel("")
        ax.set_title(f"Configuration Effects: {dataset.title()}")
        ax.grid(axis="x", linewidth=0.7)
        ax.legend(title="Preference config", loc="lower right")
        save(fig, outdir, f"configuration_effects_{dataset}")


def plot_by_scenario(metrics_dir, outdir):
    df = pd.read_csv(metrics_dir / "by_scenario_type.csv")
    df["label"] = df.apply(label_row, axis=1)
    for dataset, sub in df.groupby("dataset"):
        pivot = sub.pivot_table(
            index="label",
            columns="scenario_type",
            values="allow_rate_valid",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=["same", "close", "far"])
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(9.5, max(3.5, 0.38 * len(pivot))))
        pivot.plot(kind="barh", ax=ax, color=[PASTEL_LAVENDER, PASTEL_TEAL, PASTEL_CORAL])
        ax.set_xlim(0, 1)
        ax.set_xlabel("ALLOW_PREFERENCE rate")
        ax.set_ylabel("")
        ax.set_title(f"Country Relation Effects: {dataset.title()}")
        ax.grid(axis="x", linewidth=0.7)
        ax.legend(title="Scenario type", loc="lower right")
        save(fig, outdir, f"scenario_type_effects_{dataset}")


def plot_demographic_heatmaps(metrics_dir, outdir):
    specs = [
        ("by_actor_receiver_age.csv", "actor_age", "receiver_age", "Age Dyads", "age_dyads"),
        ("by_actor_receiver_gender.csv", "actor_gender", "receiver_gender", "Gender Dyads", "gender_dyads"),
    ]
    for filename, row_col, col_col, title, slug in specs:
        df = pd.read_csv(metrics_dir / filename)
        df["label"] = df.apply(label_row, axis=1)
        for label, sub in df.groupby("label"):
            pivot = sub.pivot_table(index=row_col, columns=col_col, values="allow_rate_valid", aggfunc="mean")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="PuBuGn")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Receiver")
            ax.set_ylabel("Actor")
            ax.set_title(f"{title}\n{label}")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color=TEXT)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ALLOW rate")
            save(fig, outdir / slug, f"{slug}_{label_row_slug(label)}")


def label_row_slug(text):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")


def plot_country_top(metrics_dir, outdir):
    df = pd.read_csv(metrics_dir / "by_base_country.csv")
    df["label"] = df.apply(label_row, axis=1)
    for label, sub in df.groupby("label"):
        sub = sub.dropna(subset=["base_country"]).copy()
        if len(sub) < 10:
            continue
        top_allow = sub.nlargest(10, "allow_rate_valid")
        top_culture = sub.nlargest(10, "culture_rate_valid")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
        axes[0].barh(top_allow["base_country"], top_allow["allow_rate_valid"], color=PASTEL_ORANGE)
        axes[0].invert_yaxis()
        axes[0].set_title("Highest preference allowance")
        axes[0].set_xlim(0, 1)
        axes[1].barh(top_culture["base_country"], top_culture["culture_rate_valid"], color=PASTEL_BLUE)
        axes[1].invert_yaxis()
        axes[1].set_title("Highest culture following")
        axes[1].set_xlim(0, 1)
        axes[0].grid(axis="x", linewidth=0.7)
        axes[1].grid(axis="x", linewidth=0.7)
        fig.suptitle(f"Country Effects\n{label}", y=1.02)
        save(fig, outdir / "country_top10", f"country_top10_{label_row_slug(label)}")


def plot_paired_flips(metrics_dir, outdir):
    df = pd.read_csv(metrics_dir / "paired_flip_overall.csv")
    df["label"] = (
        df["pair_type"]
        + " | "
        + df["dataset"].astype(str)
        + " "
        + df["model_family"].astype(str)
        + " "
        + df["left_prompt_condition"].astype(str)
        + "->"
        + df["right_prompt_condition"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.45 * len(df))))
    y = range(len(df))
    colors = [PASTEL_GREEN if v >= 0 else PASTEL_PINK for v in df["allow_rate_delta_right_minus_left"]]
    ax.barh(y, df["allow_rate_delta_right_minus_left"], color=colors)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("ALLOW_PREFERENCE delta, right minus left")
    ax.set_title("Base/Instruct and Balance/No-Balance Shifts")
    ax.grid(axis="x", linewidth=0.7)
    save(fig, outdir, "paired_allow_rate_deltas")

    pref = pd.read_csv(metrics_dir / "paired_flip_by_pref_config.csv")
    pref["label"] = pref["pair_type"] + " | " + pref["dataset"].astype(str) + " " + pref["model_family"].astype(str)
    for label, sub in pref.groupby("label"):
        sub = sub.sort_values("eval_pref_config")
        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        colors = [PASTEL_GREEN if v >= 0 else PASTEL_PINK for v in sub["allow_rate_delta_right_minus_left"]]
        ax.bar(sub["eval_pref_config"], sub["allow_rate_delta_right_minus_left"], color=colors)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_ylabel("ALLOW delta")
        ax.set_title(label)
        ax.grid(axis="y", linewidth=0.7)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        save(fig, outdir / "paired_by_pref", f"paired_pref_{label_row_slug(label)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("eval_metrics_current"))
    parser.add_argument("--outdir", type=Path, default=Path("eval_metrics_current_plots"))
    args = parser.parse_args()
    setup()
    plot_overall(args.metrics_dir, args.outdir)
    plot_by_pref(args.metrics_dir, args.outdir)
    plot_by_scenario(args.metrics_dir, args.outdir)
    plot_demographic_heatmaps(args.metrics_dir, args.outdir)
    plot_country_top(args.metrics_dir, args.outdir)
    plot_paired_flips(args.metrics_dir, args.outdir)
    print(f"Wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()
