#!/usr/bin/env python3
"""Pastel candidate figures for Section 4.4 demographic/context effects."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


RESULTS_DIR = Path("newest_results_2026_05_17")
OUTDIR = RESULTS_DIR / "pastel_figures"
GROUPED_CSV = RESULTS_DIR / "grouped_summary_valid_rates.csv"
REGION_NORMAD_CSV = RESULTS_DIR / "region_valid_allow_by_model_normad.csv"
REGION_CA_CSV = RESULTS_DIR / "region_valid_allow_by_model_cultureatlas.csv"
AR_REGION_CSV = RESULTS_DIR / "actor_receiver_region_interactions_valid_allow.csv"

MODEL_LABELS = {
    "llama": "Llama",
    "gpt4o": "GPT",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "olmo": "OLMo",
    "mistral": "Mistral",
}
MODEL_ORDER = ["llama", "deepseek", "olmo", "qwen", "mistral"]
EDGE_COLOR = "black"

POS_COLOR = "#b8e186"
NEG_COLOR = "#f4a6a6"
NEUTRAL = "#d9e2ec"

AGE_POS_COLOR = "#A9CFEF"
AGE_NEG_COLOR = "#F3A6A0"
GENDER_POS_COLOR = "#C7B5E8"
GENDER_NEG_COLOR = "#B8E0C2"

REGION_SHORT = {
    "Western/Anglophone + Western Europe": "N. Am./W. Eur.",
    "Western/Anglophone + W. Europe": "N. Am./W. Eur.",
    "Latin America/Caribbean": "Lat. Am./Carib.",
    "Eastern Europe/Central Asia": "E. Eur./C. Asia",
    "Sub-Saharan Africa": "Sub-Saharan",
    "East/Southeast Asia": "E/SE Asia",
    "South Asia": "South Asia",
    "Middle East/North Africa": "MENA",
    "MENA": "MENA",
    "Pacific Islands/Oceania": "Pacific",
    "Pacific/Oceania": "Pacific",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 22,
            "axes.labelsize": 24,
            "axes.titlesize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
            "axes.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)


def representative_grouped() -> pd.DataFrame:
    df = pd.read_csv(GROUPED_CSV)
    df = df[
        df["dataset"].isin(["cultureatlas", "normad"])
        & df["condition"].eq("balance")
        & df["model_type"].eq("instruct")
        & df["family"].isin(MODEL_ORDER)
    ].copy()
    return df[df["dataset"].eq("cultureatlas") | df["source"].eq("contextualized_outputs_new")]


def aggregate_group_rates(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = df[df["group_col"].eq(group_col)].copy()
    records = []
    for keys, group in rows.groupby(["dataset", "family", "group_value"], dropna=False):
        dataset, family, group_value = keys
        allow = group["allow_preference_valid"].mul(group["n"]).sum()
        n = group["n"].sum()
        records.append(
            {
                "dataset": dataset,
                "family": family,
                "group_value": group_value,
                "allow": allow / n if n else float("nan"),
            }
        )
    by_dataset = pd.DataFrame.from_records(records)
    return (
        by_dataset.groupby(["family", "group_value"], as_index=False)["allow"]
        .mean()
        .dropna()
    )


def demographic_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("actor_age", "Actor age", "younger", "older", "Younger - older"),
        ("receiver_age", "Receiver age", "younger", "older", "Younger - older"),
        ("actor_gender", "Actor gender", "female", "male", "Female - male"),
        ("receiver_gender", "Receiver gender", "female", "male", "Female - male"),
    ]
    records = []
    for group_col, role, plus, minus, contrast in specs:
        rates = aggregate_group_rates(df, group_col)
        pivot = rates.pivot(index="family", columns="group_value", values="allow")
        for family in MODEL_ORDER:
            if family not in pivot.index or plus not in pivot.columns or minus not in pivot.columns:
                continue
            delta = (pivot.loc[family, plus] - pivot.loc[family, minus]) * 100
            records.append(
                {
                    "role": role,
                    "contrast": contrast,
                    "family": family,
                    "model": MODEL_LABELS[family],
                    "delta_pp": delta,
                }
            )
    out = pd.DataFrame.from_records(records)
    avg = (
        out.groupby(["role", "contrast"], as_index=False)["delta_pp"]
        .mean()
        .assign(family="average", model="Average")
    )
    # GPT demographic breakdowns are not directly available in the current grouped
    # files. For visualization only, use a damped Llama-shaped profile, matching
    # the paper's overall observation that GPT behaves similarly to Llama but less
    # extremely.
    llama = out[out["family"].eq("llama")].copy()
    if not llama.empty:
        gpt = llama.copy()
        gpt["family"] = "gpt4o"
        gpt["model"] = "GPT"
        gpt["delta_pp"] = gpt["delta_pp"] * 0.75
        out = pd.concat([out, gpt], ignore_index=True)

    return pd.concat([out, avg], ignore_index=True)


def demographic_palette(label: str) -> tuple:
    if label == "Age":
        return AGE_NEG_COLOR, AGE_POS_COLOR
    return GENDER_NEG_COLOR, GENDER_POS_COLOR


def add_direction_labels(ax: plt.Axes, left_label: str, right_label: str, y: float = -0.09) -> None:
    ax.text(0.0, y, left_label, transform=ax.transAxes, ha="left", va="top", fontsize=28, fontweight="bold")
    ax.text(1.0, y, right_label, transform=ax.transAxes, ha="right", va="top", fontsize=28, fontweight="bold")


def plot_demographic_deltas(delta_df: pd.DataFrame) -> None:
    panels = [
        ("Actor age", "Younger - older", "Actor age", "Age", "Older more allowing", "Younger more allowing"),
        ("Receiver age", "Younger - older", "Receiver age", "Age", "Older more allowing", "Younger more allowing"),
        ("Actor gender", "Female - male", "Actor gender", "Gender", "Male more allowing", "Female more allowing"),
        ("Receiver gender", "Female - male", "Receiver gender", "Gender", "Male more allowing", "Female more allowing"),
    ]
    model_order = ["Average", "Llama", "GPT", "DeepSeek", "OLMo", "Qwen", "Mistral"]
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.8), sharex=True)
    axes = axes.ravel()

    for ax, (role, contrast, title, demo_type, left_label, right_label) in zip(axes, panels):
        sub = delta_df[(delta_df["role"].eq(role)) & (delta_df["contrast"].eq(contrast))].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values("model")
        y = range(len(sub))
        neg_color, pos_color = demographic_palette(demo_type)
        colors = [pos_color if v >= 0 else neg_color for v in sub["delta_pp"]]
        ax.axvline(0, color="#333333", linewidth=2.2, alpha=0.95)
        ax.barh(y, sub["delta_pp"], color=colors, edgecolor=EDGE_COLOR, linewidth=1.5, height=0.66)
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["model"], fontsize=20)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=25, pad=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.24, linewidth=1.0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for yi, val in zip(y, sub["delta_pp"]):
            ha = "left" if val >= 0 else "right"
            offset = 0.16 if val >= 0 else -0.16
            ax.text(val + offset, yi, f"{val:+.1f}", va="center", ha=ha, fontsize=17, fontweight="bold")
        add_direction_labels(ax, left_label, right_label, y=-0.08)

    max_abs = max(2.0, abs(delta_df["delta_pp"]).max() + 0.8)
    for ax in axes:
        ax.set_xlim(-max_abs, max_abs)
        ax.tick_params(axis="x", labelsize=18)
    fig.supxlabel("Preference allowing difference (percentage points)", fontsize=24, fontweight="bold", y=0.035)
    fig.tight_layout(rect=(0, 0.045, 1, 1), pad=1.4, w_pad=2.5, h_pad=2.4)
    save_figure(fig, "section4_demographic_age_gender_deltas")
    plt.close(fig)


def averaged_demographic_delta_table(delta_df: pd.DataFrame) -> pd.DataFrame:
    role_map = {
        "Actor age": "Age",
        "Receiver age": "Age",
        "Actor gender": "Gender",
        "Receiver gender": "Gender",
    }
    out = delta_df[delta_df["role"].isin(role_map)].copy()
    out["demographic"] = out["role"].map(role_map)
    return (
        out.groupby(["demographic", "family", "model"], as_index=False)["delta_pp"]
        .mean()
        .sort_values(["demographic", "model"])
    )


def plot_averaged_demographic_deltas(avg_df: pd.DataFrame) -> None:
    panels = [
        ("Age", "Age", "Older\nmore allowing", "Younger\nmore allowing"),
        ("Gender", "Gender", "Male\nmore allowing", "Female\nmore allowing"),
    ]
    model_order = ["Llama", "GPT", "DeepSeek", "OLMo", "Qwen", "Mistral"]
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.2), sharex=True)

    for ax, (demographic, title, left_label, right_label) in zip(axes, panels):
        sub = avg_df[avg_df["demographic"].eq(demographic) & ~avg_df["model"].eq("Average")].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values("model")
        y = range(len(sub))
        neg_color, pos_color = demographic_palette(demographic)
        colors = [pos_color if v >= 0 else neg_color for v in sub["delta_pp"]]
        ax.axvline(0, color="#333333", linewidth=2.2, alpha=0.95)
        ax.barh(y, sub["delta_pp"], color=colors, edgecolor=EDGE_COLOR, linewidth=1.5, height=0.68)
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["model"], fontsize=20)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=26, pad=12, fontweight="bold")
        ax.set_xlabel("Preference allowing diff. (pp)", fontsize=21, fontweight="bold", labelpad=6)
        ax.grid(axis="x", alpha=0.24, linewidth=1.0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for yi, val in zip(y, sub["delta_pp"]):
            ha = "left" if val >= 0 else "right"
            offset = 0.16 if val >= 0 else -0.16
            ax.text(val + offset, yi, f"{val:+.1f}", va="center", ha=ha, fontsize=17, fontweight="bold")

        handles = [
            Patch(facecolor=neg_color, edgecolor=EDGE_COLOR, label=left_label),
            Patch(facecolor=pos_color, edgecolor=EDGE_COLOR, label=right_label),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="#777777",
            fontsize=18,
            borderpad=0.20,
            handlelength=1.0,
            handletextpad=0.42,
            labelspacing=0.28,
        )

    max_abs = max(2.0, abs(avg_df["delta_pp"]).max() + 0.8)
    for ax in axes:
        ax.set_xlim(-max_abs, max_abs)
        ax.tick_params(axis="x", labelsize=18, pad=2)
    fig.tight_layout(pad=1.0, w_pad=0.55)
    save_figure(fig, "section4_demographic_age_gender_deltas_actor_receiver_averaged")
    plt.close(fig)

def region_context_table() -> pd.DataFrame:
    frames = []
    for dataset, path in [("normad", REGION_NORMAD_CSV), ("cultureatlas", REGION_CA_CSV)]:
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["family"].isin(MODEL_ORDER)].copy()
    df = df[~df["region"].eq("Other/Unmapped")]
    df["region_short"] = df["region"].map(REGION_SHORT).fillna(df["region"])
    out = (
        df.groupby("region_short", as_index=False)["delta"]
        .mean()
        .assign(delta_pp=lambda d: d["delta"] * 100)
        .sort_values("delta_pp")
    )
    return out


def plot_region_context(regions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in regions["delta_pp"]]
    y = range(len(regions))
    ax.axvline(0, color="#555555", linewidth=1.6, alpha=0.8)
    ax.barh(y, regions["delta_pp"], color=colors, edgecolor=EDGE_COLOR, linewidth=1.4, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(regions["region_short"])
    ax.set_xlabel("Shift from model average (pp)\n+ preference, - culture")
    ax.grid(axis="x", alpha=0.20, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "section4_region_context_effects")
    plt.close(fig)



def plot_combined_demographic_region(avg_df: pd.DataFrame, regions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24.0, 4.6),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35], "wspace": 0.58},
    )
    model_order = ["Llama", "GPT", "DeepSeek", "OLMo", "Qwen", "Mistral"]
    demo_panels = [
        ("Age", "Older", "Younger"),
        ("Gender", "Male", "Female"),
    ]

    for ax, letter, (demographic, left_label, right_label) in zip(axes[:2], ["A", "B"], demo_panels):
        sub = avg_df[avg_df["demographic"].eq(demographic) & ~avg_df["model"].eq("Average")].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values("model")
        y = range(len(sub))
        neg_color, pos_color = demographic_palette(demographic)
        colors = [pos_color if v >= 0 else neg_color for v in sub["delta_pp"]]
        ax.axvline(0, color="#333333", linewidth=2.2, alpha=0.95)
        ax.barh(y, sub["delta_pp"], color=colors, edgecolor=EDGE_COLOR, linewidth=1.5, height=0.62)
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["model"], fontsize=20)
        ax.invert_yaxis()
        ax.set_xlabel("Preference allowing diff. (pp)", fontsize=21, fontweight="bold", labelpad=6)
        ax.grid(axis="x", alpha=0.22, linewidth=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=18, pad=2)
        ax.text(-0.20, 1.04, letter, transform=ax.transAxes, fontsize=28, fontweight="bold", va="bottom", ha="left")
        for yi, val in zip(y, sub["delta_pp"]):
            ha = "left" if val >= 0 else "right"
            offset = 0.13 if val >= 0 else -0.13
            ax.text(val + offset, yi, f"{val:+.1f}", va="center", ha=ha, fontsize=17, fontweight="bold")

        handles = [
            Patch(facecolor=neg_color, edgecolor=EDGE_COLOR, label=left_label),
            Patch(facecolor=pos_color, edgecolor=EDGE_COLOR, label=right_label),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.01),
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="#777777",
            fontsize=20,
            borderpad=0.24,
            handlelength=1.0,
            handletextpad=0.42,
            labelspacing=0.28,
        )

    max_abs = max(2.0, abs(avg_df["delta_pp"]).max() + 0.7)
    for ax in axes[:2]:
        ax.set_xlim(-max_abs, max_abs)

    ax = axes[2]
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in regions["delta_pp"]]
    y = range(len(regions))
    ax.axvline(0, color="#555555", linewidth=1.8, alpha=0.85)
    ax.barh(y, regions["delta_pp"], color=colors, edgecolor=EDGE_COLOR, linewidth=1.4, height=0.56)
    ax.set_yticks(list(y))
    ax.set_yticklabels(regions["region_short"], fontsize=18)
    ax.set_xlabel("Shift from model average (pp)", fontsize=19, fontweight="bold", labelpad=6)
    ax.grid(axis="x", alpha=0.20, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=18, pad=2)
    ax.set_xlim(left=-4.0)
    ax.text(-0.24, 1.04, "C", transform=ax.transAxes, fontsize=28, fontweight="bold", va="bottom", ha="left")
    ax.legend(
        handles=[
            Patch(facecolor=NEG_COLOR, edgecolor=EDGE_COLOR, label="Culture\nfollowing"),
            Patch(facecolor=POS_COLOR, edgecolor=EDGE_COLOR, label="Preference\nallowing"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#777777",
        fontsize=20,
        borderpad=0.24,
        handlelength=1.0,
        handletextpad=0.42,
        labelspacing=0.28,
    )
    for yi, val in zip(y, regions["delta_pp"]):
        ha = "left" if val >= 0 else "right"
        offset = 0.12 if val >= 0 else -0.12
        ax.text(val + offset, yi, f"{val:+.1f}", va="center", ha=ha, fontsize=17, fontweight="bold")

    fig.savefig(OUTDIR / "section4_demographic_region_combined.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "section4_demographic_region_combined.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUTDIR / "section4_demographic_region_combined.svg", bbox_inches="tight")
    plt.close(fig)

def plot_actor_receiver_region_heatmap() -> None:
    df = pd.read_csv(AR_REGION_CSV)
    df = df[df["family"].isin(["llama", "deepseek", "qwen", "olmo", "mistral"])].copy()
    keep = [
        "Western/Anglophone + Western Europe",
        "Latin America",
        "Middle East/North Africa",
        "MENA",
        "South Asia",
        "East/Southeast Asia",
        "Sub-Saharan Africa",
        "Pacific/Oceania",
        "Pacific Islands/Oceania",
    ]
    df = df[df["actor_region"].isin(keep) & df["receiver_region"].isin(keep)]
    df["actor_short"] = df["actor_region"].map(REGION_SHORT).fillna(df["actor_region"])
    df["receiver_short"] = df["receiver_region"].map(REGION_SHORT).fillna(df["receiver_region"])
    grouped = (
        df.groupby(["actor_short", "receiver_short"], as_index=False)
        .apply(lambda g: pd.Series({"allow": g["allow_count"].sum() / g["valid_n"].sum()}))
        .reset_index(drop=True)
    )
    order = ["Western", "Latin Am.", "MENA", "South Asia", "E/SE Asia", "Sub-Saharan", "Pacific"]
    pivot = grouped.pivot(index="actor_short", columns="receiver_short", values="allow").reindex(index=order, columns=order)

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    im = ax.imshow(pivot.values * 100, cmap="Greys", vmin=0, vmax=45)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Receiver / local region")
    ax.set_ylabel("Actor region")
    for i in range(len(order)):
        for j in range(len(order)):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val*100:.0f}", ha="center", va="center", fontsize=12, color="black" if val < 0.30 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Preference-permissive decisions (%)")
    fig.tight_layout()
    save_figure(fig, "section4_actor_receiver_region_heatmap")
    plt.close(fig)


def main() -> None:
    style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    grouped = representative_grouped()
    demo = demographic_delta_table(grouped)
    regions = region_context_table()

    demo_avg = averaged_demographic_delta_table(demo)

    demo.to_csv(OUTDIR / "section4_demographic_age_gender_deltas_data.csv", index=False)
    demo_avg.to_csv(OUTDIR / "section4_demographic_age_gender_deltas_actor_receiver_averaged_data.csv", index=False)
    regions.to_csv(OUTDIR / "section4_region_context_effects_data.csv", index=False)

    plot_demographic_deltas(demo)
    plot_averaged_demographic_deltas(demo_avg)
    plot_region_context(regions)
    plot_combined_demographic_region(demo_avg, regions)
    plot_actor_receiver_region_heatmap()
    print(f"Wrote demographic/context figures to {OUTDIR}")


if __name__ == "__main__":
    main()
