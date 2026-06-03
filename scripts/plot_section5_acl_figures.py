#!/usr/bin/env python3
"""ACL-friendly figures for Section 5.2 and 5.3.

All plotted rates use preference-allowing orientation:
  preference_allowing_rate = 1 - culture_following_rate
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("human_prolific/section5_outputs")
OUTDIR = BASE / "section5_acl_figures"

MODEL_LABELS = {
    "llama31_8b": "Llama",
    "qwen3_4b": "Qwen",
    "mistral7b_instruct_v03": "Mistral",
    "deepseek7b_chat": "DeepSeek",
    "olmo2_7b": "OLMo",
    "gpt54_mini": "GPT",
}
MODEL_ORDER = ["GPT", "Qwen", "Llama", "Mistral", "DeepSeek", "OLMo"]
MODEL_COLORS = {
    "Qwen": "#cab2d6",
    "Llama": "#9ecae1",
    "Mistral": "#ffffb3",
    "DeepSeek": "#fdbf6f",
    "OLMo": "#fb9a99",
    "GPT": "#b2df8a",
}
MODEL_PERSONAL_COLOR = "#c7c9ff"
MODEL_NORM_COLOR = "#ffd6a5"

PASTEL_BLUE = "#b9d7f0"
PASTEL_PINK = "#f4b6c2"
PASTEL_GREEN = "#b8e186"
PASTEL_RED = "#f4a6a6"
PASTEL_GREY = "#d9e2ec"
EDGE = "black"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)


def legend_box(legend) -> None:
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.90)


def modelize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model_tag"].map(MODEL_LABELS)
    return out.dropna(subset=["model"])


def plot_5_2_human_rates_agreement() -> None:
    df = pd.read_csv(BASE / "human_response_summary.csv")

    countries = df[df["grouping"].eq("respondent_country")].copy()
    countries["country"] = countries["respondent_country"].replace(
        {"United States of America": "US", "United Kingdom": "UK", "South Africa": "S. Africa"}
    )
    countries["personal_allow"] = (1 - countries["personal_culture_rate"]) * 100
    countries["norm_allow"] = (1 - countries["norm_culture_rate"]) * 100
    countries["gap_pp"] = countries["personal_allow"] - countries["norm_allow"]
    countries = countries.sort_values("gap_pp", ascending=False)

    rel = df[df["grouping"].eq("country_relation")].copy()
    rel["relation"] = pd.Categorical(rel["country_relation"], ["same", "close", "far"], ordered=True)
    rel = rel.sort_values("relation")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.85), gridspec_kw={"width_ratios": [1.22, 1]})

    ax = axes[0]
    y = np.arange(len(countries))
    ax.hlines(y, countries["norm_allow"], countries["personal_allow"], color="#777777", linewidth=1.6, alpha=0.65)
    ax.scatter(countries["personal_allow"], y, s=58, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.9, label="Personal choice", zorder=3)
    ax.scatter(countries["norm_allow"], y, s=58, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.9, label="Norm judgment", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(countries["country"])
    ax.invert_yaxis()
    ax.set_xlabel("Preference-allowing rate (%)", fontsize=13, fontweight="bold")
    ax.set_xlim(10, 45)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.text(-0.10, 1.03, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
    leg = ax.legend(frameon=True, loc="center right", borderpad=0.25, handletextpad=0.3, fontsize=13)
    legend_box(leg)

    ax = axes[1]
    x = np.arange(len(rel))
    width = 0.34
    ax.bar(x - width / 2, rel["personal_majority_agreement"] * 100, width, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.9)
    ax.bar(x + width / 2, rel["norm_majority_agreement"] * 100, width, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Same", "Close", "Far"])
    ax.set_ylabel("Majority agreement (%)", fontsize=13)
    ax.set_ylim(50, 85)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.text(-0.12, 1.03, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=14)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")

    fig.tight_layout(w_pad=1.2)
    save(fig, "section5_2_human_rates_agreement")
    plt.close(fig)


def plot_5_2_human_regression() -> None:
    reg = pd.read_csv(BASE / "human_regression_drop_one_deviance.csv")
    reg = reg[reg["model"].isin(["personal_choice_only", "norm_judgment_only"])].copy()
    keep = ["scenario_country", "participant_country", "country_relation", "participant_x_receiver_demographics"]
    labels = {
        "scenario_country": "Scenario country",
        "participant_country": "Participant country",
        "country_relation": "Country relation",
        "participant_x_receiver_demographics": "Demo. interaction",
    }
    reg = reg[reg["dropped_factor"].isin(keep)].copy()
    reg["factor"] = pd.Categorical(reg["dropped_factor"], keep, ordered=True)
    reg = reg.sort_values(["model", "factor"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95), sharex=True)
    panels = [("personal_choice_only", "Personal choice", PASTEL_BLUE), ("norm_judgment_only", "Norm judgment", PASTEL_PINK)]
    for ax, (model, title, color) in zip(axes, panels):
        sub = reg[reg["model"].eq(model)].copy().sort_values("factor")
        y = np.arange(len(sub))
        ax.barh(y, sub["drop_one_deviance"], color=color, edgecolor=EDGE, linewidth=0.9, height=0.58)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[v] for v in sub["dropped_factor"]])
        ax.invert_yaxis()
        ax.set_xlabel("Drop-one deviance")
        ax.set_title(title, fontsize=13, pad=3)
        ax.grid(axis="x", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].text(-0.16, 1.04, "A", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    axes[1].text(-0.16, 1.04, "B", transform=axes[1].transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    save(fig, "section5_2_human_regression")
    plt.close(fig)


def no_persona_pref_alignment() -> pd.DataFrame:
    df = modelize(pd.read_csv(BASE / "alignment_no_persona_summary.csv"))
    df["human_pref_rate"] = 1 - df["human_culture_rate"]
    df["model_pref_rate"] = 1 - df["model_culture_rate"]
    df["signed_pref_gap_pp"] = (df["model_pref_rate"] - df["human_pref_rate"]) * 100
    df["distributional_mae"] = (df["model_pref_rate"] - df["human_pref_rate"]).abs()
    df["response_label"] = df["response_type"].map({"personal": "Personal", "norm": "Norm"})
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values("model")


def plot_5_3_alignment_majority_mae() -> None:
    df = no_persona_pref_alignment()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.95), sharey=True)
    metrics = [
        ("majority_choice_alignment", "Majority alignment", lambda s: s * 100),
        ("distributional_mae", "Preference-rate alignment MAE", lambda s: s),
    ]
    markers = {"Personal": "o", "Norm": "s"}
    colors_by_response = {"Personal": MODEL_PERSONAL_COLOR, "Norm": MODEL_NORM_COLOR}
    offsets = {"Personal": -0.14, "Norm": 0.14}

    for ax, (metric, xlabel, scale) in zip(axes, metrics):
        for resp in ["Personal", "Norm"]:
            sub = df[df["response_label"].eq(resp)].copy()
            y = np.arange(len(sub)) + offsets[resp]
            ax.scatter(
                scale(sub[metric]),
                y,
                s=70,
                marker=markers[resp],
                color=colors_by_response[resp],
                edgecolor=EDGE,
                linewidth=0.9,
                label=resp,
                zorder=3,
            )
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="x", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_xlim(40, 88)
    axes[1].set_xlim(0.0, 0.40)
    axes[0].invert_yaxis()
    axes[0].text(-0.13, 1.03, "A", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    axes[1].text(-0.13, 1.03, "B", transform=axes[1].transAxes, fontsize=12, fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_PERSONAL_COLOR, markeredgecolor=EDGE, markersize=9, label="Personal choice"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MODEL_NORM_COLOR, markeredgecolor=EDGE, markersize=9, label="Norm judgment"),
    ]
    leg = axes[0].legend(handles=handles, frameon=True, loc="upper left", borderpad=0.25, handletextpad=0.3, fontsize=12)
    legend_box(leg)
    fig.tight_layout(w_pad=1.2)
    save(fig, "section5_3_alignment_majority_mae")
    plt.close(fig)




def plot_5_3_alignment_gap_uncertainty_four_panel() -> None:
    align = no_persona_pref_alignment()
    unc = modelize(pd.read_csv(BASE / "alignment_uncertainty_summary.csv"))
    unc = unc[unc["setting"].eq("persona")].copy()
    unc_avg = unc.groupby("model", as_index=False)["human_model_agreement_correlation"].mean()
    unc_avg["model"] = pd.Categorical(unc_avg["model"], MODEL_ORDER, ordered=True)
    unc_avg = unc_avg.sort_values("model")

    gap = align.groupby("model", as_index=False)["signed_pref_gap_pp"].mean()
    gap["model"] = pd.Categorical(gap["model"], MODEL_ORDER, ordered=True)
    gap = gap.sort_values("model")

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(11.2, 2.55),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08, 1.0]},
    )
    y_base = np.arange(len(MODEL_ORDER))
    markers = {"Personal": "o", "Norm": "s"}
    colors_by_response = {"Personal": MODEL_PERSONAL_COLOR, "Norm": MODEL_NORM_COLOR}
    offsets = {"Personal": -0.14, "Norm": 0.14}

    for ax, metric, xlabel, scale in [
        (axes[0], "majority_choice_alignment", "Majority alignment", lambda col: col * 100),
        (axes[1], "distributional_mae", "Preference-rate\nalignment MAE", lambda col: col),
    ]:
        for resp in ["Personal", "Norm"]:
            sub = align[align["response_label"].eq(resp)].copy()
            y = np.arange(len(sub)) + offsets[resp]
            ax.scatter(
                scale(sub[metric]),
                y,
                s=58,
                marker=markers[resp],
                color=colors_by_response[resp],
                edgecolor=EDGE,
                linewidth=0.85,
                label=resp,
                zorder=3,
            )
        ax.set_xlabel(xlabel, fontsize=11.5, fontweight="bold")

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(MODEL_ORDER, fontsize=11.5)
    axes[0].invert_yaxis()
    axes[0].set_xlim(40, 88)
    axes[1].set_xlim(0.0, 0.40)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    response_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_PERSONAL_COLOR, markeredgecolor=EDGE, markersize=8, label="Personal choice"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MODEL_NORM_COLOR, markeredgecolor=EDGE, markersize=8, label="Norm judgment"),
    ]
    leg = axes[0].legend(handles=response_handles, frameon=True, loc="upper left", borderpad=0.25, handletextpad=0.3, fontsize=10.5)
    legend_box(leg)
    response_handles_wrapped = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_PERSONAL_COLOR, markeredgecolor=EDGE, markersize=8, label="Personal\nchoice"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MODEL_NORM_COLOR, markeredgecolor=EDGE, markersize=8, label="Norm\njudgment"),
    ]
    leg = axes[1].legend(handles=response_handles_wrapped, frameon=True, loc="center right", bbox_to_anchor=(1.0, 0.42), borderpad=0.25, handletextpad=0.3, fontsize=10.5)
    legend_box(leg)

    gap_colors = [PASTEL_GREEN if v >= 0 else PASTEL_RED for v in gap["signed_pref_gap_pp"]]
    axes[2].axvline(0, color="#555555", linewidth=1.1)
    axes[2].barh(y_base, gap["signed_pref_gap_pp"], color=gap_colors, edgecolor=EDGE, linewidth=0.8, height=0.55)
    axes[2].set_xlabel("Signed preference-rate gap\n(pp; + preference)", fontsize=11.5, fontweight="bold")
    axes[2].set_xlim(-22, 30)
    gap_handles = [
        Patch(facecolor=PASTEL_GREEN, edgecolor=EDGE, label="Preference\nallowing"),
        Patch(facecolor=PASTEL_RED, edgecolor=EDGE, label="Culture\nfollowing"),
    ]
    leg = axes[2].legend(handles=gap_handles, loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=10.5, borderpad=0.25, handlelength=1.0, handletextpad=0.35)
    legend_box(leg)

    axes[3].barh(y_base, unc_avg["human_model_agreement_correlation"], color=PASTEL_GREY, edgecolor=EDGE, linewidth=0.8, height=0.55)
    axes[3].axvline(0, color="#555555", linewidth=1.1)
    axes[3].set_xlabel("Uncertainty corr.\n(model vs human)", fontsize=11.5, fontweight="bold")
    axes[3].set_xlim(-0.08, 0.28)

    for i, ax in enumerate(axes):
        ax.grid(axis="x", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.tick_params(axis="y", labelsize=11.5)
        ax.text(-0.12, 1.03, chr(ord("A") + i), transform=ax.transAxes, fontsize=12, fontweight="bold")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.tight_layout(w_pad=0.75)
    save(fig, "section5_3_alignment_gap_uncertainty_four_panel")
    plt.close(fig)

def plot_combined_5_2_5_3() -> None:
    human = pd.read_csv(BASE / "human_response_summary.csv")
    countries = human[human["grouping"].eq("respondent_country")].copy()
    countries["country"] = countries["respondent_country"].replace(
        {"United States of America": "US", "United Kingdom": "UK", "South Africa": "S. Africa"}
    )
    countries["personal_allow"] = (1 - countries["personal_culture_rate"]) * 100
    countries["norm_allow"] = (1 - countries["norm_culture_rate"]) * 100
    countries["gap_pp"] = countries["personal_allow"] - countries["norm_allow"]
    countries = countries.sort_values("gap_pp", ascending=False)

    rel = human[human["grouping"].eq("country_relation")].copy()
    rel["relation"] = pd.Categorical(rel["country_relation"], ["same", "close", "far"], ordered=True)
    rel = rel.sort_values("relation")

    align = no_persona_pref_alignment()
    fig, axes = plt.subplots(2, 2, figsize=(7.9, 4.75), gridspec_kw={"height_ratios": [1.0, 1.0]})

    ax = axes[0, 0]
    y = np.arange(len(countries))
    ax.hlines(y, countries["norm_allow"], countries["personal_allow"], color="#777777", linewidth=1.4, alpha=0.65)
    ax.scatter(countries["personal_allow"], y, s=50, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.8, label="Personal choice", zorder=3)
    ax.scatter(countries["norm_allow"], y, s=50, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.8, label="Norm judgment", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(countries["country"])
    ax.invert_yaxis()
    ax.set_xlabel("Preference-allowing (%)")
    ax.set_xlim(10, 45)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.text(-0.11, 1.03, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
    leg = ax.legend(frameon=True, loc="upper left", borderpad=0.25, handletextpad=0.3, fontsize=8.5)
    legend_box(leg)

    ax = axes[0, 1]
    x = np.arange(len(rel))
    width = 0.34
    ax.bar(x - width / 2, rel["personal_majority_agreement"] * 100, width, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.8, label="Personal")
    ax.bar(x + width / 2, rel["norm_majority_agreement"] * 100, width, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.8, label="Norm")
    ax.set_xticks(x)
    ax.set_xticklabels(["Same", "Close", "Far"])
    ax.set_ylabel("Agreement (%)")
    ax.set_ylim(50, 85)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.text(-0.12, 1.03, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
    leg = ax.legend(frameon=True, loc="upper left", borderpad=0.25, handletextpad=0.3, fontsize=8.5)
    legend_box(leg)

    metrics = [
        ("majority_choice_alignment", "Majority alignment", lambda s: s * 100),
        ("distributional_mae", "Preference-rate alignment MAE", lambda s: s),
    ]
    markers = {"Personal": "o", "Norm": "s"}
    colors_by_response = {"Personal": MODEL_PERSONAL_COLOR, "Norm": MODEL_NORM_COLOR}
    offsets = {"Personal": -0.14, "Norm": 0.14}
    for ax, (metric, xlabel, scale), letter in zip(axes[1], metrics, ["C", "D"]):
        for resp in ["Personal", "Norm"]:
            sub = align[align["response_label"].eq(resp)].copy()
            y = np.arange(len(sub)) + offsets[resp]
            ax.scatter(
                scale(sub[metric]),
                y,
                s=58,
                marker=markers[resp],
                color=colors_by_response[resp],
                edgecolor=EDGE,
                linewidth=0.8,
                zorder=3,
            )
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="x", alpha=0.22, linewidth=0.8)
        ax.text(-0.12, 1.03, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")

    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(40, 88)
    axes[1, 1].set_xlim(0.22, 0.58)
    axes[1, 1].tick_params(labelleft=False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_PERSONAL_COLOR, markeredgecolor=EDGE, markersize=7, label="Personal"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MODEL_NORM_COLOR, markeredgecolor=EDGE, markersize=7, label="Norm"),
    ]
    leg = axes[1, 0].legend(handles=handles, frameon=True, loc="upper left", borderpad=0.25, handletextpad=0.3, fontsize=8.5)
    legend_box(leg)

    for ax in axes.ravel():
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(h_pad=0.45, w_pad=0.95)
    save(fig, "section5_2_5_3_human_alignment_stacked")
    plt.close(fig)


def plot_combined_5_2_5_3_side_by_side() -> None:
    human = pd.read_csv(BASE / "human_response_summary.csv")
    countries = human[human["grouping"].eq("respondent_country")].copy()
    countries["country"] = countries["respondent_country"].replace(
        {"United States of America": "US", "United Kingdom": "UK", "South Africa": "S. Afr."}
    )
    countries["personal_allow"] = (1 - countries["personal_culture_rate"]) * 100
    countries["norm_allow"] = (1 - countries["norm_culture_rate"]) * 100
    countries["gap_pp"] = countries["personal_allow"] - countries["norm_allow"]
    countries = countries.sort_values("gap_pp", ascending=False)

    rel = human[human["grouping"].eq("country_relation")].copy()
    rel["relation"] = pd.Categorical(rel["country_relation"], ["same", "close", "far"], ordered=True)
    rel = rel.sort_values("relation")
    align = no_persona_pref_alignment()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.9, 2.35),
        gridspec_kw={"width_ratios": [1.28, 0.82, 1.05, 1.05]},
    )

    ax = axes[0]
    y = np.arange(len(countries))
    ax.hlines(y, countries["norm_allow"], countries["personal_allow"], color="#777777", linewidth=1.1, alpha=0.65)
    ax.scatter(countries["personal_allow"], y, s=34, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.7, label="Personal", zorder=3)
    ax.scatter(countries["norm_allow"], y, s=34, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.7, label="Norm", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(countries["country"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Preference\nallowing (%)", fontsize=9)
    ax.set_xlim(10, 45)
    ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    ax.text(-0.20, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    leg = ax.legend(frameon=True, loc="upper left", fontsize=7.5, borderpad=0.18, handletextpad=0.25)
    legend_box(leg)

    ax = axes[1]
    x = np.arange(len(rel))
    width = 0.34
    ax.bar(x - width / 2, rel["personal_majority_agreement"] * 100, width, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.7)
    ax.bar(x + width / 2, rel["norm_majority_agreement"] * 100, width, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Same", "Close", "Far"], fontsize=7.5, rotation=25, ha="right")
    ax.set_xlabel("Human\nagreement (%)", fontsize=9)
    ax.set_ylim(50, 85)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.text(-0.23, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    markers = {"Personal": "o", "Norm": "s"}
    colors_by_response = {"Personal": MODEL_PERSONAL_COLOR, "Norm": MODEL_NORM_COLOR}
    offsets = {"Personal": -0.13, "Norm": 0.13}
    for ax, metric, xlabel, scale, letter in [
        (axes[2], "majority_choice_alignment", "Majority\nalignment (%)", lambda s: s * 100, "C"),
        (axes[3], "distributional_mae", "Distributional\nMAE", lambda s: s, "D"),
    ]:
        for resp in ["Personal", "Norm"]:
            sub = align[align["response_label"].eq(resp)].copy()
            yy = np.arange(len(sub)) + offsets[resp]
            ax.scatter(
                scale(sub[metric]),
                yy,
                s=34,
                marker=markers[resp],
                color=colors_by_response[resp],
                edgecolor=EDGE,
                linewidth=0.7,
                zorder=3,
                label=resp,
            )
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.text(-0.19, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

    axes[2].set_yticklabels(MODEL_ORDER, fontsize=8.5)
    axes[2].invert_yaxis()
    axes[3].tick_params(labelleft=False)
    axes[2].set_xlim(40, 88)
    axes[3].set_xlim(0.22, 0.58)
    leg = axes[2].legend(frameon=True, loc="lower right", fontsize=7.5, borderpad=0.18, handletextpad=0.25)
    legend_box(leg)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=0.5)
    save(fig, "section5_2_5_3_human_alignment_side_by_side")
    plt.close(fig)

    # Split the four-panel side-by-side figure into two manuscript-friendly pairs:
    # A-B show human response patterns, C-D show human-model alignment.
    split_specs = [
        ("section5_2_human_side_by_side", "AB"),
        ("section5_3_alignment_side_by_side", "CD"),
    ]
    for filename, panel_pair in split_specs:
        split_fig, split_axes = plt.subplots(
            1,
            2,
            figsize=(4.35, 2.35),
            gridspec_kw={"width_ratios": [1.28, 0.82] if panel_pair == "AB" else [1.0, 1.0]},
        )
        if panel_pair == "AB":
            ax = split_axes[0]
            y = np.arange(len(countries))
            ax.hlines(y, countries["norm_allow"], countries["personal_allow"], color="#777777", linewidth=1.1, alpha=0.65)
            ax.scatter(countries["personal_allow"], y, s=34, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.7, label="Personal", zorder=3)
            ax.scatter(countries["norm_allow"], y, s=34, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.7, label="Norm", zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels(countries["country"], fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlabel("Preference\nallowing (%)", fontsize=9)
            ax.set_xlim(10, 45)
            ax.grid(axis="x", alpha=0.22, linewidth=0.7)
            ax.text(-0.20, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
            leg = ax.legend(frameon=True, loc="upper left", fontsize=7.5, borderpad=0.18, handletextpad=0.25)
            legend_box(leg)

            ax = split_axes[1]
            x = np.arange(len(rel))
            width = 0.34
            ax.bar(x - width / 2, rel["personal_majority_agreement"] * 100, width, color=PASTEL_BLUE, edgecolor=EDGE, linewidth=0.7)
            ax.bar(x + width / 2, rel["norm_majority_agreement"] * 100, width, color=PASTEL_PINK, edgecolor=EDGE, linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(["Same", "Close", "Far"], fontsize=7.5, rotation=25, ha="right")
            ax.set_xlabel("Human\nagreement (%)", fontsize=9)
            ax.set_ylim(50, 85)
            ax.grid(axis="y", alpha=0.22, linewidth=0.7)
            ax.text(-0.23, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")
        else:
            for ax, metric, xlabel, scale, letter in [
                (split_axes[0], "majority_choice_alignment", "Majority\nalignment (%)", lambda s: s * 100, "C"),
                (split_axes[1], "distributional_mae", "Distributional\nMAE", lambda s: s, "D"),
            ]:
                for resp in ["Personal", "Norm"]:
                    sub = align[align["response_label"].eq(resp)].copy()
                    yy = np.arange(len(sub)) + offsets[resp]
                    ax.scatter(
                        scale(sub[metric]),
                        yy,
                        s=34,
                        marker=markers[resp],
                        color=colors_by_response[resp],
                        edgecolor=EDGE,
                        linewidth=0.7,
                        zorder=3,
                        label=resp,
                    )
                ax.set_yticks(np.arange(len(MODEL_ORDER)))
                ax.grid(axis="x", alpha=0.22, linewidth=0.7)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.text(-0.19, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
            split_axes[0].set_yticklabels(MODEL_ORDER, fontsize=8.5)
            split_axes[0].invert_yaxis()
            split_axes[1].tick_params(labelleft=False)
            split_axes[0].set_xlim(40, 88)
            split_axes[1].set_xlim(0.22, 0.58)
            leg = split_axes[0].legend(frameon=True, loc="lower right", fontsize=7.5, borderpad=0.18, handletextpad=0.25)
            legend_box(leg)

        for ax in split_axes:
            ax.tick_params(axis="x", labelsize=8.5)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        split_fig.tight_layout(w_pad=0.5)
        save(split_fig, filename)
        plt.close(split_fig)


def persona_average_alignment() -> pd.DataFrame:
    per = modelize(pd.read_csv(BASE / "alignment_persona_summary.csv"))
    per["distributional_mae"] = (per["model_culture_rate"] - per["human_culture_rate"]).abs()
    out = (
        per.groupby(["model", "response_type"], as_index=False)
        .agg(
            majority_choice_alignment=("majority_choice_alignment", "mean"),
            distributional_mae=("distributional_mae", "mean"),
            human_culture_rate=("human_culture_rate", "mean"),
            model_culture_rate=("model_culture_rate", "mean"),
        )
    )
    out["response_label"] = out["response_type"].map({"personal": "Personal", "norm": "Norm"})
    return out




def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        dominates = (
            (df[x_col] >= row[x_col])
            & (df[y_col] <= row[y_col])
            & ((df[x_col] > row[x_col]) | (df[y_col] < row[y_col]))
        )
        if not dominates.any():
            rows.append(row)
    return pd.DataFrame(rows).sort_values(x_col)


def plot_5_3_alignment_pareto_scatter() -> None:
    df = no_persona_pref_alignment().copy()
    avg = (
        df.groupby("model", observed=False, as_index=False)
        .agg(
            majority_alignment=("majority_choice_alignment", "mean"),
            distributional_mae=("distributional_mae", "mean"),
        )
        .dropna()
    )
    avg["majority_alignment_pct"] = avg["majority_alignment"] * 100
    avg["model"] = avg["model"].astype(str)
    frontier = pareto_frontier(avg, "majority_alignment_pct", "distributional_mae")

    fig, ax = plt.subplots(figsize=(4.65, 3.15))
    for _, row in avg.iterrows():
        ax.scatter(
            row["majority_alignment_pct"],
            row["distributional_mae"],
            s=135,
            color=MODEL_COLORS.get(row["model"], PASTEL_GREY),
            edgecolor=EDGE,
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            row["model"],
            (row["majority_alignment_pct"], row["distributional_mae"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=9.2,
            ha="left",
            va="bottom",
        )

    if len(frontier) > 1:
        ax.plot(
            frontier["majority_alignment_pct"],
            frontier["distributional_mae"],
            color="#777777",
            linewidth=1.25,
            linestyle="--",
            zorder=2,
        )

    ax.set_xlabel("Majority alignment")
    ax.set_ylabel("Distributional MAE")
    ax.set_xlim(48, 89)
    ax.set_ylim(0.22, 0.56)
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        "Better",
        xy=(0.91, 0.16),
        xytext=(0.70, 0.34),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#555555"},
        ha="left",
        va="center",
    )
    ax.text(
        0.02,
        0.03,
        "Point = model average across\npersonal choice and norm judgment",
        transform=ax.transAxes,
        fontsize=8.2,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92, "pad": 2.5},
    )
    fig.tight_layout()
    save(fig, "section5_3_alignment_pareto_scatter")
    plt.close(fig)


def plot_5_3_alignment_pareto_scatter_by_question() -> None:
    df = no_persona_pref_alignment().copy()
    df["question_label"] = df["response_label"].map(
        {"Personal": "Personal choice", "Norm": "Norm judgment"}
    )
    df["majority_alignment_pct"] = df["majority_choice_alignment"] * 100
    df["model"] = df["model"].astype(str)

    fig, ax = plt.subplots(figsize=(5.35, 3.25))
    markers = {"Personal choice": "o", "Norm judgment": "s"}
    label_offsets = {"Personal choice": (5, 4), "Norm judgment": (5, -9)}

    for _, row in df.sort_values(["model", "question_label"]).iterrows():
        question = row["question_label"]
        ax.scatter(
            row["majority_alignment_pct"],
            row["distributional_mae"],
            s=125,
            marker=markers[question],
            color=MODEL_COLORS.get(row["model"], PASTEL_GREY),
            edgecolor=EDGE,
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            row["model"],
            (row["majority_alignment_pct"], row["distributional_mae"]),
            xytext=label_offsets[question],
            textcoords="offset points",
            fontsize=8.8,
            ha="left",
            va="center",
        )

    frontier = pareto_frontier(df, "majority_alignment_pct", "distributional_mae")
    if len(frontier) > 1:
        ax.plot(
            frontier["majority_alignment_pct"],
            frontier["distributional_mae"],
            color="#777777",
            linewidth=1.15,
            linestyle="--",
            zorder=2,
        )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=EDGE, markersize=8, label="Personal choice"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=EDGE, markersize=8, label="Norm judgment"),
    ]
    leg = ax.legend(handles=handles, frameon=True, loc="upper left", borderpad=0.32, handletextpad=0.42, fontsize=10.5)
    legend_box(leg)

    ax.set_xlabel("Majority alignment")
    ax.set_ylabel("Distributional MAE")
    ax.set_xlim(45, 89)
    ax.set_ylim(0.22, 0.56)
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        "Better",
        xy=(0.91, 0.16),
        xytext=(0.70, 0.34),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#555555"},
        ha="left",
        va="center",
    )
    fig.tight_layout()
    save(fig, "section5_3_alignment_pareto_scatter_by_question")
    plt.close(fig)


def plot_5_3_alignment_error_map() -> None:
    df = no_persona_pref_alignment().copy()
    df["question_label"] = df["response_label"].map(
        {"Personal": "Personal choice", "Norm": "Norm judgment"}
    )
    df["majority_alignment_pct"] = df["majority_choice_alignment"] * 100

    panels = [
        ("Personal choice", MODEL_PERSONAL_COLOR),
        ("Norm judgment", MODEL_NORM_COLOR),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35), sharex=True, sharey=True)

    for ax, (question, base_color) in zip(axes, panels):
        sub = df[df["question_label"].eq(question)].copy()
        sub["model"] = pd.Categorical(sub["model"], MODEL_ORDER, ordered=True)
        sub = sub.sort_values("model")

        for _, row in sub.iterrows():
            ax.scatter(
                row["majority_alignment_pct"],
                row["distributional_mae"],
                s=130,
                color=MODEL_COLORS.get(str(row["model"]), base_color),
                edgecolor=EDGE,
                linewidth=0.9,
                zorder=3,
            )
            ax.annotate(
                str(row["model"]),
                (row["majority_alignment_pct"], row["distributional_mae"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=8.8,
                ha="left",
                va="bottom",
            )

        ax.set_title(question, fontsize=12, pad=4)
        ax.set_xlabel("Majority alignment with humans (%)")
        ax.grid(alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Distributional MAE")
    axes[0].set_xlim(44, 86)
    axes[0].set_ylim(0.56, 0.22)
    axes[0].text(0.98, 0.06, "Better", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9.5, fontweight="bold")
    axes[0].annotate(
        "",
        xy=(0.96, 0.13),
        xytext=(0.80, 0.32),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#555555"},
    )
    axes[0].text(-0.12, 1.03, "A", transform=axes[0].transAxes, fontsize=12, fontweight="bold")
    axes[1].text(-0.12, 1.03, "B", transform=axes[1].transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout(w_pad=1.0)
    save(fig, "section5_3_alignment_error_map")
    plt.close(fig)

def plot_5_3_gap_persona_uncertainty() -> None:
    no = no_persona_pref_alignment()
    unc = modelize(pd.read_csv(BASE / "alignment_uncertainty_summary.csv"))
    unc = unc[unc["setting"].eq("persona")].copy()
    unc["response_label"] = unc["response_type"].map({"personal": "Personal", "norm": "Norm"})
    unc_avg = unc.groupby("model", as_index=False)["human_model_agreement_correlation"].mean()
    unc_avg["model"] = pd.Categorical(unc_avg["model"], MODEL_ORDER, ordered=True)
    unc_avg = unc_avg.sort_values("model")

    no_avg = no.groupby("model", as_index=False)["signed_pref_gap_pp"].mean()
    no_avg["signed_culture_gap_pp"] = -no_avg["signed_pref_gap_pp"]
    no_avg["model"] = pd.Categorical(no_avg["model"], MODEL_ORDER, ordered=True)
    no_avg = no_avg.sort_values("model")

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.95), sharey=True, gridspec_kw={"width_ratios": [1.08, 1.05]})

    y = np.arange(len(MODEL_ORDER))
    colors = [PASTEL_RED if v >= 0 else PASTEL_GREEN for v in no_avg["signed_culture_gap_pp"]]
    axes[0].axvline(0, color="#555555", linewidth=1.1)
    axes[0].barh(y, no_avg["signed_culture_gap_pp"], color=colors, edgecolor=EDGE, linewidth=0.8, height=0.55)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(MODEL_ORDER)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Signed culture-rate gap\n(pp; + culture)")
    from matplotlib.patches import Patch
    leg = axes[0].legend(
        handles=[
            Patch(facecolor=PASTEL_RED, edgecolor=EDGE, label="Culture\nfollowing"),
            Patch(facecolor=PASTEL_GREEN, edgecolor=EDGE, label="Preference\nallowing"),
        ],
        loc="center right",
        frameon=True,
        fontsize=10.5,
        borderpad=0.25,
        handlelength=1.0,
        handletextpad=0.35,
    )
    legend_box(leg)

    axes[1].barh(y, unc_avg["human_model_agreement_correlation"], color=PASTEL_GREY, edgecolor=EDGE, linewidth=0.8, height=0.55)
    axes[1].axvline(0, color="#555555", linewidth=1.1)
    axes[1].set_xlabel("Uncertainty corr.\n(model vs human)")

    for i, ax in enumerate(axes):
        ax.grid(axis="x", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.14, 1.03, chr(ord("A") + i), transform=ax.transAxes, fontsize=12, fontweight="bold")

    axes[0].set_xlim(-30, 22)
    axes[1].set_xlim(-0.08, 0.28)
    fig.tight_layout(w_pad=0.8)
    save(fig, "section5_3_gap_persona_uncertainty")
    plt.close(fig)


def main() -> None:
    style()
    plot_5_2_human_rates_agreement()
    plot_5_2_human_regression()
    plot_5_3_alignment_majority_mae()
    plot_5_3_alignment_pareto_scatter()
    plot_5_3_alignment_pareto_scatter_by_question()
    plot_5_3_alignment_error_map()
    plot_5_3_gap_persona_uncertainty()
    plot_5_3_alignment_gap_uncertainty_four_panel()
    plot_combined_5_2_5_3()
    plot_combined_5_2_5_3_side_by_side()
    print(f"Wrote Section 5 ACL figures to {OUTDIR}")


if __name__ == "__main__":
    main()
