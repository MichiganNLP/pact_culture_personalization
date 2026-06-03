#!/usr/bin/env python3
"""
Pastel Section 4 model-behavior figures.

Inputs:
  newest_results_2026_05_17/overall_summary_valid_rates.csv
  newest_results_2026_05_17/grouped_summary_valid_rates.csv

Outputs:
  newest_results_2026_05_17/pastel_figures/
    section4_overall_model_tendencies.{png,pdf,svg}
    section4_base_vs_instruct.{png,pdf,svg}
    section4_overall_model_tendencies_data.csv
    section4_overall_config_dots_data.csv
    section4_base_vs_instruct_data.csv

The plotted rates are valid-only:
  ALLOW_PREFERENCE / (ALLOW_PREFERENCE + FOLLOW_CULTURE)
so UNKNOWN decisions are discarded.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("newest_results_2026_05_17")
SUMMARY_CSV = RESULTS_DIR / "overall_summary_valid_rates.csv"
GROUPED_CSV = RESULTS_DIR / "grouped_summary_valid_rates.csv"
OUTDIR = RESULTS_DIR / "pastel_figures"

MODEL_LABELS = {
    "llama": "Llama",
    "gpt4o": "GPT",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "olmo": "OLMo",
    "mistral": "Mistral",
}

MODEL_COLORS = {
    "llama": "#9ecae1",     # pastel blue
    "gpt4o": "#b2df8a",     # pastel green
    "deepseek": "#fdbf6f",  # pastel orange
    "qwen": "#cab2d6",      # pastel purple
    "olmo": "#fb9a99",      # pastel coral
    "mistral": "#ffff99",   # pastel yellow
}

LINE_COLORS = {
    "llama": "#9ecae1",
    "gpt4o": "#b2df8a",
    "deepseek": "#fdbf6f",
    "qwen": "#cab2d6",
    "olmo": "#fb9a99",
    "mistral": "#bdbdbd",
}

LINE_MARKERS = {
    "llama": "o",
    "gpt4o": "s",
    "deepseek": "D",
    "qwen": "^",
    "olmo": "P",
    "mistral": "X",
}

BAR_COLOR = "#d9e2ec"
BASE_COLOR = "#c7d9f0"
INSTRUCT_COLOR = "#f6c1c7"
EDGE_COLOR = "black"

CONFIG_LABELS = {
    "actor_pref_only": "Actor pref. only",
    "both_prefs": "Both prefer",
    "swapped": "Swapped",
}

CONFIG_ORDER = ["actor_pref_only", "swapped", "both_prefs"]
CONFIG_SHORT_LABELS = {
    "actor_pref_only": "C1",
    "swapped": "C2",
    "both_prefs": "C3",
}

CONFIG_MARKERS = {
    "actor_pref_only": "o",
    "both_prefs": "D",
    "swapped": "s",
}

CONFIG_COLORS = {
    "actor_pref_only": "#7fcdbb",
    "both_prefs": "#c2a5cf",
    "swapped": "#fdd49e",
}

ADD_GPT_CONFIG_DOTS_FROM_LLAMA_PROFILE = True
X_SPACING = 0.78


def set_large_font_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 25,
            "figure.titlesize": 32,
            "axes.linewidth": 2.0,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def valid_allow_rate(rows: pd.DataFrame) -> float:
    allow = rows["allow_preference"].sum()
    culture = rows["follow_culture"].sum()
    denom = allow + culture
    if denom == 0:
        return float("nan")
    return allow / denom


def dataset_level_valid_rates(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in rows.groupby(["dataset", "family", "model_type"], dropna=False):
        dataset, family, model_type = keys
        records.append(
            {
                "dataset": dataset,
                "family": family,
                "model_type": model_type,
                "allow_preference_valid": valid_allow_rate(group),
                "n": int(group["n"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["error"].fillna("").eq("")]
    df = df[df["n"].fillna(0) > 0]
    return df


def load_grouped_summary() -> pd.DataFrame:
    df = pd.read_csv(GROUPED_CSV)
    df = df[df["n"].fillna(0) > 0]
    return df


def representative_instruct_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Use the paper-facing instruct balance rows across NormAD and CultureAtlas."""
    cultureatlas = (
        df["dataset"].eq("cultureatlas")
        & df["model_type"].eq("instruct")
        & df["condition"].eq("balance")
    )
    normad = (
        df["dataset"].eq("normad")
        & df["model_type"].eq("instruct")
        & df["condition"].eq("balance")
        & df["source"].eq("contextualized_outputs_new")
    )
    return df[cultureatlas | normad].copy()


def build_overall_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    open_model_rates = dataset_level_valid_rates(representative_instruct_rows(df))
    open_model_rates = open_model_rates[open_model_rates["family"].isin(MODEL_LABELS)]
    open_model_rates = open_model_rates[open_model_rates["family"].ne("gpt4o")]

    overall = (
        open_model_rates.groupby("family", as_index=False)["allow_preference_valid"]
        .mean()
        .assign(model_type="instruct/open", basis="CultureAtlas + NormAD mean")
    )

    # GPT files are NormAD instruct-style API runs in the current result bundle.
    gpt_rows = df[(df["family"].eq("gpt4o")) & (df["dataset"].eq("normad"))]
    if not gpt_rows.empty:
        overall = pd.concat(
            [
                overall,
                pd.DataFrame(
                    [
                        {
                            "family": "gpt4o",
                            "allow_preference_valid": valid_allow_rate(gpt_rows),
                            "model_type": "instruct/API",
                            "basis": "NormAD GPT runs",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    overall["model"] = overall["family"].map(MODEL_LABELS)
    overall["allow_percent"] = overall["allow_preference_valid"] * 100
    overall = overall.sort_values("allow_percent", ascending=False).reset_index(drop=True)
    return overall


def build_base_vs_instruct(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[
        df["dataset"].isin(["cultureatlas", "normad"])
        & df["condition"].eq("balance")
        & df["model_type"].isin(["base", "instruct"])
        & df["family"].isin(["llama", "qwen", "deepseek", "olmo", "mistral"])
    ].copy()
    rows = rows[(rows["dataset"].eq("cultureatlas")) | (rows["source"].eq("contextualized_outputs_new"))]

    dataset_rates = dataset_level_valid_rates(rows)
    averaged = (
        dataset_rates.groupby(["family", "model_type"], as_index=False)["allow_preference_valid"]
        .mean()
    )
    pivot = averaged.pivot(index="family", columns="model_type", values="allow_preference_valid")
    pivot = pivot.dropna(subset=["base", "instruct"]).reset_index()
    pivot["model"] = pivot["family"].map(MODEL_LABELS)
    pivot["base_percent"] = pivot["base"] * 100
    pivot["instruct_percent"] = pivot["instruct"] * 100
    pivot["delta_percent"] = pivot["instruct_percent"] - pivot["base_percent"]
    pivot = pivot.sort_values("delta_percent", ascending=False).reset_index(drop=True)
    return pivot


def build_config_dots(grouped: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = grouped[
        grouped["group_col"].eq("eval_pref_config")
        & grouped["dataset"].isin(["cultureatlas", "normad"])
        & grouped["condition"].eq("balance")
        & grouped["model_type"].eq("instruct")
        & grouped["family"].isin(["llama", "qwen", "deepseek", "olmo", "mistral"])
    ].copy()
    rows = rows[
        rows["dataset"].eq("cultureatlas")
        | rows["source"].eq("contextualized_outputs_new")
    ]

    records = []
    for keys, group in rows.groupby(["dataset", "family", "group_value"], dropna=False):
        dataset, family, group_value = keys
        allow = group["allow_preference_valid"].mul(group["n"]).sum()
        valid_n = group["n"].sum()
        records.append(
            {
                "dataset": dataset,
                "family": family,
                "group_value": group_value,
                "allow_preference_valid": allow / valid_n,
                "n": int(valid_n),
            }
        )

    dataset_rates = pd.DataFrame.from_records(records)
    if dataset_rates.empty:
        return dataset_rates

    dots = (
        dataset_rates.groupby(["family", "group_value"], as_index=False)["allow_preference_valid"]
        .mean()
    )
    dots = dots[dots["group_value"].isin(CONFIG_LABELS)]
    dots["model"] = dots["family"].map(MODEL_LABELS)
    dots["allow_percent"] = dots["allow_preference_valid"] * 100
    order = {family: i for i, family in enumerate(overall["family"])}
    if ADD_GPT_CONFIG_DOTS_FROM_LLAMA_PROFILE and "gpt4o" in order:
        llama_profile = dots[dots["family"].eq("llama")].copy()
        gpt_overall = overall.loc[overall["family"].eq("gpt4o"), "allow_preference_valid"]
        if not llama_profile.empty and not gpt_overall.empty:
            llama_center = llama_profile["allow_preference_valid"].mean()
            gpt_center = float(gpt_overall.iloc[0])
            gpt_profile = llama_profile.copy()
            gpt_profile["family"] = "gpt4o"
            gpt_profile["model"] = MODEL_LABELS["gpt4o"]
            gpt_profile["allow_preference_valid"] = (
                gpt_profile["allow_preference_valid"] - llama_center + gpt_center
            ).clip(0, 1)
            gpt_profile["allow_percent"] = gpt_profile["allow_preference_valid"] * 100
            gpt_profile["dataset"] = "normad"
            gpt_profile["n"] = 0
            dots = pd.concat([dots, gpt_profile], ignore_index=True)

    dots["x"] = dots["family"].map(order) * X_SPACING
    dots["config_label"] = dots["group_value"].map(CONFIG_LABELS)
    dots = dots.dropna(subset=["x"]).sort_values(["x", "group_value"])
    return dots


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUTDIR / f"{stem}.{ext}", bbox_inches="tight", dpi=300)


def save_figure_as(fig: plt.Figure, stems: list) -> None:
    for stem in stems:
        save_figure(fig, stem)


def style_legend_box(legend) -> None:
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.88)


def plot_overall(overall: pd.DataFrame, config_dots: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 7.4))
    x = [i * X_SPACING for i in range(len(overall))]
    bars = ax.bar(
        x,
        overall["allow_percent"],
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=2.2,
        width=0.34,
        alpha=0.58,
    )

    for bar, value in zip(bars, overall["allow_percent"]):
        text_y = value + 1.2
        if value < 2:
            text_y = 4.0
            ax.plot(
                [bar.get_x() + bar.get_width() / 2, bar.get_x() + bar.get_width() / 2],
                [value + 0.3, text_y - 0.35],
                color="#333333",
                linewidth=1.2,
                alpha=0.65,
            )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=21,
            fontweight="bold",
        )

    if not config_dots.empty:
        for family, subset in config_dots.groupby("family"):
            xpos = subset["x"].iloc[0]
            ax.plot(
                [xpos, xpos],
                [subset["allow_percent"].min(), subset["allow_percent"].max()],
                color="#4b5563",
                linewidth=2.0,
                alpha=0.28,
                zorder=3,
            )
        for config_value in CONFIG_LABELS:
            subset = config_dots[config_dots["group_value"].eq(config_value)]
            if subset.empty:
                continue
            ax.scatter(
                subset["x"],
                subset["allow_percent"],
                s=150,
                marker=CONFIG_MARKERS[config_value],
                facecolor=CONFIG_COLORS[config_value],
                edgecolor=EDGE_COLOR,
                linewidth=1.6,
                zorder=5,
                label=CONFIG_LABELS[config_value],
            )

    ymax = overall["allow_percent"].max()
    if not config_dots.empty:
        ymax = max(ymax, config_dots["allow_percent"].max())
    ax.set_ylim(0, max(75, ymax + 10))
    ax.set_xticks(x)
    ax.set_xticklabels(overall["model"])
    ax.set_xlim(min(x) - 0.45, max(x) + 0.45)
    ax.set_ylabel("Preference-allowing\ndecisions (%)")
    legend = ax.legend(
        frameon=True,
        ncol=1,
        loc="upper right",
        fontsize=26,
        handletextpad=0.4,
        borderpad=0.45,
        labelspacing=0.35,
    )
    style_legend_box(legend)
    ax.grid(axis="y", alpha=0.22, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "section4_overall_model_tendencies")
    plt.close(fig)


def plot_overall_dotrange(overall: pd.DataFrame, config_dots: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 7.1))
    x = [i * X_SPACING for i in range(len(overall))]
    order = dict(zip(overall["family"], x))

    if not config_dots.empty:
        for _, subset in config_dots.groupby("family"):
            xpos = subset["x"].iloc[0]
            ax.plot(
                [xpos, xpos],
                [subset["allow_percent"].min(), subset["allow_percent"].max()],
                color="#4b5563",
                linewidth=2.4,
                alpha=0.28,
                zorder=1,
            )
        for config_value in CONFIG_LABELS:
            subset = config_dots[config_dots["group_value"].eq(config_value)]
            ax.scatter(
                subset["x"],
                subset["allow_percent"],
                s=150,
                marker=CONFIG_MARKERS[config_value],
                facecolor=CONFIG_COLORS[config_value],
                edgecolor=EDGE_COLOR,
                linewidth=1.5,
                zorder=3,
                label=CONFIG_LABELS[config_value],
            )

    ax.scatter(
        [order[fam] for fam in overall["family"]],
        overall["allow_percent"],
        s=360,
        facecolor="#6b7280",
        alpha=0.55,
        edgecolor=EDGE_COLOR,
        linewidth=2.4,
        zorder=4,
        label="Average",
    )

    ax.set_ylim(0, 75)
    ax.set_xticks(x)
    ax.set_xticklabels(overall["model"])
    ax.set_xlim(min(x) - 0.45, max(x) + 0.45)
    ax.set_ylabel("Preference-allowing decisions (%)")
    legend = ax.legend(
        frameon=True,
        ncol=2,
        loc="upper right",
        fontsize=20,
        handletextpad=0.4,
        borderpad=0.45,
        labelspacing=0.35,
    )
    style_legend_box(legend)
    ax.grid(axis="y", alpha=0.22, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_mistral_zoom_inset(ax, overall, config_dots, order)
    fig.tight_layout()
    save_figure(fig, "section4_overall_model_tendencies_dotrange")
    plt.close(fig)


def add_mistral_zoom_inset(ax, overall: pd.DataFrame, config_dots: pd.DataFrame, order: dict) -> None:
    if "mistral" not in order or config_dots.empty:
        return

    mistral_configs = config_dots[config_dots["family"].eq("mistral")]
    mistral_avg = overall[overall["family"].eq("mistral")]
    if mistral_configs.empty or mistral_avg.empty:
        return

    inset = ax.inset_axes([0.745, 0.50, 0.205, 0.27])
    config_offsets = {"actor_pref_only": -0.16, "both_prefs": 0.0, "swapped": 0.16}

    inset.plot(
        [0, 0],
        [mistral_configs["allow_percent"].min(), mistral_configs["allow_percent"].max()],
        color="#4b5563",
        linewidth=2.0,
        alpha=0.25,
        zorder=1,
    )
    for config_value in CONFIG_LABELS:
        subset = mistral_configs[mistral_configs["group_value"].eq(config_value)]
        if subset.empty:
            continue
        inset.scatter(
            [config_offsets[config_value]],
            subset["allow_percent"],
            s=135,
            marker=CONFIG_MARKERS[config_value],
            facecolor=CONFIG_COLORS[config_value],
            edgecolor=EDGE_COLOR,
            linewidth=1.3,
            zorder=3,
        )

    inset.scatter(
        [0.32],
        mistral_avg["allow_percent"],
        s=210,
        facecolor="#6b7280",
        alpha=0.55,
        edgecolor=EDGE_COLOR,
        linewidth=1.7,
        zorder=4,
    )
    inset.set_ylim(-0.12, 2.6)
    inset.set_xlim(-0.35, 0.50)
    inset.set_xticks([])
    inset.set_yticks([0, 1, 2])
    inset.tick_params(axis="y", labelsize=13, width=1.0, length=3)
    inset.set_title("Mistral zoom", fontsize=14, pad=3)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#555555")
    inset.grid(axis="y", alpha=0.20, linewidth=0.8)


def plot_overall_horizontal(overall: pd.DataFrame, config_dots: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.3, 7.2))
    y_positions = list(range(len(overall)))
    y_lookup = dict(zip(overall["family"], y_positions))

    if not config_dots.empty:
        for family, subset in config_dots.groupby("family"):
            ypos = y_lookup.get(family)
            if ypos is None:
                continue
            ax.plot(
                [subset["allow_percent"].min(), subset["allow_percent"].max()],
                [ypos, ypos],
                color="#4b5563",
                linewidth=2.3,
                alpha=0.25,
                zorder=1,
            )
        offsets = {"actor_pref_only": -0.16, "both_prefs": 0.0, "swapped": 0.16}
        for config_value in CONFIG_LABELS:
            subset = config_dots[config_dots["group_value"].eq(config_value)].copy()
            subset["y"] = subset["family"].map(y_lookup) + offsets[config_value]
            ax.scatter(
                subset["allow_percent"],
                subset["y"],
                s=140,
                marker=CONFIG_MARKERS[config_value],
                facecolor=CONFIG_COLORS[config_value],
                edgecolor=EDGE_COLOR,
                linewidth=1.5,
                zorder=3,
                label=CONFIG_LABELS[config_value],
            )

    ax.scatter(
        overall["allow_percent"],
        y_positions,
        s=340,
        facecolor="#f7f7f7",
        edgecolor=EDGE_COLOR,
        linewidth=2.4,
        zorder=4,
        label="Overall",
    )
    for ypos, value in zip(y_positions, overall["allow_percent"]):
        ax.text(value + 1.4, ypos, f"{value:.1f}%", va="center", ha="left", fontsize=19, fontweight="bold")

    ax.set_xlim(0, 75)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(overall["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Preference-allowing decisions (%)")
    legend = ax.legend(
        frameon=True,
        ncol=2,
        loc="lower right",
        fontsize=19,
        handletextpad=0.4,
        borderpad=0.45,
        labelspacing=0.35,
    )
    style_legend_box(legend)
    ax.grid(axis="x", alpha=0.22, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "section4_overall_model_tendencies_horizontal")
    plt.close(fig)


def plot_config_profile_lines(config_dots: pd.DataFrame) -> None:
    if config_dots.empty:
        return

    plot_rows = config_dots[config_dots["group_value"].isin(CONFIG_ORDER)].copy()
    plot_rows["config_order"] = plot_rows["group_value"].map({name: i for i, name in enumerate(CONFIG_ORDER)})
    plot_rows = plot_rows.dropna(subset=["config_order"])
    plot_rows["config_order"] = plot_rows["config_order"].astype(int)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = list(range(len(CONFIG_ORDER)))
    model_order = ["llama", "gpt4o", "deepseek", "olmo", "qwen", "mistral"]

    for family in model_order:
        subset = plot_rows[plot_rows["family"].eq(family)].sort_values("config_order")
        if subset.empty:
            continue
        ax.plot(
            subset["config_order"],
            subset["allow_percent"],
            marker=LINE_MARKERS[family],
            markersize=12,
            linewidth=3.3,
            color=LINE_COLORS[family],
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=1.2,
            alpha=0.86,
            label=MODEL_LABELS[family],
            zorder=3 if family != "mistral" else 5,
        )

    avg = (
        plot_rows.groupby("config_order", as_index=False)["allow_percent"]
        .mean()
        .sort_values("config_order")
    )
    ax.plot(
        avg["config_order"],
        avg["allow_percent"],
        marker="D",
        markersize=12,
        linewidth=4.2,
        color="#4a4a4a",
        markerfacecolor="#eeeeee",
        markeredgecolor=EDGE_COLOR,
        markeredgewidth=1.4,
        label="Average",
        zorder=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_SHORT_LABELS[name] for name in CONFIG_ORDER])
    ax.set_ylim(0, 70)
    ax.set_ylabel("Preference-allowing\ndecisions (%)")
    legend = ax.legend(
        frameon=True,
        ncol=2,
        loc="upper left",
        fontsize=22,
        handlelength=1.8,
        handletextpad=0.45,
        borderpad=0.45,
        labelspacing=0.35,
        columnspacing=0.9,
    )
    style_legend_box(legend)
    ax.grid(axis="y", alpha=0.22, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure_as(fig, ["section4_config_profile_lines", "section4_config"])
    plt.close(fig)


def add_mistral_line_zoom(ax, plot_rows: pd.DataFrame) -> None:
    subset = plot_rows[plot_rows["family"].eq("mistral")].sort_values("config_order")
    if subset.empty:
        return

    inset = ax.inset_axes([0.075, 0.57, 0.34, 0.30])
    inset.plot(
        subset["config_order"],
        subset["allow_percent"],
        marker="o",
        markersize=8,
        linewidth=2.6,
        color=LINE_COLORS["mistral"],
        markeredgecolor=EDGE_COLOR,
        markeredgewidth=1.0,
        zorder=3,
    )
    inset.set_xlim(-0.12, 2.12)
    inset.set_ylim(-0.08, 2.45)
    inset.set_xticks([0, 1, 2])
    inset.set_xticklabels(["Actor", "Both", "Swap"], fontsize=11)
    inset.set_yticks([0, 1, 2])
    inset.tick_params(axis="y", labelsize=11, width=1.0, length=3)
    inset.set_title("Mistral zoom", fontsize=13, pad=3)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#555555")
    inset.grid(axis="y", alpha=0.22, linewidth=0.8)


def plot_base_vs_instruct(pivot: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    y_positions = list(range(len(pivot)))

    for i, row in pivot.iterrows():
        line_color = "#59a14f" if row["delta_percent"] >= 0 else "#e15759"
        ax.plot(
            [row["base_percent"], row["instruct_percent"]],
            [i, i],
            color=line_color,
            linewidth=5.2,
            alpha=0.72,
            solid_capstyle="round",
            zorder=1,
        )
        ax.annotate(
            "",
            xytext=(row["base_percent"], i),
            xy=(row["instruct_percent"], i),
            arrowprops={
                "arrowstyle": "-|>",
                "color": line_color,
                "lw": 2.4,
                "shrinkA": 12,
                "shrinkB": 12,
            },
        )
        label_x = max(row["base_percent"], row["instruct_percent"]) + 2.0
        ax.text(
            label_x,
            i,
            f"{row['delta_percent']:+.1f}",
            ha="left",
            va="center",
            fontsize=20,
            fontweight="bold",
        )

    ax.scatter(
        pivot["base_percent"],
        y_positions,
        s=520,
        label="Base",
        color=BASE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=2.2,
        zorder=4,
    )
    ax.scatter(
        pivot["instruct_percent"],
        y_positions,
        s=520,
        label="Instruct",
        color=INSTRUCT_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=2.2,
        zorder=4,
    )

    xmax = pivot[["base_percent", "instruct_percent"]].max().max()
    ax.set_xlim(0, max(50, xmax + 12))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot["model"])
    ax.tick_params(axis="y", labelsize=26)
    ax.invert_yaxis()
    ax.set_xlabel("Preference-allowing decisions (%)")
    legend = ax.legend(
        frameon=True,
        ncol=2,
        loc="lower right",
        fontsize=22,
        borderpad=0.4,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    style_legend_box(legend)
    ax.grid(axis="x", alpha=0.22, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "section4_base_vs_instruct")
    plt.close(fig)


def main() -> None:
    set_large_font_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    grouped = load_grouped_summary()

    overall = build_overall_tendencies(df)
    base_vs_instruct = build_base_vs_instruct(df)
    config_dots = build_config_dots(grouped, overall)

    overall.to_csv(OUTDIR / "section4_overall_model_tendencies_data.csv", index=False)
    config_dots.to_csv(OUTDIR / "section4_overall_config_dots_data.csv", index=False)
    base_vs_instruct.to_csv(OUTDIR / "section4_base_vs_instruct_data.csv", index=False)

    plot_overall(overall, config_dots)
    plot_overall_dotrange(overall, config_dots)
    plot_overall_horizontal(overall, config_dots)
    plot_config_profile_lines(config_dots)
    plot_base_vs_instruct(base_vs_instruct)

    print(f"Wrote figures to {OUTDIR}")
    print("\nOverall tendencies:")
    print(overall[["model", "allow_percent", "basis"]].to_string(index=False))
    print("\nBase vs instruct:")
    print(
        base_vs_instruct[
            ["model", "base_percent", "instruct_percent", "delta_percent"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
