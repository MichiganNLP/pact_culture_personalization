#!/usr/bin/env python3
"""Model-only analogues of Section 5 human-study analyses."""
from pathlib import Path
from typing import Dict, Any, Sequence
import json
import math

import numpy as np
import pandas as pd

CHOICE_TO_NUM = {"A": 1.0, "B": 0.0}
OUTDIR = Path("human_prolific/section5_outputs/model_only_analysis")

COUNTRY_RELATIONS = {
    "Brazil": {"same": {"Brazil"}, "close": {"Myanmar", "Singapore"}, "far": set()},
    "India": {"same": {"India"}, "close": {"Bangladesh", "Nepal", "Pakistan"}, "far": {"Canada", "United States of America"}},
    "South Africa": {"same": {"South Africa"}, "close": {"Zimbabwe", "Ethiopia"}, "far": {"Argentina", "Bangladesh", "Iran"}},
    "United Kingdom": {"same": {"United Kingdom"}, "close": {"Ireland"}, "far": {"Argentina", "Bangladesh", "Iran"}},
    "United States of America": {"same": {"United States of America"}, "close": {"Canada", "Mexico"}, "far": {"Afghanistan", "Bangladesh", "Pakistan"}},
}
MODEL_LABELS = {
    "deepseek7b_chat": "DeepSeek",
    "gpt54_mini": "GPT",
    "llama31_8b": "Llama",
    "mistral7b_instruct_v03": "Mistral",
    "olmo2_7b": "OLMo",
    "qwen3_4b": "Qwen",
}

NEUTRAL_FILES = [
    "human_prolific/neutral_out/llm_predictions.csv",
    "human_prolific/mistral_deepseek_neutral_out/llm_predictions.csv",
    "human_prolific/gpt54_mini_persona_no_persona_out/llm_predictions.csv",
]
PERSONA_FILES = [
    "human_prolific/llm_predictions.csv",
    "human_prolific/mistral_deepseek_persona_out/llm_predictions.csv",
    "human_prolific/gpt54_mini_persona_no_persona_out/llm_predictions.csv",
]


def choice_to_num(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    return CHOICE_TO_NUM.get(str(x).strip().upper(), np.nan)


def country_relation(persona_country: Any, scenario_country: Any) -> str:
    pc = str(persona_country)
    sc = str(scenario_country)
    rels = COUNTRY_RELATIONS.get(pc, {})
    for label, countries in rels.items():
        if sc in countries:
            return label
    if pc == sc:
        return "same"
    return "far"


def read_prediction_files(files: Sequence[str], setting: str) -> pd.DataFrame:
    frames = []
    for file in files:
        p = Path(file)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "persona_setting" in df.columns:
            expected = "no_persona" if setting == "no_persona" else "persona"
            df = df[df["persona_setting"].astype(str).eq(expected)].copy()
        df["setting"] = setting
        df["prediction_file"] = str(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files found for {setting}")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["setting", "model_tag", "persona_country", "persona_demo", "scenario_item_id"], keep="last")
    return out


def load_predictions() -> pd.DataFrame:
    pred = pd.concat([
        read_prediction_files(NEUTRAL_FILES, "no_persona"),
        read_prediction_files(PERSONA_FILES, "persona"),
    ], ignore_index=True)
    pred["model"] = pred["model_tag"].map(MODEL_LABELS).fillna(pred["model_tag"])
    pred["persona_country"] = pred["persona_country"].fillna("none")
    pred["persona_demo"] = pred["persona_demo"].fillna("none")
    pred["persona_age"] = pred["persona_demo"].astype(str).str.extract(r"(younger|older)", expand=False).fillna("none")
    pred["persona_gender"] = pred["persona_demo"].astype(str).str.extract(r"(female|male)", expand=False).fillna("none")
    pred["country_relation"] = np.where(
        pred["setting"].eq("persona"),
        [country_relation(a, b) for a, b in zip(pred["persona_country"], pred["scenario_country"])],
        "none",
    )
    return pred


def long_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response_type, col in [("personal", "llm_personal"), ("norm", "llm_norm")]:
        tmp = pred.copy()
        tmp["response_type"] = response_type
        tmp["llm_choice"] = tmp[col]
        tmp["follow_culture"] = tmp["llm_choice"].map(choice_to_num)
        tmp["allow_preference"] = 1.0 - tmp["follow_culture"]
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["follow_culture"])


def summarize_gaps(pred_long: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["setting", "model", "model_tag"]
    rates = pred_long.groupby(grp_cols + ["response_type"], dropna=False).agg(
        n=("follow_culture", "size"),
        culture_rate=("follow_culture", "mean"),
        allow_rate=("allow_preference", "mean"),
    ).reset_index()
    wide = rates.pivot_table(index=grp_cols, columns="response_type", values=["n", "culture_rate", "allow_rate"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["allow_gap_personal_minus_norm"] = wide["allow_rate_personal"] - wide["allow_rate_norm"]
    wide["culture_gap_norm_minus_personal"] = wide["culture_rate_norm"] - wide["culture_rate_personal"]
    wide.to_csv(OUTDIR / "model_norm_personal_gaps_by_model.csv", index=False)

    setting = pred_long.groupby(["setting", "response_type"], dropna=False).agg(
        n=("follow_culture", "size"),
        culture_rate=("follow_culture", "mean"),
        allow_rate=("allow_preference", "mean"),
    ).reset_index()
    sw = setting.pivot_table(index=["setting"], columns="response_type", values=["n", "culture_rate", "allow_rate"], aggfunc="first")
    sw.columns = [f"{a}_{b}" for a, b in sw.columns]
    sw = sw.reset_index()
    sw["allow_gap_personal_minus_norm"] = sw["allow_rate_personal"] - sw["allow_rate_norm"]
    sw.to_csv(OUTDIR / "model_norm_personal_gaps_by_setting.csv", index=False)

    persona_country = pred_long[pred_long.setting.eq("persona")].groupby(["model", "model_tag", "persona_country", "response_type"], dropna=False).agg(
        n=("follow_culture", "size"),
        culture_rate=("follow_culture", "mean"),
        allow_rate=("allow_preference", "mean"),
    ).reset_index()
    pcw = persona_country.pivot_table(index=["model", "model_tag", "persona_country"], columns="response_type", values=["n", "culture_rate", "allow_rate"], aggfunc="first")
    pcw.columns = [f"{a}_{b}" for a, b in pcw.columns]
    pcw = pcw.reset_index()
    pcw["allow_gap_personal_minus_norm"] = pcw["allow_rate_personal"] - pcw["allow_rate_norm"]
    pcw.to_csv(OUTDIR / "model_norm_personal_gaps_by_persona_country.csv", index=False)
    return wide


def majority_agreement(rate: float) -> float:
    return float(max(rate, 1.0 - rate)) if not pd.isna(rate) else np.nan


def summarize_agreement(pred: pd.DataFrame, pred_long: pd.DataFrame) -> None:
    # Same-run consistency between the two question frames.
    tmp = pred.copy()
    tmp["personal_num"] = tmp["llm_personal"].map(choice_to_num)
    tmp["norm_num"] = tmp["llm_norm"].map(choice_to_num)
    tmp["personal_norm_same"] = tmp["llm_personal"].astype(str).str.upper().eq(tmp["llm_norm"].astype(str).str.upper())
    consistency = tmp.dropna(subset=["personal_num", "norm_num"]).groupby(["setting", "model", "model_tag"], dropna=False).agg(
        n=("scenario_item_id", "size"),
        personal_norm_choice_agreement=("personal_norm_same", "mean"),
        mean_abs_personal_norm_culture_gap=("personal_num", lambda s: np.nan),
    ).reset_index()
    rows = []
    for keys, grp in tmp.dropna(subset=["personal_num", "norm_num"]).groupby(["setting", "model", "model_tag"], dropna=False):
        rows.append((*keys, float((grp["personal_num"] - grp["norm_num"]).abs().mean())))
    gap_df = pd.DataFrame(rows, columns=["setting", "model", "model_tag", "mean_abs_personal_norm_culture_gap"])
    consistency = consistency.drop(columns=["mean_abs_personal_norm_culture_gap"]).merge(gap_df, on=["setting", "model", "model_tag"])
    consistency.to_csv(OUTDIR / "model_personal_norm_choice_consistency.csv", index=False)

    # Consensus across model/persona samples for each item and response type.
    all_consensus = pred_long.groupby(["setting", "response_type", "scenario_item_id"], dropna=False).agg(
        culture_rate=("follow_culture", "mean"),
        n_samples=("follow_culture", "size"),
    ).reset_index()
    all_consensus["majority_agreement"] = all_consensus["culture_rate"].map(majority_agreement)
    all_consensus.to_csv(OUTDIR / "model_item_consensus_all_samples.csv", index=False)
    setting_consensus = all_consensus.groupby(["setting", "response_type"], dropna=False).agg(
        n_items=("scenario_item_id", "nunique"),
        mean_majority_agreement=("majority_agreement", "mean"),
        mean_samples_per_item=("n_samples", "mean"),
    ).reset_index()
    setting_consensus.to_csv(OUTDIR / "model_consensus_by_setting_response.csv", index=False)

    by_model = pred_long.groupby(["setting", "model", "model_tag", "response_type", "scenario_item_id"], dropna=False).agg(
        culture_rate=("follow_culture", "mean"),
        n_samples=("follow_culture", "size"),
    ).reset_index()
    by_model["majority_agreement"] = by_model["culture_rate"].map(majority_agreement)
    by_model_summary = by_model.groupby(["setting", "model", "model_tag", "response_type"], dropna=False).agg(
        n_items=("scenario_item_id", "nunique"),
        mean_majority_agreement=("majority_agreement", "mean"),
        mean_samples_per_item=("n_samples", "mean"),
    ).reset_index()
    by_model_summary.to_csv(OUTDIR / "model_consensus_by_model_response.csv", index=False)


def factor_matrix(data: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(data[col].fillna("NA").astype(str), prefix=prefix, drop_first=True, dtype=float)


def interaction_matrix(data: pd.DataFrame, a: str, b: str, prefix: str) -> pd.DataFrame:
    vals = data[a].fillna("NA").astype(str) + ":" + data[b].fillna("NA").astype(str)
    return pd.get_dummies(vals, prefix=prefix, drop_first=True, dtype=float)


def build_design(data: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    parts = [pd.DataFrame({"Intercept": np.ones(len(data), dtype=float)}, index=data.index)]
    for term in terms:
        if term == "model": parts.append(factor_matrix(data, "model", "model"))
        elif term == "question_type": parts.append(factor_matrix(data, "response_type", "question"))
        elif term == "scenario_country": parts.append(factor_matrix(data, "scenario_country", "scenario_country"))
        elif term == "receiver_demographics":
            parts.append(factor_matrix(data, "receiver_age", "receiver_age")); parts.append(factor_matrix(data, "receiver_gender", "receiver_gender"))
        elif term == "persona_country": parts.append(factor_matrix(data, "persona_country", "persona_country"))
        elif term == "persona_demographics":
            parts.append(factor_matrix(data, "persona_age", "persona_age")); parts.append(factor_matrix(data, "persona_gender", "persona_gender"))
        elif term == "country_relation": parts.append(factor_matrix(data, "country_relation", "country_relation"))
        elif term == "question_x_model": parts.append(interaction_matrix(data, "response_type", "model", "question_x_model"))
        elif term == "question_x_persona_country": parts.append(interaction_matrix(data, "response_type", "persona_country", "question_x_persona_country"))
        elif term == "question_x_relation": parts.append(interaction_matrix(data, "response_type", "country_relation", "question_x_relation"))
        else: raise ValueError(term)
    x = pd.concat(parts, axis=1)
    return x.loc[:, ~x.columns.duplicated()]


def fit_logit(data: pd.DataFrame, terms: Sequence[str], ridge: float = 1e-6) -> Dict[str, Any]:
    x_df = build_design(data, terms)
    x = x_df.to_numpy(dtype=float)
    y = data["follow_culture"].to_numpy(dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    ridge_vec = np.full(x.shape[1], ridge, dtype=float); ridge_vec[0] = 0.0
    for _ in range(100):
        eta = np.clip(x @ beta, -35, 35)
        p = 1/(1+np.exp(-eta))
        w = np.clip(p*(1-p), 1e-7, None)
        grad = x.T @ (y-p) - ridge_vec*beta
        hess = (x.T*w) @ x + np.diag(ridge_vec)
        try: step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError: step = np.linalg.pinv(hess) @ grad
        beta += step
        if np.max(np.abs(step)) < 1e-7: break
    eta = np.clip(x @ beta, -35, 35)
    p = np.clip(1/(1+np.exp(-eta)), 1e-12, 1-1e-12)
    loglik = float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
    return {"deviance": -2*loglik, "df_model": max(int(np.linalg.matrix_rank(x))-1, 0), "nobs": len(data), "terms": list(terms)}


def regression_drop_one(pred_long: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "no_persona_combined": {
            "data": pred_long[pred_long.setting.eq("no_persona")].copy(),
            "terms": ["question_type", "model", "scenario_country", "receiver_demographics", "question_x_model"],
            "drop": {
                "question_type": ["question_type"], "model_family": ["model"], "scenario_country": ["scenario_country"],
                "receiver_demographics": ["receiver_demographics"], "question_x_model": ["question_x_model"],
            },
        },
        "persona_combined": {
            "data": pred_long[pred_long.setting.eq("persona")].copy(),
            "terms": ["question_type", "model", "persona_country", "persona_demographics", "receiver_demographics", "country_relation", "scenario_country", "question_x_model", "question_x_persona_country", "question_x_relation"],
            "drop": {
                "question_type": ["question_type"], "model_family": ["model"], "persona_country": ["persona_country"],
                "persona_demographics": ["persona_demographics"], "receiver_demographics": ["receiver_demographics"],
                "country_relation": ["country_relation"], "scenario_country": ["scenario_country"],
                "question_x_model": ["question_x_model"], "question_x_persona_country": ["question_x_persona_country"],
                "question_x_relation": ["question_x_relation"],
            },
        },
        "persona_personal_only": {
            "data": pred_long[(pred_long.setting.eq("persona")) & (pred_long.response_type.eq("personal"))].copy(),
            "terms": ["model", "persona_country", "persona_demographics", "receiver_demographics", "country_relation", "scenario_country"],
            "drop": {"model_family": ["model"], "persona_country": ["persona_country"], "persona_demographics": ["persona_demographics"], "receiver_demographics": ["receiver_demographics"], "country_relation": ["country_relation"], "scenario_country": ["scenario_country"]},
        },
        "persona_norm_only": {
            "data": pred_long[(pred_long.setting.eq("persona")) & (pred_long.response_type.eq("norm"))].copy(),
            "terms": ["model", "persona_country", "persona_demographics", "receiver_demographics", "country_relation", "scenario_country"],
            "drop": {"model_family": ["model"], "persona_country": ["persona_country"], "persona_demographics": ["persona_demographics"], "receiver_demographics": ["receiver_demographics"], "country_relation": ["country_relation"], "scenario_country": ["scenario_country"]},
        },
        "no_persona_personal_only": {
            "data": pred_long[(pred_long.setting.eq("no_persona")) & (pred_long.response_type.eq("personal"))].copy(),
            "terms": ["model", "receiver_demographics", "scenario_country"],
            "drop": {"model_family": ["model"], "receiver_demographics": ["receiver_demographics"], "scenario_country": ["scenario_country"]},
        },
        "no_persona_norm_only": {
            "data": pred_long[(pred_long.setting.eq("no_persona")) & (pred_long.response_type.eq("norm"))].copy(),
            "terms": ["model", "receiver_demographics", "scenario_country"],
            "drop": {"model_family": ["model"], "receiver_demographics": ["receiver_demographics"], "scenario_country": ["scenario_country"]},
        },
    }
    rows = []
    for name, spec in specs.items():
        data = spec["data"].dropna(subset=["follow_culture"])
        full = fit_logit(data, spec["terms"])
        for label, removed in spec["drop"].items():
            reduced_terms = [t for t in spec["terms"] if t not in removed]
            red = fit_logit(data, reduced_terms)
            rows.append({
                "model": name, "dropped_factor": label, "nobs": full["nobs"],
                "full_deviance": full["deviance"], "reduced_deviance": red["deviance"],
                "drop_one_deviance": red["deviance"] - full["deviance"],
                "full_df_model": full["df_model"], "reduced_df_model": red["df_model"],
                "df_difference": full["df_model"] - red["df_model"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "model_behavior_regression_drop_one_deviance.csv", index=False)
    return out


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def write_findings(gaps: pd.DataFrame, reg: pd.DataFrame) -> None:
    setting = pd.read_csv(OUTDIR / "model_norm_personal_gaps_by_setting.csv")
    consistency = pd.read_csv(OUTDIR / "model_personal_norm_choice_consistency.csv")
    consensus = pd.read_csv(OUTDIR / "model_consensus_by_setting_response.csv")
    by_model_consensus = pd.read_csv(OUTDIR / "model_consensus_by_model_response.csv")

    lines = ["# Model-only Section 5 analyses", ""]
    lines.append("## Norm-personal gap")
    for _, r in setting.iterrows():
        lines.append(f"- {r.setting}: personal allow={pct(r.allow_rate_personal)}, norm allow={pct(r.allow_rate_norm)}, gap personal-minus-norm={100*r.allow_gap_personal_minus_norm:+.1f} pp.")
    lines.append("")
    lines.append("By model:")
    for _, r in gaps.sort_values(["setting", "allow_gap_personal_minus_norm"]).iterrows():
        lines.append(f"- {r.setting} / {r.model}: personal allow={pct(r.allow_rate_personal)}, norm allow={pct(r.allow_rate_norm)}, gap={100*r.allow_gap_personal_minus_norm:+.1f} pp.")

    lines.append("")
    lines.append("## Regression drop-one findings")
    for model_name in ["no_persona_combined", "persona_combined", "no_persona_personal_only", "no_persona_norm_only", "persona_personal_only", "persona_norm_only"]:
        sub = reg[reg.model.eq(model_name)].sort_values("drop_one_deviance", ascending=False)
        lines.append(f"- {model_name}: top factors are " + "; ".join(f"{rr.dropped_factor} ({rr.drop_one_deviance:.1f})" for _, rr in sub.head(3).iterrows()) + ".")

    lines.append("")
    lines.append("## Agreement / consistency")
    for _, r in consistency.groupby("setting").personal_norm_choice_agreement.mean().reset_index().iterrows():
        lines.append(f"- Mean within-run personal-vs-norm choice agreement for {r.setting}: {pct(r.personal_norm_choice_agreement)}.")
    for _, r in consensus.iterrows():
        lines.append(f"- Cross-sample consensus for {r.setting}/{r.response_type}: mean majority agreement={pct(r.mean_majority_agreement)} over {int(r.n_items)} items (mean samples/item={r.mean_samples_per_item:.1f}).")
    lines.append("")
    lines.append("Lowest model/persona consensus cells:")
    for _, r in by_model_consensus.sort_values("mean_majority_agreement").head(8).iterrows():
        lines.append(f"- {r.setting} / {r.model} / {r.response_type}: {pct(r.mean_majority_agreement)} majority agreement, samples/item={r.mean_samples_per_item:.1f}.")

    (OUTDIR / "model_only_section5_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred = load_predictions()
    pred.to_csv(OUTDIR / "model_predictions_combined_wide.csv", index=False)
    pred_long = long_predictions(pred)
    pred_long.to_csv(OUTDIR / "model_predictions_combined_long.csv", index=False)
    gaps = summarize_gaps(pred_long)
    summarize_agreement(pred, pred_long)
    reg = regression_drop_one(pred_long)
    write_findings(gaps, reg)
    manifest = {
        "neutral_files": NEUTRAL_FILES,
        "persona_files": PERSONA_FILES,
        "n_wide_rows": int(len(pred)),
        "n_long_valid_rows": int(len(pred_long)),
        "outputs": sorted(p.name for p in OUTDIR.glob("*.csv")) + ["model_only_section5_findings.md"],
    }
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved model-only Section 5 analysis to {OUTDIR}")


if __name__ == "__main__":
    main()
