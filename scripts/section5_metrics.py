#!/usr/bin/env python3
"""Compute Section 5 human and human-model alignment metrics.

The script assumes Choice A is the culture-following option and Choice B is the
personal-preference option, matching the Prolific survey construction.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


CHOICE_TO_NUM = {"A": 1.0, "B": 0.0}


COUNTRY_RELATIONS = {
    "Brazil": {
        "same": {"Brazil"},
        "close": {"Myanmar", "Singapore"},
        "far": set(),
    },
    "India": {
        "same": {"India"},
        "close": {"Bangladesh", "Nepal", "Pakistan"},
        "far": {"Canada", "United States of America"},
    },
    "South Africa": {
        "same": {"South Africa"},
        "close": {"Zimbabwe", "Ethiopia"},
        "far": {"Argentina", "Bangladesh", "Iran"},
    },
    "United Kingdom": {
        "same": {"United Kingdom"},
        "close": {"Ireland"},
        "far": {"Argentina", "Bangladesh", "Iran"},
    },
    "United States of America": {
        "same": {"United States of America"},
        "close": {"Canada", "Mexico"},
        "far": {"Afghanistan", "Bangladesh", "Pakistan"},
    },
}


THEME_KEYWORDS = [
    ("food_hospitality", ["food", "meal", "dinner", "host", "home", "eat", "restaurant", "sweets"]),
    ("greetings_respect", ["greet", "hand", "elder", "older", "bow", "respect"]),
    ("workplace_education", ["meeting", "colleague", "work", "office", "study", "class", "document"]),
    ("public_etiquette", ["queue", "market", "public", "bus", "train", "line"]),
    ("privacy_helping", ["privacy", "help", "phone", "charger", "personal items", "share"]),
    ("gift_giving", ["gift", "present", "offer"]),
]


def choice_to_num(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    return CHOICE_TO_NUM.get(str(x).strip().upper(), np.nan)


def majority_choice(rate: float) -> Optional[str]:
    if pd.isna(rate):
        return None
    return "A" if rate >= 0.5 else "B"


def majority_agreement(rate: float) -> float:
    if pd.isna(rate):
        return np.nan
    return float(max(rate, 1.0 - rate))


def country_relation(respondent_country: str, scenario_country: str) -> str:
    rels = COUNTRY_RELATIONS.get(str(respondent_country), {})
    for label, countries in rels.items():
        if str(scenario_country) in countries:
            return label
    if str(respondent_country) == str(scenario_country):
        return "same"
    return "far"


def scenario_theme(text: Any) -> str:
    s = "" if pd.isna(text) else str(text).lower()
    for theme, keywords in THEME_KEYWORDS:
        if any(k in s for k in keywords):
            return theme
    return "other"


def add_human_features(human: pd.DataFrame) -> pd.DataFrame:
    human = human.copy()
    human["country_relation"] = [
        country_relation(r, s) for r, s in zip(human["respondent_country"], human["scenario_country"])
    ]
    human["scenario_theme"] = human["scenario"].map(scenario_theme)
    human["human_personal_num"] = human["human_personal"].map(choice_to_num)
    human["human_norm_num"] = human["human_norm"].map(choice_to_num)
    return human


def summarize_human_group(human: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped: Iterable = [((), human)] if not group_cols else human.groupby(list(group_cols), dropna=False)
    for keys, grp in grouped:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        personal_rate = grp["human_personal_num"].mean()
        norm_rate = grp["human_norm_num"].mean()
        row = {
            "grouping": "overall" if not group_cols else "+".join(group_cols),
            "personal_culture_rate": personal_rate,
            "norm_culture_rate": norm_rate,
            "norm_personal_gap": norm_rate - personal_rate,
            "personal_majority_agreement": majority_agreement(personal_rate),
            "norm_majority_agreement": majority_agreement(norm_rate),
            "n_judgments": len(grp),
            "n_respondents": grp["respondent_id"].nunique(),
            "n_items": grp["scenario_item_id"].nunique(),
        }
        for i, col in enumerate(group_cols):
            row[col] = keys[i]
        rows.append(row)
    return pd.DataFrame(rows)


def human_item_profiles(human: pd.DataFrame, response_type: str, group_cols: Sequence[str]) -> pd.DataFrame:
    response_col = "human_personal_num" if response_type == "personal" else "human_norm_num"
    grouped = (
        human.dropna(subset=[response_col])
        .groupby(["scenario_item_id"] + list(group_cols), dropna=False)
        .agg(
            human_culture_rate=(response_col, "mean"),
            n_humans=(response_col, "size"),
            scenario_country=("scenario_country", "first"),
            receiver_demo=("receiver_demo", "first"),
            receiver_age=("receiver_age", "first"),
            receiver_gender=("receiver_gender", "first"),
            scenario_theme=("scenario_theme", "first"),
        )
        .reset_index()
    )
    grouped["human_majority"] = grouped["human_culture_rate"].map(majority_choice)
    grouped["human_majority_agreement"] = grouped["human_culture_rate"].map(majority_agreement)
    grouped["response_type"] = response_type
    return grouped


def read_prediction_files(files: Sequence[str], setting: str) -> pd.DataFrame:
    frames = []
    for file in files:
        path = Path(file)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "persona_setting" in df.columns:
            expected = "no_persona" if setting == "no_persona" else "persona"
            df = df[df["persona_setting"].astype(str).eq(expected)].copy()
        df["setting"] = setting
        df["prediction_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(
        ["setting", "model_tag", "persona_country", "persona_demo", "scenario_item_id"],
        keep="last",
    )
    return out


def prediction_long(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response_type, col in [("personal", "llm_personal"), ("norm", "llm_norm")]:
        tmp = pred.copy()
        tmp["response_type"] = response_type
        tmp["llm_choice"] = tmp[col]
        tmp["llm_culture"] = tmp["llm_choice"].map(choice_to_num)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def summarize_alignment(joined: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    grouped = joined.groupby(list(group_cols), dropna=False)
    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        item_rates = (
            grp.groupby("scenario_item_id", dropna=False)
            .agg(
                model_culture_rate=("llm_culture", "mean"),
                human_culture_rate=("human_culture_rate", "first"),
                human_majority_agreement=("human_majority_agreement", "first"),
            )
            .reset_index()
        )
        row = {
            "n_model_rows": len(grp),
            "n_items": grp["scenario_item_id"].nunique(),
            "n_humans": int(grp["n_humans"].sum()),
            "human_culture_rate": grp["human_culture_rate"].mean(),
            "model_culture_rate": grp["llm_culture"].mean(),
            "signed_culture_rate_gap_model_minus_human": (
                item_rates["model_culture_rate"] - item_rates["human_culture_rate"]
            ).mean(),
            "majority_choice_alignment": (grp["llm_choice"] == grp["human_majority"]).mean(),
            "distributional_mae": (grp["llm_culture"] - grp["human_culture_rate"]).abs().mean(),
            "mean_human_majority_agreement": grp["human_majority_agreement"].mean(),
        }
        for i, col in enumerate(group_cols):
            row[col] = keys[i]
        rows.append(row)
    return pd.DataFrame(rows)


def model_uncertainty(pred_long_df: pd.DataFrame, profiles: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    keys = ["setting", "model_tag", "response_type"] + list(group_cols)
    pred_agree = (
        pred_long_df.dropna(subset=["llm_culture"])
        .groupby(keys + ["scenario_item_id"], dropna=False)
        .agg(model_culture_rate=("llm_culture", "mean"), n_model_samples=("llm_culture", "size"))
        .reset_index()
    )
    pred_agree["model_majority_agreement"] = pred_agree["model_culture_rate"].map(majority_agreement)
    joined = pred_agree.merge(profiles, on=["scenario_item_id", "response_type"], how="inner")
    for keys_val, grp in joined.groupby(keys, dropna=False):
        if not isinstance(keys_val, tuple):
            keys_val = (keys_val,)
        low = grp[grp["human_majority_agreement"] <= grp["human_majority_agreement"].median()]
        corr = grp["human_majority_agreement"].corr(grp["model_majority_agreement"])
        row = {
            "n_items": grp["scenario_item_id"].nunique(),
            "mean_model_majority_agreement": grp["model_majority_agreement"].mean(),
            "mean_model_majority_agreement_low_human_agreement": low["model_majority_agreement"].mean(),
            "mean_human_majority_agreement": grp["human_majority_agreement"].mean(),
            "human_model_agreement_correlation": corr,
            "mean_model_samples_per_item": grp["n_model_samples"].mean(),
        }
        for i, col in enumerate(keys):
            row[col] = keys_val[i]
        rows.append(row)
    return pd.DataFrame(rows)


def human_long_for_regression(human: pd.DataFrame) -> pd.DataFrame:
    personal = human.copy()
    personal["question_type"] = "personal"
    personal["follow_culture"] = personal["human_personal_num"]
    norm = human.copy()
    norm["question_type"] = "norm"
    norm["follow_culture"] = norm["human_norm_num"]
    cols = [
        "respondent_id",
        "scenario_item_id",
        "respondent_country",
        "respondent_age_bucket",
        "respondent_gender",
        "respondent_demo",
        "scenario_country",
        "receiver_age",
        "receiver_gender",
        "receiver_demo",
        "country_relation",
        "scenario_theme",
        "question_type",
        "follow_culture",
    ]
    return pd.concat([personal[cols], norm[cols]], ignore_index=True).dropna(subset=["follow_culture"])


def fit_glm(formula: str, data: pd.DataFrame):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    return smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()


def normal_p_value(z_value: float) -> float:
    import math

    if pd.isna(z_value):
        return np.nan
    return float(math.erfc(abs(float(z_value)) / math.sqrt(2.0)))


def factor_matrix(data: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    vals = data[col].fillna("NA").astype(str)
    return pd.get_dummies(vals, prefix=prefix, drop_first=True, dtype=float)


def interaction_matrix(data: pd.DataFrame, col_a: str, col_b: str, prefix: str) -> pd.DataFrame:
    vals = data[col_a].fillna("NA").astype(str) + ":" + data[col_b].fillna("NA").astype(str)
    return pd.get_dummies(vals, prefix=prefix, drop_first=True, dtype=float)


def build_manual_design(data: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    parts = [pd.DataFrame({"Intercept": np.ones(len(data), dtype=float)}, index=data.index)]
    for term in terms:
        if term == "question_type":
            parts.append(factor_matrix(data, "question_type", "question_type"))
        elif term == "respondent_country":
            parts.append(factor_matrix(data, "respondent_country", "respondent_country"))
        elif term == "respondent_demographics":
            parts.append(factor_matrix(data, "respondent_age_bucket", "respondent_age"))
            parts.append(factor_matrix(data, "respondent_gender", "respondent_gender"))
        elif term == "receiver_demographics":
            parts.append(factor_matrix(data, "receiver_age", "receiver_age"))
            parts.append(factor_matrix(data, "receiver_gender", "receiver_gender"))
        elif term == "country_relation":
            parts.append(factor_matrix(data, "country_relation", "country_relation"))
        elif term == "scenario_country":
            parts.append(factor_matrix(data, "scenario_country", "scenario_country"))
        elif term == "question_x_country":
            parts.append(interaction_matrix(data, "question_type", "respondent_country", "question_x_country"))
        elif term == "question_x_relation":
            parts.append(interaction_matrix(data, "question_type", "country_relation", "question_x_relation"))
        elif term == "participant_x_receiver_demographics":
            parts.append(interaction_matrix(data, "respondent_age_bucket", "receiver_age", "resp_age_x_recv_age"))
            parts.append(interaction_matrix(data, "respondent_gender", "receiver_gender", "resp_gender_x_recv_gender"))
        else:
            raise ValueError(f"Unknown manual regression term: {term}")
    design = pd.concat(parts, axis=1)
    return design.loc[:, ~design.columns.duplicated()]


def fit_manual_logit(data: pd.DataFrame, terms: Sequence[str], ridge: float = 1e-6) -> Dict[str, Any]:
    x_df = build_manual_design(data, terms)
    x = x_df.to_numpy(dtype=float)
    y = data["follow_culture"].to_numpy(dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    ridge_vec = np.full(x.shape[1], ridge, dtype=float)
    ridge_vec[0] = 0.0
    for _ in range(100):
        eta = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-7, None)
        grad = x.T @ (y - p) - ridge_vec * beta
        hess = (x.T * w) @ x + np.diag(ridge_vec)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta_new = beta + step
        if np.max(np.abs(step)) < 1e-7:
            beta = beta_new
            break
        beta = beta_new

    eta = np.clip(x @ beta, -35.0, 35.0)
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    loglik = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    deviance = -2.0 * loglik
    w = np.clip(p * (1.0 - p), 1e-7, None)
    hess = (x.T * w) @ x + np.diag(ridge_vec)
    cov = np.linalg.pinv(hess)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    coef = pd.DataFrame(
        {
            "term": x_df.columns,
            "coef": beta,
            "std_err": se,
            "z_value": z,
            "p_value": [normal_p_value(v) for v in z],
        }
    )
    return {
        "coef": coef,
        "deviance": deviance,
        "df_model": max(int(np.linalg.matrix_rank(x)) - 1, 0),
        "nobs": len(data),
        "terms": list(terms),
    }


def write_manual_regressions(
    data_for_model: Dict[str, pd.DataFrame],
    term_sets: Dict[str, List[str]],
    drop_terms: Dict[str, Dict[str, List[str]]],
    outdir: Path,
) -> None:
    summaries = []
    coef_frames = []
    deviance_rows = []
    for model_name, terms in term_sets.items():
        model_data = data_for_model[model_name]
        try:
            fit = fit_manual_logit(model_data, terms)
        except Exception as exc:
            summaries.append(f"\n=== {model_name} failed ===\n{exc}\n")
            continue
        summaries.append(
            "\n=== {} ===\nManual ridge logistic regression fallback\nnobs={}; df_model={}; deviance={:.6f}\nterms={}\n".format(
                model_name,
                fit["nobs"],
                fit["df_model"],
                fit["deviance"],
                ", ".join(terms),
            )
        )
        coef = fit["coef"].copy()
        coef["model"] = model_name
        coef_frames.append(coef)
        for label, removed_terms in drop_terms[model_name].items():
            reduced_terms = [t for t in terms if t not in removed_terms]
            try:
                reduced = fit_manual_logit(model_data, reduced_terms)
                deviance_rows.append(
                    {
                        "model": model_name,
                        "dropped_factor": label,
                        "full_deviance": fit["deviance"],
                        "reduced_deviance": reduced["deviance"],
                        "drop_one_deviance": reduced["deviance"] - fit["deviance"],
                        "full_df_model": fit["df_model"],
                        "reduced_df_model": reduced["df_model"],
                        "df_difference": fit["df_model"] - reduced["df_model"],
                    }
                )
            except Exception as exc:
                deviance_rows.append(
                    {
                        "model": model_name,
                        "dropped_factor": label,
                        "full_deviance": fit["deviance"],
                        "reduced_deviance": np.nan,
                        "drop_one_deviance": np.nan,
                        "full_df_model": fit["df_model"],
                        "reduced_df_model": np.nan,
                        "df_difference": np.nan,
                        "error": str(exc),
                    }
                )
    (outdir / "human_regression_summary.txt").write_text("\n".join(summaries), encoding="utf-8")
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(outdir / "human_regression_coefficients.csv", index=False)
    if deviance_rows:
        pd.DataFrame(deviance_rows).to_csv(outdir / "human_regression_drop_one_deviance.csv", index=False)
    (outdir / "human_regression_backend.txt").write_text(
        "Used statsmodels GLM Binomial regression backend.\n",
        encoding="utf-8",
    )


def write_regressions(human: pd.DataFrame, outdir: Path) -> None:
    reg = human_long_for_regression(human)
    manual_term_sets = {
        "combined_personal_vs_norm": [
            "question_type",
            "respondent_country",
            "respondent_demographics",
            "receiver_demographics",
            "country_relation",
            "scenario_country",
            "question_x_country",
            "question_x_relation",
            "participant_x_receiver_demographics",
        ],
        "personal_choice_only": [
            "respondent_country",
            "respondent_demographics",
            "receiver_demographics",
            "country_relation",
            "scenario_country",
            "participant_x_receiver_demographics",
        ],
        "norm_judgment_only": [
            "respondent_country",
            "respondent_demographics",
            "receiver_demographics",
            "country_relation",
            "scenario_country",
            "participant_x_receiver_demographics",
        ],
    }
    data_for_model = {
        "combined_personal_vs_norm": reg,
        "personal_choice_only": reg[reg["question_type"] == "personal"].copy(),
        "norm_judgment_only": reg[reg["question_type"] == "norm"].copy(),
    }
    manual_drop_terms = {
        "combined_personal_vs_norm": {
            "scenario_country": ["scenario_country"],
            "participant_country": ["respondent_country"],
            "country_relation": ["country_relation"],
            "question_type": ["question_type"],
            "question_x_country": ["question_x_country"],
            "question_x_relation": ["question_x_relation"],
            "participant_demographics": ["respondent_demographics"],
            "receiver_demographics": ["receiver_demographics"],
            "participant_x_receiver_demographics": ["participant_x_receiver_demographics"],
        },
        "personal_choice_only": {
            "scenario_country": ["scenario_country"],
            "participant_country": ["respondent_country"],
            "country_relation": ["country_relation"],
            "participant_demographics": ["respondent_demographics"],
            "receiver_demographics": ["receiver_demographics"],
            "participant_x_receiver_demographics": ["participant_x_receiver_demographics"],
        },
        "norm_judgment_only": {
            "scenario_country": ["scenario_country"],
            "participant_country": ["respondent_country"],
            "country_relation": ["country_relation"],
            "participant_demographics": ["respondent_demographics"],
            "receiver_demographics": ["receiver_demographics"],
            "participant_x_receiver_demographics": ["participant_x_receiver_demographics"],
        },
    }
    if os.environ.get("SECTION5_USE_STATSMODELS", "").strip() != "1":
        write_manual_regressions(data_for_model, manual_term_sets, manual_drop_terms, outdir)
        (outdir / "human_regression_backend.txt").write_text(
            "Used smaller NumPy ridge-logistic regression with scenario-country fixed effects. "
            "Set SECTION5_USE_STATSMODELS=1 to try the slower statsmodels backend.\n",
            encoding="utf-8",
        )
        return

    try:
        import statsmodels.api as sm  # noqa: F401
    except Exception as exc:
        write_manual_regressions(data_for_model, manual_term_sets, manual_drop_terms, outdir)
        (outdir / "human_regression_backend.txt").write_text(
            f"Used smaller NumPy ridge-logistic fallback because statsmodels failed: {exc}\n",
            encoding="utf-8",
        )
        return

    formulas = {
        "combined_personal_vs_norm": (
            "follow_culture ~ C(question_type) + C(respondent_country) "
            "+ C(respondent_age_bucket) + C(respondent_gender) "
            "+ C(receiver_age) + C(receiver_gender) + C(country_relation) "
            "+ C(scenario_country) + C(question_type):C(respondent_country) "
            "+ C(question_type):C(country_relation) "
            "+ C(respondent_age_bucket):C(receiver_age) "
            "+ C(respondent_gender):C(receiver_gender)"
        ),
        "personal_choice_only": (
            "follow_culture ~ C(respondent_country) + C(respondent_age_bucket) "
            "+ C(respondent_gender) + C(receiver_age) + C(receiver_gender) "
            "+ C(country_relation) + C(scenario_country) "
            "+ C(respondent_age_bucket):C(receiver_age) "
            "+ C(respondent_gender):C(receiver_gender)"
        ),
        "norm_judgment_only": (
            "follow_culture ~ C(respondent_country) + C(respondent_age_bucket) "
            "+ C(respondent_gender) + C(receiver_age) + C(receiver_gender) "
            "+ C(country_relation) + C(scenario_country) "
            "+ C(respondent_age_bucket):C(receiver_age) "
            "+ C(respondent_gender):C(receiver_gender)"
        ),
    }
    data_for_model = {
        "combined_personal_vs_norm": reg,
        "personal_choice_only": reg[reg["question_type"] == "personal"].copy(),
        "norm_judgment_only": reg[reg["question_type"] == "norm"].copy(),
    }
    drop_terms = {
        "combined_personal_vs_norm": {
            "scenario_country": "C(scenario_country)",
            "participant_country": "C(respondent_country)",
            "country_relation": "C(country_relation)",
            "question_type": "C(question_type)",
            "question_x_country": "C(question_type):C(respondent_country)",
            "question_x_relation": "C(question_type):C(country_relation)",
            "participant_demographics": "C(respondent_age_bucket) + C(respondent_gender)",
            "receiver_demographics": "C(receiver_age) + C(receiver_gender)",
            "participant_x_receiver_demographics": (
                "C(respondent_age_bucket):C(receiver_age) + "
                "C(respondent_gender):C(receiver_gender)"
            ),
        },
        "personal_choice_only": {
            "scenario_country": "C(scenario_country)",
            "participant_country": "C(respondent_country)",
            "country_relation": "C(country_relation)",
            "participant_demographics": "C(respondent_age_bucket) + C(respondent_gender)",
            "receiver_demographics": "C(receiver_age) + C(receiver_gender)",
            "participant_x_receiver_demographics": (
                "C(respondent_age_bucket):C(receiver_age) + "
                "C(respondent_gender):C(receiver_gender)"
            ),
        },
        "norm_judgment_only": {
            "scenario_country": "C(scenario_country)",
            "participant_country": "C(respondent_country)",
            "country_relation": "C(country_relation)",
            "participant_demographics": "C(respondent_age_bucket) + C(respondent_gender)",
            "receiver_demographics": "C(receiver_age) + C(receiver_gender)",
            "participant_x_receiver_demographics": (
                "C(respondent_age_bucket):C(receiver_age) + "
                "C(respondent_gender):C(receiver_gender)"
            ),
        },
    }

    summaries = []
    coef_frames = []
    deviance_rows = []
    for model_name, formula in formulas.items():
        model_data = data_for_model[model_name]
        try:
            fit = fit_glm(formula, model_data)
        except Exception as exc:
            summaries.append(f"\n=== {model_name} failed ===\n{exc}\n")
            continue
        summaries.append(f"\n=== {model_name} ===\n{fit.summary()}\n")
        coef_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "term": fit.params.index,
                    "coef": fit.params.values,
                    "std_err": fit.bse.values,
                    "z_value": fit.tvalues.values,
                    "p_value": fit.pvalues.values,
                }
            )
        )
        full_deviance = float(fit.deviance)
        for label, removed in drop_terms[model_name].items():
            reduced_formula = formula.replace(" + " + removed, "").replace(removed + " + ", "")
            try:
                reduced = fit_glm(reduced_formula, model_data)
                deviance_rows.append(
                    {
                        "model": model_name,
                        "dropped_factor": label,
                        "full_deviance": full_deviance,
                        "reduced_deviance": float(reduced.deviance),
                        "drop_one_deviance": float(reduced.deviance - full_deviance),
                        "full_df_model": float(fit.df_model),
                        "reduced_df_model": float(reduced.df_model),
                        "df_difference": float(fit.df_model - reduced.df_model),
                    }
                )
            except Exception as exc:
                deviance_rows.append(
                    {
                        "model": model_name,
                        "dropped_factor": label,
                        "full_deviance": full_deviance,
                        "reduced_deviance": np.nan,
                        "drop_one_deviance": np.nan,
                        "full_df_model": float(fit.df_model),
                        "reduced_df_model": np.nan,
                        "df_difference": np.nan,
                        "error": str(exc),
                    }
                )

    (outdir / "human_regression_summary.txt").write_text("\n".join(summaries), encoding="utf-8")
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(outdir / "human_regression_coefficients.csv", index=False)
    if deviance_rows:
        pd.DataFrame(deviance_rows).to_csv(outdir / "human_regression_drop_one_deviance.csv", index=False)


def compute_outputs(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stale_skip = outdir / "human_regression_skipped.txt"
    if stale_skip.exists():
        stale_skip.unlink()
    stale_backend = outdir / "human_regression_backend.txt"
    if stale_backend.exists():
        stale_backend.unlink()

    human = add_human_features(pd.read_csv(args.human_tidy))
    human_groupings = [
        [],
        ["respondent_country"],
        ["country_relation"],
        ["respondent_country", "country_relation"],
        ["scenario_country"],
        ["receiver_age"],
        ["receiver_gender"],
        ["respondent_demo"],
        ["scenario_theme"],
    ]
    human_summary = pd.concat(
        [summarize_human_group(human, cols) for cols in human_groupings],
        ignore_index=True,
    )
    human_summary.to_csv(outdir / "human_response_summary.csv", index=False)
    write_regressions(human, outdir)

    item_profiles = []
    for response_type in ["personal", "norm"]:
        item_profiles.append(human_item_profiles(human, response_type, []))
        item_profiles.append(
            human_item_profiles(
                human,
                response_type,
                ["respondent_country", "respondent_demo"],
            )
        )
    profiles = pd.concat(item_profiles, ignore_index=True)
    profiles.to_csv(outdir / "human_item_profiles.csv", index=False)

    neutral = read_prediction_files(args.neutral_predictions, "no_persona")
    persona = read_prediction_files(args.persona_predictions, "persona")
    pred = pd.concat([neutral, persona], ignore_index=True)
    pred_long_df = prediction_long(pred)

    overall_profiles = profiles[profiles["respondent_country"].isna() & profiles["respondent_demo"].isna()].drop(
        columns=["respondent_country", "respondent_demo"]
    )
    no_persona_join = pred_long_df[pred_long_df["setting"] == "no_persona"].merge(
        overall_profiles,
        on=["scenario_item_id", "response_type"],
        how="inner",
    )
    no_persona_summary = summarize_alignment(
        no_persona_join,
        ["setting", "model_tag", "response_type"],
    ).sort_values(["response_type", "majority_choice_alignment", "distributional_mae"], ascending=[True, False, True])
    no_persona_summary.to_csv(outdir / "alignment_no_persona_summary.csv", index=False)

    subgroup_profiles = profiles.dropna(subset=["respondent_country", "respondent_demo"])
    persona_join = pred_long_df[pred_long_df["setting"] == "persona"].merge(
        subgroup_profiles,
        left_on=["scenario_item_id", "response_type", "persona_country", "persona_demo"],
        right_on=["scenario_item_id", "response_type", "respondent_country", "respondent_demo"],
        how="inner",
    )
    persona_item_path = outdir / "alignment_persona_item_level.csv"
    persona_join.to_csv(persona_item_path, index=False)
    persona_summary = summarize_alignment(
        persona_join,
        ["setting", "model_tag", "response_type", "persona_country", "persona_demo"],
    ).sort_values(["response_type", "model_tag", "persona_country", "persona_demo"])
    persona_summary.to_csv(outdir / "alignment_persona_summary.csv", index=False)

    combined_summary = pd.concat(
        [
            no_persona_summary.assign(persona_country=np.nan, persona_demo=np.nan),
            persona_summary,
        ],
        ignore_index=True,
    )
    combined_summary.to_csv(outdir / "alignment_combined_summary.csv", index=False)

    uncertainty_profiles = profiles[profiles["respondent_country"].isna() & profiles["respondent_demo"].isna()].drop(
        columns=["respondent_country", "respondent_demo"]
    )
    uncertainty = model_uncertainty(pred_long_df, uncertainty_profiles, [])
    uncertainty.to_csv(outdir / "alignment_uncertainty_summary.csv", index=False)

    closest_rows = []
    no_persona_long = pred_long_df[pred_long_df["setting"] == "no_persona"]
    for _, pred_group in no_persona_long.groupby(["model_tag", "response_type"], dropna=False):
        joined = pred_group.merge(
            subgroup_profiles,
            on=["scenario_item_id", "response_type"],
            how="inner",
        )
        closest_rows.append(
            summarize_alignment(
                joined,
                ["setting", "model_tag", "response_type", "respondent_country", "respondent_demo"],
            )
        )
    closest = pd.concat(closest_rows, ignore_index=True)
    closest = closest.sort_values(
        ["model_tag", "response_type", "majority_choice_alignment", "distributional_mae"],
        ascending=[True, True, False, True],
    )
    closest["rank_for_model_response_type"] = closest.groupby(["model_tag", "response_type"]).cumcount() + 1
    closest.to_csv(outdir / "alignment_no_persona_closest_human_subgroups.csv", index=False)

    manifest = {
        "human_tidy": args.human_tidy,
        "neutral_predictions": args.neutral_predictions,
        "persona_predictions": args.persona_predictions,
        "outputs": sorted(p.name for p in outdir.glob("*.csv")),
        "metric_notes": {
            "culture_rate": "Share choosing Choice A, the culture-following option.",
            "norm_personal_gap": "norm_culture_rate - personal_culture_rate.",
            "majority_choice_alignment": "Share of model choices matching the human majority option.",
            "distributional_mae": "Mean absolute difference between model culture choice and human culture rate.",
            "uncertainty_alignment": "Correlation between human majority agreement and model majority agreement across scenario items.",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved Section 5 metrics in {outdir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-tidy", default="human_prolific/tidy_human_responses.csv")
    parser.add_argument("--outdir", default="human_prolific/section5_outputs")
    parser.add_argument(
        "--neutral-predictions",
        nargs="+",
        default=[
            "human_prolific/neutral_out/llm_predictions.csv",
            "human_prolific/mistral_deepseek_neutral_out/llm_predictions.csv",
            "human_prolific/gpt54_mini_persona_no_persona_out/llm_predictions.csv",
        ],
    )
    parser.add_argument(
        "--persona-predictions",
        nargs="+",
        default=[
            "human_prolific/llm_predictions.csv",
            "human_prolific/mistral_deepseek_persona_out/llm_predictions.csv",
            "human_prolific/gpt54_mini_persona_no_persona_out/llm_predictions.csv",
        ],
    )
    return parser


def main() -> None:
    compute_outputs(build_parser().parse_args())


if __name__ == "__main__":
    main()
