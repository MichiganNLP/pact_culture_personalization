#!/usr/bin/env python3
"""Appendix model-behavior analyses for demographic ablations and preference type.

Outputs are written to newest_results_2026_05_17/appendix_model_behavior.
The preference-type labels are keyword-based trace-analysis aids, not supervised
annotations.
"""

from collections import defaultdict
from pathlib import Path
import math
import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd

FOLLOW = "FOLLOW_CULTURE"
ALLOW = "ALLOW_PREFERENCE"
VALID = {FOLLOW, ALLOW}
OUTDIR = Path("newest_results_2026_05_17/appendix_model_behavior")

MODEL_FILES = [
    ("Llama", Path("res/normad_llm_evaluations_llama3_8b_last_normalsys.csv")),
    ("Qwen", Path("res/normad_llm_evaluations_qwen2514b_last_normalsys.csv")),
    ("DeepSeek", Path("res/normad_deepseek7b_last_normalsys.csv")),
    ("Mistral", Path("res/normad_mistral7b.csv")),
    ("OLMo", Path("res/normad_olmo7b_last_normalsys.csv")),
    ("GPT-4o", Path("res/normad_llm_evaluations_gpt4o_1000.csv")),
]

USECOLS = [
    "source_row",
    "base_country",
    "scenario_type",
    "actor_gender",
    "actor_age",
    "receiver_gender",
    "receiver_age",
    "situation",
    "cultural_expectation",
    "personal_preference",
    "eval_demo_config",
    "eval_pref_config",
    "eval_decision",
]

CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    (
        "health_diet_safety",
        [
            "allerg", "diet", "dietary", "vegetarian", "vegan", "halal", "kosher",
            "religious", "health", "medical", "medicine", "sick", "painful", "injury",
            "smok", "cigarette", "alcohol", "caffeine", "safety", "safe", "risk",
            "hygiene", "sanitary", "germs", "unwell",
        ],
    ),
    (
        "privacy_boundary_values",
        [
            "privacy", "private", "personal space", "boundar", "consent", "values",
            "belief", "believe", "principle", "independent", "independently", "autonomy",
            "choose their own", "choose his own", "choose her own", "openly disagree",
            "free to", "uncomfortable", "not comfortable", "decline", "refuse",
            "avoid personal", "personal topics", "earnings", "salary", "cash", "relationship",
        ],
    ),
    (
        "convenience_efficiency",
        [
            "quick", "quickly", "faster", "fast", "save time", "saving time", "efficient",
            "efficiency", "convenience", "convenient", "ease", "easy", "simpl", "hassle",
            "avoid back-and-forth", "immediately", "directly", "straight", "short", "briefly",
            "get to the destination", "flexibility", "gift card", "practical",
        ],
    ),
    (
        "comfort_habit",
        [
            "comfort", "comfortable", "used to", "habit", "usually", "typically",
            "prefers to sit", "stays seated", "standing", "left hand", "one hand",
            "shoes on", "covered", "uncovered", "casual", "relaxed", "informal",
            "own pace", "ready", "feel", "feels", "natural",
        ],
    ),
    (
        "taste_style_identity",
        [
            "style", "stylish", "appearance", "dress", "clothes", "regular clothes",
            "mask", "cap", "hijab", "scarf", "hair", "tattoo", "music", "gift",
            "book", "symbolic", "meaningful", "useful", "kitchen", "personal taste",
            "own cultural background", "same customs", "traditions visible",
        ],
    ),
    (
        "social_directness_communication",
        [
            "direct", "directly", "ask", "question", "speak", "language", "english",
            "spanish", "formal", "informal", "first name", "honorific", "hello", "bye",
            "conversation", "chat", "gesture", "neutral expression", "eye contact", "respond",
        ],
    ),
]

STRONG_TYPES = {"health_diet_safety", "privacy_boundary_values"}
LOW_TYPES = {"convenience_efficiency", "comfort_habit", "taste_style_identity", "social_directness_communication"}


def norm_decision(x: object) -> str:
    return str(x or "").strip().upper()


def classify_preference(text: object) -> Tuple[str, str]:
    s = re.sub(r"\s+", " ", str(text or "").lower())
    scores: Dict[str, int] = {}
    for category, keywords in CATEGORY_RULES:
        scores[category] = sum(1 for kw in keywords if kw in s)
    best = max(scores.items(), key=lambda kv: kv[1])
    category = best[0] if best[1] > 0 else "other"
    if category in STRONG_TYPES:
        strength = "strong"
    elif category in LOW_TYPES:
        strength = "low_stakes"
    else:
        strength = "other"
    return category, strength


def init_counter() -> Dict[str, float]:
    return {"n": 0, "valid_n": 0, "allow": 0, "follow": 0, "unknown": 0}


def add_count(store: Dict[Tuple, Dict[str, float]], key: Tuple, decisions: pd.Series) -> None:
    counts = decisions.map(norm_decision).value_counts(dropna=False)
    n = int(counts.sum())
    allow = int(counts.get(ALLOW, 0))
    follow = int(counts.get(FOLLOW, 0))
    valid_n = allow + follow
    d = store.setdefault(key, init_counter())
    d["n"] += n
    d["valid_n"] += valid_n
    d["allow"] += allow
    d["follow"] += follow
    d["unknown"] += n - valid_n


def rows_from_store(store: Dict[Tuple, Dict[str, float]], names: List[str]) -> pd.DataFrame:
    rows = []
    for key, vals in store.items():
        row = dict(zip(names, key))
        row.update(vals)
        row["allow_rate_valid"] = vals["allow"] / vals["valid_n"] if vals["valid_n"] else math.nan
        row["culture_rate_valid"] = vals["follow"] / vals["valid_n"] if vals["valid_n"] else math.nan
        row["unknown_rate"] = vals["unknown"] / vals["n"] if vals["n"] else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def model_meta(model: str, path: Path) -> Dict[str, str]:
    return {"model": model, "file": str(path)}


def process() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    demo = {}
    demo_pref = {}
    pref_type = {}
    pref_type_config = {}
    pref_strength = {}
    pref_strength_config = {}
    type_examples = {}
    inventory = []

    for model, path in MODEL_FILES:
        if not path.exists():
            continue
        print(f"Reading {model}: {path}")
        n_rows = 0
        for chunk in pd.read_csv(path, usecols=lambda c: c in USECOLS, chunksize=100000):
            missing = [c for c in USECOLS if c not in chunk.columns]
            if missing:
                raise ValueError(f"{path} missing columns {missing}")
            n_rows += len(chunk)
            chunk["pref_type"], chunk["pref_strength"] = zip(*chunk["personal_preference"].map(classify_preference))

            for keys, sub in chunk.groupby(["eval_demo_config"], dropna=False):
                add_count(demo, (model, str(keys)), sub["eval_decision"])
            for keys, sub in chunk.groupby(["eval_demo_config", "eval_pref_config"], dropna=False):
                add_count(demo_pref, (model, str(keys[0]), str(keys[1])), sub["eval_decision"])

            # Preference-type appendix uses full_demo to avoid demographic-prompt confounding.
            full = chunk[chunk["eval_demo_config"].eq("full_demo")].copy()
            if not full.empty:
                for keys, sub in full.groupby(["pref_type"], dropna=False):
                    add_count(pref_type, (model, str(keys)), sub["eval_decision"])
                for keys, sub in full.groupby(["pref_type", "eval_pref_config"], dropna=False):
                    add_count(pref_type_config, (model, str(keys[0]), str(keys[1])), sub["eval_decision"])
                for keys, sub in full.groupby(["pref_strength"], dropna=False):
                    add_count(pref_strength, (model, str(keys)), sub["eval_decision"])
                for keys, sub in full.groupby(["pref_strength", "eval_pref_config"], dropna=False):
                    add_count(pref_strength_config, (model, str(keys[0]), str(keys[1])), sub["eval_decision"])

                for category, sub in full.groupby("pref_type"):
                    bucket = type_examples.setdefault(str(category), [])
                    if len(bucket) < 8:
                        cols = ["personal_preference", "cultural_expectation", "situation", "base_country", "scenario_type"]
                        for rec in sub[cols].drop_duplicates().head(8 - len(bucket)).to_dict("records"):
                            rec["pref_type"] = str(category)
                            rec["pref_strength"] = classify_preference(rec["personal_preference"])[1]
                            bucket.append(rec)
        inventory.append({**model_meta(model, path), "rows_read": n_rows})

    demo_df = rows_from_store(demo, ["model", "eval_demo_config"]).sort_values(["model", "eval_demo_config"])
    demo_pref_df = rows_from_store(demo_pref, ["model", "eval_demo_config", "eval_pref_config"]).sort_values(["model", "eval_pref_config", "eval_demo_config"])
    type_df = rows_from_store(pref_type, ["model", "pref_type"]).sort_values(["pref_type", "model"])
    type_config_df = rows_from_store(pref_type_config, ["model", "pref_type", "eval_pref_config"]).sort_values(["pref_type", "eval_pref_config", "model"])
    strength_df = rows_from_store(pref_strength, ["model", "pref_strength"]).sort_values(["pref_strength", "model"])
    strength_config_df = rows_from_store(pref_strength_config, ["model", "pref_strength", "eval_pref_config"]).sort_values(["pref_strength", "eval_pref_config", "model"])

    # Deltas against no_demo where available.
    base = demo_df[demo_df["eval_demo_config"].eq("no_demo")][["model", "allow_rate_valid"]].rename(columns={"allow_rate_valid": "no_demo_allow_rate"})
    demo_delta = demo_df.merge(base, on="model", how="left")
    demo_delta["delta_vs_no_demo_pp"] = (demo_delta["allow_rate_valid"] - demo_delta["no_demo_allow_rate"]) * 100

    demo_pref_base = demo_pref_df[demo_pref_df["eval_demo_config"].eq("no_demo")][["model", "eval_pref_config", "allow_rate_valid"]].rename(columns={"allow_rate_valid": "no_demo_allow_rate"})
    demo_pref_delta = demo_pref_df.merge(demo_pref_base, on=["model", "eval_pref_config"], how="left")
    demo_pref_delta["delta_vs_no_demo_pp"] = (demo_pref_delta["allow_rate_valid"] - demo_pref_delta["no_demo_allow_rate"]) * 100

    examples = pd.DataFrame([rec for rows in type_examples.values() for rec in rows])

    pd.DataFrame(inventory).to_csv(OUTDIR / "model_behavior_appendix_inventory.csv", index=False)
    demo_df.to_csv(OUTDIR / "demo_ablation_model_rates.csv", index=False)
    demo_delta.to_csv(OUTDIR / "demo_ablation_model_deltas_vs_no_demo.csv", index=False)
    demo_pref_df.to_csv(OUTDIR / "demo_ablation_by_pref_config_rates.csv", index=False)
    demo_pref_delta.to_csv(OUTDIR / "demo_ablation_by_pref_config_deltas_vs_no_demo.csv", index=False)
    type_df.to_csv(OUTDIR / "preference_type_model_rates_full_demo.csv", index=False)
    type_config_df.to_csv(OUTDIR / "preference_type_by_pref_config_model_rates_full_demo.csv", index=False)
    strength_df.to_csv(OUTDIR / "preference_strength_model_rates_full_demo.csv", index=False)
    strength_config_df.to_csv(OUTDIR / "preference_strength_by_pref_config_model_rates_full_demo.csv", index=False)
    examples.to_csv(OUTDIR / "preference_type_examples.csv", index=False)

    write_findings(demo_delta, demo_pref_delta, type_df, type_config_df, strength_df, strength_config_df)
    print(f"Wrote appendix model-behavior analysis to {OUTDIR}")


def pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{100*x:.1f}%"


def pp(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:+.1f} pp"


def compact_table(df: pd.DataFrame, cols: List[str], n: int = 20) -> str:
    out = df[cols].head(n).copy()
    for col in out.columns:
        if "rate" in col and out[col].dtype.kind in "fc":
            out[col] = out[col].map(pct)
        if col.endswith("_pp"):
            out[col] = out[col].map(pp)
    headers = list(out.columns)
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in out.iterrows():
        vals = [str(row[col]) for col in headers]
        vals = [v.replace("|", "\|").replace("\n", "<br>") for v in vals]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_findings(demo_delta, demo_pref_delta, type_df, type_config_df, strength_df, strength_config_df) -> None:
    lines = []
    lines.append("# Appendix Model Behavior: Demographic Ablations and Preference Types\n")
    lines.append("Preference-type labels are keyword-based trace-analysis aids, not supervised annotations. Preference-type results use `full_demo` rows only to avoid confounding with demographic prompt ablations.\n")

    lines.append("## Demo Ablation: Overall\n")
    summary = demo_delta.copy()
    summary["allow_rate_valid_pct"] = summary["allow_rate_valid"]
    lines.append(compact_table(summary.sort_values(["model", "eval_demo_config"]), ["model", "eval_demo_config", "valid_n", "allow_rate_valid_pct", "delta_vs_no_demo_pp"], 40))
    lines.append("")

    lines.append("## Largest Demo-Ablation Shifts by Preference Config\n")
    shifts = demo_pref_delta[~demo_pref_delta["eval_demo_config"].eq("no_demo")].copy()
    shifts["abs_delta"] = shifts["delta_vs_no_demo_pp"].abs()
    lines.append(compact_table(shifts.sort_values("abs_delta", ascending=False), ["model", "eval_pref_config", "eval_demo_config", "allow_rate_valid", "delta_vs_no_demo_pp"], 30))
    lines.append("")

    lines.append("## Preference Strength\n")
    lines.append(compact_table(strength_df.sort_values(["pref_strength", "model"]), ["model", "pref_strength", "valid_n", "allow_rate_valid"], 40))
    lines.append("")

    # Average by strength across models, weighted by valid_n.
    strength_avg = []
    for strength, g in strength_df.groupby("pref_strength"):
        rate = g["allow"].sum() / g["valid_n"].sum() if g["valid_n"].sum() else math.nan
        strength_avg.append({"pref_strength": strength, "valid_n": int(g["valid_n"].sum()), "allow_rate_valid": rate})
    lines.append("## Preference Strength: Weighted Average Across Models\n")
    lines.append(compact_table(pd.DataFrame(strength_avg).sort_values("allow_rate_valid", ascending=False), ["pref_strength", "valid_n", "allow_rate_valid"], 20))
    lines.append("")

    lines.append("## Preference Type: Model-Wise Rates\n")
    lines.append(compact_table(type_df.sort_values(["pref_type", "model"]), ["model", "pref_type", "valid_n", "allow_rate_valid"], 80))
    lines.append("")

    type_avg = []
    for pref_type, g in type_df.groupby("pref_type"):
        rate = g["allow"].sum() / g["valid_n"].sum() if g["valid_n"].sum() else math.nan
        type_avg.append({"pref_type": pref_type, "valid_n": int(g["valid_n"].sum()), "allow_rate_valid": rate})
    lines.append("## Preference Type: Weighted Average Across Models\n")
    lines.append(compact_table(pd.DataFrame(type_avg).sort_values("allow_rate_valid", ascending=False), ["pref_type", "valid_n", "allow_rate_valid"], 20))
    lines.append("")

    lines.append("## Preference Type x Preference Configuration: Largest Allowance Cells\n")
    lines.append(compact_table(type_config_df.sort_values("allow_rate_valid", ascending=False), ["model", "pref_type", "eval_pref_config", "valid_n", "allow_rate_valid"], 40))
    lines.append("")

    lines.append("## Suggested Takeaways\n")
    lines.append("- Demographic ablation effects are generally smaller than preference-configuration effects; the largest shifts concentrate in Llama/GPT-style models and in `both_prefs` cells.\n")
    lines.append("- Strong preference types are not uniformly more permissive: models still follow culture when a strong preference conflicts with respect, hospitality, or family obligations.\n")
    lines.append("- Low-stakes convenience/comfort preferences can receive high allowance from flexible models, especially under `both_prefs`, but Mistral and some OLMo/Qwen settings remain culture-binding.\n")
    lines.append("- Treat the preference-type categories as qualitative trace groupings; appendix language should call them keyword-based thematic bins rather than gold labels.\n")

    (OUTDIR / "appendix_model_behavior_findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    process()
