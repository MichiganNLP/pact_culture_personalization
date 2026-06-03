#!/usr/bin/env python3
"""Summarize current CultureAtlas/NormAD balance and no-balance eval outputs.

The outputs support the paper/slides metrics that can be measured from the
existing model-eval CSVs:
  - FOLLOW_CULTURE / ALLOW_PREFERENCE / UNKNOWN rates
  - breakdowns by prompt preference config, demographics, distance, country
  - paired base-vs-instruct and balance-vs-nobalance flip rates when matching
    files are present.
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_SCRATCH = Path("/scratch/mihalcea_owned_root/mihalcea_owned1/anganab")
DEFAULT_OUTPUT = Path("eval_metrics_current")
DECISION_COL = "eval_decision"
FOLLOW = "FOLLOW_CULTURE"
ALLOW = "ALLOW_PREFERENCE"
UNKNOWN = "UNKNOWN"
VALID_DECISIONS = {FOLLOW, ALLOW}

BREAKDOWN_GROUPS = {
    "by_pref_config": ["eval_pref_config"],
    "by_demo_config": ["eval_demo_config"],
    "by_scenario_type": ["scenario_type"],
    "by_base_country": ["base_country"],
    "by_actor_gender": ["actor_gender"],
    "by_actor_age": ["actor_age"],
    "by_receiver_gender": ["receiver_gender"],
    "by_receiver_age": ["receiver_age"],
    "by_pref_and_scenario": ["eval_pref_config", "scenario_type"],
    "by_pref_and_country": ["eval_pref_config", "base_country"],
    "by_actor_receiver_gender": ["actor_gender", "receiver_gender"],
    "by_actor_receiver_age": ["actor_age", "receiver_age"],
}

PAIR_KEYS = [
    "source_row",
    "eval_demo_config",
    "eval_pref_config",
    "scenario_type",
    "actor_country",
    "receiver_country",
    "actor_gender",
    "actor_age",
    "receiver_gender",
    "receiver_age",
]

SECTION4_COLUMNS = sorted(
    set(
        [DECISION_COL, "source_row", "eval_model", "eval_backend"]
        + [col for cols in BREAKDOWN_GROUPS.values() for col in cols]
        + PAIR_KEYS
    )
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "unknown"


def norm_decision(series: pd.Series) -> pd.Series:
    return series.fillna(UNKNOWN).astype(str).str.upper().str.strip()


def classify_dataset(path: Path) -> Optional[str]:
    low = str(path).lower()
    if "cultureatlas" in low or "culture_atlas" in low or "culture_persona_atlas" in low:
        return "cultureatlas"
    if "normad" in low:
        return "normad"
    return None


def classify_prompt(path: Path) -> str:
    name = path.name.lower()
    if "nobalance" in name or "no_balance" in name or "normalsys" in name:
        return "nobalance"
    if "balance" in name:
        return "balance"
    return "nobalance"


def classify_stage(path: Path, model: str) -> str:
    low = f"{path.name} {model}".lower()
    if "base" in low and "instruct" not in low and "chat" not in low:
        return "base"
    if any(x in low for x in ["instruct", "chat", "gpt4o", "gpt-4o"]):
        return "instruct"
    if "last_normalsys" in low and any(x in low for x in ["llama", "qwen"]):
        return "instruct"
    return "unknown"


def classify_family(path: Path, model: str) -> str:
    low = f"{path.name} {model}".lower()
    if "llama" in low:
        return "llama"
    if "qwen" in low:
        return "qwen"
    if "mistral" in low:
        return "mistral"
    if "deepseek" in low:
        return "deepseek"
    if "olmo" in low:
        return "olmo"
    if "gpt" in low:
        return "gpt"
    return "unknown"


def is_obsolete_partial(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ["40k", "80k", "40000", "gpt4o_1000", "gpt4o_mini"])


def candidate_csvs(repo: Path, scratch: Path, current_only: bool = True) -> List[Path]:
    roots = [
        repo / "culture_atlas",
        repo / "contextualized_outputs_new",
        repo / "res",
        scratch / "culture_persona_atlas",
    ]
    seen = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if path in seen:
                continue
            seen.add(path)
            dataset = classify_dataset(path)
            if dataset is None:
                continue
            if current_only and is_obsolete_partial(path):
                continue
            try:
                cols = pd.read_csv(path, nrows=0).columns
            except Exception:
                continue
            if DECISION_COL in cols:
                out.append(path)
    return sorted(out, key=lambda p: str(p))


def file_meta(path: Path) -> Dict[str, str]:
    try:
        one = pd.read_csv(path, nrows=1)
    except Exception:
        one = pd.DataFrame()
    model = ""
    backend = ""
    if not one.empty:
        model = str(one.get("eval_model", pd.Series([""])).iloc[0])
        backend = str(one.get("eval_backend", pd.Series([""])).iloc[0])
    dataset = classify_dataset(path) or "unknown"
    return {
        "dataset": dataset,
        "prompt_condition": classify_prompt(path),
        "model_stage": classify_stage(path, model),
        "model_family": classify_family(path, model),
        "eval_model": model,
        "eval_backend": backend,
        "path": str(path),
        "file": path.name,
    }


def summarize_frame(df: pd.DataFrame) -> Dict[str, float]:
    decisions = norm_decision(df[DECISION_COL])
    valid = decisions.isin(VALID_DECISIONS)
    n = len(decisions)
    n_valid = int(valid.sum())
    follow = int((decisions == FOLLOW).sum())
    allow = int((decisions == ALLOW).sum())
    unknown = int((~valid).sum())
    source_items = int(df["source_row"].nunique()) if "source_row" in df.columns else math.nan
    return {
        "n_rows": n,
        "n_source_items": source_items,
        "n_valid": n_valid,
        "n_follow_culture": follow,
        "n_allow_preference": allow,
        "n_unknown_or_other": unknown,
        "culture_rate_valid": follow / n_valid if n_valid else math.nan,
        "allow_rate_valid": allow / n_valid if n_valid else math.nan,
        "unknown_rate_all": unknown / n if n else math.nan,
    }


def summarize_group(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    rows = []
    for keys, sub in df.groupby(present, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(present, keys))
        row.update(summarize_frame(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def decision_entropy(values: Iterable[str]) -> float:
    vals = [v for v in values if v in VALID_DECISIONS]
    if not vals:
        return math.nan
    total = len(vals)
    return -sum((n / total) * math.log2(n / total) for n in Counter(vals).values())


def variability_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if "source_row" not in df.columns:
        return {
            "mean_source_decision_entropy": math.nan,
            "source_decision_conflict_rate": math.nan,
        }
    decisions = norm_decision(df[DECISION_COL])
    tmp = df[["source_row"]].copy()
    tmp["decision"] = decisions
    entropies = tmp.groupby("source_row")["decision"].apply(decision_entropy)
    valid_entropies = entropies.dropna()
    return {
        "mean_source_decision_entropy": float(valid_entropies.mean()) if len(valid_entropies) else math.nan,
        "source_decision_conflict_rate": float((valid_entropies > 0).mean()) if len(valid_entropies) else math.nan,
    }


def load_decisions(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    meta = file_meta(path)
    available = set(pd.read_csv(path, nrows=0).columns)
    usecols = [col for col in SECTION4_COLUMNS if col in available]
    df = pd.read_csv(path, usecols=usecols)
    df[DECISION_COL] = norm_decision(df[DECISION_COL])
    return df, meta


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def add_meta(df: pd.DataFrame, meta: Dict[str, str]) -> pd.DataFrame:
    for key, value in reversed(list(meta.items())):
        if key != "path":
            df.insert(0, key, value)
    df.insert(0, "path", meta["path"])
    return df


def build_manifest(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        meta = file_meta(path)
        rows.append(meta)
    return pd.DataFrame(rows)


def paired_flip_summary(left: pd.DataFrame, right: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    keys = [c for c in PAIR_KEYS if c in left.columns and c in right.columns]
    if not keys:
        return pd.DataFrame()
    left_small = left[keys + [DECISION_COL]].rename(columns={DECISION_COL: "left_decision"})
    right_small = right[keys + [DECISION_COL]].rename(columns={DECISION_COL: "right_decision"})
    paired = left_small.merge(right_small, on=keys, how="inner")
    paired = paired[
        paired["left_decision"].isin(VALID_DECISIONS) & paired["right_decision"].isin(VALID_DECISIONS)
    ].copy()
    if paired.empty:
        return pd.DataFrame()
    paired["flip_pos"] = (paired["left_decision"] == FOLLOW) & (paired["right_decision"] == ALLOW)
    paired["flip_neg"] = (paired["left_decision"] == ALLOW) & (paired["right_decision"] == FOLLOW)
    paired["flip_zero"] = paired["left_decision"] == paired["right_decision"]
    paired["left_allow"] = paired["left_decision"] == ALLOW
    paired["right_allow"] = paired["right_decision"] == ALLOW
    present_groups = [c for c in group_cols if c in paired.columns]
    if present_groups:
        iterator = paired.groupby(present_groups, dropna=False)
    else:
        iterator = [((), paired)]
    rows = []
    for keys_value, sub in iterator:
        if present_groups and not isinstance(keys_value, tuple):
            keys_value = (keys_value,)
        row = dict(zip(present_groups, keys_value)) if present_groups else {}
        row.update(
            {
                "n_paired": len(sub),
                "left_allow_rate": float(sub["left_allow"].mean()),
                "right_allow_rate": float(sub["right_allow"].mean()),
                "allow_rate_delta_right_minus_left": float(sub["right_allow"].mean() - sub["left_allow"].mean()),
                "flip_pos_culture_to_preference": float(sub["flip_pos"].mean()),
                "flip_neg_preference_to_culture": float(sub["flip_neg"].mean()),
                "flip_zero_same_decision": float(sub["flip_zero"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pair_files(records: Sequence[Tuple[Path, Dict[str, str]]], pair_type: str) -> List[Tuple[Path, Path, Dict[str, str]]]:
    pairs = []
    if pair_type == "stage":
        for base_path, base_meta in records:
            if base_meta["model_stage"] != "base":
                continue
            for inst_path, inst_meta in records:
                if inst_meta["model_stage"] != "instruct":
                    continue
                if (
                    base_meta["dataset"] == inst_meta["dataset"]
                    and base_meta["prompt_condition"] == inst_meta["prompt_condition"]
                    and base_meta["model_family"] == inst_meta["model_family"]
                ):
                    pairs.append((base_path, inst_path, {"pair_type": "base_vs_instruct", **base_meta}))
    elif pair_type == "prompt":
        for nb_path, nb_meta in records:
            if nb_meta["prompt_condition"] != "nobalance":
                continue
            for bal_path, bal_meta in records:
                if bal_meta["prompt_condition"] != "balance":
                    continue
                if (
                    nb_meta["dataset"] == bal_meta["dataset"]
                    and nb_meta["model_stage"] == bal_meta["model_stage"]
                    and nb_meta["model_family"] == bal_meta["model_family"]
                ):
                    pairs.append((nb_path, bal_path, {"pair_type": "nobalance_vs_balance", **nb_meta}))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--include-partials",
        action="store_true",
        help="Include older partial/probe files such as 40k, 80k, gpt4o_1000, and 40000 outputs.",
    )
    args = ap.parse_args()

    paths = candidate_csvs(args.repo, args.scratch, current_only=not args.include_partials)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(paths)
    write_csv(manifest, outdir / "manifest.csv")

    overall_rows = []
    variability_rows = []
    breakdowns: Dict[str, List[pd.DataFrame]] = {name: [] for name in BREAKDOWN_GROUPS}
    records: List[Tuple[Path, Dict[str, str]]] = []

    for path in paths:
        df, meta = load_decisions(path)
        records.append((path, meta))
        overall_rows.append({**meta, **summarize_frame(df), **variability_metrics(df)})
        variability_rows.append({**meta, **variability_metrics(df)})
        dataset_dir = outdir / meta["dataset"]
        for name, cols in BREAKDOWN_GROUPS.items():
            grouped = summarize_group(df, cols)
            if grouped.empty:
                continue
            grouped = add_meta(grouped, meta.copy())
            breakdowns[name].append(grouped)
            write_csv(grouped, dataset_dir / name / f"{slug(path.stem)}.csv")

    write_csv(pd.DataFrame(overall_rows), outdir / "overall_metrics.csv")
    write_csv(pd.DataFrame(variability_rows), outdir / "uncertainty_proxy_by_file.csv")
    for name, frames in breakdowns.items():
        if frames:
            write_csv(pd.concat(frames, ignore_index=True), outdir / f"{name}.csv")

    pair_rows = []
    pair_group_outputs: Dict[str, List[pd.DataFrame]] = {
        "overall": [],
        "by_pref_config": [],
        "by_scenario_type": [],
        "by_base_country": [],
        "by_pref_and_scenario": [],
    }
    for pair_type in ["stage", "prompt"]:
        for left_path, right_path, pair_meta in pair_files(records, pair_type):
            left, left_meta = load_decisions(left_path)
            right, right_meta = load_decisions(right_path)
            pair_id = f"{slug(pair_meta['pair_type'])}__{slug(pair_meta['dataset'])}__{slug(pair_meta['prompt_condition'])}__{slug(pair_meta['model_family'])}"
            groups = {
                "overall": [],
                "by_pref_config": ["eval_pref_config"],
                "by_scenario_type": ["scenario_type"],
                "by_base_country": ["base_country"],
                "by_pref_and_scenario": ["eval_pref_config", "scenario_type"],
            }
            for group_name, cols in groups.items():
                summary = paired_flip_summary(left, right, cols)
                if summary.empty:
                    continue
                summary.insert(0, "right_path", str(right_path))
                summary.insert(0, "left_path", str(left_path))
                summary.insert(0, "right_model_stage", right_meta["model_stage"])
                summary.insert(0, "left_model_stage", left_meta["model_stage"])
                summary.insert(0, "right_prompt_condition", right_meta["prompt_condition"])
                summary.insert(0, "left_prompt_condition", left_meta["prompt_condition"])
                summary.insert(0, "model_family", pair_meta["model_family"])
                summary.insert(0, "dataset", pair_meta["dataset"])
                summary.insert(0, "pair_type", pair_meta["pair_type"])
                pair_group_outputs[group_name].append(summary)
                write_csv(summary, outdir / "paired" / group_name / f"{pair_id}.csv")
                if group_name == "overall":
                    pair_rows.extend(summary.to_dict("records"))

    write_csv(pd.DataFrame(pair_rows), outdir / "paired_flip_overall.csv")
    for group_name, frames in pair_group_outputs.items():
        if frames:
            write_csv(pd.concat(frames, ignore_index=True), outdir / f"paired_flip_{group_name}.csv")

    print(f"Discovered {len(paths)} eval CSVs")
    print(f"Wrote metrics to {outdir}")


if __name__ == "__main__":
    main()
