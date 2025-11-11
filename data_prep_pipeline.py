"""
Data preparation pipeline for the merged kidney DCI dataset.

This module loads the integrated dataset, standardises binary and categorical
features, engineers temporal intervals, reports missingness, and saves
reproducible artefacts to support downstream modelling.

Usage
-----
python data_prep_pipeline.py [optional_path]

If no path is provided, the script will look for the Google Drive path used in
Colab tutorials. When running locally (e.g., within this repo), it falls back
to the project copy of ``merged_kidney_dci_final.csv``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.options.mode.copy_on_write = True


# -----------------------------------------------------------------------------
# I/O configuration
# -----------------------------------------------------------------------------

GOOGLE_DRIVE_DEFAULT = Path("/content/drive/MyDrive/merged_kidney_dci_final.csv")
REPO_DEFAULT = Path(__file__).resolve().parent / "merged_kidney_dci_final.csv"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate the dataset. Expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def normalise_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns
    if not object_cols.empty:
        replacement_map = {
            "": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
            "NULL": pd.NA,
            "NONE": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
        }
        for col in object_cols:
            series = df[col].astype("string").str.strip()
            df[col] = series.replace(replacement_map)
    return df


def standardise_binary_columns(df: pd.DataFrame, columns: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    binary_map = {
        "Y": 1,
        "YES": 1,
        "TRUE": 1,
        "T": 1,
        "1": 1,
        "N": 0,
        "NO": 0,
        "FALSE": 0,
        "F": 0,
        "0": 0,
    }
    actions = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype("string").str.strip().str.upper()
        mapped = series.map(binary_map)
        df[col] = mapped.astype("Int64")
        actions[col] = "Mapped {Y/N, Yes/No, 1/0} -> Int64 binary"
    return df, actions


def standardise_gender_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    actions = {}

    if "GENDER" in df.columns:
        gender_series = df["GENDER"].astype("string").str.strip().str.upper()
        df["GENDER"] = gender_series.map({"MALE": "Male", "FEMALE": "Female"})
        df["RECIPIENT_SEX_MALE"] = (
            gender_series.map({"MALE": 1, "FEMALE": 0}).astype("Int64")
        )
        actions["GENDER"] = "Standardised to {'Male','Female'} and created RECIPIENT_SEX_MALE"

    if "DONORSEX" in df.columns:
        donor_series = df["DONORSEX"].astype("string").str.strip().str.upper()
        donor_series = donor_series.replace({"MALE": "M", "FEMALE": "F"})
        df["DONORSEX"] = donor_series.replace({"M": "M", "F": "F"})
        df["DONOR_SEX_MALE"] = donor_series.map({"M": 1, "F": 0}).astype("Int64")
        actions["DONORSEX"] = "Collapsed to {'M','F'} and created DONOR_SEX_MALE binary"

    return df, actions


def group_primary_diagnosis(value: str | float) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).lower()
    patterns = [
        ("Diabetes", ["diabetes", "dm"]),
        ("Hypertension", ["hypertens"]),
        ("Polycystic Kidney Disease", ["polycystic"]),
        ("Glomerular Disease", ["glomerul", "fsg", "iga", "mesangio", "membran"]),
        ("Autoimmune/Systemic", ["lupus", "wegener", "polyarteritis", "henoch", "goodpasture", "scleroderma", "sarcoidosis"]),
        ("Congenital/Genetic", ["congenital", "alport", "fabry", "prune", "hypoplasia", "dysplasia", "agenesis", "nephronophthisis"]),
        ("Obstructive/Urologic", ["obstruct", "nephrolith", "urol", "reflux", "oxalate"]),
        ("Toxic/Drug Related", ["toxicity", "drug", "lithium", "calcineurin", "chemotherapy"]),
        ("Infectious/Inflammatory", ["hiv", "pyelonephritis", "post-infect", "nephritis", "hantavirus"]),
        ("Malignancy/Plasma Cell", ["carcinoma", "tumor", "myeloma", "amyloid"]),
    ]
    for label, keywords in patterns:
        if any(keyword in text for keyword in keywords):
            return label
    return "Other/Unknown"


def group_insurance_type(value: str | float) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).lower()
    if any(keyword in text for keyword in ["medicare", "advantra", "security blue", "community blue"]):
        return "Medicare / Medicare Advantage"
    if any(keyword in text for keyword in ["medicaid", "gateway", "forward", "managed"]):
        return "Medicaid / State Plan"
    if any(keyword in text for keyword in ["upmc", "aetna", "cigna", "highmark", "united", "bcbs", "commercial", "employ", "global care"]):
        return "Commercial / Employer"
    if "self" in text:
        return "Self-Pay"
    if any(keyword in text for keyword in ["pending", "other", "hmi", "147", "158"]):
        return "Pending / Other"
    if any(keyword in text for keyword in ["workers", "auto"]):
        return "Workers Comp / Auto"
    return "Pending / Other"


def group_donor_relation(value: str | float) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).lower()
    if "biological" in text:
        if "twin" in text:
            return "Twin (Biological)"
        return "Biological Relative"
    if any(keyword in text for keyword in ["spouse", "partner"]):
        return "Spouse/Partner"
    if "paired" in text:
        return "Paired Donation"
    if any(keyword in text for keyword in ["anonymous", "humanitarian"]):
        return "Anonymous Non-Biological"
    return "Other Non-Biological"


def engineer_temporal_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    actions = {}

    date_columns = [
        "KIDNEY_TRANSPLANTSTX_EVAL_DT9",
        "LISTING_DATE",
        "KIDNEY_TRANSPLANTSTX_SURG_DT9",
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if set(date_columns).issubset(df.columns):
        df["EVAL_TO_LISTING_DAYS"] = (
            df["LISTING_DATE"] - df["KIDNEY_TRANSPLANTSTX_EVAL_DT9"]
        ).dt.days.astype("Int64")
        df["LISTING_TO_TRANSPLANT_DAYS"] = (
            df["KIDNEY_TRANSPLANTSTX_SURG_DT9"] - df["LISTING_DATE"]
        ).dt.days.astype("Int64")
        df["EVAL_TO_TRANSPLANT_DAYS"] = (
            df["KIDNEY_TRANSPLANTSTX_SURG_DT9"] - df["KIDNEY_TRANSPLANTSTX_EVAL_DT9"]
        ).dt.days.astype("Int64")
        actions["temporal"] = "Engineered interval features in days and removed raw dates"
        df = df.drop(columns=date_columns)

    dob_col = "KIDNEY_TRANSPLANTSDONORDOB9"
    if dob_col in df.columns:
        df = df.drop(columns=[dob_col])
        actions[dob_col] = "Dropped; donor age available and DOB not needed after de-identification"

    return df, actions


def apply_groupings(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    actions = {}
    if "PRIMARY_DX" in df.columns:
        df["PRIMARY_DX_GROUP"] = df["PRIMARY_DX"].map(group_primary_diagnosis)
        actions["PRIMARY_DX"] = "Grouped into PRIMARY_DX_GROUP (10 clinically coherent buckets)"
    if "INSURANCE_TYPE" in df.columns:
        df["INSURANCE_PAYOR_GROUP"] = df["INSURANCE_TYPE"].map(group_insurance_type)
        actions["INSURANCE_TYPE"] = "Grouped into INSURANCE_PAYOR_GROUP (payer macro-categories)"
    if "DONOR_RELATION" in df.columns:
        df["DONOR_RELATION_GROUP"] = df["DONOR_RELATION"].map(group_donor_relation)
        actions["DONOR_RELATION"] = "Grouped into DONOR_RELATION_GROUP (5 categories)"
    return df, actions


def identify_constant_and_empty_columns(df: pd.DataFrame) -> List[str]:
    constant_cols = []
    for col in df.columns:
        if df[col].isna().all():
            constant_cols.append(col)
            continue
        uniques = df[col].dropna().unique()
        if len(uniques) <= 1:
            constant_cols.append(col)
    return constant_cols


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    summary = []
    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / total) * 100
        nunique = df[col].nunique(dropna=True)
        summary.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(nunique),
                "missing_count": int(missing),
                "missing_pct": round(missing_pct, 2),
            }
        )
    return pd.DataFrame(summary).sort_values(by="missing_pct", ascending=False)


def high_cardinality_plan(df: pd.DataFrame, threshold: int = 20) -> pd.DataFrame:
    plan_rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique >= threshold:
            plan_rows.append(
                {
                    "column": col,
                    "n_unique": int(nunique),
                    "missing_pct": round(df[col].isna().mean() * 100, 2),
                    "proposed_strategy": "See grouped companion column / manual review",
                }
            )
    return pd.DataFrame(plan_rows).sort_values(by="n_unique", ascending=False)


def descriptive_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_summary = (
        df[numeric_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        if len(numeric_cols) > 0
        else pd.DataFrame()
    )

    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    categorical_summary: Dict[str, pd.DataFrame] = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
        vc["percent"] = (vc["count"] / len(df) * 100).round(2)
        categorical_summary[col] = vc

    return numeric_summary, categorical_summary


def plot_transplants_by_year(df: pd.DataFrame, output_path: Path) -> None:
    if "KIDNEY_TRANSPLANTSTX_SURG_DT9" in df.columns:
        date_series = pd.to_datetime(df["KIDNEY_TRANSPLANTSTX_SURG_DT9"], errors="coerce")
    else:
        return
    timeline = (
        pd.DataFrame({"year": date_series.dt.year})
        .dropna()
        .groupby("year")
        .size()
        .reset_index(name="transplant_count")
        .sort_values("year")
    )
    if timeline.empty:
        return
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=timeline, x="year", y="transplant_count", marker="o")
    plt.title("Kidney Transplants by Shifted Surgery Year")
    plt.xlabel("Surgery Year (shifted)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_artifacts(
    df_clean: pd.DataFrame,
    missingness: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    categorical_summary: Dict[str, pd.DataFrame],
    binary_actions: Dict[str, str],
    grouping_actions: Dict[str, str],
    temporal_actions: Dict[str, str],
    dropped_columns: List[str],
) -> None:
    df_clean.to_parquet(ARTIFACT_DIR / "cleaned_kidney_features.parquet", index=False)
    df_clean.to_csv(ARTIFACT_DIR / "cleaned_kidney_features.csv", index=False)
    missingness.to_csv(ARTIFACT_DIR / "missingness_summary.csv", index=False)
    numeric_summary.to_csv(ARTIFACT_DIR / "numeric_summary.csv")
    high_cardinality = high_cardinality_plan(df_clean)
    high_cardinality.to_csv(ARTIFACT_DIR / "high_cardinality_plan.csv", index=False)

    categorical_dir = ARTIFACT_DIR / "categorical_distributions"
    categorical_dir.mkdir(exist_ok=True)
    for col, summary in categorical_summary.items():
        summary.to_csv(categorical_dir / f"{col}_distribution.csv", index=False)

    metadata = {
        "binary_transformations": binary_actions,
        "grouping_transformations": grouping_actions,
        "temporal_transformations": temporal_actions,
        "dropped_columns": dropped_columns,
    }
    with open(ARTIFACT_DIR / "data_prep_actions.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def apply_inclusion_exclusion_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"STATUS", "PHASE"}.issubset(df.columns):
        df["EXCLUDE_NOT_FOLLOWED"] = df["STATUS"].isin(["Not Followed"]).astype("Int64")
        df["EXCLUDE_NON_TRANSPLANT_PHASE"] = (~df["PHASE"].eq("Transplanted")).astype("Int64")
    return df


def main(raw_path: Path) -> None:
    df_raw = load_data(raw_path)

    df = normalise_object_columns(df_raw)

    binary_columns = ["PREV_TX", "PREV_KID_TX", "HD", "IMPORTED", "KID_PUMPED", "DGF"]
    df, binary_actions = standardise_binary_columns(df, binary_columns)
    df, gender_actions = standardise_gender_columns(df)
    df, grouping_actions = apply_groupings(df)

    plot_transplants_by_year(df, ARTIFACT_DIR / "transplants_by_year.png")

    df, temporal_actions = engineer_temporal_features(df)

    df = apply_inclusion_exclusion_flags(df)

    constant_cols = identify_constant_and_empty_columns(df)
    df = df.drop(columns=constant_cols)

    missingness = compute_missingness(df)
    numeric_summary, categorical_summary = descriptive_statistics(df)

    save_artifacts(
        df_clean=df,
        missingness=missingness,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        binary_actions=binary_actions | gender_actions,
        grouping_actions=grouping_actions,
        temporal_actions=temporal_actions,
        dropped_columns=constant_cols,
    )

    print(f"Raw shape: {df_raw.shape}")
    print(f"Cleaned shape: {df.shape}")
    print("\nDropped columns:", constant_cols)
    print("\nBinary transformations:", json.dumps(binary_actions | gender_actions, indent=2))
    print("\nGrouping transformations:", json.dumps(grouping_actions, indent=2))
    print("\nTemporal transformations:", json.dumps(temporal_actions, indent=2))
    print(f"\nMissingness summary saved to {ARTIFACT_DIR / 'missingness_summary.csv'}")
    print(f"High-cardinality plan saved to {ARTIFACT_DIR / 'high_cardinality_plan.csv'}")
    print(f"Numeric summary saved to {ARTIFACT_DIR / 'numeric_summary.csv'}")
    print(f"Categorical distributions stored under {ARTIFACT_DIR / 'categorical_distributions/'}")
    print(f"Cleaned feature file saved to {ARTIFACT_DIR / 'cleaned_kidney_features.parquet'}")


if __name__ == "__main__":
    user_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if user_path is None:
        target_path = _first_existing_path(GOOGLE_DRIVE_DEFAULT, REPO_DEFAULT)
    else:
        target_path = _first_existing_path(user_path, GOOGLE_DRIVE_DEFAULT, REPO_DEFAULT)
    main(target_path)
