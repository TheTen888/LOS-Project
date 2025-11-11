"""
Kidney transplant feature engineering and exploratory reporting pipeline.

This script ingests the integrated kidney transplant dataset, enforces the
cleaning requirements outlined by the clinical data science team, and exports
both a modeling-ready feature table and supporting profiling artefacts.

Key capabilities:
* Standardises binary/string encodings (e.g., Y/N -> 1/0, donor sex harmonisation)
* Drops structurally uninformative features (all-missing or constant)
* Engineers time-interval features from de-identified dates and removes raw timestamps
* Collapses high-cardinality text features (payer, diagnosis, donor relation, employment)
* Quantifies missingness, descriptive statistics, and categorical distributions
* Generates transplant volume trends over time for sanity checking

The default input path matches the Google Drive location shared by the team.
If that file is not present locally (e.g., when running inside this repo),
the script will fall back to the repo copy at ./merged_kidney_dci_final.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Required default path (per user request). Fallback to local copy if not present.
csv_file_path = Path("/content/drive/MyDrive/merged_kidney_dci_final.csv")
LOCAL_FALLBACK = Path(__file__).resolve().parent / "merged_kidney_dci_final.csv"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Date columns provided in the raw extract
DATE_COLUMNS = [
    "KIDNEY_TRANSPLANTSTX_EVAL_DT9",
    "LISTING_DATE",
    "KIDNEY_TRANSPLANTSTX_SURG_DT9",
    "KIDNEY_TRANSPLANTSDONORDOB9",
]


# High-cardinality mappings ------------------------------------------------------------

PRIMARY_DX_GROUP_MAP: Dict[str, str] = {
    "Acquired Obstructive Nephropathy": "Structural/Obstructive",
    "Acute Tubular Necrosis": "Acute Kidney Injury",
    "Alport's Syndrome": "Hereditary/Congenital",
    "Amyloidosis - Kidney": "Metabolic/Depositional",
    "Analgesic Nephropathy": "Drug/Toxin-Induced",
    "Anti-GBM": "Autoimmune/Vasculitis",
    "Calcineurin Inhibitor Nephrotoxicity": "Drug/Toxin-Induced",
    "Cancer Chemotherapy Induced Nephritis": "Drug/Toxin-Induced",
    "Chronic Glomerulonephritis - Unspecified": "Glomerular Disease",
    "Chronic Glomerulosclerosis - Unspecified": "Glomerular Disease",
    "Chronic Nephrosclerosis - Unspecified": "Hypertensive Disease",
    "Chronic Pyelonephritis (Reflux Nephropathy)": "Structural/Obstructive",
    "Congenital Obstructive Uropathy": "Hereditary/Congenital",
    "Cortical Necrosis": "Acute Kidney Injury",
    "Diabetes - Type I Insulin Dep/Juvenile Onset": "Diabetes",
    "Diabetes - Type I Non-Insulin Dep/ Juvenile Onset": "Diabetes",
    "Diabetes - Type II Non-Insulin Dep/ Adult Onset": "Diabetes",
    "Diabetes Mellitus - Type I": "Diabetes",
    "Diabetes Mellitus - Type II": "Diabetes",
    "Diabetes Mellitus - Type Other / Unknown": "Diabetes",
    "Drug Related Interstitial Nephritis": "Drug/Toxin-Induced",
    "Fabry's Disease": "Metabolic/Depositional",
    "Focal Glomerular Sclerosis (Focal Segmental - FSG)": "Glomerular Disease",
    "Goodpasture's Syndrome": "Autoimmune/Vasculitis",
    "HIV Nephropathy": "Infectious",
    "Hemolytic Uremic Syndrome": "Thrombotic Microangiopathy",
    "Henoch-Schonlein Purpura": "Autoimmune/Vasculitis",
    "Hepatorenal Syndrome": "Systemic/Secondary",
    "Hypertensive Nephrosclerosis": "Hypertensive Disease",
    "Hypoplasia/Dysplasia/Dysgenesis/Agenesis": "Hereditary/Congenital",
    "Idiopathic and Post-infectious Crescentic Glomerulonephritis": "Glomerular Disease",
    "IgA Nephropathy": "Glomerular Disease",
    "Incidental Carcinoma": "Neoplasm",
    "Lithium Toxicity": "Drug/Toxin-Induced",
    "Malignant Hypertension": "Hypertensive Disease",
    "Medullary Cystic Disease": "Hereditary/Congenital",
    "Membranous Glomerulonephritis": "Glomerular Disease",
    "Membranous Nephropathy": "Glomerular Disease",
    "Mesangio-capillary Glomerulonephritis Type II": "Glomerular Disease",
    "Missing": "Missing",
    "Myeloma": "Neoplasm",
    "Nephritis": "Glomerular Disease",
    "Nephrolithiasis": "Structural/Obstructive",
    "Nephronophthisis": "Hereditary/Congenital",
    "Other, Specify": "Other/Unspecified",
    "Oxalate Nephropathy (Includes Hereditary Oxalosis)": "Metabolic/Depositional",
    "Polyarteritis": "Autoimmune/Vasculitis",
    "Polycystic Kidneys": "Polycystic Kidney Disease",
    "Prune Belly Syndrome": "Hereditary/Congenital",
    "Renal Cell Carcinoma": "Neoplasm",
    "Sarcoidosis - Kidney": "Autoimmune/Vasculitis",
    "Scleroderma - Kidney": "Autoimmune/Vasculitis",
    "Systemic Lupus Erythematosus": "Autoimmune/Vasculitis",
    "Wegener's Granulomatosis": "Autoimmune/Vasculitis",
    "Wilms' Tumor": "Neoplasm",
}

DONOR_RELATION_GROUP_MAP: Dict[str, str] = {
    "Other - Non Biological": "Non-Biological/Unrelated",
    "Related Other - Biological": "Biological Relative",
    "Brother - Biological": "Biological Relative",
    "Sister - Biological": "Biological Relative",
    "Mother - Biological": "Biological Relative",
    "Father - Biological": "Biological Relative",
    "Son - Biological": "Biological Relative",
    "Daughter - Biological": "Biological Relative",
    "Half Brother - Biological": "Biological Relative",
    "Half Sister - Biological": "Biological Relative",
    "Grandson - Biological": "Biological Relative",
    "Cousin - Biological": "Biological Relative",
    "Nephew - Biological": "Biological Relative",
    "Spouse - Non Biological": "Non-Biological/Unrelated",
    "Friend - Non Biological": "Non-Biological/Unrelated",
    "Anonymous - Non Biological": "Non-Biological/Unrelated",
    "Paired Donation - Non Biological": "Paired/Exchange",
    "Unrelated Directed Donor - Non Biological": "Non-Biological/Unrelated",
    "Missing": "Unknown",
    np.nan: "Unknown",
}

EMPLOYMENT_GROUP_MAP: Dict[str, str] = {
    "Full Time": "Employed",
    "Part Time": "Employed",
    "Self Employed": "Employed",
    "Not Employed": "Not Employed",
    "Retired": "Retired",
    "Student Full Time": "Student",
    "Unknown": "Unknown",
    "Missing": "Unknown",
}

RACE_GROUP_MAP: Dict[str, str] = {
    "White": "White",
    "Black/African American": "Black",
    "Asian Indian": "Asian / Pacific Islander",
    "Chinese": "Asian / Pacific Islander",
    "Japanese": "Asian / Pacific Islander",
    "Vietnamese": "Asian / Pacific Islander",
    "Other Asian": "Asian / Pacific Islander",
    "Other Pacific Islander": "Asian / Pacific Islander",
    "American Indian/Alaska Native": "American Indian / Alaska Native",
    "Other": "Other/Multiple",
    "Unreported,Chose Not to Disclose Race": "Unknown",
    np.nan: "Unknown",
}


# --------------------------------------------------------------------------------------
# Data classes for reporting
# --------------------------------------------------------------------------------------

@dataclass
class PreparationArtifacts:
    dropped_columns: List[str]
    binary_columns: List[str]
    engineered_columns: List[str]
    grouped_categorical_columns: Dict[str, List[str]]
    missingness: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load the dataset from Drive (preferred) or local fallback."""
    path = csv_file_path
    if not path.exists():
        if LOCAL_FALLBACK.exists():
            path = LOCAL_FALLBACK
        else:
            raise FileNotFoundError(
                f"Input dataset not found at {csv_file_path} or {LOCAL_FALLBACK}"
            )

    df = pd.read_csv(path, parse_dates=DATE_COLUMNS, infer_datetime_format=True)
    # Ensure pandas keeps NA values when strings like "NaN" appear in text columns
    df = df.replace({"": pd.NA, " ": pd.NA})
    return df


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace in object columns without altering numeric types."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace({"": pd.NA})
    return df


def _standardise_binary_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convert binary-like columns to 0/1 integers."""
    df = df.copy()
    binary_maps = {
        "HD": {"Y": 1, "N": 0},
        "IMPORTED": {"Y": 1, "N": 0},
        "KID_PUMPED": {"Yes": 1, "No": 0},
        "PREV_KID_TX": {"Y": 1, "N": 0},
        "PREV_TX": {"Y": 1, "N": 0},
    }
    converted_columns: List[str] = []
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            converted_columns.append(col)

    # Ensure existing numeric binaries are ints
    for col in ["DM", "HTN", "NUM_OF_READMITS_12M", "NUM_OF_READMITS_6M", "NUM_OF_READMITS_3M"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
            converted_columns.append(col)

    return df, sorted(set(converted_columns))


def _harmonise_sex_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Standardise recipient and donor sex encodings and derive binary indicators."""
    df = df.copy()

    engineered = []

    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].replace({"M": "Male", "F": "Female"})
        df["RECIPIENT_SEX_MALE"] = df["GENDER"].map({"Male": 1, "Female": 0})
        engineered.append("RECIPIENT_SEX_MALE")

    if "DONORSEX" in df.columns:
        df["DONORSEX"] = df["DONORSEX"].replace(
            {"M": "Male", "F": "Female", "male": "Male", "female": "Female"}
        )
        df["DONOR_SEX_MALE"] = df["DONORSEX"].map({"Male": 1, "Female": 0})
        engineered.append("DONOR_SEX_MALE")

    return df, engineered


def _impute_domain_driven_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply rule-based imputations that are clinically justified.

    * Dialysis duration is zero for non-hemodialysis patients.
    * KDPI is undefined for living donors; retain missing but add indicator.
    """
    df = df.copy()
    engineered = []

    if {"DIALYSIS_TIME_MONTHS", "HD"}.issubset(df.columns):
        mask = df["DIALYSIS_TIME_MONTHS"].isna() & (df["HD"] == 0)
        df.loc[mask, "DIALYSIS_TIME_MONTHS"] = 0.0
        df["DIALYSIS_TIME_MONTHS"] = df["DIALYSIS_TIME_MONTHS"].astype("float64")

    if "DONORTYPE" in df.columns:
        df["DONORTYPE"] = df["DONORTYPE"].fillna("Missing")

    if "KDPI" in df.columns and "DONORTYPE" in df.columns:
        indicator_name = "KDPI_MISSING_FOR_LIVING"
        df[indicator_name] = np.where(df["KDPI"].isna() & (df["DONORTYPE"] == "Living"), 1, 0)
        engineered.append(indicator_name)

    return df, engineered


def _engineer_time_intervals(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert raw transplant timeline dates to interval-based features.
    Original dates are removed to prevent leakage of shifted absolute dates.
    """
    df = df.copy()
    engineered = []

    eval_col = "KIDNEY_TRANSPLANTSTX_EVAL_DT9"
    listing_col = "LISTING_DATE"
    tx_col = "KIDNEY_TRANSPLANTSTX_SURG_DT9"

    if set([eval_col, listing_col, tx_col]).issubset(df.columns):
        df["eval_to_listing_days"] = (
            df[listing_col] - df[eval_col]
        ).dt.days.astype("float64")
        df["listing_to_transplant_days"] = (
            df[tx_col] - df[listing_col]
        ).dt.days.astype("float64")
        df["eval_to_transplant_days"] = (
            df[tx_col] - df[eval_col]
        ).dt.days.astype("float64")

        df["listing_year"] = df[listing_col].dt.year
        df["transplant_year"] = df[tx_col].dt.year
        df["transplant_month"] = df[tx_col].dt.month

        engineered.extend(
            [
                "eval_to_listing_days",
                "listing_to_transplant_days",
                "eval_to_transplant_days",
                "listing_year",
                "transplant_year",
                "transplant_month",
            ]
        )

        df = df.drop(columns=[eval_col, listing_col, tx_col, "KIDNEY_TRANSPLANTSDONORDOB9"], errors="ignore")

    return df, engineered


def _map_insurance_type(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    val = value.lower()
    if "medicare" in val:
        return "Medicare"
    if "medicaid" in val or "gateway" in val or "upmc for you" in val:
        return "Medicaid"
    if "self-pay" in val:
        return "Self-Pay"
    if "workers comp" in val:
        return "Workers Comp"
    if "auto" in val:
        return "Auto/Other"
    if "pending medicaid" in val:
        return "Medicaid"
    if val in {"147", "158"}:
        return "Other/Unknown"
    if "champus" in val or "hmi" in val or "global care" in val:
        return "Commercial/Other"
    if "other" == val:
        return "Other/Unknown"
    if "upmc employees" in val:
        return "Commercial/Other"
    if "security blue" in val:
        return "Commercial/Other"
    if "upmc for life" in val:
        return "Medicare"
    if "united healthcare medicare" in val:
        return "Medicare"
    if "united healthcare" in val:
        return "Commercial/Other"
    if "community blue" in val:
        return "Medicare"
    if "highmark" in val or "aetna" in val or "cigna" in val or "bcbs" in val or "upmc employees" in val:
        return "Commercial/Other"
    return "Commercial/Other"


def _group_high_cardinality_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Collapse sparse categorical features into modeling friendly groupings."""
    df = df.copy()
    grouped_levels: Dict[str, List[str]] = {}

    if "PRIMARY_DX" in df.columns:
        df["PRIMARY_DX_GROUP"] = df["PRIMARY_DX"].map(
            lambda v: PRIMARY_DX_GROUP_MAP.get(v, "Other/Unspecified") if pd.notna(v) else "Missing"
        )
        grouped_levels["PRIMARY_DX_GROUP"] = sorted(df["PRIMARY_DX_GROUP"].dropna().unique())

    if "INSURANCE_TYPE" in df.columns:
        df["INSURANCE_GROUP"] = df["INSURANCE_TYPE"].map(_map_insurance_type)
        grouped_levels["INSURANCE_GROUP"] = sorted(df["INSURANCE_GROUP"].dropna().unique())

    if "DONOR_RELATION" in df.columns:
        df["DONOR_RELATION_GROUP"] = df["DONOR_RELATION"].map(
            lambda v: DONOR_RELATION_GROUP_MAP.get(v, "Unknown")
        )
        grouped_levels["DONOR_RELATION_GROUP"] = sorted(df["DONOR_RELATION_GROUP"].dropna().unique())

    if "EMPLOYMENT_STATUS" in df.columns:
        df["EMPLOYMENT_GROUP"] = df["EMPLOYMENT_STATUS"].map(
            lambda v: EMPLOYMENT_GROUP_MAP.get(v, "Unknown")
        )
        grouped_levels["EMPLOYMENT_GROUP"] = sorted(df["EMPLOYMENT_GROUP"].dropna().unique())

    if "MARITAL_STATUS" in df.columns:
        df["MARITAL_STATUS_GROUP"] = df["MARITAL_STATUS"].replace(
            {"Separated": "Previously Married", "Divorced": "Previously Married"}
        )
        grouped_levels["MARITAL_STATUS_GROUP"] = sorted(df["MARITAL_STATUS_GROUP"].dropna().unique())

    if "RACE" in df.columns:
        df["RACE_GROUP"] = df["RACE"].map(lambda v: RACE_GROUP_MAP.get(v, "Other/Multiple"))
        grouped_levels["RACE_GROUP"] = sorted(df["RACE_GROUP"].dropna().unique())

    if "ETHNICITY" in df.columns:
        df["ETHNICITY_GROUP"] = df["ETHNICITY"].fillna("Unknown")
        df["IS_HISPANIC"] = (df["ETHNICITY_GROUP"] == "Hispanic/Latino").astype("Int64")
        grouped_levels["ETHNICITY_GROUP"] = sorted(df["ETHNICITY_GROUP"].dropna().unique())

    return df, grouped_levels


def prepare_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, PreparationArtifacts]:
    """Full preparation pipeline orchestrating all helper steps."""
    df = _strip_strings(df_raw)

    # Drop uninformative columns (all missing or constant) plus those flagged by SMEs.
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    sme_drop_cols = [
        "TRANSPLANTED",
        "PHASE",
        "KAS_SCORE",
        "TERMINAL_CREAT",
        "GAIT_SPEED_AT_EVAL",
        "DGF",
    ]
    drop_cols = sorted(set(constant_cols + sme_drop_cols))
    df = df.drop(columns=drop_cols, errors="ignore")

    df, binary_cols = _standardise_binary_columns(df)
    df, sex_cols = _harmonise_sex_columns(df)
    df, domain_engineered = _impute_domain_driven_values(df)
    df, interval_cols = _engineer_time_intervals(df)
    df, grouped_levels = _group_high_cardinality_categories(df)

    # Final touches on categorical NA handling
    for col in ["DONORSEX", "DONOR_RELATION", "INSURANCE_TYPE", "EMPLOYMENT_STATUS"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Missingness summary (retain percentages for report)
    missingness = (
        df.isna()
        .mean()
        .reset_index(name="missing_pct")
        .rename(columns={"index": "feature"})
        .sort_values("missing_pct", ascending=False)
    )

    numeric_summary = (
        df.select_dtypes(include=[np.number])
        .describe()
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )

    categorical_summary = build_categorical_summary(df)

    artifacts = PreparationArtifacts(
        dropped_columns=drop_cols,
        binary_columns=binary_cols,
        engineered_columns=sex_cols + domain_engineered + interval_cols,
        grouped_categorical_columns=grouped_levels,
        missingness=missingness,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
    )

    # Sort columns alphabetically to improve schema stability downstream
    df = df.reindex(sorted(df.columns), axis=1)

    return df, artifacts


def build_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten categorical value counts into a tidy table."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    records: List[Dict[str, object]] = []
    for col in categorical_cols:
        value_counts = df[col].fillna("Missing").value_counts(dropna=False, normalize=False)
        total = len(df)
        for level, count in value_counts.items():
            records.append(
                {
                    "feature": col,
                    "level": level,
                    "count": int(count),
                    "percent": round((count / total) * 100, 2),
                }
            )
    return pd.DataFrame(records).sort_values(["feature", "count"], ascending=[True, False])


def plot_transplants_by_year(df: pd.DataFrame, output_path: Path) -> None:
    """Create a frequency plot of transplant counts per year."""
    if "transplant_year" not in df.columns:
        return
    year_counts = (
        df["transplant_year"]
        .value_counts(dropna=True)
        .rename_axis("transplant_year")
        .sort_index()
        .reset_index(name="count")
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=year_counts, x="transplant_year", y="count", marker="o")
    plt.title("Kidney Transplants by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Transplants")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_artifacts(clean_df: pd.DataFrame, artifacts: PreparationArtifacts) -> None:
    """Persist cleaned dataset and artefacts to disk."""
    clean_df.to_csv(OUTPUT_DIR / "cleaned_kidney_features.csv", index=False)
    artifacts.missingness.to_csv(OUTPUT_DIR / "missingness_summary.csv", index=False)
    artifacts.numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv", index=False)
    artifacts.categorical_summary.to_csv(OUTPUT_DIR / "categorical_summary.csv", index=False)

    grouping_records = []
    for column, levels in artifacts.grouped_categorical_columns.items():
        grouping_records.append({"column": column, "levels": "|".join(levels)})
    pd.DataFrame(grouping_records).to_csv(OUTPUT_DIR / "grouping_levels.csv", index=False)

    plot_transplants_by_year(clean_df, OUTPUT_DIR / "transplants_by_year.png")


def main() -> None:
    print("Loading dataset...")
    df_raw = load_dataset()
    print(f"Raw shape: {df_raw.shape}")

    print("Running preparation pipeline...")
    clean_df, artifacts = prepare_dataset(df_raw)
    print(f"Cleaned shape: {clean_df.shape}")

    export_artifacts(clean_df, artifacts)

    print("\nDropped columns:")
    for col in artifacts.dropped_columns:
        print(f"  - {col}")

    print("\nEngineered columns:")
    for col in artifacts.engineered_columns:
        print(f"  - {col}")

    print("\nHigh-cardinality grouping levels:")
    for col, levels in artifacts.grouped_categorical_columns.items():
        preview = ", ".join(levels[:8])
        suffix = "..." if len(levels) > 8 else ""
        print(f"  - {col}: {preview}{suffix}")

    print("\nTop 10 features by missingness:")
    print(artifacts.missingness.head(10).to_string(index=False))

    print("\nPreparation complete. Artefacts written to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

