"""
Compare predictive power of clinical versus community feature sets for 12-month GFR.

Loads the cleaned kidney transplant feature table, builds train/test splits, applies
basic feature selection, and evaluates both a baseline linear regression model and a
RandomForestRegressor on each feature subset.

Outputs performance metrics and the top contributing predictors per model to stdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("outputs/cleaned_kidney_features.csv")
TARGET = "GFR_12MON"
RANDOM_STATE = 42
FEATURE_SELECTION_K = 20


COMMUNITY_FEATURES: List[str] = [
    "DistressScore",
    "Quintile5Distressed",
    "UrbanSuburbanSmallTownRural",
    "TotalPopulation_Round_1k",
    "ofAdultswoaHighSchoolDe",
    "PovertyRate",
    "ofPrimeAgeAdultsNotinWor",
    "VacancyRate",
    "MedianIncomeRatio",
    "ChangeinEmployment",
    "ChangeinEstablishments",
    "NonHispanicWhiteofPopulati",
    "HispanicorLatinoShareofPo",
    "BlackorAfricanAmericanofP",
    "AmericanIndianorAlaskaNative",
    "AsianorPacificIslanderofP",
    "OtherRaceorTwoorMoreRaces",
    "ofthePopulationForeignBorn",
    "of25PopulationwaBachelo",
]

NON_PREDICTOR_COLUMNS: List[str] = [
    TARGET,
    "GFR_3MON",
    "GFR_6MON",
    "NUM_OF_READMITS_12M",
    "NUM_OF_READMITS_6M",
    "NUM_OF_READMITS_3M",
    "LENGTHOFSTAY",
    "PRIMARY_DX",
    "INSURANCE_TYPE",
    "DONOR_RELATION",
    "EMPLOYMENT_STATUS",
    "MARITAL_STATUS",
    "RACE",
    "ETHNICITY",
    "STATUS",
    "DONORSEX",
    "GENDER",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def build_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    community = [col for col in COMMUNITY_FEATURES if col in df.columns]
    drop_cols = set(NON_PREDICTOR_COLUMNS)
    clinical = sorted(
        [
            col
            for col in df.columns
            if col not in community and col not in drop_cols
        ]
    )
    return clinical, community


def preprocess_and_select(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, List[str], SelectKBest]:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    k = min(FEATURE_SELECTION_K, X_train_processed.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_test_selected = selector.transform(X_test_processed)
    selected_features = feature_names[selector.get_support()]

    return X_train_selected, X_test_selected, selected_features.tolist(), selector


def evaluate_models(
    feature_group_name: str,
    feature_columns: List[str],
    df: pd.DataFrame,
) -> Dict[str, object]:
    data = df[feature_columns + [TARGET]].dropna(subset=[TARGET]).copy()
    X = data[feature_columns]
    y = data[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train_sel, X_test_sel, selected_features, selector = preprocess_and_select(
        X_train, X_test, y_train
    )

    results: Dict[str, object] = {
        "feature_group": feature_group_name,
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    # Baseline model: Linear Regression
    baseline_model = LinearRegression()
    baseline_model.fit(X_train_sel, y_train)
    baseline_train_r2 = baseline_model.score(X_train_sel, y_train)
    baseline_test_r2 = r2_score(y_test, baseline_model.predict(X_test_sel))

    results["baseline_model"] = {
        "name": "LinearRegression",
        "train_r2": float(baseline_train_r2),
        "test_r2": float(baseline_test_r2),
    }

    # High-capacity model: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    rf_model.fit(X_train_sel, y_train)
    rf_train_r2 = rf_model.score(X_train_sel, y_train)
    rf_test_r2 = r2_score(y_test, rf_model.predict(X_test_sel))

    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [
        {
            "feature": selected_features[idx],
            "importance": float(importances[idx]),
        }
        for idx in top_indices
    ]

    results["advanced_model"] = {
        "name": "RandomForestRegressor",
        "train_r2": float(rf_train_r2),
        "test_r2": float(rf_test_r2),
        "top_features": top_features,
    }

    return results


def main() -> None:
    df = load_data()
    clinical_features, community_features = build_feature_sets(df)

    if not clinical_features:
        raise ValueError("No clinical features identified for modeling.")
    if not community_features:
        raise ValueError("No community features identified for modeling.")

    clinical_results = evaluate_models("clinical", clinical_features, df)
    community_results = evaluate_models("community", community_features, df)

    for result in [clinical_results, community_results]:
        print(f"\n=== Feature Group: {result['feature_group'].title()} ===")
        print(f"Train samples: {result['n_train']} | Test samples: {result['n_test']}")
        print(f"Selected features ({result['n_selected_features']}): {', '.join(result['selected_features'])}")
        baseline = result["baseline_model"]
        print(
            f"Baseline ({baseline['name']}): "
            f"train R^2={baseline['train_r2']:.3f}, test R^2={baseline['test_r2']:.3f}"
        )
        advanced = result["advanced_model"]
        print(
            f"Advanced ({advanced['name']}): "
            f"train R^2={advanced['train_r2']:.3f}, test R^2={advanced['test_r2']:.3f}"
        )
        print("Top contributing features (Random Forest):")
        for item in advanced["top_features"]:
            print(f"  - {item['feature']}: {item['importance']:.3f}")


if __name__ == "__main__":
    main()

