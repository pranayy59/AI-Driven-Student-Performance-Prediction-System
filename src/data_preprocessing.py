"""Data loading, cleaning, feature engineering, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_FILES, GRADE_COLUMN, PASSING_GRADE, RAW_DATA_DIR, TARGET_COLUMN


def load_student_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load available Kaggle student performance CSV files."""
    frames: list[pd.DataFrame] = []

    for subject, filename in DATA_FILES.items():
        file_path = raw_dir / filename
        if file_path.exists():
            frame = pd.read_csv(file_path, sep=";")
            if frame.empty:
                raise ValueError(f"Dataset file is empty: {file_path}")
            frame["subject"] = subject
            frames.append(frame)

    if not frames:
        expected = ", ".join(DATA_FILES.values())
        raise FileNotFoundError(
            f"No dataset files found in {raw_dir}. Add one or more of: {expected}"
        )

    return pd.concat(frames, ignore_index=True)


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw student data and create the binary pass/fail target."""
    if df.empty:
        raise ValueError("Input dataset is empty.")

    cleaned = df.copy()

    # Remove duplicate student-course records while preserving first occurrence.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Normalize object columns to reduce accidental category duplication.
    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    if GRADE_COLUMN not in cleaned.columns:
        raise ValueError(f"Required final grade column '{GRADE_COLUMN}' was not found.")

    cleaned[GRADE_COLUMN] = pd.to_numeric(cleaned[GRADE_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[GRADE_COLUMN]).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(f"No valid rows remain after cleaning '{GRADE_COLUMN}'.")

    cleaned[TARGET_COLUMN] = (cleaned[GRADE_COLUMN] >= PASSING_GRADE).astype(int)
    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target, excluding the final grade to avoid leakage."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    drop_columns = [TARGET_COLUMN, GRADE_COLUMN]
    X = df.drop(columns=[column for column in drop_columns if column in df.columns])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical features."""
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract readable feature names after preprocessing."""
    names: list[str] = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue

        if transformer_name == "numeric":
            names.extend(list(columns))
        elif transformer_name == "categorical":
            encoder = transformer.named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(columns).tolist())

    return names
