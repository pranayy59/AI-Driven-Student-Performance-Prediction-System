"""Train and evaluate student pass/fail prediction models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    ensure_directories,
)
from backend.services.data_preprocessing import (
    build_preprocessor,
    clean_student_data,
    get_feature_names,
    load_student_data,
    split_features_target,
)
from backend.services.eda import (
    run_eda,
    save_confusion_matrix,
    save_feature_importance,
    save_model_accuracy_comparison,
    save_precision_recall_curve,
    save_roc_curve,
)


def build_models() -> dict[str, object]:
    """Create candidate models for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a fitted model with common classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def extract_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Extract model-native or permutation-based feature importance."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = get_feature_names(preprocessor)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).ravel()
    else:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
        )
        return (
            pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_select_model() -> tuple[Pipeline, pd.DataFrame]:
    """Run the full training workflow and return the best model with comparison results."""
    ensure_directories()

    raw_df = load_student_data()
    cleaned_df = clean_student_data(raw_df)
    cleaned_df.to_csv(PROCESSED_DATA_DIR / "student_performance_processed.csv", index=False)

    run_eda(cleaned_df)

    X, y = split_features_target(cleaned_df)

    if y.nunique() < 2:
        raise ValueError("Training requires both pass and fail examples.")

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model_results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        model_results.append({"model": model_name, **metrics})
        fitted_models[model_name] = pipeline

    comparison_df = (
        pd.DataFrame(model_results)
        .sort_values(["roc_auc", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    save_model_accuracy_comparison(comparison_df)

    best_model_name = str(comparison_df.loc[0, "model"])
    best_model = fitted_models[best_model_name]

    y_pred = best_model.predict(X_test)
    report_text = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])
    (REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    feature_importance = extract_feature_importance(best_model, X_test, y_test)
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    save_feature_importance(feature_importance)
    save_confusion_matrix(best_model, X_test, y_test)
    save_roc_curve(best_model, X_test, y_test)
    save_precision_recall_curve(best_model, X_test, y_test)

    metadata = {
        "best_model": best_model_name,
        "estimator_class": best_model.named_steps["model"].__class__.__name__,
        "metrics": comparison_df.loc[0].to_dict(),
        "all_model_metrics": comparison_df.to_dict(orient="records"),
        "feature_columns": X.columns.tolist(),
        "class_labels": {"0": "Fail", "1": "Pass"},
        "target_definition": "pass = 1 when G3 >= 10, else 0",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(metadata, MODELS_DIR / "model_metadata.pkl")
    (REPORTS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return best_model, comparison_df


def main() -> None:
    """CLI entry point for training."""
    best_model, comparison_df = train_and_select_model()
    best_model_name = best_model.named_steps["model"].__class__.__name__

    print("Training complete.")
    print(f"Best estimator: {best_model_name}")
    print(comparison_df.to_string(index=False))
    print(f"Saved model to: {Path(MODELS_DIR / 'best_model.pkl')}")


if __name__ == "__main__":
    main()
