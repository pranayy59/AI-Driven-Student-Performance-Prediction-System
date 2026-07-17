"""Exploratory data analysis and model visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

from backend.config.config import FIGURES_DIR, GRADE_COLUMN, TARGET_COLUMN


PLOT_STYLE = "whitegrid"
PRIMARY_COLOR = "#2f80ed"
SUCCESS_COLOR = "#27ae60"
DANGER_COLOR = "#eb5757"


def _apply_plot_style() -> None:
    """Apply a consistent style to all saved project figures."""
    sns.set_theme(style=PLOT_STYLE, palette="deep")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a correlation heatmap for numeric variables."""
    _apply_plot_style()
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        numeric_df.corr(),
        cmap="vlag",
        center=0,
        linewidths=0.3,
        square=False,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def save_score_distributions(df: pd.DataFrame) -> None:
    """Save score distributions for period and final grades."""
    _apply_plot_style()
    grade_columns = [column for column in ["G1", "G2", GRADE_COLUMN] if column in df.columns]

    if not grade_columns:
        return

    melted = df.melt(value_vars=grade_columns, var_name="Grade Type", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=melted,
        x="Score",
        hue="Grade Type",
        bins=20,
        kde=True,
        multiple="layer",
    )
    plt.title("Score Distributions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "score_distributions.png", dpi=300)
    plt.close()


def save_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Save the top feature importance chart."""
    _apply_plot_style()
    if feature_importance.empty:
        return

    top_features = feature_importance.head(20).sort_values("importance")

    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="importance", y="feature", color=PRIMARY_COLOR)
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300)
    plt.close()


def save_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a confusion matrix for the best model."""
    _apply_plot_style()
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
        display_labels=["Fail", "Pass"],
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def save_roc_curve(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save the ROC curve for the best model."""
    _apply_plot_style()
    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.close()


def save_precision_recall_curve(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save the precision-recall curve for the best model."""
    _apply_plot_style()
    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()


def save_model_accuracy_comparison(comparison_df: pd.DataFrame) -> None:
    """Save a model accuracy comparison bar chart."""
    _apply_plot_style()
    ordered = comparison_df.sort_values("accuracy", ascending=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=ordered, x="accuracy", y="model", color=SUCCESS_COLOR)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_accuracy_comparison.png", dpi=300)
    plt.close()


def run_eda(df: pd.DataFrame) -> None:
    """Generate all EDA figures."""
    _apply_plot_style()
    save_correlation_heatmap(df)
    save_score_distributions(df)

    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        palette=[DANGER_COLOR, SUCCESS_COLOR],
        legend=False,
    )
    plt.title("Pass/Fail Class Distribution")
    plt.xticks([0, 1], ["Fail", "Pass"])
    plt.xlabel("Outcome")
    plt.ylabel("Student Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()
