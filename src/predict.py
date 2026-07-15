"""Predict pass/fail outcomes for new student records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODELS_DIR


MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.pkl"


def confidence_level(probability: float) -> str:
    """Convert a class probability into a user-friendly confidence label."""
    confidence = max(probability, 1 - probability)

    if confidence >= 0.9:
        return "Very High"
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.6:
        return "Moderate"
    return "Low"


def risk_level(pass_probability: float) -> str:
    """Map pass probability to an academic risk level."""
    if pass_probability >= 0.8:
        return "Low Risk"
    if pass_probability >= 0.6:
        return "Moderate Risk"
    return "High Risk"


def recommendation_text(prediction: str, pass_probability: float) -> str:
    """Generate a concise recommendation for the prediction result."""
    if prediction == "Pass" and pass_probability >= 0.8:
        return (
            "Student is highly likely to pass based on academic history and "
            "support indicators."
        )
    if prediction == "Pass":
        return (
            "Student is likely to pass, but continued academic monitoring is "
            "recommended."
        )
    return (
        "Student may be at risk of failing. Consider targeted academic support, "
        "attendance review, and mentoring."
    )


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load the trained model and metadata from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train` before predicting.")

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH) if METADATA_PATH.exists() else {}
    return model, metadata


def prepare_input(record: dict[str, Any], feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Convert one input record to a DataFrame with the training feature columns."""
    frame = pd.DataFrame([record])

    if feature_columns:
        for column in feature_columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[feature_columns]

    return frame


def predict_student(record: dict[str, Any]) -> dict[str, Any]:
    """Predict whether a student will pass."""
    model, metadata = load_artifacts()
    feature_columns = metadata.get("feature_columns")
    input_frame = prepare_input(record, feature_columns)

    prediction = int(model.predict(input_frame)[0])
    probability = float(model.predict_proba(input_frame)[0][1])
    label = "Pass" if prediction == 1 else "Fail"

    return {
        "prediction": label,
        "pass_probability": round(probability, 4),
        "confidence": confidence_level(probability),
        "risk": risk_level(probability),
        "model": metadata.get("best_model", "trained_model"),
        "recommendation": recommendation_text(label, probability),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict student pass/fail outcome.")
    parser.add_argument("--input", type=str, help="Student record as a JSON string.")
    parser.add_argument("--file", type=Path, help="Path to a JSON file containing one student record.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for prediction."""
    args = parse_args()

    if args.file:
        record = json.loads(args.file.read_text(encoding="utf-8"))
    elif args.input:
        record = json.loads(args.input)
    else:
        raise SystemExit("Provide either --input JSON or --file path/to/student.json")

    result = predict_student(record)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
