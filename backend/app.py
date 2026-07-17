"""Backend application entry points for prediction services."""

from __future__ import annotations

from typing import Any

from backend.services.predict import predict_student


def predict(record: dict[str, Any]) -> dict[str, Any]:
    """Return a pass/fail prediction for one student record."""
    return predict_student(record)


__all__ = ["predict"]
