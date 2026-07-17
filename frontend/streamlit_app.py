"""Streamlit dashboard for student performance prediction."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.config import FIGURES_DIR
from backend.services.predict import MODEL_PATH, load_artifacts, prepare_input
from backend.services.predict import confidence_level, recommendation_text, risk_level


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp {
        background: #0e1117;
        color: #f5f7fb;
    }
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #263244;
        border-radius: 10px;
        background: linear-gradient(135deg, #111827 0%, #172033 100%);
        margin-bottom: 1rem;
    }
    .hero p {
        color: #cbd5e1;
        margin-bottom: 0;
    }
    .metric-card {
        padding: 1rem;
        border: 1px solid #263244;
        border-radius: 10px;
        background: #151b2b;
        min-height: 120px;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .success {
        color: #4ade80;
    }
    .danger {
        color: #fb7185;
    }
    .footer {
        color: #94a3b8;
        text-align: center;
        padding: 1.4rem 0 0.4rem;
        border-top: 1px solid #263244;
        margin-top: 2rem;
    }
</style>
"""


@st.cache_resource
def cached_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load model artifacts once per Streamlit session."""
    return load_artifacts()


def render_metric_card(label: str, value: str, detail: str = "") -> None:
    """Render a small custom metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div>{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def collect_student_input() -> dict[str, Any]:
    """Render input controls and return one student record."""
    left, middle, right = st.columns(3)

    with left:
        subject = st.selectbox("Subject", ["mathematics", "portuguese"])
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.number_input("Age", min_value=15, max_value=22, value=17)
        address = st.selectbox("Address", ["U", "R"])
        family_size = st.selectbox("Family size", ["GT3", "LE3"])
        parent_status = st.selectbox("Parent cohabitation status", ["T", "A"])
        mother_education = st.slider("Mother education", 0, 4, 3)
        father_education = st.slider("Father education", 0, 4, 3)
        mother_job = st.selectbox(
            "Mother job",
            ["teacher", "health", "services", "at_home", "other"],
        )
        father_job = st.selectbox(
            "Father job",
            ["teacher", "health", "services", "at_home", "other"],
        )

    with middle:
        reason = st.selectbox("Reason for school choice", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        travel_time = st.slider("Travel time", 1, 4, 1)
        study_time = st.slider("Weekly study time", 1, 4, 2)
        failures = st.slider("Past class failures", 0, 4, 0)
        school_support = st.selectbox("Extra school support", ["yes", "no"])
        family_support = st.selectbox("Family support", ["yes", "no"])
        paid_classes = st.selectbox("Extra paid classes", ["yes", "no"])
        activities = st.selectbox("Extracurricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])

    with right:
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access", ["yes", "no"])
        romantic = st.selectbox("Romantic relationship", ["yes", "no"])
        family_relationship = st.slider("Family relationship quality", 1, 5, 4)
        free_time = st.slider("Free time", 1, 5, 3)
        going_out = st.slider("Going out", 1, 5, 3)
        workday_alcohol = st.slider("Workday alcohol consumption", 1, 5, 1)
        weekend_alcohol = st.slider("Weekend alcohol consumption", 1, 5, 1)
        health = st.slider("Current health", 1, 5, 4)
        absences = st.number_input("Absences", min_value=0, max_value=100, value=2)
        first_grade = st.slider("First period grade", 0, 20, 12)
        second_grade = st.slider("Second period grade", 0, 20, 12)

    return {
        "subject": subject,
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": family_size,
        "Pstatus": parent_status,
        "Medu": mother_education,
        "Fedu": father_education,
        "Mjob": mother_job,
        "Fjob": father_job,
        "reason": reason,
        "guardian": guardian,
        "traveltime": travel_time,
        "studytime": study_time,
        "failures": failures,
        "schoolsup": school_support,
        "famsup": family_support,
        "paid": paid_classes,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": family_relationship,
        "freetime": free_time,
        "goout": going_out,
        "Dalc": workday_alcohol,
        "Walc": weekend_alcohol,
        "health": health,
        "absences": absences,
        "G1": first_grade,
        "G2": second_grade,
    }


def render_model_information(metadata: dict[str, Any]) -> None:
    """Display model metrics and training metadata."""
    metrics = metadata.get("metrics", {})
    columns = st.columns(5)
    metric_map = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1 Score", metrics.get("f1_score")),
        ("ROC-AUC", metrics.get("roc_auc")),
    ]

    for column, (label, value) in zip(columns, metric_map):
        with column:
            display_value = f"{value:.3f}" if isinstance(value, float) else "N/A"
            render_metric_card(label, display_value)

    st.markdown("### Model Information")
    st.write(f"**Best model:** {metadata.get('best_model', 'Not available')}")
    st.write(f"**Target:** {metadata.get('target_definition', 'G3 >= 10')}")
    st.write(f"**Features:** {len(metadata.get('feature_columns', []))}")

    model_rows = metadata.get("all_model_metrics", [])
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True)


def render_visualizations() -> None:
    """Display generated evaluation figures if they exist."""
    figure_names = [
        "correlation_heatmap.png",
        "feature_importance.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "class_distribution.png",
        "model_accuracy_comparison.png",
    ]

    for figure_name in figure_names:
        figure_path = FIGURES_DIR / figure_name
        if figure_path.exists():
            st.image(str(figure_path), caption=figure_name.replace("_", " ").replace(".png", "").title())
        else:
            st.info(
                f"{figure_name} will appear here after running "
                "`python backend/models/train.py`."
            )


def create_prediction_download(result: dict[str, Any], record: dict[str, Any]) -> bytes:
    """Create a CSV payload containing the prediction and input record."""
    output = {**record, **result, "created_at": datetime.now().isoformat(timespec="seconds")}
    return pd.DataFrame([output]).to_csv(index=False).encode("utf-8")


st.markdown(APP_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("Student AI")
    page = st.radio(
        "Navigation",
        ["Prediction Dashboard", "Model Information", "Visualizations"],
    )
    st.caption("Production-style ML dashboard for academic risk prediction.")

st.markdown(
    """
    <div class="hero">
        <h1>AI-Driven Student Performance Prediction System</h1>
        <p>Machine learning dashboard for predicting pass probability, confidence, and academic risk.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run `python backend/models/train.py` "
        "before opening the dashboard."
    )
    st.stop()

model, metadata = cached_artifacts()

if page == "Prediction Dashboard":
    st.markdown("### Student Profile")
    student_record = collect_student_input()

    if st.button("Predict Outcome", type="primary", use_container_width=True):
        input_frame = prepare_input(student_record, metadata.get("feature_columns"))
        prediction_value = int(model.predict(input_frame)[0])
        pass_probability = float(model.predict_proba(input_frame)[0][1])
        prediction = "PASS" if prediction_value == 1 else "FAIL"
        probability_text = f"{pass_probability:.1%}"
        confidence = confidence_level(pass_probability)
        risk = risk_level(pass_probability)
        recommendation = recommendation_text(prediction.title(), pass_probability)

        result = {
            "prediction": prediction,
            "pass_probability": round(pass_probability, 4),
            "confidence": confidence,
            "risk": risk,
            "model": metadata.get("best_model", "trained_model"),
            "recommendation": recommendation,
        }

        st.markdown("### Prediction Result")
        card_columns = st.columns(5)
        values = [
            ("Prediction", f"{prediction} {'[OK]' if prediction == 'PASS' else '[!]'}"),
            ("Probability", probability_text),
            ("Confidence", confidence),
            ("Risk", risk),
            ("Model", str(result["model"])),
        ]

        for column, (label, value) in zip(card_columns, values):
            with column:
                render_metric_card(label, value)

        st.progress(pass_probability)
        st.success(f"Recommendation: {recommendation}")

        st.download_button(
            label="Download prediction as CSV",
            data=create_prediction_download(result, student_record),
            file_name="student_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Prediction details"):
            st.dataframe(pd.DataFrame([{**student_record, **result}]), use_container_width=True)

elif page == "Model Information":
    render_model_information(metadata)

elif page == "Visualizations":
    render_visualizations()

st.markdown(
    """
    <div class="footer">
        Built for AI/ML portfolio demonstration | Student Performance Prediction System
    </div>
    """,
    unsafe_allow_html=True,
)
