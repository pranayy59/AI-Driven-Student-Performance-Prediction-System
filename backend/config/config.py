"""Central project paths and constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

DATA_DIR = BACKEND_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BACKEND_DIR / "models"
OUTPUTS_DIR = BACKEND_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

DATA_FILES = {
    "mathematics": "student-mat.csv",
    "portuguese": "student-por.csv",
}

TARGET_COLUMN = "pass"
GRADE_COLUMN = "G3"
PASSING_GRADE = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_directories() -> None:
    """Create runtime directories used by the pipeline."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        SCREENSHOTS_DIR,
        SAMPLE_DATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
