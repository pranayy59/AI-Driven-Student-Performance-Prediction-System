# Contributing

Thank you for helping improve this project.

## Development Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the Kaggle dataset files in `backend/data/raw/`:

```text
backend/data/raw/student-mat.csv
backend/data/raw/student-por.csv
```

4. Run the training workflow:

```bash
python backend/models/train.py
```

5. Run the Streamlit dashboard:

```bash
streamlit run frontend/streamlit_app.py
```

## Guidelines

- Preserve the existing backend/frontend architecture.
- Do not commit raw dataset files, generated model artifacts, reports, or figures.
- Keep preprocessing and model settings reproducible unless there is a documented bug.
- Update documentation when commands, paths, or user-facing behavior changes.
