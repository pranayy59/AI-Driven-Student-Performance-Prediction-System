# Contributing

Thank you for considering a contribution to this project.

## Development Setup

1. Fork or clone the repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add the Kaggle dataset files to `data/raw/`.
5. Run the training workflow:

```bash
python -m src.train
```

## Contribution Guidelines

- Keep the existing folder structure.
- Follow PEP8 and use descriptive variable names.
- Add type hints for new functions.
- Include docstrings for public functions.
- Keep generated model files and raw datasets out of Git.
- Update `README.md` when user-facing behavior changes.

## Pull Request Checklist

- Code compiles successfully.
- Training pipeline runs with the Kaggle dataset.
- Streamlit dashboard opens without errors.
- New outputs are documented.
- No raw dataset files or `.pkl` artifacts are committed.
