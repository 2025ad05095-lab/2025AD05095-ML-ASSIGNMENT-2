# ML Assignment 2 Workspace

Interactive Streamlit workspace for Adult Census Income classification.

## Features

- Accepts standard Adult dataset files with or without a header
- Explores data quality and target distribution
- Applies imputation, standardization, and one-hot encoding in reusable pipelines
- Compares Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and Random Forest
- Reports Accuracy, ROC AUC, Precision, Recall, F1 Score, and MCC
- Tunes Random Forest with GridSearchCV
- Makes interactive income predictions with the best baseline model

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` and upload the Adult dataset from the sidebar.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the entry point to `streamlit_app.py`.
4. Deploy. No secrets are required.

## Files

- `streamlit_app.py`: deployable application
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: visual theme and server configuration
- `2025AD05095_ML_assignment_2.ipynb`: original assignment notebook
- `ML_Assignment_2.pdf`: assignment document

Student ID: 2025AD05095
