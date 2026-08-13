"""Interactive Streamlit workspace for ML Assignment 2."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="ML Assignment 2", page_icon="📊", layout="wide")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
NUMERIC = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
    "hours_per_week",
]
CATEGORICAL = [column for column in COLUMNS if column not in NUMERIC + ["income"]]


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Read an Adult dataset CSV, accepting either headered or headerless files."""
    raw = pd.read_csv(
        io.BytesIO(file_bytes), header=None, names=COLUMNS, skipinitialspace=True,
        na_values=["?", " ?"], comment="|",
    )
    if not raw.empty and str(raw.iloc[0]["age"]).strip().lower() == "age":
        raw = raw.iloc[1:].copy()
    for column in NUMERIC:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["income"] = raw["income"].astype("string").str.strip().str.rstrip(".")
    raw = raw[raw["income"].isin(["<=50K", ">50K"])].reset_index(drop=True)
    if raw.empty:
        raise ValueError("No valid Adult Census rows were found in this file.")
    return raw


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])


def model_catalog() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }


def scores(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, prediction),
        "ROC AUC": roc_auc_score(y_test, probability),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1 Score": f1_score(y_test, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y_test, prediction),
    }


@st.cache_resource(show_spinner="Training five classification models...")
def train_models(file_bytes: bytes):
    data = load_data(file_bytes)
    x = data.drop(columns="income")
    y = data["income"].eq(">50K").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )
    trained = {}
    results = {}
    for name, estimator in model_catalog().items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        trained[name] = pipeline
        results[name] = scores(pipeline, x_test, y_test)
    return trained, pd.DataFrame(results).T, x_train, x_test, y_train, y_test


def show_overview(data: pd.DataFrame) -> None:
    st.header("Dataset overview")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(data):,}")
    metric_columns[1].metric("Features", len(COLUMNS) - 1)
    metric_columns[2].metric("Missing values", f"{int(data.isna().sum().sum()):,}")
    metric_columns[3].metric("Income >50K", f"{data['income'].eq('>50K').mean():.1%}")
    st.subheader("Data preview")
    st.dataframe(data.head(20), width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Numeric summary")
        st.dataframe(data[NUMERIC].describe().T, width="stretch")
    with right:
        st.subheader("Target distribution")
        counts = data["income"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values, color=["#287271", "#E07A5F"])
        ax.set_ylabel("Records")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")


def show_preprocessing(data: pd.DataFrame) -> None:
    st.header("Data preprocessing")
    st.write("Missing numeric values are replaced by the median. Missing categorical values use the most frequent value. Numeric features are standardized and categorical features are one-hot encoded.")
    left, right = st.columns(2)
    with left:
        st.subheader("Numerical features")
        st.dataframe(pd.DataFrame({"Feature": NUMERIC, "Missing": data[NUMERIC].isna().sum().values}), hide_index=True, width="stretch")
    with right:
        st.subheader("Categorical features")
        st.dataframe(pd.DataFrame({"Feature": CATEGORICAL, "Categories": [data[c].nunique() for c in CATEGORICAL]}), hide_index=True, width="stretch")


def show_comparison(file_bytes: bytes) -> None:
    st.header("Model comparison")
    trained, results, _, x_test, _, y_test = train_models(file_bytes)
    st.dataframe(results.style.format("{:.4f}").highlight_max(axis=0), width="stretch")
    best_name = results["ROC AUC"].idxmax()
    st.success(f"Best ROC AUC: {best_name} ({results.loc[best_name, 'ROC AUC']:.4f})")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    results[["Accuracy", "ROC AUC", "F1 Score"]].plot.bar(ax=ax, color=["#287271", "#E07A5F", "#3D405B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch")
    st.subheader("Confusion matrix")
    selected = st.selectbox("Model", list(trained), index=list(trained).index(best_name))
    matrix = confusion_matrix(y_test, trained[selected].predict(x_test))
    st.dataframe(pd.DataFrame(matrix, index=["Actual <=50K", "Actual >50K"], columns=["Predicted <=50K", "Predicted >50K"]), width="stretch")


def show_tuning(file_bytes: bytes) -> None:
    st.header("Random Forest tuning")
    _, _, x_train, x_test, y_train, y_test = train_models(file_bytes)
    if not st.button("Run GridSearchCV", type="primary"):
        st.info("Run the search to compare a tuned Random Forest with the baseline model.")
        return
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    with st.spinner("Running cross-validation..."):
        search = GridSearchCV(pipeline, grid, cv=3, scoring="roc_auc", n_jobs=-1)
        search.fit(x_train, y_train)
    st.success("Tuning complete")
    st.json({key.removeprefix("model__"): value for key, value in search.best_params_.items()})
    st.metric("Best cross-validation ROC AUC", f"{search.best_score_:.4f}")
    st.dataframe(pd.DataFrame([scores(search.best_estimator_, x_test, y_test)], index=["Tuned Random Forest"]).style.format("{:.4f}"), width="stretch")


def show_prediction(data: pd.DataFrame, file_bytes: bytes) -> None:
    st.header("Income prediction")
    trained, results, _, _, _, _ = train_models(file_bytes)
    model_name = results["ROC AUC"].idxmax()
    model = trained[model_name]
    st.caption(f"Prediction model: {model_name}")
    defaults = data.iloc[0]
    values = {}
    cols = st.columns(3)
    for index, feature in enumerate(NUMERIC):
        series = data[feature].dropna()
        values[feature] = cols[index % 3].number_input(feature.replace("_", " ").title(), value=float(series.median()), step=1.0)
    for index, feature in enumerate(CATEGORICAL):
        options = sorted(data[feature].dropna().astype(str).unique())
        default = str(defaults[feature])
        values[feature] = cols[index % 3].selectbox(feature.replace("_", " ").title(), options, index=options.index(default) if default in options else 0)
    if st.button("Predict income", type="primary"):
        row = pd.DataFrame([values], columns=NUMERIC + CATEGORICAL)
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0, 1])
        label = ">50K" if prediction else "<=50K"
        st.metric("Predicted income", label, help=f"Probability of income >50K: {probability:.1%}")
        st.progress(probability, text=f"Probability of >50K: {probability:.1%}")


st.title("Machine Learning Assignment 2")
st.caption("Adult Census income classification and performance evaluation")

uploaded = st.sidebar.file_uploader("Adult dataset", type=["csv", "data", "txt"])
page = st.sidebar.radio("Workspace", ["Overview", "Preprocessing", "Model comparison", "Hyperparameter tuning", "Prediction"])
st.sidebar.caption("Student ID: 2025AD05095")

if uploaded is None:
    st.info("Upload the Adult Census Income dataset from the sidebar to begin.")
    st.markdown("The file may be headered or headerless and should contain the standard 15 Adult dataset columns.")
    st.stop()

try:
    content = uploaded.getvalue()
    dataset = load_data(content)
except Exception as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()

if page == "Overview":
    show_overview(dataset)
elif page == "Preprocessing":
    show_preprocessing(dataset)
elif page == "Model comparison":
    show_comparison(content)
elif page == "Hyperparameter tuning":
    show_tuning(content)
else:
    show_prediction(dataset, content)
