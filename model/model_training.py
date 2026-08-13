"""Train and save every classifier required for ML Assignment 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
TARGET = "income"


def build_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    categorical = [column for column in data.columns if column not in NUMERIC + [TARGET]]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def main(data_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(data_path, na_values="?")
    data[TARGET] = data[TARGET].astype(str).str.strip().str.rstrip(".")
    x = data.drop(columns=TARGET)
    y = data[TARGET].eq(">50K").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42,
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "gaussian_naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(data)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1]
        results[name] = {
            "Accuracy": accuracy_score(y_test, prediction),
            "AUC": roc_auc_score(y_test, probability),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1": f1_score(y_test, prediction, zero_division=0),
            "MCC": matthews_corrcoef(y_test, prediction),
        }
        with (output_dir / f"{name}.pkl").open("wb") as file:
            pickle.dump(pipeline, file)
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("adult_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("model/artifacts"))
    arguments = parser.parse_args()
    main(arguments.data, arguments.output)
