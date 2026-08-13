# Adult Census Income Classification

**Student ID:** 2025AD05095  
**Course:** M.Tech (AIML/DSE) - Machine Learning  
**Assignment:** Assignment 2 - Classification Models and Streamlit Deployment

## a. Problem Statement

The objective is to predict whether a person's annual income is greater than USD 50,000 from census attributes. The project implements the five classifiers listed in the assignment, evaluates each model using six classification metrics, and provides an interactive Streamlit application for uploading test data, comparing models, inspecting confusion matrices, tuning Random Forest, and making predictions.

## b. Dataset Description

The project uses the public **UCI Adult Census Income dataset**. It contains 48,842 records and 14 input features plus the binary target `income` (`<=50K` or `>50K`). The dataset exceeds the assignment minimum of 500 instances and 12 features.

| Feature type | Features |
|---|---|
| Numerical | age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week |
| Categorical | workclass, education, marital-status, occupation, relationship, race, sex, native-country |
| Target | income |

Missing numerical values are replaced with the median. Missing categorical values are replaced with the most frequent category. Numerical features are standardized and categorical features are one-hot encoded. The split uses 80% training data and 20% test data with `random_state=42`.

## c. Project Links

- **GitHub repository:** [2025AD05095-ML-ASSIGNMENT-2](https://github.com/2025ad05095-lab/2025AD05095-ML-ASSIGNMENT-2)
- **Live Streamlit application:** [Open the deployed application](https://2025ad05095-ml-assignment-2-n2n9yb6qu5cuqyxacft9r9.streamlit.app/)
- **Dataset source:** [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)

## d. Models Used and Results

The following results were produced using the same preprocessing and held-out test split for every model.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8511 | **0.9057** | **0.7422** | 0.5856 | 0.6546 | 0.5678 |
| Decision Tree | 0.8165 | 0.7498 | 0.6189 | 0.6212 | 0.6200 | 0.4991 |
| k-Nearest Neighbors | 0.8292 | 0.8485 | 0.6668 | 0.5822 | 0.6216 | 0.5139 |
| Gaussian Naive Bayes | 0.6447 | 0.8533 | 0.3978 | **0.9223** | 0.5559 | 0.4125 |
| Random Forest (Ensemble) | **0.8513** | 0.8998 | 0.7325 | 0.6034 | **0.6617** | **0.5719** |

## Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the best AUC and precision. It is a strong, interpretable baseline with well-balanced overall performance. |
| Decision Tree | Produced moderate recall but the lowest AUC, indicating overfitting and weaker probability ranking than the other models. |
| k-Nearest Neighbors | Outperformed the single Decision Tree in accuracy and AUC, but its F1 score remained below the linear and ensemble models. |
| Gaussian Naive Bayes | Achieved the highest recall, detecting most high-income records, but low precision caused many false positives and reduced accuracy. |
| Random Forest (Ensemble) | Achieved the best accuracy, F1, and MCC, giving the strongest balance between both classes. |
| **Overall winner** | **Random Forest**, because it leads on accuracy, F1, and MCC. Logistic Regression is a close alternative and has the highest AUC. |

## Streamlit Features

- CSV upload for test data
- Dataset overview and preprocessing summary
- Model comparison with all six required metrics
- Model selection dropdown
- Confusion matrix for the selected model
- Random Forest hyperparameter tuning with GridSearchCV
- Interactive income prediction

## Repository Structure

```text
.
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- adult_data.csv
|-- model/
|   |-- model_training.py
|   `-- 2025AD05095_ML_assignment_2.ipynb
|-- submission/
|   `-- 2025AD05095_ML_Assignment_2_Submission.pdf
`-- .streamlit/
    `-- config.toml
```

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` and upload `test_data.csv` from the sidebar.

## Reproduce Model Results

```powershell
python model/model_training.py --data adult_data.csv
```

The script prints the comparison table and saves trained pipelines in `model/artifacts/`.
