from pathlib import Path
from typing import Any
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
CATEGORICAL_FEATURES = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def train_and_select_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[Pipeline, dict[str, float]]:
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    }

    best_pipeline = None
    best_score = -1.0
    results: dict[str, float] = {}

    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_proba)
        accuracy = accuracy_score(y_val, y_pred)
        results[name] = roc_auc
        if roc_auc > best_score:
            best_score = roc_auc
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained successfully.")

    return best_pipeline, results


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    print("Evaluation results")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_proba))
    print(classification_report(y, y_pred, target_names=["Rejected", "Approved"]))


def save_model(pipeline: Pipeline, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, model_path)


def load_model(model_path: Path) -> Pipeline:
    return load(model_path)
