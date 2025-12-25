import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)


DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/logistic_regression.joblib")


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Features not found at {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    print(f"\n ROC-AUC: {auc:.4f}\n")

    print(" Classification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))


def main():
    X, y = load_data(DATA_PATH)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = load_model(MODEL_PATH)
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
