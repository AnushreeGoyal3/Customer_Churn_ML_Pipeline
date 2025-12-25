import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib


DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/logistic_regression.joblib")


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Features not found at {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_model(X, y):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    print(f" Training ROC-AUC: {auc:.4f}")


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
